"""
Step 16 — Task 1 (lesion segmentation) evaluation.

Loads the best checkpoint saved by step 15, runs it over the validation
fold, and reports Dice / IoU / HD95 both as summary statistics and as a
per-image CSV. Also saves four-panel visuals (input, ground truth,
prediction, error map) for the worst-performing cases, which are the ones
worth looking at.

Run it the same way you ran step 15 — from PyCharm, or:

    python step16_task1_evaluation.py
"""

from step14_task1_training import build_task1_model
from step12_data_augmentation import LesionDataset, build_val_transform
from torch.utils.data import DataLoader, Subset
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # no interactive window; just write PNG files


# =====================================================================
# 0. Settings  — change these, not the code below
# =====================================================================

# MUST match the image_size used in the training run you are evaluating.
IMAGE_SIZE = 384

FOLD = 0
THRESHOLD = 0.5                # probability cut-off for "lesion"
SEED = 42                      # makes fast mode reproducible
N_FAST = 10                    # images sampled in fast mode
N_WORST_VISUALS = 10           # worst-Dice cases to draw in full mode
N_RANDOM_VISUALS = 5           # extra random cases, so visuals aren't all failures
RUN_THRESHOLD_SWEEP = True     # report Dice at several thresholds
CHECKPOINT_NAME = "task1_best_model.pth"


# =====================================================================
# 1. Reuse the metric code from step 15
# =====================================================================
# Importing rather than re-implementing guarantees the numbers here are
# the same quantity as the ones printed during training. If your step 15
# file has a different name, change the import below.

try:
    from step15_task1_train_loop import (
        select_device,
        hausdorff_95_from_masks,
    )
    print("Using metric functions imported from step 15.")
except ImportError:
    print("Could not import step 15 — using local copies of the metrics.")
    from scipy.ndimage import binary_erosion, distance_transform_edt

    def select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _boundary(mask_bool: np.ndarray) -> np.ndarray:
        if mask_bool.sum() == 0:
            return mask_bool
        return mask_bool & ~binary_erosion(mask_bool)

    def hausdorff_95_from_masks(pred_bool: np.ndarray,
                                target_bool: np.ndarray) -> float:
        """95th-percentile symmetric Hausdorff distance, in pixels."""
        if pred_bool.sum() == 0 or target_bool.sum() == 0:
            return float("nan")
        pred_b = _boundary(pred_bool)
        target_b = _boundary(target_bool)
        dt_to_target = distance_transform_edt(~target_b)
        dt_to_pred = distance_transform_edt(~pred_b)
        distances = np.concatenate(
            [dt_to_target[pred_b], dt_to_pred[target_b]]
        )
        return float(np.percentile(distances, 95))


# =====================================================================
# 2. Metrics for one image
# =====================================================================

def metrics_from_arrays(pred_bool: np.ndarray,
                        target_bool: np.ndarray,
                        compute_hd95: bool = True):
    """
    Dice, IoU and HD95 for a single pair of binary masks.

    Note the empty-mask convention: if the model predicts nothing and
    there is nothing to find, that is a perfect result (1.0), not a zero.
    """
    pred_sum = int(pred_bool.sum())
    target_sum = int(target_bool.sum())

    if pred_sum == 0 and target_sum == 0:
        return 1.0, 1.0, 0.0

    intersection = int(np.logical_and(pred_bool, target_bool).sum())
    union = pred_sum + target_sum - intersection

    dice = (2.0 * intersection) / (pred_sum + target_sum)
    iou = intersection / union if union > 0 else 0.0

    hd95 = (hausdorff_95_from_masks(pred_bool, target_bool)
            if compute_hd95 else float("nan"))

    return float(dice), float(iou), hd95


# =====================================================================
# 3. Four-panel visual
# =====================================================================

def save_evaluation_visual(image, mask_gt_bool, mask_pred_bool,
                           save_path, title=""):
    """image: CHW tensor. The two masks: 2-D boolean numpy arrays."""
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # Undo normalisation just enough to be viewable.
    spread = image_np.max() - image_np.min()
    if spread > 0:
        image_np = (image_np - image_np.min()) / spread
    image_np = np.clip(image_np, 0.0, 1.0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image_np)
    axes[0].set_title(f"Input: {title}")

    axes[1].imshow(mask_gt_bool, cmap="gray")
    axes[1].set_title("Ground truth")

    axes[2].imshow(mask_pred_bool, cmap="gray")
    axes[2].set_title("Prediction")

    error_map = np.zeros((*mask_gt_bool.shape, 3), dtype=np.float32)
    error_map[mask_pred_bool & mask_gt_bool] = [1.0, 1.0, 1.0]    # correct
    error_map[mask_pred_bool & ~mask_gt_bool] = [0.6, 0.0, 0.8]   # extra
    error_map[~mask_pred_bool & mask_gt_bool] = [1.0, 0.0, 0.0]   # missed

    axes[3].imshow(error_map)
    axes[3].set_title("Error map (white: correct, purple: extra, red: missed)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_dice_histogram(dice_values, save_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dice_values, bins=25, range=(0.0, 1.0),
            color="#4a7ba7", edgecolor="black")
    ax.set_xlabel("Dice score")
    ax.set_ylabel("Number of images")
    ax.set_title("Distribution of per-image Dice on the validation fold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# =====================================================================
# 4. Locating the checkpoint
# =====================================================================
# Step 15 wrote to a path relative to wherever it was launched from, so
# the checkpoint could be under Images/ or under the project root. Check
# the likely places instead of guessing.

def locate_checkpoint() -> Path:
    script_dir = Path(__file__).resolve().parent
    relative = Path("outputs") / "training_results" / CHECKPOINT_NAME
    candidate_roots = [script_dir, script_dir.parent, Path.cwd()]

    for root in candidate_roots:
        candidate = root / relative
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(root / relative) for root in candidate_roots)
    raise FileNotFoundError(
        f"Could not find {CHECKPOINT_NAME}. Looked in:\n  {searched}"
    )


# =====================================================================
# 5. Main evaluation
# =====================================================================

def evaluate_model(fast_mode: bool):
    device = select_device()
    print(f"\nEvaluating on {device}")

    checkpoint_path = locate_checkpoint()
    print(f"Checkpoint: {checkpoint_path}")

    # Put evaluation outputs next to the training outputs.
    output_dir = checkpoint_path.parent.parent / "evaluation"
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)

    # --- data -------------------------------------------------------
    val_transform = build_val_transform(image_size=IMAGE_SIZE)
    full_val_dataset = LesionDataset(
        fold=FOLD, role="val", transform=val_transform, include_task2=False
    )

    if fast_mode:
        random.seed(SEED)
        indices = random.sample(
            range(len(full_val_dataset)),
            min(N_FAST, len(full_val_dataset)),
        )
        dataset = Subset(full_val_dataset, indices)
        print(f"Fast mode: {len(dataset)} images (seed {SEED}).")
    else:
        dataset = full_val_dataset
        print(f"Full mode: {len(dataset)} images.")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # --- model ------------------------------------------------------
    model = build_task1_model().to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # --- loop -------------------------------------------------------
    sweep_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    sweep_dice = {t: [] for t in sweep_thresholds}
    rows = []
    cached = []          # (image_id, image_tensor, gt_bool, pred_bool)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            images = batch["image"].to(device)
            masks = batch["task1_segmentation"].to(device)
            image_id = batch["image_id"][0]

            probs = torch.sigmoid(model(images))

            prob_np = probs[0].squeeze().cpu().numpy()
            gt_bool = masks[0].squeeze().cpu().numpy() > 0.5
            pred_bool = prob_np > THRESHOLD

            dice, iou, hd95 = metrics_from_arrays(pred_bool, gt_bool)

            rows.append({
                "image_id": image_id,
                "dice": dice,
                "iou": iou,
                "hd95": hd95,
                "gt_pixels": int(gt_bool.sum()),
                "pred_pixels": int(pred_bool.sum()),
            })

            if RUN_THRESHOLD_SWEEP:
                for t in sweep_thresholds:
                    p = prob_np > t
                    d, _, _ = metrics_from_arrays(p, gt_bool,
                                                  compute_hd95=False)
                    sweep_dice[t].append(d)

            cached.append((image_id, images[0].cpu(), gt_bool, pred_bool))

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(loader)} images")

    results = pd.DataFrame(rows)

    # --- per-image CSV ----------------------------------------------
    csv_path = output_dir / "task1_val_per_image_metrics.csv"
    results.to_csv(csv_path, index=False)

    # --- visuals ----------------------------------------------------
    to_draw = results.nsmallest(N_WORST_VISUALS, "dice")["image_id"].tolist()
    if not fast_mode:
        remaining = [r for r in results["image_id"] if r not in to_draw]
        random.seed(SEED)
        to_draw += random.sample(
            remaining, min(N_RANDOM_VISUALS, len(remaining))
        )

    draw_set = set(to_draw)
    for image_id, image_tensor, gt_bool, pred_bool in cached:
        if image_id not in draw_set:
            continue
        dice = float(results.loc[results["image_id"]
                     == image_id, "dice"].iloc[0])
        save_evaluation_visual(
            image_tensor, gt_bool, pred_bool,
            visual_dir / f"dice{dice:.3f}_{image_id}.png",
            title=f"{image_id} (Dice {dice:.3f})",
        )

    save_dice_histogram(results["dice"].to_numpy(),
                        output_dir / "task1_val_dice_histogram.png")

    # --- summary ----------------------------------------------------
    hd_valid = results["hd95"].dropna()

    print("\n" + "=" * 62)
    print("TASK 1 — VALIDATION METRICS")
    print("=" * 62)
    print(f"Images evaluated      : {len(results)}")
    print(f"Image size            : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Threshold             : {THRESHOLD}")
    print("-" * 62)
    print(f"Dice   mean / median  : {results['dice'].mean():.4f} / "
          f"{results['dice'].median():.4f}")
    print(f"Dice   std / min      : {results['dice'].std():.4f} / "
          f"{results['dice'].min():.4f}")
    print(f"IoU    mean / median  : {results['iou'].mean():.4f} / "
          f"{results['iou'].median():.4f}")
    if len(hd_valid) > 0:
        print(f"HD95   mean / median  : {hd_valid.mean():.2f} / "
              f"{hd_valid.median():.2f} px  ({len(hd_valid)} valid)")
    else:
        print("HD95                  : not computable (empty masks)")
    print("-" * 62)
    print(f"Images with Dice < 0.65 : "
          f"{(results['dice'] < 0.65).sum()} "
          f"({100 * (results['dice'] < 0.65).mean():.1f}%)")
    print(f"Images with Dice < 0.50 : "
          f"{(results['dice'] < 0.50).sum()} "
          f"({100 * (results['dice'] < 0.50).mean():.1f}%)")

    if RUN_THRESHOLD_SWEEP:
        print("-" * 62)
        print("Mean Dice by threshold:")
        for t in sweep_thresholds:
            print(f"  {t:.1f} : {np.mean(sweep_dice[t]):.4f}")

    print("=" * 62)
    print(f"Per-image metrics : {csv_path}")
    print(f"Visuals           : {visual_dir}")
    print("=" * 62)

    return results


if __name__ == "__main__":
    print("\n--- STEP 16: TASK 1 EVALUATION ---")
    print(f"1) Fast mode: {N_FAST} random images (fixed seed)")
    print("2) Full mode: the entire validation fold")

    choice = input("Select an option (1 or 2): ").strip()

    if choice == "1":
        evaluate_model(fast_mode=True)
    elif choice == "2":
        evaluate_model(fast_mode=False)
    else:
        print("Invalid choice. Run again and enter 1 or 2.")
