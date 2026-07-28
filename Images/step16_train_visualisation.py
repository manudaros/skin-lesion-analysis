"""
Step 16 — Task 1 evaluation and error analysis.

Step 15 writes its own per-image metrics for the best epoch, so this is
not where your headline numbers come from. Its job is what step 15
doesn't do:

  * Visuals of the WORST cases, not the first ten alphabetically. Those
    are the images worth looking at and worth putting in the report.
  * A threshold sweep. 0.5 is an arbitrary operating point; the sweep
    shows what moving it would gain, on Dice and on the official
    thresholded Jaccard, which do not always peak at the same place.
  * A Dice histogram. A mean of 0.85 means something very different if
    everything sits at 0.85 than if most images are at 0.92 with a tail
    of total failures — and the ISIC organisers specifically note that
    aggregate statistics hide exactly this.
  * Evaluating any checkpoint you like: best or last, any fold.

Metrics are imported from step 15 rather than reimplemented, so the
numbers here are the same quantity reported during training.

    python step16_train_visualisation.py
"""

from step15_train import (
    JACCARD_FAILURE_THRESHOLD,
    calculate_metrics_single,
    select_device,
)
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
matplotlib.use("Agg")


# =====================================================================
# 0. Settings
# =====================================================================

FOLD = 0
IMAGE_SIZE = 384           # must match the run you are evaluating
WHICH_CHECKPOINT = "best"  # "best" or "last"
THRESHOLD = 0.5

SEED = 42                  # fixed, so fast mode is comparable across runs
N_FAST = 10                # images sampled in fast mode
N_WORST_VISUALS = 10       # worst-Dice cases to draw
N_RANDOM_VISUALS = 5       # extra random cases, so visuals aren't all failures

RUN_THRESHOLD_SWEEP = True
SWEEP_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

COMPUTE_HD95 = True


# =====================================================================
# 1. Where Task 1 checkpoints live
# =====================================================================
# These duplicate the output layout defined in step15_train.py. If you
# ever change OUTPUT_DIR there, change task1_output_dir here to match —
# otherwise this script silently evaluates a stale checkpoint from an
# earlier run instead of raising, which is a hard bug to notice.

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def task1_output_dir(fold: int, image_size: int) -> Path:
    """Must stay identical to OUTPUT_DIR in step15_train.py."""
    return (PROJECT_ROOT / "outputs" / "task1_training"
            / f"fold_{fold}_{image_size}px")


def locate_task1_checkpoint(fold: int, image_size: int,
                            which: str = "best") -> Path | None:
    """
    Find a trained Task 1 checkpoint, or return None.

    Current layout first, then the legacy flat location so a checkpoint
    trained before the paths changed still resolves.
    """
    name = f"task1_{which}_model.pth"
    candidates = [
        task1_output_dir(fold, image_size) / name,
        SCRIPT_DIR / "outputs" / "task1_training"
        / f"fold_{fold}_{image_size}px" / name,
        PROJECT_ROOT / "outputs" / "training_results" / name,
        SCRIPT_DIR / "outputs" / "training_results" / name,
        Path.cwd() / "outputs" / "training_results" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


# =====================================================================
# 2. Figures
# =====================================================================

def save_evaluation_visual(image, mask_gt_bool, mask_pred_bool,
                           save_path, title=""):
    """image: CHW tensor. The two masks: 2-D boolean numpy arrays."""
    image_np = image.permute(1, 2, 0).cpu().numpy()
    spread = image_np.max() - image_np.min()
    if spread > 0:
        image_np = (image_np - image_np.min()) / spread
    image_np = np.clip(image_np, 0.0, 1.0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image_np)
    axes[0].set_title(f"Input: {title}")

    axes[1].imshow(mask_gt_bool, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth")

    axes[2].imshow(mask_pred_bool, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Prediction")

    error_map = np.zeros((*mask_gt_bool.shape, 3), dtype=np.float32)
    error_map[mask_pred_bool & mask_gt_bool] = [1.0, 1.0, 1.0]    # correct
    error_map[mask_pred_bool & ~mask_gt_bool] = [0.6, 0.0, 0.8]   # extra
    error_map[~mask_pred_bool & mask_gt_bool] = [1.0, 0.0, 0.0]   # missed

    axes[3].imshow(error_map)
    axes[3].set_title("Error map (white correct, purple extra, red missed)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_dice_histogram(dice_values, save_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dice_values, bins=25, range=(0.0, 1.0),
            color="#4a7ba7", edgecolor="black")
    ax.axvline(JACCARD_FAILURE_THRESHOLD, color="crimson",
               linestyle="--", linewidth=1.5,
               label=f"ISIC failure threshold ({JACCARD_FAILURE_THRESHOLD})")
    ax.set_xlabel("Dice score")
    ax.set_ylabel("Number of images")
    ax.set_title("Per-image Dice on the validation fold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def save_threshold_plot(sweep, save_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["threshold"], sweep["dice"], marker="o", label="Dice")
    ax.plot(sweep["threshold"], sweep["thresholded_jaccard"],
            marker="s", label="thresholded Jaccard")
    ax.set_xlabel("probability threshold")
    ax.set_ylabel("mean score")
    ax.set_title("Operating point sweep")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# =====================================================================
# 3. Evaluation
# =====================================================================

def evaluate(fast_mode: bool):
    device = select_device()
    print(f"Evaluating on {device}")

    checkpoint_path = locate_task1_checkpoint(
        fold=FOLD, image_size=IMAGE_SIZE, which=WHICH_CHECKPOINT
    )
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No '{WHICH_CHECKPOINT}' Task 1 checkpoint for fold {FOLD} at "
            f"{IMAGE_SIZE}px. Has step 15 finished?"
        )
    print(f"Checkpoint: {checkpoint_path}")

    output_dir = task1_output_dir(FOLD, IMAGE_SIZE) / "evaluation"
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)

    # --- data -------------------------------------------------------
    full_val_dataset = LesionDataset(
        fold=FOLD, role="val",
        transform=build_val_transform(image_size=IMAGE_SIZE),
        include_task2=False,
    )

    if fast_mode:
        random.seed(SEED)
        indices = random.sample(
            range(len(full_val_dataset)),
            min(N_FAST, len(full_val_dataset)),
        )
        dataset = Subset(full_val_dataset, indices)
        print(f"Fast mode: {len(dataset)} images (seed {SEED})")
    else:
        dataset = full_val_dataset
        print(f"Full mode: {len(dataset)} images")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # --- model ------------------------------------------------------
    model = build_task1_model().to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    # --- loop -------------------------------------------------------
    rows = []
    sweep_scores = {t: {"dice": [], "tj": []} for t in SWEEP_THRESHOLDS}
    cached = []          # (image_id, image, gt_bool, pred_bool)

    with torch.inference_mode():
        for index, batch in enumerate(loader):
            images = batch["image"].to(device)
            masks = batch["task1_segmentation"].to(device,
                                                   dtype=torch.float32)
            image_id = str(batch["image_id"][0])

            probabilities = torch.sigmoid(model(images))

            dice, iou, thresholded, hd95 = calculate_metrics_single(
                probabilities, masks,
                threshold=THRESHOLD, compute_hd95=COMPUTE_HD95,
            )

            gt_bool = masks[0].squeeze().cpu().numpy() > 0.5
            pred_bool = (probabilities[0].squeeze().cpu().numpy()
                         > THRESHOLD)

            rows.append({
                "image_id": image_id,
                "dice": dice,
                "iou": iou,
                "thresholded_jaccard": thresholded,
                "hd95_pixels": hd95 if np.isfinite(hd95) else "",
                "gt_pixels": int(gt_bool.sum()),
                "pred_pixels": int(pred_bool.sum()),
            })

            if RUN_THRESHOLD_SWEEP:
                for value in SWEEP_THRESHOLDS:
                    d, _, tj, _ = calculate_metrics_single(
                        probabilities, masks,
                        threshold=value, compute_hd95=False,
                    )
                    sweep_scores[value]["dice"].append(d)
                    sweep_scores[value]["tj"].append(tj)

            cached.append((image_id, images[0].cpu(), gt_bool, pred_bool))

            if (index + 1) % 50 == 0:
                print(f"  {index + 1}/{len(loader)} images")

    results = pd.DataFrame(rows)
    csv_path = output_dir / f"task1_{WHICH_CHECKPOINT}_per_image_metrics.csv"
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
        dice = float(
            results.loc[results["image_id"] == image_id, "dice"].iloc[0]
        )
        save_evaluation_visual(
            image_tensor, gt_bool, pred_bool,
            visual_dir / f"dice{dice:.3f}_{image_id}.png",
            title=f"{image_id} (Dice {dice:.3f})",
        )

    save_dice_histogram(results["dice"].to_numpy(),
                        output_dir / "task1_dice_histogram.png")

    # --- summary ----------------------------------------------------
    hd_valid = pd.to_numeric(results["hd95_pixels"],
                             errors="coerce").dropna()
    failures = (results["iou"] < JACCARD_FAILURE_THRESHOLD).sum()

    print("\n" + "=" * 66)
    print(f"TASK 1 — fold {FOLD}, {IMAGE_SIZE}px, "
          f"{WHICH_CHECKPOINT} checkpoint")
    print("=" * 66)
    print(f"Images evaluated       : {len(results)}")
    print(f"Threshold              : {THRESHOLD}")
    print("-" * 66)
    print(f"Dice   mean / median   : {results['dice'].mean():.4f} / "
          f"{results['dice'].median():.4f}")
    print(f"Dice   std / min       : {results['dice'].std():.4f} / "
          f"{results['dice'].min():.4f}")
    print(f"IoU    mean            : {results['iou'].mean():.4f}")
    print(f"Thresholded Jaccard    : "
          f"{results['thresholded_jaccard'].mean():.4f}")
    if len(hd_valid) > 0:
        print(f"HD95   mean / median   : {hd_valid.mean():.2f} / "
              f"{hd_valid.median():.2f} px  ({len(hd_valid)} valid)")
    print("-" * 66)
    print(f"Failures (IoU < {JACCARD_FAILURE_THRESHOLD})   : {failures} "
          f"({100 * failures / len(results):.1f}%)")
    print(f"Empty predictions      : "
          f"{(results['pred_pixels'] == 0).sum()}")

    if RUN_THRESHOLD_SWEEP:
        sweep = pd.DataFrame({
            "threshold": SWEEP_THRESHOLDS,
            "dice": [float(np.mean(sweep_scores[t]["dice"]))
                     for t in SWEEP_THRESHOLDS],
            "thresholded_jaccard": [float(np.mean(sweep_scores[t]["tj"]))
                                    for t in SWEEP_THRESHOLDS],
        })
        sweep.to_csv(output_dir / "task1_threshold_sweep.csv", index=False)
        save_threshold_plot(sweep, output_dir / "task1_threshold_sweep.png")

        best_dice_row = sweep.loc[sweep["dice"].idxmax()]
        best_tj_row = sweep.loc[sweep["thresholded_jaccard"].idxmax()]

        print("-" * 66)
        print(f"{'threshold':>10} {'Dice':>10} {'thr. Jaccard':>14}")
        for _, row in sweep.iterrows():
            print(f"{row['threshold']:>10.2f} {row['dice']:>10.4f} "
                  f"{row['thresholded_jaccard']:>14.4f}")
        print(f"Best Dice at threshold {best_dice_row['threshold']:.2f} "
              f"({best_dice_row['dice']:.4f})")
        print(f"Best thresholded Jaccard at "
              f"{best_tj_row['threshold']:.2f} "
              f"({best_tj_row['thresholded_jaccard']:.4f})")

    print("=" * 66)
    print(f"Per-image metrics : {csv_path}")
    print(f"Visuals           : {visual_dir}")
    print("=" * 66)

    return results


if __name__ == "__main__":
    print("\n--- STEP 16: TASK 1 EVALUATION ---")
    print(f"1) Fast mode: {N_FAST} random images (fixed seed)")
    print("2) Full mode: the entire validation fold")

    choice = input("Select an option (1 or 2): ").strip()

    if choice == "1":
        evaluate(fast_mode=True)
    elif choice == "2":
        evaluate(fast_mode=False)
    else:
        print("Invalid choice. Run again and enter 1 or 2.")
