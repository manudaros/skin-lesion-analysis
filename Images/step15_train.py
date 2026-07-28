"""
Step 15 — Task 1 training loop (lesion segmentation).

Trains the ResNet34 U-Net from step 14 on folds 1-4, validating on fold 0.

Condensed from the longer draft, with the same behaviour plus three
additions:

  * RESUME — the run already saved everything needed to continue but had
    no way to use it. It does now, which matters for a multi-hour job.
  * Thresholded Jaccard — the official ISIC 2018 Task 1 metric. Per-image
    Jaccard is set to zero when it falls below 0.65, so the score tracks
    the failure rate rather than hiding it behind a good average. The
    challenge's top submission scored 0.802 on this; plain mean Jaccard
    flatters a model that fails badly on a minority of images.
  * OUTPUT_DIR includes the fold and resolution, so a second run at a
    different size doesn't overwrite the first.

    python step15_task1_train_loop.py
"""

from __future__ import annotations
from step14_task1_training import build_task1_model
from step12_data_augmentation import (
    LesionDataset,
    build_train_transform,
    build_val_transform,
)
from torch.utils.data import DataLoader
from scipy.ndimage import binary_erosion, distance_transform_edt
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

matplotlib.use("Agg")


# ---------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------

RANDOM_SEED = 42
IMAGE_SIZE = 384
BATCH_SIZE = 8
DEVELOPMENT_FOLD = 0          # fold 0 validates, folds 1-4 train

MAX_EPOCHS = 35
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5
PREDICTION_THRESHOLD = 0.5

# ISIC scores a segmentation as a failure below this Jaccard.
JACCARD_FAILURE_THRESHOLD = 0.65

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
MINIMUM_LEARNING_RATE = 1e-6

EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 1e-4

COMPUTE_HD95 = True
RESUME = True                 # continue from the last checkpoint if present

# Required on Apple Silicon: any other value hangs the MPS dataloader.
NUM_WORKERS = 0

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "task1_training"
    / f"fold_{DEVELOPMENT_FOLD}_{IMAGE_SIZE}px"
)


# ---------------------------------------------------------------------
# 1. Reproducibility and device
# ---------------------------------------------------------------------

def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for repeatable runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    """CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------
# 2. Loss
# ---------------------------------------------------------------------

class BCEDiceLoss(nn.Module):
    """Weighted sum of binary cross-entropy and soft Dice loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("Loss weights must be non-negative.")
        if bce_weight + dice_weight <= 0:
            raise ValueError("At least one loss weight must be positive.")

        total = bce_weight + dice_weight
        self.bce_weight = bce_weight / total
        self.dice_weight = dice_weight / total
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                smooth: float = 1e-6) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: {tuple(logits.shape)} vs "
                f"{tuple(targets.shape)}"
            )

        bce_loss = self.bce(logits, targets)

        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(2, 3))
        denominator = (probabilities.sum(dim=(2, 3))
                       + targets.sum(dim=(2, 3)))
        dice_score = (2.0 * intersection + smooth) / (denominator + smooth)
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ---------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------

def extract_boundary(mask_bool: np.ndarray) -> np.ndarray:
    """One-pixel-wide boundary of a binary mask."""
    mask_bool = np.asarray(mask_bool, dtype=bool)
    if not mask_bool.any():
        return np.zeros_like(mask_bool, dtype=bool)

    eroded = binary_erosion(
        mask_bool, structure=np.ones((3, 3), dtype=bool),
        iterations=1, border_value=0,
    )
    return mask_bool & ~eroded


def hausdorff_95_from_masks(prediction: np.ndarray,
                            target: np.ndarray) -> float:
    """
    Symmetric 95th-percentile boundary Hausdorff distance, in pixels.

    Group policy retained: NaN when either mask is empty, and those cases
    are excluded from the HD95 average rather than counted as zero.

    Uses distance transforms rather than pairwise point comparison — the
    O(n^2) version stalls for tens of minutes at this resolution.
    """
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    if not prediction.any() or not target.any():
        return float("nan")

    prediction_boundary = extract_boundary(prediction)
    target_boundary = extract_boundary(target)

    distance_to_target = distance_transform_edt(~target_boundary)
    distance_to_prediction = distance_transform_edt(~prediction_boundary)

    all_distances = np.concatenate([
        distance_to_target[prediction_boundary],
        distance_to_prediction[target_boundary],
    ])
    return float(np.percentile(all_distances, 95))


def calculate_metrics_single(
    probability: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    compute_hd95: bool = True,
) -> tuple[float, float, float, float]:
    """
    Dice, IoU, thresholded Jaccard and HD95 for one validation image.

    An empty prediction on an empty target scores 1.0, not 0.0 — that is
    a correct answer, and the 1e-6 denominator trick would punish it.
    """
    prediction = (probability > threshold).float().reshape(-1)
    target_flat = (target > 0.5).float().reshape(-1)

    intersection = (prediction * target_flat).sum().item()
    prediction_sum = prediction.sum().item()
    target_sum = target_flat.sum().item()
    union = prediction_sum + target_sum - intersection

    dice = (1.0 if prediction_sum + target_sum == 0
            else 2.0 * intersection / (prediction_sum + target_sum))
    iou = 1.0 if union == 0 else intersection / union

    # Official ISIC metric: a below-threshold overlap counts as a failure.
    thresholded_jaccard = iou if iou >= JACCARD_FAILURE_THRESHOLD else 0.0

    if not compute_hd95:
        return dice, iou, thresholded_jaccard, float("nan")

    hd95 = hausdorff_95_from_masks(
        (probability > threshold).squeeze().detach().cpu().numpy(),
        (target > 0.5).squeeze().detach().cpu().numpy(),
    )
    return dice, iou, thresholded_jaccard, hd95


def calculate_statistics(values: list[float]) -> dict[str, Any]:
    """Report-friendly descriptive statistics."""
    if not values:
        return {"count": 0, "mean": None, "std": None,
                "median": None, "minimum": None, "maximum": None}

    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


# ---------------------------------------------------------------------
# 4. Output helpers
# ---------------------------------------------------------------------

def replace_non_finite(value: Any) -> Any:
    """NaN and inf are not valid JSON; write null instead."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: replace_non_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_non_finite(item) for item in value]
    return value


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(replace_non_finite(data), indent=2,
                   ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def save_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a list of dicts as CSV. Used for both history and per-image."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_error_map(image: torch.Tensor, mask_gt: torch.Tensor,
                   mask_pred: torch.Tensor, save_path: Path) -> None:
    """Input, target, prediction and a colour-coded error overlay."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    image_array = image.detach().cpu().permute(1, 2, 0).numpy()
    spread = image_array.max() - image_array.min()
    if spread > 0:
        image_array = (image_array - image_array.min()) / spread
    image_array = np.clip(image_array, 0.0, 1.0)

    gt = mask_gt.detach().cpu().squeeze().numpy() > 0.5
    pred = mask_pred.detach().cpu().squeeze().numpy() > 0.5

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image_array)
    axes[0].set_title("Input image")
    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth")
    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Prediction")

    error_map = np.zeros((*gt.shape, 3), dtype=np.float32)
    error_map[pred & gt] = [0.0, 1.0, 0.0]      # correct
    error_map[pred & ~gt] = [1.0, 0.0, 0.0]     # extra
    error_map[~pred & gt] = [0.0, 0.0, 1.0]     # missed

    axes[3].imshow(image_array)
    axes[3].imshow(error_map, alpha=0.5)
    axes[3].set_title("Error map (G correct, R extra, B missed)")

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------
# 5. Train and validate one epoch
# ---------------------------------------------------------------------

def train_one_epoch(model, train_loader, criterion, optimizer,
                    device, epoch) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    started = time.time()

    for step, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        masks = batch["task1_segmentation"].to(device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), masks)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss: {loss.item()}")

        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size

        if step % 25 == 0 or step == len(train_loader) - 1:
            print(f"  epoch {epoch:02d} | step {step + 1:03d}/"
                  f"{len(train_loader):03d} | loss {loss.item():.4f} | "
                  f"{time.time() - started:.0f}s")

    if total_samples == 0:
        raise RuntimeError("The training DataLoader produced no samples.")
    return total_loss / total_samples


def evaluate_model(model, val_loader, criterion, device, threshold,
                   compute_hd95, visualization_path=None,
                   collect_per_image=False):
    """Loss, Dice, IoU, thresholded Jaccard and HD95 on the validation fold."""
    model.eval()

    total_loss = 0.0
    total_samples = 0
    dice_scores, iou_scores, thresholded_scores, hd95_scores = [], [], [], []
    per_image_rows: list[dict[str, Any]] = []
    visual_saved = False

    with torch.inference_mode():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["task1_segmentation"].to(device,
                                                   dtype=torch.float32)
            logits = model(images)
            probabilities = torch.sigmoid(logits)

            batch_size = images.shape[0]
            total_loss += criterion(logits, masks).item() * batch_size
            total_samples += batch_size

            image_ids = batch.get("image_id")

            for index in range(batch_size):
                dice, iou, thresholded, hd95 = calculate_metrics_single(
                    probabilities[index:index + 1],
                    masks[index:index + 1],
                    threshold=threshold,
                    compute_hd95=compute_hd95,
                )
                dice_scores.append(dice)
                iou_scores.append(iou)
                thresholded_scores.append(thresholded)
                if np.isfinite(hd95):
                    hd95_scores.append(hd95)

                if collect_per_image:
                    per_image_rows.append({
                        "image_id": (str(image_ids[index])
                                     if image_ids is not None
                                     else f"val_{len(per_image_rows):05d}"),
                        "dice": dice,
                        "iou": iou,
                        "thresholded_jaccard": thresholded,
                        "hd95_pixels": hd95 if np.isfinite(hd95) else "",
                    })

            if visualization_path is not None and not visual_saved:
                save_error_map(images[0], masks[0],
                               (probabilities[0] > threshold).float(),
                               visualization_path)
                visual_saved = True

    if total_samples == 0:
        raise RuntimeError("The validation DataLoader produced no samples.")

    failures = sum(1 for value in iou_scores
                   if value < JACCARD_FAILURE_THRESHOLD)

    metrics = {
        "validation_loss": total_loss / total_samples,
        "dice": calculate_statistics(dice_scores),
        "iou": calculate_statistics(iou_scores),
        "thresholded_jaccard": calculate_statistics(thresholded_scores),
        "hd95_pixels": calculate_statistics(hd95_scores),
        "hd95_valid_case_count": len(hd95_scores),
        "hd95_excluded_case_count": total_samples - len(hd95_scores),
        "failure_count": failures,
        "failure_rate": failures / total_samples,
        "prediction_threshold": threshold,
        "validation_sample_count": total_samples,
    }
    return metrics, per_image_rows


# ---------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------

def train_model(train_loader, val_loader, output_dir: Path):
    device = select_device()
    print(f"Training device: {device}")

    model = build_task1_model().to(device)
    total_parameters = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_parameters:,}")

    criterion = BCEDiceLoss(BCE_WEIGHT, DICE_WEIGHT)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE, min_lr=MINIMUM_LEARNING_RATE,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = output_dir / "task1_best_model.pth"
    best_checkpoint_path = output_dir / "task1_best_checkpoint.pth"
    last_checkpoint_path = output_dir / "task1_last_checkpoint.pth"

    run_config = {
        "task": "Task 1 lesion segmentation",
        "validation_strategy": (
            f"Fixed development split: fold {DEVELOPMENT_FOLD} validation, "
            "remaining folds training"
        ),
        "full_five_fold_cross_validation": False,
        "training_sample_count": len(train_loader.dataset),
        "validation_sample_count": len(val_loader.dataset),
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "optimizer": "AdamW",
        "initial_learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "ReduceLROnPlateau (mode=max on validation Dice)",
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "early_stopping_monitor": "validation Dice",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "loss_function": f"{BCE_WEIGHT} BCEWithLogits + {DICE_WEIGHT} soft Dice",
        "prediction_threshold": PREDICTION_THRESHOLD,
        "jaccard_failure_threshold": JACCARD_FAILURE_THRESHOLD,
        "empty_mask_policy": (
            "Dice and IoU score 1.0 when prediction and target are both "
            "empty. HD95 is NaN when either is empty and excluded."
        ),
        "random_seed": RANDOM_SEED,
        "num_workers": NUM_WORKERS,
        "device": str(device),
        "total_model_parameters": total_parameters,
    }
    save_json(run_config, output_dir / "run_config.json")

    start_epoch = 1
    best_dice = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    if RESUME and last_checkpoint_path.is_file():
        print(f"Resuming from {last_checkpoint_path.name}")
        state = torch.load(last_checkpoint_path, map_location=device,
                           weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"] + 1
        best_dice = state["best_validation_dice"]
        best_epoch = state.get("best_epoch", 0)
        epochs_without_improvement = state["epochs_without_improvement"]
        history = state["history"]
        print(f"Continuing at epoch {start_epoch}, "
              f"best Dice so far {best_dice:.4f}")

    training_start = time.time()

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_start = time.time()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        print("\n" + "=" * 78)
        print(f"Epoch {epoch}/{MAX_EPOCHS} | learning rate {learning_rate:.2e}")

        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device, epoch)

        metrics, _ = evaluate_model(
            model, val_loader, criterion, device,
            threshold=PREDICTION_THRESHOLD, compute_hd95=COMPUTE_HD95,
            visualization_path=(output_dir / "visualizations"
                                / f"epoch_{epoch:03d}.png"),
        )

        mean_dice = float(metrics["dice"]["mean"])
        improved = mean_dice > best_dice + EARLY_STOPPING_MIN_DELTA

        if improved:
            best_dice, best_epoch = mean_dice, epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        scheduler.step(mean_dice)
        epoch_seconds = time.time() - epoch_start
        hd95_mean = metrics["hd95_pixels"]["mean"]

        history.append({
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "validation_loss": metrics["validation_loss"],
            "validation_dice_mean": mean_dice,
            "validation_dice_std": metrics["dice"]["std"],
            "validation_iou_mean": metrics["iou"]["mean"],
            "validation_thresholded_jaccard":
                metrics["thresholded_jaccard"]["mean"],
            "validation_failure_rate": metrics["failure_rate"],
            "validation_hd95_mean_pixels": hd95_mean,
            "hd95_valid_case_count": metrics["hd95_valid_case_count"],
            "best_validation_dice": best_dice,
            "best_epoch_so_far": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "epoch_seconds": epoch_seconds,
        })

        remaining = (MAX_EPOCHS - epoch) * epoch_seconds
        print(
            f"Epoch {epoch:02d} done in {epoch_seconds:.0f}s "
            f"(~{remaining / 3600:.1f}h left)\n"
            f"  train loss           {train_loss:.4f}\n"
            f"  validation loss      {metrics['validation_loss']:.4f}\n"
            f"  Dice                 {mean_dice:.4f} "
            f"+/- {metrics['dice']['std']:.4f}\n"
            f"  IoU                  {metrics['iou']['mean']:.4f}\n"
            f"  thresholded Jaccard  "
            f"{metrics['thresholded_jaccard']['mean']:.4f}\n"
            f"  failures (<{JACCARD_FAILURE_THRESHOLD} IoU)  "
            f"{metrics['failure_count']}/"
            f"{metrics['validation_sample_count']} "
            f"({100 * metrics['failure_rate']:.1f}%)\n"
            f"  HD95                 "
            f"{'n/a' if hd95_mean is None else f'{hd95_mean:.2f} px'} "
            f"({metrics['hd95_valid_case_count']} valid)\n"
            f"  best Dice            {best_dice:.4f} (epoch {best_epoch})\n"
            f"  no improvement       {epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_validation_dice": best_dice,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "run_config": run_config,
        }
        torch.save(checkpoint, last_checkpoint_path)
        torch.save(model.state_dict(), output_dir / "task1_last_model.pth")

        if improved:
            # Plain state_dict for downstream scripts; full checkpoint too.
            torch.save(model.state_dict(), best_model_path)
            torch.save(checkpoint, best_checkpoint_path)
            print("  new best model saved")

        save_json(history, output_dir / "training_history.json")
        save_rows_csv(history, output_dir / "training_history.csv")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping: no improvement for "
                  f"{EARLY_STOPPING_PATIENCE} epochs.")
            break

    total_seconds = time.time() - training_start

    if not best_checkpoint_path.is_file():
        raise RuntimeError("Training ended without a best checkpoint.")

    # Final numbers come from the best epoch, not the last one.
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device,
                                 weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_epoch = int(best_checkpoint["epoch"])

    print("\n" + "=" * 78)
    print(f"Final validation using the best model (epoch {best_epoch})")

    final_metrics, per_image_rows = evaluate_model(
        model, val_loader, criterion, device,
        threshold=PREDICTION_THRESHOLD, compute_hd95=True,
        visualization_path=output_dir / "best_model_visualization.png",
        collect_per_image=True,
    )

    save_json({
        "best_epoch": best_epoch,
        "completed_epochs": len(history),
        "maximum_epochs": MAX_EPOCHS,
        "stopped_early": len(history) < MAX_EPOCHS,
        "total_training_minutes": total_seconds / 60.0,
        "final_best_model_validation": final_metrics,
        "run_config": run_config,
    }, output_dir / "final_validation_report.json")

    save_rows_csv(per_image_rows,
                  output_dir / "final_validation_per_image.csv")

    hd95_mean = final_metrics["hd95_pixels"]["mean"]
    print(
        f"\nTraining complete in {total_seconds / 60:.1f} minutes.\n"
        f"  best epoch           {best_epoch}\n"
        f"  Dice                 {final_metrics['dice']['mean']:.4f} "
        f"+/- {final_metrics['dice']['std']:.4f}\n"
        f"  IoU                  {final_metrics['iou']['mean']:.4f}\n"
        f"  thresholded Jaccard  "
        f"{final_metrics['thresholded_jaccard']['mean']:.4f}\n"
        f"  failure rate         "
        f"{100 * final_metrics['failure_rate']:.1f}%\n"
        f"  HD95                 "
        f"{'n/a' if hd95_mean is None else f'{hd95_mean:.2f} px'}\n"
        f"  output directory     {output_dir}"
    )
    return model, history


# ---------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------

def main() -> None:
    set_random_seed(RANDOM_SEED)

    print(f"Task 1 training | fold {DEVELOPMENT_FOLD} | "
          f"{IMAGE_SIZE}px | batch {BATCH_SIZE} | "
          f"up to {MAX_EPOCHS} epochs | seed {RANDOM_SEED}")

    train_dataset = LesionDataset(
        fold=DEVELOPMENT_FOLD, role="train",
        transform=build_train_transform(image_size=IMAGE_SIZE),
        include_task2=False,
    )
    val_dataset = LesionDataset(
        fold=DEVELOPMENT_FOLD, role="val",
        transform=build_val_transform(image_size=IMAGE_SIZE),
        include_task2=False,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("A dataset is empty — check the fold CSV.")

    print(f"{len(train_dataset)} training, {len(val_dataset)} validation")

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        generator=generator, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    train_model(train_loader, val_loader, OUTPUT_DIR)


if __name__ == "__main__":
    main()
