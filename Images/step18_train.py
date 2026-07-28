"""
Step 18 — Task 2 training loop (dermoscopic attribute detection).

Trains the five-channel U-Net from step 17. Structurally this mirrors
step 15, but attribute sparsity forces three real changes:

  1. Tversky + weighted BCE instead of Dice + plain BCE. Streaks appear
     in ~5% of images and cover ~0.04% of pixels even then, so predicting
     nothing scores well on plain BCE. Tversky penalises missed pixels
     harder than spurious ones; pos_weight rescales BCE by the measured
     imbalance.

  2. Per-attribute metrics throughout. One pooled number would be
     dominated by pigment network and would hide streaks failing.

  3. Per-attribute operating thresholds, tuned after training. 0.5 is
     arbitrary and badly wrong for rare attributes.

Class weights are measured over the whole training fold with validation
transforms — augmented, randomly sampled masks give noticeably different
rates run to run, and those rates feed straight into the loss.

No HD95: the briefing scores 95% Hausdorff on Task 1 only.

Task 1 checkpoint location is not defined here; step 17 owns it and this
file asks through the model config.

    python step18_train.py
"""

from __future__ import annotations
from step17_task2_training import (
    ATTRIBUTES,
    NUM_ATTRIBUTES,
    Task2ModelConfig,
    build_task2_model,
    select_device,
)
from step12_data_augmentation import (
    LesionDataset,
    build_train_transform,
    build_val_transform,
)
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

matplotlib.use("Agg")


# =====================================================================
# 0. Settings
# =====================================================================

SMOKE_TEST = False

IMAGE_SIZE = 256 if SMOKE_TEST else 384
EPOCHS = 2 if SMOKE_TEST else 30
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
FOLD = 0
RANDOM_SEED = 42
NUM_WORKERS = 0                 # required on MPS; anything else hangs

TRAIN_THRESHOLD = 0.5           # used during training and checkpointing
THRESHOLD_CANDIDATES = np.arange(0.10, 0.91, 0.05)

# Which sweep column picks each attribute's threshold.
#   dice_all — every image, correct silence on empty truth counts as 1.0.
#              Penalises both failure modes, so it is the default.
#   dice_pos — positive images only. Ignores false positives entirely and
#              will drive thresholds down; use only if you can justify it.
SELECTION_METRIC = "dice_all"

EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 1e-4
RESUME = True

TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7
SEG_LOSS_WEIGHT = 1.0
CLS_LOSS_WEIGHT = 0.5
BCE_WEIGHT = 0.5
MAX_POS_WEIGHT = 50.0

USE_TASK1_ENCODER = True
TASK1_IMAGE_SIZE = 384          # resolution the Task 1 run used

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_DIR = (PROJECT_ROOT / "outputs" / "task2_training"
              / f"fold_{FOLD}_{IMAGE_SIZE}px")

# Where earlier versions of this script wrote; checked when resuming.
LEGACY_OUTPUT_DIRS = [
    SCRIPT_DIR / "outputs" / "task2_results",
    PROJECT_ROOT / "outputs" / "task2_results",
]


# =====================================================================
# 1. Reproducibility
# =====================================================================

def set_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# =====================================================================
# 2. Locating the attribute masks in a batch
# =====================================================================

_STACKED_KEYS = [
    "task2_attributes", "task2_attribute_masks", "task2_masks",
    "task2", "attributes", "attribute_masks",
]


def find_attribute_layout(batch: dict):
    """Return ('stacked', key) or ('separate', [key, ...]) or raise."""
    for key in _STACKED_KEYS:
        value = batch.get(key)
        if (torch.is_tensor(value) and value.dim() == 4
                and value.shape[1] == NUM_ATTRIBUTES):
            return "stacked", key

    for prefix in ("", "task2_", "attribute_"):
        keys = [f"{prefix}{name}" for name in ATTRIBUTES]
        if all(torch.is_tensor(batch.get(key)) for key in keys):
            return "separate", keys

    raise KeyError(
        "Could not find the five Task 2 masks.\n"
        f"Available keys: {sorted(batch.keys())}"
    )


def get_attribute_masks(batch: dict, layout) -> torch.Tensor:
    """Return a [B, 5, H, W] float tensor in ATTRIBUTES order."""
    kind, keys = layout
    if kind == "stacked":
        masks = batch[keys]
    else:
        parts = []
        for key in keys:
            mask = batch[key]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            parts.append(mask)
        masks = torch.cat(parts, dim=1)
    return (masks > 0.5).float()


# =====================================================================
# 3. Loss
# =====================================================================

class AttributeSegLoss(nn.Module):
    """Weighted BCE plus Tversky, computed independently per attribute."""

    def __init__(self, pos_weight: torch.Tensor,
                 alpha: float = TVERSKY_ALPHA,
                 beta: float = TVERSKY_BETA,
                 bce_weight: float = BCE_WEIGHT):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.view(1, -1, 1, 1)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                smooth: float = 1.0) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probabilities = torch.sigmoid(logits)
        true_positive = (probabilities * targets).sum(dim=(2, 3))
        false_positive = (probabilities * (1.0 - targets)).sum(dim=(2, 3))
        false_negative = ((1.0 - probabilities) * targets).sum(dim=(2, 3))

        tversky = (true_positive + smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + smooth
        )
        tversky_loss = 1.0 - tversky.mean()

        return (self.bce_weight * bce_loss
                + (1.0 - self.bce_weight) * tversky_loss)


# =====================================================================
# 4. Class weights
# =====================================================================

def estimate_class_weights(loader: DataLoader, layout, cache_path: Path):
    """
    Measure attribute imbalance over the entire training fold.

    The loader must use validation transforms and shuffle=False. Measuring
    on augmented, randomly sampled masks gives materially different rates
    between runs, and those rates go straight into the loss weights.

    Cached, because a full pass costs several minutes and this runs again
    on every resume.
    """
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("attribute_order") == list(ATTRIBUTES) and \
                cached.get("image_size") == IMAGE_SIZE:
            print(f"\nUsing cached class weights ({cache_path.name}).")
            print_weight_table(
                np.array(cached["pixel_rate"]),
                np.array(cached["image_rate"]),
                np.array(cached["seg_weight"]),
                np.array(cached["cls_weight"]),
                cached["total_images"],
            )
            return (torch.tensor(cached["seg_weight"], dtype=torch.float32),
                    torch.tensor(cached["cls_weight"], dtype=torch.float32))

    positive_pixels = torch.zeros(NUM_ATTRIBUTES, dtype=torch.float64)
    positive_images = torch.zeros(NUM_ATTRIBUTES, dtype=torch.float64)
    total_pixels = 0
    total_images = 0

    print("\nMeasuring class weights over the full training fold...")

    for batch in loader:
        masks = get_attribute_masks(batch, layout)
        positive_pixels += masks.sum(dim=(0, 2, 3)).double()
        positive_images += (masks.sum(dim=(2, 3)) > 0).sum(dim=0).double()
        total_pixels += masks.shape[0] * masks.shape[2] * masks.shape[3]
        total_images += masks.shape[0]

    pixel_rate = (positive_pixels / max(total_pixels, 1)).clamp(min=1e-8)
    image_rate = (positive_images / max(total_images, 1)).clamp(min=1e-8)

    seg_weight = ((1.0 - pixel_rate) / pixel_rate).clamp(max=MAX_POS_WEIGHT)
    cls_weight = ((1.0 - image_rate) / image_rate).clamp(max=MAX_POS_WEIGHT)

    print_weight_table(pixel_rate.numpy(), image_rate.numpy(),
                       seg_weight.numpy(), cls_weight.numpy(), total_images)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "attribute_order": list(ATTRIBUTES),
        "image_size": IMAGE_SIZE,
        "total_images": total_images,
        "pixel_rate": pixel_rate.tolist(),
        "image_rate": image_rate.tolist(),
        "seg_weight": seg_weight.tolist(),
        "cls_weight": cls_weight.tolist(),
        "max_pos_weight": MAX_POS_WEIGHT,
    }, indent=2), encoding="utf-8")

    return seg_weight.float(), cls_weight.float()


def print_weight_table(pixel_rate, image_rate, seg_weight, cls_weight,
                       total_images) -> None:
    uncapped = (1.0 - pixel_rate) / np.maximum(pixel_rate, 1e-8)
    print(f"Measured on {total_images} training images:")
    print(f"{'attribute':<20}{'pixel %':>10}{'image %':>10}"
          f"{'seg w':>9}{'uncapped':>10}{'cls w':>9}")
    for index, name in enumerate(ATTRIBUTES):
        print(f"{name:<20}{100 * pixel_rate[index]:>9.4f}%"
              f"{100 * image_rate[index]:>9.2f}%"
              f"{seg_weight[index]:>9.2f}{uncapped[index]:>10.0f}"
              f"{cls_weight[index]:>9.2f}")


# =====================================================================
# 5. Validation
# =====================================================================
# Dice is undefined when an image has no ground truth for an attribute,
# which is the majority of cases. Two conventions, both reported:
#
#   dice_pos  positive images only. "When it's there, do we find it?"
#             The honest segmentation number; checkpoints select on it.
#   dice_all  every image, scoring empty-on-empty as 1.0. Higher, and
#             largely measures how often the model correctly stays quiet.
#
# fire% against true% is the clearest read on the failure mode: pinned at
# 100% means the model sprays positives, 0% means it has gone silent.

def make_threshold_tensor(thresholds, device) -> torch.Tensor:
    values = ([float(thresholds)] * NUM_ATTRIBUTES
              if np.isscalar(thresholds)
              else [float(value) for value in thresholds])
    return torch.tensor(values, dtype=torch.float32,
                        device=device).view(1, NUM_ATTRIBUTES, 1, 1)


def per_image_scores(probabilities, targets, threshold_tensor):
    """Per-image, per-channel Dice and IoU, empty-aware."""
    predictions = probabilities > threshold_tensor
    targets_binary = targets > 0.5

    intersection = (predictions & targets_binary).sum(dim=(2, 3)).float()
    prediction_sum = predictions.sum(dim=(2, 3)).float()
    target_sum = targets_binary.sum(dim=(2, 3)).float()

    denominator = prediction_sum + target_sum
    union = denominator - intersection

    # Both empty is a correct answer, not a zero.
    dice = torch.where(denominator > 0,
                       2.0 * intersection / denominator.clamp_min(1.0),
                       torch.ones_like(denominator))
    iou = torch.where(union > 0,
                      intersection / union.clamp_min(1.0),
                      torch.ones_like(union))

    return dice, iou, prediction_sum, target_sum


def evaluate(model, loader, layout, criterion, cls_criterion, device,
             thresholds=TRAIN_THRESHOLD):
    model.eval()
    threshold_tensor = make_threshold_tensor(thresholds, device)

    val_loss = 0.0
    dice_pos = [[] for _ in range(NUM_ATTRIBUTES)]
    dice_all = [[] for _ in range(NUM_ATTRIBUTES)]
    iou_pos = [[] for _ in range(NUM_ATTRIBUTES)]
    iou_all = [[] for _ in range(NUM_ATTRIBUTES)]
    fired = np.zeros(NUM_ATTRIBUTES)
    total_images = 0
    presence_true, presence_score = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = get_attribute_masks(batch, layout).to(device)

            seg_logits, cls_logits = model(images)
            target_presence = (targets.sum(dim=(2, 3)) > 0).float()

            loss = SEG_LOSS_WEIGHT * criterion(seg_logits, targets)
            if cls_logits is not None:
                loss = loss + CLS_LOSS_WEIGHT * cls_criterion(
                    cls_logits, target_presence)
                scores = torch.sigmoid(cls_logits)
            else:
                scores = torch.sigmoid(seg_logits).mean(dim=(2, 3))
            val_loss += loss.item()

            dice, iou, prediction_sum, target_sum = per_image_scores(
                torch.sigmoid(seg_logits), targets, threshold_tensor)

            dice_array = dice.cpu().numpy()
            iou_array = iou.cpu().numpy()
            has_truth = (target_sum > 0).cpu().numpy()
            fired += (prediction_sum > 0).cpu().numpy().sum(axis=0)
            total_images += images.shape[0]

            for channel in range(NUM_ATTRIBUTES):
                rows = has_truth[:, channel]
                dice_all[channel].extend(dice_array[:, channel].tolist())
                iou_all[channel].extend(iou_array[:, channel].tolist())
                dice_pos[channel].extend(dice_array[rows, channel].tolist())
                iou_pos[channel].extend(iou_array[rows, channel].tolist())

            presence_true.append(target_presence.cpu().numpy())
            presence_score.append(scores.cpu().numpy())

    presence_true = np.concatenate(presence_true)
    presence_score = np.concatenate(presence_score)

    # Average precision, not accuracy: always answering "absent" gives
    # 95% accuracy on streaks and an AP near zero.
    average_precision = [float("nan")] * NUM_ATTRIBUTES
    try:
        from sklearn.metrics import average_precision_score
        for channel in range(NUM_ATTRIBUTES):
            labels = presence_true[:, channel]
            if labels.min() != labels.max():
                average_precision[channel] = float(average_precision_score(
                    labels, presence_score[:, channel]))
    except ImportError:
        pass

    threshold_values = ([float(thresholds)] * NUM_ATTRIBUTES
                        if np.isscalar(thresholds) else list(thresholds))

    per_attribute = []
    for channel, name in enumerate(ATTRIBUTES):
        positive_count = len(dice_pos[channel])
        per_attribute.append({
            "attribute": name,
            "threshold": float(threshold_values[channel]),
            "dice_pos": (float(np.mean(dice_pos[channel]))
                         if positive_count else float("nan")),
            "iou_pos": (float(np.mean(iou_pos[channel]))
                        if positive_count else float("nan")),
            "dice_all": float(np.mean(dice_all[channel])),
            "iou_all": float(np.mean(iou_all[channel])),
            "base_all": (total_images - positive_count)
            / max(total_images, 1),
            "n_positive": positive_count,
            "true_rate": positive_count / max(total_images, 1),
            "fire_rate": float(fired[channel]) / max(total_images, 1),
            "presence_ap": average_precision[channel],
        })

    return val_loss / max(len(loader), 1), per_attribute


def print_attribute_table(per_attribute) -> None:
    print(f"  {'attribute':<20}{'thr':>6}{'Dice+':>9}{'IoU+':>9}"
          f"{'Dice(all)':>11}{'n+':>6}{'true%':>8}{'fire%':>8}{'AP':>9}")
    for row in per_attribute:
        print(f"  {row['attribute']:<20}{row['threshold']:>6.2f}"
              f"{row['dice_pos']:>9.4f}{row['iou_pos']:>9.4f}"
              f"{row['dice_all']:>11.4f}{row['n_positive']:>6}"
              f"{100 * row['true_rate']:>7.1f}%"
              f"{100 * row['fire_rate']:>7.1f}%"
              f"{row['presence_ap']:>9.4f}")


# =====================================================================
# 6. Threshold sweep
# =====================================================================

def sweep_thresholds(model, loader, layout, device) -> pd.DataFrame:
    """
    Score every candidate threshold on every validation image.

    Deliberately scores ALL images, not only the positives. A sweep that
    looks only at images where the attribute is present has no term for
    false positives, so it always favours a lower threshold and makes
    over-prediction worse while the reported number improves.
    """
    model.eval()

    dice_all_sum = np.zeros((len(THRESHOLD_CANDIDATES), NUM_ATTRIBUTES))
    dice_pos_sum = np.zeros((len(THRESHOLD_CANDIDATES), NUM_ATTRIBUTES))
    fired = np.zeros((len(THRESHOLD_CANDIDATES), NUM_ATTRIBUTES))
    positive_counts = np.zeros(NUM_ATTRIBUTES, dtype=np.int64)
    total_images = 0

    print("\nSweeping operating thresholds...")

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = get_attribute_masks(batch, layout).to(device)

            seg_logits, _ = model(images)
            probabilities = torch.sigmoid(seg_logits)

            has_truth = (targets.sum(dim=(2, 3)) > 0).cpu().numpy()
            positive_counts += has_truth.sum(axis=0)
            total_images += images.shape[0]

            for index, threshold in enumerate(THRESHOLD_CANDIDATES):
                tensor = make_threshold_tensor(float(threshold), device)
                dice, _, prediction_sum, _ = per_image_scores(
                    probabilities, targets, tensor)

                dice_array = dice.cpu().numpy()
                dice_all_sum[index] += dice_array.sum(axis=0)
                fired[index] += (prediction_sum > 0).cpu().numpy().sum(axis=0)

                for channel in range(NUM_ATTRIBUTES):
                    rows = has_truth[:, channel]
                    if rows.any():
                        dice_pos_sum[index, channel] += \
                            dice_array[rows, channel].sum()

    rows = []
    for index, threshold in enumerate(THRESHOLD_CANDIDATES):
        for channel, name in enumerate(ATTRIBUTES):
            rows.append({
                "attribute": name,
                "threshold": float(threshold),
                "dice_all": dice_all_sum[index, channel]
                / max(total_images, 1),
                "dice_pos": dice_pos_sum[index, channel]
                / max(positive_counts[channel], 1),
                "fire_rate": fired[index, channel] / max(total_images, 1),
                "true_rate": positive_counts[channel] / max(total_images, 1),
            })
    return pd.DataFrame(rows)


def select_thresholds(sweep: pd.DataFrame, metric: str = SELECTION_METRIC):
    """Pick the best threshold per attribute and report the trade-off."""
    chosen, report = [], {
        "attribute_order": list(ATTRIBUTES),
        "selection_metric": metric,
        "note": ("Tuned on the validation fold and reported on the same "
                 "fold, so these numbers are optimistic."),
        "thresholds": {},
    }

    print(f"\nTuned thresholds (selected on {metric}):")
    print(f"  {'attribute':<20}{'thr':>6}{'Dice(all)':>11}"
          f"{'Dice+':>9}{'fire%':>8}{'true%':>8}")

    for name in ATTRIBUTES:
        subset = sweep[sweep["attribute"] == name]
        best = subset.loc[subset[metric].idxmax()]
        chosen.append(float(best["threshold"]))

        report["thresholds"][name] = {
            "threshold": float(best["threshold"]),
            "dice_all": float(best["dice_all"]),
            "dice_pos": float(best["dice_pos"]),
            "fire_rate": float(best["fire_rate"]),
            "true_rate": float(best["true_rate"]),
        }

        print(f"  {name:<20}{best['threshold']:>6.2f}"
              f"{best['dice_all']:>11.4f}{best['dice_pos']:>9.4f}"
              f"{100 * best['fire_rate']:>7.1f}%"
              f"{100 * best['true_rate']:>7.1f}%")

    return chosen, report


# =====================================================================
# 7. Figures
# =====================================================================

def save_attribute_visual(image, targets, predictions, save_path) -> None:
    """Ground truth on the top row, prediction below, one column each."""
    image_array = image.permute(1, 2, 0).cpu().numpy()
    spread = image_array.max() - image_array.min()
    if spread > 0:
        image_array = (image_array - image_array.min()) / spread
    image_array = np.clip(image_array, 0, 1)

    figure, axes = plt.subplots(2, NUM_ATTRIBUTES + 1, figsize=(18, 6))
    axes[0, 0].imshow(image_array)
    axes[0, 0].set_title("Input", fontsize=9)

    for channel, name in enumerate(ATTRIBUTES):
        axes[0, channel + 1].imshow(targets[channel].cpu().numpy(),
                                    cmap="gray", vmin=0, vmax=1)
        axes[0, channel + 1].set_title(f"GT {name}", fontsize=8)
        axes[1, channel + 1].imshow(predictions[channel].cpu().numpy(),
                                    cmap="gray", vmin=0, vmax=1)
        axes[1, channel + 1].set_title(f"Pred {name}", fontsize=8)

    for axis in axes.ravel():
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def save_history_plot(history, save_path) -> None:
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=8)

    for name in ATTRIBUTES:
        column = f"dice_pos_{name}"
        if column in frame:
            axes[1].plot(frame["epoch"], frame[column], label=name)
    axes[1].plot(frame["epoch"], frame["mean_dice_pos"],
                 color="black", linewidth=2, label="mean")
    axes[1].set_title("Dice on positive images")
    axes[1].set_xlabel("epoch")
    axes[1].legend(fontsize=7)

    # Firing rate is the clearest read on collapse vs over-prediction.
    for name in ATTRIBUTES:
        column = f"fire_rate_{name}"
        if column in frame:
            axes[2].plot(frame["epoch"], 100 * frame[column], label=name)
    axes[2].set_title("Firing rate")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("% of images fired on")
    axes[2].set_ylim(0, 105)
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(figure)


# =====================================================================
# 8. Training
# =====================================================================

def find_resume_checkpoint(output_dir: Path):
    candidates = [output_dir / "task2_checkpoint_last.pth"]
    candidates += [directory / "task2_checkpoint_last.pth"
                   for directory in LEGACY_OUTPUT_DIRS]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_loaders():
    train_dataset = LesionDataset(
        fold=FOLD, role="train",
        transform=build_train_transform(image_size=IMAGE_SIZE),
        include_task2=True)
    val_dataset = LesionDataset(
        fold=FOLD, role="val",
        transform=build_val_transform(image_size=IMAGE_SIZE),
        include_task2=True)
    # Deterministic, unaugmented copy of the training fold, for weights.
    weight_dataset = LesionDataset(
        fold=FOLD, role="train",
        transform=build_val_transform(image_size=IMAGE_SIZE),
        include_task2=True)

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "worker_init_fn": seed_worker if NUM_WORKERS > 0 else None,
    }

    return (
        DataLoader(train_dataset, shuffle=True, generator=generator, **kwargs),
        DataLoader(val_dataset, shuffle=False, **kwargs),
        DataLoader(weight_dataset, shuffle=False, **kwargs),
    )


def train():
    set_random_seed(RANDOM_SEED)
    device = select_device()

    print(f"Training on {device}")
    print(f"Image size {IMAGE_SIZE}, batch {BATCH_SIZE}, "
          f"up to {EPOCHS} epochs, seed {RANDOM_SEED}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")

    train_loader, val_loader, weight_loader = build_loaders()
    print(f"{len(train_loader.dataset)} training, "
          f"{len(val_loader.dataset)} validation")

    probe = next(iter(weight_loader))
    layout = find_attribute_layout(probe)
    print(f"Attribute masks: {layout[0]} layout, shape "
          f"{tuple(get_attribute_masks(probe, layout).shape)}")
    print("Channel order: " + ", ".join(ATTRIBUTES))

    seg_pos_weight, cls_pos_weight = estimate_class_weights(
        weight_loader, layout, OUTPUT_DIR / "task2_class_weights.json")

    resume_path = find_resume_checkpoint(OUTPUT_DIR) if RESUME else None
    resuming = resume_path is not None

    # No warm start when resuming: the checkpoint already holds trained
    # encoder weights and Task 1's would overwrite them.
    config = Task2ModelConfig(
        auto_locate_task1=(USE_TASK1_ENCODER and not resuming),
        fold=FOLD, image_size=TASK1_IMAGE_SIZE)
    model = build_task2_model(config).to(device)

    criterion = AttributeSegLoss(seg_pos_weight.to(device))
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    start_epoch, best_dice, no_improvement, history = 0, -1.0, 0, []

    if resuming:
        print(f"\nResuming from {resume_path}")
        state = torch.load(resume_path, map_location=device,
                           weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"]
        best_dice = state["best_dice"]
        history = state["history"]
        no_improvement = state.get("no_improvement", 0)
        print(f"Continuing at epoch {start_epoch + 1}; "
              f"best Dice {best_dice:.4f}")

    run_started = time.time()

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        started = time.time()

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            targets = get_attribute_masks(batch, layout).to(device)

            optimizer.zero_grad(set_to_none=True)
            seg_logits, cls_logits = model(images)

            loss = SEG_LOSS_WEIGHT * criterion(seg_logits, targets)
            if cls_logits is not None:
                presence = (targets.sum(dim=(2, 3)) > 0).float()
                loss = loss + CLS_LOSS_WEIGHT * cls_criterion(
                    cls_logits, presence)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if step % 50 == 0:
                print(f"  epoch {epoch + 1} step {step}/{len(train_loader)} "
                      f"loss {loss.item():.4f} "
                      f"({time.time() - started:.0f}s)")

        val_loss, per_attribute = evaluate(
            model, val_loader, layout, criterion, cls_criterion, device,
            thresholds=TRAIN_THRESHOLD)

        mean_dice_pos = float(np.nanmean(
            [row["dice_pos"] for row in per_attribute]))
        mean_iou_pos = float(np.nanmean(
            [row["iou_pos"] for row in per_attribute]))

        elapsed = time.time() - started
        remaining = (EPOCHS - epoch - 1) * elapsed

        print(f"\nEpoch {epoch + 1}/{EPOCHS} | "
              f"train {train_loss / len(train_loader):.4f} | "
              f"val {val_loss:.4f} | Dice(pos) {mean_dice_pos:.4f} | "
              f"IoU(pos) {mean_iou_pos:.4f} | {elapsed:.0f}s | "
              f"~{remaining / 3600:.1f}h left")
        print_attribute_table(per_attribute)

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss,
            "mean_dice_pos": mean_dice_pos,
            "mean_iou_pos": mean_iou_pos,
            "lr": optimizer.param_groups[0]["lr"],
        }
        for row in per_attribute:
            name = row["attribute"]
            for field in ("dice_pos", "iou_pos", "dice_all",
                          "iou_all", "fire_rate"):
                record[f"{field}_{name}"] = row[field]
            record[f"ap_{name}"] = row["presence_ap"]
        history.append(record)

        pd.DataFrame(history).to_csv(
            OUTPUT_DIR / "task2_training_history.csv", index=False)
        save_history_plot(history, OUTPUT_DIR / "task2_training_curves.png")

        scheduler.step(mean_dice_pos)

        if mean_dice_pos > best_dice + EARLY_STOPPING_MIN_DELTA:
            best_dice = mean_dice_pos
            no_improvement = 0
            torch.save(model.state_dict(),
                       OUTPUT_DIR / "task2_best_model.pth")
            print(f"  New best model saved (Dice {best_dice:.4f})")
        else:
            no_improvement += 1

        model.eval()
        with torch.no_grad():
            sample = next(iter(val_loader))
            images = sample["image"].to(device)
            targets = get_attribute_masks(sample, layout).to(device)
            seg_logits, _ = model(images)
            predictions = (torch.sigmoid(seg_logits)
                           > TRAIN_THRESHOLD).float()
            save_attribute_visual(
                images[0], targets[0], predictions[0],
                OUTPUT_DIR / f"epoch_{epoch + 1:02d}_attributes.png")

        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_dice": best_dice,
            "no_improvement": no_improvement,
            "history": history,
            "seed": RANDOM_SEED,
        }, OUTPUT_DIR / "task2_checkpoint_last.pth")

        print(f"  Early-stop counter: {no_improvement}/"
              f"{EARLY_STOPPING_PATIENCE}")

        if no_improvement >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping triggered.")
            break

    # -----------------------------------------------------------------
    # Final evaluation, from the best checkpoint
    # -----------------------------------------------------------------
    best_model_path = OUTPUT_DIR / "task2_best_model.pth"
    if not best_model_path.is_file():
        raise FileNotFoundError(f"Best model not found: {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device,
                                     weights_only=True))

    sweep = sweep_thresholds(model, val_loader, layout, device)
    sweep.to_csv(OUTPUT_DIR / "task2_threshold_sweep.csv", index=False)

    best_thresholds, threshold_report = select_thresholds(sweep)
    (OUTPUT_DIR / "task2_best_thresholds.json").write_text(
        json.dumps(threshold_report, indent=2), encoding="utf-8")

    # Both operating points, so the report can quote the honest pair.
    results = {}
    for label, thresholds in (("default_0.5", TRAIN_THRESHOLD),
                              ("tuned", best_thresholds)):
        loss, metrics = evaluate(model, val_loader, layout, criterion,
                                 cls_criterion, device,
                                 thresholds=thresholds)
        print(f"\nFinal validation — {label} thresholds:")
        print_attribute_table(metrics)

        results[label] = {
            "validation_loss": loss,
            "mean_dice_pos": float(np.nanmean(
                [row["dice_pos"] for row in metrics])),
            "mean_iou_pos": float(np.nanmean(
                [row["iou_pos"] for row in metrics])),
            "mean_dice_all": float(np.nanmean(
                [row["dice_all"] for row in metrics])),
            "per_attribute": metrics,
        }
        pd.DataFrame(metrics).to_csv(
            OUTPUT_DIR / f"task2_final_metrics_{label}.csv", index=False)

    (OUTPUT_DIR / "task2_final_metrics.json").write_text(
        json.dumps({
            "selection_metric": SELECTION_METRIC,
            "thresholds_tuned_on": "validation fold (same fold as reported)",
            "results": results,
        }, indent=2), encoding="utf-8")

    torch.save(model.state_dict(), OUTPUT_DIR / "task2_final_model.pth")

    print(f"\nDone in {(time.time() - run_started) / 3600:.1f}h.")
    print(f"  mean Dice(pos) at 0.5   "
          f"{results['default_0.5']['mean_dice_pos']:.4f}")
    print(f"  mean Dice(pos) tuned    "
          f"{results['tuned']['mean_dice_pos']:.4f}")
    print(f"Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
