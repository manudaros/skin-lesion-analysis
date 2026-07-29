"""
Step 18 — Task 2 training loop (dermoscopic attribute detection).

Trains the five-channel U-Net from step 17. The core problem is sparsity:
streaks appear in ~5% of images and cover ~0.05% of pixels even then, so a
model that predicts nothing scores well on plain BCE, while one pushed too
hard the other way fires on every image. Four mechanisms address it:

  1. Tversky with alpha > beta, penalising false positives harder than
     false negatives.
  2. pos_weight from measured imbalance, sqrt-compressed and capped, so
     the rare attributes don't all collapse onto the same cap value.
  3. WeightedRandomSampler, so rare-positive images appear more often.
  4. Classification gating: the presence head vetoes the segmentation
     map. If the classifier says an attribute is absent, its mask is
     zeroed regardless of the pixels. This separates "is it present"
     from "where is it", and is what keeps the firing rate near the true
     presence rate.

All four push the same direction, so watch fire% against true%. If firing
drops well below the true rate the correction has overshot; relax the
Tversky asymmetry first, since gating now does that job.

Metrics are per attribute throughout. A pooled number would be dominated
by pigment network (~61% of images) and would hide streaks failing.

No HD95: the briefing scores 95% Hausdorff on Task 1 only.

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
from torch.utils.data import DataLoader, WeightedRandomSampler
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
EPOCHS = 2 if SMOKE_TEST else 35
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
FOLD = 0
SEED = 42
NUM_WORKERS = 0                 # required on MPS; anything else hangs

RESUME = False                  # False for the first run of a new recipe

SEG_LOSS_WEIGHT = 1.0
CLS_LOSS_WEIGHT = 0.5
BCE_WEIGHT = 0.5

# alpha > beta penalises false positives harder than false negatives.
TVERSKY_ALPHA = 0.6
TVERSKY_BETA = 0.4

MAX_SEG_POS_WEIGHT = 20.0
MAX_CLS_POS_WEIGHT = 50.0
MAX_SAMPLE_WEIGHT = 6.0

EARLY_STOP_PATIENCE = 8
EARLY_STOP_MIN_DELTA = 1e-4

DEFAULT_THRESHOLD = 0.5
THRESHOLD_CANDIDATES = np.arange(0.10, 0.91, 0.05)

# Checkpoint selection: segmentation stays the main objective, AP rewards
# reliable image-level presence, which is what Task 3 consumes.
SEG_SELECTION_WEIGHT = 0.7
AP_SELECTION_WEIGHT = 0.3

USE_TASK1_ENCODER = True
TASK1_IMAGE_SIZE = 384

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = (PROJECT_ROOT / "outputs" / "task2_training"
              / f"fold_{FOLD}_{IMAGE_SIZE}px")


# =====================================================================
# 1. Reproducibility
# =====================================================================

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =====================================================================
# 2. Attribute-mask access
# =====================================================================

_STACKED_KEYS = [
    "task2_attributes", "task2_attribute_masks", "task2_masks",
    "task2", "attributes", "attribute_masks",
]


def find_layout(batch: dict):
    """Return ('stacked', key) or ('separate', [key, ...]) or raise."""
    for key in _STACKED_KEYS:
        value = batch.get(key)
        if (torch.is_tensor(value) and value.ndim == 4
                and value.shape[1] == NUM_ATTRIBUTES):
            return "stacked", key

    for prefix in ("", "task2_", "attribute_"):
        keys = [f"{prefix}{name}" for name in ATTRIBUTES]
        if all(torch.is_tensor(batch.get(key)) for key in keys):
            return "separate", keys

    raise KeyError(
        "Task 2 masks were not found. "
        f"Available batch keys: {sorted(batch.keys())}"
    )


def get_masks(batch: dict, layout) -> torch.Tensor:
    """Return binary masks with shape [B, 5, H, W] in ATTRIBUTES order."""
    kind, keys = layout
    if kind == "stacked":
        masks = batch[keys]
    else:
        parts = []
        for key in keys:
            mask = batch[key]
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            parts.append(mask)
        masks = torch.cat(parts, dim=1)
    return (masks > 0.5).float()


# =====================================================================
# 3. Loss
# =====================================================================

class AttributeSegLoss(nn.Module):
    """Weighted BCE plus false-positive-aware Tversky."""

    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.view(1, -1, 1, 1))

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probabilities = torch.sigmoid(logits)
        true_positive = (probabilities * targets).sum(dim=(2, 3))
        false_positive = (probabilities * (1.0 - targets)).sum(dim=(2, 3))
        false_negative = ((1.0 - probabilities) * targets).sum(dim=(2, 3))

        tversky = (true_positive + 1.0) / (
            true_positive
            + TVERSKY_ALPHA * false_positive
            + TVERSKY_BETA * false_negative
            + 1.0
        )
        tversky_loss = 1.0 - tversky.mean()

        return BCE_WEIGHT * bce_loss + (1.0 - BCE_WEIGHT) * tversky_loss


# =====================================================================
# 4. Class weights and rare-positive sampling
# =====================================================================

def measure_training_distribution(loader: DataLoader, layout,
                                  cache_path: Path):
    """
    Measure attribute imbalance over the whole training fold.

    The loader must use validation transforms and shuffle=False: measuring
    on augmented, randomly sampled masks gives materially different rates
    between runs, and those rates feed straight into the loss.

    Returns segmentation and classification pos_weights, plus one sampling
    weight per training image so rare-positive images appear more often.

    Cached, because the full pass costs several minutes and would repeat
    on every restart. Delete the JSON to force a re-measure.
    """
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (cached.get("attribute_order") == list(ATTRIBUTES)
                and cached.get("image_size") == IMAGE_SIZE
                and len(cached.get("sample_weight", [])) > 0):
            print(f"\nUsing cached class statistics ({cache_path.name}).")
            print_weight_table(cached["per_attribute"],
                               cached["total_images"])
            return (
                torch.tensor(cached["seg_pos_weight"], dtype=torch.float32),
                torch.tensor(cached["cls_pos_weight"], dtype=torch.float32),
                torch.tensor(cached["sample_weight"], dtype=torch.float64),
                cached["per_attribute"],
            )

    positive_pixels = torch.zeros(NUM_ATTRIBUTES, dtype=torch.float64)
    positive_images = torch.zeros(NUM_ATTRIBUTES, dtype=torch.float64)
    presence_rows = []
    total_pixels = 0
    total_images = 0

    print("\nMeasuring class statistics over the full training fold...")

    for batch in loader:
        masks = get_masks(batch, layout)
        presence = (masks.sum(dim=(2, 3)) > 0).float()

        positive_pixels += masks.sum(dim=(0, 2, 3)).double()
        positive_images += presence.sum(dim=0).double()
        presence_rows.append(presence)

        total_pixels += masks.shape[0] * masks.shape[2] * masks.shape[3]
        total_images += masks.shape[0]

    presence_matrix = torch.cat(presence_rows, dim=0)
    pixel_rate = (positive_pixels / total_pixels).clamp_min(1e-8)
    image_rate = (positive_images / total_images).clamp_min(1e-8)

    raw_seg_weight = (1.0 - pixel_rate) / pixel_rate
    raw_cls_weight = (1.0 - image_rate) / image_rate

    # sqrt compresses the extremes: without it, four of five attributes
    # sit on the cap at the same value despite very different rarity.
    seg_weight = torch.sqrt(raw_seg_weight).clamp(
        max=MAX_SEG_POS_WEIGHT).float()
    cls_weight = raw_cls_weight.clamp(max=MAX_CLS_POS_WEIGHT).float()

    # An image containing rare attributes is sampled more often.
    rarity = torch.sqrt(1.0 / image_rate.float()).clamp(max=4.0)
    sample_weight = (
        1.0 + (presence_matrix * (rarity - 1.0)).sum(dim=1)
    ).clamp(max=MAX_SAMPLE_WEIGHT).double()

    per_attribute = [{
        "attribute": name,
        "pixel_rate": float(pixel_rate[index]),
        "image_rate": float(image_rate[index]),
        "raw_seg_pos_weight": float(raw_seg_weight[index]),
        "seg_pos_weight": float(seg_weight[index]),
        "cls_pos_weight": float(cls_weight[index]),
    } for index, name in enumerate(ATTRIBUTES)]

    print_weight_table(per_attribute, total_images)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "attribute_order": list(ATTRIBUTES),
        "image_size": IMAGE_SIZE,
        "total_images": total_images,
        "per_attribute": per_attribute,
        "seg_pos_weight": seg_weight.tolist(),
        "cls_pos_weight": cls_weight.tolist(),
        "sample_weight": sample_weight.tolist(),
        "max_seg_pos_weight": MAX_SEG_POS_WEIGHT,
        "sqrt_compressed": True,
    }, indent=2), encoding="utf-8")

    return seg_weight, cls_weight, sample_weight, per_attribute


def print_weight_table(per_attribute, total_images: int) -> None:
    print(f"Measured on {total_images} training images:")
    print(f"{'attribute':<20}{'pixel%':>10}{'image%':>10}"
          f"{'raw w':>10}{'seg w':>9}{'cls w':>9}")
    for row in per_attribute:
        print(f"{row['attribute']:<20}"
              f"{100 * row['pixel_rate']:>9.4f}%"
              f"{100 * row['image_rate']:>9.2f}%"
              f"{row['raw_seg_pos_weight']:>10.0f}"
              f"{row['seg_pos_weight']:>9.2f}"
              f"{row['cls_pos_weight']:>9.2f}")


# =====================================================================
# 5. Metrics
# =====================================================================

def make_threshold_tensor(values, device) -> torch.Tensor:
    if np.isscalar(values):
        values = [float(values)] * NUM_ATTRIBUTES
    return torch.tensor(values, dtype=torch.float32,
                        device=device).view(1, NUM_ATTRIBUTES, 1, 1)


def binary_metrics(predictions: torch.Tensor, targets: torch.Tensor):
    """Per-image, per-attribute Dice and IoU. Empty-on-empty scores 1.0."""
    intersection = (predictions & targets).sum(dim=(2, 3)).float()
    prediction_sum = predictions.sum(dim=(2, 3)).float()
    target_sum = targets.sum(dim=(2, 3)).float()

    denominator = prediction_sum + target_sum
    union = denominator - intersection

    dice = torch.where(denominator > 0,
                       2.0 * intersection / denominator.clamp_min(1.0),
                       torch.ones_like(denominator))
    iou = torch.where(union > 0,
                      intersection / union.clamp_min(1.0),
                      torch.ones_like(union))

    return dice, iou, prediction_sum, target_sum


def fallback_presence_score(probabilities: torch.Tensor) -> torch.Tensor:
    """Top 5% of map probabilities, used only if there is no cls head."""
    flattened = probabilities.flatten(start_dim=2)
    top_count = max(1, flattened.shape[2] // 20)
    return flattened.topk(top_count, dim=2).values.mean(dim=2)


def safe_mean(values) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def presence_scores_from(model_output, seg_probabilities):
    cls_logits = model_output
    if cls_logits is None:
        return fallback_presence_score(seg_probabilities)
    return torch.sigmoid(cls_logits)


def evaluate(model, loader, layout, seg_criterion, cls_criterion, device,
             pixel_thresholds=DEFAULT_THRESHOLD,
             presence_thresholds=DEFAULT_THRESHOLD,
             gate: bool = True):
    """
    Validation metrics, per attribute.

    seg_dice_pos  Dice on positive images BEFORE classification gating.
    dice_pos      Dice on positive images after gating — the headline.
    dice_all      every image, empty-on-empty scoring 1.0.
    fire_rate     % of images predicted non-empty, against true_rate.

    gate=False evaluates the raw segmentation output, which is the
    baseline the gated numbers should be reported against.
    """
    model.eval()

    pixel_tensor = make_threshold_tensor(pixel_thresholds, device)
    presence_tensor = make_threshold_tensor(
        presence_thresholds, device).view(1, NUM_ATTRIBUTES)

    raw_dice_pos = [[] for _ in ATTRIBUTES]
    final_dice_pos = [[] for _ in ATTRIBUTES]
    final_iou_pos = [[] for _ in ATTRIBUTES]
    dice_all = [[] for _ in ATTRIBUTES]
    iou_all = [[] for _ in ATTRIBUTES]

    fired = np.zeros(NUM_ATTRIBUTES)
    presence_labels, presence_probabilities = [], []
    total_images = 0
    validation_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = get_masks(batch, layout).to(device)

            seg_logits, cls_logits = model(images)
            target_presence = (targets.sum(dim=(2, 3)) > 0).float()

            loss = SEG_LOSS_WEIGHT * seg_criterion(seg_logits, targets)
            seg_probabilities = torch.sigmoid(seg_logits)

            if cls_logits is not None:
                loss = loss + CLS_LOSS_WEIGHT * cls_criterion(
                    cls_logits, target_presence)
            cls_probabilities = presence_scores_from(
                cls_logits, seg_probabilities)
            validation_loss += loss.item()

            target_masks = targets > 0.5
            raw_predictions = seg_probabilities > pixel_tensor

            if gate:
                predicted_presence = cls_probabilities > presence_tensor
                final_predictions = (
                    raw_predictions
                    & predicted_presence[:, :, None, None]
                )
            else:
                final_predictions = raw_predictions

            raw_dice, _, _, target_sum = binary_metrics(
                raw_predictions, target_masks)
            final_dice, final_iou, prediction_sum, _ = binary_metrics(
                final_predictions, target_masks)

            raw_dice = raw_dice.cpu().numpy()
            final_dice = final_dice.cpu().numpy()
            final_iou = final_iou.cpu().numpy()
            has_truth = (target_sum > 0).cpu().numpy()

            fired += (prediction_sum > 0).cpu().numpy().sum(axis=0)
            total_images += images.shape[0]

            presence_labels.append(target_presence.cpu().numpy())
            presence_probabilities.append(cls_probabilities.cpu().numpy())

            for channel in range(NUM_ATTRIBUTES):
                rows = has_truth[:, channel]
                raw_dice_pos[channel].extend(
                    raw_dice[rows, channel].tolist())
                final_dice_pos[channel].extend(
                    final_dice[rows, channel].tolist())
                final_iou_pos[channel].extend(
                    final_iou[rows, channel].tolist())
                dice_all[channel].extend(final_dice[:, channel].tolist())
                iou_all[channel].extend(final_iou[:, channel].tolist())

    true_presence = np.concatenate(presence_labels)
    presence_probabilities = np.concatenate(presence_probabilities)

    # Average precision, not accuracy: always answering "absent" gives
    # 95% accuracy on streaks and an AP near zero.
    average_precision = [float("nan")] * NUM_ATTRIBUTES
    try:
        from sklearn.metrics import average_precision_score
        for channel in range(NUM_ATTRIBUTES):
            labels = true_presence[:, channel]
            if np.unique(labels).size >= 2:
                average_precision[channel] = float(average_precision_score(
                    labels, presence_probabilities[:, channel]))
    except ImportError:
        pass

    pixel_values = ([float(pixel_thresholds)] * NUM_ATTRIBUTES
                    if np.isscalar(pixel_thresholds)
                    else list(pixel_thresholds))
    presence_values = ([float(presence_thresholds)] * NUM_ATTRIBUTES
                       if np.isscalar(presence_thresholds)
                       else list(presence_thresholds))

    rows = []
    for channel, name in enumerate(ATTRIBUTES):
        labels = true_presence[:, channel].astype(bool)
        predicted = (presence_probabilities[:, channel]
                     > presence_values[channel])
        precision, recall, f1 = precision_recall_f1(predicted, labels)

        positive_count = len(final_dice_pos[channel])
        rows.append({
            "attribute": name,
            "presence_threshold": float(presence_values[channel]),
            "pixel_threshold": float(pixel_values[channel]),
            "gated": bool(gate),
            "seg_dice_pos": safe_mean(raw_dice_pos[channel]),
            "dice_pos": safe_mean(final_dice_pos[channel]),
            "iou_pos": safe_mean(final_iou_pos[channel]),
            "dice_all": safe_mean(dice_all[channel]),
            "iou_all": safe_mean(iou_all[channel]),
            "n_positive": positive_count,
            "true_rate": positive_count / max(total_images, 1),
            "fire_rate": float(fired[channel]) / max(total_images, 1),
            "presence_ap": average_precision[channel],
            "presence_precision": precision,
            "presence_recall": recall,
            "presence_f1": f1,
        })

    return validation_loss / max(len(loader), 1), rows


def precision_recall_f1(predicted: np.ndarray, labels: np.ndarray):
    true_positive = int(np.sum(predicted & labels))
    false_positive = int(np.sum(predicted & ~labels))
    false_negative = int(np.sum(~predicted & labels))

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return float(precision), float(recall), float(f1)


def print_metrics(rows) -> None:
    print(f"  {'attribute':<20}{'preT':>6}{'pixT':>6}{'SegD+':>8}"
          f"{'Dice+':>8}{'IoU+':>8}{'true%':>8}{'fire%':>8}"
          f"{'AP':>8}{'F1':>8}")
    for row in rows:
        print(f"  {row['attribute']:<20}"
              f"{row['presence_threshold']:>6.2f}"
              f"{row['pixel_threshold']:>6.2f}"
              f"{row['seg_dice_pos']:>8.4f}{row['dice_pos']:>8.4f}"
              f"{row['iou_pos']:>8.4f}"
              f"{100 * row['true_rate']:>7.1f}%"
              f"{100 * row['fire_rate']:>7.1f}%"
              f"{row['presence_ap']:>8.4f}{row['presence_f1']:>8.4f}")


# =====================================================================
# 6. Threshold tuning
# =====================================================================
# Tuned on the validation fold and reported on the same fold, so the
# tuned numbers are optimistic — ten free parameters fitted to the data
# being scored. The final report therefore also gives the untuned,
# ungated baseline. State both in the write-up.

def tune_presence_thresholds(model, loader, layout, device):
    """Image-level F1 per attribute. Gating decides presence, so F1 is
    the right objective: it balances missing the attribute against
    hallucinating it."""
    model.eval()
    all_labels, all_scores = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = get_masks(batch, layout).to(device)

            seg_logits, cls_logits = model(images)
            scores = presence_scores_from(
                cls_logits, torch.sigmoid(seg_logits))

            all_labels.append(
                (targets.sum(dim=(2, 3)) > 0).cpu().numpy())
            all_scores.append(scores.cpu().numpy())

    labels = np.concatenate(all_labels).astype(bool)
    scores = np.concatenate(all_scores)

    chosen = []
    for channel, name in enumerate(ATTRIBUTES):
        best_threshold, best_f1 = DEFAULT_THRESHOLD, -1.0
        for threshold in THRESHOLD_CANDIDATES:
            predicted = scores[:, channel] > float(threshold)
            _, _, f1 = precision_recall_f1(predicted, labels[:, channel])
            if f1 > best_f1:
                best_threshold, best_f1 = float(threshold), f1

        chosen.append(best_threshold)
        print(f"  {name:<20} presence={best_threshold:.2f}, "
              f"F1={best_f1:.4f}")
    return chosen


def tune_pixel_thresholds(model, loader, layout, device):
    """Dice on positive images only. False positives on empty images are
    gating's problem, not this threshold's."""
    model.eval()
    dice_sums = np.zeros((len(THRESHOLD_CANDIDATES), NUM_ATTRIBUTES))
    positive_counts = np.zeros(NUM_ATTRIBUTES, dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = get_masks(batch, layout).to(device).bool()

            seg_logits, _ = model(images)
            probabilities = torch.sigmoid(seg_logits)
            target_sum = targets.sum(dim=(2, 3))

            for channel in range(NUM_ATTRIBUTES):
                rows = target_sum[:, channel] > 0
                count = int(rows.sum().item())
                if count == 0:
                    continue
                positive_counts[channel] += count

                channel_probabilities = probabilities[rows, channel]
                channel_targets = targets[rows, channel]
                target_pixels = channel_targets.sum(dim=(1, 2)).float()

                for index, threshold in enumerate(THRESHOLD_CANDIDATES):
                    predicted = channel_probabilities > float(threshold)
                    intersection = (predicted & channel_targets).sum(
                        dim=(1, 2)).float()
                    predicted_pixels = predicted.sum(dim=(1, 2)).float()
                    dice = (2.0 * intersection) / (
                        predicted_pixels + target_pixels + 1e-8)
                    dice_sums[index, channel] += dice.sum().item()

    mean_dice = dice_sums / np.maximum(positive_counts[None, :], 1)
    best_indices = np.argmax(mean_dice, axis=0)

    chosen = []
    for channel, name in enumerate(ATTRIBUTES):
        threshold = float(THRESHOLD_CANDIDATES[best_indices[channel]])
        chosen.append(threshold)
        print(f"  {name:<20} pixel={threshold:.2f}, Dice(pos)="
              f"{mean_dice[best_indices[channel], channel]:.4f}")
    return chosen


# =====================================================================
# 7. Figures
# =====================================================================

def save_history(history) -> None:
    frame = pd.DataFrame(history)
    frame.to_csv(OUTPUT_DIR / "task2_training_history.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=8)

    axes[1].plot(frame["epoch"], frame["macro_seg_dice_pos"],
                 label="segmentation Dice")
    axes[1].plot(frame["epoch"], frame["macro_dice_pos"],
                 label="Dice after gating")
    axes[1].plot(frame["epoch"], frame["macro_presence_ap"],
                 label="presence AP")
    axes[1].plot(frame["epoch"], frame["selection_score"],
                 color="black", linewidth=2, label="selection score")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("epoch")
    axes[1].legend(fontsize=7)

    # Firing rate against true rate is the clearest read on whether the
    # anti-over-prediction machinery has overshot into silence.
    for name in ATTRIBUTES:
        column = f"fire_rate_{name}"
        if column in frame:
            axes[2].plot(frame["epoch"], 100 * frame[column], label=name)
        true_column = f"true_rate_{name}"
        if true_column in frame:
            axes[2].axhline(100 * frame[true_column].iloc[-1],
                            linestyle=":", linewidth=0.8, alpha=0.4)
    axes[2].set_title("Firing rate (dotted: true presence rate)")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("% of images fired on")
    axes[2].set_ylim(0, 105)
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task2_training_curves.png", dpi=130)
    plt.close(figure)


def save_visual(image, targets, predictions, save_path: Path) -> None:
    """Ground truth on the top row, prediction below, one column each."""
    array = image.permute(1, 2, 0).cpu().numpy()
    array = (array - array.min()) / max(array.max() - array.min(), 1e-8)

    figure, axes = plt.subplots(2, NUM_ATTRIBUTES + 1, figsize=(18, 6))
    axes[0, 0].imshow(np.clip(array, 0, 1))
    axes[0, 0].set_title("Input", fontsize=9)

    for channel, name in enumerate(ATTRIBUTES):
        axes[0, channel + 1].imshow(targets[channel].cpu(), cmap="gray",
                                    vmin=0, vmax=1)
        axes[0, channel + 1].set_title(f"GT {name}", fontsize=8)
        axes[1, channel + 1].imshow(predictions[channel].cpu(), cmap="gray",
                                    vmin=0, vmax=1)
        axes[1, channel + 1].set_title(f"Pred {name}", fontsize=8)

    for axis in axes.ravel():
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(figure)


# =====================================================================
# 8. Training
# =====================================================================

def train():
    set_seed(SEED)
    device = select_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training on {device}")
    print(f"Image size {IMAGE_SIZE}, batch {BATCH_SIZE}, "
          f"up to {EPOCHS} epochs, seed {SEED}")
    print(f"Output: {OUTPUT_DIR}")

    train_dataset = LesionDataset(
        fold=FOLD, role="train",
        transform=build_train_transform(image_size=IMAGE_SIZE),
        include_task2=True)
    val_dataset = LesionDataset(
        fold=FOLD, role="val",
        transform=build_val_transform(image_size=IMAGE_SIZE),
        include_task2=True)
    # Unaugmented copy of the training fold, for statistics only.
    weight_dataset = LesionDataset(
        fold=FOLD, role="train",
        transform=build_val_transform(image_size=IMAGE_SIZE),
        include_task2=True)

    weight_loader = DataLoader(weight_dataset, batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)

    layout = find_layout(next(iter(weight_loader)))
    print("Channel order: " + ", ".join(ATTRIBUTES))

    seg_pos_weight, cls_pos_weight, sample_weight, _ = \
        measure_training_distribution(
            weight_loader, layout,
            OUTPUT_DIR / "task2_class_weights.json")

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(SEED)
    sampler = WeightedRandomSampler(
        weights=sample_weight, num_samples=len(sample_weight),
        replacement=True, generator=sampler_generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=NUM_WORKERS)

    print(f"{len(train_dataset)} training, {len(val_dataset)} validation")

    checkpoint_path = OUTPUT_DIR / "task2_checkpoint_last.pth"
    resuming = RESUME and checkpoint_path.is_file()

    # No warm start when resuming: the checkpoint already holds trained
    # encoder weights and Task 1's would overwrite them.
    config = Task2ModelConfig(
        auto_locate_task1=(USE_TASK1_ENCODER and not resuming),
        fold=FOLD, image_size=TASK1_IMAGE_SIZE)
    model = build_task2_model(config).to(device)

    seg_criterion = AttributeSegLoss(seg_pos_weight.to(device))
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    start_epoch, best_score, no_improvement, history = 0, -1.0, 0, []

    if resuming:
        state = torch.load(checkpoint_path, map_location=device,
                           weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"]
        best_score = state.get("best_score", -1.0)
        no_improvement = state.get("no_improvement", 0)
        history = state.get("history", [])
        print(f"Resuming at epoch {start_epoch + 1}")

    run_start = time.time()

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_start = time.time()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            targets = get_masks(batch, layout).to(device)

            optimizer.zero_grad(set_to_none=True)
            seg_logits, cls_logits = model(images)

            loss = SEG_LOSS_WEIGHT * seg_criterion(seg_logits, targets)
            if cls_logits is not None:
                presence = (targets.sum(dim=(2, 3)) > 0).float()
                loss = loss + CLS_LOSS_WEIGHT * cls_criterion(
                    cls_logits, presence)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if step % 50 == 0:
                print(f"  epoch {epoch + 1} step {step}/"
                      f"{len(train_loader)} loss {loss.item():.4f} "
                      f"({time.time() - epoch_start:.0f}s)")

        train_loss /= len(train_loader)

        validation_loss, rows = evaluate(
            model, val_loader, layout, seg_criterion, cls_criterion, device)

        macro_seg_dice = float(np.nanmean(
            [row["seg_dice_pos"] for row in rows]))
        macro_dice = float(np.nanmean([row["dice_pos"] for row in rows]))
        macro_ap = float(np.nanmean([row["presence_ap"] for row in rows]))
        selection_score = (SEG_SELECTION_WEIGHT * macro_seg_dice
                           + AP_SELECTION_WEIGHT * macro_ap)

        scheduler.step(selection_score)
        epoch_seconds = time.time() - epoch_start
        remaining = (EPOCHS - epoch - 1) * epoch_seconds

        print(f"\nEpoch {epoch + 1}/{EPOCHS} | train {train_loss:.4f} | "
              f"val {validation_loss:.4f} | seg Dice(pos) "
              f"{macro_seg_dice:.4f} | gated Dice(pos) {macro_dice:.4f} | "
              f"AP {macro_ap:.4f} | score {selection_score:.4f} | "
              f"{epoch_seconds:.0f}s | ~{remaining / 3600:.1f}h left")
        print_metrics(rows)

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": validation_loss,
            "macro_seg_dice_pos": macro_seg_dice,
            "macro_dice_pos": macro_dice,
            "macro_presence_ap": macro_ap,
            "selection_score": selection_score,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        for row in rows:
            name = row["attribute"]
            for field in ("seg_dice_pos", "dice_pos", "iou_pos",
                          "fire_rate", "true_rate", "presence_ap",
                          "presence_f1"):
                record[f"{field}_{name}"] = row[field]
        history.append(record)
        save_history(history)

        if selection_score > best_score + EARLY_STOP_MIN_DELTA:
            best_score = selection_score
            no_improvement = 0
            torch.save(model.state_dict(),
                       OUTPUT_DIR / "task2_best_model.pth")
            print(f"  New best model saved (score {best_score:.4f})")
        else:
            no_improvement += 1

        save_epoch_visual(model, val_loader, layout, device,
                          OUTPUT_DIR / f"epoch_{epoch + 1:02d}_attributes.png")

        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_score": best_score,
            "no_improvement": no_improvement,
            "history": history,
            "seed": SEED,
            "tversky_alpha": TVERSKY_ALPHA,
            "tversky_beta": TVERSKY_BETA,
        }, checkpoint_path)

        print(f"  Early-stop counter: {no_improvement}/"
              f"{EARLY_STOP_PATIENCE}")

        if no_improvement >= EARLY_STOP_PATIENCE:
            print("\nEarly stopping triggered.")
            break

    finalise(model, val_loader, layout, seg_criterion, cls_criterion,
             device, run_start)


def save_epoch_visual(model, loader, layout, device, save_path) -> None:
    model.eval()
    with torch.no_grad():
        sample = next(iter(loader))
        images = sample["image"].to(device)
        targets = get_masks(sample, layout).to(device)
        seg_logits, cls_logits = model(images)

        seg_probabilities = torch.sigmoid(seg_logits)
        cls_probabilities = presence_scores_from(
            cls_logits, seg_probabilities)

        predictions = (
            (seg_probabilities > DEFAULT_THRESHOLD)
            & (cls_probabilities > DEFAULT_THRESHOLD)[:, :, None, None]
        )
        save_visual(images[0], targets[0], predictions[0], save_path)


def finalise(model, val_loader, layout, seg_criterion, cls_criterion,
             device, run_start) -> None:
    """Reload the best checkpoint, tune thresholds, report both settings."""
    best_model_path = OUTPUT_DIR / "task2_best_model.pth"
    if not best_model_path.is_file():
        raise FileNotFoundError(f"Best model not found: {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device,
                                     weights_only=True))

    print("\nTuning presence thresholds on image-level F1...")
    presence_thresholds = tune_presence_thresholds(
        model, val_loader, layout, device)

    print("\nTuning pixel thresholds on Dice over positive images...")
    pixel_thresholds = tune_pixel_thresholds(
        model, val_loader, layout, device)

    results = {}

    # Baseline: no gating, both thresholds at 0.5.
    baseline_loss, baseline_rows = evaluate(
        model, val_loader, layout, seg_criterion, cls_criterion, device,
        gate=False)
    print("\nBaseline — ungated, threshold 0.5:")
    print_metrics(baseline_rows)
    results["baseline_ungated_0.5"] = summarise(baseline_loss, baseline_rows)

    # Gated, with tuned thresholds.
    tuned_loss, tuned_rows = evaluate(
        model, val_loader, layout, seg_criterion, cls_criterion, device,
        pixel_thresholds=pixel_thresholds,
        presence_thresholds=presence_thresholds, gate=True)
    print("\nGated, tuned thresholds:")
    print_metrics(tuned_rows)
    results["gated_tuned"] = summarise(tuned_loss, tuned_rows)

    pd.DataFrame(baseline_rows).to_csv(
        OUTPUT_DIR / "task2_final_metrics_baseline.csv", index=False)
    pd.DataFrame(tuned_rows).to_csv(
        OUTPUT_DIR / "task2_final_metrics_tuned.csv", index=False)

    (OUTPUT_DIR / "task2_best_thresholds.json").write_text(json.dumps({
        "attribute_order": list(ATTRIBUTES),
        "presence_threshold_metric": "image-level F1",
        "pixel_threshold_metric": "Dice on positive validation images",
        "tuned_on": "validation fold — same fold as reported, so the "
                    "tuned figures are optimistic",
        "presence_thresholds": dict(zip(ATTRIBUTES, presence_thresholds)),
        "pixel_thresholds": dict(zip(ATTRIBUTES, pixel_thresholds)),
    }, indent=2), encoding="utf-8")

    (OUTPUT_DIR / "task2_final_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    save_epoch_visual(model, val_loader, layout, device,
                      OUTPUT_DIR / "task2_best_model_attributes.png")

    print(f"\nDone in {(time.time() - run_start) / 3600:.1f}h.")
    print(f"  mean Dice(pos) ungated @0.5  "
          f"{results['baseline_ungated_0.5']['macro_dice_pos']:.4f}")
    print(f"  mean Dice(pos) gated + tuned "
          f"{results['gated_tuned']['macro_dice_pos']:.4f}")
    print(f"Outputs in {OUTPUT_DIR}")


def summarise(loss: float, rows) -> dict:
    return {
        "validation_loss": loss,
        "macro_seg_dice_pos": float(np.nanmean(
            [row["seg_dice_pos"] for row in rows])),
        "macro_dice_pos": float(np.nanmean(
            [row["dice_pos"] for row in rows])),
        "macro_iou_pos": float(np.nanmean(
            [row["iou_pos"] for row in rows])),
        "macro_dice_all": float(np.nanmean(
            [row["dice_all"] for row in rows])),
        "macro_presence_ap": float(np.nanmean(
            [row["presence_ap"] for row in rows])),
        "macro_presence_f1": float(np.nanmean(
            [row["presence_f1"] for row in rows])),
        "per_attribute": rows,
    }


if __name__ == "__main__":
    train()
