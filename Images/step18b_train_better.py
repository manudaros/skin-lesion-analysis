"""
Step 18b — Task 2 at high resolution on predicted lesion crops.

Written to fix the sparse attributes. Milia-like cysts and streaks scored
near 0.1 Dice at 384px full-image, and the cause is resolution rather
than tuning: a cyst is a handful of bright pixels, and after four encoder
downsamples it occupies a fraction of one feature-map cell. Four changes:

  1. Crop to the predicted Task 1 lesion box, then resize to 512. Small
     structures get several times the pixels. The crop comes from
     predicted masks, never ground truth, so nothing leaks.
  2. Attribute-specific losses. clDice for the curvilinear attributes
     (negative network, streaks), focal for the blob-like ones (milia,
     globules), Dice everywhere.
  3. Soft classification gating. Hard gating multiplies a mask by a
     binary decision, so a classifier false negative erases the mask and
     that image scores Dice 0 — and since Dice(pos) averages only over
     positive images, every such error lands on the headline number.
     Scaling instead of zeroing removes that.
  4. Crop and augmentation retries, so a transform can never silently
     delete a tiny positive target.

IMPORTANT — metrics here are computed on the CROP, not the full image.
Cropping removes background where false positives would have counted, so
Dice is higher than a full-image number for the same model quality. These
figures are NOT comparable with the 384px full-image run. Say so in the
report, or map predictions back to full coordinates before scoring.

    python step18b_task2_highres.py
"""

from __future__ import annotations
from step17_task2_training import (
    ATTRIBUTES,
    NUM_ATTRIBUTES,
    Task2ModelConfig,
    build_task2_model,
    locate_task1_checkpoint,
    select_device,
)
from step14_task1_training import build_task1_model
from step12_data_augmentation import LesionDataset, ResizeLongestSideAndPad
from torchvision.transforms.v2 import functional as VF
from torchvision.transforms import InterpolationMode, v2
from torchvision import tv_tensors
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from scipy.ndimage import label as connected_components
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

matplotlib.use("Agg")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

EXPECTED_ATTRIBUTES = ("pigment_network", "negative_network", "streaks",
                       "milia_like_cysts", "globules")
if tuple(ATTRIBUTES) != EXPECTED_ATTRIBUTES:
    raise ValueError(f"Attribute order changed: {tuple(ATTRIBUTES)}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =====================================================================
# 0. Configuration
# =====================================================================

@dataclass(frozen=True)
class Config:
    fold: int = 0
    image_size: int = 512
    batch_size: int = 4
    accumulation_steps: int = 2      # effective batch 8
    max_epochs: int = 30
    num_workers: int = 0
    seed: int = 42

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0

    classification_loss_weight: float = 0.20
    max_seg_pos_weight: float = 20.0
    max_cls_pos_weight: float = 10.0
    focal_gamma: float = 2.0
    cldice_iterations: int = 5

    # Task 1 crop and soft ROI.
    task1_size: int = 384
    task1_threshold: float = 0.50
    val_crop_margin: float = 0.25
    train_crop_margin_min: float = 0.18
    train_crop_margin_max: float = 0.30
    train_crop_jitter: float = 0.04
    transform_retries: int = 3
    roi_dilation: int = 15
    outside_roi_scale: float = 0.25

    # Per-attribute soft gate strength. The classifier may attenuate a
    # weak mask but never erases it.
    gate_strengths: tuple[float, ...] = (0.20, 0.55, 0.65, 0.25, 0.25)

    max_sample_weight: float = 4.0
    max_rarity_boost: float = 3.0

    component_percentile: float = 5.0
    component_percentile_scale: float = 0.50
    component_area_floors: tuple[int, ...] = (12, 4, 2, 2, 3)

    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    min_learning_rate: float = 1e-6
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    training_threshold: float = 0.50

    threshold_min: float = 0.15
    threshold_max: float = 0.90
    threshold_step: float = 0.05
    operational_dice_weight: float = 0.80
    operational_presence_weight: float = 0.20
    use_final_tta: bool = True

    resume: bool = False
    smoke_test: bool = False         # 2 epochs, for checking plumbing

    @property
    def output_dir(self) -> Path:
        return (PROJECT_ROOT / "outputs" / "task2_training"
                / f"fold_{self.fold}_{self.image_size}px_crop")


CFG = Config()


# =====================================================================
# 1. Helpers
# =====================================================================

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint_file(path: Path, device, weights_only: bool = True):
    try:
        return torch.load(path, map_location=device,
                          weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dict or state_dict.")
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise KeyError("No state_dict found in the checkpoint.")


def clear_cache(device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def normalise_output(output):
    """Task 2 returns (seg, cls); Task 1 returns a bare tensor."""
    if isinstance(output, (tuple, list)):
        return output[0], output[1]
    return output, None


# =====================================================================
# 2. Task 1 predicted masks and lesion boxes
# =====================================================================

def build_task1_preprocess(size: int) -> v2.Compose:
    return v2.Compose([
        ResizeLongestSideAndPad(output_size=size),
        v2.ToDtype(dtype={tv_tensors.Image: torch.float32,
                          tv_tensors.Mask: torch.float32, "others": None},
                   scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        v2.ToPureTensor(),
    ])


def largest_component(mask: np.ndarray) -> np.ndarray | None:
    labels, count = connected_components(mask.astype(bool))
    if count == 0:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def unpad_and_restore(padded: np.ndarray, height: int, width: int,
                      size: int) -> np.ndarray:
    """Undo ResizeLongestSideAndPad to get back to original resolution."""
    scale = min(size / height, size / width)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    top = (size - new_height) // 2
    left = (size - new_width) // 2

    cropped = padded[top:top + new_height, left:left + new_width]
    tensor = torch.from_numpy(cropped.astype(np.float32))[None, None]
    return F.interpolate(tensor, size=(height, width),
                         mode="nearest")[0, 0].numpy() >= 0.5


def mask_filename(image_id: str) -> str:
    return f"{hashlib.sha1(image_id.encode('utf-8')).hexdigest()}.png"


def build_task1_cache(datasets, model, device, checkpoint_path,
                      config: Config):
    """
    Predict and cache a lesion mask and bounding box per image.

    Cached against the checkpoint, inference size and threshold that
    produced it, so changing any of those forces a rebuild rather than
    silently reusing boxes from a different model.
    """
    cache_path = config.output_dir / "task1_roi_cache.json"
    mask_dir = config.output_dir / "task1_predicted_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    expected = {str(row["image_id"]) for dataset in datasets
                for row in dataset.data}

    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        metadata = cached.get("metadata", {})
        boxes, files = cached.get("boxes", {}), cached.get("mask_files", {})
        if (metadata.get("checkpoint") == str(checkpoint_path.resolve())
                and metadata.get("size") == config.task1_size
                and metadata.get("threshold") == config.task1_threshold
                and expected.issubset(boxes) and expected.issubset(files)
                and all((config.output_dir / files[k]).is_file()
                        for k in expected)):
            print(f"Using cached Task 1 crops: {cache_path.name}")
            return ({k: list(map(int, v)) for k, v in boxes.items()},
                    {k: str(v) for k, v in files.items()})

    preprocess = build_task1_preprocess(config.task1_size)
    boxes: dict[str, list[int]] = {}
    files: dict[str, str] = {}
    started = time.time()

    print(f"Predicting Task 1 masks for {len(expected)} images...")

    with torch.inference_mode():
        for dataset in datasets:
            for index in range(len(dataset)):
                sample = dataset[index]
                image_id = str(sample["image_id"])
                if image_id in boxes:
                    continue

                raw = sample["image"]
                height, width = raw.shape[-2:]
                image, _ = preprocess(
                    tv_tensors.Image(raw),
                    tv_tensors.Mask(torch.zeros((1, height, width),
                                                dtype=torch.uint8)))

                logits, _ = normalise_output(
                    model(image.unsqueeze(0).to(device)))
                padded = (torch.sigmoid(logits)[0, 0]
                          >= config.task1_threshold).cpu().numpy()
                prediction = unpad_and_restore(padded, height, width,
                                               config.task1_size)

                component = largest_component(prediction)
                box = bbox_from_mask(component) if component is not None \
                    else None

                if component is None or box is None:
                    # An empty Task 1 prediction must not suppress the
                    # whole image — fall back to using all of it.
                    component = np.ones((height, width), dtype=bool)
                    box = (0, 0, width, height)

                relative = str(Path("task1_predicted_masks")
                               / mask_filename(image_id))
                Image.fromarray(component.astype(np.uint8) * 255).save(
                    config.output_dir / relative)

                boxes[image_id] = list(map(int, box))
                files[image_id] = relative

                if len(boxes) % 250 == 0:
                    print(f"  {len(boxes)}/{len(expected)} "
                          f"({time.time() - started:.0f}s)")

    cache_path.write_text(json.dumps({
        "metadata": {"checkpoint": str(checkpoint_path.resolve()),
                     "size": config.task1_size,
                     "threshold": config.task1_threshold,
                     "count": len(boxes)},
        "boxes": boxes, "mask_files": files}, indent=2), encoding="utf-8")

    print(f"Cached {len(boxes)} Task 1 masks.")
    return boxes, files


# =====================================================================
# 3. Crop dataset
# =====================================================================

def build_transform(size: int, training: bool) -> v2.Compose:
    """Conservative augmentation — small structures are easily destroyed."""
    operations: list[Any] = [ResizeLongestSideAndPad(output_size=size)]

    if training:
        operations += [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomApply([v2.RandomAffine(
                degrees=(-10, 10), translate=(0.03, 0.03),
                scale=(0.97, 1.03),
                interpolation=InterpolationMode.BILINEAR,
                fill={tv_tensors.Image: [124, 116, 104],
                      tv_tensors.Mask: 0})], p=0.5),
            v2.RandomApply([v2.ColorJitter(
                brightness=0.10, contrast=0.10,
                saturation=0.08, hue=0.02)], p=0.4),
        ]

    operations += [
        v2.ToDtype(dtype={tv_tensors.Image: torch.float32,
                          tv_tensors.Mask: torch.float32, "others": None},
                   scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        v2.ToPureTensor(),
    ]
    return v2.Compose(operations)


def expand_box(box, height: int, width: int, margin: float, jitter: float):
    x1, y1, x2, y2 = box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)

    centre_x = 0.5 * (x1 + x2)
    centre_y = 0.5 * (y1 + y2)
    if jitter > 0:
        centre_x += random.uniform(-jitter, jitter) * box_width
        centre_y += random.uniform(-jitter, jitter) * box_height

    half_width = box_width * (1.0 + 2.0 * margin) / 2.0
    half_height = box_height * (1.0 + 2.0 * margin) / 2.0

    nx1 = max(0, int(math.floor(centre_x - half_width)))
    ny1 = max(0, int(math.floor(centre_y - half_height)))
    nx2 = min(width, int(math.ceil(centre_x + half_width)))
    ny2 = min(height, int(math.ceil(centre_y + half_height)))

    return (0, 0, width, height) if nx2 <= nx1 or ny2 <= ny1 \
        else (nx1, ny1, nx2, ny2)


class PredictedCropDataset(Dataset):
    """Crop each raw image to its predicted lesion, then resize."""

    def __init__(self, base: LesionDataset, boxes, mask_files,
                 config: Config, training: bool):
        self.base = base
        self.boxes = boxes
        self.mask_files = mask_files
        self.config = config
        self.training = training
        self.transform = build_transform(config.image_size, training)
        self.safe_transform = build_transform(config.image_size, False)
        self.data = base.data

    def __len__(self) -> int:
        return len(self.base)

    def _load_roi(self, image_id: str, height: int,
                  width: int) -> torch.Tensor:
        relative = self.mask_files.get(image_id)
        path = self.config.output_dir / relative if relative else None

        if path is None or not path.is_file():
            return torch.ones((1, height, width), dtype=torch.uint8)

        array = np.asarray(Image.open(path).convert("L")) > 0
        if array.shape != (height, width):
            tensor = torch.from_numpy(array.astype(np.float32))[None, None]
            array = F.interpolate(tensor, size=(height, width),
                                  mode="nearest")[0, 0].numpy() >= 0.5
        return torch.from_numpy(array.astype(np.uint8))[None]

    @staticmethod
    def _crop(image, masks, box):
        x1, y1, x2, y2 = box
        cropped_image = VF.crop(tv_tensors.Image(image), top=y1, left=x1,
                                height=y2 - y1, width=x2 - x1)
        cropped_masks = VF.crop(tv_tensors.Mask(masks), top=y1, left=x1,
                                height=y2 - y1, width=x2 - x1)

        before = masks[1:].sum(dim=(1, 2)).float()
        after = torch.as_tensor(cropped_masks)[1:].sum(dim=(1, 2)).float()
        retention = torch.where(before > 0, after / before.clamp_min(1.0),
                                torch.full_like(before, float("nan")))
        return cropped_image, cropped_masks, retention

    def _transform_preserving_positives(self, image, masks):
        """Retry augmentation if it would delete a positive mask entirely."""
        present = torch.as_tensor(masks)[1:].sum(dim=(1, 2)) > 0
        attempts = self.config.transform_retries if self.training else 1

        for _ in range(max(attempts, 1)):
            out_image, out_masks = self.transform(tv_tensors.Image(image),
                                                  tv_tensors.Mask(masks))
            still = torch.as_tensor(out_masks)[1:].sum(dim=(1, 2)) > 0
            if not torch.any(present & ~still):
                return out_image, out_masks

        return self.safe_transform(tv_tensors.Image(image),
                                   tv_tensors.Mask(masks))

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        image_id = str(sample["image_id"])
        image = sample["image"]
        attributes = sample["task2_attributes"]
        height, width = image.shape[-2:]

        roi = self._load_roi(image_id, height, width)
        masks = torch.cat([roi, attributes], dim=0)

        tight = tuple(int(v) for v in
                      self.boxes.get(image_id, [0, 0, width, height]))

        if self.training:
            margin = random.uniform(self.config.train_crop_margin_min,
                                    self.config.train_crop_margin_max)
            jitter = self.config.train_crop_jitter
        else:
            margin, jitter = self.config.val_crop_margin, 0.0

        box = expand_box(tight, height, width, margin, jitter)
        cropped_image, cropped_masks, retention = self._crop(image, masks,
                                                             box)

        # Widen, then abandon the crop entirely, rather than lose a target.
        if self.training:
            present = masks[1:].sum(dim=(1, 2)) > 0
            for fallback in (0.50, None):
                still = torch.as_tensor(cropped_masks)[1:].sum(
                    dim=(1, 2)) > 0
                if not torch.any(present & ~still):
                    break
                next_box = (expand_box(tight, height, width, fallback, 0.0)
                            if fallback is not None
                            else (0, 0, width, height))
                cropped_image, cropped_masks, retention = self._crop(
                    image, masks, next_box)

        out_image, out_masks = self._transform_preserving_positives(
            cropped_image, cropped_masks)

        out_masks = (torch.as_tensor(out_masks) > 0.5).float().contiguous()

        return {
            "image_id": image_id,
            "image": torch.as_tensor(out_image).contiguous(),
            "task1_predicted_roi": out_masks[0:1],
            "task2_attributes": out_masks[1:6],
            "crop_retention": retention.float(),
        }


# =====================================================================
# 4. Class statistics
# =====================================================================

def component_areas(mask: np.ndarray) -> list[int]:
    labels, count = connected_components(mask.astype(bool))
    if count == 0:
        return []
    return [int(v) for v in np.bincount(labels.ravel())[1:] if v > 0]


def measure_distribution(dataset, config: Config):
    """
    Class prevalence, sampling weights and component-area floors.

    Cached: a full pass now includes loading a ROI mask and cropping every
    image, so it is far more expensive than it was at 384px full-image.
    """
    cache_path = config.output_dir / "task2_class_statistics.json"

    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (cached.get("attribute_order") == list(ATTRIBUTES)
                and cached.get("image_size") == config.image_size
                and len(cached.get("sample_weight", [])) == len(dataset)):
            print(f"Using cached class statistics ({cache_path.name}).")
            print_distribution(cached["attributes"])
            return (torch.tensor(cached["seg_pos_weight"],
                                 dtype=torch.float32),
                    torch.tensor(cached["cls_pos_weight"],
                                 dtype=torch.float32),
                    torch.tensor(cached["sample_weight"],
                                 dtype=torch.float64),
                    cached["minimum_component_areas"],
                    cached["attributes"])

    loader = DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=False, num_workers=config.num_workers)

    positive_pixels = torch.zeros(NUM_ATTRIBUTES, dtype=torch.float64)
    positive_images = torch.zeros(NUM_ATTRIBUTES, dtype=torch.float64)
    presence_rows: list[torch.Tensor] = []
    areas: list[list[int]] = [[] for _ in ATTRIBUTES]
    total_pixels = total_images = 0

    print("Measuring class statistics over deterministic crops...")

    for batch in loader:
        masks = batch["task2_attributes"]
        presence = masks.sum(dim=(2, 3)) > 0

        positive_pixels += masks.sum(dim=(0, 2, 3)).double()
        positive_images += presence.sum(dim=0).double()
        presence_rows.append(presence.float())
        total_pixels += masks.shape[0] * masks.shape[2] * masks.shape[3]
        total_images += masks.shape[0]

        numpy_masks = masks.numpy() > 0.5
        for sample in range(numpy_masks.shape[0]):
            for channel in range(NUM_ATTRIBUTES):
                areas[channel].extend(
                    component_areas(numpy_masks[sample, channel]))

    presence_matrix = torch.cat(presence_rows, dim=0)
    pixel_rate = (positive_pixels / max(total_pixels, 1)).clamp_min(1e-8)
    image_rate = (positive_images / max(total_images, 1)).clamp_min(1e-8)

    raw_seg = (1.0 - pixel_rate) / pixel_rate
    raw_cls = (1.0 - image_rate) / image_rate
    seg_weight = torch.sqrt(raw_seg).clamp(
        max=config.max_seg_pos_weight).float()
    cls_weight = torch.sqrt(raw_cls).clamp(
        max=config.max_cls_pos_weight).float()

    rarity = torch.sqrt(1.0 / image_rate.float()).clamp(
        max=config.max_rarity_boost)
    sample_weight = (1.0 + (presence_matrix * (rarity - 1.0)).sum(dim=1)
                     ).clamp(max=config.max_sample_weight).double()

    minimum_areas: list[int] = []
    rows: list[dict[str, float]] = []

    for channel, name in enumerate(ATTRIBUTES):
        channel_areas = areas[channel]
        percentile = (float(np.percentile(channel_areas,
                                          config.component_percentile))
                      if channel_areas else 0.0)
        derived = int(round(percentile * config.component_percentile_scale))
        minimum = max(int(config.component_area_floors[channel]), derived)
        minimum_areas.append(minimum)

        rows.append({
            "attribute": name,
            "pixel_rate": float(pixel_rate[channel]),
            "image_rate": float(image_rate[channel]),
            "raw_seg_weight": float(raw_seg[channel]),
            "seg_pos_weight": float(seg_weight[channel]),
            "cls_pos_weight": float(cls_weight[channel]),
            "component_count": len(channel_areas),
            "component_area_percentile": percentile,
            "minimum_component_area": minimum,
        })

    print_distribution(rows)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "attribute_order": list(ATTRIBUTES),
        "image_size": config.image_size,
        "attributes": rows,
        "seg_pos_weight": seg_weight.tolist(),
        "cls_pos_weight": cls_weight.tolist(),
        "sample_weight": sample_weight.tolist(),
        "minimum_component_areas": minimum_areas,
    }, indent=2), encoding="utf-8")

    return seg_weight, cls_weight, sample_weight, minimum_areas, rows


def print_distribution(rows) -> None:
    print(f"{'attribute':<20}{'pixel%':>10}{'image%':>10}"
          f"{'raw w':>10}{'seg w':>9}{'cls w':>9}{'min cc':>8}")
    for row in rows:
        print(f"{row['attribute']:<20}{100 * row['pixel_rate']:>9.4f}%"
              f"{100 * row['image_rate']:>9.2f}%"
              f"{row['raw_seg_weight']:>10.0f}"
              f"{row['seg_pos_weight']:>9.2f}{row['cls_pos_weight']:>9.2f}"
              f"{row['minimum_component_area']:>8d}")


# =====================================================================
# 5. Loss
# =====================================================================

def weighted_bce(logits, targets, positive_weight):
    loss = F.binary_cross_entropy_with_logits(logits, targets,
                                              reduction="none")
    return (loss * (1.0 + (positive_weight - 1.0) * targets)).mean()


def focal_bce(logits, targets, positive_weight, gamma: float):
    base = F.binary_cross_entropy_with_logits(logits, targets,
                                              reduction="none")
    probabilities = torch.sigmoid(logits)
    probability_true = (probabilities * targets
                        + (1.0 - probabilities) * (1.0 - targets))
    weights = 1.0 + (positive_weight - 1.0) * targets
    return (base * (1.0 - probability_true).pow(gamma) * weights).mean()


def positive_dice_loss(probabilities, targets, smooth: float = 1.0):
    """Soft Dice on images where the attribute is actually present."""
    present = targets.sum(dim=(1, 2)) > 0
    if not torch.any(present):
        return probabilities.sum() * 0.0

    probabilities, targets = probabilities[present], targets[present]
    intersection = (probabilities * targets).sum(dim=(1, 2))
    denominator = probabilities.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    return 1.0 - ((2.0 * intersection + smooth)
                  / (denominator + smooth)).mean()


def soft_erode(image):
    vertical = -F.max_pool2d(-image, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-image, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def soft_skeletonize(image, iterations: int):
    """Differentiable skeleton approximation for clDice."""
    opened = F.max_pool2d(soft_erode(image), 3, stride=1, padding=1)
    skeleton = F.relu(image - opened)

    for _ in range(iterations):
        image = soft_erode(image)
        opened = F.max_pool2d(soft_erode(image), 3, stride=1, padding=1)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)

    return skeleton


def positive_cldice_loss(probabilities, targets, iterations: int,
                         smooth: float = 1.0):
    """
    Centreline Dice, for curvilinear structures.

    Standard Dice barely penalises a broken line, because the missing
    pixels are few. clDice compares skeletons, so connectivity matters —
    which is what a pigment network or a streak actually is.
    """
    present = targets.sum(dim=(1, 2)) > 0
    if not torch.any(present):
        return probabilities.sum() * 0.0

    probabilities = probabilities[present, None]
    targets = targets[present, None]
    skeleton_prediction = soft_skeletonize(probabilities, iterations)
    skeleton_target = soft_skeletonize(targets, iterations)

    precision = ((skeleton_prediction * targets).sum(dim=(1, 2, 3)) + smooth
                 ) / (skeleton_prediction.sum(dim=(1, 2, 3)) + smooth)
    sensitivity = ((skeleton_target * probabilities).sum(dim=(1, 2, 3))
                   + smooth) / (skeleton_target.sum(dim=(1, 2, 3)) + smooth)

    return 1.0 - ((2.0 * precision * sensitivity + smooth)
                  / (precision + sensitivity + smooth)).mean()


class AttributeLoss(nn.Module):
    """
    A different loss mixture per attribute, matched to its shape.

    Networks and streaks are curvilinear, so they get clDice. Milia and
    globules are small blobs against a busy background, so they get focal
    BCE, which concentrates on the hard pixels instead of the easy ones.
    """

    SPECIFICATIONS = (
        {"bce": 0.35, "dice": 0.65},                    # pigment network
        {"bce": 0.30, "dice": 0.50, "cldice": 0.20},    # negative network
        {"bce": 0.25, "dice": 0.45, "cldice": 0.30},    # streaks
        {"focal": 0.50, "dice": 0.50},                  # milia-like cysts
        {"focal": 0.40, "dice": 0.60},                  # globules
    )

    def __init__(self, positive_weights: torch.Tensor, focal_gamma: float,
                 cldice_iterations: int):
        super().__init__()
        self.register_buffer("positive_weights", positive_weights.float())
        self.focal_gamma = focal_gamma
        self.cldice_iterations = cldice_iterations

        for specification in self.SPECIFICATIONS:
            if not math.isclose(sum(specification.values()), 1.0,
                                abs_tol=1e-6):
                raise ValueError("Each loss specification must sum to 1.")

    def forward(self, logits, targets):
        if logits.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {logits.shape} vs "
                             f"{targets.shape}")

        losses = []
        for channel, specification in enumerate(self.SPECIFICATIONS):
            channel_logits = logits[:, channel]
            channel_targets = targets[:, channel]
            probabilities = torch.sigmoid(channel_logits)
            weight = self.positive_weights[channel]
            total = logits.sum() * 0.0

            if "bce" in specification:
                total = total + specification["bce"] * weighted_bce(
                    channel_logits, channel_targets, weight)
            if "focal" in specification:
                total = total + specification["focal"] * focal_bce(
                    channel_logits, channel_targets, weight,
                    self.focal_gamma)
            if "dice" in specification:
                total = total + specification["dice"] * positive_dice_loss(
                    probabilities, channel_targets)
            if "cldice" in specification:
                total = total + specification["cldice"] * \
                    positive_cldice_loss(probabilities, channel_targets,
                                         self.cldice_iterations)

            losses.append(total)

        return torch.stack(losses).mean()


# =====================================================================
# 6. Prediction and metrics
# =====================================================================

def predict(model, images, use_tta: bool):
    """Segmentation and presence probabilities, optionally flip-averaged."""
    flips = [((), ())]
    if use_tta:
        flips += [((-1,), (-1,)), ((-2,), (-2,))]

    segmentations, classifications = [], []
    for input_dims, output_dims in flips:
        augmented = torch.flip(images, dims=input_dims) if input_dims \
            else images
        seg_logits, cls_logits = normalise_output(model(augmented))
        probabilities = torch.sigmoid(seg_logits)
        if output_dims:
            probabilities = torch.flip(probabilities, dims=output_dims)
        segmentations.append(probabilities)
        if cls_logits is not None:
            classifications.append(torch.sigmoid(cls_logits))

    return (torch.stack(segmentations).mean(dim=0),
            torch.stack(classifications).mean(dim=0)
            if classifications else None)


def fallback_presence(probabilities):
    """Mean of the top 5% of pixels, if there is no classification head."""
    flattened = probabilities.flatten(start_dim=2)
    return flattened.topk(max(1, flattened.shape[2] // 20),
                          dim=2).values.mean(dim=2)


def dilate(roi, pixels: int):
    if pixels <= 0:
        return roi > 0.5
    return F.max_pool2d(roi.float(), 2 * pixels + 1, stride=1,
                        padding=pixels) > 0.5


def apply_operational(segmentation, classification, roi, config: Config):
    """
    Soft gating and soft ROI suppression.

    Both are multiplicative rather than binary. Hard gating zeroes a mask
    whenever the classifier is wrong, and on a positive image that scores
    Dice 0 — which lands straight on the headline, since Dice(pos)
    averages over positive images only.
    """
    adjusted = segmentation

    if classification is not None:
        strengths = torch.tensor(config.gate_strengths,
                                 dtype=adjusted.dtype,
                                 device=adjusted.device
                                 ).view(1, NUM_ATTRIBUTES, 1, 1)
        adjusted = adjusted * ((1.0 - strengths)
                               + strengths * classification[:, :, None, None])

    inside = dilate(roi, config.roi_dilation)
    return torch.where(inside, adjusted,
                       adjusted * config.outside_roi_scale)


def overlap_scores(predictions, targets):
    predictions, targets = predictions.bool(), targets.bool()
    intersection = (predictions & targets).sum(dim=(2, 3)).float()
    prediction_sum = predictions.sum(dim=(2, 3)).float()
    target_sum = targets.sum(dim=(2, 3)).float()
    union = prediction_sum + target_sum - intersection

    dice = torch.where(prediction_sum + target_sum > 0,
                       2.0 * intersection
                       / (prediction_sum + target_sum).clamp_min(1.0),
                       torch.ones_like(intersection))
    iou = torch.where(union > 0, intersection / union.clamp_min(1.0),
                      torch.ones_like(intersection))
    return dice, iou, prediction_sum, target_sum


def threshold_tensor(values, device):
    if np.isscalar(values):
        values = [float(values)] * NUM_ATTRIBUTES
    return torch.tensor(values, dtype=torch.float32,
                        device=device).view(1, NUM_ATTRIBUTES, 1, 1)


def filter_components(predictions, minimum_areas):
    """Drop predicted blobs smaller than the smallest real ones."""
    device = predictions.device
    array = predictions.detach().cpu().numpy().astype(bool)
    filtered = np.zeros_like(array, dtype=bool)

    for sample in range(array.shape[0]):
        for channel in range(array.shape[1]):
            labels, count = connected_components(array[sample, channel])
            if count == 0:
                continue
            sizes = np.bincount(labels.ravel())
            keep = np.where(sizes >= int(minimum_areas[channel]))[0]
            keep = keep[keep != 0]
            if len(keep):
                filtered[sample, channel] = np.isin(labels, keep)

    return torch.from_numpy(filtered).to(device=device)


def presence_scores(truth: np.ndarray, prediction: np.ndarray):
    truth, prediction = truth.astype(bool), prediction.astype(bool)
    true_positive = int(np.sum(truth & prediction))
    false_positive = int(np.sum(~truth & prediction))
    false_negative = int(np.sum(truth & ~prediction))

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return float(precision), float(recall), float(f1)


def evaluate(model, loader, criterion, cls_criterion, device,
             pixel_thresholds, mode: str, use_tta: bool,
             minimum_areas=None, config: Config = CFG):
    """mode: 'raw' for the bare segmentation, 'operational' for gated."""
    if mode not in {"raw", "operational"}:
        raise ValueError("mode must be 'raw' or 'operational'.")

    model.eval()
    thresholds = threshold_tensor(pixel_thresholds, device)

    dice_positive = [[] for _ in ATTRIBUTES]
    iou_positive = [[] for _ in ATTRIBUTES]
    dice_all = [[] for _ in ATTRIBUTES]
    fired = np.zeros(NUM_ATTRIBUTES)
    truths, scores, mask_presence, retentions = [], [], [], []
    total_images = 0
    total_loss = 0.0

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["task2_attributes"].to(device)
            roi = batch["task1_predicted_roi"].to(device)

            seg_logits, cls_logits = normalise_output(model(images))
            target_presence = (targets.sum(dim=(2, 3)) > 0).float()

            loss = criterion(seg_logits, targets)
            if cls_logits is not None:
                loss = loss + config.classification_loss_weight * \
                    cls_criterion(cls_logits, target_presence)
            total_loss += float(loss.item())

            if use_tta:
                segmentation, classification = predict(model, images, True)
            else:
                segmentation = torch.sigmoid(seg_logits)
                classification = (torch.sigmoid(cls_logits)
                                  if cls_logits is not None else None)

            probabilities = (apply_operational(segmentation, classification,
                                               roi, config)
                             if mode == "operational" else segmentation)

            predictions = probabilities >= thresholds
            if minimum_areas is not None:
                predictions = filter_components(predictions, minimum_areas)

            dice, iou, prediction_sum, target_sum = overlap_scores(
                predictions, targets > 0.5)

            dice_np = dice.cpu().numpy()
            iou_np = iou.cpu().numpy()
            positive = (target_sum > 0).cpu().numpy()
            fired += (prediction_sum > 0).cpu().numpy().sum(axis=0)
            total_images += images.shape[0]

            for channel in range(NUM_ATTRIBUTES):
                dice_all[channel].extend(dice_np[:, channel].tolist())
                dice_positive[channel].extend(
                    dice_np[positive[:, channel], channel].tolist())
                iou_positive[channel].extend(
                    iou_np[positive[:, channel], channel].tolist())

            truths.append(target_presence.cpu().numpy())
            scores.append((classification if classification is not None
                           else fallback_presence(segmentation)
                           ).cpu().numpy())
            mask_presence.append((prediction_sum > 0).cpu().numpy())
            retentions.append(batch["crop_retention"].numpy())

    truth = np.concatenate(truths)
    score = np.concatenate(scores)
    mask_truth = np.concatenate(mask_presence)
    retention = np.concatenate(retentions)

    average_precision = [float("nan")] * NUM_ATTRIBUTES
    try:
        from sklearn.metrics import average_precision_score
        for channel in range(NUM_ATTRIBUTES):
            if np.unique(truth[:, channel]).size > 1:
                average_precision[channel] = float(average_precision_score(
                    truth[:, channel], score[:, channel]))
    except ImportError:
        pass

    values = ([float(pixel_thresholds)] * NUM_ATTRIBUTES
              if np.isscalar(pixel_thresholds)
              else [float(v) for v in pixel_thresholds])

    rows = []
    for channel, name in enumerate(ATTRIBUTES):
        count = len(dice_positive[channel])
        precision, recall, f1 = presence_scores(truth[:, channel],
                                                mask_truth[:, channel])
        rows.append({
            "attribute": name, "mode": mode,
            "pixel_threshold": values[channel],
            "dice_pos": float(np.mean(dice_positive[channel]))
            if count else float("nan"),
            "iou_pos": float(np.mean(iou_positive[channel]))
            if count else float("nan"),
            "dice_all": float(np.mean(dice_all[channel])),
            "n_positive": count,
            "true_rate": count / max(total_images, 1),
            "fire_rate": float(fired[channel]) / max(total_images, 1),
            "mask_presence_precision": precision,
            "mask_presence_recall": recall,
            "mask_presence_f1": f1,
            "classification_ap": average_precision[channel],
            "mean_crop_retention": float(np.nanmean(retention[:, channel])),
        })

    return total_loss / max(len(loader), 1), rows


def print_metrics(rows) -> None:
    print(f"  {'attribute':<20}{'thr':>6}{'Dice+':>9}{'IoU+':>9}"
          f"{'true%':>8}{'fire%':>8}{'maskF1':>9}{'AP':>8}{'crop%':>8}")
    for row in rows:
        print(f"  {row['attribute']:<20}{row['pixel_threshold']:>6.2f}"
              f"{row['dice_pos']:>9.4f}{row['iou_pos']:>9.4f}"
              f"{100 * row['true_rate']:>7.1f}%"
              f"{100 * row['fire_rate']:>7.1f}%"
              f"{row['mask_presence_f1']:>9.4f}"
              f"{row['classification_ap']:>8.4f}"
              f"{100 * row['mean_crop_retention']:>7.1f}%")


# =====================================================================
# 7. Threshold tuning
# =====================================================================

def candidates(config: Config) -> np.ndarray:
    count = int(round((config.threshold_max - config.threshold_min)
                      / config.threshold_step))
    return np.round(np.linspace(config.threshold_min, config.threshold_max,
                                count + 1), 4)


def tune_pixel_thresholds(model, loader, device, config: Config,
                          operational: bool):
    """
    Pick a pixel threshold per attribute.

    Raw thresholds maximise Dice on positive images. Operational
    thresholds also weigh mask-presence F1, since a gated mask decides
    whether the attribute is claimed at all.
    """
    grid = candidates(config)
    dice_sums = np.zeros((len(grid), NUM_ATTRIBUTES))
    positive_counts = np.zeros(NUM_ATTRIBUTES, dtype=np.int64)
    true_positive = np.zeros((len(grid), NUM_ATTRIBUTES), dtype=np.int64)
    false_positive = np.zeros((len(grid), NUM_ATTRIBUTES), dtype=np.int64)
    false_negative = np.zeros((len(grid), NUM_ATTRIBUTES), dtype=np.int64)

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["task2_attributes"].to(device) > 0.5
            segmentation, classification = predict(model, images,
                                                   config.use_final_tta)

            probabilities = (apply_operational(segmentation, classification,
                                               batch["task1_predicted_roi"]
                                               .to(device), config)
                             if operational else segmentation)

            target_sum = targets.sum(dim=(2, 3))
            present = target_sum > 0

            for index, threshold in enumerate(grid):
                predictions = probabilities >= float(threshold)
                prediction_sum = predictions.sum(dim=(2, 3))
                predicted_presence = prediction_sum > 0

                if operational:
                    true_positive[index] += (predicted_presence & present
                                             ).sum(dim=0).cpu().numpy()
                    false_positive[index] += (predicted_presence & ~present
                                              ).sum(dim=0).cpu().numpy()
                    false_negative[index] += (~predicted_presence & present
                                              ).sum(dim=0).cpu().numpy()

                intersection = (predictions & targets).sum(dim=(2, 3)).float()
                for channel in range(NUM_ATTRIBUTES):
                    rows = present[:, channel]
                    count = int(rows.sum().item())
                    if count == 0:
                        continue
                    if index == 0:
                        positive_counts[channel] += count
                    dice = 2.0 * intersection[rows, channel] / (
                        prediction_sum[rows, channel].float()
                        + target_sum[rows, channel].float() + 1e-8)
                    dice_sums[index, channel] += float(dice.sum().item())

    mean_dice = dice_sums / np.maximum(positive_counts[None, :], 1)

    if operational:
        precision = true_positive / np.maximum(
            true_positive + false_positive, 1)
        recall = true_positive / np.maximum(
            true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
        objective = (config.operational_dice_weight * mean_dice
                     + config.operational_presence_weight * f1)
    else:
        f1 = np.zeros_like(mean_dice)
        objective = mean_dice

    best = np.argmax(objective, axis=0)
    thresholds = [float(grid[best[channel]])
                  for channel in range(NUM_ATTRIBUTES)]

    label = "operational" if operational else "raw"
    print(f"\n{label.capitalize()} pixel thresholds:")
    report = {}
    for channel, name in enumerate(ATTRIBUTES):
        index = best[channel]
        report[name] = {"threshold": thresholds[channel],
                        "dice_pos": float(mean_dice[index, channel]),
                        "mask_f1": float(f1[index, channel])}
        print(f"  {name:<20} threshold={thresholds[channel]:.2f}, "
              f"Dice(pos)={mean_dice[index, channel]:.4f}"
              + (f", mask F1={f1[index, channel]:.4f}"
                 if operational else ""))

    return thresholds, report


def tune_presence_thresholds(model, loader, device, config: Config):
    """Image-level thresholds for Task 3, chosen on F1."""
    grid = candidates(config)
    truths, scores = [], []

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["task2_attributes"].to(device)
            segmentation, classification = predict(model, images,
                                                   config.use_final_tta)
            truths.append((targets.sum(dim=(2, 3)) > 0).cpu().numpy())
            scores.append((classification if classification is not None
                           else fallback_presence(segmentation)
                           ).cpu().numpy())

    truth = np.concatenate(truths).astype(bool)
    score = np.concatenate(scores)

    thresholds, report = [], {}
    print("\nClassification thresholds for Task 3:")

    for channel, name in enumerate(ATTRIBUTES):
        best = (0.5, -1.0, 0.0, 0.0)
        for threshold in grid:
            precision, recall, f1 = presence_scores(
                truth[:, channel], score[:, channel] >= threshold)
            if f1 > best[1]:
                best = (float(threshold), f1, precision, recall)

        thresholds.append(best[0])
        report[name] = {"threshold": best[0], "f1": best[1],
                        "precision": best[2], "recall": best[3]}
        print(f"  {name:<20} threshold={best[0]:.2f}, F1={best[1]:.4f}, "
              f"P={best[2]:.4f}, R={best[3]:.4f}")

    return thresholds, report


# =====================================================================
# 8. Figures
# =====================================================================

def save_history(history, output_dir: Path) -> None:
    frame = pd.DataFrame(history)
    frame.to_csv(output_dir / "task2_training_history.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=8)

    axes[1].plot(frame["epoch"], frame["macro_dice_pos"], label="Dice(pos)")
    axes[1].plot(frame["epoch"], frame["macro_presence_ap"], label="AP")
    axes[1].plot(frame["epoch"], frame["balanced_score"],
                 color="black", linewidth=2, label="balanced")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("epoch")
    axes[1].legend(fontsize=8)

    for name in ATTRIBUTES:
        column = f"dice_pos_{name}"
        if column in frame:
            axes[2].plot(frame["epoch"], frame[column], label=name)
    axes[2].set_title("Per-attribute Dice(pos)")
    axes[2].set_xlabel("epoch")
    axes[2].legend(fontsize=7)

    figure.tight_layout()
    figure.savefig(output_dir / "task2_training_curves.png", dpi=140)
    plt.close(figure)


def save_visualisation(model, loader, device, raw_thresholds,
                       operational_thresholds, minimum_areas,
                       path: Path, config: Config) -> None:
    batch = next(iter(loader))
    images = batch["image"].to(device)
    targets = batch["task2_attributes"]
    roi = batch["task1_predicted_roi"].to(device)

    with torch.inference_mode():
        segmentation, classification = predict(model, images,
                                               config.use_final_tta)
        raw = segmentation >= threshold_tensor(raw_thresholds, device)
        operational = filter_components(
            apply_operational(segmentation, classification, roi, config)
            >= threshold_tensor(operational_thresholds, device),
            minimum_areas)

    count = min(2, images.shape[0])
    figure, axes = plt.subplots(count * 2, NUM_ATTRIBUTES + 1,
                                figsize=(18, 7 * count), squeeze=False)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    for sample in range(count):
        image = (images[sample].cpu() * std + mean).clamp(0, 1)
        for offset, (label, predictions) in enumerate(
                (("raw", raw), ("operational", operational))):
            row = sample * 2 + offset
            axes[row, 0].imshow(image.permute(1, 2, 0).numpy())
            axes[row, 0].set_title(
                f"{batch['image_id'][sample]}\n{label}", fontsize=9)
            axes[row, 0].axis("off")

            for channel, name in enumerate(ATTRIBUTES):
                overlay = np.zeros((*targets.shape[-2:], 3),
                                   dtype=np.float32)
                overlay[..., 0] = predictions[sample, channel].cpu().numpy()
                overlay[..., 1] = targets[sample, channel].numpy()
                axes[row, channel + 1].imshow(overlay)
                axes[row, channel + 1].set_title(
                    f"{name}\nred pred, green GT", fontsize=8)
                axes[row, channel + 1].axis("off")

    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


# =====================================================================
# 9. Training
# =====================================================================

def train() -> None:
    set_seed(CFG.seed)
    device = select_device()
    CFG.output_dir.mkdir(parents=True, exist_ok=True)
    epochs = 2 if CFG.smoke_test else CFG.max_epochs

    print(f"Training on {device}")
    print(f"Predicted lesion crop -> {CFG.image_size}px, batch "
          f"{CFG.batch_size} x {CFG.accumulation_steps} accumulation")
    print(f"Output: {CFG.output_dir}")

    task1_checkpoint = locate_task1_checkpoint(fold=CFG.fold,
                                               image_size=CFG.task1_size)
    if task1_checkpoint is None:
        raise FileNotFoundError("No Task 1 checkpoint — run step 15 first.")
    print(f"Task 1 checkpoint: {task1_checkpoint}")

    raw_train = LesionDataset(fold=CFG.fold, role="train", transform=None,
                              include_task2=True)
    raw_val = LesionDataset(fold=CFG.fold, role="val", transform=None,
                            include_task2=True)

    task1_model = build_task1_model().to(device)
    task1_model.load_state_dict(extract_state_dict(
        load_checkpoint_file(task1_checkpoint, device)), strict=True)
    task1_model.eval()

    boxes, mask_files = build_task1_cache(
        [LesionDataset(fold=CFG.fold, role=role, transform=None,
                       include_task2=False) for role in ("train", "val")],
        task1_model, device, task1_checkpoint, CFG)

    del task1_model
    clear_cache(device)

    train_dataset = PredictedCropDataset(raw_train, boxes, mask_files,
                                         CFG, training=True)
    val_dataset = PredictedCropDataset(raw_val, boxes, mask_files,
                                       CFG, training=False)
    stats_dataset = PredictedCropDataset(raw_train, boxes, mask_files,
                                         CFG, training=False)

    (seg_weight, cls_weight, sample_weight, minimum_areas,
     _) = measure_distribution(stats_dataset, CFG)

    generator = torch.Generator().manual_seed(CFG.seed)
    sampler = WeightedRandomSampler(sample_weight,
                                    num_samples=len(sample_weight),
                                    replacement=True, generator=generator)

    loader_kwargs = {"batch_size": CFG.batch_size,
                     "num_workers": CFG.num_workers}
    train_loader = DataLoader(train_dataset, sampler=sampler,
                              **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    print(f"{len(train_dataset)} training, {len(val_dataset)} validation")

    model = build_task2_model(Task2ModelConfig(
        auto_locate_task1=True, fold=CFG.fold,
        image_size=CFG.task1_size)).to(device)

    criterion = AttributeLoss(seg_weight.to(device), CFG.focal_gamma,
                              CFG.cldice_iterations)
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=cls_weight.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=CFG.learning_rate,
                            weight_decay=CFG.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=CFG.scheduler_factor,
        patience=CFG.scheduler_patience, min_lr=CFG.min_learning_rate)

    checkpoint_path = CFG.output_dir / "task2_checkpoint_last.pth"
    segmentation_path = CFG.output_dir / "task2_best_segmentation.pth"
    balanced_path = CFG.output_dir / "task2_best_balanced.pth"

    start_epoch = 0
    best_dice = best_balanced = -1.0
    best_dice_epoch = best_balanced_epoch = 0
    no_improvement = 0
    history: list[dict[str, float]] = []

    if CFG.resume and checkpoint_path.is_file():
        state = load_checkpoint_file(checkpoint_path, device,
                                     weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"])
        best_dice = float(state["best_dice"])
        best_balanced = float(state["best_balanced"])
        best_dice_epoch = int(state.get("best_dice_epoch", 0))
        best_balanced_epoch = int(state.get("best_balanced_epoch", 0))
        no_improvement = int(state.get("no_improvement", 0))
        history = list(state.get("history", []))
        print(f"Resuming at epoch {start_epoch + 1}")

    run_started = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_started = time.time()
        loss_sum = sample_count = 0

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            targets = batch["task2_attributes"].to(device)
            presence = (targets.sum(dim=(2, 3)) > 0).float()

            seg_logits, cls_logits = normalise_output(model(images))
            loss = criterion(seg_logits, targets)
            if cls_logits is not None:
                loss = loss + CFG.classification_loss_weight * \
                    cls_criterion(cls_logits, presence)

            (loss / CFG.accumulation_steps).backward()

            if ((step + 1) % CFG.accumulation_steps == 0
                    or step + 1 == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               CFG.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            loss_sum += float(loss.detach().item()) * images.shape[0]
            sample_count += images.shape[0]

            if step % 50 == 0:
                print(f"  epoch {epoch + 1} step {step}/"
                      f"{len(train_loader)} loss {loss.item():.4f} "
                      f"({time.time() - epoch_started:.0f}s)")

        train_loss = loss_sum / max(sample_count, 1)
        val_loss, rows = evaluate(model, val_loader, criterion,
                                  cls_criterion, device,
                                  CFG.training_threshold, "raw",
                                  use_tta=False)

        macro_dice = float(np.nanmean([r["dice_pos"] for r in rows]))
        macro_iou = float(np.nanmean([r["iou_pos"] for r in rows]))
        macro_ap = float(np.nanmean([r["classification_ap"] for r in rows]))
        balanced = 0.70 * macro_dice + 0.30 * macro_ap
        scheduler.step(macro_dice)

        elapsed = time.time() - epoch_started
        print(f"\nEpoch {epoch + 1}/{epochs} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | Dice(pos) {macro_dice:.4f} | "
              f"IoU(pos) {macro_iou:.4f} | AP {macro_ap:.4f} | "
              f"balanced {balanced:.4f} | {elapsed:.0f}s | "
              f"~{(epochs - epoch - 1) * elapsed / 3600:.1f}h left")
        print_metrics(rows)

        record = {"epoch": epoch + 1, "train_loss": train_loss,
                  "val_loss": val_loss, "macro_dice_pos": macro_dice,
                  "macro_iou_pos": macro_iou, "macro_presence_ap": macro_ap,
                  "balanced_score": balanced,
                  "learning_rate": optimizer.param_groups[0]["lr"]}
        for row in rows:
            record[f"dice_pos_{row['attribute']}"] = row["dice_pos"]
        history.append(record)
        save_history(history, CFG.output_dir)

        if macro_dice > best_dice + CFG.early_stopping_min_delta:
            best_dice, best_dice_epoch = macro_dice, epoch + 1
            no_improvement = 0
            torch.save(model.state_dict(), segmentation_path)
            print(f"  New best segmentation model (Dice {best_dice:.4f})")
        else:
            no_improvement += 1

        if balanced > best_balanced + CFG.early_stopping_min_delta:
            best_balanced, best_balanced_epoch = balanced, epoch + 1
            torch.save(model.state_dict(), balanced_path)
            print(f"  New best balanced model (score {best_balanced:.4f})")

        torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_dice": best_dice, "best_balanced": best_balanced,
                    "best_dice_epoch": best_dice_epoch,
                    "best_balanced_epoch": best_balanced_epoch,
                    "no_improvement": no_improvement, "history": history,
                    "config": asdict(CFG)}, checkpoint_path)

        print(f"  Early-stop counter: {no_improvement}/"
              f"{CFG.early_stopping_patience}")
        if no_improvement >= CFG.early_stopping_patience:
            print("\nEarly stopping triggered.")
            break

    finalise(model, val_loader, criterion, cls_criterion, device,
             segmentation_path, balanced_path, minimum_areas,
             best_dice_epoch, best_balanced_epoch, run_started)


def finalise(model, val_loader, criterion, cls_criterion, device,
             segmentation_path: Path, balanced_path: Path, minimum_areas,
             best_dice_epoch: int, best_balanced_epoch: int,
             run_started: float) -> None:
    """Tune thresholds and report raw and operational results."""
    for path in (segmentation_path, balanced_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")

    model.load_state_dict(load_checkpoint_file(segmentation_path, device))
    model.eval()

    raw_thresholds, raw_report = tune_pixel_thresholds(
        model, val_loader, device, CFG, operational=False)
    operational_thresholds, operational_report = tune_pixel_thresholds(
        model, val_loader, device, CFG, operational=True)

    raw_loss, raw_rows = evaluate(model, val_loader, criterion,
                                  cls_criterion, device, raw_thresholds,
                                  "raw", CFG.use_final_tta)
    operational_loss, operational_rows = evaluate(
        model, val_loader, criterion, cls_criterion, device,
        operational_thresholds, "operational", CFG.use_final_tta,
        minimum_areas)

    print("\nFinal validation — raw segmentation:")
    print_metrics(raw_rows)
    print("\nFinal validation — soft-gated operational masks:")
    print_metrics(operational_rows)

    model.load_state_dict(load_checkpoint_file(balanced_path, device))
    model.eval()
    presence_thresholds, presence_report = tune_presence_thresholds(
        model, val_loader, device, CFG)

    (CFG.output_dir / "task2_best_thresholds.json").write_text(json.dumps({
        "attribute_order": list(ATTRIBUTES),
        "metrics_computed_on": "predicted lesion crop, NOT the full image",
        "comparable_to_full_image_run": False,
        "tuned_on": "validation fold — same fold as reported",
        "pixel_thresholds": dict(zip(ATTRIBUTES, raw_thresholds)),
        "operational_pixel_thresholds": dict(zip(ATTRIBUTES,
                                                 operational_thresholds)),
        "classification_thresholds": dict(zip(ATTRIBUTES,
                                              presence_thresholds)),
        "soft_gate_strengths": dict(zip(ATTRIBUTES, CFG.gate_strengths)),
        "minimum_component_areas": dict(zip(ATTRIBUTES, minimum_areas)),
        "hard_gating": False,
        "final_tta": CFG.use_final_tta,
        "details": {"raw": raw_report, "operational": operational_report,
                    "classification": presence_report},
    }, indent=2), encoding="utf-8")

    pd.DataFrame(raw_rows).to_csv(
        CFG.output_dir / "task2_raw_metrics.csv", index=False)
    pd.DataFrame(operational_rows).to_csv(
        CFG.output_dir / "task2_operational_metrics.csv", index=False)

    def macro(rows, key):
        return float(np.nanmean([row[key] for row in rows]))

    (CFG.output_dir / "task2_final_metrics.json").write_text(json.dumps({
        "metrics_computed_on": "predicted lesion crop, NOT the full image",
        "comparable_to_full_image_run": False,
        "best_segmentation_epoch": best_dice_epoch,
        "best_balanced_epoch": best_balanced_epoch,
        "raw_validation_loss": raw_loss,
        "operational_validation_loss": operational_loss,
        "macro_raw_dice_pos": macro(raw_rows, "dice_pos"),
        "macro_raw_iou_pos": macro(raw_rows, "iou_pos"),
        "macro_operational_dice_pos": macro(operational_rows, "dice_pos"),
        "macro_operational_mask_f1": macro(operational_rows,
                                           "mask_presence_f1"),
        "raw_per_attribute": raw_rows,
        "operational_per_attribute": operational_rows,
        "config": asdict(CFG),
    }, indent=2), encoding="utf-8")

    model.load_state_dict(load_checkpoint_file(segmentation_path, device))
    save_visualisation(model, val_loader, device, raw_thresholds,
                       operational_thresholds, minimum_areas,
                       CFG.output_dir / "task2_final_predictions.png", CFG)

    print(f"\nDone in {(time.time() - run_started) / 3600:.1f}h.")
    print(f"  raw Dice(pos)          {macro(raw_rows, 'dice_pos'):.4f}")
    print(f"  operational Dice(pos)  "
          f"{macro(operational_rows, 'dice_pos'):.4f}")
    print(f"  operational mask F1    "
          f"{macro(operational_rows, 'mask_presence_f1'):.4f}")
    print("\nNote: these are crop-based metrics and are NOT comparable "
          "with the 384px full-image run.")
    print(f"Output: {CFG.output_dir}")


if __name__ == "__main__":
    train()
