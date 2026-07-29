"""
Step 22 — bonus: CLIP retrieval-augmented reporting.

Retrieves visually similar training cases for each validation image and
uses their ground-truth attribute labels as a weak prior, blended into
the Task 2 prediction. The briefing's constraint is that retrieval may
stabilise an estimate but must never supply a fact: no claim in a report
comes from a neighbour, only a nudge to a probability the model already
produced. The confidence gate enforces that — fusion only applies where
Task 2 was undecided in the first place.

Leakage safety is the thing that matters most here:

  * The reference index is built from the training folds only. Fold 0 is
    query-only and never indexed.
  * Any accidental self-match is excluded at retrieval time.
  * Query IDs are checked against reference IDs as they are processed,
    and the audit records how many neighbours fell outside the reference
    split. Both should be zero.

Three probability sets are evaluated: Task 2 alone, retrieval alone, and
fused. Retrieval alone is what shows whether fusion beats either part.

    python step22_bonus.py
"""

from __future__ import annotations
from step19_task3_rubric import (
    attribute_status,
    border_category,
    build_report_text,
    lesion_geometry,
    size_category,
)
from step20_task3_report import (
    load_models,
    load_tuned_pixel_thresholds,
    roi_probability,
)
from step17_task2_training import ATTRIBUTES, select_device
from step12_data_augmentation import LesionDataset, build_val_transform
from transformers import AutoProcessor, CLIPModel
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import csv
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

matplotlib.use("Agg")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

EXPECTED_ATTRIBUTES = (
    "pigment_network", "negative_network", "streaks",
    "milia_like_cysts", "globules",
)


# =====================================================================
# 0. Configuration
# =====================================================================

@dataclass(frozen=True)
class BonusConfig:
    fold: int = 0
    image_size: int = 384
    batch_size: int = 8
    seed: int = 42

    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_batch_size: int = 16
    top_k: int = 5
    temperature: float = 0.07

    lesion_threshold: float = 0.50

    # "classifier" or "roi".
    #
    # The ROI mean is the briefing's suggestion, but for sparse
    # attributes it sits near 0.05 on almost every image. That leaves it
    # permanently outside the confidence gate below, so no fusion ever
    # happens and the bonus reports a null result caused by the settings
    # rather than by retrieval. The classifier spans the range.
    probability_source: str = "classifier"

    absent_at: float = 0.40
    present_at: float = 0.60

    # Retrieval is a prior, not a source of facts: a small weight, and
    # only where Task 2 was already undecided.
    fusion_alpha: float = 0.20
    use_confidence_gate: bool = True
    gate_low: float = 0.25
    gate_high: float = 0.75

    force_rebuild_index: bool = False
    max_query_images: Optional[int] = None
    save_montages: int = 10

    output_root: Path = field(default_factory=lambda: (
        PROJECT_ROOT / "outputs" / "bonus_clip" / "fold_0_384px"))


def validate(config: BonusConfig) -> None:
    if tuple(ATTRIBUTES) != EXPECTED_ATTRIBUTES:
        raise ValueError(
            f"Attribute order changed.\nExpected {EXPECTED_ATTRIBUTES}\n"
            f"Received {tuple(ATTRIBUTES)}")
    if config.top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if config.temperature <= 0:
        raise ValueError("temperature must be positive.")
    if not 0.0 <= config.fusion_alpha <= 1.0:
        raise ValueError("fusion_alpha must be in [0, 1].")
    if not 0.0 <= config.absent_at < config.present_at <= 1.0:
        raise ValueError("Need 0 <= absent_at < present_at <= 1.")
    if config.probability_source not in {"classifier", "roi"}:
        raise ValueError("probability_source must be 'classifier' or 'roi'.")


# =====================================================================
# 1. Utilities
# =====================================================================

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2,
                               ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def append_jsonl(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(record), ensure_ascii=False,
                                allow_nan=False) + "\n")


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


# =====================================================================
# 2. CLIP encoding
# =====================================================================

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Undo the project's normalisation and return an RGB image."""
    restored = (image.detach().cpu() * IMAGENET_STD
                + IMAGENET_MEAN).clamp(0.0, 1.0)
    array = (restored.permute(1, 2, 0).numpy() * 255.0).round()
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def encode_images(images: list[Image.Image], processor, clip_model,
                  device) -> np.ndarray:
    """
    L2-normalised CLIP image embeddings.

    transformers versions differ in what get_image_features returns: a
    plain tensor, an object with image_embeds, or one with pooler_output.
    Whether pooler_output has already had the visual projection applied
    also varies, so decide from its width rather than assuming — 768 is
    the vision encoder's own space and needs projecting into CLIP's
    shared space, 512 is already there.
    """
    inputs = processor(images=images, return_tensors="pt")
    model_inputs = {key: value.to(device)
                    for key, value in inputs.items()
                    if isinstance(value, torch.Tensor)}

    with torch.inference_mode():
        output = clip_model.get_image_features(**model_inputs)

        if torch.is_tensor(output):
            features = output
        elif hasattr(output, "image_embeds"):
            features = output.image_embeds
        elif hasattr(output, "pooler_output"):
            features = output.pooler_output
        else:
            raise TypeError(
                "Could not extract image embeddings from "
                f"{type(output).__name__}")

        projection = getattr(clip_model, "visual_projection", None)
        if projection is not None and \
                features.shape[-1] == projection.in_features:
            features = projection(features)

        features = F.normalize(features.float(), p=2, dim=-1)

    return features.detach().cpu().numpy().astype(np.float32)


def presence_labels(masks: torch.Tensor) -> np.ndarray:
    """Five image-level labels from [5, H, W] ground-truth masks."""
    if masks.ndim != 3 or masks.shape[0] != len(ATTRIBUTES):
        raise ValueError(f"Expected [5, H, W], got {tuple(masks.shape)}")
    return (masks.reshape(len(ATTRIBUTES), -1).sum(dim=1) > 0
            ).to(dtype=torch.uint8).cpu().numpy()


# =====================================================================
# 3. Reference index
# =====================================================================

def index_paths(config: BonusConfig) -> dict[str, Path]:
    directory = config.output_root / "reference"
    return {
        "archive": directory / "clip_reference_index.npz",
        "metadata": directory / "clip_reference_metadata.csv",
        "config": directory / "clip_reference_config.json",
    }


def build_reference_index(dataset, processor, clip_model, device,
                          config: BonusConfig) -> dict[str, Any]:
    """
    Encode the training folds. Fold 0 is never added.

    Cached, keyed on the settings that would invalidate it — a different
    CLIP model, fold, resolution or attribute order forces a rebuild.
    """
    paths = index_paths(config)

    if (paths["archive"].is_file() and paths["config"].is_file()
            and not config.force_rebuild_index):
        saved = json.loads(paths["config"].read_text(encoding="utf-8"))
        matches = (saved.get("clip_model_name") == config.clip_model_name
                   and saved.get("fold") == config.fold
                   and saved.get("image_size") == config.image_size
                   and saved.get("attribute_order") == list(ATTRIBUTES))
        if matches:
            print(f"Loading cached reference index: {paths['archive']}")
            archive = np.load(paths["archive"], allow_pickle=False)
            return {
                "embeddings": archive["embeddings"].astype(np.float32),
                "labels": archive["labels"].astype(np.uint8),
                "image_ids": archive["image_ids"].astype(str),
                "dataset_indices": archive["dataset_indices"].astype(np.int64),
            }
        print("Cached index does not match this run — rebuilding.")

    print(f"Encoding {len(dataset)} reference images with CLIP...")

    embeddings, labels, image_ids, dataset_indices = [], [], [], []
    metadata_rows = []
    indices = list(range(len(dataset)))
    total_batches = math.ceil(len(indices) / config.clip_batch_size)
    started = time.time()

    for batch_number, index_batch in enumerate(
            chunks(indices, config.clip_batch_size), start=1):
        samples = [dataset[index] for index in index_batch]
        embeddings.append(encode_images(
            [tensor_to_pil(sample["image"]) for sample in samples],
            processor, clip_model, device))

        for position, sample in enumerate(samples):
            if "task2_attributes" not in sample:
                raise KeyError(
                    "The reference dataset needs include_task2=True.")

            label = presence_labels(sample["task2_attributes"])
            image_id = str(sample["image_id"])

            labels.append(label)
            image_ids.append(image_id)
            dataset_indices.append(index_batch[position])

            row = {"reference_row": len(image_ids) - 1,
                   "dataset_index": index_batch[position],
                   "image_id": image_id,
                   "role": "train_reference"}
            row.update({name: int(label[i])
                        for i, name in enumerate(ATTRIBUTES)})
            metadata_rows.append(row)

        if batch_number % 20 == 0 or batch_number == total_batches:
            print(f"  batch {batch_number}/{total_batches} "
                  f"({time.time() - started:.0f}s)")

    embedding_matrix = np.concatenate(embeddings, axis=0).astype(np.float32)
    label_matrix = np.stack(labels, axis=0).astype(np.uint8)

    if len(set(image_ids)) != len(image_ids):
        raise RuntimeError("Duplicate image IDs in the reference index.")

    paths["archive"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["archive"],
        embeddings=embedding_matrix,
        labels=label_matrix,
        image_ids=np.asarray(image_ids, dtype=str),
        dataset_indices=np.asarray(dataset_indices, dtype=np.int64))

    save_csv(metadata_rows, paths["metadata"])
    save_json({
        "clip_model_name": config.clip_model_name,
        "fold": config.fold,
        "image_size": config.image_size,
        "reference_role": "all folds except the validation fold",
        "reference_image_count": len(dataset),
        "embedding_dimension": int(embedding_matrix.shape[1]),
        "attribute_order": list(ATTRIBUTES),
        "seed": config.seed,
    }, paths["config"])

    dimension = int(embedding_matrix.shape[1])
    print(f"Reference index saved: {paths['archive']}")
    print(f"Embedding dimension: {dimension} "
          "(expect 512 for clip-vit-base-patch32; 768 would mean the "
          "visual projection was skipped)")

    return {"embeddings": embedding_matrix, "labels": label_matrix,
            "image_ids": np.asarray(image_ids, dtype=str),
            "dataset_indices": np.asarray(dataset_indices, dtype=np.int64)}


# =====================================================================
# 4. Retrieval
# =====================================================================

def stable_softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values.astype(np.float64) / temperature
    scaled -= scaled.max()
    exponentials = np.exp(scaled)
    total = exponentials.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(values.shape, 1.0 / values.size)
    return exponentials / total


def retrieve(query_embedding: np.ndarray, query_image_id: str,
             index: dict, config: BonusConfig) -> dict[str, Any]:
    """Top-K neighbours and their similarity-weighted attribute prior."""
    reference_ids = index["image_ids"]
    similarities = (index["embeddings"]
                    @ np.asarray(query_embedding,
                                 dtype=np.float32).reshape(-1)
                    ).astype(np.float64)

    # The split already makes this impossible; belt and braces.
    similarities[reference_ids == query_image_id] = -np.inf

    if int(np.isfinite(similarities).sum()) < config.top_k:
        raise RuntimeError("Fewer eligible references than top_k.")

    candidates = np.argpartition(
        similarities, kth=similarities.size - config.top_k)[-config.top_k:]
    ordered = candidates[np.argsort(similarities[candidates])[::-1]]

    top_similarities = similarities[ordered]
    weights = stable_softmax(top_similarities, config.temperature)
    prior = (weights[:, None]
             * index["labels"][ordered].astype(np.float64)).sum(axis=0)

    neighbours = []
    for rank, row in enumerate(ordered, start=1):
        neighbours.append({
            "rank": rank,
            "reference_row": int(row),
            "dataset_index": int(index["dataset_indices"][row]),
            "image_id": str(reference_ids[row]),
            "cosine_similarity": float(top_similarities[rank - 1]),
            "softmax_weight": float(weights[rank - 1]),
            "attributes": {name: int(index["labels"][row, i])
                           for i, name in enumerate(ATTRIBUTES)},
        })

    return {
        "neighbours": neighbours,
        "prior": {name: float(prior[i])
                  for i, name in enumerate(ATTRIBUTES)},
    }


def fuse(base: dict[str, float], prior: dict[str, float],
         config: BonusConfig) -> tuple[dict[str, float], dict[str, bool]]:
    """
    Blend the retrieval prior into the model probability.

    The gate is what keeps this a stabiliser rather than a source of
    facts: where Task 2 is confident, its answer stands untouched.
    """
    fused, applied = {}, {}
    for name in ATTRIBUTES:
        base_value = float(base[name])
        open_gate = (not config.use_confidence_gate
                     or config.gate_low < base_value < config.gate_high)
        value = ((1.0 - config.fusion_alpha) * base_value
                 + config.fusion_alpha * float(prior[name])
                 if open_gate else base_value)
        fused[name] = float(np.clip(value, 0.0, 1.0))
        applied[name] = bool(open_gate)
    return fused, applied


# =====================================================================
# 5. Evaluation
# =====================================================================

def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)

    true_positive = int(np.logical_and(truth, prediction).sum())
    true_negative = int(np.logical_and(~truth, ~prediction).sum())
    false_positive = int(np.logical_and(~truth, prediction).sum())
    false_negative = int(np.logical_and(truth, ~prediction).sum())

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)

    return {
        "true_positive": true_positive, "true_negative": true_negative,
        "false_positive": false_positive, "false_negative": false_negative,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": (true_positive + true_negative) / max(truth.size, 1),
    }


def selective_metrics(truth: np.ndarray, probabilities: np.ndarray,
                      config: BonusConfig) -> dict:
    """Coverage and accuracy treating 'uncertain' as an abstention."""
    present = probabilities >= config.present_at
    decided = np.logical_or(present, probabilities <= config.absent_at)
    correct = int((present[decided] == truth.astype(bool)[decided]).sum())
    decided_count = int(decided.sum())

    return {
        "total": int(truth.size),
        "decided": decided_count,
        "uncertain": int(truth.size) - decided_count,
        "coverage": decided_count / max(truth.size, 1),
        "accuracy_on_decided": correct / max(decided_count, 1),
    }


def evaluate_sets(truth: np.ndarray, sets: dict[str, np.ndarray],
                  config: BonusConfig) -> tuple[dict, list[dict]]:
    summary, rows = {}, []

    for method, probabilities in sets.items():
        per_attribute, scores = {}, {k: [] for k in
                                     ("precision", "recall", "f1",
                                      "accuracy")}

        for index, name in enumerate(ATTRIBUTES):
            column_truth = truth[:, index]
            column = probabilities[:, index]

            metrics = binary_metrics(column_truth, column >= 0.50)
            selective = selective_metrics(column_truth, column, config)
            per_attribute[name] = {**metrics, "report_status": selective}

            for key in scores:
                scores[key].append(float(metrics[key]))

            rows.append({
                "method": method, "attribute": name,
                **{k: metrics[k] for k in
                   ("precision", "recall", "f1", "accuracy",
                    "true_positive", "true_negative",
                    "false_positive", "false_negative")},
                "report_coverage": selective["coverage"],
                "report_accuracy_on_decided": selective[
                    "accuracy_on_decided"],
                "report_uncertain": selective["uncertain"],
            })

        summary[method] = {
            "per_attribute": per_attribute,
            "macro": {key: float(np.mean(values))
                      for key, values in scores.items()},
        }

    return summary, rows


# =====================================================================
# 6. Montage
# =====================================================================

def save_montage(query_image, query_id, neighbours, reference_dataset,
                 path: Path) -> None:
    columns = len(neighbours) + 1
    figure, axes = plt.subplots(1, columns, figsize=(4 * columns, 4),
                                squeeze=False)

    axes[0, 0].imshow(tensor_to_pil(query_image))
    axes[0, 0].set_title(f"Query\n{query_id}", fontsize=10)
    axes[0, 0].axis("off")

    for column, neighbour in enumerate(neighbours, start=1):
        sample = reference_dataset[neighbour["dataset_index"]]
        axes[0, column].imshow(tensor_to_pil(sample["image"]))
        axes[0, column].set_title(
            f"Rank {neighbour['rank']}\n{neighbour['image_id']}\n"
            f"sim {neighbour['cosine_similarity']:.3f}", fontsize=9)
        axes[0, column].axis("off")

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


# =====================================================================
# 7. Pipeline
# =====================================================================

def run(config: BonusConfig) -> None:
    validate(config)
    set_seed(config.seed)

    root = config.output_root
    root.mkdir(parents=True, exist_ok=True)

    audit_path = root / "audit" / "bonus_audit.jsonl"
    for path in (audit_path,
                 root / "retrieval" / "topk_neighbours.csv",
                 root / "evaluation" / "changed_fields.csv"):
        if path.exists():
            path.unlink()

    device = select_device()
    print("=" * 74)
    print("CLIP retrieval bonus")
    print("=" * 74)
    print(f"Device: {device}   CLIP: {config.clip_model_name}")
    print(f"Reference: folds other than {config.fold}. "
          f"Query: fold {config.fold} only.")

    transform = build_val_transform(image_size=config.image_size)
    reference_dataset = LesionDataset(fold=config.fold, role="train",
                                      transform=transform,
                                      include_task2=True)
    query_dataset = LesionDataset(fold=config.fold, role="val",
                                  transform=transform, include_task2=True)

    if len(reference_dataset) == 0 or len(query_dataset) == 0:
        raise RuntimeError("A dataset is empty — check the fold CSV.")

    query_count = (min(config.max_query_images, len(query_dataset))
                   if config.max_query_images else len(query_dataset))
    print(f"Reference images: {len(reference_dataset)}   "
          f"Query images: {query_count}")

    task1_model, task2_model, task1_path, task2_path = load_models(device)
    pixel_thresholds = load_tuned_pixel_thresholds(task2_path)

    print("Loading CLIP...")
    processor = AutoProcessor.from_pretrained(config.clip_model_name)
    clip_model = CLIPModel.from_pretrained(
        config.clip_model_name).to(device).eval()

    index = build_reference_index(reference_dataset, processor, clip_model,
                                  device, config)
    reference_id_set = set(index["image_ids"].tolist())

    if config.top_k > index["embeddings"].shape[0]:
        raise ValueError("top_k exceeds the reference image count.")

    loader = DataLoader(query_dataset, batch_size=config.batch_size,
                        shuffle=False, num_workers=0)

    truth_rows, base_rows, prior_rows, fused_rows = [], [], [], []
    neighbour_rows, changed_rows, audit_records = [], [], []
    leaked_query_ids = []
    processed = 0
    started = time.time()

    with torch.inference_mode():
        for batch in loader:
            if processed >= query_count:
                break

            limit = min(batch["image"].shape[0], query_count - processed)
            images_cpu = batch["image"][:limit]
            images = images_cpu.to(device, dtype=torch.float32)
            image_ids = [str(i) for i in batch["image_id"][:limit]]
            truth_masks = batch["task2_attributes"][:limit]

            lesion_masks = (torch.sigmoid(task1_model(images).float())
                            >= config.lesion_threshold)

            attribute_logits, presence_logits = task2_model(images)
            attribute_logits = attribute_logits.float()
            attribute_probabilities = torch.sigmoid(attribute_logits)
            classifier_probabilities = (
                torch.sigmoid(presence_logits.float())
                if presence_logits is not None else None)

            query_embeddings = encode_images(
                [tensor_to_pil(image) for image in images_cpu],
                processor, clip_model, device)

            for position in range(limit):
                image_id = image_ids[position]

                # Leakage check as we go, rather than a separate pass
                # over every image just to collect identifiers.
                if image_id in reference_id_set:
                    leaked_query_ids.append(image_id)

                lesion_mask = lesion_masks[position, 0].cpu().numpy(
                ).astype(np.uint8)
                geometry = lesion_geometry(lesion_mask)
                size, area_ratio = size_category(
                    geometry["lesion_pixels"], lesion_mask.size)
                border, _ = border_category(
                    geometry["component_area"],
                    geometry["perimeter_smoothed"])

                logits = attribute_logits[position].cpu().numpy()
                probabilities = attribute_probabilities[
                    position].cpu().numpy()

                base, details = {}, {}
                for index_attribute, name in enumerate(ATTRIBUTES):
                    roi_value = roi_probability(
                        logits[index_attribute], lesion_mask)
                    classifier_value = (
                        float(classifier_probabilities[
                            position, index_attribute])
                        if classifier_probabilities is not None
                        else float("nan"))

                    base[name] = (classifier_value
                                  if config.probability_source == "classifier"
                                  else roi_value)
                    details[name] = {
                        "classifier": classifier_value,
                        "roi_logit_mean": roi_value,
                        "pixel_threshold": pixel_thresholds[index_attribute],
                        "mask_pixels_in_roi": int(np.logical_and(
                            probabilities[index_attribute]
                            > pixel_thresholds[index_attribute],
                            lesion_mask == 1).sum()),
                    }

                retrieval = retrieve(query_embeddings[position], image_id,
                                     index, config)
                prior = retrieval["prior"]
                fused, applied = fuse(base, prior, config)

                base_statuses = {name: attribute_status(base[name])
                                 for name in ATTRIBUTES}
                fused_statuses = {name: attribute_status(fused[name])
                                  for name in ATTRIBUTES}

                truth = presence_labels(truth_masks[position])
                truth_rows.append(truth)
                base_rows.append(np.array([base[n] for n in ATTRIBUTES],
                                          dtype=np.float32))
                prior_rows.append(np.array([prior[n] for n in ATTRIBUTES],
                                           dtype=np.float32))
                fused_rows.append(np.array([fused[n] for n in ATTRIBUTES],
                                           dtype=np.float32))

                changed_status = [n for n in ATTRIBUTES
                                  if base_statuses[n] != fused_statuses[n]]
                changed_binary = [n for n in ATTRIBUTES
                                  if (base[n] >= 0.5) != (fused[n] >= 0.5)]

                for name in changed_binary:
                    truth_value = bool(truth[ATTRIBUTES.index(name)])
                    before = base[name] >= 0.5
                    after = fused[name] >= 0.5
                    changed_rows.append({
                        "image_id": image_id, "attribute": name,
                        "ground_truth": int(truth_value),
                        "base_probability": base[name],
                        "prior_probability": prior[name],
                        "fused_probability": fused[name],
                        "base_correct": int(before == truth_value),
                        "fused_correct": int(after == truth_value),
                        "effect": ("corrected"
                                   if before != truth_value
                                   and after == truth_value
                                   else "damaged"
                                   if before == truth_value
                                   and after != truth_value
                                   else "neutral"),
                    })

                lesion_summary = {
                    "size_category": size,
                    "area_ratio": round(area_ratio, 4),
                    "border_category": border,
                    "border_index": round(
                        geometry["border_index_smoothed"], 4),
                    "border_index_raw": round(
                        geometry["border_index_raw"], 4),
                    "lesion_pixels": geometry["lesion_pixels"],
                }

                metadata = {
                    "image_id": image_id,
                    "split": "val",
                    "model_version": f"{task1_path.name}, {task2_path.name}",
                    "attributes_order": list(ATTRIBUTES),
                }

                for label, probabilities_map, statuses in (
                        ("baseline", base, base_statuses),
                        ("bonus", fused, fused_statuses)):
                    presence = {
                        name: {"prob": round(probabilities_map[name], 4),
                               "status": statuses[name],
                               **({"base_prob": round(base[name], 4),
                                   "prior_prob": round(prior[name], 4),
                                   "fusion_applied": applied[name]}
                                  if label == "bonus" else details[name])}
                        for name in ATTRIBUTES}

                    report = {**metadata,
                              "report_type": label,
                              "outputs": {"presence": presence,
                                          "lesion": lesion_summary}}

                    if label == "bonus":
                        report["retrieval"] = {
                            "clip_model": config.clip_model_name,
                            "top_k": config.top_k,
                            "temperature": config.temperature,
                            "fusion_alpha": config.fusion_alpha,
                            "confidence_gate": [config.gate_low,
                                                config.gate_high],
                            "neighbours": retrieval["neighbours"],
                            "changed_status_fields": changed_status,
                        }

                    text = build_report_text(size, border, statuses)
                    report["template_text"] = text

                    save_json(report,
                              root / "reports" / label / f"{image_id}.json")
                    (root / "reports" / label
                     / f"{image_id}.txt").write_text(text, encoding="utf-8")

                record = {
                    "query_image_id": image_id,
                    "neighbours": retrieval["neighbours"],
                    "lesion": lesion_summary,
                    "probability_source": config.probability_source,
                    "probability_details": details,
                    "base": base, "prior": prior, "fused": fused,
                    "base_statuses": base_statuses,
                    "fused_statuses": fused_statuses,
                    "fusion_applied": applied,
                    "changed_status_fields": changed_status,
                    "changed_binary_fields": changed_binary,
                    "ground_truth_for_evaluation_only": {
                        name: int(truth[i])
                        for i, name in enumerate(ATTRIBUTES)},
                }
                append_jsonl(record, audit_path)
                audit_records.append(record)

                for neighbour in retrieval["neighbours"]:
                    row = {"query_image_id": image_id,
                           "neighbour_rank": neighbour["rank"],
                           "neighbour_image_id": neighbour["image_id"],
                           "cosine_similarity":
                               neighbour["cosine_similarity"],
                           "softmax_weight": neighbour["softmax_weight"]}
                    row.update({f"neighbour_{n}": neighbour["attributes"][n]
                                for n in ATTRIBUTES})
                    neighbour_rows.append(row)

                if processed < config.save_montages:
                    save_montage(images_cpu[position], image_id,
                                 retrieval["neighbours"], reference_dataset,
                                 root / "montages" / f"{image_id}.png")

                processed += 1
                if processed % 25 == 0 or processed == query_count:
                    print(f"  {processed}/{query_count} "
                          f"({time.time() - started:.0f}s)")

    if leaked_query_ids:
        raise RuntimeError(
            "Leakage: query images found in the reference index — "
            f"{sorted(set(leaked_query_ids))[:10]}")

    save_csv(neighbour_rows, root / "retrieval" / "topk_neighbours.csv")
    save_csv(changed_rows, root / "evaluation" / "changed_fields.csv")

    summarise(config, root, task1_path, task2_path, reference_dataset,
              audit_records, changed_rows, reference_id_set,
              np.stack(truth_rows), np.stack(base_rows),
              np.stack(prior_rows), np.stack(fused_rows),
              processed, time.time() - started)


def summarise(config, root, task1_path, task2_path, reference_dataset,
              audit_records, changed_rows, reference_id_set,
              truth, base, prior, fused, processed, seconds) -> None:
    evaluation, metric_rows = evaluate_sets(
        truth,
        {"task2_baseline": base,
         "clip_retrieval_only": prior,
         "task2_clip_fused": fused},
        config)

    def count(effect: str) -> int:
        return sum(row["effect"] == effect for row in changed_rows)

    corrected, damaged, neutral = (count("corrected"), count("damaged"),
                                   count("neutral"))

    checks = {
        "query_count": processed,
        "unique_queries": len({r["query_image_id"] for r in audit_records}),
        "all_queries_have_top_k": all(
            len(r["neighbours"]) == config.top_k for r in audit_records),
        "queries_with_duplicate_neighbours": sum(
            len({n["image_id"] for n in r["neighbours"]})
            != len(r["neighbours"]) for r in audit_records),
        "self_matches": sum(n["image_id"] == r["query_image_id"]
                            for r in audit_records
                            for n in r["neighbours"]),
        "neighbours_outside_reference": sum(
            n["image_id"] not in reference_id_set
            for r in audit_records for n in r["neighbours"]),
        "fusion_applied_rate": float(np.mean(
            [sum(r["fusion_applied"].values()) / len(ATTRIBUTES)
             for r in audit_records])) if audit_records else 0.0,
    }

    save_json({
        "configuration": asdict(config),
        "software": {"python": platform.python_version(),
                     "pytorch": torch.__version__,
                     "numpy": np.__version__,
                     "platform": platform.platform()},
        "models": {"task1": str(task1_path), "task2": str(task2_path)},
        "data": {"reference_images": len(reference_dataset),
                 "query_images": processed},
        "evaluation": evaluation,
        "retrieval_impact": {
            "changed_binary_fields": len(changed_rows),
            "corrected": corrected, "damaged": damaged, "neutral": neutral,
            "net_corrected": corrected - damaged},
        "audit_checks": checks,
        "runtime_minutes": seconds / 60.0,
    }, root / "evaluation" / "bonus_evaluation_summary.json")

    save_csv(metric_rows, root / "evaluation" / "per_attribute_metrics.csv")

    print("\n" + "=" * 74)
    print("BONUS RESULTS")
    print("=" * 74)
    print(f"{'method':<24}{'macro F1':>10}{'precision':>11}{'recall':>9}")
    for method in ("task2_baseline", "clip_retrieval_only",
                   "task2_clip_fused"):
        macro = evaluation[method]["macro"]
        print(f"{method:<24}{macro['f1']:>10.4f}"
              f"{macro['precision']:>11.4f}{macro['recall']:>9.4f}")

    print("-" * 74)
    print(f"Fusion applied to {100 * checks['fusion_applied_rate']:.1f}% "
          "of attribute decisions")
    print(f"Changed binary fields : {len(changed_rows)}")
    print(f"  corrected           : {corrected}")
    print(f"  damaged             : {damaged}")
    print(f"  net                 : {corrected - damaged}")
    print("-" * 74)
    print(f"Self matches                  : {checks['self_matches']}")
    print(f"Neighbours outside reference  : "
          f"{checks['neighbours_outside_reference']}")
    print(f"Queries with duplicate nbrs   : "
          f"{checks['queries_with_duplicate_neighbours']}")
    print("=" * 74)
    print(f"Runtime {seconds / 60:.1f} min. Outputs in {root}")


if __name__ == "__main__":
    run(BonusConfig(
        # Set to 10 for a smoke test, None for all 540 validation images.
        max_query_images=None,
    ))
