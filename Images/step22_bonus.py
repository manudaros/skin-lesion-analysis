"""
Step 22 — bonus: CLIP retrieval-augmented reporting.

For each validation image, retrieves visually similar training cases and
uses their ground-truth attribute labels as a weak prior, blended into the
Task 2 presence probability. The briefing's constraint is that retrieval
may stabilise an estimate but must never supply a fact: no claim in a
report comes from a neighbour, only a nudge to a probability the model
already produced. The confidence gate enforces that — fusion only applies
where Task 2 was undecided in the first place.

Rewritten to read step 20's cache. That means no Task 2 model is loaded
here, and the baseline probabilities are the step 18b classification head
rather than a separate estimator — so the baseline in this file is the same
number as the one in the Task 3 reports. Reports are built with step 20's
own build_report, so the bonus and baseline reports differ only in the
probabilities that went in.

Leakage safety is the thing that matters most:

  * The reference index is built from the training folds only. The
    validation fold is query-only and never indexed.
  * Any self-match is excluded at retrieval time.
  * Query ids are checked against reference ids as they are processed, and
    the audit counts neighbours falling outside the reference split. Both
    should be zero.

Three probability sets are evaluated: Task 2 alone, retrieval alone, and
fused. Retrieval alone is what shows whether fusion beats either part.

Run step 20 first so the cache exists.

    python step22_bonus.py
"""

from __future__ import annotations

import copy
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPModel

from step12_data_augmentation import LesionDataset
from step17_task2_training import ATTRIBUTES, NUM_ATTRIBUTES, select_device
from step19_task3_rubric import STATUS_PRESENT_AT, attribute_status
from step20_task3_report import (
    CACHE_MASKS,
    CFG,
    FOLD,
    build_report,
    load_or_run_inference,
    load_task1_cache,
    read_mask,
    unpack_masks,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =====================================================================
# 0. Configuration
# =====================================================================

@dataclass(frozen=True)
class BonusConfig:
    seed: int = 42

    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_batch_size: int = 16
    top_k: int = 5
    temperature: float = 0.07

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
        PROJECT_ROOT / "outputs" / "bonus_clip" / f"fold_{FOLD}_from_18b"))


def validate(config: BonusConfig) -> None:
    if config.top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if config.temperature <= 0:
        raise ValueError("temperature must be positive.")
    if not 0.0 <= config.fusion_alpha <= 1.0:
        raise ValueError("fusion_alpha must be in [0, 1].")


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
# Raw dataset images go straight to PIL. With transform=None they are
# unnormalised, so there is no ImageNet round trip to get wrong, and CLIP's
# own processor handles the resize.

def to_pil(tensor: torch.Tensor) -> Image.Image:
    array = torch.as_tensor(tensor).detach().cpu()
    if array.dtype != torch.uint8:
        array = (array.clamp(0, 1) * 255).to(torch.uint8)
    return Image.fromarray(array.permute(1, 2, 0).numpy(), mode="RGB")


def encode_images(images: list[Image.Image], processor, clip_model,
                  device) -> np.ndarray:
    """
    L2-normalised CLIP image embeddings.

    transformers versions differ in what get_image_features returns, and in
    whether pooler_output has already had the visual projection applied.
    Decide from the tensor width rather than assuming: 768 is the vision
    encoder's own space and needs projecting, 512 is already there.
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
                f"Cannot get embeddings from {type(output).__name__}")

        projection = getattr(clip_model, "visual_projection", None)
        if projection is not None and \
                features.shape[-1] == projection.in_features:
            features = projection(features)

        features = F.normalize(features.float(), p=2, dim=-1)

    return features.detach().cpu().numpy().astype(np.float32)


def presence_labels(masks) -> np.ndarray:
    """Five image-level labels from [5, H, W] ground-truth masks."""
    tensor = torch.as_tensor(masks)
    if tensor.ndim != 3 or tensor.shape[0] != NUM_ATTRIBUTES:
        raise ValueError(f"Expected [5, H, W], got {tuple(tensor.shape)}")
    return (tensor.reshape(NUM_ATTRIBUTES, -1).sum(dim=1) > 0
            ).to(dtype=torch.uint8).cpu().numpy()


# =====================================================================
# 3. Reference index
# =====================================================================

def index_paths(config: BonusConfig) -> dict[str, Path]:
    directory = config.output_root / "reference"
    return {"archive": directory / "clip_reference_index.npz",
            "metadata": directory / "clip_reference_metadata.csv",
            "config": directory / "clip_reference_config.json"}


def build_reference_index(dataset, processor, clip_model, device,
                          config: BonusConfig) -> dict[str, Any]:
    """
    Encode the training folds. The validation fold is never added.

    Cached, keyed on the settings that would invalidate it — a different
    CLIP model, fold or attribute order forces a rebuild.
    """
    paths = index_paths(config)

    if (paths["archive"].is_file() and paths["config"].is_file()
            and not config.force_rebuild_index):
        saved = json.loads(paths["config"].read_text(encoding="utf-8"))
        if (saved.get("clip_model_name") == config.clip_model_name
                and saved.get("fold") == FOLD
                and saved.get("attribute_order") == list(ATTRIBUTES)
                and saved.get("reference_image_count") == len(dataset)):
            print(f"Loading cached reference index: {paths['archive'].name}")
            archive = np.load(paths["archive"], allow_pickle=False)
            return {"embeddings": archive["embeddings"].astype(np.float32),
                    "labels": archive["labels"].astype(np.uint8),
                    "image_ids": archive["image_ids"].astype(str),
                    "dataset_indices":
                        archive["dataset_indices"].astype(np.int64)}
        print("Cached index does not match this run — rebuilding.")

    print(f"Encoding {len(dataset)} reference images with CLIP...")

    embeddings, labels, image_ids, indices = [], [], [], []
    metadata_rows = []
    total_batches = math.ceil(len(dataset) / config.clip_batch_size)
    started = time.time()

    for batch_number, index_batch in enumerate(
            chunks(list(range(len(dataset))), config.clip_batch_size),
            start=1):
        samples = [dataset[index] for index in index_batch]
        embeddings.append(encode_images(
            [to_pil(sample["image"]) for sample in samples],
            processor, clip_model, device))

        for position, sample in enumerate(samples):
            if "task2_attributes" not in sample:
                raise KeyError("Reference dataset needs include_task2=True.")

            label = presence_labels(sample["task2_attributes"])
            image_id = str(sample["image_id"])

            labels.append(label)
            image_ids.append(image_id)
            indices.append(index_batch[position])

            row = {"reference_row": len(image_ids) - 1,
                   "dataset_index": index_batch[position],
                   "image_id": image_id, "role": "train_reference"}
            row.update({name: int(label[i])
                        for i, name in enumerate(ATTRIBUTES)})
            metadata_rows.append(row)

        if batch_number % 20 == 0 or batch_number == total_batches:
            print(f"  batch {batch_number}/{total_batches} "
                  f"({time.time() - started:.0f}s)")

    embedding_matrix = np.concatenate(embeddings, axis=0).astype(np.float32)
    label_matrix = np.stack(labels, axis=0).astype(np.uint8)

    if len(set(image_ids)) != len(image_ids):
        raise RuntimeError("Duplicate image ids in the reference index.")

    paths["archive"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["archive"], embeddings=embedding_matrix, labels=label_matrix,
        image_ids=np.asarray(image_ids, dtype=str),
        dataset_indices=np.asarray(indices, dtype=np.int64))

    save_csv(metadata_rows, paths["metadata"])
    save_json({"clip_model_name": config.clip_model_name, "fold": FOLD,
               "reference_role": "all folds except the validation fold",
               "reference_image_count": len(dataset),
               "embedding_dimension": int(embedding_matrix.shape[1]),
               "attribute_order": list(ATTRIBUTES),
               "seed": config.seed}, paths["config"])

    print(f"Reference index saved. Embedding dimension: "
          f"{embedding_matrix.shape[1]} "
          "(expect 512 for clip-vit-base-patch32)")

    return {"embeddings": embedding_matrix, "labels": label_matrix,
            "image_ids": np.asarray(image_ids, dtype=str),
            "dataset_indices": np.asarray(indices, dtype=np.int64)}


# =====================================================================
# 4. Retrieval and fusion
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

    neighbours = [{
        "rank": rank,
        "dataset_index": int(index["dataset_indices"][row]),
        "image_id": str(reference_ids[row]),
        "cosine_similarity": float(top_similarities[rank - 1]),
        "softmax_weight": float(weights[rank - 1]),
        "attributes": {name: int(index["labels"][row, i])
                       for i, name in enumerate(ATTRIBUTES)},
    } for rank, row in enumerate(ordered, start=1)]

    return {"neighbours": neighbours,
            "prior": {name: float(prior[i])
                      for i, name in enumerate(ATTRIBUTES)}}


def fuse(base: dict[str, float], prior: dict[str, float],
         config: BonusConfig) -> tuple[dict[str, float], dict[str, bool]]:
    """
    Blend the retrieval prior into the model probability.

    The gate is what keeps this a stabiliser rather than a source of facts:
    where Task 2 is confident, its answer stands untouched.
    """
    fused, applied = {}, {}
    for name in ATTRIBUTES:
        value = float(base[name])
        open_gate = (not config.use_confidence_gate
                     or config.gate_low < value < config.gate_high)
        blended = ((1.0 - config.fusion_alpha) * value
                   + config.fusion_alpha * float(prior[name])
                   if open_gate else value)
        fused[name] = float(np.clip(blended, 0.0, 1.0))
        applied[name] = bool(open_gate)
    return fused, applied


# =====================================================================
# 5. Evaluation
# =====================================================================

def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth, prediction = truth.astype(bool), prediction.astype(bool)
    true_positive = int(np.logical_and(truth, prediction).sum())
    true_negative = int(np.logical_and(~truth, ~prediction).sum())
    false_positive = int(np.logical_and(~truth, prediction).sum())
    false_negative = int(np.logical_and(truth, ~prediction).sum())

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)

    return {"true_positive": true_positive, "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": (true_positive + true_negative) / max(truth.size, 1)}


def selective_metrics(truth: np.ndarray,
                      probabilities: np.ndarray) -> dict:
    """
    Coverage and accuracy treating 'uncertain' as an abstention.

    Uses the rubric's fixed 0.60/0.40, since that is what the reports
    actually apply.
    """
    present = probabilities >= STATUS_PRESENT_AT
    decided = np.logical_or(present, probabilities <= 0.40)
    correct = int((present[decided] == truth.astype(bool)[decided]).sum())
    count = int(decided.sum())

    return {"total": int(truth.size), "decided": count,
            "uncertain": int(truth.size) - count,
            "coverage": count / max(truth.size, 1),
            "accuracy_on_decided": correct / max(count, 1)}


def evaluate_sets(truth: np.ndarray,
                  sets: dict[str, np.ndarray]) -> tuple[dict, list[dict]]:
    summary, rows = {}, []

    for method, probabilities in sets.items():
        per_attribute = {}
        scores = {k: [] for k in ("precision", "recall", "f1", "accuracy")}

        for index, name in enumerate(ATTRIBUTES):
            column_truth = truth[:, index]
            column = probabilities[:, index]

            metrics = binary_metrics(column_truth, column >= 0.50)
            selective = selective_metrics(column_truth, column)
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
                "report_accuracy_on_decided":
                    selective["accuracy_on_decided"],
                "report_uncertain": selective["uncertain"]})

        summary[method] = {
            "per_attribute": per_attribute,
            "macro": {key: float(np.mean(values))
                      for key, values in scores.items()}}

    return summary, rows


# =====================================================================
# 6. Montage
# =====================================================================

def save_montage(query_image, query_id: str, neighbours,
                 reference_dataset, path: Path) -> None:
    columns = len(neighbours) + 1
    figure, axes = plt.subplots(1, columns, figsize=(4 * columns, 4),
                                squeeze=False)

    axes[0, 0].imshow(to_pil(query_image))
    axes[0, 0].set_title(f"Query\n{query_id}", fontsize=10)
    axes[0, 0].axis("off")

    for column, neighbour in enumerate(neighbours, start=1):
        sample = reference_dataset[neighbour["dataset_index"]]
        axes[0, column].imshow(to_pil(sample["image"]))
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
    for path in (audit_path, root / "retrieval" / "topk_neighbours.csv",
                 root / "evaluation" / "changed_fields.csv"):
        if path.exists():
            path.unlink()

    device = select_device()
    print("=" * 74)
    print("CLIP retrieval bonus")
    print("=" * 74)
    print(f"Device: {device}   CLIP: {config.clip_model_name}")
    print(f"Reference: folds other than {FOLD}. Query: fold {FOLD} only.")

    # Task 2 baseline probabilities come from step 20's cache, so the
    # baseline here is the same number the Task 3 reports use.
    payload = load_or_run_inference()
    metadata, cached = payload["metadata"], payload["images"]
    _, mask_files = load_task1_cache()
    print(f"Task 2 baseline: {metadata['presence_checkpoint']} "
          f"from {metadata['source_run']}")

    reference_dataset = LesionDataset(fold=FOLD, role="train",
                                      transform=None, include_task2=True)
    query_dataset = LesionDataset(fold=FOLD, role="val", transform=None,
                                  include_task2=True)

    query_count = (min(config.max_query_images, len(query_dataset))
                   if config.max_query_images else len(query_dataset))
    print(f"Reference images: {len(reference_dataset)}   "
          f"Query images: {query_count}")

    print("Loading CLIP...")
    processor = AutoProcessor.from_pretrained(config.clip_model_name)
    clip_model = CLIPModel.from_pretrained(
        config.clip_model_name).to(device).eval()

    index = build_reference_index(reference_dataset, processor, clip_model,
                                 device, config)
    reference_id_set = set(index["image_ids"].tolist())

    truth_rows, base_rows, prior_rows, fused_rows = [], [], [], []
    neighbour_rows, changed_rows, audit_records = [], [], []
    leaked: list[str] = []
    processed = 0
    started = time.time()

    for position in range(query_count):
        sample = query_dataset[position]
        image_id = str(sample["image_id"])

        if image_id not in cached:
            raise KeyError(
                f"{image_id} is not in the Task 2 cache. Re-run step 20 "
                "over the full fold.")

        # Leakage check as we go, rather than a separate pass.
        if image_id in reference_id_set:
            leaked.append(image_id)

        record = cached[image_id]
        base = {name: float(record["attributes"][name]["prob"])
                for name in ATTRIBUTES}

        embedding = encode_images([to_pil(sample["image"])], processor,
                                  clip_model, device)[0]
        retrieval = retrieve(embedding, image_id, index, config)
        prior = retrieval["prior"]
        fused, applied = fuse(base, prior, config)

        truth = presence_labels(sample["task2_attributes"])
        truth_rows.append(truth)
        base_rows.append(np.array([base[n] for n in ATTRIBUTES],
                                  dtype=np.float32))
        prior_rows.append(np.array([prior[n] for n in ATTRIBUTES],
                                   dtype=np.float32))
        fused_rows.append(np.array([fused[n] for n in ATTRIBUTES],
                                   dtype=np.float32))

        # --- reports, both built by step 20's own code ----------------
        task1_mask = read_mask(CFG.output_dir / mask_files[image_id])
        attribute_masks = unpack_masks(
            np.asarray(Image.open(CACHE_MASKS / f"{image_id}.png")))

        baseline_report, baseline_text, _ = build_report(
            image_id, record, task1_mask, attribute_masks, metadata)

        fused_record = copy.deepcopy(record)
        for name in ATTRIBUTES:
            fused_record["attributes"][name]["prob"] = fused[name]
        bonus_report, bonus_text, bonus_checks = build_report(
            image_id, fused_record, task1_mask, attribute_masks, metadata)

        bonus_report["retrieval"] = {
            "clip_model": config.clip_model_name,
            "top_k": config.top_k,
            "temperature": config.temperature,
            "fusion_alpha": config.fusion_alpha,
            "confidence_gate": [config.gate_low, config.gate_high],
            "neighbours": retrieval["neighbours"],
            "base_probabilities": base,
            "prior_probabilities": prior,
            "fusion_applied": applied,
        }

        for label, report, text in (("baseline", baseline_report,
                                     baseline_text),
                                    ("bonus", bonus_report, bonus_text)):
            save_json(report, root / "reports" / label / f"{image_id}.json")
            (root / "reports" / label / f"{image_id}.txt").write_text(
                text, encoding="utf-8")

        # --- what retrieval changed -----------------------------------
        base_statuses = {n: attribute_status(base[n]) for n in ATTRIBUTES}
        fused_statuses = {n: attribute_status(fused[n]) for n in ATTRIBUTES}
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
                "effect": ("corrected" if before != truth_value
                           and after == truth_value
                           else "damaged" if before == truth_value
                           and after != truth_value else "neutral")})

        record_out = {
            "query_image_id": image_id,
            "neighbours": retrieval["neighbours"],
            "base": base, "prior": prior, "fused": fused,
            "base_statuses": base_statuses,
            "fused_statuses": fused_statuses,
            "fusion_applied": applied,
            "changed_status_fields": changed_status,
            "changed_binary_fields": changed_binary,
            "bonus_checks_passed": bonus_checks["passed"],
            "ground_truth_for_evaluation_only": {
                name: int(truth[i]) for i, name in enumerate(ATTRIBUTES)}}
        append_jsonl(record_out, audit_path)
        audit_records.append(record_out)

        for neighbour in retrieval["neighbours"]:
            row = {"query_image_id": image_id,
                   "neighbour_rank": neighbour["rank"],
                   "neighbour_image_id": neighbour["image_id"],
                   "cosine_similarity": neighbour["cosine_similarity"],
                   "softmax_weight": neighbour["softmax_weight"]}
            row.update({f"neighbour_{n}": neighbour["attributes"][n]
                        for n in ATTRIBUTES})
            neighbour_rows.append(row)

        if processed < config.save_montages:
            save_montage(sample["image"], image_id,
                         retrieval["neighbours"], reference_dataset,
                         root / "montages" / f"{image_id}.png")

        processed += 1
        if processed % 25 == 0 or processed == query_count:
            print(f"  {processed}/{query_count} "
                  f"({time.time() - started:.0f}s)")

    if leaked:
        raise RuntimeError(
            "Leakage: query images found in the reference index — "
            f"{sorted(set(leaked))[:10]}")

    save_csv(neighbour_rows, root / "retrieval" / "topk_neighbours.csv")
    save_csv(changed_rows, root / "evaluation" / "changed_fields.csv")

    summarise(config, root, metadata, reference_dataset, audit_records,
              changed_rows, reference_id_set, np.stack(truth_rows),
              np.stack(base_rows), np.stack(prior_rows),
              np.stack(fused_rows), processed, time.time() - started)


def summarise(config, root, metadata, reference_dataset, audit_records,
              changed_rows, reference_id_set, truth, base, prior, fused,
              processed, seconds) -> None:
    evaluation, metric_rows = evaluate_sets(
        truth, {"task2_baseline": base, "clip_retrieval_only": prior,
                "task2_clip_fused": fused})

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
            [sum(r["fusion_applied"].values()) / NUM_ATTRIBUTES
             for r in audit_records])) if audit_records else 0.0,
    }

    save_json({
        "configuration": asdict(config),
        "task2_baseline_source": {
            "checkpoint": metadata["presence_checkpoint"],
            "run": metadata["source_run"],
            "trained_on": f"{metadata['crop_image_size']}px lesion crops"},
        "software": {"python": platform.python_version(),
                     "pytorch": torch.__version__,
                     "numpy": np.__version__,
                     "platform": platform.platform()},
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
    print(f"Self matches                 : {checks['self_matches']}")
    print(f"Neighbours outside reference : "
          f"{checks['neighbours_outside_reference']}")
    print(f"Queries with duplicate nbrs  : "
          f"{checks['queries_with_duplicate_neighbours']}")
    print("=" * 74)
    print(f"Runtime {seconds / 60:.1f} min. Outputs in {root}")


if __name__ == "__main__":
    run(BonusConfig(
        # Set to 10 for a smoke test, None for all 540 validation images.
        max_query_images=None,
    ))