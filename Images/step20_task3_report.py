"""
Step 19 — Task 3 anchored findings report.

Runs both trained models over the validation fold and writes, per image:
a JSON report matching the briefing's schema, a line in a summary CSV,
and a row in a consistency audit.

Two things differ from the obvious implementation.

Presence probability. The briefing suggests p_attr = mean of the
attribute logits inside the predicted lesion ROI. That estimator degrades
badly for the sparse attributes: streaks cover roughly 0.05% of pixels,
so averaged across a whole lesion the probability never approaches 0.60
and every report reads "streaks are absent" regardless of the image. The
step 17 model has a presence head trained directly on the image-level
question, so PRESENCE_SOURCE defaults to it. The ROI estimator is still
computed and stored alongside, so the two can be compared in the report.

Checkpoint loading is strict. A missing file raises rather than quietly
leaving a randomly initialised decoder in place, which produces perfectly
plausible-looking output from an untrained model.

    python step19_task3_report.py
"""

from __future__ import annotations
from step19_task3_rubric import (
    attribute_status,
    border_category,
    build_report_text,
    check_report,
    lesion_geometry,
    size_category,
    summarise_audit,
)
from step17_task2_training import (
    ATTRIBUTES,
    Task2ModelConfig,
    build_task2_model,
    locate_task1_checkpoint,
    select_device,
)
from step14_task1_training import build_task1_model
from step12_data_augmentation import LesionDataset, build_val_transform
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


# =====================================================================
# 0. Settings
# =====================================================================

FOLD = 0
IMAGE_SIZE = 384              # must match the resolution both models saw
SPLIT_NAME = "val"            # the fold being reported on
LESION_THRESHOLD = 0.5

# "classifier" uses the step 17 presence head.
# "roi" uses the briefing's mean-logit-inside-ROI estimator.
PRESENCE_SOURCE = "classifier"

# Use per-attribute pixel thresholds from step 18 if they were saved.
USE_TUNED_PIXEL_THRESHOLDS = True
DEFAULT_PIXEL_THRESHOLD = 0.5

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task3_reports"


# =====================================================================
# 1. Locating the trained models
# =====================================================================

def locate_task2_checkpoint(fold: int = FOLD,
                            image_size: int = IMAGE_SIZE) -> Path | None:
    name = "task2_best_model.pth"
    candidates = [
        PROJECT_ROOT / "outputs" / "task2_training"
        / f"fold_{fold}_{image_size}px" / name,
        SCRIPT_DIR / "outputs" / "task2_training"
        / f"fold_{fold}_{image_size}px" / name,
        SCRIPT_DIR / "outputs" / "task2_results" / name,
        PROJECT_ROOT / "outputs" / "training_results" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_tuned_pixel_thresholds(checkpoint_path: Path) -> list[float]:
    """Read step 18's tuned per-attribute pixel thresholds, if present."""
    path = checkpoint_path.parent / "task2_best_thresholds.json"
    if not (USE_TUNED_PIXEL_THRESHOLDS and path.is_file()):
        return [DEFAULT_PIXEL_THRESHOLD] * len(ATTRIBUTES)

    data = json.loads(path.read_text(encoding="utf-8"))
    tuned = data.get("pixel_thresholds", {})
    thresholds = [float(tuned.get(name, DEFAULT_PIXEL_THRESHOLD))
                  for name in ATTRIBUTES]
    print(f"Using tuned pixel thresholds from {path.name}: "
          + ", ".join(f"{n}={t:.2f}"
                      for n, t in zip(ATTRIBUTES, thresholds)))
    return thresholds


def load_models(device):
    """Load both trained models, or raise. No silent untrained fallback."""
    task1_path = locate_task1_checkpoint(fold=FOLD, image_size=IMAGE_SIZE)
    if task1_path is None:
        raise FileNotFoundError(
            f"No Task 1 checkpoint for fold {FOLD} at {IMAGE_SIZE}px. "
            "Run step 15 first."
        )

    task2_path = locate_task2_checkpoint()
    if task2_path is None:
        raise FileNotFoundError(
            f"No Task 2 checkpoint for fold {FOLD} at {IMAGE_SIZE}px. "
            "Run step 18 first."
        )

    print(f"Task 1 model: {task1_path}")
    print(f"Task 2 model: {task2_path}")

    task1_model = build_task1_model().to(device)
    task1_model.load_state_dict(
        torch.load(task1_path, map_location=device, weights_only=True))
    task1_model.eval()

    # The head must match how the model was trained, or the load fails.
    task2_model = build_task2_model(
        Task2ModelConfig(use_classification_head=True)).to(device)
    task2_model.load_state_dict(
        torch.load(task2_path, map_location=device, weights_only=True))
    task2_model.eval()

    return task1_model, task2_model, task1_path, task2_path


# =====================================================================
# 2. Presence probability
# =====================================================================

def roi_probability(attribute_logits: np.ndarray,
                    lesion_mask: np.ndarray) -> float:
    """
    The briefing's estimator: mean of the attribute logits over the
    predicted lesion ROI, squashed to a probability.

    Averaging the logits and then applying sigmoid is not the same as
    averaging the probabilities; the former is what the briefing
    describes, and it is less dominated by the many near-zero pixels.
    Falls back to the whole image when the lesion mask is empty.
    """
    region = attribute_logits[lesion_mask == 1]
    if region.size == 0:
        region = attribute_logits.reshape(-1)
    return float(1.0 / (1.0 + np.exp(-region.mean())))


# =====================================================================
# 3. One image
# =====================================================================

def process_image(batch, task1_model, task2_model, device,
                  pixel_thresholds, model_version: str):
    images = batch["image"].to(device)
    image_id = str(batch["image_id"][0])
    total_pixels = images.shape[2] * images.shape[3]

    lesion_probabilities = torch.sigmoid(task1_model(images))
    lesion_mask = (lesion_probabilities > LESION_THRESHOLD
                   ).squeeze().cpu().numpy().astype(np.uint8)

    geometry = lesion_geometry(lesion_mask)
    size, area_ratio = size_category(geometry["lesion_pixels"], total_pixels)
    border, _ = border_category(geometry["component_area"],
                                geometry["perimeter_smoothed"])

    attribute_logits, presence_logits = task2_model(images)
    attribute_logits = attribute_logits.squeeze(0).cpu().numpy()
    attribute_probabilities = 1.0 / (1.0 + np.exp(-attribute_logits))

    if presence_logits is not None:
        classifier_probabilities = torch.sigmoid(
            presence_logits).squeeze(0).cpu().numpy()
    else:
        classifier_probabilities = np.full(len(ATTRIBUTES), float("nan"))

    presence, statuses, evidence_pixels = {}, {}, {}

    for index, name in enumerate(ATTRIBUTES):
        roi_probability_value = roi_probability(
            attribute_logits[index], lesion_mask)
        classifier_value = float(classifier_probabilities[index])

        probability = (classifier_value if PRESENCE_SOURCE == "classifier"
                       else roi_probability_value)
        status = attribute_status(probability)
        statuses[name] = status

        attribute_mask = (attribute_probabilities[index]
                          > pixel_thresholds[index])
        evidence_pixels[name] = int(
            np.logical_and(attribute_mask, lesion_mask == 1).sum())

        presence[name] = {
            "prob": round(float(probability), 4),
            "status": status,
            "prob_classifier": round(classifier_value, 4),
            "prob_roi_logit_mean": round(roi_probability_value, 4),
            "evidence_pixels_in_roi": evidence_pixels[name],
            "pixel_threshold": round(float(pixel_thresholds[index]), 2),
        }

    report = {
        "image_id": image_id,
        "split": SPLIT_NAME,
        "model_version": model_version,
        "attributes_order": list(ATTRIBUTES),
        "outputs": {
            "presence": presence,
            "lesion": {
                "size_category": size,
                "area_ratio": round(area_ratio, 4),
                "border_category": border,
                "border_index": round(geometry["border_index_smoothed"], 4),
                "border_index_raw": round(geometry["border_index_raw"], 4),
                "lesion_pixels": geometry["lesion_pixels"],
                "n_components": geometry["n_components"],
            },
        },
        "presence_source": PRESENCE_SOURCE,
    }

    text = build_report_text(size, border, statuses)
    checks = check_report(report, text, evidence_pixels)

    return report, text, checks


# =====================================================================
# 4. Main
# =====================================================================

def generate(sample_mode: bool = False):
    device = select_device()
    print(f"\nGenerating Task 3 reports on {device}")

    json_dir = OUTPUT_DIR / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    task1_model, task2_model, task1_path, task2_path = load_models(device)
    pixel_thresholds = load_tuned_pixel_thresholds(task2_path)
    model_version = f"{task1_path.name}, {task2_path.name}"

    dataset = LesionDataset(
        fold=FOLD, role="val",
        transform=build_val_transform(image_size=IMAGE_SIZE),
        include_task2=False)
    expected_count = len(dataset)

    if sample_mode:
        dataset = Subset(dataset, range(min(10, expected_count)))
        expected_count = len(dataset)
        print(f"Sample mode: {expected_count} images")
    else:
        print(f"Processing all {expected_count} images")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    audit_rows = []
    summary_csv = OUTPUT_DIR / "task3_reports.csv"

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "findings", "size_category",
                         "border_category", "checks_passed"])

        with torch.no_grad():
            for index, batch in enumerate(loader):
                report, text, checks = process_image(
                    batch, task1_model, task2_model, device,
                    pixel_thresholds, model_version)

                image_id = report["image_id"]
                (json_dir / f"{image_id}.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8")

                writer.writerow([
                    image_id, text,
                    report["outputs"]["lesion"]["size_category"],
                    report["outputs"]["lesion"]["border_category"],
                    checks["passed"],
                ])

                audit_rows.append({"image_id": image_id,
                                   "text": text,
                                   "checks": checks})

                if sample_mode:
                    print(f"  {image_id}: {text}")
                elif (index + 1) % 50 == 0:
                    print(f"  {index + 1}/{expected_count}")

    audit = summarise_audit(audit_rows, expected_count)
    (OUTPUT_DIR / "task3_audit.json").write_text(
        json.dumps({"summary": audit, "per_image": audit_rows}, indent=2),
        encoding="utf-8")

    report_distribution(audit_rows, audit)
    print(f"\nOutputs in {OUTPUT_DIR}")


def report_distribution(audit_rows, audit) -> None:
    """Print the audit result and how the categories came out."""
    print("\n" + "=" * 62)
    print("CONSISTENCY AUDIT")
    print("=" * 62)
    print(f"Reports generated        : {audit['reports_generated']}"
          f" / {audit['expected_reports']}")
    print(f"All images covered       : {audit['all_images_covered']}")
    print(f"No duplicate image ids   : {audit['no_duplicate_ids']}")
    print(f"Fully passing reports    : {audit['reports_fully_passing']} "
          f"({100 * audit['pass_rate']:.1f}%)")
    print("-" * 62)
    for field in ("failed_all_terms_present", "failed_statuses_match_json",
                  "failed_evidence_supports_claims",
                  "failed_probabilities_in_range"):
        print(f"{field:<34}: {audit[field]}")

    # A rubric that assigns every image the same label is not
    # discriminating, and is worth catching before the write-up.
    sizes, borders, statuses = {}, {}, {}
    for row in audit_rows:
        text = row["text"]
        size = text.split("The lesion is ")[1].split(" with")[0]
        border = text.split("with ")[1].split(" borders")[0]
        sizes[size] = sizes.get(size, 0) + 1
        borders[border] = borders.get(border, 0) + 1

    total = max(len(audit_rows), 1)
    print("-" * 62)
    print("Category distribution:")
    for label, counts in (("size", sizes), ("border", borders)):
        parts = ", ".join(f"{key} {100 * value / total:.0f}%"
                          for key, value in sorted(counts.items()))
        print(f"  {label:<8}: {parts}")
    print("=" * 62)


if __name__ == "__main__":
    print("=" * 62)
    print("TASK 3 — ANCHORED FINDINGS REPORT")
    print("=" * 62)
    print("1. Full validation fold")
    print("2. Sample of 10 images")

    while True:
        choice = input("Select an option (1 or 2): ").strip()
        if choice == "1":
            generate(sample_mode=False)
            break
        if choice == "2":
            generate(sample_mode=True)
            break
        print("Enter 1 or 2.")
