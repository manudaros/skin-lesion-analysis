"""
Step 20 — Task 3 anchored findings report.

Uses the step 18b model, which was trained on 512px crops of the predicted
lesion. Task 3 works in full-image coordinates, so this file translates:

  1. Crop each validation image to its cached Task 1 lesion box, resize to
     512, run the 18b model with test-time augmentation.
  2. Soft-gate, threshold and component-filter exactly as 18b did.
  3. Map the attribute masks back to full-image coordinates.
  4. Apply the rubric and write the reports.

Steps 1-3 are cached to task2_cache/. They cost 15-20 minutes; the report
takes seconds, and you will run it repeatedly while checking wording and
distributions. Set RECOMPUTE = True to redo the inference.

None of the preprocessing is reimplemented — the dataset, gating,
component filter and TTA are all imported from 18b, so the inputs match
the run that produced the thresholds.

Two variants of 18b exist in this project, with different function names,
checkpoint filenames and threshold JSON keys. Section 1 resolves all
three and prints what it found, so this file works with either. If you
only ever have one, the aliases are harmless.

Two consequences worth knowing.

Geometry is measured at ORIGINAL image resolution, because that is what
the cached Task 1 masks are stored at. This is closer to the briefing's
lesion_area_ratio, and the staircase inflation of the border perimeter is
smaller at full resolution — so some images will change border category
compared with the 384px version.

The rubric fixes "present" at 0.60, while 18b's tuning put the best
operating point between 0.25 and 0.50. The rubric is specified, so it is
applied as given, and the audit reports what the difference costs.

    python step20_task3_report.py
"""

from __future__ import annotations

import csv
import importlib
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from step12_data_augmentation import LesionDataset
from step17_task2_training import (
    ATTRIBUTES,
    NUM_ATTRIBUTES,
    Task2ModelConfig,
    build_task2_model,
    select_device,
)
from step19_task3_rubric import (
    STATUS_PRESENT_AT,
    attribute_status,
    border_category,
    build_report_text,
    check_report,
    lesion_geometry,
    size_category,
    summarise_audit,
)


# =====================================================================
# 1. Resolving the step 18b module
# =====================================================================
# Everything below is called positionally, and both known variants take
# their arguments in the same order, so only the names differ.

TASK2_MODULE_CANDIDATES = (
    "step18b_train_better",
    "step18b_task2_highres",
    "step18b",
)

_ALIASES = {
    "CFG": ("CFG", "CONFIG", "cfg"),
    "CropDataset": ("PredictedCropDataset", "PredictedLesionCropDataset"),
    "apply_operational": ("apply_operational",
                          "apply_operational_adjustments"),
    "expand_box": ("expand_box", "expanded_crop_box"),
    "filter_components": ("filter_components", "filter_small_components"),
    "predict": ("predict", "predict_probabilities"),
    "unpad_and_restore": ("unpad_and_restore",
                          "restore_padded_mask_to_original"),
}


def load_task2_module():
    """Import whichever 18b variant is present."""
    errors = []
    for name in TASK2_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(name)
            print(f"Task 2 training module: {name}")
            return module
        except ImportError as error:
            errors.append(f"{name}: {error}")

    raise ImportError(
        "Could not import a step 18b module. Tried:\n  "
        + "\n  ".join(errors)
        + "\nAdd your filename (without .py) to TASK2_MODULE_CANDIDATES."
    )


def resolve(module, key: str):
    for candidate in _ALIASES[key]:
        if hasattr(module, candidate):
            return getattr(module, candidate)

    available = sorted(n for n in dir(module) if not n.startswith("_"))
    raise ImportError(
        f"None of {_ALIASES[key]} found in {module.__name__}.\n"
        f"Available names: {available}"
    )


_T2 = load_task2_module()
CFG = resolve(_T2, "CFG")
CropDataset = resolve(_T2, "CropDataset")
apply_operational = resolve(_T2, "apply_operational")
expand_box = resolve(_T2, "expand_box")
filter_components = resolve(_T2, "filter_components")
predict = resolve(_T2, "predict")
unpad_and_restore = resolve(_T2, "unpad_and_restore")


# =====================================================================
# 2. Settings
# =====================================================================

FOLD = CFG.fold
SPLIT_NAME = "val"
BATCH_SIZE = 4
SAMPLE_SIZE = 10
RECOMPUTE = False              # True forces the inference to run again

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task3_reports"
CACHE_DIR = OUTPUT_DIR / "task2_cache"
CACHE_JSON = CACHE_DIR / "presence.json"
CACHE_MASKS = CACHE_DIR / "masks"

# Filenames differ between the two 18b variants.
SEGMENTATION_CHECKPOINTS = ("task2_best_segmentation.pth",
                            "task2_best_segmentation_model.pth")
BALANCED_CHECKPOINTS = ("task2_best_balanced.pth",
                        "task2_best_balanced_model.pth")
TASK1_CACHE_NAMES = ("task1_roi_cache.json",
                     "task1_predicted_roi_cache.json")
PIXEL_THRESHOLD_KEYS = ("operational_pixel_thresholds",
                        "pixel_thresholds", "raw_pixel_thresholds")


def first_existing(names, description: str) -> Path:
    for name in names:
        path = CFG.output_dir / name
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"No {description} in {CFG.output_dir}. Looked for: "
        + ", ".join(names) + ". Has step 18b finished?"
    )


# =====================================================================
# 3. Mask packing
# =====================================================================
# Five full-resolution masks per image across 540 images. Packing them
# into the bit-planes of one uint8 PNG keeps the size manageable and keeps
# the channels together so they cannot be reordered by accident.

def pack_masks(masks: list[np.ndarray]) -> np.ndarray:
    packed = np.zeros(masks[0].shape, dtype=np.uint8)
    for index, mask in enumerate(masks):
        packed |= (mask.astype(np.uint8) << index)
    return packed


def unpack_masks(packed: np.ndarray) -> np.ndarray:
    return np.stack([(packed >> index) & 1
                     for index in range(NUM_ATTRIBUTES)]).astype(bool)


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def threshold_tensor(values, device) -> torch.Tensor:
    return torch.tensor([float(v) for v in values], dtype=torch.float32,
                        device=device).view(1, NUM_ATTRIBUTES, 1, 1)


# =====================================================================
# 4. Inputs from step 18b
# =====================================================================

def load_thresholds() -> dict:
    """Tuned thresholds and component-area floors."""
    path = first_existing(("task2_best_thresholds.json",),
                          "threshold file")
    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("attribute_order") != list(ATTRIBUTES):
        raise ValueError("Threshold file attribute order does not match.")

    pixel_block = next((data[key] for key in PIXEL_THRESHOLD_KEYS
                        if key in data), None)
    if pixel_block is None:
        raise KeyError(
            f"No pixel thresholds in {path.name}. Looked for: "
            + ", ".join(PIXEL_THRESHOLD_KEYS)
            + f". Keys present: {sorted(data)}")

    presence_block = data.get("classification_thresholds")
    if presence_block is None:
        raise KeyError(f"No classification_thresholds in {path.name}.")

    minimum_block = data.get("minimum_component_areas", {})

    return {
        "pixel": [float(pixel_block[name]) for name in ATTRIBUTES],
        "presence": [float(presence_block[name]) for name in ATTRIBUTES],
        "minimum_areas": [int(minimum_block.get(name, 1))
                          for name in ATTRIBUTES],
    }


def load_task1_cache() -> tuple[dict, dict]:
    """
    Step 18b's predicted lesion boxes and masks.

    Required, never rebuilt. If the boxes differed from the ones the model
    was validated against, every mask would paste to the wrong place.
    """
    path = first_existing(TASK1_CACHE_NAMES, "Task 1 crop cache")
    cached = json.loads(path.read_text(encoding="utf-8"))
    return ({k: list(map(int, v)) for k, v in cached["boxes"].items()},
            {k: str(v) for k, v in cached["mask_files"].items()})


def state_dict_from(path: Path, device) -> dict:
    """Unwrap whichever checkpoint layout 18b wrote."""
    try:
        checkpoint = torch.load(path, map_location=device,
                                weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location=device,
                                weights_only=False)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path.name} is not a dictionary.")

    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(checkpoint.get(key), dict):
            return checkpoint[key]

    if all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise KeyError(f"No state_dict found in {path.name}.")


def load_model(path: Path, device) -> torch.nn.Module:
    model = build_task2_model(Task2ModelConfig(
        use_classification_head=True, auto_locate_task1=False)).to(device)
    model.load_state_dict(state_dict_from(path, device), strict=True)
    model.eval()
    return model


# =====================================================================
# 5. Crop geometry
# =====================================================================

def original_size(mask_files: dict, image_id: str) -> tuple[int, int]:
    """
    Original (height, width) from the cached Task 1 mask header.

    Cheaper than decoding the source JPEG, and the cached mask is stored
    at original resolution by construction.
    """
    with Image.open(CFG.output_dir / mask_files[image_id]) as handle:
        width, height = handle.size
    return height, width


def crop_box_for(boxes: dict, mask_files: dict, image_id: str):
    """
    Recompute the crop the validation dataset used.

    Deterministic for validation — fixed margin, no jitter. Calls 18b's own
    expand_box, because a mismatch here would paste every mask wrongly.
    """
    height, width = original_size(mask_files, image_id)
    tight = tuple(int(v) for v in boxes.get(image_id,
                                            [0, 0, width, height]))
    box = expand_box(tight, height, width, CFG.val_crop_margin, 0.0)
    return box, height, width


def paste_to_full(mask_512: np.ndarray, box, height: int,
                  width: int) -> np.ndarray:
    """Undo the pad and resize, then place the mask back in the image."""
    x1, y1, x2, y2 = box
    restored = unpad_and_restore(mask_512, y2 - y1, x2 - x1,
                                 CFG.image_size)
    canvas = np.zeros((height, width), dtype=bool)
    canvas[y1:y2, x1:x2] = restored
    return canvas


# =====================================================================
# 6. Inference
# =====================================================================

def cache_is_valid(expected: int) -> bool:
    if RECOMPUTE or not CACHE_JSON.is_file():
        return False

    data = json.loads(CACHE_JSON.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    return (metadata.get("attribute_order") == list(ATTRIBUTES)
            and metadata.get("source_run") == str(CFG.output_dir)
            and metadata.get("crop_image_size") == CFG.image_size
            and metadata.get("count") == expected
            and all((CACHE_MASKS / f"{i}.png").is_file()
                    for i in data["images"]))


def run_inference(thresholds: dict, boxes: dict, mask_files: dict) -> dict:
    """
    Run 18b over the validation fold and cache the results.

    Two checkpoints, matching how 18b tuned its two threshold sets: the
    segmentation model for masks, the balanced model for presence.
    """
    device = select_device()
    segmentation_path = first_existing(SEGMENTATION_CHECKPOINTS,
                                       "segmentation checkpoint")
    balanced_path = first_existing(BALANCED_CHECKPOINTS,
                                   "balanced checkpoint")

    print(f"Running Task 2 inference on {device}")
    print(f"  masks    : {segmentation_path.name}")
    print(f"  presence : {balanced_path.name}")

    segmentation_model = load_model(segmentation_path, device)
    balanced_model = load_model(balanced_path, device)

    raw_val = LesionDataset(fold=FOLD, role="val", transform=None,
                            include_task2=True)
    dataset = CropDataset(raw_val, boxes, mask_files, CFG, False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0)

    CACHE_MASKS.mkdir(parents=True, exist_ok=True)
    pixel_tensor = threshold_tensor(thresholds["pixel"], device)
    records: dict[str, dict] = {}
    processed = 0

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            roi = batch["task1_predicted_roi"].to(device)
            image_ids = [str(i) for i in batch["image_id"]]

            segmentation, gating = predict(segmentation_model, images,
                                           CFG.use_final_tta)
            _, presence = predict(balanced_model, images,
                                  CFG.use_final_tta)

            adjusted = apply_operational(segmentation, gating, roi, CFG)
            predictions = filter_components(adjusted >= pixel_tensor,
                                            thresholds["minimum_areas"])

            crop_pixels = predictions.sum(dim=(2, 3)).cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            presence_np = (presence.cpu().numpy() if presence is not None
                           else np.full((len(image_ids), NUM_ATTRIBUTES),
                                        float("nan")))

            for position, image_id in enumerate(image_ids):
                box, height, width = crop_box_for(boxes, mask_files,
                                                  image_id)
                full = [paste_to_full(predictions_np[position, channel],
                                      box, height, width)
                        for channel in range(NUM_ATTRIBUTES)]

                Image.fromarray(pack_masks(full)).save(
                    CACHE_MASKS / f"{image_id}.png")

                records[image_id] = {
                    "crop_box": list(map(int, box)),
                    "original_size": [int(height), int(width)],
                    "attributes": {
                        name: {
                            "prob": round(float(presence_np[position, c]), 6),
                            "presence_threshold": thresholds["presence"][c],
                            "pixel_threshold": thresholds["pixel"][c],
                            "mask_pixels_crop": int(crop_pixels[position, c]),
                            "mask_pixels_full": int(full[c].sum()),
                        } for c, name in enumerate(ATTRIBUTES)
                    },
                }

                processed += 1
                if processed % 50 == 0:
                    print(f"  {processed}/{len(dataset)}")

    payload = {
        "metadata": {
            "attribute_order": list(ATTRIBUTES),
            "source_run": str(CFG.output_dir),
            "task2_module": _T2.__name__,
            "segmentation_checkpoint": segmentation_path.name,
            "presence_checkpoint": balanced_path.name,
            "crop_image_size": CFG.image_size,
            "val_crop_margin": CFG.val_crop_margin,
            "test_time_augmentation": CFG.use_final_tta,
            "masks_in": "original image resolution, bit-packed uint8 PNG",
            "count": processed,
        },
        "images": records,
    }
    CACHE_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_inference_summary(records)
    return payload


def print_inference_summary(records: dict) -> None:
    """Sanity figures. Check these before trusting any report."""
    print("\n" + "=" * 68)
    print(f"{'attribute':<20}{'mean prob':>11}{'above thr':>11}"
          f"{'has mask':>10}{'mean px':>12}")

    for name in ATTRIBUTES:
        probabilities = np.array([r["attributes"][name]["prob"]
                                  for r in records.values()])
        threshold = next(iter(records.values()))["attributes"][name][
            "presence_threshold"]
        pixels = np.array([r["attributes"][name]["mask_pixels_full"]
                           for r in records.values()])

        print(f"{name:<20}{np.nanmean(probabilities):>11.4f}"
              f"{100 * np.mean(probabilities >= threshold):>10.1f}%"
              f"{100 * np.mean(pixels > 0):>9.1f}%"
              f"{pixels.mean():>12.0f}")

    print("=" * 68)
    print("'above thr' should sit near the true presence rates:")
    print("  pigment 61%, negative 7%, streaks 5%, milia 21%, globules 22%")


def load_or_run_inference() -> dict:
    thresholds = load_thresholds()
    boxes, mask_files = load_task1_cache()

    expected = len(LesionDataset(fold=FOLD, role="val", transform=None,
                                 include_task2=False))

    if cache_is_valid(expected):
        print(f"Using cached Task 2 inference ({CACHE_JSON.name}). "
              "Set RECOMPUTE = True to redo it.")
        return json.loads(CACHE_JSON.read_text(encoding="utf-8"))

    return run_inference(thresholds, boxes, mask_files)


# =====================================================================
# 7. One report
# =====================================================================

def build_report(image_id: str, record: dict, task1_mask: np.ndarray,
                 attribute_masks: np.ndarray, metadata: dict):
    geometry = lesion_geometry(task1_mask)
    size, area_ratio = size_category(geometry["lesion_pixels"],
                                     task1_mask.size)
    border, _ = border_category(geometry["component_area"],
                                geometry["perimeter_smoothed"])

    presence, statuses, evidence = {}, {}, {}

    for index, name in enumerate(ATTRIBUTES):
        entry = record["attributes"][name]
        probability = float(entry["prob"])
        statuses[name] = attribute_status(probability)

        # Evidence for the consistency check: predicted attribute pixels
        # falling inside the predicted lesion.
        inside = int(np.logical_and(attribute_masks[index],
                                    task1_mask).sum())
        evidence[name] = inside

        presence[name] = {
            "prob": round(probability, 4),
            "status": statuses[name],
            "evidence_pixels_in_roi": inside,
            "mask_pixels_full": int(attribute_masks[index].sum()),
            "tuned_presence_threshold": entry["presence_threshold"],
            "pixel_threshold": entry["pixel_threshold"],
        }

    report = {
        "image_id": image_id,
        "split": SPLIT_NAME,
        "model_version": (f"{metadata['segmentation_checkpoint']}, "
                          f"{metadata['presence_checkpoint']}"),
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
        "provenance": {
            "presence_source": "step 18b classification head",
            "trained_on": f"{metadata['crop_image_size']}px lesion crops",
            "geometry_measured_at": "original image resolution",
            "test_time_augmentation": metadata["test_time_augmentation"],
        },
    }

    text = build_report_text(size, border, statuses)
    checks = check_report(report, text, evidence)
    return report, text, checks


# =====================================================================
# 8. Main
# =====================================================================

def generate(sample_mode: bool = False) -> None:
    payload = load_or_run_inference()
    metadata, images = payload["metadata"], payload["images"]
    _, mask_files = load_task1_cache()

    json_dir = OUTPUT_DIR / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    image_ids = sorted(images)
    if sample_mode:
        image_ids = image_ids[:SAMPLE_SIZE]
        print(f"\nSample mode: {len(image_ids)} reports")
    else:
        print(f"\nGenerating {len(image_ids)} reports")

    audit_rows = []
    summary_csv = OUTPUT_DIR / "task3_reports.csv"

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "findings", "size_category",
                         "border_category", "checks_passed"])

        for position, image_id in enumerate(image_ids):
            task1_mask = read_mask(CFG.output_dir / mask_files[image_id])
            packed = np.asarray(Image.open(CACHE_MASKS / f"{image_id}.png"))
            attribute_masks = unpack_masks(packed)

            if attribute_masks.shape[1:] != task1_mask.shape:
                raise ValueError(
                    f"{image_id}: attribute masks "
                    f"{attribute_masks.shape[1:]} do not match the lesion "
                    f"mask {task1_mask.shape}")

            report, text, checks = build_report(
                image_id, images[image_id], task1_mask, attribute_masks,
                metadata)

            (json_dir / f"{image_id}.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")

            lesion = report["outputs"]["lesion"]
            writer.writerow([image_id, text, lesion["size_category"],
                             lesion["border_category"], checks["passed"]])
            audit_rows.append({"image_id": image_id, "text": text,
                               "checks": checks})

            if sample_mode:
                print(f"  {image_id}: {text}")
            elif (position + 1) % 100 == 0:
                print(f"  {position + 1}/{len(image_ids)}")

    audit = summarise_audit(audit_rows, len(image_ids))
    calibration = threshold_calibration(images)

    (OUTPUT_DIR / "task3_audit.json").write_text(json.dumps({
        "summary": audit,
        "threshold_calibration": calibration,
        "per_image": audit_rows,
    }, indent=2), encoding="utf-8")

    print_audit(audit, audit_rows, calibration)
    print(f"\nOutputs in {OUTPUT_DIR}")


def threshold_calibration(images: dict) -> dict:
    """
    What the fixed rubric threshold costs against the tuned one.

    The briefing requires present at >= 0.60. 18b's tuning found a lower
    optimum for every attribute. This quantifies the gap rather than
    leaving it as an assertion.
    """
    calibration = {}
    for name in ATTRIBUTES:
        probabilities = np.array([r["attributes"][name]["prob"]
                                  for r in images.values()])
        tuned = next(iter(images.values()))["attributes"][name][
            "presence_threshold"]

        calibration[name] = {
            "rubric_threshold": STATUS_PRESENT_AT,
            "tuned_threshold": tuned,
            "present_under_rubric": int(np.sum(
                probabilities >= STATUS_PRESENT_AT)),
            "present_under_tuned": int(np.sum(probabilities >= tuned)),
            "mean_probability": float(np.mean(probabilities)),
        }
    return calibration


def print_audit(audit: dict, audit_rows: list, calibration: dict) -> None:
    print("\n" + "=" * 68)
    print("CONSISTENCY AUDIT")
    print("=" * 68)
    print(f"Reports generated       : {audit['reports_generated']} / "
          f"{audit['expected_reports']}")
    print(f"No duplicate image ids  : {audit['no_duplicate_ids']}")
    print(f"Fully passing reports   : {audit['reports_fully_passing']} "
          f"({100 * audit['pass_rate']:.1f}%)")
    print("-" * 68)
    for field in ("failed_all_terms_present", "failed_statuses_match_json",
                  "failed_evidence_supports_claims",
                  "failed_probabilities_in_range"):
        print(f"{field:<36}: {audit[field]}")

    sizes: dict[str, int] = {}
    borders: dict[str, int] = {}
    for row in audit_rows:
        size = row["text"].split("The lesion is ")[1].split(" with")[0]
        border = row["text"].split("with ")[1].split(" borders")[0]
        sizes[size] = sizes.get(size, 0) + 1
        borders[border] = borders.get(border, 0) + 1

    total = max(len(audit_rows), 1)
    print("-" * 68)
    print("Category distribution:")
    for label, counts in (("size", sizes), ("border", borders)):
        print(f"  {label:<8}: " + ", ".join(
            f"{k} {100 * v / total:.0f}%"
            for k, v in sorted(counts.items())))

    print("-" * 68)
    print("Threshold calibration — attributes called present:")
    print(f"  {'attribute':<20}{'rubric 0.60':>13}{'tuned':>10}{'thr':>7}")
    for name, entry in calibration.items():
        print(f"  {name:<20}{entry['present_under_rubric']:>13}"
              f"{entry['present_under_tuned']:>10}"
              f"{entry['tuned_threshold']:>7.2f}")
    print("=" * 68)


if __name__ == "__main__":
    print("=" * 68)
    print("TASK 3 — ANCHORED FINDINGS REPORT")
    print("=" * 68)
    print("1. Full validation fold")
    print(f"2. Sample of {SAMPLE_SIZE} images")

    while True:
        choice = input("Select an option (1 or 2): ").strip()
        if choice == "1":
            generate(sample_mode=False)
            break
        if choice == "2":
            generate(sample_mode=True)
            break
        print("Enter 1 or 2.")