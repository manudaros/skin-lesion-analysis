"""
Step 21 — end-to-end pipeline visualisation.

One figure per image: the Task 1 segmentation with its error map, the five
Task 2 attribute masks against ground truth, and the Task 3 report text
underneath. This is the figure for the write-up.

Reads step 20's cache rather than running any model. The rubric, the
geometry and the presence probabilities all come from step 20's own
functions, so the picture cannot disagree with the numbers in the tables.

Because the cached masks are stored at original image resolution, the Dice
and IoU shown here are full-resolution figures. They are therefore NOT the
same quantity as the crop-based metrics in step 18b's tables. Both are
valid; they measure different things, and the caption says which.

Every figure is written to disk whether or not a window opens: step 18b
sets matplotlib's backend to Agg at import, and this file inherits that
through step 20.

Run step 20 first so the cache exists.

    python step21_visualize.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import matplotlib
import numpy as np
import torch
from PIL import Image

from step12_data_augmentation import LesionDataset
from step15_train import calculate_metrics_single
from step17_task2_training import ATTRIBUTES
from step18b_train_better import CFG
from step20_task3_report import (
    CACHE_MASKS,
    FOLD,
    build_report,
    load_or_run_inference,
    load_task1_cache,
    read_mask,
    unpack_masks,
)

OUTPUT_DIR = (Path(__file__).resolve().parent.parent
              / "outputs" / "task3_reports" / "figures")
AUDIT_PATH = (Path(__file__).resolve().parent.parent / "outputs"
              / "task3_reports" / "task3_audit.json")

# Original ISIC images are around 1000px. Downscaling for display keeps
# rendering fast; nothing measured passes through it.
DISPLAY_MAX_SIDE = 700


# =====================================================================
# 1. Setup
# =====================================================================

def enable_interactive_backend() -> bool:
    """
    Restore a window-capable backend if there is one.

    step18b sets Agg on import, which silently disables plt.show(). This
    must run before any figure is created, since a figure binds to
    whichever canvas was active when it was made.
    """
    for backend in ("macosx", "TkAgg", "QtAgg"):
        try:
            matplotlib.use(backend, force=True)
            return True
        except Exception:
            continue
    return False


def build_id_index(dataset: LesionDataset) -> dict[str, int]:
    """Map image_id to position, so a figure can be requested by id."""
    return {str(row["image_id"]): index
            for index, row in enumerate(dataset.data)}


def to_display_rgb(tensor: torch.Tensor) -> np.ndarray:
    """
    The raw dataset image as a viewable array.

    With transform=None the image is unnormalised, so this is a dtype
    conversion rather than an inverse normalisation — nothing to undo.
    """
    array = torch.as_tensor(tensor).detach().cpu()
    if array.dtype == torch.uint8:
        array = array.float() / 255.0
    return np.clip(array.permute(1, 2, 0).numpy(), 0, 1)


def downscale(array: np.ndarray, nearest: bool = False) -> np.ndarray:
    """Shrink for display only."""
    height, width = array.shape[:2]
    longest = max(height, width)
    if longest <= DISPLAY_MAX_SIDE:
        return array

    scale = DISPLAY_MAX_SIDE / longest
    size = (max(1, int(width * scale)), max(1, int(height * scale)))

    if array.dtype == bool:
        resized = Image.fromarray(array.astype(np.uint8) * 255).resize(
            size, Image.NEAREST if nearest else Image.BILINEAR)
        return np.asarray(resized) > 127

    resized = Image.fromarray((array * 255).astype(np.uint8)).resize(
        size, Image.NEAREST if nearest else Image.BILINEAR)
    return np.asarray(resized).astype(np.float32) / 255.0


# =====================================================================
# 2. Panels
# =====================================================================

def error_map(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """White correct, red spurious, blue missed."""
    canvas = np.zeros((*prediction.shape, 3), dtype=np.float32)
    canvas[prediction & truth] = [1.0, 1.0, 1.0]
    canvas[prediction & ~truth] = [1.0, 0.0, 0.0]
    canvas[~prediction & truth] = [0.0, 0.0, 1.0]
    return canvas


def attribute_overlay(prediction: np.ndarray, truth: np.ndarray,
                      lesion: np.ndarray) -> np.ndarray:
    """
    Red prediction, green ground truth, yellow where they agree.

    The lesion sits underneath in dark grey, so a mask predicted outside
    the lesion is immediately obvious — which is the main thing that could
    go wrong in the coordinate mapping.
    """
    canvas = np.zeros((*prediction.shape, 3), dtype=np.float32)
    canvas[lesion] = [0.16, 0.16, 0.16]
    canvas[truth] = [0.0, 1.0, 0.0]
    canvas[prediction] = [1.0, 0.0, 0.0]
    canvas[prediction & truth] = [1.0, 1.0, 0.0]
    return canvas


# =====================================================================
# 3. One figure
# =====================================================================

def resolve_selector(selector, cached_ids: list[str]) -> str | None:
    """Accept 'random', an integer position, or an image id."""
    if selector == "random":
        return str(np.random.choice(cached_ids))

    if isinstance(selector, int):
        if not 0 <= selector < len(cached_ids):
            print(f"Index out of range: 0 to {len(cached_ids) - 1}")
            return None
        return cached_ids[selector]

    if str(selector) in cached_ids:
        return str(selector)

    print(f"No cached inference for '{selector}'.")
    return None


def visualise(selector, show: bool = True,
              save_dir: Path = OUTPUT_DIR) -> None:
    if show and not enable_interactive_backend():
        print("No interactive backend available — saving only.")
        show = False

    import matplotlib.pyplot as plt          # after the backend is settled

    payload = load_or_run_inference()
    metadata, images = payload["metadata"], payload["images"]
    _, mask_files = load_task1_cache()

    dataset = LesionDataset(fold=FOLD, role="val", transform=None,
                            include_task2=True)
    id_index = build_id_index(dataset)

    image_id = resolve_selector(selector, sorted(images))
    if image_id is None:
        return
    if image_id not in id_index:
        print(f"{image_id} is not in the validation fold.")
        return

    sample = dataset[id_index[image_id]]

    # --- masks, all at original resolution ---------------------------
    lesion_prediction = read_mask(CFG.output_dir / mask_files[image_id])
    attribute_predictions = unpack_masks(
        np.asarray(Image.open(CACHE_MASKS / f"{image_id}.png")))

    lesion_truth = (torch.as_tensor(sample["task1_segmentation"])
                    .squeeze().numpy() > 0.5)
    attribute_truth = (torch.as_tensor(sample["task2_attributes"]).numpy()
                       > 0.5)

    if attribute_predictions.shape != attribute_truth.shape:
        raise ValueError(
            f"{image_id}: predicted masks {attribute_predictions.shape} do "
            f"not match ground truth {attribute_truth.shape}")

    # --- report, from step 20's own code -----------------------------
    report, text, checks = build_report(
        image_id, images[image_id], lesion_prediction,
        attribute_predictions, metadata)

    # Step 15's metric, so the number matches the Task 1 tables.
    dice, iou, thresholded, _ = calculate_metrics_single(
        torch.from_numpy(lesion_prediction.astype(np.float32))[None, None],
        torch.from_numpy(lesion_truth.astype(np.float32))[None, None],
        threshold=0.5, compute_hd95=False)

    lesion = report["outputs"]["lesion"]
    presence = report["outputs"]["presence"]

    # --- figure ------------------------------------------------------
    rgb = downscale(to_display_rgb(sample["image"]))
    small_prediction = downscale(lesion_prediction, nearest=True)
    small_truth = downscale(lesion_truth, nearest=True)

    figure = plt.figure(figsize=(18, 11))
    figure.suptitle(f"End-to-end output — {image_id}", fontsize=18)

    axis = plt.subplot2grid((3, 20), (0, 0), colspan=5)
    axis.imshow(rgb)
    axis.set_title("Input image", fontsize=11)
    axis.axis("off")

    axis = plt.subplot2grid((3, 20), (0, 5), colspan=5)
    axis.imshow(small_truth, cmap="gray", vmin=0, vmax=1)
    axis.set_title("Task 1 ground truth", fontsize=11)
    axis.axis("off")

    axis = plt.subplot2grid((3, 20), (0, 10), colspan=5)
    overlay = rgb.copy()
    overlay[small_prediction] = (overlay[small_prediction] * 0.5
                                 + np.array([1.0, 0.0, 0.0]) * 0.5)
    axis.imshow(overlay)
    axis.set_title(f"Prediction — Dice {dice:.3f}, IoU {iou:.3f}",
                   fontsize=11)
    axis.axis("off")

    axis = plt.subplot2grid((3, 20), (0, 15), colspan=5)
    axis.imshow(error_map(small_prediction, small_truth))
    axis.set_title("Error map (white correct, red extra, blue missed)",
                   fontsize=10)
    axis.axis("off")

    for position, name in enumerate(ATTRIBUTES):
        axis = plt.subplot2grid((3, 20), (1, position * 4), colspan=4)
        axis.imshow(attribute_overlay(
            downscale(attribute_predictions[position], nearest=True),
            downscale(attribute_truth[position], nearest=True),
            small_prediction))

        entry = presence[name]
        axis.set_title(f"{name.replace('_', ' ')}\n{entry['status']}",
                       fontsize=11)
        axis.text(0.5, -0.10,
                  f"p {entry['prob']:.2f}   "
                  f"{entry['evidence_pixels_in_roi']} px in lesion",
                  fontsize=9, ha="center", transform=axis.transAxes)
        axis.axis("off")

    axis = plt.subplot2grid((3, 20), (2, 0), colspan=20)
    axis.axis("off")
    caption = (
        f"{text}\n\n"
        f"area ratio {lesion['area_ratio']:.3f}  |  "
        f"border index {lesion['border_index']:.2f} "
        f"(raw {lesion['border_index_raw']:.2f})  |  "
        f"Task 1 thresholded Jaccard {thresholded:.3f}  |  "
        f"consistency checks "
        f"{'passed' if checks['passed'] else 'FAILED'}\n"
        "Attribute panels: red predicted, green ground truth, yellow both, "
        "grey lesion. Metrics at original resolution."
    )
    axis.text(0.5, 0.55, caption, fontsize=12, ha="center", va="center",
              wrap=True,
              bbox=dict(boxstyle="round,pad=1.2", fc="#f6f6f4",
                        ec="#d8d8d2", lw=1.5))

    plt.tight_layout(pad=2.5)

    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"pipeline_{image_id}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")

    if show:
        plt.show()
    plt.close(figure)


# =====================================================================
# 4. Worst cases
# =====================================================================

def save_worst_cases(count: int = 5) -> None:
    """
    Figures for reports whose consistency checks failed, then the
    least-confident ones. A page of successes says less in a report than
    one honestly presented failure.
    """
    payload = load_or_run_inference()
    images = payload["images"]

    chosen: list[str] = []
    if AUDIT_PATH.is_file():
        audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        failed = [row["image_id"] for row in audit.get("per_image", [])
                  if not row["checks"]["passed"]]
        print(f"{len(failed)} reports failed their consistency checks")
        chosen = failed[:count]
    else:
        print(f"No audit at {AUDIT_PATH.name} — run step 20 first for "
              "failure-ranked figures. Falling back to least confident.")

    if len(chosen) < count:
        mean_probability = {
            image_id: float(np.mean([entry["prob"] for entry
                                     in record["attributes"].values()]))
            for image_id, record in images.items()}
        for image_id in sorted(mean_probability,
                               key=mean_probability.get):
            if image_id not in chosen:
                chosen.append(image_id)
            if len(chosen) == count:
                break

    for image_id in chosen:
        visualise(image_id, show=False)


if __name__ == "__main__":
    print("=" * 68)
    print("STEP 21 — PIPELINE VISUALISATION")
    print("=" * 68)
    print("1. Random image")
    print("2. Specific image by id or index")
    print("3. Save five random figures without displaying")
    print("4. Save five worst cases (failed checks first)")

    while True:
        choice = input("Select an option (1-4): ").strip()

        if choice == "1":
            visualise("random")
            break
        if choice == "2":
            entry = input("Image id or index: ").strip()
            visualise(int(entry) if entry.isdigit() else entry)
            break
        if choice == "3":
            for _ in range(5):
                visualise("random", show=False)
            break
        if choice == "4":
            save_worst_cases(5)
            break
        print("Enter 1, 2, 3 or 4.")