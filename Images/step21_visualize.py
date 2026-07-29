"""
Step 21 — end-to-end pipeline visualisation.

One figure per image: the Task 1 segmentation with its error map, the
five Task 2 attribute heatmaps, and the Task 3 report text underneath.
This is the figure for the write-up, so it uses the same rubric module
and the same loaders as step 19 — nothing is recomputed here, which is
what keeps the picture and the reported numbers in agreement.

Every figure is written to disk whether or not a window opens. Step 15
sets matplotlib's backend to Agg at import, and this file imports from
it, so an interactive window needs the backend switched back before the
figure is built. If that isn't possible the PNG is still there.

    python step21_visualize.py
"""

from __future__ import annotations
from step20_task3_report import (
    FOLD,
    IMAGE_SIZE,
    LESION_THRESHOLD,
    load_models,
    load_tuned_pixel_thresholds,
    process_image,
)
from step17_task2_training import ATTRIBUTES, select_device
from step15_train import calculate_metrics_single
from step12_data_augmentation import LesionDataset, build_val_transform
import torch
import numpy as np
import matplotlib

import os
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


OUTPUT_DIR = (Path(__file__).resolve().parent.parent
              / "outputs" / "task3_reports" / "figures")

# ImageNet statistics, matching the normalisation in step 12. If your
# transform uses different values, change these or the colours will look
# wrong — display only, no metric depends on it.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def enable_interactive_backend() -> bool:
    """
    Try to restore a window-capable backend.

    step15_train sets Agg on import, which silently disables plt.show().
    This must run before any figure is created, since a figure is bound
    to whichever canvas was active when it was made.
    """
    for backend in ("macosx", "TkAgg", "QtAgg"):
        try:
            matplotlib.use(backend, force=True)
            return True
        except Exception:
            continue
    return False


def unnormalise(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(IMAGENET_STD * array + IMAGENET_MEAN, 0, 1)


def error_map(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """White correct, red spurious, blue missed."""
    canvas = np.zeros((*prediction.shape, 3), dtype=np.float32)
    canvas[(prediction == 1) & (truth == 1)] = [1.0, 1.0, 1.0]
    canvas[(prediction == 1) & (truth == 0)] = [1.0, 0.0, 0.0]
    canvas[(prediction == 0) & (truth == 1)] = [0.0, 0.0, 1.0]
    return canvas


def visualise(index, show: bool = True, save_dir: Path = OUTPUT_DIR):
    if show:
        if not enable_interactive_backend():
            print("No interactive backend available — saving only.")
            show = False

    import matplotlib.pyplot as plt      # after the backend is settled

    device = select_device()
    task1_model, task2_model, _, task2_path = load_models(device)
    pixel_thresholds = load_tuned_pixel_thresholds(task2_path)

    dataset = LesionDataset(
        fold=FOLD, role="val",
        transform=build_val_transform(image_size=IMAGE_SIZE),
        include_task2=False)

    if index == "random":
        index = int(np.random.randint(0, len(dataset)))
    if not 0 <= index < len(dataset):
        print(f"Index out of range: 0 to {len(dataset) - 1}")
        return

    sample = dataset[index]
    batch = {
        "image": sample["image"].unsqueeze(0),
        "image_id": [sample["image_id"]],
    }

    with torch.no_grad():
        report, text, checks = process_image(
            batch, task1_model, task2_model, device,
            pixel_thresholds, model_version="visualisation")

        images = batch["image"].to(device)
        lesion_probabilities = torch.sigmoid(task1_model(images))
        prediction = (lesion_probabilities > LESION_THRESHOLD
                      ).squeeze().cpu().numpy().astype(np.uint8)

        attribute_logits, _ = task2_model(images)
        attribute_probabilities = torch.sigmoid(
            attribute_logits).squeeze(0).cpu().numpy()

    truth_tensor = sample["task1_segmentation"]
    truth = (truth_tensor.squeeze().numpy() > 0.5).astype(np.uint8)

    dice, iou, thresholded, _ = calculate_metrics_single(
        lesion_probabilities.cpu(), truth_tensor.unsqueeze(0),
        threshold=LESION_THRESHOLD, compute_hd95=False)

    image_id = report["image_id"]
    rgb = unnormalise(sample["image"])
    lesion = report["outputs"]["lesion"]
    presence = report["outputs"]["presence"]

    # One 20-column grid throughout: row 0 takes 4 panels of 5 columns,
    # row 1 takes 5 panels of 4. Mixing 4- and 5-column grids is what
    # makes tight_layout refuse to run.
    figure = plt.figure(figsize=(18, 11))
    figure.suptitle(f"End-to-end output — {image_id}", fontsize=18)

    axis = plt.subplot2grid((3, 20), (0, 0), colspan=5)
    axis.imshow(rgb)
    axis.set_title("Input image", fontsize=11)
    axis.axis("off")

    axis = plt.subplot2grid((3, 20), (0, 5), colspan=5)
    axis.imshow(truth, cmap="gray", vmin=0, vmax=1)
    axis.set_title("Task 1 ground truth", fontsize=11)
    axis.axis("off")

    axis = plt.subplot2grid((3, 20), (0, 10), colspan=5)
    overlay = rgb.copy()
    overlay[prediction == 1] = (overlay[prediction == 1] * 0.5
                                + np.array([1, 0, 0]) * 0.5)
    axis.imshow(overlay)
    axis.set_title(f"Prediction — Dice {dice:.3f}, IoU {iou:.3f}",
                   fontsize=11)
    axis.axis("off")

    axis = plt.subplot2grid((3, 20), (0, 15), colspan=5)
    axis.imshow(error_map(prediction, truth))
    axis.set_title("Error map (white correct, red extra, blue missed)",
                   fontsize=10)
    axis.axis("off")

    for position, name in enumerate(ATTRIBUTES):
        axis = plt.subplot2grid((3, 20), (1, position * 4), colspan=4)
        heatmap = attribute_probabilities[position].copy()
        heatmap[prediction == 0] = 0        # restrict to the lesion ROI

        axis.imshow(heatmap, cmap="magma", vmin=0, vmax=1)
        entry = presence[name]
        axis.set_title(f"{name.replace('_', ' ')}\n{entry['status']}",
                       fontsize=11)
        axis.text(0.5, -0.10,
                  f"classifier {entry['prob_classifier']:.2f}   "
                  f"ROI {entry['prob_roi_logit_mean']:.2f}",
                  fontsize=9, ha="center", transform=axis.transAxes)
        axis.axis("off")

    axis = plt.subplot2grid((3, 20), (2, 0), colspan=20)
    axis.axis("off")
    caption = (
        f"{text}\n\n"
        f"area ratio {lesion['area_ratio']:.3f}  |  "
        f"border index {lesion['border_index']:.2f} "
        f"(raw {lesion['border_index_raw']:.2f})  |  "
        f"thresholded Jaccard {thresholded:.3f}  |  "
        f"consistency checks {'passed' if checks['passed'] else 'FAILED'}"
    )
    axis.text(0.5, 0.55, caption, fontsize=13, ha="center", va="center",
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


if __name__ == "__main__":
    print("=" * 62)
    print("STEP 21 — PIPELINE VISUALISATION")
    print("=" * 62)
    print("1. Random image")
    print("2. Specific image by index")
    print("3. Save five random figures without displaying")

    while True:
        choice = input("Select an option (1, 2 or 3): ").strip()

        if choice == "1":
            visualise("random")
            break
        if choice == "2":
            try:
                visualise(int(input("Image index: ").strip()))
                break
            except ValueError:
                print("Enter a whole number.")
        elif choice == "3":
            for _ in range(5):
                visualise("random", show=False)
            break
        else:
            print("Enter 1, 2 or 3.")
