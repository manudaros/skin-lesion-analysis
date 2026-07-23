# This file visually checks whether Task 1 lesion masks align correctly
# with the corresponding lesion regions in the original images.

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Locate the project root by searching for the data directory.
here = Path(__file__).resolve().parent
PROJECT = None

for candidate in [
    here,
    *here.parents,
]:
    if (
        candidate
        / "data"
    ).is_dir():
        PROJECT = candidate
        break

if PROJECT is None:
    raise FileNotFoundError(
        "Could not locate the project root. "
        "No parent directory containing "
        "a 'data' folder was found."
    )


# Define the Task 1 image and mask directories.
DATA = (
    PROJECT
    / "data"
)

T1_IMG = (
    DATA
    / "Task1_Segmentation"
    / "images"
)

T1_MASK = (
    DATA
    / "Task1_Segmentation"
    / "masks"
)


# Confirm that the required directories exist.
if not T1_IMG.is_dir():
    raise FileNotFoundError(
        "Task 1 image directory "
        f"does not exist: {T1_IMG}"
    )

if not T1_MASK.is_dir():
    raise FileNotFoundError(
        "Task 1 mask directory "
        f"does not exist: {T1_MASK}"
    )


# Collect all Task 1 image IDs from JPG filenames.
ids = sorted(
    image_path.stem
    for image_path
    in T1_IMG.glob("*.jpg")
)

if not ids:
    raise FileNotFoundError(
        "No JPG images were found in: "
        f"{T1_IMG}"
    )


# Use a fixed random seed so the same samples are selected every time.
random.seed(0)

sample_count = min(
    6,
    len(ids),
)

sample = random.sample(
    ids,
    sample_count,
)


# Create one row for original images and one row for overlays.
fig, axes = plt.subplots(
    2,
    sample_count,
    figsize=(
        2.7 * sample_count,
        6,
    ),
    squeeze=False,
)


for (
    col,
    img_id,
) in enumerate(sample):
    image_path = (
        T1_IMG
        / f"{img_id}.jpg"
    )

    mask_path = (
        T1_MASK
        / f"{img_id}_segmentation.png"
    )

    if not mask_path.exists():
        raise FileNotFoundError(
            "No corresponding Task 1 mask "
            f"was found for {img_id}: "
            f"{mask_path}"
        )

    # Open the image and mask using context managers.
    with Image.open(
        image_path
    ) as image_file:
        img = image_file.convert(
            "RGB"
        )

    with Image.open(
        mask_path
    ) as mask_file:
        mask = mask_file.convert(
            "L"
        )

    # Check the original dimensions before resizing.
    # Resizing both files independently could otherwise hide a mismatch.
    if img.size != mask.size:
        raise ValueError(
            "Image-mask size mismatch "
            f"for {img_id}: "
            f"image size={img.size}, "
            f"mask size={mask.size}"
        )

    # Resize both files to a common display size.
    # Bilinear interpolation is suitable for the RGB image.
    # Nearest-neighbor interpolation preserves binary mask values.
    img = img.resize(
        (256, 256),
        Image.Resampling.BILINEAR,
    )

    mask = mask.resize(
        (256, 256),
        Image.Resampling.NEAREST,
    )

    img_array = np.asarray(
        img
    )

    mask_array = (
        np.asarray(mask)
        > 0
    )

    # Display the original image in the top row.
    axes[0, col].imshow(
        img_array
    )

    axes[0, col].set_title(
        img_id,
        fontsize=9,
    )

    # Create a red semi-transparent wash over the lesion region.
    overlay = img_array.copy()

    overlay[mask_array] = (
        0.5
        * overlay[mask_array]
        + 0.5
        * np.array(
            [255, 0, 0]
        )
    ).astype(
        np.uint8
    )

    axes[1, col].imshow(
        overlay
    )

    # Calculate the fraction of the image covered by the lesion mask.
    coverage = float(
        mask_array.mean()
    )

    axes[1, col].set_title(
        f"Lesion coverage: "
        f"{coverage:.1%}",
        fontsize=9,
    )


# Remove axes for cleaner visual inspection.
for axis in axes.ravel():
    axis.axis("off")


fig.suptitle(
    "Task 1 Image-Mask Alignment Check",
    fontsize=14,
)

plt.tight_layout()
plt.show()