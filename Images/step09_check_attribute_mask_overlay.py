# This file visually checks whether Task 2 attribute masks are stored
# in the correct folders and align with the corresponding images.

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from qc_config import (
    ATTRIBUTES,
    RANDOM_SEED,
    TASK1_IMAGE_DIR,
    TASK1_MASK_DIR,
    TASK2_MASK_ROOT,
)
from qc_utils import (
    build_id_map,
    first_path,
    list_image_files,
    load_binary_mask,
)


def find_positive_ids(
    attribute_map: dict[
        str,
        list[Path],
    ],
) -> list[str]:
    """
    Return image IDs whose attribute
    masks contain foreground pixels.
    """
    positive_ids = []

    for sample_id in sorted(
        attribute_map
    ):
        mask_path = first_path(
            attribute_map,
            sample_id,
        )

        if mask_path is None:
            continue

        try:
            mask = load_binary_mask(
                mask_path
            )

            if mask.any():
                positive_ids.append(
                    sample_id
                )

        except Exception:
            continue

    return positive_ids


def load_resized_image(
    image_path: Path,
    size: tuple[
        int,
        int,
    ] = (256, 256),
) -> np.ndarray:
    """Load and resize one RGB image."""
    with Image.open(
        image_path
    ) as image_file:
        image = image_file.convert(
            "RGB"
        )

        image = image.resize(
            size,
            Image.Resampling.BILINEAR,
        )

    return np.asarray(
        image
    )


def load_resized_mask(
    mask_path: Path,
    size: tuple[
        int,
        int,
    ] = (256, 256),
) -> np.ndarray:
    """Load and resize one binary mask."""
    with Image.open(
        mask_path
    ) as mask_file:
        mask = mask_file.convert(
            "L"
        )

        mask = mask.resize(
            size,
            Image.Resampling.NEAREST,
        )

    return (
        np.asarray(mask)
        > 0
    )


def create_combined_overlay(
    image_array: np.ndarray,
    lesion_mask: np.ndarray,
    attribute_mask: np.ndarray,
) -> np.ndarray:
    """
    Add a light green lesion wash
    and a stronger red attribute wash.
    """
    overlay = image_array.copy()

    overlay[lesion_mask] = (
        0.80
        * overlay[lesion_mask]
        + 0.20
        * np.array(
            [0, 255, 0]
        )
    ).astype(
        np.uint8
    )

    overlay[attribute_mask] = (
        0.45
        * overlay[attribute_mask]
        + 0.55
        * np.array(
            [255, 0, 0]
        )
    ).astype(
        np.uint8
    )

    return overlay


def run_attribute_overlay_check() -> None:
    """
    Display one positive example
    for each Task 2 attribute.
    """
    image_map = build_id_map(
        list_image_files(
            TASK1_IMAGE_DIR
        )
    )

    lesion_map = build_id_map(
        list_image_files(
            TASK1_MASK_DIR
        )
    )

    attribute_maps = {
        attribute: build_id_map(
            list_image_files(
                TASK2_MASK_ROOT
                / attribute
            )
        )
        for attribute in ATTRIBUTES
    }

    random_generator = random.Random(
        RANDOM_SEED
    )

    selected_samples = {}

    for attribute in ATTRIBUTES:
        positive_ids = find_positive_ids(
            attribute_maps[
                attribute
            ]
        )

        valid_ids = [
            sample_id
            for sample_id in positive_ids
            if (
                sample_id in image_map
                and sample_id in lesion_map
            )
        ]

        if valid_ids:
            selected_samples[
                attribute
            ] = random_generator.choice(
                valid_ids
            )
        else:
            selected_samples[
                attribute
            ] = None

    figure, axes = plt.subplots(
        2,
        len(ATTRIBUTES),
        figsize=(16, 7),
        squeeze=False,
    )

    for (
        column,
        attribute,
    ) in enumerate(
        ATTRIBUTES
    ):
        sample_id = (
            selected_samples[
                attribute
            ]
        )

        if sample_id is None:
            axes[0, column].text(
                0.5,
                0.5,
                "No positive sample",
                ha="center",
                va="center",
            )

            axes[1, column].text(
                0.5,
                0.5,
                "No positive sample",
                ha="center",
                va="center",
            )

            axes[0, column].set_title(
                attribute,
                fontsize=9,
            )

            continue

        image_path = first_path(
            image_map,
            sample_id,
        )

        lesion_path = first_path(
            lesion_map,
            sample_id,
        )

        attribute_path = first_path(
            attribute_maps[
                attribute
            ],
            sample_id,
        )

        if (
            image_path is None
            or lesion_path is None
            or attribute_path is None
        ):
            continue

        with Image.open(
            image_path
        ) as image_file:
            original_image_size = (
                image_file.size
            )

        with Image.open(
            lesion_path
        ) as lesion_file:
            original_lesion_size = (
                lesion_file.size
            )

        with Image.open(
            attribute_path
        ) as attribute_file:
            original_attribute_size = (
                attribute_file.size
            )

        if not (
            original_image_size
            == original_lesion_size
            == original_attribute_size
        ):
            raise ValueError(
                f"Size mismatch for "
                f"{sample_id}, {attribute}: "
                f"image={original_image_size}, "
                f"lesion={original_lesion_size}, "
                f"attribute={original_attribute_size}"
            )

        image_array = (
            load_resized_image(
                image_path
            )
        )

        lesion_mask = (
            load_resized_mask(
                lesion_path
            )
        )

        attribute_mask = (
            load_resized_mask(
                attribute_path
            )
        )

        overlay = (
            create_combined_overlay(
                image_array,
                lesion_mask,
                attribute_mask,
            )
        )

        axes[0, column].imshow(
            image_array
        )

        axes[0, column].set_title(
            f"{attribute}\n"
            f"{sample_id}",
            fontsize=9,
        )

        axes[1, column].imshow(
            overlay
        )

        attribute_coverage = float(
            attribute_mask.mean()
        )

        axes[1, column].set_title(
            "Attribute coverage: "
            f"{attribute_coverage:.2%}",
            fontsize=9,
        )

    for axis in axes.ravel():
        axis.axis("off")

    figure.suptitle(
        "Task 2 Attribute-Mask Alignment Check\n"
        "Green: lesion area, Red: selected attribute",
        fontsize=14,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_attribute_overlay_check()