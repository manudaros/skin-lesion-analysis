import csv
from pathlib import Path

import numpy as np
from PIL import Image

from qc_config import (
    NEAR_UNIFORM_IMAGE_STD_THRESHOLD,
    OUTPUT_ROOT,
    READABILITY_REPORT,
    TASK1_IMAGE_DIR,
    TASK2_IMAGE_DIR,
)
from qc_utils import list_image_files


def inspect_original_image(
    dataset_name: str,
    path: Path,
) -> dict[str, object]:
    """
    Fully decode one original image and record
    basic pixel statistics.
    """
    result = {
        "dataset": dataset_name,
        "file_path": str(path),
        "readable": False,
        "format": "",
        "mode": "",
        "mode_is_rgb": "",
        "width": "",
        "height": "",
        "pixel_min": "",
        "pixel_max": "",
        "pixel_mean": "",
        "pixel_std": "",
        "all_black": "",
        "all_white": "",
        "near_uniform": "",
        "error": "",
    }

    try:
        with Image.open(path) as image:
            image.load()

            result["format"] = (
                image.format
                or ""
            )

            result["mode"] = image.mode
            result["mode_is_rgb"] = (
                image.mode == "RGB"
            )

            result["width"] = image.width
            result["height"] = image.height

            rgb_array = np.asarray(
                image.convert("RGB"),
                dtype=np.float32,
            )

        pixel_min = float(
            rgb_array.min()
        )

        pixel_max = float(
            rgb_array.max()
        )

        pixel_mean = float(
            rgb_array.mean()
        )

        pixel_std = float(
            rgb_array.std()
        )

        result["pixel_min"] = pixel_min
        result["pixel_max"] = pixel_max
        result["pixel_mean"] = pixel_mean
        result["pixel_std"] = pixel_std

        result["all_black"] = (
            pixel_max == 0.0
        )

        result["all_white"] = (
            pixel_min == 255.0
        )

        result["near_uniform"] = (
            pixel_std
            < NEAR_UNIFORM_IMAGE_STD_THRESHOLD
        )

        result["readable"] = True

    except Exception as error:
        result["error"] = (
            f"{type(error).__name__}: "
            f"{error}"
        )

    return result


def collect_original_image_datasets(
) -> list[tuple[str, Path]]:
    """
    Return all original-image folders
    that should be inspected.
    """
    datasets = [
        (
            "task1_images",
            TASK1_IMAGE_DIR,
        )
    ]

    task2_files = list_image_files(
        TASK2_IMAGE_DIR
    )

    if task2_files:
        datasets.append(
            (
                "task2_images",
                TASK2_IMAGE_DIR,
            )
        )

    return datasets


def run_image_readability_check() -> Path:
    """
    Inspect all original images and save
    a readability report.
    """
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print("=" * 70)
    print(
        "ORIGINAL IMAGE READABILITY CHECK"
    )
    print("=" * 70)

    datasets = (
        collect_original_image_datasets()
    )

    for (
        dataset_name,
        folder,
    ) in datasets:
        if not folder.exists():
            print(
                f"Missing folder: "
                f"{folder}"
            )
            continue

        files = list_image_files(
            folder
        )

        print(
            f"{dataset_name}: "
            f"{len(files)} files"
        )

        for (
            index,
            path,
        ) in enumerate(
            files,
            start=1,
        ):
            rows.append(
                inspect_original_image(
                    dataset_name,
                    path,
                )
            )

            if (
                index % 500 == 0
                or index == len(files)
            ):
                print(
                    f"  Processed "
                    f"{index}/{len(files)} files"
                )

    fieldnames = [
        "dataset",
        "file_path",
        "readable",
        "format",
        "mode",
        "mode_is_rgb",
        "width",
        "height",
        "pixel_min",
        "pixel_max",
        "pixel_mean",
        "pixel_std",
        "all_black",
        "all_white",
        "near_uniform",
        "error",
    ]

    with READABILITY_REPORT.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    unreadable_count = sum(
        row["readable"] is False
        for row in rows
    )

    black_count = sum(
        row["all_black"] is True
        for row in rows
    )

    white_count = sum(
        row["all_white"] is True
        for row in rows
    )

    near_uniform_count = sum(
        row["near_uniform"] is True
        for row in rows
    )

    non_rgb_count = sum(
        row["mode_is_rgb"] is False
        for row in rows
        if row["readable"] is True
    )

    print(
        f"\nInspected files: "
        f"{len(rows)}"
    )

    print(
        f"Unreadable files: "
        f"{unreadable_count}"
    )

    print(
        f"Non-RGB files: "
        f"{non_rgb_count}"
    )

    print(
        f"All-black files: "
        f"{black_count}"
    )

    print(
        f"All-white files: "
        f"{white_count}"
    )

    print(
        f"Near-uniform files: "
        f"{near_uniform_count}"
    )

    print(
        f"Report saved to: "
        f"{READABILITY_REPORT}"
    )

    return READABILITY_REPORT


if __name__ == "__main__":
    run_image_readability_check()