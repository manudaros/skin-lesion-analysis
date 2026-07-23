import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from qc_config import (
    ATTRIBUTES,
    LARGE_LESION_THRESHOLD,
    OUTSIDE_LESION_WARNING_THRESHOLD,
    OUTPUT_ROOT,
    SMALL_LESION_THRESHOLD,
    TASK1_IMAGE_DIR,
    TASK1_MASK_DIR,
    TASK1_MASK_REPORT,
    TASK2_MASK_REPORT,
    TASK2_MASK_ROOT,
)
from qc_utils import (
    build_id_map,
    combine_status,
    first_path,
    list_image_files,
)


def inspect_mask(
    path: Path | None,
) -> tuple[
    dict[str, object],
    np.ndarray | None,
]:
    """
    Inspect one mask and return statistics
    plus a Boolean mask array.
    """
    result = {
        "readable": False,
        "width": None,
        "height": None,
        "mode": "",
        "unique_value_count": None,
        "unique_values": "",
        "binary_values": None,
        "foreground_pixels": None,
        "total_pixels": None,
        "foreground_ratio": None,
        "empty_mask": None,
        "full_mask": None,
        "connected_components": None,
        "largest_component_fraction": None,
        "touches_image_border": None,
        "error": "",
    }

    if path is None:
        result["error"] = (
            "File is missing."
        )
        return result, None

    try:
        with Image.open(path) as image:
            image.load()

            result["width"] = (
                image.width
            )

            result["height"] = (
                image.height
            )

            result["mode"] = (
                image.mode
            )

            grayscale = np.asarray(
                image.convert("L"),
                dtype=np.uint8,
            )

        unique_values = np.unique(
            grayscale
        )

        result["unique_value_count"] = int(
            len(unique_values)
        )

        result["unique_values"] = ",".join(
            str(int(value))
            for value in unique_values[:30]
        )

        allowed_values = {
            0,
            1,
            255,
        }

        actual_values = {
            int(value)
            for value in unique_values
        }

        result["binary_values"] = (
            actual_values.issubset(
                allowed_values
            )
        )

        binary_mask = (
            grayscale > 0
        )

        foreground_pixels = int(
            binary_mask.sum()
        )

        total_pixels = int(
            binary_mask.size
        )

        result["foreground_pixels"] = (
            foreground_pixels
        )

        result["total_pixels"] = (
            total_pixels
        )

        result["foreground_ratio"] = (
            foreground_pixels
            / total_pixels
            if total_pixels > 0
            else 0.0
        )

        result["empty_mask"] = (
            foreground_pixels == 0
        )

        result["full_mask"] = (
            foreground_pixels
            == total_pixels
        )

        if foreground_pixels > 0:
            # Use 8-connectivity for connected-component analysis.
            structure = np.ones(
                (3, 3),
                dtype=np.uint8,
            )

            (
                labeled_mask,
                component_count,
            ) = ndimage.label(
                binary_mask,
                structure=structure,
            )

            component_sizes = (
                np.bincount(
                    labeled_mask.ravel()
                )[1:]
            )

            largest_component = (
                int(
                    component_sizes.max()
                )
                if len(component_sizes) > 0
                else 0
            )

            result[
                "connected_components"
            ] = int(
                component_count
            )

            result[
                "largest_component_fraction"
            ] = (
                largest_component
                / foreground_pixels
            )

            result[
                "touches_image_border"
            ] = bool(
                binary_mask[0, :].any()
                or binary_mask[-1, :].any()
                or binary_mask[:, 0].any()
                or binary_mask[:, -1].any()
            )

        else:
            result[
                "connected_components"
            ] = 0

            result[
                "largest_component_fraction"
            ] = 0.0

            result[
                "touches_image_border"
            ] = False

        result["readable"] = True

        return (
            result,
            binary_mask,
        )

    except Exception as error:
        result["error"] = (
            f"{type(error).__name__}: "
            f"{error}"
        )

        return (
            result,
            None,
        )


def run_task1_mask_check(
) -> list[dict[str, object]]:
    """Inspect all Task 1 lesion masks."""
    image_map = build_id_map(
        list_image_files(
            TASK1_IMAGE_DIR
        )
    )

    mask_map = build_id_map(
        list_image_files(
            TASK1_MASK_DIR
        )
    )

    rows = []

    all_ids = sorted(
        set(image_map)
        | set(mask_map)
    )

    for (
        index,
        sample_id,
    ) in enumerate(
        all_ids,
        start=1,
    ):
        mask_path = first_path(
            mask_map,
            sample_id,
        )

        (
            mask_info,
            _,
        ) = inspect_mask(
            mask_path
        )

        issues = []

        if mask_path is None:
            issues.append(
                "missing_mask"
            )

        elif not mask_info["readable"]:
            issues.append(
                "unreadable_mask"
            )

        else:
            if not mask_info[
                "binary_values"
            ]:
                issues.append(
                    "non_binary_values"
                )

            if mask_info[
                "empty_mask"
            ]:
                issues.append(
                    "empty_lesion_mask"
                )

            if mask_info[
                "full_mask"
            ]:
                issues.append(
                    "full_lesion_mask"
                )

            area_ratio = mask_info[
                "foreground_ratio"
            ]

            if (
                area_ratio is not None
                and 0
                < area_ratio
                < SMALL_LESION_THRESHOLD
            ):
                issues.append(
                    "very_small_lesion"
                )

            if (
                area_ratio is not None
                and area_ratio
                > LARGE_LESION_THRESHOLD
            ):
                issues.append(
                    "very_large_lesion"
                )

        rows.append(
            {
                "image_id": sample_id,
                "mask_path": (
                    str(mask_path)
                    if mask_path
                    else ""
                ),
                **mask_info,
                "status": combine_status(
                    issues
                ),
            }
        )

        if (
            index % 250 == 0
            or index == len(all_ids)
        ):
            print(
                "Task 1 masks processed: "
                f"{index}/{len(all_ids)}"
            )

    fieldnames = (
        list(rows[0].keys())
        if rows
        else []
    )

    with TASK1_MASK_REPORT.open(
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

    return rows


def run_task2_mask_check(
) -> list[dict[str, object]]:
    """Inspect all Task 2 attribute masks."""
    task1_mask_map = build_id_map(
        list_image_files(
            TASK1_MASK_DIR
        )
    )

    task1_image_map = build_id_map(
        list_image_files(
            TASK1_IMAGE_DIR
        )
    )

    lesion_cache: dict[
        str,
        np.ndarray | None,
    ] = {}

    rows = []

    for attribute in ATTRIBUTES:
        attribute_map = build_id_map(
            list_image_files(
                TASK2_MASK_ROOT
                / attribute
            )
        )

        all_ids = sorted(
            set(task1_image_map)
            | set(attribute_map)
        )

        for sample_id in all_ids:
            attribute_path = first_path(
                attribute_map,
                sample_id,
            )

            (
                attribute_info,
                attribute_binary,
            ) = inspect_mask(
                attribute_path
            )

            if sample_id not in lesion_cache:
                lesion_path = first_path(
                    task1_mask_map,
                    sample_id,
                )

                (
                    _,
                    lesion_binary,
                ) = inspect_mask(
                    lesion_path
                )

                lesion_cache[
                    sample_id
                ] = lesion_binary

            lesion_binary = (
                lesion_cache[
                    sample_id
                ]
            )

            outside_pixels = None
            outside_fraction = None
            lesion_attribute_size_match = None

            issues = []

            if attribute_path is None:
                issues.append(
                    "missing_attribute_mask"
                )

            elif not attribute_info[
                "readable"
            ]:
                issues.append(
                    "unreadable_attribute_mask"
                )

            else:
                if not attribute_info[
                    "binary_values"
                ]:
                    issues.append(
                        "non_binary_values"
                    )

                if attribute_info[
                    "full_mask"
                ]:
                    issues.append(
                        "full_attribute_mask"
                    )

            if (
                lesion_binary is not None
                and attribute_binary
                is not None
            ):
                lesion_attribute_size_match = (
                    lesion_binary.shape
                    == attribute_binary.shape
                )

                if not lesion_attribute_size_match:
                    issues.append(
                        "lesion_attribute_size_mismatch"
                    )

                else:
                    outside_mask = (
                        attribute_binary
                        & ~lesion_binary
                    )

                    outside_pixels = int(
                        outside_mask.sum()
                    )

                    foreground_pixels = int(
                        attribute_binary.sum()
                    )

                    if foreground_pixels > 0:
                        outside_fraction = (
                            outside_pixels
                            / foreground_pixels
                        )
                    else:
                        outside_fraction = 0.0

                    if (
                        outside_fraction
                        > OUTSIDE_LESION_WARNING_THRESHOLD
                    ):
                        issues.append(
                            "attribute_outside_lesion"
                        )

            rows.append(
                {
                    "image_id": sample_id,
                    "attribute": attribute,
                    "mask_path": (
                        str(attribute_path)
                        if attribute_path
                        else ""
                    ),
                    **attribute_info,
                    "lesion_attribute_size_match": (
                        lesion_attribute_size_match
                    ),
                    "outside_lesion_pixels": (
                        outside_pixels
                    ),
                    "outside_lesion_fraction": (
                        outside_fraction
                    ),
                    "status": combine_status(
                        issues
                    ),
                }
            )

        print(
            f"Task 2 {attribute}: "
            f"{len(all_ids)} masks processed"
        )

    fieldnames = (
        list(rows[0].keys())
        if rows
        else []
    )

    with TASK2_MASK_REPORT.open(
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

    return rows


def run_mask_quality_checks(
) -> tuple[Path, Path]:
    """
    Run Task 1 and Task 2 mask-quality
    inspections.
    """
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("MASK QUALITY CHECK")
    print("=" * 70)

    task1_rows = (
        run_task1_mask_check()
    )

    task2_rows = (
        run_task2_mask_check()
    )

    task1_warnings = sum(
        row["status"] != "OK"
        for row in task1_rows
    )

    task2_warnings = sum(
        row["status"] != "OK"
        for row in task2_rows
    )

    print(
        "\nTask 1 rows with warnings: "
        f"{task1_warnings}"
    )

    print(
        "Task 2 rows with warnings: "
        f"{task2_warnings}"
    )

    print(
        f"Task 1 report: "
        f"{TASK1_MASK_REPORT}"
    )

    print(
        f"Task 2 report: "
        f"{TASK2_MASK_REPORT}"
    )

    return (
        TASK1_MASK_REPORT,
        TASK2_MASK_REPORT,
    )


if __name__ == "__main__":
    run_mask_quality_checks()