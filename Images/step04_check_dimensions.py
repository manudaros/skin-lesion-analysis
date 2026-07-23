import csv
from pathlib import Path

from qc_config import (
    ATTRIBUTES,
    DIMENSION_REPORT,
    OUTPUT_ROOT,
    TASK1_IMAGE_DIR,
    TASK1_MASK_DIR,
    TASK2_IMAGE_DIR,
    TASK2_MASK_ROOT,
)
from qc_utils import (
    build_id_map,
    combine_status,
    first_path,
    list_image_files,
    read_image_size,
)


def safe_read_size(
    path: Path | None,
) -> tuple[
    int | None,
    int | None,
    str,
]:
    """
    Read image dimensions while safely
    recording decoding errors.
    """
    if path is None:
        return (
            None,
            None,
            "File is missing.",
        )

    try:
        width, height = read_image_size(
            path
        )

        return (
            width,
            height,
            "",
        )

    except Exception as error:
        return (
            None,
            None,
            f"{type(error).__name__}: "
            f"{error}",
        )


def inspect_pair(
    task: str,
    sample_id: str,
    attribute: str,
    image_path: Path | None,
    mask_path: Path | None,
) -> dict[str, object]:
    """Compare the dimensions of one image-mask pair."""
    issues = []

    if image_path is None:
        issues.append(
            "missing_image"
        )

    if mask_path is None:
        issues.append(
            "missing_mask"
        )

    (
        image_width,
        image_height,
        image_error,
    ) = safe_read_size(
        image_path
    )

    (
        mask_width,
        mask_height,
        mask_error,
    ) = safe_read_size(
        mask_path
    )

    if (
        image_path is not None
        and image_error
    ):
        issues.append(
            "unreadable_image"
        )

    if (
        mask_path is not None
        and mask_error
    ):
        issues.append(
            "unreadable_mask"
        )

    size_match = None

    if (
        image_width is not None
        and image_height is not None
        and mask_width is not None
        and mask_height is not None
    ):
        size_match = (
            image_width == mask_width
            and image_height == mask_height
        )

        if not size_match:
            issues.append(
                "size_mismatch"
            )

    return {
        "task": task,
        "image_id": sample_id,
        "attribute": attribute,
        "image_path": (
            str(image_path)
            if image_path
            else ""
        ),
        "mask_path": (
            str(mask_path)
            if mask_path
            else ""
        ),
        "image_width": image_width,
        "image_height": image_height,
        "mask_width": mask_width,
        "mask_height": mask_height,
        "size_match": size_match,
        "image_error": image_error,
        "mask_error": mask_error,
        "status": combine_status(
            issues
        ),
    }


def run_dimension_check() -> Path:
    """
    Check Task 1 and Task 2 image-mask
    dimension consistency.
    """
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    task1_image_map = build_id_map(
        list_image_files(
            TASK1_IMAGE_DIR
        )
    )

    task1_mask_map = build_id_map(
        list_image_files(
            TASK1_MASK_DIR
        )
    )

    task2_image_map = build_id_map(
        list_image_files(
            TASK2_IMAGE_DIR
        )
    )

    # Use Task 1 images if Task 2 has no separate image folder.
    reference_image_map = (
        task2_image_map
        if task2_image_map
        else task1_image_map
    )

    rows = []

    task1_ids = sorted(
        set(task1_image_map)
        | set(task1_mask_map)
    )

    for sample_id in task1_ids:
        rows.append(
            inspect_pair(
                task="task1",
                sample_id=sample_id,
                attribute="",
                image_path=first_path(
                    task1_image_map,
                    sample_id,
                ),
                mask_path=first_path(
                    task1_mask_map,
                    sample_id,
                ),
            )
        )

    for attribute in ATTRIBUTES:
        attribute_map = build_id_map(
            list_image_files(
                TASK2_MASK_ROOT
                / attribute
            )
        )

        task2_ids = sorted(
            set(reference_image_map)
            | set(attribute_map)
        )

        for sample_id in task2_ids:
            rows.append(
                inspect_pair(
                    task="task2",
                    sample_id=sample_id,
                    attribute=attribute,
                    image_path=first_path(
                        reference_image_map,
                        sample_id,
                    ),
                    mask_path=first_path(
                        attribute_map,
                        sample_id,
                    ),
                )
            )

    fieldnames = [
        "task",
        "image_id",
        "attribute",
        "image_path",
        "mask_path",
        "image_width",
        "image_height",
        "mask_width",
        "mask_height",
        "size_match",
        "image_error",
        "mask_error",
        "status",
    ]

    with DIMENSION_REPORT.open(
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

    warning_count = sum(
        row["status"] != "OK"
        for row in rows
    )

    mismatch_count = sum(
        row["size_match"] is False
        for row in rows
    )

    print("=" * 70)
    print("DIMENSION CHECK")
    print("=" * 70)

    print(
        f"Checked pairs: "
        f"{len(rows)}"
    )

    print(
        f"Dimension mismatches: "
        f"{mismatch_count}"
    )

    print(
        f"Rows with warnings: "
        f"{warning_count}"
    )

    print(
        f"Report saved to: "
        f"{DIMENSION_REPORT}"
    )

    return DIMENSION_REPORT


if __name__ == "__main__":
    run_dimension_check()