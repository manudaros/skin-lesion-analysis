import csv
from pathlib import Path

from qc_config import (
    DIMENSION_REPORT,
    DUPLICATE_ID_REPORT,
    DUPLICATE_IMAGE_REPORT,
    IMAGE_COPY_REPORT,
    INDEX_VALIDATION_REPORT,
    READABILITY_REPORT,
    SUMMARY_REPORT,
    TASK1_MASK_REPORT,
    TASK2_MASK_REPORT,
)


def read_csv_rows(
    path: Path,
) -> list[dict[str, str]]:
    """Read a CSV report if it exists."""
    if not path.exists():
        return []

    with path.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        return list(
            csv.DictReader(
                csv_file
            )
        )


def count_exact(
    rows: list[
        dict[str, str]
    ],
    column: str,
    value: str,
) -> int:
    """Count rows containing an exact value."""
    return sum(
        row.get(column)
        == value
        for row in rows
    )


def count_status_issue(
    rows: list[
        dict[str, str]
    ],
    issue: str,
) -> int:
    """
    Count rows whose status field
    contains a given issue.
    """
    return sum(
        issue
        in row.get(
            "status",
            "",
        )
        for row in rows
    )


def generate_qc_summary() -> Path:
    """
    Generate a final human-readable
    quality-control summary.
    """
    index_rows = read_csv_rows(
        INDEX_VALIDATION_REPORT
    )

    readability_rows = read_csv_rows(
        READABILITY_REPORT
    )

    dimension_rows = read_csv_rows(
        DIMENSION_REPORT
    )

    task1_rows = read_csv_rows(
        TASK1_MASK_REPORT
    )

    task2_rows = read_csv_rows(
        TASK2_MASK_REPORT
    )

    duplicate_id_rows = read_csv_rows(
        DUPLICATE_ID_REPORT
    )

    duplicate_image_rows = (
        read_csv_rows(
            DUPLICATE_IMAGE_REPORT
        )
    )

    image_copy_rows = read_csv_rows(
        IMAGE_COPY_REPORT
    )

    lines = [
        "DATASET QUALITY-CONTROL SUMMARY",
        "=" * 70,
        "",
        "INDEX VALIDATION",
        (
            "Report available: "
            f"{INDEX_VALIDATION_REPORT.exists()}"
        ),
        (
            "Index issues: "
            f"{len(index_rows)}"
        ),
        "",
        "ORIGINAL IMAGE READABILITY",
        (
            "Report available: "
            f"{READABILITY_REPORT.exists()}"
        ),
        (
            "Images inspected: "
            f"{len(readability_rows)}"
        ),
        (
            "Unreadable images: "
            f"{count_exact(readability_rows, 'readable', 'False')}"
        ),
        (
            "Non-RGB images: "
            f"{count_exact(readability_rows, 'mode_is_rgb', 'False')}"
        ),
        (
            "All-black images: "
            f"{count_exact(readability_rows, 'all_black', 'True')}"
        ),
        (
            "All-white images: "
            f"{count_exact(readability_rows, 'all_white', 'True')}"
        ),
        (
            "Near-uniform images: "
            f"{count_exact(readability_rows, 'near_uniform', 'True')}"
        ),
        "",
        "DIMENSIONS",
        (
            "Report available: "
            f"{DIMENSION_REPORT.exists()}"
        ),
        (
            "Image-mask pairs inspected: "
            f"{len(dimension_rows)}"
        ),
        (
            "Dimension mismatches: "
            f"{count_exact(dimension_rows, 'size_match', 'False')}"
        ),
        (
            "Missing images: "
            f"{count_status_issue(dimension_rows, 'missing_image')}"
        ),
        (
            "Missing masks: "
            f"{count_status_issue(dimension_rows, 'missing_mask')}"
        ),
        "",
        "TASK 1 MASK QUALITY",
        (
            "Report available: "
            f"{TASK1_MASK_REPORT.exists()}"
        ),
        (
            "Task 1 masks inspected: "
            f"{len(task1_rows)}"
        ),
        (
            "Unreadable Task 1 masks: "
            f"{count_status_issue(task1_rows, 'unreadable_mask')}"
        ),
        (
            "Non-binary Task 1 masks: "
            f"{count_status_issue(task1_rows, 'non_binary_values')}"
        ),
        (
            "Empty lesion masks: "
            f"{count_status_issue(task1_rows, 'empty_lesion_mask')}"
        ),
        (
            "Full lesion masks: "
            f"{count_status_issue(task1_rows, 'full_lesion_mask')}"
        ),
        (
            "Very small lesions: "
            f"{count_status_issue(task1_rows, 'very_small_lesion')}"
        ),
        (
            "Very large lesions: "
            f"{count_status_issue(task1_rows, 'very_large_lesion')}"
        ),
        "",
        "TASK 2 MASK QUALITY",
        (
            "Report available: "
            f"{TASK2_MASK_REPORT.exists()}"
        ),
        (
            "Task 2 mask rows inspected: "
            f"{len(task2_rows)}"
        ),
        (
            "Missing attribute masks: "
            f"{count_status_issue(task2_rows, 'missing_attribute_mask')}"
        ),
        (
            "Unreadable attribute masks: "
            f"{count_status_issue(task2_rows, 'unreadable_attribute_mask')}"
        ),
        (
            "Non-binary attribute masks: "
            f"{count_status_issue(task2_rows, 'non_binary_values')}"
        ),
        (
            "Full attribute masks: "
            f"{count_status_issue(task2_rows, 'full_attribute_mask')}"
        ),
        (
            "Lesion-attribute size mismatches: "
            f"{count_status_issue(task2_rows, 'lesion_attribute_size_mismatch')}"
        ),
        (
            "Outside-lesion warnings: "
            f"{count_status_issue(task2_rows, 'attribute_outside_lesion')}"
        ),
        "",
        "DUPLICATES",
        (
            "Duplicate ID report available: "
            f"{DUPLICATE_ID_REPORT.exists()}"
        ),
        (
            "Duplicate ID groups: "
            f"{len(duplicate_id_rows)}"
        ),
        (
            "Exact duplicate Task 1 image groups: "
            f"{len(duplicate_image_rows)}"
        ),
        (
            "Different Task 1 and Task 2 image copies: "
            f"{count_exact(image_copy_rows, 'same_content', 'False')}"
        ),
        "",
        "INTERPRETATION NOTES",
        (
            "- Empty Task 2 masks are expected when "
            "an attribute is absent."
        ),
        (
            "- A small number of attribute pixels outside "
            "the lesion may reflect annotation-boundary differences."
        ),
        (
            "- The two overlay scripts must still be "
            "reviewed manually."
        ),
    ]

    SUMMARY_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_REPORT.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print(
        "\n".join(
            lines
        )
    )

    print(
        "\nSummary saved to: "
        f"{SUMMARY_REPORT}"
    )

    return SUMMARY_REPORT


if __name__ == "__main__":
    generate_qc_summary()