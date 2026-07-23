import csv
from pathlib import Path

import numpy as np

from qc_config import (
    ATTRIBUTES,
    DATASET_STATISTICS_REPORT,
    OUTPUT_ROOT,
    READABILITY_REPORT,
    TASK1_MASK_REPORT,
    TASK2_ATTRIBUTE_SUMMARY_REPORT,
    TASK2_MASK_REPORT,
)


def read_csv_rows(
    path: Path,
) -> list[dict[str, str]]:
    """Read all rows from a CSV report."""
    if not path.exists():
        raise FileNotFoundError(
            "Required report does not exist: "
            f"{path}"
        )

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


def parse_float(
    value: str | None,
) -> float | None:
    """
    Convert a CSV value to float
    while safely handling empty values.
    """
    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    try:
        return float(value)

    except ValueError:
        return None


def numeric_summary(
    values: list[float],
) -> dict[str, float | int | None]:
    """Calculate basic descriptive statistics."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p05": float(
            np.percentile(
                array,
                5,
            )
        ),
        "p25": float(
            np.percentile(
                array,
                25,
            )
        ),
        "median": float(
            np.percentile(
                array,
                50,
            )
        ),
        "p75": float(
            np.percentile(
                array,
                75,
            )
        ),
        "p95": float(
            np.percentile(
                array,
                95,
            )
        ),
        "max": float(array.max()),
    }


def format_optional_number(
    value: float | int | None,
    digits: int = 4,
) -> str:
    """Format an optional number for a text report."""
    if value is None:
        return "N/A"

    if isinstance(
        value,
        int,
    ):
        return str(value)

    return f"{value:.{digits}f}"


def summarize_image_dimensions(
    readability_rows: list[
        dict[str, str]
    ],
) -> list[str]:
    """
    Summarize original Task 1 image
    dimensions and aspect ratios.
    """
    task1_rows = [
        row
        for row in readability_rows
        if (
            row.get("dataset")
            == "task1_images"
            and row.get("readable")
            == "True"
        )
    ]

    widths = []
    heights = []
    aspect_ratios = []

    for row in task1_rows:
        width = parse_float(
            row.get("width")
        )

        height = parse_float(
            row.get("height")
        )

        if (
            width is None
            or height is None
            or height == 0
        ):
            continue

        widths.append(width)
        heights.append(height)

        aspect_ratios.append(
            width / height
        )

    width_summary = numeric_summary(
        widths
    )

    height_summary = numeric_summary(
        heights
    )

    ratio_summary = numeric_summary(
        aspect_ratios
    )

    return [
        "ORIGINAL IMAGE DIMENSIONS",
        (
            "Readable Task 1 images: "
            f"{len(task1_rows)}"
        ),
        (
            "Width, median [min, max]: "
            f"{format_optional_number(width_summary['median'], 1)} "
            f"[{format_optional_number(width_summary['min'], 1)}, "
            f"{format_optional_number(width_summary['max'], 1)}]"
        ),
        (
            "Height, median [min, max]: "
            f"{format_optional_number(height_summary['median'], 1)} "
            f"[{format_optional_number(height_summary['min'], 1)}, "
            f"{format_optional_number(height_summary['max'], 1)}]"
        ),
        (
            "Aspect ratio, median [p05, p95]: "
            f"{format_optional_number(ratio_summary['median'])} "
            f"[{format_optional_number(ratio_summary['p05'])}, "
            f"{format_optional_number(ratio_summary['p95'])}]"
        ),
    ]


def summarize_task1_masks(
    task1_rows: list[
        dict[str, str]
    ],
) -> list[str]:
    """
    Summarize lesion-area and morphology
    statistics.
    """
    area_values = []
    component_values = []

    border_touch_count = 0
    readable_count = 0

    for row in task1_rows:
        if (
            row.get("readable")
            != "True"
        ):
            continue

        readable_count += 1

        area = parse_float(
            row.get(
                "foreground_ratio"
            )
        )

        components = parse_float(
            row.get(
                "connected_components"
            )
        )

        if area is not None:
            area_values.append(
                area
            )

        if components is not None:
            component_values.append(
                components
            )

        if (
            row.get(
                "touches_image_border"
            )
            == "True"
        ):
            border_touch_count += 1

    area_summary = numeric_summary(
        area_values
    )

    component_summary = (
        numeric_summary(
            component_values
        )
    )

    border_touch_rate = (
        border_touch_count
        / readable_count
        if readable_count > 0
        else 0.0
    )

    return [
        "TASK 1 LESION MASKS",
        (
            "Readable lesion masks: "
            f"{readable_count}"
        ),
        (
            "Lesion area ratio, mean: "
            f"{format_optional_number(area_summary['mean'])}"
        ),
        (
            "Lesion area ratio, median [p05, p95]: "
            f"{format_optional_number(area_summary['median'])} "
            f"[{format_optional_number(area_summary['p05'])}, "
            f"{format_optional_number(area_summary['p95'])}]"
        ),
        (
            "Connected components, median [min, max]: "
            f"{format_optional_number(component_summary['median'], 1)} "
            f"[{format_optional_number(component_summary['min'], 1)}, "
            f"{format_optional_number(component_summary['max'], 1)}]"
        ),
        (
            "Masks touching the image border: "
            f"{border_touch_count}/"
            f"{readable_count} "
            f"({border_touch_rate:.1%})"
        ),
    ]


def summarize_task2_attributes(
    task2_rows: list[
        dict[str, str]
    ],
) -> list[dict[str, object]]:
    """
    Create one dataset-level summary
    for each Task 2 attribute.
    """
    summary_rows = []

    for attribute in ATTRIBUTES:
        attribute_rows = [
            row
            for row in task2_rows
            if (
                row.get("attribute")
                == attribute
            )
        ]

        readable_rows = [
            row
            for row in attribute_rows
            if (
                row.get("readable")
                == "True"
            )
        ]

        positive_rows = []

        for row in readable_rows:
            foreground_pixels = (
                parse_float(
                    row.get(
                        "foreground_pixels"
                    )
                )
            )

            if (
                foreground_pixels
                is not None
                and foreground_pixels > 0
            ):
                positive_rows.append(
                    row
                )

        positive_areas = []

        for row in positive_rows:
            area = parse_float(
                row.get(
                    "foreground_ratio"
                )
            )

            if area is not None:
                positive_areas.append(
                    area
                )

        outside_warning_count = sum(
            "attribute_outside_lesion"
            in row.get(
                "status",
                "",
            )
            for row in attribute_rows
        )

        positive_area_summary = (
            numeric_summary(
                positive_areas
            )
        )

        readable_count = len(
            readable_rows
        )

        positive_count = len(
            positive_rows
        )

        empty_count = (
            readable_count
            - positive_count
        )

        positive_rate = (
            positive_count
            / readable_count
            if readable_count > 0
            else 0.0
        )

        summary_rows.append(
            {
                "attribute": attribute,
                "total_rows": len(
                    attribute_rows
                ),
                "readable_masks": (
                    readable_count
                ),
                "positive_masks": (
                    positive_count
                ),
                "empty_masks": (
                    empty_count
                ),
                "positive_rate": (
                    positive_rate
                ),
                "mean_positive_area": (
                    positive_area_summary[
                        "mean"
                    ]
                ),
                "median_positive_area": (
                    positive_area_summary[
                        "median"
                    ]
                ),
                "p95_positive_area": (
                    positive_area_summary[
                        "p95"
                    ]
                ),
                "outside_lesion_warnings": (
                    outside_warning_count
                ),
            }
        )

    return summary_rows


def run_dataset_statistics(
) -> tuple[Path, Path]:
    """Generate dataset-level descriptive statistics."""
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    readability_rows = read_csv_rows(
        READABILITY_REPORT
    )

    task1_rows = read_csv_rows(
        TASK1_MASK_REPORT
    )

    task2_rows = read_csv_rows(
        TASK2_MASK_REPORT
    )

    text_lines = [
        "DATASET STATISTICS",
        "=" * 70,
        "",
    ]

    text_lines.extend(
        summarize_image_dimensions(
            readability_rows
        )
    )

    text_lines.append("")

    text_lines.extend(
        summarize_task1_masks(
            task1_rows
        )
    )

    text_lines.append("")
    text_lines.append(
        "TASK 2 ATTRIBUTE MASKS"
    )

    task2_summary_rows = (
        summarize_task2_attributes(
            task2_rows
        )
    )

    for row in task2_summary_rows:
        text_lines.append(
            f"{row['attribute']}: "
            f"{row['positive_masks']}/"
            f"{row['readable_masks']} positive "
            f"({row['positive_rate']:.1%}); "
            f"mean positive area="
            f"{format_optional_number(row['mean_positive_area'])}; "
            f"outside-lesion warnings="
            f"{row['outside_lesion_warnings']}"
        )

    DATASET_STATISTICS_REPORT.write_text(
        "\n".join(
            text_lines
        ),
        encoding="utf-8",
    )

    fieldnames = [
        "attribute",
        "total_rows",
        "readable_masks",
        "positive_masks",
        "empty_masks",
        "positive_rate",
        "mean_positive_area",
        "median_positive_area",
        "p95_positive_area",
        "outside_lesion_warnings",
    ]

    with TASK2_ATTRIBUTE_SUMMARY_REPORT.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            task2_summary_rows
        )

    print(
        "\n".join(
            text_lines
        )
    )

    print(
        "\nStatistics report: "
        f"{DATASET_STATISTICS_REPORT}"
    )

    print(
        "Task 2 summary CSV: "
        f"{TASK2_ATTRIBUTE_SUMMARY_REPORT}"
    )

    return (
        DATASET_STATISTICS_REPORT,
        TASK2_ATTRIBUTE_SUMMARY_REPORT,
    )


if __name__ == "__main__":
    run_dataset_statistics()