import csv

from qc_config import (
    DATA_ROOT,
    INDEX_CSV,
    INDEX_VALIDATION_REPORT,
    OUTPUT_ROOT,
    TASK1_IMAGE_DIR,
)
from qc_utils import (
    extract_image_id,
    list_image_files,
)


REQUIRED_COLUMNS = [
    "image_id",
    "task1_image_path",
    "task1_mask_path",
    "pigment_network_mask",
    "negative_network_mask",
    "streaks_mask",
    "milia_like_cysts_mask",
    "globules_mask",
]

PATH_COLUMNS = [
    "task1_image_path",
    "task1_mask_path",
    "pigment_network_mask",
    "negative_network_mask",
    "streaks_mask",
    "milia_like_cysts_mask",
    "globules_mask",
]


def validate_index_csv() -> bool:
    """
    Validate required columns, image IDs, paths,
    and file existence.
    """
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INDEX_CSV.exists():
        print(
            f"Error: '{INDEX_CSV.resolve()}' "
            "does not exist."
        )
        print(
            "Please run "
            "step01_create_index_csv.py first."
        )
        return False

    validation_rows = []
    seen_ids = set()
    seen_paths = set()
    index_ids = set()
    row_count = 0

    print(
        "Starting index validation...\n"
    )

    with INDEX_CSV.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )

        fieldnames = (
            reader.fieldnames
            or []
        )

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in fieldnames
        ]

        if missing_columns:
            for column in missing_columns:
                validation_rows.append(
                    {
                        "row": "",
                        "image_id": "",
                        "column": column,
                        "status": (
                            "missing_required_column"
                        ),
                        "path": "",
                    }
                )

        else:
            for (
                row_index,
                row,
            ) in enumerate(
                reader,
                start=1,
            ):
                row_count += 1

                image_id = (
                    row.get("image_id")
                    or ""
                ).strip()

                if not image_id:
                    validation_rows.append(
                        {
                            "row": row_index,
                            "image_id": "",
                            "column": "image_id",
                            "status": (
                                "missing_entry"
                            ),
                            "path": "",
                        }
                    )

                elif image_id in seen_ids:
                    validation_rows.append(
                        {
                            "row": row_index,
                            "image_id": image_id,
                            "column": "image_id",
                            "status": (
                                "duplicate_image_id"
                            ),
                            "path": "",
                        }
                    )

                else:
                    seen_ids.add(
                        image_id
                    )
                    index_ids.add(
                        image_id
                    )

                for column_name in PATH_COLUMNS:
                    relative_path = (
                        row.get(column_name)
                        or ""
                    ).strip()

                    if not relative_path:
                        validation_rows.append(
                            {
                                "row": row_index,
                                "image_id": image_id,
                                "column": column_name,
                                "status": (
                                    "missing_entry"
                                ),
                                "path": "",
                            }
                        )
                        continue

                    relative_path_object = (
                        DATA_ROOT
                        / relative_path
                    )

                    if relative_path in seen_paths:
                        validation_rows.append(
                            {
                                "row": row_index,
                                "image_id": image_id,
                                "column": column_name,
                                "status": (
                                    "duplicate_path"
                                ),
                                "path": relative_path,
                            }
                        )
                    else:
                        seen_paths.add(
                            relative_path
                        )

                    if not relative_path_object.exists():
                        validation_rows.append(
                            {
                                "row": row_index,
                                "image_id": image_id,
                                "column": column_name,
                                "status": (
                                    "file_not_found"
                                ),
                                "path": relative_path,
                            }
                        )

    disk_image_ids = {
        extract_image_id(path)
        for path in list_image_files(
            TASK1_IMAGE_DIR
        )
    }

    missing_from_index = (
        disk_image_ids
        - index_ids
    )

    for image_id in sorted(
        missing_from_index
    ):
        validation_rows.append(
            {
                "row": "",
                "image_id": image_id,
                "column": "image_id",
                "status": (
                    "disk_image_missing_from_index"
                ),
                "path": "",
            }
        )

    fieldnames = [
        "row",
        "image_id",
        "column",
        "status",
        "path",
    ]

    with INDEX_VALIDATION_REPORT.open(
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
            validation_rows
        )

    print("-" * 50)
    print(
        "Index validation results"
    )
    print("-" * 50)

    print(
        f"Rows checked: "
        f"{row_count}"
    )

    print(
        f"Issues found: "
        f"{len(validation_rows)}"
    )

    print(
        "Validation report: "
        f"{INDEX_VALIDATION_REPORT}"
    )

    if validation_rows:
        print(
            "\nFirst five issues:"
        )

        for issue in validation_rows[:5]:
            print(
                f"- Row {issue['row']} "
                f"({issue['image_id']}), "
                f"column "
                f"'{issue['column']}': "
                f"{issue['status']} "
                f"{issue['path']}"
            )

        return False

    print(
        "Success! Every indexed file exists, "
        "and all image IDs are unique."
    )

    return True


if __name__ == "__main__":
    validate_index_csv()