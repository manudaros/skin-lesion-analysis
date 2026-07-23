import csv

from qc_config import (
    DATA_ROOT,
    INDEX_CSV,
    TASK1_IMAGE_DIR,
)


def create_index_csv() -> None:
    """Create the central dataset index CSV file."""
    task1_dir = "Task1_Segmentation"
    task2_dir = "Task2_Attributes"

    columns = [
        "image_id",
        "task1_image_path",
        "task1_mask_path",
        "pigment_network_mask",
        "negative_network_mask",
        "streaks_mask",
        "milia_like_cysts_mask",
        "globules_mask",
    ]

    rows = []

    if not TASK1_IMAGE_DIR.exists():
        print(
            "Error: Directory "
            f"'{TASK1_IMAGE_DIR.resolve()}' "
            "was not found."
        )
        return

    image_files = sorted(
        TASK1_IMAGE_DIR.glob("*.jpg")
    )

    for image_path in image_files:
        image_id = image_path.stem

        row = {
            "image_id": image_id,
            "task1_image_path": (
                f"{task1_dir}/images/"
                f"{image_id}.jpg"
            ),
            "task1_mask_path": (
                f"{task1_dir}/masks/"
                f"{image_id}_segmentation.png"
            ),
            "pigment_network_mask": (
                f"{task2_dir}/masks/"
                f"pigment_network/"
                f"{image_id}_attribute_"
                f"pigment_network.png"
            ),
            "negative_network_mask": (
                f"{task2_dir}/masks/"
                f"negative_network/"
                f"{image_id}_attribute_"
                f"negative_network.png"
            ),
            "streaks_mask": (
                f"{task2_dir}/masks/"
                f"streaks/"
                f"{image_id}_attribute_"
                f"streaks.png"
            ),
            "milia_like_cysts_mask": (
                f"{task2_dir}/masks/"
                f"milia_like_cysts/"
                f"{image_id}_attribute_"
                f"milia_like_cyst.png"
            ),
            "globules_mask": (
                f"{task2_dir}/masks/"
                f"globules/"
                f"{image_id}_attribute_"
                f"globules.png"
            ),
        }

        rows.append(row)

    with INDEX_CSV.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "Successfully created "
        f"'index.csv' at: "
        f"{INDEX_CSV.resolve()}"
    )

    print(
        f"Total rows processed: "
        f"{len(rows)}"
    )

    print(
        f"Data root used: "
        f"{DATA_ROOT.resolve()}"
    )


if __name__ == "__main__":
    create_index_csv()