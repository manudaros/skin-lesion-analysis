from pathlib import Path
import shutil

from qc_config import TASK2_MASK_ROOT


# Source directory containing all downloaded Task 2 masks.
SOURCE_DIR = (
    Path.home()
    / "Downloads"
    / "summer_school_project_train"
    / "train"
    / "task2_gt"
)

# Destination directory containing the five attribute folders.
DESTINATION_ROOT = TASK2_MASK_ROOT

# Map filename keywords to destination folder names.
ATTRIBUTE_MAPPING = {
    "attribute_globules": "globules",
    "attribute_milia_like_cyst": "milia_like_cysts",
    "attribute_negative_network": "negative_network",
    "attribute_pigment_network": "pigment_network",
    "attribute_streaks": "streaks",
}


def create_destination_folders() -> None:
    """Create all destination folders if needed."""
    for folder_name in ATTRIBUTE_MAPPING.values():
        folder_path = (
            DESTINATION_ROOT
            / folder_name
        )

        folder_path.mkdir(
            parents=True,
            exist_ok=True,
        )


def classify_and_copy_masks() -> None:
    """
    Classify Task 2 masks by filename and copy them
    to their corresponding attribute folders.
    """
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            "Source directory does not exist: "
            f"{SOURCE_DIR}"
        )

    create_destination_folders()

    counts = {
        folder_name: 0
        for folder_name
        in ATTRIBUTE_MAPPING.values()
    }

    unmatched_files = []

    for source_file in SOURCE_DIR.iterdir():
        if not source_file.is_file():
            continue

        if source_file.suffix.lower() != ".png":
            continue

        filename_lower = (
            source_file.name.lower()
        )

        matched = False

        for (
            keyword,
            folder_name,
        ) in ATTRIBUTE_MAPPING.items():
            if keyword not in filename_lower:
                continue

            destination_file = (
                DESTINATION_ROOT
                / folder_name
                / source_file.name
            )

            shutil.copy2(
                source_file,
                destination_file,
            )

            counts[folder_name] += 1
            matched = True
            break

        if not matched:
            unmatched_files.append(
                source_file.name
            )

    print("\nClassification completed.")
    print(
        f"Source directory: "
        f"{SOURCE_DIR}"
    )
    print(
        f"Destination directory: "
        f"{DESTINATION_ROOT}\n"
    )

    total_copied = 0

    for (
        folder_name,
        count,
    ) in counts.items():
        print(
            f"{folder_name}: "
            f"{count}"
        )
        total_copied += count

    print(
        f"\nTotal copied files: "
        f"{total_copied}"
    )

    if unmatched_files:
        print(
            f"\nUnmatched files: "
            f"{len(unmatched_files)}"
        )

        for filename in unmatched_files[:20]:
            print(f"  {filename}")

        if len(unmatched_files) > 20:
            remaining = (
                len(unmatched_files)
                - 20
            )
            print(
                f"  ... and "
                f"{remaining} more files"
            )
    else:
        print("Unmatched files: 0")

    expected_per_attribute = 2700

    expected_total = (
        expected_per_attribute
        * len(ATTRIBUTE_MAPPING)
    )

    if total_copied == expected_total:
        print(
            "\nDataset count check passed."
        )
    else:
        print(
            "\nWarning: Expected "
            f"{expected_total} files, "
            f"but copied {total_copied}."
        )


if __name__ == "__main__":
    classify_and_copy_masks()