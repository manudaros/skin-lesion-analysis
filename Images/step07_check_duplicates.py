import csv
from collections import defaultdict
from pathlib import Path

from qc_config import (
    ATTRIBUTES,
    DUPLICATE_ID_REPORT,
    DUPLICATE_IMAGE_REPORT,
    IMAGE_COPY_REPORT,
    OUTPUT_ROOT,
    TASK1_IMAGE_DIR,
    TASK1_MASK_DIR,
    TASK2_IMAGE_DIR,
    TASK2_MASK_ROOT,
)
from qc_utils import (
    build_id_map,
    calculate_sha256,
    extract_image_id,
    first_path,
    list_image_files,
)


def collect_duplicate_ids(
    dataset_name: str,
    id_map: dict[
        str,
        list[Path],
    ],
) -> list[dict[str, object]]:
    """
    Find image IDs associated with
    more than one file.
    """
    rows = []

    for (
        sample_id,
        paths,
    ) in sorted(
        id_map.items()
    ):
        if len(paths) <= 1:
            continue

        rows.append(
            {
                "dataset": dataset_name,
                "image_id": sample_id,
                "file_count": len(paths),
                "file_paths": " | ".join(
                    str(path)
                    for path in paths
                ),
            }
        )

    return rows


def check_duplicate_ids(
) -> list[dict[str, object]]:
    """
    Check duplicate IDs in every image
    and mask directory.
    """
    datasets = {
        "task1_images": (
            TASK1_IMAGE_DIR
        ),
        "task1_masks": (
            TASK1_MASK_DIR
        ),
        "task2_images": (
            TASK2_IMAGE_DIR
        ),
    }

    for attribute in ATTRIBUTES:
        datasets[
            f"task2_{attribute}"
        ] = (
            TASK2_MASK_ROOT
            / attribute
        )

    rows = []

    for (
        dataset_name,
        folder,
    ) in datasets.items():
        id_map = build_id_map(
            list_image_files(
                folder
            )
        )

        rows.extend(
            collect_duplicate_ids(
                dataset_name,
                id_map,
            )
        )

    fieldnames = [
        "dataset",
        "image_id",
        "file_count",
        "file_paths",
    ]

    with DUPLICATE_ID_REPORT.open(
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


def check_exact_duplicate_images(
) -> list[dict[str, object]]:
    """
    Find Task 1 images with
    exactly identical file content.
    """
    image_files = list_image_files(
        TASK1_IMAGE_DIR
    )

    hash_groups: dict[
        str,
        list[Path],
    ] = defaultdict(list)

    for (
        index,
        path,
    ) in enumerate(
        image_files,
        start=1,
    ):
        try:
            file_hash = (
                calculate_sha256(
                    path
                )
            )

            hash_groups[
                file_hash
            ].append(
                path
            )

        except Exception as error:
            print(
                f"Could not hash "
                f"{path}: {error}"
            )

        if (
            index % 500 == 0
            or index == len(image_files)
        ):
            print(
                "Task 1 images hashed: "
                f"{index}/{len(image_files)}"
            )

    rows = []

    for (
        file_hash,
        paths,
    ) in hash_groups.items():
        if len(paths) <= 1:
            continue

        rows.append(
            {
                "sha256": file_hash,
                "file_count": len(paths),
                "image_ids": " | ".join(
                    extract_image_id(
                        path
                    )
                    for path in paths
                ),
                "file_paths": " | ".join(
                    str(path)
                    for path in paths
                ),
            }
        )

    fieldnames = [
        "sha256",
        "file_count",
        "image_ids",
        "file_paths",
    ]

    with DUPLICATE_IMAGE_REPORT.open(
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


def compare_task1_task2_images(
) -> list[dict[str, object]]:
    """
    Compare Task 1 and Task 2 copies
    of each original image.

    SHA-256 only detects byte-identical files.
    Differently encoded copies may have different hashes
    even when their visual content is the same.
    """
    task1_map = build_id_map(
        list_image_files(
            TASK1_IMAGE_DIR
        )
    )

    task2_map = build_id_map(
        list_image_files(
            TASK2_IMAGE_DIR
        )
    )

    all_ids = sorted(
        set(task1_map)
        | set(task2_map)
    )

    rows = []
    hash_cache: dict[
        Path,
        str,
    ] = {}

    def cached_hash(
        path: Path,
    ) -> str:
        if path not in hash_cache:
            hash_cache[path] = (
                calculate_sha256(
                    path
                )
            )

        return hash_cache[path]

    for sample_id in all_ids:
        task1_path = first_path(
            task1_map,
            sample_id,
        )

        task2_path = first_path(
            task2_map,
            sample_id,
        )

        same_content = None
        error = ""

        if (
            task1_path is not None
            and task2_path is not None
        ):
            try:
                same_content = (
                    cached_hash(
                        task1_path
                    )
                    == cached_hash(
                        task2_path
                    )
                )

            except Exception as exception:
                error = (
                    f"{type(exception).__name__}: "
                    f"{exception}"
                )

        rows.append(
            {
                "image_id": sample_id,
                "task1_path": (
                    str(task1_path)
                    if task1_path
                    else ""
                ),
                "task2_path": (
                    str(task2_path)
                    if task2_path
                    else ""
                ),
                "task1_exists": (
                    task1_path
                    is not None
                ),
                "task2_exists": (
                    task2_path
                    is not None
                ),
                "same_content": (
                    same_content
                ),
                "error": error,
            }
        )

    fieldnames = [
        "image_id",
        "task1_path",
        "task2_path",
        "task1_exists",
        "task2_exists",
        "same_content",
        "error",
    ]

    with IMAGE_COPY_REPORT.open(
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


def run_duplicate_checks(
) -> tuple[
    Path,
    Path,
    Path,
]:
    """
    Run all duplicate-ID and
    duplicate-image checks.
    """
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("DUPLICATE CHECK")
    print("=" * 70)

    duplicate_id_rows = (
        check_duplicate_ids()
    )

    duplicate_image_rows = (
        check_exact_duplicate_images()
    )

    task2_image_files = (
        list_image_files(
            TASK2_IMAGE_DIR
        )
    )

    if task2_image_files:
        image_copy_rows = (
            compare_task1_task2_images()
        )
    else:
        image_copy_rows = []

        print(
            "Task 2 image directory is absent "
            "or empty. Image-copy comparison "
            "was skipped."
        )

    different_copies = sum(
        row["same_content"] is False
        for row in image_copy_rows
    )

    print(
        "\nDuplicate ID groups: "
        f"{len(duplicate_id_rows)}"
    )

    print(
        "Exact duplicate Task 1 "
        "image groups: "
        f"{len(duplicate_image_rows)}"
    )

    print(
        "Different Task 1 and Task 2 "
        "image copies: "
        f"{different_copies}"
    )

    print(
        "Duplicate ID report: "
        f"{DUPLICATE_ID_REPORT}"
    )

    print(
        "Exact duplicate report: "
        f"{DUPLICATE_IMAGE_REPORT}"
    )

    print(
        "Image-copy report: "
        f"{IMAGE_COPY_REPORT}"
    )

    return (
        DUPLICATE_ID_REPORT,
        DUPLICATE_IMAGE_REPORT,
        IMAGE_COPY_REPORT,
    )


if __name__ == "__main__":
    run_duplicate_checks()