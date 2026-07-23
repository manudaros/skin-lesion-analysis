from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from PIL import Image

from qc_config import (
    ATTRIBUTES,
    INDEX_CSV,
    PROJECT_ROOT,
    RANDOM_SEED,
)


# ============================================================
# Split configuration
# ============================================================

NUMBER_OF_FOLDS = 5

# During model development, fold 0 is used as validation data.
# All remaining folds are used as training data.
DEVELOPMENT_VALIDATION_FOLD = 0

# Use four approximately equally sized lesion-area groups.
NUMBER_OF_LESION_AREA_BINS = 4

DATA_ROOT = PROJECT_ROOT / "data"

SPLIT_OUTPUT_DIR = PROJECT_ROOT / "splits"

FOLD_OUTPUT_CSV = (
    SPLIT_OUTPUT_DIR
    / "task1_task2_folds.csv"
)

FOLD_SUMMARY_CSV = (
    SPLIT_OUTPUT_DIR
    / "task1_task2_fold_summary.csv"
)

FOLD_SUMMARY_TXT = (
    SPLIT_OUTPUT_DIR
    / "task1_task2_fold_summary.txt"
)

SPLIT_METADATA_JSON = (
    SPLIT_OUTPUT_DIR
    / "task1_task2_split_metadata.json"
)

TRAIN_IDS_TXT = (
    SPLIT_OUTPUT_DIR
    / "train.txt"
)

VAL_IDS_TXT = (
    SPLIT_OUTPUT_DIR
    / "val.txt"
)


# ============================================================
# Dataset columns
# ============================================================

IMAGE_ID_COLUMN = "image_id"
TASK1_IMAGE_COLUMN = "task1_image_path"
TASK1_MASK_COLUMN = "task1_mask_path"

ATTRIBUTE_MASK_COLUMNS = {
    attribute: f"{attribute}_mask"
    for attribute in ATTRIBUTES
}

REQUIRED_COLUMNS = [
    IMAGE_ID_COLUMN,
    TASK1_IMAGE_COLUMN,
    TASK1_MASK_COLUMN,
    *ATTRIBUTE_MASK_COLUMNS.values(),
]


# ============================================================
# General utility functions
# ============================================================

def require_file(
    path: Path,
    description: str,
) -> None:
    """Raise an error when a required file does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found: {path}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"{description} is not a regular file: {path}"
        )


def clean_path_value(
    value: object,
) -> str:
    """Convert one CSV path value into a clean string."""
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
    }:
        return ""

    return text


def resolve_dataset_path(
    value: object,
    description: str,
    image_id: str,
) -> Path:
    """
    Resolve an index.csv path.

    Relative paths are first interpreted relative to data/,
    matching the path handling used by Step 12. A project-root
    fallback is retained for compatibility with older index files.
    """
    path_text = clean_path_value(
        value
    )

    if not path_text:
        raise ValueError(
            f"{description} is empty for image ID "
            f"'{image_id}'."
        )

    raw_path = Path(
        path_text
    )

    if raw_path.is_absolute():
        candidates = [
            raw_path,
        ]

    else:
        candidates = [
            DATA_ROOT / raw_path,
            PROJECT_ROOT / raw_path,
        ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    candidate_text = "\n".join(
        f"  - {candidate}"
        for candidate in candidates
    )

    raise FileNotFoundError(
        f"{description} was not found for image ID "
        f"'{image_id}'. Checked:\n"
        f"{candidate_text}"
    )


def calculate_sha256(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate the SHA-256 hash of one image file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_binary_mask(
    path: Path,
    description: str,
    image_id: str,
) -> np.ndarray:
    """
    Load one segmentation mask as a two-dimensional Boolean array.

    Both 0/1 and 0/255 masks are supported because all non-zero
    pixels are interpreted as foreground.
    """
    try:
        with Image.open(path) as image:
            image.load()

            grayscale = np.asarray(
                image.convert("L"),
                dtype=np.uint8,
            )

    except Exception as error:
        raise RuntimeError(
            f"Could not read {description} for image ID "
            f"'{image_id}': {path}\n"
            f"{type(error).__name__}: {error}"
        ) from error

    if grayscale.ndim != 2:
        raise ValueError(
            f"{description} for image ID '{image_id}' "
            f"does not have two dimensions. "
            f"Received shape: {grayscale.shape}"
        )

    if grayscale.size == 0:
        raise ValueError(
            f"{description} for image ID '{image_id}' "
            "contains no pixels."
        )

    return grayscale > 0


# ============================================================
# Index loading and validation
# ============================================================

def load_index_csv() -> pd.DataFrame:
    """Load and validate index.csv."""
    require_file(
        INDEX_CSV,
        "Dataset index",
    )

    index_df = pd.read_csv(
        INDEX_CSV,
        dtype=str,
        keep_default_na=False,
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in index_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "index.csv is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
            + "\n\nAvailable columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in index_df.columns
            )
        )

    index_df[IMAGE_ID_COLUMN] = (
        index_df[IMAGE_ID_COLUMN]
        .astype(str)
        .str.strip()
    )

    empty_id_rows = index_df[
        index_df[IMAGE_ID_COLUMN] == ""
    ]

    if not empty_id_rows.empty:
        raise ValueError(
            "index.csv contains rows with empty image IDs. "
            f"Row indices: "
            f"{empty_id_rows.index[:10].tolist()}"
        )

    duplicate_ids = index_df[
        index_df[IMAGE_ID_COLUMN].duplicated(
            keep=False
        )
    ]

    if not duplicate_ids.empty:
        examples = (
            duplicate_ids[IMAGE_ID_COLUMN]
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            "index.csv contains duplicate image IDs. "
            f"Examples: {examples}"
        )

    if len(index_df) < NUMBER_OF_FOLDS:
        raise ValueError(
            f"At least {NUMBER_OF_FOLDS} samples are required "
            f"to create {NUMBER_OF_FOLDS} folds. "
            f"Only {len(index_df)} samples were found."
        )

    return index_df.reset_index(
        drop=True
    )


# ============================================================
# Sample-level label construction
# ============================================================

def inspect_samples(
    index_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inspect all masks and construct sample-level split labels.

    The split labels include:
    - presence or absence of each Task 2 attribute;
    - Task 1 lesion area;
    - the number of positive attributes;
    - the SHA-256 hash of the Task 1 image.
    """
    records: list[dict[str, object]] = []

    total_samples = len(
        index_df
    )

    print(
        "=" * 70
    )
    print(
        "INSPECTING DATA FOR FOLD CREATION"
    )
    print(
        "=" * 70
    )
    print(
        f"Samples to inspect: {total_samples}"
    )

    for row_number, row in index_df.iterrows():
        image_id = str(
            row[IMAGE_ID_COLUMN]
        )

        image_path = resolve_dataset_path(
            value=row[TASK1_IMAGE_COLUMN],
            description="Task 1 image",
            image_id=image_id,
        )

        task1_mask_path = resolve_dataset_path(
            value=row[TASK1_MASK_COLUMN],
            description="Task 1 mask",
            image_id=image_id,
        )

        task1_mask = load_binary_mask(
            path=task1_mask_path,
            description="Task 1 mask",
            image_id=image_id,
        )

        task1_foreground_pixels = int(
            task1_mask.sum()
        )

        task1_total_pixels = int(
            task1_mask.size
        )

        task1_area_ratio = (
            task1_foreground_pixels
            / task1_total_pixels
        )

        record: dict[str, object] = {
            "image_id": image_id,
            "image_sha256": calculate_sha256(
                image_path
            ),
            "task1_foreground_pixels": (
                task1_foreground_pixels
            ),
            "task1_total_pixels": (
                task1_total_pixels
            ),
            "task1_lesion_area_ratio": (
                task1_area_ratio
            ),
            "task1_empty_mask": int(
                task1_foreground_pixels == 0
            ),
            "task1_full_mask": int(
                task1_foreground_pixels
                == task1_total_pixels
            ),
        }

        attribute_count = 0

        for attribute in ATTRIBUTES:
            mask_column = ATTRIBUTE_MASK_COLUMNS[
                attribute
            ]

            attribute_mask_path = resolve_dataset_path(
                value=row[mask_column],
                description=(
                    f"Task 2 '{attribute}' mask"
                ),
                image_id=image_id,
            )

            attribute_mask = load_binary_mask(
                path=attribute_mask_path,
                description=(
                    f"Task 2 '{attribute}' mask"
                ),
                image_id=image_id,
            )

            if (
                attribute_mask.shape
                != task1_mask.shape
            ):
                raise ValueError(
                    f"Mask size mismatch for image ID "
                    f"'{image_id}' and attribute "
                    f"'{attribute}'.\n"
                    f"Task 1 mask shape: "
                    f"{task1_mask.shape}\n"
                    f"Task 2 mask shape: "
                    f"{attribute_mask.shape}"
                )

            foreground_pixels = int(
                attribute_mask.sum()
            )

            is_present = int(
                foreground_pixels > 0
            )

            record[
                f"{attribute}_present"
            ] = is_present

            record[
                f"{attribute}_foreground_pixels"
            ] = foreground_pixels

            record[
                f"{attribute}_area_ratio"
            ] = (
                foreground_pixels
                / attribute_mask.size
            )

            attribute_count += is_present

        record["attribute_count"] = (
            attribute_count
        )

        record["no_attribute_present"] = int(
            attribute_count == 0
        )

        record[
            "multiple_attributes_present"
        ] = int(
            attribute_count >= 2
        )

        records.append(
            record
        )

        completed = row_number + 1

        if (
            completed == 1
            or completed % 250 == 0
            or completed == total_samples
        ):
            print(
                f"Processed {completed}/{total_samples} samples."
            )

    sample_df = pd.DataFrame(
        records
    )

    return sample_df


# ============================================================
# Stratification target construction
# ============================================================

def add_lesion_area_bins(
    sample_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Divide Task 1 lesion-area ratios into approximately equal bins.

    Ranking before qcut makes the operation robust when many masks
    have exactly the same foreground ratio.
    """
    result = sample_df.copy()

    number_of_bins = min(
        NUMBER_OF_LESION_AREA_BINS,
        len(result),
    )

    ranked_area = result[
        "task1_lesion_area_ratio"
    ].rank(
        method="first"
    )

    result["task1_area_bin"] = pd.qcut(
        ranked_area,
        q=number_of_bins,
        labels=False,
    ).astype(int)

    area_label_columns: list[str] = []

    for area_bin in range(
        number_of_bins
    ):
        column = (
            f"task1_area_bin_{area_bin}"
        )

        result[column] = (
            result["task1_area_bin"]
            == area_bin
        ).astype(int)

        area_label_columns.append(
            column
        )

    return result, area_label_columns


def build_stratification_targets(
    sample_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Construct the multilabel target matrix used for fold creation.
    """
    result, area_label_columns = (
        add_lesion_area_bins(
            sample_df
        )
    )

    attribute_label_columns = [
        f"{attribute}_present"
        for attribute in ATTRIBUTES
    ]

    additional_label_columns = [
        "no_attribute_present",
        "multiple_attributes_present",
    ]

    target_columns = [
        *attribute_label_columns,
        *additional_label_columns,
        *area_label_columns,
    ]

    target_matrix = result[
        target_columns
    ].to_numpy(
        dtype=np.int64
    )

    if not np.isin(
        target_matrix,
        [0, 1],
    ).all():
        raise ValueError(
            "The stratification target matrix must contain "
            "only binary values."
        )

    rows_without_labels = np.where(
        target_matrix.sum(axis=1) == 0
    )[0]

    if len(rows_without_labels) > 0:
        raise ValueError(
            "Some samples have no stratification labels. "
            f"Row indices: "
            f"{rows_without_labels[:10].tolist()}"
        )

    return result, target_columns


# ============================================================
# Duplicate-aware grouping
# ============================================================

def build_image_group_table(
    sample_df: pd.DataFrame,
    target_columns: list[str],
) -> tuple[
    pd.DataFrame,
    dict[str, str],
]:
    """
    Collapse exact duplicate Task 1 images into groups.

    Every image in the same SHA-256 group will later receive the
    same fold, preventing exact image copies from appearing in
    both training and validation data.
    """
    grouped_rows: list[dict[str, object]] = []

    unique_hashes = sorted(
        sample_df["image_sha256"]
        .unique()
        .tolist()
    )

    hash_to_group_id = {
        image_hash: (
            f"image_group_{index:06d}"
        )
        for index, image_hash in enumerate(
            unique_hashes,
            start=1,
        )
    }

    inconsistent_duplicate_groups = 0

    for image_hash, group in sample_df.groupby(
        "image_sha256",
        sort=True,
    ):
        group_record: dict[str, object] = {
            "image_sha256": image_hash,
            "duplicate_group_id": (
                hash_to_group_id[image_hash]
            ),
            "group_size": int(
                len(group)
            ),
            "representative_image_id": str(
                group.iloc[0]["image_id"]
            ),
            "mean_task1_lesion_area_ratio": float(
                group[
                    "task1_lesion_area_ratio"
                ].mean()
            ),
        }

        has_inconsistent_labels = False

        for column in target_columns:
            unique_values = group[
                column
            ].unique()

            if len(unique_values) > 1:
                has_inconsistent_labels = True

            # Use the union of labels for a duplicate group.
            group_record[column] = int(
                group[column].max()
            )

        if (
            len(group) > 1
            and has_inconsistent_labels
        ):
            inconsistent_duplicate_groups += 1

        grouped_rows.append(
            group_record
        )

    group_df = pd.DataFrame(
        grouped_rows
    )

    duplicate_group_count = int(
        (
            group_df["group_size"] > 1
        ).sum()
    )

    duplicate_sample_count = int(
        group_df.loc[
            group_df["group_size"] > 1,
            "group_size",
        ].sum()
    )

    print(
        "\nDuplicate-aware grouping:"
    )
    print(
        f"Unique image groups: {len(group_df)}"
    )
    print(
        "Groups containing exact duplicates: "
        f"{duplicate_group_count}"
    )
    print(
        "Samples belonging to duplicate groups: "
        f"{duplicate_sample_count}"
    )
    print(
        "Duplicate groups with inconsistent labels: "
        f"{inconsistent_duplicate_groups}"
    )

    return group_df, hash_to_group_id


# ============================================================
# Fold assignment
# ============================================================

def create_fold_assignments(
    sample_df: pd.DataFrame,
    group_df: pd.DataFrame,
    target_columns: list[str],
    hash_to_group_id: dict[str, str],
) -> pd.DataFrame:
    """
    Create duplicate-aware multilabel-stratified folds.
    """
    if len(group_df) < NUMBER_OF_FOLDS:
        raise ValueError(
            f"Only {len(group_df)} unique image groups were "
            f"found, but {NUMBER_OF_FOLDS} folds were requested."
        )

    group_targets = group_df[
        target_columns
    ].to_numpy(
        dtype=np.int64
    )

    print(
        "\nStratification label counts at group level:"
    )

    for column_index, column in enumerate(
        target_columns
    ):
        positive_count = int(
            group_targets[
                :,
                column_index,
            ].sum()
        )

        warning = ""

        if positive_count < NUMBER_OF_FOLDS:
            warning = (
                "  [WARNING: fewer positives than folds]"
            )

        print(
            f"  {column}: "
            f"{positive_count}{warning}"
        )

    splitter = MultilabelStratifiedKFold(
        n_splits=NUMBER_OF_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    placeholder_features = np.zeros(
        shape=(
            len(group_df),
            1,
        ),
        dtype=np.float32,
    )

    group_fold = np.full(
        shape=len(group_df),
        fill_value=-1,
        dtype=np.int64,
    )

    for fold, (
        _,
        validation_indices,
    ) in enumerate(
        splitter.split(
            placeholder_features,
            group_targets,
        )
    ):
        group_fold[
            validation_indices
        ] = fold

    if (
        group_fold < 0
    ).any():
        missing_indices = np.where(
            group_fold < 0
        )[0]

        raise RuntimeError(
            "Some image groups were not assigned to a fold. "
            f"Indices: {missing_indices[:10].tolist()}"
        )

    group_df = group_df.copy()
    group_df["fold"] = group_fold

    hash_to_fold = dict(
        zip(
            group_df["image_sha256"],
            group_df["fold"],
        )
    )

    hash_to_group_size = dict(
        zip(
            group_df["image_sha256"],
            group_df["group_size"],
        )
    )

    assignments = sample_df.copy()

    assignments[
        "duplicate_group_id"
    ] = assignments[
        "image_sha256"
    ].map(
        hash_to_group_id
    )

    assignments[
        "duplicate_group_size"
    ] = assignments[
        "image_sha256"
    ].map(
        hash_to_group_size
    ).astype(int)

    assignments["fold"] = assignments[
        "image_sha256"
    ].map(
        hash_to_fold
    )

    if assignments["fold"].isna().any():
        missing_ids = assignments.loc[
            assignments["fold"].isna(),
            "image_id",
        ].head(
            10
        ).tolist()

        raise RuntimeError(
            "Some samples could not be mapped back from "
            "image groups to folds. "
            f"Examples: {missing_ids}"
        )

    assignments["fold"] = assignments[
        "fold"
    ].astype(int)

    assignments[
        "development_role"
    ] = np.where(
        assignments["fold"]
        == DEVELOPMENT_VALIDATION_FOLD,
        "val",
        "train",
    )

    assignments = assignments.sort_values(
        by="image_id",
        kind="stable",
    ).reset_index(
        drop=True
    )

    return assignments


# ============================================================
# Fold validation
# ============================================================

def validate_fold_assignments(
    assignments: pd.DataFrame,
    expected_image_ids: set[str],
) -> None:
    """Run consistency checks on the completed fold assignment."""
    print(
        "\n"
        + "=" * 70
    )
    print(
        "VALIDATING FOLD ASSIGNMENTS"
    )
    print(
        "=" * 70
    )

    assigned_ids = set(
        assignments[
            "image_id"
        ].astype(str)
    )

    missing_ids = (
        expected_image_ids
        - assigned_ids
    )

    unexpected_ids = (
        assigned_ids
        - expected_image_ids
    )

    if missing_ids:
        raise RuntimeError(
            "Some index.csv IDs are absent from the fold file. "
            f"Examples: {sorted(missing_ids)[:10]}"
        )

    if unexpected_ids:
        raise RuntimeError(
            "The fold file contains IDs not found in index.csv. "
            f"Examples: {sorted(unexpected_ids)[:10]}"
        )

    if assignments[
        "image_id"
    ].duplicated().any():
        duplicate_ids = assignments.loc[
            assignments[
                "image_id"
            ].duplicated(
                keep=False
            ),
            "image_id",
        ].head(
            10
        ).tolist()

        raise RuntimeError(
            "The fold assignment contains duplicate image IDs. "
            f"Examples: {duplicate_ids}"
        )

    expected_folds = set(
        range(
            NUMBER_OF_FOLDS
        )
    )

    observed_folds = set(
        assignments["fold"]
        .unique()
        .tolist()
    )

    if observed_folds != expected_folds:
        raise RuntimeError(
            "Unexpected fold values.\n"
            f"Expected: {sorted(expected_folds)}\n"
            f"Observed: {sorted(observed_folds)}"
        )

    duplicate_fold_counts = (
        assignments.groupby(
            "duplicate_group_id"
        )["fold"]
        .nunique()
    )

    leaking_groups = duplicate_fold_counts[
        duplicate_fold_counts > 1
    ]

    if not leaking_groups.empty:
        raise RuntimeError(
            "Exact duplicate images were assigned to more "
            "than one fold. "
            f"Examples: "
            f"{leaking_groups.index[:10].tolist()}"
        )

    train_ids = set(
        assignments.loc[
            assignments["fold"]
            != DEVELOPMENT_VALIDATION_FOLD,
            "image_id",
        ]
    )

    val_ids = set(
        assignments.loc[
            assignments["fold"]
            == DEVELOPMENT_VALIDATION_FOLD,
            "image_id",
        ]
    )

    overlap = (
        train_ids
        & val_ids
    )

    if overlap:
        raise RuntimeError(
            "Development training and validation IDs overlap. "
            f"Examples: {sorted(overlap)[:10]}"
        )

    print(
        f"Expected sample count: "
        f"{len(expected_image_ids)}"
    )
    print(
        f"Assigned sample count: "
        f"{len(assignments)}"
    )
    print(
        f"Train samples: "
        f"{len(train_ids)}"
    )
    print(
        f"Validation samples: "
        f"{len(val_ids)}"
    )
    print(
        f"Train/validation overlap: "
        f"{len(overlap)}"
    )
    print(
        "Duplicate groups spanning multiple folds: "
        f"{len(leaking_groups)}"
    )
    print(
        "Fold assignment validation passed."
    )


# ============================================================
# Summary construction
# ============================================================

def build_fold_summary(
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Create one summary row for each fold."""
    summary_rows: list[dict[str, object]] = []

    total_samples = len(
        assignments
    )

    for fold in range(
        NUMBER_OF_FOLDS
    ):
        fold_data = assignments[
            assignments["fold"] == fold
        ]

        row: dict[str, object] = {
            "fold": fold,
            "sample_count": int(
                len(fold_data)
            ),
            "sample_fraction": (
                len(fold_data)
                / total_samples
            ),
            "unique_image_group_count": int(
                fold_data[
                    "duplicate_group_id"
                ].nunique()
            ),
            "mean_task1_lesion_area_ratio": float(
                fold_data[
                    "task1_lesion_area_ratio"
                ].mean()
            ),
            "median_task1_lesion_area_ratio": float(
                fold_data[
                    "task1_lesion_area_ratio"
                ].median()
            ),
            "empty_task1_mask_count": int(
                fold_data[
                    "task1_empty_mask"
                ].sum()
            ),
            "full_task1_mask_count": int(
                fold_data[
                    "task1_full_mask"
                ].sum()
            ),
            "no_attribute_count": int(
                fold_data[
                    "no_attribute_present"
                ].sum()
            ),
            "multiple_attribute_count": int(
                fold_data[
                    "multiple_attributes_present"
                ].sum()
            ),
        }

        for attribute in ATTRIBUTES:
            presence_column = (
                f"{attribute}_present"
            )

            positive_count = int(
                fold_data[
                    presence_column
                ].sum()
            )

            prevalence = (
                positive_count
                / len(fold_data)
                if len(fold_data) > 0
                else 0.0
            )

            row[
                f"{attribute}_positive_count"
            ] = positive_count

            row[
                f"{attribute}_prevalence"
            ] = prevalence

        area_bins = sorted(
            assignments[
                "task1_area_bin"
            ].unique()
            .tolist()
        )

        for area_bin in area_bins:
            row[
                f"task1_area_bin_{area_bin}_count"
            ] = int(
                (
                    fold_data[
                        "task1_area_bin"
                    ]
                    == area_bin
                ).sum()
            )

        summary_rows.append(
            row
        )

    return pd.DataFrame(
        summary_rows
    )


def format_percentage(
    value: float,
) -> str:
    """Format a proportion as a percentage."""
    return f"{100.0 * value:.2f}%"


def write_text_summary(
    assignments: pd.DataFrame,
    summary_df: pd.DataFrame,
    target_columns: list[str],
) -> None:
    """Write a human-readable fold summary."""
    lines: list[str] = []

    lines.append(
        "TASK 1 + TASK 2 FOLD SUMMARY"
    )
    lines.append(
        "=" * 70
    )
    lines.append(
        f"Number of folds: {NUMBER_OF_FOLDS}"
    )
    lines.append(
        f"Random seed: {RANDOM_SEED}"
    )
    lines.append(
        "Development validation fold: "
        f"{DEVELOPMENT_VALIDATION_FOLD}"
    )
    lines.append(
        f"Total samples: {len(assignments)}"
    )
    lines.append(
        "Unique image groups: "
        f"{assignments['duplicate_group_id'].nunique()}"
    )

    duplicate_groups = assignments.loc[
        assignments[
            "duplicate_group_size"
        ] > 1,
        "duplicate_group_id",
    ].nunique()

    lines.append(
        "Exact-duplicate image groups: "
        f"{duplicate_groups}"
    )
    lines.append(
        ""
    )

    lines.append(
        "Stratification labels:"
    )

    for column in target_columns:
        lines.append(
            f"  - {column}"
        )

    lines.append(
        ""
    )

    for _, row in summary_df.iterrows():
        fold = int(
            row["fold"]
        )

        lines.append(
            "-" * 70
        )
        lines.append(
            f"Fold {fold}"
        )
        lines.append(
            "-" * 70
        )
        lines.append(
            "Samples: "
            f"{int(row['sample_count'])} "
            f"({format_percentage(float(row['sample_fraction']))})"
        )
        lines.append(
            "Unique image groups: "
            f"{int(row['unique_image_group_count'])}"
        )
        lines.append(
            "Mean lesion area ratio: "
            f"{float(row['mean_task1_lesion_area_ratio']):.6f}"
        )
        lines.append(
            "Median lesion area ratio: "
            f"{float(row['median_task1_lesion_area_ratio']):.6f}"
        )
        lines.append(
            "Images without positive Task 2 attributes: "
            f"{int(row['no_attribute_count'])}"
        )
        lines.append(
            "Images with multiple Task 2 attributes: "
            f"{int(row['multiple_attribute_count'])}"
        )

        lines.append(
            "Attribute prevalence:"
        )

        for attribute in ATTRIBUTES:
            count = int(
                row[
                    f"{attribute}_positive_count"
                ]
            )

            prevalence = float(
                row[
                    f"{attribute}_prevalence"
                ]
            )

            lines.append(
                f"  {attribute}: "
                f"{count} "
                f"({format_percentage(prevalence)})"
            )

        lines.append(
            ""
        )

    FOLD_SUMMARY_TXT.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


def write_id_file(
    path: Path,
    image_ids: list[str],
) -> None:
    """Write one image ID per line."""
    content = "\n".join(
        image_ids
    )

    if content:
        content += "\n"

    path.write_text(
        content,
        encoding="utf-8",
    )


def write_metadata(
    assignments: pd.DataFrame,
    target_columns: list[str],
) -> None:
    """Write split configuration and provenance metadata."""
    metadata = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "index_csv": str(
            INDEX_CSV
        ),
        "fold_output_csv": str(
            FOLD_OUTPUT_CSV
        ),
        "number_of_folds": (
            NUMBER_OF_FOLDS
        ),
        "development_validation_fold": (
            DEVELOPMENT_VALIDATION_FOLD
        ),
        "random_seed": int(
            RANDOM_SEED
        ),
        "splitter": (
            "MultilabelStratifiedKFold"
        ),
        "duplicate_handling": (
            "Task 1 images with identical SHA-256 hashes "
            "are assigned to the same fold."
        ),
        "total_samples": int(
            len(assignments)
        ),
        "unique_image_groups": int(
            assignments[
                "duplicate_group_id"
            ].nunique()
        ),
        "attribute_names": list(
            ATTRIBUTES
        ),
        "stratification_targets": (
            target_columns
        ),
        "development_train_count": int(
            (
                assignments["fold"]
                != DEVELOPMENT_VALIDATION_FOLD
            ).sum()
        ),
        "development_validation_count": int(
            (
                assignments["fold"]
                == DEVELOPMENT_VALIDATION_FOLD
            ).sum()
        ),
        "output_files": {
            "fold_assignments": str(
                FOLD_OUTPUT_CSV
            ),
            "fold_summary_csv": str(
                FOLD_SUMMARY_CSV
            ),
            "fold_summary_txt": str(
                FOLD_SUMMARY_TXT
            ),
            "train_ids": str(
                TRAIN_IDS_TXT
            ),
            "validation_ids": str(
                VAL_IDS_TXT
            ),
        },
    }

    with SPLIT_METADATA_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Output writing
# ============================================================

def save_split_outputs(
    assignments: pd.DataFrame,
    summary_df: pd.DataFrame,
    target_columns: list[str],
) -> None:
    """Save all fold and development-split outputs."""
    SPLIT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = [
        "image_id",
        "fold",
        "development_role",
        "duplicate_group_id",
        "duplicate_group_size",
        "image_sha256",
        "task1_lesion_area_ratio",
        "task1_area_bin",
        "task1_empty_mask",
        "task1_full_mask",
        "attribute_count",
        "no_attribute_present",
        "multiple_attributes_present",
    ]

    for attribute in ATTRIBUTES:
        output_columns.extend(
            [
                f"{attribute}_present",
                f"{attribute}_foreground_pixels",
                f"{attribute}_area_ratio",
            ]
        )

    area_label_columns = [
        column
        for column in target_columns
        if column.startswith(
            "task1_area_bin_"
        )
    ]

    for column in area_label_columns:
        if column not in output_columns:
            output_columns.append(
                column
            )

    assignments[
        output_columns
    ].to_csv(
        FOLD_OUTPUT_CSV,
        index=False,
        encoding="utf-8",
    )

    summary_df.to_csv(
        FOLD_SUMMARY_CSV,
        index=False,
        encoding="utf-8",
    )

    train_ids = (
        assignments.loc[
            assignments["fold"]
            != DEVELOPMENT_VALIDATION_FOLD,
            "image_id",
        ]
        .astype(str)
        .tolist()
    )

    val_ids = (
        assignments.loc[
            assignments["fold"]
            == DEVELOPMENT_VALIDATION_FOLD,
            "image_id",
        ]
        .astype(str)
        .tolist()
    )

    write_id_file(
        TRAIN_IDS_TXT,
        train_ids,
    )

    write_id_file(
        VAL_IDS_TXT,
        val_ids,
    )

    write_text_summary(
        assignments=assignments,
        summary_df=summary_df,
        target_columns=target_columns,
    )

    write_metadata(
        assignments=assignments,
        target_columns=target_columns,
    )


# ============================================================
# Main pipeline
# ============================================================

def create_task1_task2_folds() -> None:
    """Create reproducible Task 1 and Task 2 fold assignments."""
    print(
        "=" * 70
    )
    print(
        "STEP 11: TASK 1 + TASK 2 FOLD CREATION"
    )
    print(
        "=" * 70
    )
    print(
        f"Project root: {PROJECT_ROOT}"
    )
    print(
        f"Index CSV: {INDEX_CSV}"
    )
    print(
        f"Number of folds: {NUMBER_OF_FOLDS}"
    )
    print(
        f"Random seed: {RANDOM_SEED}"
    )
    print(
        "Development validation fold: "
        f"{DEVELOPMENT_VALIDATION_FOLD}"
    )

    index_df = load_index_csv()

    print(
        f"\nLoaded {len(index_df)} rows from index.csv."
    )

    sample_df = inspect_samples(
        index_df
    )

    (
        sample_df,
        target_columns,
    ) = build_stratification_targets(
        sample_df
    )

    (
        group_df,
        hash_to_group_id,
    ) = build_image_group_table(
        sample_df=sample_df,
        target_columns=target_columns,
    )

    assignments = create_fold_assignments(
        sample_df=sample_df,
        group_df=group_df,
        target_columns=target_columns,
        hash_to_group_id=hash_to_group_id,
    )

    expected_image_ids = set(
        index_df[
            IMAGE_ID_COLUMN
        ].astype(str)
    )

    validate_fold_assignments(
        assignments=assignments,
        expected_image_ids=expected_image_ids,
    )

    summary_df = build_fold_summary(
        assignments
    )

    save_split_outputs(
        assignments=assignments,
        summary_df=summary_df,
        target_columns=target_columns,
    )

    print(
        "\n"
        + "=" * 70
    )
    print(
        "FOLD CREATION COMPLETED"
    )
    print(
        "=" * 70
    )

    print(
        f"Fold assignments:\n"
        f"  {FOLD_OUTPUT_CSV}"
    )

    print(
        f"Fold summary CSV:\n"
        f"  {FOLD_SUMMARY_CSV}"
    )

    print(
        f"Fold summary text:\n"
        f"  {FOLD_SUMMARY_TXT}"
    )

    print(
        f"Split metadata:\n"
        f"  {SPLIT_METADATA_JSON}"
    )

    print(
        f"Development training IDs:\n"
        f"  {TRAIN_IDS_TXT}"
    )

    print(
        f"Development validation IDs:\n"
        f"  {VAL_IDS_TXT}"
    )

    print(
        "\nFold sample counts:"
    )

    for _, row in summary_df.iterrows():
        print(
            f"  Fold {int(row['fold'])}: "
            f"{int(row['sample_count'])} samples"
        )

    train_count = int(
        (
            assignments["fold"]
            != DEVELOPMENT_VALIDATION_FOLD
        ).sum()
    )

    val_count = int(
        (
            assignments["fold"]
            == DEVELOPMENT_VALIDATION_FOLD
        ).sum()
    )

    print(
        "\nDevelopment split:"
    )
    print(
        f"  Train: {train_count}"
    )
    print(
        f"  Validation: {val_count}"
    )


if __name__ == "__main__":
    create_task1_task2_folds()