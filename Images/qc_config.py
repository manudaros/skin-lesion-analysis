from pathlib import Path


# The scripts are stored in Biology-AI/Images/.
SCRIPT_DIR = Path(__file__).resolve().parent

# The project root is one directory above the Images folder.
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_ROOT = PROJECT_ROOT / "data"
INDEX_CSV = PROJECT_ROOT / "index.csv"

TASK1_IMAGE_DIR = (
    DATA_ROOT
    / "Task1_Segmentation"
    / "images"
)

TASK1_MASK_DIR = (
    DATA_ROOT
    / "Task1_Segmentation"
    / "masks"
)

TASK2_IMAGE_DIR = (
    DATA_ROOT
    / "Task2_Attributes"
    / "images"
)

TASK2_MASK_ROOT = (
    DATA_ROOT
    / "Task2_Attributes"
    / "masks"
)

ATTRIBUTES = [
    "pigment_network",
    "negative_network",
    "streaks",
    "milia_like_cysts",
    "globules",
]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "data_check"
)

INDEX_VALIDATION_REPORT = (
    OUTPUT_ROOT
    / "index_validation_report.csv"
)

READABILITY_REPORT = (
    OUTPUT_ROOT
    / "readability_report.csv"
)

DIMENSION_REPORT = (
    OUTPUT_ROOT
    / "dimension_report.csv"
)

TASK1_MASK_REPORT = (
    OUTPUT_ROOT
    / "task1_mask_quality.csv"
)

TASK2_MASK_REPORT = (
    OUTPUT_ROOT
    / "task2_mask_quality.csv"
)

DATASET_STATISTICS_REPORT = (
    OUTPUT_ROOT
    / "dataset_statistics.txt"
)

TASK2_ATTRIBUTE_SUMMARY_REPORT = (
    OUTPUT_ROOT
    / "task2_attribute_summary.csv"
)

DUPLICATE_ID_REPORT = (
    OUTPUT_ROOT
    / "duplicate_ids.csv"
)

DUPLICATE_IMAGE_REPORT = (
    OUTPUT_ROOT
    / "exact_duplicate_images.csv"
)

IMAGE_COPY_REPORT = (
    OUTPUT_ROOT
    / "task1_task2_image_comparison.csv"
)

SUMMARY_REPORT = (
    OUTPUT_ROOT
    / "qc_summary.txt"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

RANDOM_SEED = 42

# Task 1 masks below this ratio are flagged as very small.
SMALL_LESION_THRESHOLD = 0.005

# Task 1 masks above this ratio are flagged as very large.
LARGE_LESION_THRESHOLD = 0.95

# Attribute masks exceeding this outside-lesion fraction are flagged.
OUTSIDE_LESION_WARNING_THRESHOLD = 0.05

# Images below this pixel standard deviation may be nearly uniform.
NEAR_UNIFORM_IMAGE_STD_THRESHOLD = 1.0