from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

from qc_config import IMAGE_EXTENSIONS


def extract_image_id(path: Path) -> str:
    """
    Extract the shared image ID from an image or mask filename.

    Examples:
        ISIC_0000000.jpg
        -> ISIC_0000000

        ISIC_0000000_segmentation.png
        -> ISIC_0000000

        ISIC_0000000_attribute_globules.png
        -> ISIC_0000000
    """
    stem = path.stem

    if "_attribute_" in stem:
        return stem.split(
            "_attribute_",
            maxsplit=1,
        )[0]

    if stem.endswith("_segmentation"):
        return stem[
            : -len("_segmentation")
        ]

    return stem


def list_image_files(
    folder: Path,
) -> list[Path]:
    """
    Return all supported image files inside a directory.

    Recursive search is used so that files in unexpected
    subdirectories are still discovered.
    """
    if not folder.exists():
        return []

    files = [
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in IMAGE_EXTENSIONS
    ]

    return sorted(files)


def build_id_map(
    files: list[Path],
) -> dict[str, list[Path]]:
    """
    Group files by their extracted image ID.

    A list is stored for each ID so duplicate IDs can be detected.
    """
    id_map: dict[
        str,
        list[Path],
    ] = defaultdict(list)

    for path in files:
        sample_id = extract_image_id(
            path
        )
        id_map[sample_id].append(path)

    return dict(id_map)


def first_path(
    id_map: dict[str, list[Path]],
    sample_id: str,
) -> Path | None:
    """
    Return the first file associated with an image ID.

    Duplicate IDs are reported separately by the duplicate checker.
    """
    paths = id_map.get(
        sample_id,
        [],
    )

    if not paths:
        return None

    return sorted(paths)[0]


def read_image_size(
    path: Path,
) -> tuple[int, int]:
    """Read image width and height."""
    with Image.open(path) as image:
        return image.size


def load_rgb_array(
    path: Path,
) -> np.ndarray:
    """Load an image as an RGB NumPy array."""
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        return np.asarray(rgb)


def load_binary_mask(
    path: Path,
) -> np.ndarray:
    """
    Load a mask and convert every positive pixel to foreground.

    The returned array contains Boolean values.
    """
    with Image.open(path) as image:
        image.load()
        grayscale = np.asarray(
            image.convert("L")
        )

    return grayscale > 0


def calculate_sha256(
    path: Path,
) -> str:
    """Calculate a SHA-256 hash for exact duplicate detection."""
    hasher = sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            hasher.update(block)

    return hasher.hexdigest()


def combine_status(
    issues: list[str],
) -> str:
    """Convert a list of QC issues into one CSV value."""
    if not issues:
        return "OK"

    return ";".join(issues)