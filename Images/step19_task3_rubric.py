"""
Task 3 — shared rubric.

Every threshold and phrasing rule for the findings report lives here, so
the report generator and the visualiser cannot drift apart. Deliberately
free of torch: this is pure geometry and text, and can be tested on its
own.

All thresholds come from the project briefing:

  status   present if p >= 0.60, absent if p <= 0.40, uncertain between
  border   index = perimeter^2 / (4 * pi * area); irregular if >= 1.60;
           index is 0.0 when area is 0
  size     ratio = lesion_pixels / total_pixels
           small < 0.08, moderate 0.08-0.25, large > 0.25
"""

from __future__ import annotations

import math
import re

import cv2
import numpy as np

ATTRIBUTE_PHRASING = {
    "pigment_network": ("pigment network", "is"),
    "negative_network": ("negative network", "is"),
    "streaks": ("streaks", "are"),
    "milia_like_cysts": ("milia-like cysts", "are"),
    "globules": ("globules", "are"),
}

STATUS_PRESENT_AT = 0.60
STATUS_ABSENT_AT = 0.40
BORDER_IRREGULAR_AT = 1.60
SIZE_SMALL_BELOW = 0.08
SIZE_LARGE_ABOVE = 0.25

# An attribute called present should have some visible mask behind it.
MIN_EVIDENCE_PIXELS = 20


# =====================================================================
# 1. Categories
# =====================================================================

def size_category(lesion_pixels: int, total_pixels: int) -> tuple[str, float]:
    """Return (category, ratio) from the predicted lesion mask."""
    ratio = lesion_pixels / max(total_pixels, 1)
    if ratio < SIZE_SMALL_BELOW:
        return "small", ratio
    if ratio <= SIZE_LARGE_ABOVE:
        return "moderate", ratio
    return "large", ratio


def border_category(area: float, perimeter: float) -> tuple[str, float]:
    """
    Return (category, irregularity index).

    The briefing sets the index to 0.0 when area is 0, which falls below
    the threshold and therefore reads as regular.
    """
    if area <= 0:
        return "regular", 0.0
    index = (perimeter ** 2) / (4.0 * math.pi * area)
    category = "irregular" if index >= BORDER_IRREGULAR_AT else "regular"
    return category, float(index)


def attribute_status(probability: float) -> str:
    if probability >= STATUS_PRESENT_AT:
        return "present"
    if probability <= STATUS_ABSENT_AT:
        return "absent"
    return "uncertain"


# =====================================================================
# 2. Geometry from the Task 1 mask
# =====================================================================

def lesion_geometry(mask: np.ndarray, smooth: bool = True) -> dict:
    """
    Measure area, perimeter and shape from a binary lesion mask.

    Area for the size ratio is the pixel count of the whole mask, which
    is how the briefing defines lesion_area_ratio.

    The shape index instead uses the largest connected component, so the
    perimeter and the area it is compared against describe the same
    object. Reported separately to keep that explicit.

    A raw pixel boundary is a staircase, and its perimeter is inflated by
    roughly a factor of 4/pi for a smooth shape — enough to push almost
    everything over the 1.60 threshold. Polygon approximation removes
    most of that. Both indices are returned so the effect is visible.
    """
    mask_uint8 = (mask > 0.5).astype(np.uint8)
    lesion_pixels = int(mask_uint8.sum())

    result = {
        "lesion_pixels": lesion_pixels,
        "component_area": 0.0,
        "perimeter_raw": 0.0,
        "perimeter_smoothed": 0.0,
        "border_index_raw": 0.0,
        "border_index_smoothed": 0.0,
        "n_components": 0,
    }

    if lesion_pixels == 0:
        return result

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    result["n_components"] = len(contours)
    if not contours:
        return result

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    perimeter_raw = float(cv2.arcLength(largest, True))

    epsilon = 0.01 * perimeter_raw
    approximated = cv2.approxPolyDP(largest, epsilon, True)
    perimeter_smoothed = float(cv2.arcLength(approximated, True))

    result["component_area"] = area
    result["perimeter_raw"] = perimeter_raw
    result["perimeter_smoothed"] = perimeter_smoothed
    result["border_index_raw"] = border_category(area, perimeter_raw)[1]
    result["border_index_smoothed"] = border_category(
        area, perimeter_smoothed)[1]

    return result


# =====================================================================
# 3. Report text
# =====================================================================

def build_report_text(size: str, border: str, statuses: dict) -> str:
    """
    Produce the controlled sentence pair from the briefing.

    Example: "The lesion is moderate with irregular borders. Pigment
    network is present; negative network is absent; streaks are
    uncertain; milia-like cysts are absent; globules are present."
    """
    clauses = []
    for key, (name, verb) in ATTRIBUTE_PHRASING.items():
        clauses.append(f"{name} {verb} {statuses[key]}")

    clauses[0] = clauses[0][0].upper() + clauses[0][1:]
    return (f"The lesion is {size} with {border} borders. "
            + "; ".join(clauses) + ".")


# =====================================================================
# 4. Consistency audit
# =====================================================================
# The briefing scores report consistency on three conditions: all five
# terms appear, the text agrees with the JSON, and the claims are backed
# by predicted masks. Checking them here means a failure is a reported
# number rather than something a marker finds first.

def check_report(report: dict, text: str,
                 evidence_pixels: dict | None = None) -> dict:
    """
    Verify one report. Returns a dict of boolean checks plus details.

    evidence_pixels maps attribute -> predicted mask pixels inside the
    lesion ROI. Omit it to skip the evidence check.
    """
    presence = report["outputs"]["presence"]
    lowered = text.lower()

    missing_terms = [name for name, _ in ATTRIBUTE_PHRASING.values()
                     if name not in lowered]

    mismatched = []
    for key, (name, verb) in ATTRIBUTE_PHRASING.items():
        expected = f"{name} {verb} {presence[key]['status']}"
        if expected.lower() not in lowered:
            mismatched.append(key)

    unsupported = []
    if evidence_pixels is not None:
        for key, entry in presence.items():
            if entry["status"] == "present" and \
                    evidence_pixels.get(key, 0) < MIN_EVIDENCE_PIXELS:
                unsupported.append(key)

    out_of_range = [key for key, entry in presence.items()
                    if not 0.0 <= entry["prob"] <= 1.0]

    size_word = re.search(r"the lesion is (\w+)", lowered)
    border_word = re.search(r"with (\w+) borders", lowered)

    checks = {
        "all_terms_present": not missing_terms,
        "statuses_match_json": not mismatched,
        "evidence_supports_claims": not unsupported,
        "probabilities_in_range": not out_of_range,
        "schema_fields_present": all(
            field in report for field in
            ("image_id", "split", "model_version",
             "attributes_order", "outputs")),
        "size_stated": size_word is not None,
        "border_stated": border_word is not None,
    }
    checks["passed"] = all(checks.values())
    checks["missing_terms"] = missing_terms
    checks["mismatched_attributes"] = mismatched
    checks["unsupported_claims"] = unsupported
    return checks


def summarise_audit(rows: list[dict], expected_count: int | None = None
                    ) -> dict:
    """Dataset-level sanity: completeness, duplicates, failure counts."""
    image_ids = [row["image_id"] for row in rows]
    duplicates = sorted({i for i in image_ids if image_ids.count(i) > 1})

    def failures(field: str) -> int:
        return sum(1 for row in rows if not row["checks"][field])

    summary = {
        "reports_generated": len(rows),
        "expected_reports": expected_count,
        "all_images_covered": (expected_count is None
                               or len(rows) == expected_count),
        "duplicate_image_ids": duplicates,
        "no_duplicate_ids": not duplicates,
        "failed_all_terms_present": failures("all_terms_present"),
        "failed_statuses_match_json": failures("statuses_match_json"),
        "failed_evidence_supports_claims": failures(
            "evidence_supports_claims"),
        "failed_probabilities_in_range": failures("probabilities_in_range"),
        "reports_fully_passing": sum(1 for row in rows
                                     if row["checks"]["passed"]),
    }
    summary["pass_rate"] = (summary["reports_fully_passing"]
                            / max(len(rows), 1))
    return summary
