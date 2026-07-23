from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from qc_config import (
    ATTRIBUTES,
    DUPLICATE_IMAGE_REPORT,
    INDEX_CSV,
    PROJECT_ROOT,
    RANDOM_SEED,
    TASK1_MASK_REPORT,
    TASK2_MASK_REPORT,
)

NUMBER_OF_FOLDS = 5
DEVELOPMENT_VALIDATION_FOLD = 0

SPLIT_OUTPUT_DIR = PROJECT_ROOT / "splits"
FOLD_OUTPUT_CSV = SPLIT_OUTPUT_DIR / "task1_task2_folds.csv"
FOLD_SUMMARY_CSV = SPLIT_OUTPUT_DIR / "task1_task2_fold_summary.csv"
FOLD_SUMMARY_TXT = SPLIT_OUTPUT_DIR / "task1_task2_fold_summary.txt"
SPLIT_METADATA_JSON = SPLIT_OUTPUT_DIR / "task1_task2_split_metadata.json"

TASK1_BLOCKING_ISSUES = [
    "missing_mask", "unreadable_mask",
    "non_binary_values", "empty_lesion_mask",
]
TASK2_BLOCKING_ISSUES = [
    "missing_attribute_mask", "unreadable_attribute_mask",
    "non_binary_values", "full_attribute_mask",
    "lesion_attribute_size_mismatch",
]


def read_required_csv(path: Path, report_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{report_name} does not exist: {path}\n"
            "Run the QC pipeline (steps 1-7) before creating folds."
        )
    frame = pd.read_csv(path, dtype={"image_id": str})
    if frame.empty:
        raise ValueError(f"{report_name} contains no rows: {path}")
    return frame


def status_has_issue(status_series: pd.Series, issue: str) -> pd.Series:
    return status_series.fillna("").astype(str).str.contains(issue, regex=False)


def convert_to_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False}
    return (series.astype(str).str.strip().str.lower().map(mapping)
            .fillna(False).astype(bool))


# ---------- load and validate ----------

index_df = read_required_csv(INDEX_CSV, "index.csv")
task1_df = read_required_csv(TASK1_MASK_REPORT, "Task 1 mask report")
task2_df = read_required_csv(TASK2_MASK_REPORT, "Task 2 mask report")

for issue in TASK1_BLOCKING_ISSUES:
    if status_has_issue(task1_df["status"], issue).any():
        raise ValueError(
            f"Task 1 has blocking issue '{issue}'. Resolve before splitting.")
for issue in TASK2_BLOCKING_ISSUES:
    if status_has_issue(task2_df["status"], issue).any():
        raise ValueError(
            f"Task 2 has blocking issue '{issue}'. Resolve before splitting.")

# ---------- build sample table ----------

sample = index_df[["image_id"]].copy()
sample["image_id"] = sample["image_id"].astype(str)

t1 = task1_df[["image_id", "foreground_ratio",
               "touches_image_border", "status"]].copy()
t1["image_id"] = t1["image_id"].astype(str)
t1 = t1.rename(columns={"foreground_ratio": "lesion_area_ratio",
                        "status": "task1_status"})
t1["lesion_area_ratio"] = pd.to_numeric(
    t1["lesion_area_ratio"], errors="coerce")
t1["touches_image_border"] = convert_to_boolean(
    t1["touches_image_border"]).astype(int)
t1["very_small_lesion"] = status_has_issue(
    t1["task1_status"], "very_small_lesion").astype(int)
t1["very_large_lesion"] = status_has_issue(
    t1["task1_status"], "very_large_lesion").astype(int)

sample = sample.merge(t1, on="image_id", how="left", validate="one_to_one")

# quantile bins on ranked area (rank avoids ties collapsing bins)
area_rank = sample["lesion_area_ratio"].rank(method="first")
sample["area_bin"] = pd.qcut(
    area_rank, q=NUMBER_OF_FOLDS, labels=False).astype(int)

# attribute presence, pivoted to one column per attribute
t2 = task2_df[["image_id", "attribute", "foreground_pixels"]].copy()
t2["image_id"] = t2["image_id"].astype(str)
t2["foreground_pixels"] = pd.to_numeric(
    t2["foreground_pixels"], errors="coerce")
t2["present"] = t2["foreground_pixels"].gt(0).astype(int)

presence = (t2.pivot(index="image_id", columns="attribute", values="present")
            .reindex(columns=ATTRIBUTES).reset_index())
presence.columns = ["image_id"] + [f"{a}_present" for a in ATTRIBUTES]

sample = sample.merge(presence, on="image_id",
                      how="left", validate="one_to_one")

presence_cols = [f"{a}_present" for a in ATTRIBUTES]
sample[presence_cols] = sample[presence_cols].fillna(0).astype(int)

# ---------- build the stratification matrix ----------

area_onehot = pd.get_dummies(sample["area_bin"], prefix="area").astype(int)
strat_cols = presence_cols + ["very_small_lesion", "very_large_lesion",
                              "touches_image_border"]
y = pd.concat([sample[strat_cols].reset_index(drop=True),
               area_onehot.reset_index(drop=True)], axis=1)

# ---------- assign folds ----------

mskf = MultilabelStratifiedKFold(n_splits=NUMBER_OF_FOLDS, shuffle=True,
                                 random_state=RANDOM_SEED)
sample["fold"] = -1
for fold, (_, val_idx) in enumerate(mskf.split(sample["image_id"], y.values)):
    sample.loc[val_idx, "fold"] = fold

assert (sample["fold"] >= 0).all(), "some images were not assigned a fold"

sample["development_split"] = np.where(
    sample["fold"] == DEVELOPMENT_VALIDATION_FOLD, "val", "train")

# ---------- save ----------

SPLIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sample = sample.sort_values("image_id").reset_index(drop=True)
sample.to_csv(FOLD_OUTPUT_CSV, index=False)

# per-fold summary
rows = []
for fold in list(range(NUMBER_OF_FOLDS)) + ["ALL"]:
    sub = sample if fold == "ALL" else sample[sample["fold"] == fold]
    row = {"fold": fold, "n": len(sub),
           "lesion_area_mean": sub["lesion_area_ratio"].mean()}
    for a in ATTRIBUTES:
        row[f"{a}_rate"] = sub[f"{a}_present"].mean()
    rows.append(row)
summary = pd.DataFrame(rows)
summary.to_csv(FOLD_SUMMARY_CSV, index=False)
FOLD_SUMMARY_TXT.write_text(summary.to_string(index=False), encoding="utf-8")

SPLIT_METADATA_JSON.write_text(json.dumps({
    "n_images": len(sample),
    "n_folds": NUMBER_OF_FOLDS,
    "random_seed": RANDOM_SEED,
    "development_validation_fold": DEVELOPMENT_VALIDATION_FOLD,
    "method": "MultilabelStratifiedKFold",
    "stratified_on": strat_cols + list(area_onehot.columns),
}, indent=2), encoding="utf-8")

# ---------- report ----------

print(f"{len(sample)} images split into {NUMBER_OF_FOLDS} folds\n")
print(f"{'fold':>5} {'n':>6}  " + "  ".join(f"{a[:9]:>9}" for a in ATTRIBUTES))
for fold in range(NUMBER_OF_FOLDS):
    sub = sample[sample["fold"] == fold]
    print(f"{fold:>5} {len(sub):>6}  " +
          "  ".join(f"{sub[f'{a}_present'].mean():>8.1%}" for a in ATTRIBUTES))
print(f"{'all':>5} {len(sample):>6}  " +
      "  ".join(f"{sample[f'{a}_present'].mean():>8.1%}" for a in ATTRIBUTES))

print(f"\nwritten to {SPLIT_OUTPUT_DIR}")
