from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

here = Path(__file__).resolve().parent
for candidate in [here, *here.parents]:
    if (candidate / "data").is_dir():
        PROJECT = candidate
        break

DATA = PROJECT / "data"
T1_MASK = DATA / "Task1_Segmentation" / "masks"
T2_MASK = DATA / "Task2_Attributes" / "masks"

ATTRIBUTES = ["pigment_network", "negative_network", "streaks",
              "milia_like_cysts", "globules"]

ids = sorted(f.stem for f in
             (DATA / "Task1_Segmentation" / "images").glob("*.jpg"))
index_of = {img_id: i for i, img_id in enumerate(ids)}
print(f"{len(ids)} images")

# --- attribute presence ---
table = {"image_id": ids}
for attr in ATTRIBUTES:
    table[attr] = [0] * len(ids)

for attr in ATTRIBUTES:
    print(f"scanning {attr}...")
    for path in sorted((T2_MASK / attr).glob("*.png")):
        img_id = path.stem.split("_")[0]
        if img_id not in index_of:
            continue
        mask = np.array(Image.open(path).convert("L")) > 127
        if mask.any():
            table[attr][index_of[img_id]] = 1

# --- lesion area ratio from Task 1 masks ---
print("scanning lesion masks...")
table["lesion_area_ratio"] = [0.0] * len(ids)
for path in sorted(T1_MASK.glob("*.png")):
    img_id = path.stem.split("_")[0]
    if img_id not in index_of:
        continue
    mask = np.array(Image.open(path).convert("L")) > 127
    table["lesion_area_ratio"][index_of[img_id]] = float(mask.mean())

df = pd.DataFrame(table)

# --- derived stratification columns ---
df["area_bin"] = pd.qcut(df["lesion_area_ratio"], q=5,
                         labels=False, duplicates="drop")

df["n_attributes"] = df[ATTRIBUTES].sum(axis=1)
df["no_attributes"] = (df["n_attributes"] == 0).astype(int)
df["many_attributes"] = (df["n_attributes"] >= 3).astype(int)
df["tiny_lesion"] = (df["lesion_area_ratio"] < 0.02).astype(int)
df["huge_lesion"] = (df["lesion_area_ratio"] > 0.50).astype(int)

out = PROJECT / "labels.csv"
df.to_csv(out, index=False)
print(f"\nwritten to {out}\n")

print(df[ATTRIBUTES].sum())
print("\narea_bin counts:")
print(df["area_bin"].value_counts().sort_index())
print("\nindicators:")
for col in ["no_attributes", "many_attributes", "tiny_lesion", "huge_lesion"]:
    print(f"  {col:<16} {df[col].sum():>5}  ({df[col].mean():.1%})")
print("\nlesion_area_ratio:")
print(df["lesion_area_ratio"].describe())