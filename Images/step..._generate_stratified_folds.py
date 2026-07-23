from pathlib import Path
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

here = Path(__file__).resolve().parent
for candidate in [here, *here.parents]:
    if (candidate / "data").is_dir():
        PROJECT = candidate
        break
else:
    raise SystemExit(f"No 'data' folder found above {here}")

ATTRIBUTES = ["pigment_network", "negative_network", "streaks",
              "milia_like_cysts", "globules"]

df = pd.read_csv(PROJECT / "labels.csv", dtype={"image_id": str})
X = df["image_id"].values

# --- build the stratification matrix ---
STRAT_COLS = ATTRIBUTES + ["no_attributes", "many_attributes",
                           "tiny_lesion", "huge_lesion"]

area_onehot = pd.get_dummies(df["area_bin"], prefix="area").astype(int)
y = pd.concat([df[STRAT_COLS], area_onehot], axis=1).values

print(f"{len(df)} images, stratifying on {y.shape[1]} binary labels\n")

SPLITS = PROJECT / "splits"
SPLITS.mkdir(exist_ok=True)

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- attribute rates per fold ---
print("Attribute presence rate in each validation fold:")
print(f"{'fold':<6} {'train':>6} {'val':>5}   " +
      "  ".join(f"{a[:9]:>9}" for a in ATTRIBUTES))

fold_val_idx = []

for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y)):
    fold_val_idx.append(val_idx)

    val_ids = sorted(X[val_idx])
    train_ids = sorted(X[train_idx])

    (SPLITS / f"fold{fold}_train.txt").write_text("\n".join(train_ids))
    (SPLITS / f"fold{fold}_val.txt").write_text("\n".join(val_ids))

    rates = df.iloc[val_idx][ATTRIBUTES].mean()
    print(f"{fold:<6} {len(train_ids):>6} {len(val_ids):>5}   " +
          "  ".join(f"{rates[a]:>8.1%}" for a in ATTRIBUTES))

print(f"\n{'overall':<6} {'':>6} {len(df):>5}   " +
      "  ".join(f"{df[a].mean():>8.1%}" for a in ATTRIBUTES))

# --- lesion size balance per fold ---
print("\n\nLesion area bin counts in each validation fold:")
print(f"{'fold':<6} " + "  ".join(f"{'bin'+str(b):>7}" for b in sorted(df['area_bin'].unique()))
      + f"  {'mean area':>10}")

for fold, val_idx in enumerate(fold_val_idx):
    counts = df.iloc[val_idx]["area_bin"].value_counts().sort_index()
    mean_area = df.iloc[val_idx]["lesion_area_ratio"].mean()
    print(f"{fold:<6} " +
          "  ".join(f"{counts.get(b, 0):>7}" for b in sorted(df['area_bin'].unique()))
          + f"  {mean_area:>9.3f}")

print(f"\n{'overall':<6} " +
      "  ".join(f"{(df['area_bin'] == b).sum():>7}" for b in sorted(df['area_bin'].unique()))
      + f"  {df['lesion_area_ratio'].mean():>9.3f}")

# --- sanity checks ---
all_val = sorted(i for idx in fold_val_idx for i in idx)
print(f"\nEach image held out exactly once: {all_val == list(range(len(df)))}")
print(f"Written to {SPLITS}")

for fold in range(5):
    tr = set((SPLITS / f"fold{fold}_train.txt").read_text().split())
    va = set((SPLITS / f"fold{fold}_val.txt").read_text().split())
    assert not (tr & va), f"fold {fold} leaks!"
print("No train/val overlap in any fold.")