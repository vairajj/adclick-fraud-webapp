
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train_sample.csv"
OUTPUT_DIR = BASE_DIR / "c2idtl_experiment" / "images"

df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().dropna(subset=["is_attributed"]).reset_index(drop=True)

y = df["is_attributed"].astype(int)

# Reproduce exactly the same split used by c2idtl_train.py
indices = pd.Series(range(len(df)))

train_val_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.25,
    random_state=42,
    stratify=y.iloc[train_val_idx],
)

splits = {
    "train": train_idx,
    "validation": val_idx,
    "test": test_idx,
}

for split_name, split_idx in splits.items():
    labels = y.iloc[split_idx].reset_index(drop=True)

    manifest = pd.DataFrame({
        "filename": [
            f"click_{i:06d}.png" for i in range(len(labels))
        ],
        "label": labels,
    })

    split_dir = OUTPUT_DIR / split_name
    manifest_path = split_dir / "labels.csv"

    # Save labels beside the corresponding images.
    manifest.to_csv(manifest_path, index=False)

    print(
        f"{split_name}: {len(manifest):,} labels saved; "
        f"positive labels = {int(labels.sum())}"
    )

print("\nLabel manifests created.")
