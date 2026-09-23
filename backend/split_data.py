import pandas as pd
from pathlib import Path

print("=" * 70)
print("AD CLICK - TIME-AWARE TRAIN / TEST SPLIT")
print("=" * 70)

INPUT_FILE = Path("data/train_features_v3.csv")

TRAIN_FILE = Path("data/train_v3.csv")
TEST_FILE = Path("data/test_v3.csv")

print("\n[1/5] Loading V3 dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["click_time"]
)

print(f"Total rows: {len(df):,}")

# -------------------------------------------------------------------
# Ensure chronological order
# -------------------------------------------------------------------

print("\n[2/5] Sorting chronologically...")

df = df.sort_values("click_time").reset_index(drop=True)

# -------------------------------------------------------------------
# Time-aware split
# Train: before Nov 9
# Test: Nov 9
# -------------------------------------------------------------------

print("\n[3/5] Creating chronological split...")

split_date = pd.Timestamp("2017-11-09")

train_df = df[df["click_time"] < split_date].copy()
test_df = df[df["click_time"] >= split_date].copy()

# -------------------------------------------------------------------
# Save
# -------------------------------------------------------------------

print("\n[4/5] Saving datasets...")

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------

print("\n[5/5] Validating split...")

print("\nTRAIN")
print("-" * 40)
print(f"Rows:       {len(train_df):,}")
print(f"Positives:  {train_df['is_attributed'].sum():,}")
print(f"Negatives:  {(train_df['is_attributed'] == 0).sum():,}")
print(f"Start:      {train_df['click_time'].min()}")
print(f"End:        {train_df['click_time'].max()}")

print("\nTEST")
print("-" * 40)
print(f"Rows:       {len(test_df):,}")
print(f"Positives:  {test_df['is_attributed'].sum():,}")
print(f"Negatives:  {(test_df['is_attributed'] == 0).sum():,}")
print(f"Start:      {test_df['click_time'].min()}")
print(f"End:        {test_df['click_time'].max()}")

print("\nPositive rate:")
print(f"Train: {train_df['is_attributed'].mean() * 100:.4f}%")
print(f"Test:  {test_df['is_attributed'].mean() * 100:.4f}%")

print("\nFiles created:")
print(f"  {TRAIN_FILE}")
print(f"  {TEST_FILE}")

print("\n" + "=" * 70)
print("TIME-AWARE SPLIT COMPLETE")
print("=" * 70)