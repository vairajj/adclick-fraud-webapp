import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path("data/train_sample.csv")
OUTPUT_PATH = Path("data/train_clean.csv")


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("AD CLICK DATA CLEANING")
print("=" * 60)

print("\n[1/8] Loading dataset...")

df = pd.read_csv(INPUT_PATH)

print(f"Original shape: {df.shape}")


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

print("\n[2/8] Checking required columns...")

required_columns = [
    "ip",
    "app",
    "device",
    "os",
    "channel",
    "click_time",
    "attributed_time",
    "is_attributed",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("All required columns are present.")


# ============================================================
# 3. REMOVE EXACT DUPLICATES
# ============================================================

print("\n[3/8] Removing exact duplicate rows...")

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows found: {duplicate_count}")

df = df.drop_duplicates().copy()

print(f"Rows after duplicate removal: {len(df):,}")


# ============================================================
# 4. VALIDATE CATEGORICAL / ID COLUMNS
# ============================================================

print("\n[4/8] Validating ID columns...")

id_columns = [
    "ip",
    "app",
    "device",
    "os",
    "channel",
]

for column in id_columns:

    # Convert invalid values to NaN
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    invalid_count = df[column].isna().sum()

    print(
        f"{column:10s} -> "
        f"invalid/missing: {invalid_count}"
    )

    # These IDs should never be negative
    negative_count = (df[column] < 0).sum()

    if negative_count > 0:
        print(
            f"WARNING: {column} has "
            f"{negative_count} negative values"
        )


# ============================================================
# 5. CONVERT TIMESTAMPS
# ============================================================

print("\n[5/8] Converting timestamps...")

# Click time should always exist
df["click_time"] = pd.to_datetime(
    df["click_time"],
    errors="coerce"
)

invalid_click_time = df["click_time"].isna().sum()

print(
    f"Invalid click_time values: "
    f"{invalid_click_time}"
)

# A click without a valid timestamp cannot be used
# reliably for temporal analysis.
if invalid_click_time > 0:
    df = df.dropna(
        subset=["click_time"]
    ).copy()


# attributed_time is naturally missing for
# non-attributed clicks, so DO NOT drop those rows.
df["attributed_time"] = pd.to_datetime(
    df["attributed_time"],
    errors="coerce"
)

print(
    f"Missing attributed_time: "
    f"{df['attributed_time'].isna().sum():,}"
)


# ============================================================
# 6. VALIDATE TARGET
# ============================================================

print("\n[6/8] Validating target...")

valid_targets = [0, 1]

invalid_target = ~df["is_attributed"].isin(
    valid_targets
)

invalid_target_count = invalid_target.sum()

print(
    f"Invalid target values: "
    f"{invalid_target_count}"
)

if invalid_target_count > 0:
    df = df.loc[
        ~invalid_target
    ].copy()


# ============================================================
# 7. SORT CHRONOLOGICALLY
# ============================================================

print("\n[7/8] Sorting by click time...")

df = df.sort_values(
    "click_time"
).reset_index(drop=True)


# ============================================================
# 8. SAVE CLEAN DATASET
# ============================================================

print("\n[8/8] Saving cleaned dataset...")

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"\nSaved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)

print(f"\nFinal shape: {df.shape}")

print("\nFinal data types:")
print(df.dtypes)

print("\nFinal missing values:")
print(df.isnull().sum())

print("\nFinal target distribution:")
print(df["is_attributed"].value_counts())

print("\nFinal target percentage:")
print(
    df["is_attributed"]
    .value_counts(normalize=True)
    .mul(100)
    .round(3)
)

print("\nTime range:")
print(f"First click: {df['click_time'].min()}")
print(f"Last click:  {df['click_time'].max()}")

print("\nDone.")