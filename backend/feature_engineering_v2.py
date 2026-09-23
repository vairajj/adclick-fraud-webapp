import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path("data/train_clean.csv")
OUTPUT_PATH = Path("data/train_features_v2.csv")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AD CLICK - TIME-AWARE FEATURE ENGINEERING V2")
print("=" * 70)

print("\n[1/8] Loading cleaned dataset...")

df = pd.read_csv(
    INPUT_PATH,
    parse_dates=[
        "click_time",
        "attributed_time"
    ]
)

print(f"Input shape: {df.shape}")


# ============================================================
# 1. SORT CHRONOLOGICALLY
# ============================================================

print("\n[2/8] Sorting clicks chronologically...")

df = (
    df.sort_values("click_time")
      .reset_index(drop=True)
)

print(
    f"First click: {df['click_time'].min()}"
)

print(
    f"Last click:  {df['click_time'].max()}"
)


# ============================================================
# 2. TEMPORAL FEATURES
# ============================================================

print("\n[3/8] Creating temporal features...")

df["hour"] = df["click_time"].dt.hour

df["day"] = df["click_time"].dt.day

df["day_of_week"] = (
    df["click_time"].dt.dayofweek
)

# Dataset only contains Monday-Thursday,
# so we intentionally do not add is_weekend.

print("Created:")
print("  - hour")
print("  - day")
print("  - day_of_week")


# ============================================================
# 3. PREVIOUS CLICK COUNTS
# ============================================================

print("\n[4/8] Creating historical click counts...")

# ------------------------------------------------------------
# IP
# ------------------------------------------------------------

df["ip_prev_clicks"] = (
    df.groupby("ip")
      .cumcount()
)

# ------------------------------------------------------------
# APP
# ------------------------------------------------------------

df["app_prev_clicks"] = (
    df.groupby("app")
      .cumcount()
)

# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

df["device_prev_clicks"] = (
    df.groupby("device")
      .cumcount()
)

# ------------------------------------------------------------
# OS
# ------------------------------------------------------------

df["os_prev_clicks"] = (
    df.groupby("os")
      .cumcount()
)

# ------------------------------------------------------------
# CHANNEL
# ------------------------------------------------------------

df["channel_prev_clicks"] = (
    df.groupby("channel")
      .cumcount()
)

print("Created:")
print("  - ip_prev_clicks")
print("  - app_prev_clicks")
print("  - device_prev_clicks")
print("  - os_prev_clicks")
print("  - channel_prev_clicks")


# ============================================================
# 4. PREVIOUS INTERACTION COUNTS
# ============================================================

print("\n[5/8] Creating historical interaction counts...")


def previous_group_count(df, columns):
    """
    Number of previous observations for a combination.

    The current row is excluded because cumcount()
    counts only previous rows.
    """
    return (
        df.groupby(columns)
          .cumcount()
    )


df["ip_app_prev_clicks"] = previous_group_count(
    df,
    ["ip", "app"]
)

df["ip_device_prev_clicks"] = previous_group_count(
    df,
    ["ip", "device"]
)

df["ip_os_prev_clicks"] = previous_group_count(
    df,
    ["ip", "os"]
)

df["ip_channel_prev_clicks"] = previous_group_count(
    df,
    ["ip", "channel"]
)

print("Created:")
print("  - ip_app_prev_clicks")
print("  - ip_device_prev_clicks")
print("  - ip_os_prev_clicks")
print("  - ip_channel_prev_clicks")


# ============================================================
# 5. TIME SINCE PREVIOUS CLICK
# ============================================================

print("\n[6/8] Creating time-gap features...")


# ------------------------------------------------------------
# Previous click from same IP
# ------------------------------------------------------------

df["prev_ip_click_time"] = (
    df.groupby("ip")["click_time"]
      .shift(1)
)

df["seconds_since_prev_ip_click"] = (
    (
        df["click_time"]
        - df["prev_ip_click_time"]
    )
    .dt.total_seconds()
)


# ------------------------------------------------------------
# Previous click from same IP + APP
# ------------------------------------------------------------

df["prev_ip_app_click_time"] = (
    df.groupby(
        ["ip", "app"]
    )["click_time"]
    .shift(1)
)

df["seconds_since_prev_ip_app_click"] = (
    (
        df["click_time"]
        - df["prev_ip_app_click_time"]
    )
    .dt.total_seconds()
)


# ------------------------------------------------------------
# Previous click from same IP + DEVICE
# ------------------------------------------------------------

df["prev_ip_device_click_time"] = (
    df.groupby(
        ["ip", "device"]
    )["click_time"]
    .shift(1)
)

df["seconds_since_prev_ip_device_click"] = (
    (
        df["click_time"]
        - df["prev_ip_device_click_time"]
    )
    .dt.total_seconds()
)


print("Created:")
print("  - seconds_since_prev_ip_click")
print("  - seconds_since_prev_ip_app_click")
print("  - seconds_since_prev_ip_device_click")


# ============================================================
# 6. CLEAN TIME-GAP VALUES
# ============================================================

print("\n[7/8] Cleaning time-gap features...")

time_features = [
    "seconds_since_prev_ip_click",
    "seconds_since_prev_ip_app_click",
    "seconds_since_prev_ip_device_click",
]

for feature in time_features:

    # First click for a group has no previous click.
    # Represent this as -1 rather than NaN.
    df[feature] = (
        df[feature]
        .fillna(-1)
    )

    # Protect against impossible negative values.
    invalid_negative = (
        df[feature] < -1
    ).sum()

    if invalid_negative > 0:
        print(
            f"WARNING: {feature} has "
            f"{invalid_negative} invalid values"
        )

        df.loc[
            df[feature] < -1,
            feature
        ] = -1


# ============================================================
# 7. BEHAVIORAL INTENSITY FEATURES
# ============================================================

print("\nCreating behavioral intensity features...")


# ------------------------------------------------------------
# IP click velocity
#
# Approximate clicks per hour since the first observed
# click from the same IP.
# ------------------------------------------------------------

first_ip_click_time = (
    df.groupby("ip")["click_time"]
      .transform("min")
)

ip_observation_hours = (
    (
        df["click_time"]
        - first_ip_click_time
    )
    .dt.total_seconds()
    / 3600
)

# Avoid division by zero for an IP's first click.
ip_observation_hours = (
    ip_observation_hours.clip(lower=1 / 3600)
)

df["ip_clicks_per_hour"] = (
    df["ip_prev_clicks"]
    / ip_observation_hours
)


# ------------------------------------------------------------
# IP + APP click velocity
# ------------------------------------------------------------

first_ip_app_click_time = (
    df.groupby(
        ["ip", "app"]
    )["click_time"]
    .transform("min")
)

ip_app_observation_hours = (
    (
        df["click_time"]
        - first_ip_app_click_time
    )
    .dt.total_seconds()
    / 3600
)

ip_app_observation_hours = (
    ip_app_observation_hours.clip(lower=1 / 3600)
)

df["ip_app_clicks_per_hour"] = (
    df["ip_app_prev_clicks"]
    / ip_app_observation_hours
)


print("Created:")
print("  - ip_clicks_per_hour")
print("  - ip_app_clicks_per_hour")


# ============================================================
# 8. VALIDATION
# ============================================================

print("\n[8/8] Validating V2 features...")


new_features = [
    "hour",
    "day",
    "day_of_week",

    "ip_prev_clicks",
    "app_prev_clicks",
    "device_prev_clicks",
    "os_prev_clicks",
    "channel_prev_clicks",

    "ip_app_prev_clicks",
    "ip_device_prev_clicks",
    "ip_os_prev_clicks",
    "ip_channel_prev_clicks",

    "seconds_since_prev_ip_click",
    "seconds_since_prev_ip_app_click",
    "seconds_since_prev_ip_device_click",

    "ip_clicks_per_hour",
    "ip_app_clicks_per_hour",
]


print(
    f"\nNumber of new features: "
    f"{len(new_features)}"
)


print("\nMissing values:")

print(
    df[new_features]
    .isnull()
    .sum()
)


print("\nInfinite values:")

infinite_counts = (
    df[new_features]
    .isin([np.inf, -np.inf])
    .sum()
)

print(infinite_counts)


# Replace accidental infinite values
df[new_features] = (
    df[new_features]
    .replace([np.inf, -np.inf], np.nan)
)


# For engineered numerical features, replace any
# unexpected NaN with 0.
df[new_features] = (
    df[new_features]
    .fillna(0)
)


# ============================================================
# FEATURE STATISTICS
# ============================================================

print("\nFeature statistics:")

print(
    df[new_features]
    .describe()
    .transpose()
    .round(4)
    .to_string()
)


# ============================================================
# SAVE DATASET
# ============================================================

print("\nSaving V2 dataset...")

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"\nSaved to:")
print(OUTPUT_PATH)

print(
    f"\nFinal dataset shape: "
    f"{df.shape}"
)


# ============================================================
# SHOW EXAMPLE
# ============================================================

print("\n" + "=" * 70)
print("EXAMPLE OF TIME-AWARE FEATURES")
print("=" * 70)

example_columns = [
    "ip",
    "app",
    "click_time",
    "ip_prev_clicks",
    "ip_app_prev_clicks",
    "seconds_since_prev_ip_click",
    "seconds_since_prev_ip_app_click",
    "ip_clicks_per_hour",
]

print(
    df[example_columns]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING V2 COMPLETE")
print("=" * 70)

print("""
V2 uses historical information available before each click.

This makes the behavioral features more suitable for
real-time prediction and reduces future-information leakage.
""")