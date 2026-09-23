import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path("data/train_clean.csv")
OUTPUT_PATH = Path("data/train_features_v3.csv")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AD CLICK - TIME-AWARE FEATURE ENGINEERING V3")
print("=" * 70)

print("\n[1/9] Loading cleaned dataset...")

df = pd.read_csv(
    INPUT_PATH,
    parse_dates=[
        "click_time",
        "attributed_time"
    ]
)

print(f"Input shape: {df.shape}")


# ============================================================
# 2. SORT CHRONOLOGICALLY
# ============================================================

print("\n[2/9] Sorting clicks chronologically...")

df = (
    df.sort_values("click_time")
      .reset_index(drop=True)
)

print(f"First click: {df['click_time'].min()}")
print(f"Last click:  {df['click_time'].max()}")


# ============================================================
# 3. TEMPORAL FEATURES
# ============================================================

print("\n[3/9] Creating temporal features...")

df["hour"] = df["click_time"].dt.hour

df["day"] = df["click_time"].dt.day

df["day_of_week"] = (
    df["click_time"].dt.dayofweek
)

print("Created:")
print("  - hour")
print("  - day")
print("  - day_of_week")


# ============================================================
# 4. HISTORICAL COUNTS
# ============================================================

print("\n[4/9] Creating historical counts...")

df["ip_prev_clicks"] = (
    df.groupby("ip")
      .cumcount()
)

df["app_prev_clicks"] = (
    df.groupby("app")
      .cumcount()
)

df["device_prev_clicks"] = (
    df.groupby("device")
      .cumcount()
)

df["os_prev_clicks"] = (
    df.groupby("os")
      .cumcount()
)

df["channel_prev_clicks"] = (
    df.groupby("channel")
      .cumcount()
)


# ============================================================
# 5. HISTORICAL INTERACTION COUNTS
# ============================================================

print("\n[5/9] Creating interaction history...")

df["ip_app_prev_clicks"] = (
    df.groupby(["ip", "app"])
      .cumcount()
)

df["ip_device_prev_clicks"] = (
    df.groupby(["ip", "device"])
      .cumcount()
)

df["ip_os_prev_clicks"] = (
    df.groupby(["ip", "os"])
      .cumcount()
)

df["ip_channel_prev_clicks"] = (
    df.groupby(["ip", "channel"])
      .cumcount()
)


# ============================================================
# 6. PREVIOUS CLICK INFORMATION
# ============================================================

print("\n[6/9] Creating previous-click features...")


# ------------------------------------------------------------
# Previous IP click
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

df["has_previous_ip_click"] = (
    df["prev_ip_click_time"]
    .notna()
    .astype("int8")
)


# ------------------------------------------------------------
# Previous IP + APP click
# ------------------------------------------------------------

df["prev_ip_app_click_time"] = (
    df.groupby(["ip", "app"])["click_time"]
      .shift(1)
)

df["seconds_since_prev_ip_app_click"] = (
    (
        df["click_time"]
        - df["prev_ip_app_click_time"]
    )
    .dt.total_seconds()
)

df["has_previous_ip_app_click"] = (
    df["prev_ip_app_click_time"]
    .notna()
    .astype("int8")
)


# ------------------------------------------------------------
# Previous IP + DEVICE click
# ------------------------------------------------------------

df["prev_ip_device_click_time"] = (
    df.groupby(["ip", "device"])["click_time"]
      .shift(1)
)

df["seconds_since_prev_ip_device_click"] = (
    (
        df["click_time"]
        - df["prev_ip_device_click_time"]
    )
    .dt.total_seconds()
)

df["has_previous_ip_device_click"] = (
    df["prev_ip_device_click_time"]
    .notna()
    .astype("int8")
)


# ============================================================
# 7. FIXED-WINDOW ACTIVITY
# ============================================================

print("\n[7/9] Creating recent activity features...")

# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
#
# For every click we count only clicks that happened BEFORE
# the current click.
#
# The current click itself is excluded.
#
# ------------------------------------------------------------


# Keep original row order so we can assign results safely.
df["_row_order"] = np.arange(len(df))


def previous_window_count(
    timestamps,
    window_seconds
):
    """
    Count previous timestamps inside a fixed time window.

    The current timestamp is excluded.
    """

    values = timestamps.astype("int64").to_numpy()

    result = np.zeros(len(values), dtype=np.int32)

    left = 0

    for right in range(len(values)):

        current_time = values[right]

        window_start = (
            current_time
            - window_seconds * 1_000_000_000
        )

        while (
            left < right
            and values[left] <= window_start
        ):
            left += 1

        result[right] = right - left

    return result


def add_window_count(
    dataframe,
    group_columns,
    output_column,
    window_seconds
):
    """
    Calculate previous-event count for each group.
    """

    result = pd.Series(
        0,
        index=dataframe.index,
        dtype="int32"
    )

    for _, group in dataframe.groupby(
        group_columns,
        sort=False
    ):

        indices = group.index

        timestamps = group["click_time"]

        counts = previous_window_count(
            timestamps,
            window_seconds
        )

        result.loc[indices] = counts

    dataframe[output_column] = result


# ------------------------------------------------------------
# IP activity
# ------------------------------------------------------------

print("Calculating IP activity windows...")

add_window_count(
    df,
    ["ip"],
    "ip_clicks_last_1min",
    60
)

add_window_count(
    df,
    ["ip"],
    "ip_clicks_last_5min",
    5 * 60
)

add_window_count(
    df,
    ["ip"],
    "ip_clicks_last_1hour",
    60 * 60
)


# ------------------------------------------------------------
# IP + APP activity
# ------------------------------------------------------------

print("Calculating IP + APP activity windows...")

add_window_count(
    df,
    ["ip", "app"],
    "ip_app_clicks_last_1min",
    60
)

add_window_count(
    df,
    ["ip", "app"],
    "ip_app_clicks_last_5min",
    5 * 60
)

add_window_count(
    df,
    ["ip", "app"],
    "ip_app_clicks_last_1hour",
    60 * 60
)


# ============================================================
# 8. CLEAN AND VALIDATE
# ============================================================

print("\n[8/9] Validating V3 features...")


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

    "has_previous_ip_click",
    "has_previous_ip_app_click",
    "has_previous_ip_device_click",

    "ip_clicks_last_1min",
    "ip_clicks_last_5min",
    "ip_clicks_last_1hour",

    "ip_app_clicks_last_1min",
    "ip_app_clicks_last_5min",
    "ip_app_clicks_last_1hour",
]


print(
    f"\nNumber of engineered features: "
    f"{len(new_features)}"
)


# Time differences are undefined when no previous
# event exists. Use 0 together with the corresponding
# has_previous_* indicator.

time_features = [
    "seconds_since_prev_ip_click",
    "seconds_since_prev_ip_app_click",
    "seconds_since_prev_ip_device_click",
]

for feature in time_features:

    df[feature] = (
        df[feature]
        .fillna(0)
    )


# Check missing values
print("\nMissing values:")

missing = (
    df[new_features]
    .isnull()
    .sum()
)

print(missing)


# Check infinite values
print("\nInfinite values:")

infinite = (
    df[new_features]
    .isin([np.inf, -np.inf])
    .sum()
)

print(infinite)


# Replace unexpected infinite values
df[new_features] = (
    df[new_features]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
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
# REMOVE INTERNAL COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "_row_order",
        "prev_ip_click_time",
        "prev_ip_app_click_time",
        "prev_ip_device_click_time",
    ]
)


# ============================================================
# 9. SAVE
# ============================================================

print("\n[9/9] Saving V3 dataset...")

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
# SHOW EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("EXAMPLE OF V3 BEHAVIORAL FEATURES")
print("=" * 70)

example_columns = [
    "ip",
    "app",
    "click_time",

    "ip_prev_clicks",
    "ip_app_prev_clicks",

    "has_previous_ip_click",

    "seconds_since_prev_ip_click",

    "ip_clicks_last_1min",
    "ip_clicks_last_5min",
    "ip_clicks_last_1hour",

    "ip_app_clicks_last_1min",
    "ip_app_clicks_last_5min",
    "ip_app_clicks_last_1hour",
]

print(
    df[example_columns]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING V3 COMPLETE")
print("=" * 70)

print("""
V3 adds fixed-window behavioral activity features.

The windows are:
  - Previous 1 minute
  - Previous 5 minutes
  - Previous 1 hour

Only events occurring before the current click are counted.

The unstable clicks-per-hour features from V2
are intentionally not included.
""")