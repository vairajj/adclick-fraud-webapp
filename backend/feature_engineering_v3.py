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
# 4. HISTORICAL COUNTS — STRICTLY EARLIER TIMESTAMPS
# ============================================================

print("\n[4/9] Creating historical counts...")

def add_previous_timestamp_counts(dataframe, group_columns, output_column):
    """Count group events with timestamps strictly before each row."""
    counts = pd.Series(0, index=dataframe.index, dtype="int64")

    for _, group in dataframe.groupby(group_columns, sort=False):
        group = group.sort_values("click_time", kind="stable")
        timestamp_counts = group.groupby("click_time", sort=False).size()
        cumulative = timestamp_counts.cumsum().shift(1, fill_value=0)
        counts.loc[group.index] = (
            group["click_time"].map(cumulative).to_numpy()
        )

    dataframe[output_column] = counts


add_previous_timestamp_counts(df, ["ip"], "ip_prev_clicks")
add_previous_timestamp_counts(df, ["app"], "app_prev_clicks")
add_previous_timestamp_counts(df, ["device"], "device_prev_clicks")
add_previous_timestamp_counts(df, ["os"], "os_prev_clicks")
add_previous_timestamp_counts(df, ["channel"], "channel_prev_clicks")



# ============================================================
# 5. HISTORICAL INTERACTION COUNTS
# ============================================================

print("\n[5/9] Creating interaction history...")

add_previous_timestamp_counts(df, ["ip", "app"], "ip_app_prev_clicks")
add_previous_timestamp_counts(df, ["ip", "device"], "ip_device_prev_clicks")
add_previous_timestamp_counts(df, ["ip", "os"], "ip_os_prev_clicks")
add_previous_timestamp_counts(df, ["ip", "channel"], "ip_channel_prev_clicks")



# ============================================================
# 6. PREVIOUS CLICK INFORMATION
# ============================================================

print("\n[6/9] Creating previous-click features...")


def add_previous_click_features(dataframe, group_columns, prefix):
    """Use only timestamps strictly earlier than the current click."""
    previous_time = pd.Series(
        pd.NaT, index=dataframe.index, dtype="datetime64[ns]"
    )

    for _, group in dataframe.groupby(group_columns, sort=False):
        group = group.sort_values("click_time", kind="stable")

        # One row per distinct timestamp, so tied clicks share
        # the same strictly previous timestamp.
        unique_times = pd.Index(group["click_time"].drop_duplicates())
        prior_times = pd.Series(unique_times).shift(1)
        previous_by_time = dict(zip(unique_times, prior_times))

        previous_time.loc[group.index] = (
            group["click_time"].map(previous_by_time).to_numpy()
        )

    dataframe[f"prev_{prefix}_click_time"] = previous_time
    dataframe[f"seconds_since_prev_{prefix}_click"] = (
        dataframe["click_time"] - previous_time
    ).dt.total_seconds()
    dataframe[f"has_previous_{prefix}_click"] = (
        previous_time.notna().astype("int8")
    )


add_previous_click_features(df, ["ip"], "ip")
add_previous_click_features(df, ["ip", "app"], "ip_app")
add_previous_click_features(df, ["ip", "device"], "ip_device")

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



def previous_window_count(timestamps, window_seconds):
    """
    Count events strictly before the current timestamp,
    within the specified time window.

    Events sharing the current timestamp are excluded.
    """
    values = (
        timestamps
        .astype("datetime64[ns]")
        .astype("int64")
     .to_numpy()
    )
    result = np.zeros(len(values), dtype=np.int32)

    left = 0
    right = 0
    n = len(values)
    window_ns = window_seconds * 1_000_000_000

    while right < n:
        current_time = values[right]

        # Find all rows with this exact timestamp.
        end = right + 1
        while end < n and values[end] == current_time:
            end += 1

        # Keep only events strictly newer than the window start.
        window_start = current_time - window_ns
        while left < right and values[left] <= window_start:
            left += 1

        # Every tied row gets the same count. None of the
        # current-timestamp rows are counted as previous events.
        count = right - left
        result[right:end] = count

        right = end

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