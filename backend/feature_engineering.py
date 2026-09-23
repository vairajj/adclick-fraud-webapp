import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path("data/train_clean.csv")
OUTPUT_PATH = Path("data/train_features_v1.csv")


# ============================================================
# LOAD CLEAN DATA
# ============================================================

print("=" * 70)
print("AD CLICK - FEATURE ENGINEERING V1")
print("=" * 70)

print("\n[1/7] Loading cleaned dataset...")

df = pd.read_csv(
    INPUT_PATH,
    parse_dates=[
        "click_time",
        "attributed_time"
    ]
)

print(f"Input shape: {df.shape}")


# ============================================================
# 1. TEMPORAL FEATURES
# ============================================================

print("\n[2/7] Creating temporal features...")

df["hour"] = df["click_time"].dt.hour

df["day"] = df["click_time"].dt.day

df["day_of_week"] = df["click_time"].dt.dayofweek

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype("int8")

print("Created:")
print("  - hour")
print("  - day")
print("  - day_of_week")
print("  - is_weekend")


# ============================================================
# 2. BASIC FREQUENCY FEATURES
# ============================================================

print("\n[3/7] Creating basic frequency features...")

# Number of times each IP appears
df["ip_click_count"] = (
    df.groupby("ip")["ip"]
    .transform("count")
)

# Number of times each app appears
df["app_click_count"] = (
    df.groupby("app")["app"]
    .transform("count")
)

# Number of times each device appears
df["device_click_count"] = (
    df.groupby("device")["device"]
    .transform("count")
)

# Number of times each OS appears
df["os_click_count"] = (
    df.groupby("os")["os"]
    .transform("count")
)

# Number of times each channel appears
df["channel_click_count"] = (
    df.groupby("channel")["channel"]
    .transform("count")
)

print("Created:")
print("  - ip_click_count")
print("  - app_click_count")
print("  - device_click_count")
print("  - os_click_count")
print("  - channel_click_count")


# ============================================================
# 3. INTERACTION FREQUENCY FEATURES
# ============================================================

print("\n[4/7] Creating interaction frequency features...")

# How many times an IP used an app
df["ip_app_count"] = (
    df.groupby(["ip", "app"])["ip"]
    .transform("count")
)

# How many times an IP used a device
df["ip_device_count"] = (
    df.groupby(["ip", "device"])["ip"]
    .transform("count")
)

# How many times an IP used an OS
df["ip_os_count"] = (
    df.groupby(["ip", "os"])["ip"]
    .transform("count")
)

# How many times an IP used a channel
df["ip_channel_count"] = (
    df.groupby(["ip", "channel"])["ip"]
    .transform("count")
)

print("Created:")
print("  - ip_app_count")
print("  - ip_device_count")
print("  - ip_os_count")
print("  - ip_channel_count")


# ============================================================
# 4. BEHAVIORAL RATIO FEATURES
# ============================================================

print("\n[5/7] Creating behavioral ratio features...")

# These describe how concentrated an IP's activity is
# around a particular app/device/OS/channel.

df["ip_app_ratio"] = (
    df["ip_app_count"]
    / df["ip_click_count"]
)

df["ip_device_ratio"] = (
    df["ip_device_count"]
    / df["ip_click_count"]
)

df["ip_os_ratio"] = (
    df["ip_os_count"]
    / df["ip_click_count"]
)

df["ip_channel_ratio"] = (
    df["ip_channel_count"]
    / df["ip_click_count"]
)

print("Created:")
print("  - ip_app_ratio")
print("  - ip_device_ratio")
print("  - ip_os_ratio")
print("  - ip_channel_ratio")


# ============================================================
# 5. DATA VALIDATION
# ============================================================

print("\n[6/7] Validating engineered features...")

# List of features created by this script
new_features = [
    "hour",
    "day",
    "day_of_week",
    "is_weekend",

    "ip_click_count",
    "app_click_count",
    "device_click_count",
    "os_click_count",
    "channel_click_count",

    "ip_app_count",
    "ip_device_count",
    "ip_os_count",
    "ip_channel_count",

    "ip_app_ratio",
    "ip_device_ratio",
    "ip_os_ratio",
    "ip_channel_ratio",
]

print(f"\nNumber of new features: {len(new_features)}")

print("\nNew features:")
for feature in new_features:
    print(f"  {feature}")


# Check missing values
print("\nMissing values in new features:")

missing_features = df[new_features].isnull().sum()

print(missing_features)


# Check infinite values
infinite_counts = (
    df[new_features]
    .isin([float("inf"), float("-inf")])
    .sum()
)

print("\nInfinite values in new features:")

print(infinite_counts)


# Replace any accidental infinite values
df[new_features] = df[new_features].replace(
    [float("inf"), float("-inf")],
    0
)


# ============================================================
# 6. FEATURE SUMMARY
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
# 7. SAVE
# ============================================================

print("\n[7/7] Saving feature-engineered dataset...")

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"\nSaved to:")
print(OUTPUT_PATH)

print("\nFinal dataset shape:")
print(df.shape)

print("\nFinal columns:")

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING V1 COMPLETE")
print("=" * 70)

print("""
The following feature groups were created:

1. Temporal features
2. Basic frequency features
3. IP interaction features
4. Behavioral ratio features

No target-based statistics were used.
""")