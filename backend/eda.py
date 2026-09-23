import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/train_clean.csv")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AD CLICK FRAUD - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_PATH, parse_dates=[
    "click_time",
    "attributed_time"
])

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 1. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC DATASET INFORMATION")
print("=" * 70)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 2. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("2. TARGET DISTRIBUTION")
print("=" * 70)

target_counts = df["is_attributed"].value_counts()

print("\nCounts:")
print(target_counts)

print("\nPercentages:")
print(
    df["is_attributed"]
    .value_counts(normalize=True)
    .mul(100)
    .round(3)
)

positive = (df["is_attributed"] == 1).sum()
negative = (df["is_attributed"] == 0).sum()

print(f"\nAttributed clicks:     {positive:,}")
print(f"Non-attributed clicks: {negative:,}")

if positive > 0:
    print(
        f"Class ratio: "
        f"1 positive for every {negative / positive:.1f} negative clicks"
    )


# ============================================================
# 3. TIME RANGE
# ============================================================

print("\n" + "=" * 70)
print("3. TIME RANGE")
print("=" * 70)

print(f"First click: {df['click_time'].min()}")
print(f"Last click:  {df['click_time'].max()}")

duration = (
    df["click_time"].max()
    - df["click_time"].min()
)

print(f"Duration: {duration}")


# ============================================================
# 4. UNIQUE VALUES
# ============================================================

print("\n" + "=" * 70)
print("4. UNIQUE VALUES")
print("=" * 70)

columns_to_check = [
    "ip",
    "app",
    "device",
    "os",
    "channel"
]

for column in columns_to_check:
    print(
        f"{column:10s}: "
        f"{df[column].nunique():,} unique values"
    )


# ============================================================
# 5. TOP IPs
# ============================================================

print("\n" + "=" * 70)
print("5. TOP 20 IPS BY CLICK COUNT")
print("=" * 70)

ip_counts = (
    df["ip"]
    .value_counts()
    .head(20)
)

print(ip_counts.to_string())


# ============================================================
# 6. IP ATTRIBUTION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("6. IP ATTRIBUTION ANALYSIS")
print("=" * 70)

ip_stats = (
    df.groupby("ip")
    .agg(
        clicks=("is_attributed", "size"),
        attributed=("is_attributed", "sum")
    )
)

ip_stats["attribution_rate"] = (
    ip_stats["attributed"]
    / ip_stats["clicks"]
)

print("\nIPs with the most clicks:")

print(
    ip_stats
    .sort_values("clicks", ascending=False)
    .head(20)
    .to_string()
)


# ============================================================
# 7. TOP APPS
# ============================================================

print("\n" + "=" * 70)
print("7. TOP 20 APPS BY CLICK COUNT")
print("=" * 70)

app_stats = (
    df.groupby("app")
    .agg(
        clicks=("is_attributed", "size"),
        attributed=("is_attributed", "sum")
    )
)

app_stats["attribution_rate"] = (
    app_stats["attributed"]
    / app_stats["clicks"]
)

print(
    app_stats
    .sort_values("clicks", ascending=False)
    .head(20)
    .to_string()
)


# ============================================================
# 8. CHANNEL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("8. TOP 20 CHANNELS BY CLICK COUNT")
print("=" * 70)

channel_stats = (
    df.groupby("channel")
    .agg(
        clicks=("is_attributed", "size"),
        attributed=("is_attributed", "sum")
    )
)

channel_stats["attribution_rate"] = (
    channel_stats["attributed"]
    / channel_stats["clicks"]
)

print(
    channel_stats
    .sort_values("clicks", ascending=False)
    .head(20)
    .to_string()
)


# ============================================================
# 9. DEVICE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("9. DEVICE ANALYSIS")
print("=" * 70)

device_stats = (
    df.groupby("device")
    .agg(
        clicks=("is_attributed", "size"),
        attributed=("is_attributed", "sum")
    )
)

device_stats["attribution_rate"] = (
    device_stats["attributed"]
    / device_stats["clicks"]
)

print(
    device_stats
    .sort_values("clicks", ascending=False)
    .head(20)
    .to_string()
)


# ============================================================
# 10. OS ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("10. OS ANALYSIS")
print("=" * 70)

os_stats = (
    df.groupby("os")
    .agg(
        clicks=("is_attributed", "size"),
        attributed=("is_attributed", "sum")
    )
)

os_stats["attribution_rate"] = (
    os_stats["attributed"]
    / os_stats["clicks"]
)

print(
    os_stats
    .sort_values("clicks", ascending=False)
    .head(20)
    .to_string()
)


# ============================================================
# 11. HOURLY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("11. HOURLY CLICK ANALYSIS")
print("=" * 70)

df["hour"] = df["click_time"].dt.hour

hour_stats = (
    df.groupby("hour")
    .agg(
        clicks=("is_attributed", "size"),
        attributed=("is_attributed", "sum")
    )
)

hour_stats["attribution_rate"] = (
    hour_stats["attributed"]
    / hour_stats["clicks"]
)

print(hour_stats.to_string())


# ============================================================
# 12. DAY OF WEEK ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("12. DAY OF WEEK ANALYSIS")
print("=" * 70)

df["day_of_week"] = df["click_time"].dt.dayofweek

day_names = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}

day_stats = (
    df.groupby("day_of_week")
    .agg(
        clicks=("is_attributed", "size"),
        attributed=("is_attributed", "sum")
    )
)

day_stats["day"] = day_stats.index.map(day_names)

day_stats["attribution_rate"] = (
    day_stats["attributed"]
    / day_stats["clicks"]
)

print(
    day_stats[
        ["day", "clicks", "attributed", "attribution_rate"]
    ].to_string(index=False)
)


# ============================================================
# 13. TOP IPs WITH ATTRIBUTED CLICKS
# ============================================================

print("\n" + "=" * 70)
print("13. IPS WITH MOST ATTRIBUTED CLICKS")
print("=" * 70)

attributed_ips = (
    ip_stats[ip_stats["attributed"] > 0]
    .sort_values(
        ["attributed", "clicks"],
        ascending=False
    )
)

print(
    attributed_ips.head(20).to_string()
)


# ============================================================
# 14. TOP APPS BY ATTRIBUTION RATE
# ============================================================

print("\n" + "=" * 70)
print("14. APPS WITH ATTRIBUTION RATE")
print("=" * 70)

# Require at least 50 clicks so that tiny groups
# don't produce misleading 100% rates.

reliable_apps = app_stats[
    app_stats["clicks"] >= 50
]

print(
    reliable_apps
    .sort_values(
        "attribution_rate",
        ascending=False
    )
    .head(20)
    .to_string()
)


# ============================================================
# 15. TOP CHANNELS BY ATTRIBUTION RATE
# ============================================================

print("\n" + "=" * 70)
print("15. CHANNELS WITH ATTRIBUTION RATE")
print("=" * 70)

reliable_channels = channel_stats[
    channel_stats["clicks"] >= 50
]

print(
    reliable_channels
    .sort_values(
        "attribution_rate",
        ascending=False
    )
    .head(20)
    .to_string()
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print("""
The dataset has now been analyzed without modifying
the cleaned CSV.

Next we will use these results to design behavioral
features for the machine-learning pipeline.
""")