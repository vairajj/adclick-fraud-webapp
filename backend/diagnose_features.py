
import pandas as pd

df = pd.read_csv("data/click_fraud_dataset.csv")

target = "is_fraudulent"

# Numeric feature comparison
numeric_columns = df.select_dtypes(include="number").columns.drop(
    target
).tolist()

print("\n========== NUMERIC FEATURES BY CLASS ==========")

for column in numeric_columns:
    print(f"\n{column}")
    print(
        df.groupby(target)[column]
        .agg(["mean", "median", "std", "min", "max"])
        .round(3)
        .to_string()
    )

# Categorical feature comparison
categorical_columns = df.select_dtypes(include="object").columns.tolist()

categorical_columns = [
    column for column in categorical_columns
    if column not in ["click_id", "user_id", "ip_address",
                      "timestamp", "referrer_url", "page_url"]
]

print("\n========== CATEGORICAL FRAUD RATES ==========")

for column in categorical_columns:
    print(f"\n{column}")
    rates = pd.crosstab(
        df[column],
        df[target],
        normalize="index"
    )

    if 1 in rates.columns:
        rates = rates[1].sort_values(ascending=False)
        print((rates * 100).round(2).to_string())

# Bot score investigation
print("\n========== BOT SCORE BY FRAUD LABEL ==========")

print(
    df.groupby(target)["bot_likelihood_score"]
    .describe()
    .round(3)
    .to_string()
)

# Check whether fraud is concentrated in particular months
print("\n========== FRAUD RATE BY MONTH ==========")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["month"] = df["timestamp"].dt.to_period("M").astype(str)

print(
    df.groupby("month")[target]
    .agg(["count", "mean"])
    .round(3)
    .to_string()
)
