
import pandas as pd

df = pd.read_csv("data/click_fraud_dataset.csv")

print("\n========== NUMERIC FEATURE CORRELATION ==========")

numeric_df = df.select_dtypes(include="number")

correlations = (
    numeric_df.corr()["is_fraudulent"]
    .drop("is_fraudulent")
    .sort_values(ascending=False)
)

print(correlations.to_string())

print("\n========== CATEGORICAL FEATURE VALUES ==========")

categorical_columns = [
    "device_type",
    "browser",
    "operating_system",
    "ad_position",
    "device_ip_reputation",
]

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts().to_string())

print("\n========== DUPLICATE RECORDS ==========")
print("Duplicate rows:", df.duplicated().sum())

print("\n========== UNIQUE IDENTIFIERS ==========")
for column in ["click_id", "user_id", "ip_address"]:
    print(f"{column}: {df[column].nunique()} unique values")
