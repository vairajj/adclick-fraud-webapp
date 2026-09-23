import pandas as pd

DATA_PATH = "data/train_sample.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATASET INSPECTION")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nTarget distribution:")
print(df["is_attributed"].value_counts())

print("\nTarget percentage:")
print(df["is_attributed"].value_counts(normalize=True) * 100)

print("\nUnique values:")
for column in df.columns:
    print(f"{column}: {df[column].nunique():,}")

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())