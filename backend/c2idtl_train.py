from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train_sample.csv"
OUTPUT_DIR = BASE_DIR / "c2idtl_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "is_attributed"


def prepare_features(X_train, X_val, X_test):
    """Create numerical features and scale using training data only."""

    feature_sets = []

    for split in (X_train, X_val, X_test):
        split = split.copy()

        # Convert click_time into numerical components
        click_time = pd.to_datetime(
            split["click_time"],
            errors="coerce"
        )

        split["year"] = click_time.dt.year
        split["month"] = click_time.dt.month
        split["day"] = click_time.dt.day
        split["hour"] = click_time.dt.hour
        split["minute"] = click_time.dt.minute
        split["second"] = click_time.dt.second

        split = split.drop(columns=["click_time"])

        # Use numeric model inputs only
        split = split.apply(pd.to_numeric, errors="coerce")
        split = split.replace([np.inf, -np.inf], np.nan)

        feature_sets.append(split)

    train_features, val_features, test_features = feature_sets

    # Fill missing values using training-set medians only
    medians = train_features.median().fillna(0)

    train_features = train_features.fillna(medians)
    val_features = val_features.fillna(medians)
    test_features = test_features.fillna(medians)

    # Fit scaler only on training data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_features)
    val_scaled = scaler.transform(val_features)
    test_scaled = scaler.transform(test_features)

    return (
        train_scaled.astype(np.float32),
        val_scaled.astype(np.float32),
        test_scaled.astype(np.float32),
        list(train_features.columns),
        scaler,
    )


def generate_click_images(features, feature_names, importance, output_dir):
    """Convert each feature vector into a 224x224 RGB image."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = 224
    total_pixels = image_size * image_size

    # Allocate pixels proportionally; rounding must total 224*224.
    raw_counts = importance * total_pixels
    pixel_counts = np.floor(raw_counts).astype(int)

    remaining = total_pixels - int(pixel_counts.sum())
    fractions = raw_counts - pixel_counts

    if remaining > 0:
        order = np.argsort(-fractions)
        pixel_counts[order[:remaining]] += 1

    generated = 0

    for row_index, row in enumerate(features):
        flat_image = np.zeros(total_pixels, dtype=np.uint8)
        position = 0

        for feature_index, pixel_count in enumerate(pixel_counts):
            end = position + pixel_count

            # Features are scaled to [0, 1]; convert to grayscale [0, 255].
            value = int(np.clip(row[feature_index], 0, 1) * 255)

            flat_image[position:end] = value
            position = end

        grayscale = flat_image.reshape(image_size, image_size)

        # Duplicate the grayscale image across RGB channels.
        rgb_image = np.stack([grayscale] * 3, axis=-1)

        Image.fromarray(rgb_image, mode="RGB").save(
            output_dir / f"click_{row_index:06d}.png"
        )

        generated += 1

    print(f"Generated {generated:,} images in {output_dir}")

def main():
    print("Loading TalkingData dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Remove duplicate records
    df = df.drop_duplicates().reset_index(drop=True)

    # Remove rows without a target label
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    print(f"\nDataset after cleaning: {df.shape}")
    print("\nLabel distribution:")
    print(df[TARGET].value_counts().sort_index())

    # Separate input features and target
    X = df.drop(columns=[TARGET, "attributed_time"])
    y = df[TARGET].astype(int)

    # First: reserve 20% for the final test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Then: reserve 20% of the remaining data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    print("\nSplit sizes:")
    print(f"Training:   {len(X_train):,} rows")
    print(f"Validation: {len(X_val):,} rows")
    print(f"Test:       {len(X_test):,} rows")

    print("\nPositive-label counts (is_attributed = 1):")
    print(f"Training:   {int(y_train.sum()):,}")
    print(f"Validation: {int(y_val.sum()):,}")
    print(f"Test:       {int(y_test.sum()):,}")

    print("\nSplit preparation completed.")
    print("\nPreparing numerical features...")

    X_train_scaled, X_val_scaled, X_test_scaled, feature_names, scaler = (
    prepare_features(X_train, X_val, X_test)
    )

    print("\nFeature preparation results:")
    print(f"Number of features: {len(feature_names)}")
    print(f"Training shape:   {X_train_scaled.shape}")
    print(f"Validation shape: {X_val_scaled.shape}")
    print(f"Test shape:       {X_test_scaled.shape}")
    print(f"Features: {feature_names}")

    print("\nCalculating feature importance...")

    rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    )

    rf.fit(X_train_scaled, y_train)

    importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_,
    })

    importance_df = importance_df.sort_values(
    "importance", ascending=False
    )

    importance_df["importance_percent"] = (
    importance_df["importance"] / importance_df["importance"].sum() * 100
    )

    print("\nFeature importance:")
    print(importance_df.to_string(index=False))

    importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False,
    )

    print(f"\nSaved feature importance to: {OUTPUT_DIR / 'feature_importance.csv'}")

    importance = (
    importance_df.set_index("feature")
    .loc[feature_names, "importance"]
    .to_numpy(dtype=np.float64)
    )

    if importance.sum() <= 0:
        raise ValueError("Feature importance sum must be positive.")

    importance = importance / importance.sum()

    generate_click_images(
        X_train_scaled,
        feature_names,
        importance,
        OUTPUT_DIR / "images" / "train",
    )

    generate_click_images(
        X_val_scaled,
        feature_names,
        importance,
        OUTPUT_DIR / "images" / "validation",
    )

    generate_click_images(
    X_test_scaled,
    feature_names,
    importance,
    OUTPUT_DIR / "images" / "test",
    )



    print("\nFeature preparation completed.")
    print("No model has been trained yet.")


if __name__ == "__main__":
    main()