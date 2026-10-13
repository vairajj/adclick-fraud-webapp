from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


DATA_FILE = Path("data/train_sample.csv")
OUTPUT_DIR = Path("c2idtl_prototype_output")
IMAGE_SIZE = 224
NUMBER_OF_IMAGES = 5
RANDOM_STATE = 42


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the TalkingData sample.
    df = pd.read_csv(DATA_FILE)

    required = {
        "ip", "app", "device", "os", "channel",
        "click_time", "is_attributed",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Convert click_time to date/time features.
    click_time = pd.to_datetime(df["click_time"], errors="coerce")
    df["year"] = click_time.dt.year
    df["month"] = click_time.dt.month
    df["day"] = click_time.dt.day
    df["hour"] = click_time.dt.hour
    df["minute"] = click_time.dt.minute
    df["second"] = click_time.dt.second

    # Drop incomplete rows after time feature extraction.
    df = df.dropna(
        subset=[
            "ip", "app", "device", "os", "channel",
            "year", "month", "day", "hour", "minute",
            "second", "is_attributed",
        ]
    ).copy()

    # Exclude attributed_time: it is overwhelmingly missing and
    # is not used as an input feature in this prototype.
    feature_columns = [
        "ip", "app", "device", "os", "channel",
        "year", "month", "day", "hour", "minute", "second",
    ]

    X = df[feature_columns].astype(np.float64)
    y = df["is_attributed"].astype(int)

    # Fit the scaler on this prototype dataset.
    # A rigorous evaluation must fit preprocessing only on each
    # training fold to prevent leakage.
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # Random Forest feature importance.
    # Class weighting is used here to account for severe imbalance;
    # the paper's exact RF settings are not assumed.
    forest = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    forest.fit(X_scaled, y)

    importances = forest.feature_importances_
    if importances.sum() == 0:
        raise RuntimeError("Random Forest produced zero total feature importance.")

    # Implementation choice: allocate pixels in proportion to
    # normalized feature importance, using a flat sequential layout.
    weights = importances / importances.sum()
    total_pixels = IMAGE_SIZE * IMAGE_SIZE
    raw_counts = weights * total_pixels
    pixel_counts = np.floor(raw_counts).astype(int)

    # Assign remaining pixels by largest fractional remainder so
    # allocations sum to exactly 224*224.
    remaining = total_pixels - int(pixel_counts.sum())
    fractions = raw_counts - pixel_counts
    for idx in np.argsort(fractions)[::-1][:remaining]:
        pixel_counts[idx] += 1

    # Save metadata for reproducibility.
    metadata = {
        "source_dataset": str(DATA_FILE),
        "rows_used": int(len(df)),
        "feature_columns": feature_columns,
        "label_column": "is_attributed",
        "label_warning": (
            "App-install attribution is not independently verified click fraud."
        ),
        "image_size": [IMAGE_SIZE, IMAGE_SIZE, 3],
        "number_of_images": min(NUMBER_OF_IMAGES, len(df)),
        "initial_matrix_dimensions": (
            "Not specified by the paper; prototype uses a flat feature sequence."
        ),
        "pixel_allocation": (
            "Normalized Random Forest importance; largest-remainder rounding."
        ),
        "features": {
            name: {
                "importance": float(importance),
                "pixel_count": int(count),
            }
            for name, importance, count in zip(
                feature_columns, importances, pixel_counts
            )
        },
    }

    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Generate only a few images for validation.
    # Every feature occupies a consecutive block of pixels.
    for row_idx in range(min(NUMBER_OF_IMAGES, len(df))):
        pixels = np.empty(total_pixels, dtype=np.float32)
        start = 0

        for feature_idx, count in enumerate(pixel_counts):
            end = start + int(count)
            pixels[start:end] = X_scaled[row_idx, feature_idx]
            start = end

        grayscale = np.clip(
            np.round(pixels.reshape(IMAGE_SIZE, IMAGE_SIZE) * 255),
            0,
            255,
        ).astype(np.uint8)

        # Repeat the same grayscale values in R, G, and B.
        rgb = np.repeat(grayscale[:, :, None], 3, axis=2)
        image = Image.fromarray(rgb, mode="RGB")
        image.save(OUTPUT_DIR / f"click_{row_idx:03d}.png")

    print("C2IDTL prototype completed.")
    print("Rows processed:", len(df))
    print("Features:", len(feature_columns))
    print("Class counts:", y.value_counts().sort_index().to_dict())
    print("Images generated:", min(NUMBER_OF_IMAGES, len(df)))
    print("Image dimensions:", f"{IMAGE_SIZE}x{IMAGE_SIZE} RGB")
    print("Output directory:", OUTPUT_DIR.resolve())
    print("Feature importance saved to metadata.json")


if __name__ == "__main__":
    main()