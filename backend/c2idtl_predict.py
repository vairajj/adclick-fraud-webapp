
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train_sample.csv"
EXPERIMENT_DIR = BASE_DIR / "c2idtl_experiment"
IMAGE_DIR = EXPERIMENT_DIR / "images"
MODEL_DIR = EXPERIMENT_DIR / "models"
BASELINE_MODEL_PATH = MODEL_DIR / "resnet50_best.pth"
IMPROVED_MODEL_PATH = MODEL_DIR / "resnet50_lr1e3_best.pth"
MODEL_PATH = Path(os.getenv(
    "C2IDTL_MODEL_PATH",
    str(IMPROVED_MODEL_PATH if IMPROVED_MODEL_PATH.exists()
        else BASELINE_MODEL_PATH),
))
METRICS_PATH = MODEL_PATH.with_suffix(".json")
EXPERIMENT_METADATA_PATH = (
    EXPERIMENT_DIR / "resnet50_lr1e3_experiment.json"
)
IMPORTANCE_PATH = EXPERIMENT_DIR / "feature_importance.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def build_model():
    """Build the same ResNet50 architecture used during training."""
    model = models.resnet50(weights=None)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Linear(512, 64),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(64, 1),
    )

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    return model


def load_preprocessing():
    """Reconstruct preprocessing using the original training split."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not IMPORTANCE_PATH.exists():
        raise FileNotFoundError(
            f"Feature importance file not found: {IMPORTANCE_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().dropna(subset=["is_attributed"])
    df = df.reset_index(drop=True)

    X = df.drop(columns=["is_attributed", "attributed_time"])
    y = df["is_attributed"].astype(int)

    X_train_val, _, y_train_val, _ = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train, _, _, _ = train_test_split(
        X_train_val, y_train_val,
        test_size=0.25,
        random_state=42,
        stratify=y_train_val,
    )

    def numeric_features(frame):
        frame = frame.copy()
        click_time = pd.to_datetime(
            frame["click_time"], errors="coerce", utc=True
        ).dt.tz_localize(None)
        frame["year"] = click_time.dt.year
        frame["month"] = click_time.dt.month
        frame["day"] = click_time.dt.day
        frame["hour"] = click_time.dt.hour
        frame["minute"] = click_time.dt.minute
        frame["second"] = click_time.dt.second

        frame = frame.drop(columns=["click_time"])
        frame = frame.apply(pd.to_numeric, errors="coerce")
        return frame.replace([np.inf, -np.inf], np.nan)

    X_train = numeric_features(X_train)
    medians = X_train.median().fillna(0)
    X_train = X_train.fillna(medians)

    scaler = MinMaxScaler()
    scaler.fit(X_train)

    importance_df = pd.read_csv(IMPORTANCE_PATH)
    importance_map = importance_df.set_index("feature")["importance"]
    feature_names = list(X_train.columns)

    missing = [name for name in feature_names if name not in importance_map.index]
    if missing:
        raise ValueError(f"Missing feature importance values: {missing}")

    importance = importance_map.loc[feature_names].to_numpy(dtype=np.float64)
    if not np.isfinite(importance).all() or importance.sum() <= 0:
        raise ValueError("Invalid feature importance values.")

    importance /= importance.sum()

    # Ensure the saved allocation can fill exactly 224 × 224 pixels.
    total_pixels = IMAGE_SIZE * IMAGE_SIZE
    raw_counts = importance * total_pixels
    pixel_counts = np.floor(raw_counts).astype(int)
    remaining = total_pixels - int(pixel_counts.sum())

    if remaining > 0:
        fractions = raw_counts - pixel_counts
        order = np.argsort(-fractions)
        pixel_counts[order[:remaining]] += 1

    if pixel_counts.sum() != total_pixels:
        raise ValueError("Pixel allocation does not equal 224 × 224.")

    return feature_names, medians, scaler, importance



def row_to_image(row, feature_names, medians, scaler, importance):
    """Convert one click record to the same flat-block grayscale RGB image."""
    frame = pd.DataFrame([row])

    if "click_time" not in frame.columns:
        raise ValueError("click_time is required (example: 2017-11-08 12:34:56).")

    click_time = pd.to_datetime(frame["click_time"], errors="coerce")

    frame["year"] = click_time.dt.year
    frame["month"] = click_time.dt.month
    frame["day"] = click_time.dt.day
    frame["hour"] = click_time.dt.hour
    frame["minute"] = click_time.dt.minute
    frame["second"] = click_time.dt.second

    frame = frame.drop(columns=["click_time"])
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)

    for name in feature_names:
        if name not in frame.columns:
            frame[name] = np.nan

    frame = frame[feature_names].fillna(medians)
    scaled = scaler.transform(frame).astype(np.float32)[0]

    total_pixels = IMAGE_SIZE * IMAGE_SIZE
    raw_counts = importance * total_pixels
    pixel_counts = np.floor(raw_counts).astype(int)

    remaining = total_pixels - int(pixel_counts.sum())
    fractions = raw_counts - pixel_counts
    if remaining > 0:
        order = np.argsort(-fractions)
        pixel_counts[order[:remaining]] += 1

    flat = np.zeros(total_pixels, dtype=np.uint8)
    position = 0

    for value, count in zip(scaled, pixel_counts):
        end = position + count
        flat[position:end] = int(np.clip(value, 0, 1) * 255)
        position = end

    gray = flat.reshape(IMAGE_SIZE, IMAGE_SIZE)
    rgb = np.stack([gray] * 3, axis=-1)
    return Image.fromarray(rgb)


class C2IDTLPredictor:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

        self.feature_names, self.medians, self.scaler, self.importance = (
            load_preprocessing()
        )
        self.model = build_model()
        self.threshold = 0.5
        metadata_path = (
            METRICS_PATH
            if METRICS_PATH.exists()
            else EXPERIMENT_METADATA_PATH
        )
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            self.threshold = float(
                metadata.get("validation_threshold", self.threshold)
            )

    def predict(self, click):
        image = row_to_image(
            click,
            self.feature_names,
            self.medians,
            self.scaler,
            self.importance,
        )

        tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probability = torch.sigmoid(self.model(tensor)).item()

        return {
            "probability": round(float(probability), 6),
            "probability_percent": round(float(probability) * 100, 2),
            "predicted_class": int(probability >= self.threshold),
            "label_meaning": {
                "0": "Not attributed",
                "1": "Attributed",
            },
            "model": "C2IDTL-inspired ResNet50",
            "decision_threshold": self.threshold,
            "warning": (
                "Experimental. is_attributed is an attribution label, "
                "not independently verified click fraud."
            ),
        }


if __name__ == "__main__":
    predictor = C2IDTLPredictor()
    print("C2IDTL ResNet50 loaded successfully.")
    print("Device:", DEVICE)
    print("Features:", predictor.feature_names)
