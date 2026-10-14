
from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "c2idtl_experiment" / "images"
OUTPUT_DIR = BASE_DIR / "c2idtl_experiment" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 1e-5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ClickImageDataset(Dataset):
    def __init__(self, split):
        self.directory = IMAGE_DIR / split
        self.manifest = pd.read_csv(self.directory / "labels.csv")

        # Augmentation is applied only to training images.
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(
                    degrees=0,
                    shear=0.2,
                    scale=(0.8, 1.2),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest.iloc[index]
        image = Image.open(
            self.directory / row["filename"]
        ).convert("RGB")

        image = self.transform(image)
        label = torch.tensor(
            int(row["label"]), dtype=torch.float32
        )
        return image, label


def evaluate(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities.extend(
                torch.sigmoid(logits).cpu().squeeze(1).tolist()
            )
            labels.extend(batch_labels.tolist())
    return labels, probabilities


def select_threshold(labels, probabilities):
    rows = []
    for threshold in np.linspace(0.01, 0.99, 197):
        predictions = [int(p >= threshold) for p in probabilities]
        rows.append((
            f1_score(labels, predictions, zero_division=0),
            float(threshold),
        ))
    return max(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Train the C2IDTL ResNet50 attribution model."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--output-name",
        default="resnet50_best.pth",
        help="Checkpoint filename under c2idtl_experiment/models.",
    )
    args = parser.parse_args()
    set_seed(args.seed)

    print(f"Device: {DEVICE}")
    print(f"Seed: {args.seed}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")

    train_dataset = ClickImageDataset("train")
    val_dataset = ClickImageDataset("validation")
    test_dataset = ClickImageDataset("test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # ImageNet-pretrained ResNet-50
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # Freeze the pretrained backbone.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace the classifier with the paper-inspired head.
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Linear(512, 64),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(64, 1),
    )

    model = model.to(DEVICE)

    # Handle imbalance using training labels only.
    positives = int(train_dataset.manifest["label"].sum())
    negatives = len(train_dataset) - positives

    pos_weight = torch.tensor(
        [negatives / max(positives, 1)],
        device=DEVICE,
        dtype=torch.float32,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimize the new classifier only.
    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=args.learning_rate,
    )

    best_val_pr_auc = float("-inf")
    best_path = OUTPUT_DIR / args.output_name
    metrics_path = best_path.with_suffix(".json")

    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(
                DEVICE, non_blocking=True
            ).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)

        train_loss = total_loss / len(train_dataset)

        model.eval()
        val_loss = 0.0
        val_probs, val_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels_gpu = labels.to(
                    DEVICE, non_blocking=True
                ).unsqueeze(1)

                logits = model(images)
                loss = criterion(logits, labels_gpu)
                val_loss += loss.item() * len(labels)

                val_probs.extend(
                    torch.sigmoid(logits)
                    .cpu().squeeze(1).tolist()
                )
                val_labels.extend(labels.tolist())

        val_loss /= len(val_dataset)
        val_auc = roc_auc_score(val_labels, val_probs)
        val_pr_auc = average_precision_score(val_labels, val_probs)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_roc_auc": val_auc,
            "val_pr_auc": val_pr_auc,
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_auc={val_auc:.4f} | "
            f"val_pr_auc={val_pr_auc:.4f}"
        )

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            torch.save(model.state_dict(), best_path)

    # Evaluate the best checkpoint. Select the classification threshold on
    # validation data only, then apply it once to the held-out test split.
    model.load_state_dict(
        torch.load(best_path, map_location=DEVICE)
    )
    val_labels, val_probs = evaluate(model, val_loader, DEVICE)
    test_labels, test_probs = evaluate(model, test_loader, DEVICE)
    val_f1, threshold = select_threshold(val_labels, val_probs)
    predictions = [int(p >= threshold) for p in test_probs]
    test_metrics = {
        "roc_auc": roc_auc_score(test_labels, test_probs),
        "pr_auc": average_precision_score(test_labels, test_probs),
        "precision": precision_score(
            test_labels, predictions, zero_division=0
        ),
        "recall": recall_score(test_labels, predictions, zero_division=0),
        "f1": f1_score(test_labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(
            test_labels, predictions
        ).tolist(),
    }

    print(f"\nValidation-selected threshold: {threshold:.2f}")
    print(f"Validation F1 at threshold: {val_f1:.4f}")
    print("\nHeld-out test metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    metrics_path.write_text(json.dumps({
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "checkpoint": str(best_path),
        "validation_threshold": threshold,
        "validation_f1": val_f1,
        "history": history,
        "test": test_metrics,
    }, indent=2))
    print(f"Saved checkpoint: {best_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
