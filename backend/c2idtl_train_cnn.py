
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "c2idtl_experiment" / "images"
OUTPUT_DIR = BASE_DIR / "c2idtl_experiment" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-4


class ClickImageDataset(Dataset):
    def __init__(self, split):
        self.directory = IMAGE_DIR / split
        self.manifest = pd.read_csv(self.directory / "labels.csv")
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
        label = torch.tensor(int(row["label"]), dtype=torch.float32)

        return image, label


def main():
    print(f"Device: {DEVICE}")

    train_dataset = ClickImageDataset("train")
    val_dataset = ClickImageDataset("validation")
    test_dataset = ClickImageDataset("test")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # ImageNet-pretrained ResNet18
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(DEVICE)

    # Address class imbalance using training labels only
    positives = int(train_dataset.manifest["label"].sum())
    negatives = len(train_dataset) - positives
    pos_weight = torch.tensor(
        [negatives / max(positives, 1)],
        device=DEVICE,
        dtype=torch.float32,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    best_path = OUTPUT_DIR / "resnet18_best.pth"

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).unsqueeze(1)

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
                    torch.sigmoid(logits).cpu().squeeze(1).tolist()
                )
                val_labels.extend(labels.tolist())

        val_loss /= len(val_dataset)
        val_auc = roc_auc_score(val_labels, val_probs)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_auc={val_auc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

    # Evaluate the best checkpoint on the held-out test set
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    test_probs, test_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            logits = model(images)

            test_probs.extend(
                torch.sigmoid(logits).cpu().squeeze(1).tolist()
            )
            test_labels.extend(labels.tolist())

    predictions = [int(p >= 0.5) for p in test_probs]

    print("\nTest metrics at threshold 0.5:")
    print(f"Accuracy:  {accuracy_score(test_labels, predictions):.4f}")
    print(f"Precision: {precision_score(test_labels, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(test_labels, predictions, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(test_labels, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(test_labels, test_probs):.4f}")
    print(f"\nSaved checkpoint: {best_path}")


if __name__ == "__main__":
    main()
