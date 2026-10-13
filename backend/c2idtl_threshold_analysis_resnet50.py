
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from c2idtl_train_resnet50 import ClickImageDataset, DEVICE, BATCH_SIZE

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "c2idtl_experiment" / "models" / "resnet50_best.pth"
OUTPUT_DIR = BASE_DIR / "c2idtl_experiment"

def main():
    val_dataset = ClickImageDataset("validation")
    test_dataset = ClickImageDataset("test")

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = models.resnet50(weights=None)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Linear(512, 64),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(64, 1),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    def predict(loader):
        probabilities, labels = [], []
        with torch.no_grad():
            for images, batch_labels in loader:
                images = images.to(DEVICE, non_blocking=True)
                logits = model(images)
                probabilities.extend(
                    torch.sigmoid(logits).cpu().squeeze(1).tolist()
                )
                labels.extend(batch_labels.tolist())
        return labels, probabilities

    val_y, val_p = predict(val_loader)
    test_y, test_p = predict(test_loader)

    # Select a threshold using VALIDATION data only.
    rows = []
    for threshold in [i / 100 for i in range(1, 100)]:
        pred = [int(p >= threshold) for p in val_p]
        rows.append({
            "threshold": threshold,
            "precision": precision_score(
                val_y, pred, zero_division=0
            ),
            "recall": recall_score(
                val_y, pred, zero_division=0
            ),
            "f1": f1_score(val_y, pred, zero_division=0),
        })

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "validation_thresholds.csv", index=False)

    # Best F1 threshold on validation, not test.
    best = results.loc[results["f1"].idxmax()]
    threshold = float(best["threshold"])

    print("VALIDATION RESULTS")
    print(f"ROC-AUC: {roc_auc_score(val_y, val_p):.4f}")
    print(f"PR-AUC (average precision): "
          f"{average_precision_score(val_y, val_p):.4f}")
    print(f"Best F1 threshold: {threshold:.2f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall: {best['recall']:.4f}")
    print(f"F1-score: {best['f1']:.4f}")

    # Apply the chosen validation threshold to test exactly once.
    test_pred = [int(p >= threshold) for p in test_p]

    print("\nHELD-OUT TEST RESULTS")
    print(f"Threshold: {threshold:.2f}")
    print(f"ROC-AUC: {roc_auc_score(test_y, test_p):.4f}")
    print(f"PR-AUC (average precision): "
          f"{average_precision_score(test_y, test_p):.4f}")
    print(f"Precision: "
          f"{precision_score(test_y, test_pred, zero_division=0):.4f}")
    print(f"Recall: "
          f"{recall_score(test_y, test_pred, zero_division=0):.4f}")
    print(f"F1-score: "
          f"{f1_score(test_y, test_pred, zero_division=0):.4f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(test_y, test_pred))

    print(f"\nThreshold table saved to: "
          f"{OUTPUT_DIR / 'validation_thresholds.csv'}")

if __name__ == "__main__":
    main()
