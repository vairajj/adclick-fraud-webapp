import pandas as pd
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ================================================================
# CONFIGURATION
# ================================================================

INPUT_FILE = Path("data/train_features_v2.csv")
RESULTS_FILE = Path("data/model_results_v2.csv")
THRESHOLD_FILE = Path("data/threshold_results_v2.csv")
MODEL_FILE = Path("model/v2_random_forest.joblib")

SPLIT_DATE = pd.Timestamp("2017-11-09")


print("=" * 75)
print("AD CLICK - V2 MODEL TRAINING")
print("HISTORICAL BEHAVIORAL FEATURES")
print("=" * 75)


# ================================================================
# 1. LOAD DATA
# ================================================================

print("\n[1/9] Loading V2 dataset...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["click_time"]
)

df = df.sort_values("click_time").reset_index(drop=True)

print(f"Total rows: {len(df):,}")


# ================================================================
# 2. CHRONOLOGICAL TRAIN / TEST SPLIT
# ================================================================

print("\n[2/9] Creating chronological train/test split...")

train_df = df[
    df["click_time"] < SPLIT_DATE
].copy()

test_df = df[
    df["click_time"] >= SPLIT_DATE
].copy()

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows:  {len(test_df):,}")

print(
    f"Training positives: "
    f"{train_df['is_attributed'].sum():,}"
)

print(
    f"Testing positives:  "
    f"{test_df['is_attributed'].sum():,}"
)


# ================================================================
# 3. SELECT FEATURES
# ================================================================

print("\n[3/9] Selecting V2 features...")

TARGET = "is_attributed"

DROP_COLUMNS = [
    TARGET,
    "attributed_time",
    "click_time",

    # Internal helper timestamps
    "prev_ip_click_time",
    "prev_ip_app_click_time",
    "prev_ip_device_click_time",

    # Unstable rate features
    "ip_clicks_per_hour",
    "ip_app_clicks_per_hour"
]

DROP_COLUMNS = [
    col
    for col in DROP_COLUMNS
    if col in train_df.columns
]

X_train = train_df.drop(columns=DROP_COLUMNS)
y_train = train_df[TARGET]

X_test = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df[TARGET]

print(f"Features used: {X_train.shape[1]}")


# ================================================================
# 4. FEATURE TYPES
# ================================================================

print("\n[4/9] Identifying feature types...")

categorical_features = [
    "app",
    "device",
    "os",
    "channel"
]

categorical_features = [
    col
    for col in categorical_features
    if col in X_train.columns
]

numerical_features = [
    col
    for col in X_train.columns
    if col not in categorical_features
]

print(
    f"Categorical features: {len(categorical_features)}"
)

print(
    f"Numerical features:   {len(numerical_features)}"
)


# ================================================================
# 5. PREPROCESSING
# ================================================================

print("\n[5/9] Building preprocessing pipeline...")

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ================================================================
# 6. RANDOM FOREST
# ================================================================

print("\n[6/9] Building Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ================================================================
# 7. TRAIN MODEL
# ================================================================

print("\n[7/9] Training Random Forest...")

pipeline.fit(
    X_train,
    y_train
)

print("Training complete.")


# ================================================================
# 8. EVALUATE + THRESHOLD TUNING
# ================================================================

print("\n[8/9] Evaluating model...")

probabilities = pipeline.predict_proba(
    X_test
)[:, 1]


# ------------------------------------------------
# Threshold-independent metrics
# ------------------------------------------------

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)

print("\nThreshold-independent metrics")
print("-" * 50)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")


# ------------------------------------------------
# Test multiple thresholds
# ------------------------------------------------

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

threshold_results = []

print("\nThreshold analysis")
print("=" * 90)

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Predicted_Positive": int(predictions.sum())
    })


threshold_df = pd.DataFrame(
    threshold_results
)

print(
    threshold_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ------------------------------------------------
# Find highest F1 threshold
# ------------------------------------------------

best_row = threshold_df.loc[
    threshold_df["F1"].idxmax()
]

best_threshold = float(
    best_row["Threshold"]
)

print("\n" + "=" * 90)
print("BEST F1 THRESHOLD")
print("=" * 90)

print(
    f"Threshold:          {best_threshold:.2f}"
)

print(
    f"Precision:          {best_row['Precision']:.4f}"
)

print(
    f"Recall:             {best_row['Recall']:.4f}"
)

print(
    f"F1:                 {best_row['F1']:.4f}"
)

print(
    f"True Positives:     {int(best_row['TP'])}"
)

print(
    f"False Positives:    {int(best_row['FP'])}"
)

print(
    f"False Negatives:    {int(best_row['FN'])}"
)

print(
    f"Predicted Positive: {int(best_row['Predicted_Positive'])}"
)


# ------------------------------------------------
# Save threshold results
# ------------------------------------------------

threshold_df.to_csv(
    THRESHOLD_FILE,
    index=False
)

print(
    f"\nThreshold results saved to: "
    f"{THRESHOLD_FILE}"
)


# ================================================================
# 9. SAVE MODEL
# ================================================================

print("\n[9/9] Saving trained model...")

MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    pipeline,
    MODEL_FILE
)

print(
    f"Model saved to: {MODEL_FILE}"
)


# ================================================================
# SAVE MAIN RESULT
# ================================================================

default_predictions = (
    probabilities >= 0.50
).astype(int)

precision = precision_score(
    y_test,
    default_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    default_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    default_predictions,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    default_predictions
).ravel()

result = pd.DataFrame([
    {
        "Version": "V2",
        "Model": "Random Forest",
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }
])

result.to_csv(
    RESULTS_FILE,
    index=False
)

print(
    f"Main results saved to: {RESULTS_FILE}"
)

print("\n" + "=" * 75)
print("V2 TRAINING + THRESHOLD ANALYSIS COMPLETE")
print("=" * 75)