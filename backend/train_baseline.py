import pandas as pd
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


TRAIN_FILE = Path("data/train_v3.csv")
TEST_FILE = Path("data/test_v3.csv")

print("=" * 75)
print("AD CLICK - BASELINE MODEL")
print("RAW CLICK + BASIC TIME FEATURES")
print("=" * 75)


# ================================================================
# 1. LOAD DATA
# ================================================================

print("\n[1/7] Loading data...")

train_df = pd.read_csv(
    TRAIN_FILE,
    parse_dates=["click_time"]
)

test_df = pd.read_csv(
    TEST_FILE,
    parse_dates=["click_time"]
)

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows:  {len(test_df):,}")

print(f"Training positives: {train_df['is_attributed'].sum():,}")
print(f"Testing positives:  {test_df['is_attributed'].sum():,}")


# ================================================================
# 2. CREATE BASELINE FEATURES
# ================================================================

print("\n[2/7] Selecting baseline features...")

TARGET = "is_attributed"

# Baseline deliberately excludes all engineered behavioral features.
#
# These are the basic click attributes plus simple temporal features.

BASELINE_FEATURES = [
    "ip",
    "app",
    "device",
    "os",
    "channel",
    "hour",
    "day",
    "day_of_week"
]

X_train = train_df[BASELINE_FEATURES].copy()
y_train = train_df[TARGET]

X_test = test_df[BASELINE_FEATURES].copy()
y_test = test_df[TARGET]

print("\nBaseline features:")
for feature in BASELINE_FEATURES:
    print(f"  - {feature}")


# ================================================================
# 3. FEATURE TYPES
# ================================================================

print("\n[3/7] Preparing feature types...")

categorical_features = [
    "ip",
    "app",
    "device",
    "os",
    "channel"
]

numerical_features = [
    "hour",
    "day",
    "day_of_week"
]


# ================================================================
# 4. PREPROCESSING
# ================================================================

print("\n[4/7] Building preprocessing pipeline...")

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
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
# 5. MODELS
# ================================================================

print("\n[5/7] Creating baseline models...")

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )
}


# ================================================================
# 6. TRAIN + EVALUATE
# ================================================================

print("\n[6/7] Training and evaluating...")
print("=" * 75)

results = []

for model_name, model in models.items():

    print(f"\n{'-' * 75}")
    print(f"MODEL: {model_name}")
    print(f"{'-' * 75}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    print("Training...")
    pipeline.fit(X_train, y_train)

    print("Predicting...")

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Default threshold = 0.5
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

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

    print("\nRESULTS")
    print("-" * 40)

    print(f"ROC-AUC:             {roc_auc:.4f}")
    print(f"PR-AUC:              {pr_auc:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1 Score:            {f1:.4f}")

    print("\nConfusion Matrix:")
    print(
        f"TN = {tn:,}    FP = {fp:,}"
    )
    print(
        f"FN = {fn:,}    TP = {tp:,}"
    )

    print("\nPredicted positives:", predictions.sum())

    results.append({
        "Model": model_name,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })


# ================================================================
# 7. SAVE RESULTS
# ================================================================

print("\n[7/7] Baseline comparison")
print("=" * 75)

results_df = pd.DataFrame(results)

print(
    results_df[
        [
            "Model",
            "ROC_AUC",
            "PR_AUC",
            "Precision",
            "Recall",
            "F1"
        ]
    ].to_string(index=False)
)

output_file = Path(
    "data/model_results_baseline.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print(f"\nSaved to: {output_file}")

print("\n" + "=" * 75)
print("BASELINE TRAINING COMPLETE")
print("=" * 75)