import pandas as pd
import numpy as np

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
    confusion_matrix,
    classification_report
)


# ================================================================
# CONFIGURATION
# ================================================================

TRAIN_FILE = Path("data/train_v3.csv")
TEST_FILE = Path("data/test_v3.csv")

print("=" * 75)
print("AD CLICK - V3-A MODEL TRAINING")
print("RAW IP REMOVED")
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
# 2. PREPARE FEATURES
# ================================================================

print("\n[2/7] Preparing features...")

TARGET = "is_attributed"

# IMPORTANT:
# attributed_time is unavailable when predicting a new click.
#
# click_time itself is removed because we already extracted:
# hour, day, day_of_week.
#
# Raw IP is deliberately removed because it is an identifier,
# not a meaningful continuous numerical quantity.

DROP_COLUMNS = [
    TARGET,
    "attributed_time",
    "click_time",
    "ip"
]

X_train = train_df.drop(columns=DROP_COLUMNS)
y_train = train_df[TARGET]

X_test = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df[TARGET]

print(f"Features after removing raw IP: {X_train.shape[1]}")


# ================================================================
# 3. FEATURE TYPES
# ================================================================

print("\n[3/7] Identifying feature types...")

categorical_features = [
    "app",
    "device",
    "os",
    "channel"
]

categorical_features = [
    col for col in categorical_features
    if col in X_train.columns
]

numerical_features = [
    col for col in X_train.columns
    if col not in categorical_features
]

print(f"Categorical features: {len(categorical_features)}")
print(f"Numerical features:   {len(numerical_features)}")

print("\nCategorical:")
print(categorical_features)


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

print("\n[5/7] Creating models...")

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

    # Default threshold
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

    cm = confusion_matrix(
        y_test,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    print("\nRESULTS")
    print("-" * 40)

    print(f"ROC-AUC:             {roc_auc:.4f}")
    print(f"PR-AUC:              {pr_auc:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1 Score:            {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nPrediction counts:")
    print(f"Actual positives:    {y_test.sum():,}")
    print(f"Predicted positives: {predictions.sum():,}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    results.append({
        "Model": model_name,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "True_Negative": tn,
        "False_Positive": fp,
        "False_Negative": fn,
        "True_Positive": tp
    })


# ================================================================
# 7. SAVE RESULTS
# ================================================================

print("\n[7/7] Saving results...")

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print("=" * 75)

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

output_file = Path("data/model_results_v3_no_ip.csv")

results_df.to_csv(
    output_file,
    index=False
)

print(f"\nSaved to: {output_file}")

print("\n" + "=" * 75)
print("V3-A TRAINING COMPLETE")
print("=" * 75)