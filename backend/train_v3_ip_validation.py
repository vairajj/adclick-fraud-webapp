import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

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
TARGET = "is_attributed"

print("=" * 75)
print("AD CLICK - V3 MODEL TRAINING")
print("=" * 75)


# ================================================================
# 1. LOAD DATA
# ================================================================

print("\n[1/8] Loading data...")

train_source = pd.read_csv(
    TRAIN_FILE,
    parse_dates=["click_time"]
)

test_source = pd.read_csv(
    TEST_FILE,
    parse_dates=["click_time"]
)

df = pd.concat([train_source, test_source], ignore_index=True)
df = df.sort_values("click_time").reset_index(drop=True)

VALIDATION_START = pd.Timestamp("2017-11-08")
TEST_START = pd.Timestamp("2017-11-09")

train_df = df[df["click_time"] < VALIDATION_START].copy()

validation_df = df[
    (df["click_time"] >= VALIDATION_START) &
    (df["click_time"] < TEST_START)
].copy()

test_df = df[df["click_time"] >= TEST_START].copy()

print(f"Training rows:   {len(train_df):,}")
print(f"Validation rows: {len(validation_df):,}")
print(f"Testing rows:    {len(test_df):,}")

print(f"Training positives:   {train_df[TARGET].sum():,}")
print(f"Validation positives: {validation_df[TARGET].sum():,}")
print(f"Testing positives:    {test_df[TARGET].sum():,}")

# ================================================================
# 2. REMOVE TARGET / NON-PREDICTIVE / LEAKAGE COLUMNS
# ================================================================

print("\n[2/8] Preparing features...")

TARGET = "is_attributed"

# attributed_time is only known after attribution.
# It must NOT be available when predicting a new click.
DROP_COLUMNS = [
    TARGET,
    "attributed_time",
    "click_time"
]

X_train = train_df.drop(columns=DROP_COLUMNS)
y_train = train_df[TARGET]

X_validation = validation_df.drop(columns=DROP_COLUMNS)
y_validation = validation_df[TARGET]

X_test = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df[TARGET]

print(f"Feature columns before preprocessing: {X_train.shape[1]}")


# ================================================================
# 3. IDENTIFY FEATURE TYPES
# ================================================================

print("\n[3/8] Identifying numerical and categorical features...")

categorical_features = [
    "app",
    "device",
    "os",
    "channel"
]

# Raw IP is intentionally excluded from the categorical list.
# It has very high cardinality and behaves more like an identifier.
#
# The behavioral IP features such as:
#   ip_prev_clicks
#   ip_clicks_last_1min
#   ip_clicks_last_5min
#   ip_clicks_last_1hour
# are retained.

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

print("\nNumerical:")
print(numerical_features)


# ================================================================
# 4. PREPROCESSING
# ================================================================

print("\n[4/8] Building preprocessing pipeline...")

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
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ================================================================
# 5. DEFINE MODELS
# ================================================================

print("\n[5/8] Creating models...")

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
    ),

}



# ================================================================
# 6. TRAIN AND EVALUATE
# ================================================================

print("\n[6/8] Training models...")
print("=" * 75)

results = []
trained_pipelines = {}

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
    trained_pipelines[model_name] = pipeline

    # Select threshold using validation data only
    print("Selecting threshold on validation data...")

    val_probabilities = pipeline.predict_proba(X_validation)[:, 1]


    val_precision, val_recall, thresholds = precision_recall_curve(
        y_validation, val_probabilities
    )

    f1_scores = (
        2 * val_precision[:-1] * val_recall[:-1]
        / (
            val_precision[:-1]
            + val_recall[:-1]
            + 1e-12
    )
)

    if len(thresholds) == 0:
        raise ValueError("No validation thresholds were generated.")

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_val_f1 = float(f1_scores[best_idx])

    print(f"Best validation threshold: {best_threshold:.6f}")
    print(f"Validation F1: {best_val_f1:.4f}")
    print(
    "Validation predicted positives:",
    int((val_probabilities >= best_threshold).sum())
    )


    # Evaluate on the untouched test set
    print("Evaluating on test data...")

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    test_roc_auc = roc_auc_score(y_test, probabilities)
    test_pr_auc = average_precision_score(y_test, probabilities)
    test_precision = precision_score(
        y_test, predictions, zero_division=0
    )
    test_recall = recall_score(
        y_test, predictions, zero_division=0
    )
    test_f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"Test PR-AUC:  {test_pr_auc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1:        {test_f1:.4f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    results.append({
        "Model": model_name,
        "Threshold": best_threshold,
        "Validation_F1": best_val_f1,
        "ROC_AUC": test_roc_auc,
        "PR_AUC": test_pr_auc,
        "Precision": test_precision,
        "Recall": test_recall,
        "F1": test_f1,
        "True_Negative": tn,
        "False_Positive": fp,
        "False_Negative": fn,
        "True_Positive": tp
    })

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

    print("\nDetailed classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )



# ================================================================
# FEATURE IMPORTANCE — RANDOM FOREST
# ================================================================

print("\n[Feature Importance] Random Forest")
print("=" * 75)

rf_pipeline = trained_pipelines.get("Random Forest")

if rf_pipeline is not None:
    rf_model = rf_pipeline.named_steps["model"]
    fitted_preprocessor = rf_pipeline.named_steps["preprocessor"]

    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    print("\nTop 20 features:")
    print(importance_df.head(20).to_string(index=False))

    importance_df.to_csv(
        "data/v3_ip_feature_importance.csv",
        index=False
    )
    print("\nSaved: data/v3_ip_feature_importance.csv")



# ================================================================
# 7. COMPARE MODELS
# ================================================================

print("\n[7/8] Model comparison")
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



# ================================================================
# 8. SAVE RESULTS
# ================================================================

OUTPUT_FILE = Path("data/model_results_v3_ip_validation.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nResults saved to: {OUTPUT_FILE}")
print(results_df.to_string(index=False))