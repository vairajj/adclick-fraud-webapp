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


TRAIN_SOURCE = Path("data/train_features_v1.csv")
TEST_SOURCE = Path("data/train_features_v1.csv")

TRAIN_OUTPUT = Path("data/train_v1.csv")
TEST_OUTPUT = Path("data/test_v1.csv")


print("=" * 75)
print("AD CLICK - V1 MODEL TRAINING")
print("GLOBAL FREQUENCY + INTERACTION FEATURES")
print("=" * 75)


# ================================================================
# 1. LOAD V1 DATA
# ================================================================

print("\n[1/8] Loading V1 dataset...")

df = pd.read_csv(
    TRAIN_SOURCE,
    parse_dates=["click_time"]
)

df = df.sort_values("click_time").reset_index(drop=True)

print(f"Total rows: {len(df):,}")


# ================================================================
# 2. CREATE SAME CHRONOLOGICAL SPLIT
# ================================================================

print("\n[2/8] Creating chronological train/test split...")

split_date = pd.Timestamp("2017-11-09")

train_df = df[df["click_time"] < split_date].copy()
test_df = df[df["click_time"] >= split_date].copy()

train_df.to_csv(TRAIN_OUTPUT, index=False)
test_df.to_csv(TEST_OUTPUT, index=False)

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows:  {len(test_df):,}")

print(f"Training positives: {train_df['is_attributed'].sum():,}")
print(f"Testing positives:  {test_df['is_attributed'].sum():,}")


# ================================================================
# 3. PREPARE FEATURES
# ================================================================

print("\n[3/8] Preparing V1 features...")

TARGET = "is_attributed"

# Do NOT use attributed_time.
# Do NOT use click_time directly.
#
# V1 features:
# - temporal features
# - global frequency features
# - global interaction counts
# - global ratios

DROP_COLUMNS = [
    TARGET,
    "attributed_time",
    "click_time"
]

X_train = train_df.drop(columns=DROP_COLUMNS)
y_train = train_df[TARGET]

X_test = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df[TARGET]


# ================================================================
# 4. FEATURE TYPES
# ================================================================

print("\n[4/8] Identifying feature types...")

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
# 5. PREPROCESSING
# ================================================================

print("\n[5/8] Building preprocessing pipeline...")

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
# 6. MODELS
# ================================================================

print("\n[6/8] Creating models...")

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
# 7. TRAIN + EVALUATE
# ================================================================

print("\n[7/8] Training and evaluating...")
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

    predictions = (
        probabilities >= 0.5
    ).astype(int)

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
    print(f"TN = {tn:,}    FP = {fp:,}")
    print(f"FN = {fn:,}    TP = {tp:,}")

    print(f"\nPredicted positives: {predictions.sum():,}")

    results.append({
        "Version": "V1",
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
# 8. SAVE RESULTS
# ================================================================

print("\n[8/8] Saving V1 results...")

results_df = pd.DataFrame(results)

print("\nV1 MODEL COMPARISON")
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

output_file = Path(
    "data/model_results_v1.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print(f"\nSaved to: {output_file}")

print("\n" + "=" * 75)
print("V1 TRAINING COMPLETE")
print("=" * 75)