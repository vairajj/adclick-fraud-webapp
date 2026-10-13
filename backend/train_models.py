
import pandas as pd
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# 1. Load dataset
df = pd.read_csv("data/click_fraud_dataset.csv")

TARGET = "is_fraudulent"

# 2. Extract useful timestamp features
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["click_hour"] = df["timestamp"].dt.hour
df["click_day"] = df["timestamp"].dt.dayofweek
df["click_month"] = df["timestamp"].dt.month

# 3. Remove identifiers, raw URLs, timestamp and suspicious score
drop_columns = [
    "click_id",
    "user_id",
    "ip_address",
    "timestamp",
    "referrer_url",
    "page_url",
    "bot_likelihood_score",
]

X = df.drop(columns=drop_columns + [TARGET])
y = df[TARGET]

# 4. Identify feature types
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# 5. Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features),
])

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 7. Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        class_weight="balanced",
        verbosity=-1,
        random_state=42,
        n_jobs=-1,
    ),
}

# 8. Train and evaluate
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)
    os.makedirs("model", exist_ok=True)

    model_filename = (
    name.lower().replace(" ", "_").replace("-", "_") + ".joblib"
    )

    joblib.dump(pipeline, os.path.join("model", model_filename))

    print(f"Saved model: model/{model_filename}")

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    result = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "PR-AUC": average_precision_score(y_test, probabilities),
    }

    results.append(result)

    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))


# 9. Display comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("PR-AUC", ascending=False)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# 10. Save results
results_df.to_csv("model_comparison.csv", index=False)

print("\nResults saved to model_comparison.csv")
