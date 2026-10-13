
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/click_fraud_dataset.csv")

y = df["is_fraudulent"]

excluded = [
    "click_id",
    "user_id",
    "ip_address",
    "timestamp",
    "referrer_url",
    "page_url",
    "is_fraudulent",
]

X = df.drop(columns=excluded)

X["timestamp_hour"] = pd.to_datetime(
    df["timestamp"]
).dt.hour

# Model A: exclude bot score
X_without_bot = X.drop(columns=["bot_likelihood_score"])

# Model B: include bot score (diagnostic only)
X_with_bot = X.copy()

def evaluate(X_data, name):
    categorical = X_data.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numeric = X_data.select_dtypes(
        include=["number"]
    ).columns.tolist()

    preprocessor = ColumnTransformer([
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical,
        ),
        ("num", "passthrough", numeric),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X_data,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    print(f"\n========== {name} ==========")
    print(classification_report(y_test, predictions))
    print("ROC-AUC:", roc_auc_score(y_test, probabilities))
    print("PR-AUC:", average_precision_score(y_test, probabilities))


evaluate(X_without_bot, "MODEL A - WITHOUT BOT SCORE")
evaluate(X_with_bot, "MODEL B - WITH BOT SCORE (DIAGNOSTIC ONLY)")
