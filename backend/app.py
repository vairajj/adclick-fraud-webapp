import os
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline import ClickHistory, V2_FEATURES, build_v2_features


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "v2_random_forest.joblib",
)
DATA_DIR = os.path.join(BASE_DIR, "data")

RESULT_FILES = [
    ("Baseline", "model_results_baseline.csv"),
    ("V1", "model_results_v1.csv"),
    ("V2", "model_results_v2.csv"),
    ("V3 no-IP", "model_results_v3_no_ip.csv"),
]


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Ad Click Attribution & Suspicious Click Detection API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Model + History
# ============================================================

_model = None
history = ClickHistory()

# Recent predictions for the current API session.
prediction_history = []
MAX_PREDICTION_HISTORY = 20

def load_model():
    global _model

    if _model is None:

        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=(
                    "V2 model not found. "
                    "Expected backend/model/v2_random_forest.joblib"
                ),
            )

        _model = joblib.load(MODEL_PATH)

    return _model


def load_comparison_results():
    """Read experiment metrics from the saved result CSVs."""
    results = []

    for version, filename in RESULT_FILES:
        path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(path):
            continue

        frame = pd.read_csv(path)

        for _, row in frame.iterrows():
            results.append({
                "version": version,
                "model": str(row["Model"]),
                "precision": float(row["Precision"]),
                "recall": float(row["Recall"]),
                "f1": float(row["F1"]),
                "roc_auc": float(row["ROC_AUC"]),
                "pr_auc": float(row["PR_AUC"]),
                "deployed": version == "V2" and str(row["Model"]) == "Random Forest",
            })

    return results


# ============================================================
# Request
# ============================================================

class PredictRequest(BaseModel):

    ip: int = Field(..., ge=0)
    app: int = Field(..., ge=0)
    device: int = Field(..., ge=0)
    os: int = Field(..., ge=0)
    channel: int = Field(..., ge=0)

    click_time: datetime


# ============================================================
# Health
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "model_ready": os.path.exists(MODEL_PATH),
        "model": "v2_random_forest",
        "version": "2.0.0",
        "history": history.stats(),
    }


# ============================================================
# Model information
# ============================================================

@app.get("/api/model")
def model_info():

    return {
        "model": "Random Forest",
        "version": "V2",
        "features": V2_FEATURES,
        "feature_count": len(V2_FEATURES),
        "threshold": 0.50,
        "target": "is_attributed",
        "target_note": (
            "is_attributed represents attribution/app download "
            "in the TalkingData dataset; it is not direct "
            "ground-truth fraud."
        ),
    }


# ============================================================
# Statistics
# ============================================================

@app.get("/api/stats")
def stats():

    load_model()

    return {
        "model": "V2 Random Forest",
        "model_ready": True,
        "feature_count": len(V2_FEATURES),
        "features": V2_FEATURES,
        "threshold": 0.50,
        "history": history.stats(),

        "performance": {
            "roc_auc": 0.977599,
            "pr_auc": 0.300797,
            "precision": 0.212871,
            "recall": 0.728814,
            "f1": 0.329502,
        },
        "comparison": load_comparison_results(),
    }


# ============================================================
# Prediction
# ============================================================

@app.post("/api/predict")
def predict(req: PredictRequest):

    model = load_model()

    try:

        # Build historical features BEFORE recording
        # the current click.
        feature_row = build_v2_features(
            history=history,
            ip=req.ip,
            app=req.app,
            device=req.device,
            os_value=req.os,
            channel=req.channel,
            click_time=req.click_time,
        )

        feature_row = feature_row[V2_FEATURES]

        probability = float(
            model.predict_proba(feature_row)[0][1]
        )

        threshold = 0.50

        if probability >= threshold:
            label = "Attributed / Suspicious Pattern"
        else:
            label = "Not Attributed / Lower Risk Pattern"

        if probability >= 0.75:
            risk = "high"
        elif probability >= 0.50:
            risk = "medium"
        elif probability >= 0.25:
            risk = "low"
        else:
            risk = "minimal"

        # Behavioral signals
        signals = [
            {
                "feature": "ip_prev_clicks",
                "value": float(
                    feature_row.iloc[0]["ip_prev_clicks"]
                ),
                "description": "Previous clicks from this IP",
            },
            {
                "feature": "app_prev_clicks",
                "value": float(
                    feature_row.iloc[0]["app_prev_clicks"]
                ),
                "description": "Previous clicks from this app",
            },
            {
                "feature": "ip_app_prev_clicks",
                "value": float(
                    feature_row.iloc[0]["ip_app_prev_clicks"]
                ),
                "description": (
                    "Previous clicks from this IP/app combination"
                ),
            },
            {
                "feature": "ip_device_prev_clicks",
                "value": float(
                    feature_row.iloc[0]["ip_device_prev_clicks"]
                ),
                "description": (
                    "Previous clicks from this IP/device combination"
                ),
            },
            {
                "feature": "seconds_since_prev_ip_click",
                "value": float(
                    feature_row.iloc[0][
                        "seconds_since_prev_ip_click"
                    ]
                ),
                "description": (
                    "Time since previous click from this IP"
                ),
            },
        ]

        # Record current click AFTER prediction.
        history.record_click(
            ip=req.ip,
            app=req.app,
            device=req.device,
            os_value=req.os,
            channel=req.channel,
            click_time=pd.Timestamp(req.click_time),
        )

         # Store a lightweight record of this prediction.
        prediction_history.append({
            "click_time": req.click_time.isoformat(),
            "ip": req.ip,
            "app": req.app,
            "device": req.device,
            "channel": req.channel,
            "probability": probability,
            "label": label,
            "risk": risk,
        })

        # Keep only the most recent predictions.
        if len(prediction_history) > MAX_PREDICTION_HISTORY:
            del prediction_history[:-MAX_PREDICTION_HISTORY]

        return {
            "probability": probability,
            "threshold": threshold,
            "label": label,
            "risk": risk,
            "features": {
                key: float(value)
                for key, value in feature_row.iloc[0].items()
            },
            "behavioral_signals": signals,
            "history": history.stats(),
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )
    



# ============================================================
# Reset history
# ============================================================
@app.get("/api/history/recent")
def recent_history():

    return {
        "count": len(prediction_history),
        "predictions": prediction_history,
    }

@app.post("/api/history/reset")
def reset_history():

    history.clear()
    prediction_history.clear()

    return {
        "status": "ok",
        "message": "Click history and prediction history have been reset.",
        "history": history.stats(),
        "prediction_count": 0,
    }


# ============================================================
# Frontend
# ============================================================

if os.path.isdir(FRONTEND_DIR):

    app.mount(
        "/",
        StaticFiles(
            directory=FRONTEND_DIR,
            html=True,
        ),
        name="frontend",
    )
