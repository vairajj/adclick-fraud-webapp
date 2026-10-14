import os
import joblib
import pandas as pd

from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from c2idtl_predict import C2IDTLPredictor

from database import engine, Base, get_db
import models


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "lightgbm.joblib")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")


app = FastAPI(
    title="Ad Click Fraud Detection API",
    description="Behavioural analysis and machine learning-based click fraud detection.",
    version="3.0.0",
)
# Initialize database tables
Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_c2idtl_model = None


def load_c2idtl_model():
    global _c2idtl_model

    if _c2idtl_model is None:
        try:
            _c2idtl_model = C2IDTLPredictor()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"C2IDTL model unavailable: {exc}",
            ) from exc

    return _c2idtl_model


def load_model():
    global _model

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail="LightGBM model not found.",
            )

        _model = joblib.load(MODEL_PATH)

    return _model


class PredictRequest(BaseModel):
    device_type: str = Field(..., pattern="^(Desktop|Mobile|Tablet)$")
    browser: str = Field(..., pattern="^(Chrome|Edge|Firefox|Opera|Safari)$")
    operating_system: str = Field(..., pattern="^(Android|Linux|Windows|iOS|macOS)$")
    ad_position: str = Field(..., pattern="^(Bottom|Side|Top)$")
    device_ip_reputation: str = Field(..., pattern="^(Bad|Good|Suspicious)$")

    click_duration: float = Field(..., ge=0, le=18.28)
    scroll_depth: float = Field(..., ge=0, le=99)
    mouse_movement: float = Field(..., ge=0, le=499)
    keystrokes_detected: int = Field(..., ge=0, le=49)
    click_frequency: int = Field(..., ge=1, le=9)
    time_since_last_click: float = Field(..., ge=1, le=599)
    VPN_usage: int = Field(..., ge=0, le=1)
    proxy_usage: int = Field(..., ge=0, le=1)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_ready": os.path.exists(MODEL_PATH),
        "model": "LightGBM",
        "version": "3.0.0",
    }


@app.get("/api/model")
def model_info():
    return {
        "model": "LightGBM",
        "version": "New Kaggle dataset",
        "target": "is_fraudulent",
        "features": [
            "click_duration",
            "scroll_depth",
            "mouse_movement",
            "keystrokes_detected",
            "click_frequency",
            "time_since_last_click",
            "VPN_usage",
            "proxy_usage",
            "device_type",
            "browser",
            "operating_system",
            "ad_position",
            "device_ip_reputation",
            "click_hour",
            "click_day",
            "click_month",
        ],
        "warning": "Experimental model. Dataset labels and predictive features require independent validation.",
    }


@app.post("/api/predict")
def predict(
    request: PredictRequest,
    db: Session = Depends(get_db),
):
    model = load_model()

    try:
        data = request.model_dump()

        now = datetime.now()
        data["click_hour"] = now.hour
        data["click_day"] = now.weekday()
        data["click_month"] = now.month

        feature_row = pd.DataFrame([data])

        probability = float(model.predict_proba(feature_row)[0][1])

        threshold = 0.50

        if probability >= threshold:
            label = "Potentially Suspicious"
        else:
            label = "Likely Legitimate"

        if probability >= 0.75:
            risk = "high"
        elif probability >= 0.50:
            risk = "medium"
        elif probability >= 0.25:
            risk = "low"
        else:
            risk = "minimal"

        # Save prediction to SQLite
        new_prediction = models.Prediction(
            **request.model_dump(),
            probability=probability,
            label=label,
            risk=risk,
        )

        db.add(new_prediction)
        db.commit()
        db.refresh(new_prediction)

        return {
            "probability": probability,
            "probability_percent": round(probability * 100, 2),
            "threshold": threshold,
            "label": label,
            "risk": risk,
            "model": "LightGBM",
            "warning": "Experimental prediction; not independently validated for real-world fraud detection.",
        }

    ##except Exception as e:
        ##raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/recent")
def get_recent_history(db: Session = Depends(get_db)):
    predictions = (
        db.query(models.Prediction)
        .order_by(models.Prediction.id.desc())
        .limit(20)
        .all()
    )

    return {
        "count": len(predictions),
        "predictions": [
            {
                "id": p.id,
                "click_time": p.click_time.isoformat(),
                "device_type": p.device_type,
                "browser": p.browser,
                "operating_system": p.operating_system,
                "probability": p.probability,
                "label": p.label,
                "risk": p.risk,
            }
            for p in predictions
        ],
    }


@app.get("/api/dashboard/stats")
def get_dashboard_stats(db: Session = Depends(get_db)):
    predictions = db.query(models.Prediction).all()

    total = len(predictions)

    suspicious = sum(1 for p in predictions if p.label == "Potentially Suspicious")

    legitimate = sum(1 for p in predictions if p.label == "Likely Legitimate")

    average_probability = (
        sum(p.probability for p in predictions) / total if total > 0 else 0
    )

    risk_distribution = {
        "high": sum(1 for p in predictions if p.risk == "high"),
        "medium": sum(1 for p in predictions if p.risk == "medium"),
        "low": sum(1 for p in predictions if p.risk == "low"),
        "minimal": sum(1 for p in predictions if p.risk == "minimal"),
    }

    return {
        "total_predictions": total,
        "suspicious_clicks": suspicious,
        "legitimate_clicks": legitimate,
        "average_probability": round(average_probability, 4),
        "risk_distribution": risk_distribution,
    }


@app.post("/api/history/reset")
def reset_history(db: Session = Depends(get_db)):
    deleted_count = db.query(models.Prediction).delete()

    db.commit()

    return {
        "message": "Prediction history cleared successfully.",
        "deleted_count": deleted_count,
    }

class C2IDTLPredictRequest(BaseModel):
    ip: int = Field(..., ge=0)
    app: int = Field(..., ge=0)
    device: int = Field(..., ge=0)
    os: int = Field(..., ge=0)
    channel: int = Field(..., ge=0)
    click_time: datetime



@app.post("/api/c2idtl/predict")
def c2idtl_predict(
    request: C2IDTLPredictRequest,
    db: Session = Depends(get_db),
):
    predictor = load_c2idtl_model()

    try:
        result = predictor.predict(request.model_dump())

        # Support either a dictionary or a Pydantic response object.
        if hasattr(result, "model_dump"):
            result_data = result.model_dump()
        elif isinstance(result, dict):
            result_data = result
        else:
            raise TypeError("Unexpected C2IDTL prediction response format")

        probability = result_data.get("probability")
        predicted_class = result_data.get(
            "predicted_class",
            result_data.get("prediction"),
        )

        if probability is None or predicted_class is None:
            raise ValueError(
                "C2IDTL result must contain probability and predicted class"
            )

        record = models.C2IDTLPrediction(
            click_time=request.click_time,
            ip=request.ip,
            app=request.app,
            device=request.device,
            os=request.os,
            channel=request.channel,
            probability=float(probability),
            predicted_class=int(predicted_class),
        )

        db.add(record)
        db.commit()

        return result_data

    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"C2IDTL prediction failed: {exc}",
        ) from exc

@app.get("/api/c2idtl/history")
def c2idtl_history(db: Session = Depends(get_db)):
    records = (
        db.query(models.C2IDTLPrediction)
        .order_by(models.C2IDTLPrediction.id.desc())
        .limit(50)
        .all()
    )

    return {
        "count": len(records),
        "predictions": [
            {
                "id": record.id,
                "click_time": record.click_time.isoformat(),
                "ip": record.ip,
                "app": record.app,
                "device": record.device,
                "os": record.os,
                "channel": record.channel,
                "probability": record.probability,
                "probability_percent": round(record.probability * 100, 2),
                "predicted_class": record.predicted_class,
            }
            for record in records
        ],
    }


if os.path.isdir(FRONTEND_DIR):
    app.mount(
        "/",
        StaticFiles(directory=FRONTEND_DIR, html=True),
        name="frontend",
    )
