# Ad Click Fraud Detection — Web App

A full-stack version of the original [adclick-fraud](https://github.com/vairajj/adclick-fraud) ML
pipeline: same 8-model comparison and evaluation, now wrapped in a FastAPI backend and a
dashboard + live prediction UI.

## What's inside

```
backend/
  pipeline.py      # feature engineering, shared by training and the live API
  train.py         # trains & compares all 8 models, saves the best one
  app.py           # FastAPI server: /api/stats, /api/predict, serves the frontend
  requirements.txt
frontend/
  index.html       # Dashboard + Live Check tabs
  style.css
  script.js
```

## Setup

```bash
cd backend
pip install -r requirements.txt

# Drop your dataset in backend/, named advertising.csv
# (download it from https://www.kaggle.com/datasets/gabrielsantello/advertisement-click-on-ad)
python3 train.py

uvicorn app:app --reload
```

Open **http://127.0.0.1:8000** — the Dashboard tab shows the model comparison, feature
importance and confusion matrix; the Live Check tab lets you enter a visitor's session details
and get a fraud probability with the top contributing factors.

If you run `train.py` without `advertising.csv` present, it falls back to a synthetic dataset
so you can see the whole app working immediately — swap in the real dataset before you present
your results.

## How it fits together

- `pipeline.py` has the exact preprocessing/feature-engineering steps from the original
  `adclick.py` (timestamp → hour/day/month, `time_per_usage`, `age_bin`, `income_per_usage`,
  `is_peak_hour`, `is_weekend`). Both `train.py` and `app.py` import it, so the live predictions
  always see features shaped exactly like what the model was trained on.
- `train.py` reruns the original 10-fold CV comparison across Logistic Regression, Decision
  Tree, Random Forest, KNN, ANN, Gradient Boosting, Naive Bayes and SVM, then **deploys the
  model with the best ROC-AUC** and writes `backend/model/`:
  - `model.pkl`, `scaler.pkl`, `features.json` — what `/api/predict` loads
  - `stats.json` — what `/api/stats` serves to the dashboard
- `app.py` is a thin FastAPI layer: health check, stats, and a `/api/predict` endpoint that
  engineers features for one row, scales it, and returns a probability + risk band + the
  features that influenced it most (importance-weighted distance from the mean).

## Ideas for extending this as your final project

Pick one or two of these to give the report a "we went beyond the baseline" angle:

- **Class imbalance**: real ad-fraud datasets are heavily skewed. Try the larger
  [TalkingData AdTracking Fraud dataset](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection)
  and add SMOTE/undersampling to `train.py`.
- **Explainability**: swap the rough importance-weighted contribution in `/api/predict` for real
  SHAP values — `shap.TreeExplainer` for RF/GB, `shap.LinearExplainer` for LR.
- **Sequence modeling**: add an LSTM over a simulated per-user click history to catch bursty,
  bot-like click patterns the current row-level models can't see.
- **Auth + history**: add a login and store past checks in SQLite so the dashboard can show
  "checks over time," which is a natural demo moment for a viva.
- **Deploy it**: put the backend on Render/Railway (free tier) and the frontend anywhere static,
  so you can share a live link instead of running it locally for the demo.

## Notes

- CORS is wide open (`allow_origins=["*"]`) for local development — tighten this before
  deploying anywhere public.
- The model is retrained by running `train.py` again; `app.py` doesn't retrain on the fly.
