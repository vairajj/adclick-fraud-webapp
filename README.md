# Ad Click Attribution & Suspicious-Click Analysis Web App

A full-stack machine learning system for **ad-click attribution prediction and behavioral analysis** using clickstream data.

This project is the **7th-semester/final-year continuation** of the earlier [`adclick-fraud`](https://github.com/vairajj/adclick-fraud) project. The previous project focused mainly on comparing multiple machine-learning algorithms. This continuation focuses on improving the data-processing pipeline, chronological behavioral feature engineering, leakage prevention, model evaluation, and deployment through a FastAPI web application.

> **Important:** The target variable `is_attributed` in the TalkingData dataset represents whether a click resulted in an attributed app download. It is **not a verified ground-truth fraud label**. Therefore, this project uses the terminology **ad-click attribution prediction** and **suspicious-click behavioral analysis** rather than claiming to directly identify confirmed fraud.

---

## 📌 Project Overview

The system takes clickstream information such as:

- IP address
- Application
- Device
- Operating system
- Channel
- Click timestamp

and generates historical behavioral features based on events that occurred **before the current click**.

A trained Random Forest model then predicts the probability of attribution.

The result is presented through a web dashboard that includes:

- Model performance
- Dataset statistics
- Feature groups
- Model comparison
- Behavioral feature pipeline
- Live prediction
- Prediction history
- Session reset

---

# 🔄 Project Evolution

## 6th-Semester Project

The previous project compared eight machine-learning algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors
5. Artificial Neural Network
6. Gradient Boosting
7. Naive Bayes
8. Support Vector Machine

The main focus was broad algorithm comparison and model evaluation.

Previous repository:

https://github.com/vairajj/adclick-fraud

---

## 7th-Semester / Final-Year Continuation

Instead of simply repeating the eight-model comparison, the project was extended toward a more realistic clickstream machine-learning pipeline.

The current work focuses on:

- Data cleaning
- Exploratory data analysis
- Chronological feature engineering
- Historical behavioral features
- Leakage-aware feature construction
- Baseline, V1, V2 and V3 experiments
- Imbalanced classification evaluation
- Random Forest deployment
- FastAPI backend
- Interactive web dashboard
- Live prediction
- Prediction history

### Current pipeline

```text
                 CLICKSTREAM DATA
                        │
                        ▼
                ┌───────────────┐
                │ Data Cleaning │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │      EDA      │
                └───────┬───────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Feature Engineering │
             │     V1 / V2 / V3    │
             └──────────┬──────────┘
                        │
                        ▼
                ┌───────────────┐
                │ Model Training│
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │ Random Forest │
                │      V2       │
                └───────┬───────┘
                        │
                        ▼
                 ┌─────────────┐
                 │   FastAPI   │
                 │   Backend   │
                 └──────┬──────┘
                        │
                        ▼
                 ┌─────────────┐
                 │  Web        │
                 │  Dashboard  │
                 └─────────────┘# Ad Click Attribution & Suspicious-Click Analysis Web App

A full-stack machine-learning system for predicting **ad-click attribution** from clickstream data and analyzing historical behavioral patterns associated with suspicious clicking activity.

This project is the **7th-semester/final-year continuation** of the earlier `adclick-fraud` project. The previous project focused on comparing eight machine-learning algorithms. This continuation focuses on improving the data pipeline, chronological behavioral feature engineering, leakage prevention, model evaluation, and deployment through a FastAPI web application.

> **Important:** The TalkingData `is_attributed` label represents whether a click resulted in an attributed app download. It is not a verified ground-truth fraud label. Therefore, this project uses attribution prediction and behavioral analysis rather than claiming to directly identify confirmed fraud.

---

## Project Evolution

### 6th-Semester Project

The previous project compared eight machine-learning algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors
5. Artificial Neural Network
6. Gradient Boosting
7. Naive Bayes
8. Support Vector Machine

The focus was primarily on broad model comparison and evaluation.

### 7th-Semester / Final-Year Continuation

The project was extended by focusing on the data and deployment pipeline:

```text
TalkingData Clickstream Dataset
              ↓
       Data Cleaning
              ↓
       Exploratory Analysis
              ↓
   Chronological Feature Engineering
              ↓
     Baseline / V1 / V2 / V3
              ↓
       Model Evaluation
              ↓
      Random Forest V2
              ↓
        FastAPI Backend
              ↓
       Web Dashboard
              ↓
      Live Prediction System
```

## C2IDTL ResNet50 experiment

The C2IDTL attribution model uses a fixed, stratified 60/20/20
train/validation/held-out-test split (`random_state=42`). Numeric click
features are converted to a 224 x 224 grayscale feature-block image, copied
across RGB channels, and normalized with the ImageNet transform expected by
ResNet50. Training-only medians, scaling, and feature allocation are recorded
in `backend/c2idtl_experiment/feature_importance.csv`.

The original checkpoint trained only the classifier head for three epochs at
`1e-5`. A controlled experiment kept the same architecture, split, class
weighting, and augmentation but trained for five epochs at `1e-3` with seed
42. The improved checkpoint is saved separately as
`backend/c2idtl_experiment/models/resnet50_lr1e3_best.pth`; it does not
overwrite `resnet50_best.pth`. Configuration and measured results are
recorded in
`backend/c2idtl_experiment/resnet50_lr1e3_experiment.json`.

| Held-out test metric | Original (`0.50`) | Improved (`0.79` validation threshold) |
| --- | ---: | ---: |
| ROC-AUC | 0.7351 | 0.8331 |
| PR-AUC | 0.0106 | 0.0364 |
| Precision | 0.0070 | 0.0756 |
| Recall | 0.4222 | 0.2000 |
| F1 | 0.0138 | 0.1098 |

The improved threshold is selected on validation data only. The attribution
probability remains the model's raw sigmoid output; changing the threshold
does not inflate displayed probabilities. C2IDTL scores represent
`is_attributed` likelihood, not verified fraud probability.

### Reproduce the experiment

From the repository root, with the C2IDTL virtual environment configured:

```bash
cd backend
python c2idtl_train.py
python c2idtl_train_resnet50.py \
  --epochs 5 \
  --learning-rate 0.001 \
  --batch-size 64 \
  --seed 42 \
  --output-name resnet50_lr1e3_best.pth
```

The API automatically selects `resnet50_lr1e3_best.pth` when it exists and
falls back to `resnet50_best.pth` for rollback. Set `C2IDTL_MODEL_PATH` to
explicitly select either checkpoint.
