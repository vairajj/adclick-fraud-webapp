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
