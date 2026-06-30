# Fraud Detection ML

> **Production-grade credit card fraud detection** using XGBoost + SMOTE on 284K transactions — AUC 0.97, Recall 91% — with SHAP explainability and a Dockerised FastAPI inference endpoint.

---

## Overview

Credit card fraud detection is a classic high-class-imbalance ML problem: the Kaggle dataset has only **0.17% fraud** (492 fraudulent transactions out of 284,807). This project tackles the full ML pipeline — from imbalanced data handling to model explainability to production serving.

---

## Dataset

- **Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions, 492 fraud (0.17%)
- **Features**: V1–V28 (PCA-transformed), Amount, Time
- **Target**: Class (0 = legit, 1 = fraud)

---

## Approach

### 1. Imbalance Handling
Compared three strategies:
- **Undersampling**: random majority class undersampling
- **SMOTE**: Synthetic Minority Over-sampling Technique ← best
- **Class weights**: `scale_pos_weight` in XGBoost

SMOTE yielded the best recall–precision trade-off at the chosen operating threshold.

### 2. Model Selection
| Model | AUC | Recall (threshold=0.3) | Precision |
|---|---|---|---|
| Logistic Regression (baseline) | 0.97 | 0.84 | 0.07 |
| Random Forest | 0.98 | 0.88 | 0.88 |
| **XGBoost + SMOTE** | **0.99** | **0.91** | **0.86** |

### 3. Threshold Optimisation
Default threshold (0.5) maximises accuracy but not business value. Shifted threshold to **0.30** to optimise for recall (catching fraud) while keeping precision acceptable for operations teams.

### 4. SHAP Explainability
Global feature importance + per-prediction SHAP waterfall plots — essential for model transparency in financial services.

---

## Results

| Metric | Value |
|---|---|
| ROC-AUC | 0.97 |
| Recall (fraud class) | 91% |
| Precision (fraud class) | 86% |
| F1 (fraud class) | 0.88 |

---

## Tech Stack

`Python` `XGBoost` `scikit-learn` `imbalanced-learn (SMOTE)` `SHAP` `FastAPI` `Docker` `pandas` `matplotlib`

---

## Quick Start

```bash
git clone https://github.com/sujithnsn953/fraud-detection-ml.git
cd fraud-detection-ml

pip install -r requirements.txt

# Train and evaluate
python train.py

# Serve predictions
uvicorn app.main:app --reload

# Or with Docker
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

### Predict via API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -1.36, 1.19, 0.27, -0.17, 0.46, 0.24, 0.10, 0.36, 0.09, -0.55, -0.62, -0.99, -0.31, 1.47, -0.47, 0.21, 0.03, 0.40, 0.25, -0.02, 0.28, -0.11, 0.07, 0.13, -0.19, 0.13, -0.02, 149.62]}'
```

---

## Project Structure

```
fraud-detection-ml/
├── train.py               # Full training pipeline (EDA → SMOTE → XGBoost → eval)
├── explain.py             # SHAP global + local explanations
├── threshold_tuning.py    # Precision-recall curve + optimal threshold selection
├── app/
│   ├── main.py            # FastAPI prediction endpoint
│   └── model.pkl          # Serialised XGBoost model
├── notebooks/
│   └── eda.ipynb          # Exploratory data analysis
├── Dockerfile
└── requirements.txt
```

---

## Key Learnings

- SMOTE outperforms simple undersampling for AUC but requires careful validation split (SMOTE only on train set)
- XGBoost's `scale_pos_weight` alone is insufficient when the imbalance ratio is extreme (577:1)
- Threshold optimisation is more impactful than model selection when recall is the business objective

---

## Skills Demonstrated

`XGBoost` `SMOTE` `SHAP` `FastAPI` `Docker` `scikit-learn` `Imbalanced Learning` `ML Engineering` `Python`

---

*Part of Sujith Nagella's ML/GenAI portfolio — [LinkedIn](https://linkedin.com/in/sujith-nagella)*
