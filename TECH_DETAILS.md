# Technical Documentation

This document provides a detailed technical overview of the model pipeline, API implementation, decision logic and deployment architecture to support maintainability, auditability and production readiness of the Loan Risk Monitoring & Scoring System.

---

## 1. Model Training

File: `src/train_baseline.py`

- Dataset: Loan Default Dataset (Kaggle) https://www.kaggle.com/datasets/yasserh/loan-default-dataset/data
- Target: `Status`
- Preprocessing:
  - Missing value imputation
  - One-hot encoding for categorical variables
  - Standard scaling for numeric features
- Model: Logistic Regression
- Train-test split: 80/20
- Evaluation metric: ROC-AUC

Performance:
- ROC-AUC: 0.8675
- Balanced precision/recall
- Threshold analysis performed

---

## 2. Threshold Tuning

Risk policy thresholds:

- APPROVE_MAX = 0.30
- REJECT_MIN = 0.60
- LGD = 0.60

These thresholds simulate configurable institutional risk appetite.

---

## 3. API Layer (FastAPI)

File: `src/api.py`

Features:
- Single prediction endpoint
- Batch portfolio scoring endpoint
- Input alignment to handle missing fields
- Expected loss computation
- Robust error handling
- Health monitoring endpoint

---

## 4. Batch Scoring

The `/batch_predict` endpoint allows:

- Portfolio-level evaluation
- Stress testing
- Risk aggregation
- Operational scaling

Returns:
- PD
- Decision
- Expected Loss
- Count of records

---

## 5. Deployment

Cloud: Render  
Containerized using Docker  

Command:

uvicorn src.api:app --host 0.0.0.0 --port $PORT

Dependencies frozen using:

pip freeze > requirements.txt

This ensures model compatibility and reproducibility.

---

## 6. Governance & Fairness

Approval rate analysis performed across:

- Gender
- Region
- Age group

Bias diagnostics highlight potential model sensitivity to demographic attributes.

---

## 7. Production Considerations

Future improvements:

- SHAP explainability endpoint
- Model drift detection
- Automated retraining pipeline
- Authentication & rate limiting
- Logging & audit trail
- CI/CD integration

---

## 8. Project Structure

Loan Risk Monitoring System/
├── src/
│   ├── api.py                  # FastAPI inference API
│   ├── train_baseline.py       # Model training & serialization
│   ├── threshold_tuning.py     # Threshold analysis & optimization
│   ├── risk_policy.py          # Business rule logic
│   ├── policy_impact.py        # Policy simulation & impact analysis
│   ├── fairness_check.py       # Bias & fairness diagnostics
│   ├── explainability.py       # Feature importance & interpretability
│   └── check_data.py           # Data validation checks
│
├── model_pipeline.joblib       # Serialized sklearn pipeline
├── feature_importance_logreg.csv
├── threshold_results.csv
├── scored_with_policy.csv
│
├── Dockerfile                  # Container configuration
├── requirements.txt            # Frozen Python dependencies
├── README.md                   # Business + usage documentation
├── TECH_DETAILS.md             # Technical architecture documentation
└── .gitignore