# Loan Risk Monitoring System

End-to-end credit default prediction system including:

- Logistic Regression model with preprocessing pipeline
- Threshold tuning and business decision policy
- Expected loss calculation
- Fairness diagnostics
- FastAPI scoring service (single & batch)
- Streamlit dashboard

## Architecture

Client → FastAPI (/predict or /batch_predict)
→ Model Pipeline (Impute + Encode + Scale + Logistic Regression)
→ Outputs:
    - Probability of Default (PD)
    - Decision (Approve / Review / Reject)
    - Expected Loss

## Run Locally

### Create environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

### Train model
python src/train_baseline.py

### Run API
uvicorn src.api:app --reload

Open:
http://127.0.0.1:8000/docs