from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

MODEL_PATH = "model_pipeline.joblib"

APPROVE_MAX = 0.30
REJECT_MIN = 0.60
LGD = 0.60  # assumption for expected loss

app = FastAPI(title="Loan Risk Scoring API", version="1.0")

# Load model once when the API starts
model = joblib.load(MODEL_PATH)


def decision_from_pd(pd_default: float) -> str:
    if pd_default < APPROVE_MAX:
        return "APPROVE"
    elif pd_default >= REJECT_MIN:
        return "REJECT"
    else:
        return "MANUAL_REVIEW"


def expected_loss(pd_default: float, loan_amount: float) -> float:
    return float(pd_default) * float(loan_amount) * float(LGD)


class LoanApplication(BaseModel):
    # Keep it flexible: accept any fields as a dict
    # Because your dataset has many columns and we want the API to be easy to use.
    data: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: LoanApplication):
    # Convert dict -> DataFrame (single row)
    df = pd.DataFrame([payload.data])

    # Predict PD
    pd_default = float(model.predict_proba(df)[:, 1][0])

    # Decision + expected loss (requires loan_amount)
    decision = decision_from_pd(pd_default)

    loan_amount = payload.data.get("loan_amount", None)
    if loan_amount is None:
        return {
            "pd_default": pd_default,
            "decision": decision,
            "expected_loss": None,
            "note": "loan_amount not provided so expected_loss cannot be calculated"
        }

    return {
        "pd_default": pd_default,
        "decision": decision,
        "expected_loss": expected_loss(pd_default, float(loan_amount)),
        "assumptions": {"LGD": LGD, "APPROVE_MAX": APPROVE_MAX, "REJECT_MIN": REJECT_MIN},
    }