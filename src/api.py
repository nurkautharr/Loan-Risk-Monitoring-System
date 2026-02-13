from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List, Dict, Any, Optional

MODEL_PATH = "model_pipeline.joblib"

APPROVE_MAX = 0.30
REJECT_MIN = 0.60
LGD = 0.60

app = FastAPI(title="Loan Risk Scoring API", version="1.1")

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
    data: Dict[str, Any]


class BatchLoanApplication(BaseModel):
    items: List[Dict[str, Any]]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: LoanApplication):
    df = pd.DataFrame([payload.data])
    pd_default = float(model.predict_proba(df)[:, 1][0])
    decision = decision_from_pd(pd_default)

    loan_amount = payload.data.get("loan_amount", None)
    if loan_amount is None:
        return {
            "pd_default": pd_default,
            "decision": decision,
            "expected_loss": None,
            "note": "loan_amount not provided so expected_loss cannot be calculated",
            "assumptions": {
                "LGD": LGD,
                "APPROVE_MAX": APPROVE_MAX,
                "REJECT_MIN": REJECT_MIN
            }
        }

    return {
        "pd_default": pd_default,
        "decision": decision,
        "expected_loss": expected_loss(pd_default, float(loan_amount)),
        "assumptions": {
            "LGD": LGD,
            "APPROVE_MAX": APPROVE_MAX,
            "REJECT_MIN": REJECT_MIN
        }
    }


@app.post("/batch_predict")
def batch_predict(payload: BatchLoanApplication):
    if not payload.items:
        return {"results": [], "count": 0}

    df = pd.DataFrame(payload.items)
    probs = model.predict_proba(df)[:, 1]

    results = []
    for item, p in zip(payload.items, probs):
        pd_default = float(p)
        decision = decision_from_pd(pd_default)

        loan_amount = item.get("loan_amount", None)
        eloss: Optional[float] = None
        if loan_amount is not None:
            try:
                eloss = expected_loss(pd_default, float(loan_amount))
            except Exception:
                eloss = None

        results.append({
            "pd_default": pd_default,
            "decision": decision,
            "expected_loss": eloss
        })

    return {
        "results": results,
        "count": len(results),
        "assumptions": {
            "LGD": LGD,
            "APPROVE_MAX": APPROVE_MAX,
            "REJECT_MIN": REJECT_MIN
        }
    }