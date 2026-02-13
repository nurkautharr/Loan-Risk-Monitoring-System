from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List, Dict, Any, Optional

MODEL_PATH = "model_pipeline.joblib"

APPROVE_MAX = 0.30
REJECT_MIN = 0.60
LGD = 0.60

app = FastAPI(title="Loan Risk Scoring API", version="1.3")

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


def get_expected_raw_columns() -> List[str]:
    """
    Return raw input columns expected by the preprocessing step.
    Uses feature_names_in_ if available; otherwise falls back to ColumnTransformer config.
    """
    pre = model.named_steps.get("preprocessor", None)
    if pre is None:
        return []

    expected = getattr(pre, "feature_names_in_", None)
    if expected is not None:
        return list(expected)

    cols: List[str] = []
    for _, _, col_list in pre.transformers_:
        if col_list is None or col_list == "drop":
            continue
        if isinstance(col_list, (list, tuple)):
            cols.extend(list(col_list))

    # remove duplicates preserve order
    seen = set()
    ordered = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


EXPECTED_COLS = get_expected_raw_columns()


def align_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has all columns the pipeline expects.
    Missing columns are created as NA so the pipeline can impute.
    Reorders columns to expected first.
    """
    if not EXPECTED_COLS:
        return df

    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[EXPECTED_COLS + [c for c in df.columns if c not in EXPECTED_COLS]]
    return df


def safe_float(x) -> Optional[float]:
    """
    Convert to float safely.
    Returns None if x is None/NA/blank/unparseable.
    """
    if x is None:
        return None
    if pd.isna(x):
        return None
    if isinstance(x, str) and x.strip() == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


class LoanApplication(BaseModel):
    data: Dict[str, Any]


class BatchLoanApplication(BaseModel):
    items: List[Dict[str, Any]]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: LoanApplication):
    try:
        df = pd.DataFrame([payload.data])
        df = align_input_columns(df)

        pd_default = float(model.predict_proba(df)[:, 1][0])
        decision = decision_from_pd(pd_default)

        loan_amount = safe_float(payload.data.get("loan_amount", None))
        if loan_amount is None:
            return {
                "pd_default": pd_default,
                "decision": decision,
                "expected_loss": None,
                "note": "loan_amount missing/invalid so expected_loss cannot be calculated",
                "assumptions": {"LGD": LGD, "APPROVE_MAX": APPROVE_MAX, "REJECT_MIN": REJECT_MIN},
            }

        return {
            "pd_default": pd_default,
            "decision": decision,
            "expected_loss": expected_loss(pd_default, loan_amount),
            "assumptions": {"LGD": LGD, "APPROVE_MAX": APPROVE_MAX, "REJECT_MIN": REJECT_MIN},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
def batch_predict(payload: BatchLoanApplication):
    try:
        if not payload.items:
            return {"results": [], "count": 0}

        df = pd.DataFrame(payload.items)
        df = align_input_columns(df)

        probs = model.predict_proba(df)[:, 1]

        results = []
        for item, p in zip(payload.items, probs):
            pd_default = float(p)
            decision = decision_from_pd(pd_default)

            loan_amount = safe_float(item.get("loan_amount", None))
            eloss: Optional[float] = None
            if loan_amount is not None:
                eloss = expected_loss(pd_default, loan_amount)

            results.append({
                "pd_default": pd_default,
                "decision": decision,
                "expected_loss": eloss
            })

        return {
            "results": results,
            "count": len(results),
            "assumptions": {"LGD": LGD, "APPROVE_MAX": APPROVE_MAX, "REJECT_MIN": REJECT_MIN},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))