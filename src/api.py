from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union

MODEL_PATH = "model_pipeline.joblib"

APPROVE_MAX = 0.30
REJECT_MIN = 0.60
LGD = 0.60

# ID-like keys (metadata) that should NOT be used as model features
ID_KEYS = {"id", "ID", "application_id", "loan_id", "client_id", "customer_id", "reference_id"}

app = FastAPI(title="Loan Risk Scoring API", version="1.4")

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


def safe_float(x) -> Optional[float]:
    """
    Convert to float safely.
    Returns None if x is None/NA/blank/unparseable.
    """
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, str) and x.strip() == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


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
    Missing columns are created as np.nan so sklearn can impute.
    Reorders columns to expected first.
    """
    if not EXPECTED_COLS:
        return df

    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = np.nan  # IMPORTANT: use np.nan (not pd.NA)

    df = df[EXPECTED_COLS + [c for c in df.columns if c not in EXPECTED_COLS]]

    # Ensure any pandas missing types become np.nan
    df = df.replace({pd.NA: np.nan})
    return df


def extract_id(item: Dict[str, Any]) -> Optional[str]:
    """
    Try to find an ID/application_id in the incoming item.
    """
    for k in ["application_id", "ID", "id", "loan_id", "client_id", "customer_id", "reference_id"]:
        if k in item and item[k] is not None and str(item[k]).strip() != "":
            return str(item[k])
    return None


def remove_id_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of item without ID keys so the model never sees identifiers.
    """
    return {k: v for k, v in item.items() if k not in ID_KEYS}


# -----------------------
# Request Schemas
# -----------------------

class LoanApplication(BaseModel):
    # New optional ID support (keeps backward compatibility)
    application_id: Optional[str] = None
    data: Dict[str, Any]


class BatchItemNew(BaseModel):
    application_id: Optional[str] = None
    data: Dict[str, Any]


class BatchLoanApplication(BaseModel):
    # support both styles:
    # 1) old: items = [ {feature dict}, ... ]
    # 2) new: items = [ {application_id, data}, ... ]
    items: List[Union[Dict[str, Any], BatchItemNew]]


# -----------------------
# Endpoints
# -----------------------

@app.get("/health")
def health():
    # include version to confirm you are running the latest code
    return {"status": "ok", "api_version": "1.4"}


@app.post("/predict")
def predict(payload: LoanApplication):
    try:
        app_id = payload.application_id

        # remove ID fields from features in case user accidentally includes them
        features = remove_id_fields(payload.data)

        df = pd.DataFrame([features])
        df = align_input_columns(df)

        pd_default = float(model.predict_proba(df)[:, 1][0])
        decision = decision_from_pd(pd_default)

        loan_amount = safe_float(payload.data.get("loan_amount", None))
        eloss = expected_loss(pd_default, loan_amount) if loan_amount is not None else None

        return {
            "application_id": app_id,
            "pd_default": pd_default,
            "decision": decision,
            "expected_loss": eloss,
            "assumptions": {"LGD": LGD, "APPROVE_MAX": APPROVE_MAX, "REJECT_MIN": REJECT_MIN},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
def batch_predict(payload: BatchLoanApplication):
    try:
        if not payload.items:
            return {"results": [], "count": 0}

        ids: List[Optional[str]] = []
        rows: List[Dict[str, Any]] = []
        raw_for_loss: List[Dict[str, Any]] = []

        for it in payload.items:
            # old-style: dict
            if isinstance(it, dict):
                app_id = extract_id(it)
                ids.append(app_id)

                feat = remove_id_fields(it)
                rows.append(feat)
                raw_for_loss.append(it)  # keep original for loan_amount lookup

            # new-style: {application_id, data}
            else:
                ids.append(it.application_id)

                feat = remove_id_fields(it.data)
                rows.append(feat)
                raw_for_loss.append(it.data)

        df = pd.DataFrame(rows)
        df = align_input_columns(df)

        probs = model.predict_proba(df)[:, 1]

        results = []
        for app_id, original_item, p in zip(ids, raw_for_loss, probs):
            pd_default = float(p)
            decision = decision_from_pd(pd_default)

            loan_amount = safe_float(original_item.get("loan_amount", None))
            eloss: Optional[float] = None
            if loan_amount is not None:
                eloss = expected_loss(pd_default, loan_amount)

            results.append({
                "application_id": app_id,
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