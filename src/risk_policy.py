import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

DATA_PATH = "data/Loan_Default.csv"
MODEL_PATH = "model_pipeline.joblib"

# Decision policy thresholds (you can change later)
APPROVE_MAX = 0.30
REJECT_MIN = 0.60

# Business assumption
LGD = 0.60  # 60% loss given default


def decision_from_pd(pd_default: float) -> str:
    """Convert probability of default into a decision label."""
    if pd_default < APPROVE_MAX:
        return "APPROVE"
    elif pd_default >= REJECT_MIN:
        return "REJECT"
    else:
        return "MANUAL_REVIEW"


def expected_loss(pd_default: float, loan_amount: float, lgd: float = LGD) -> float:
    """Expected Loss = PD x Exposure(loan_amount) x LGD."""
    if pd.isna(loan_amount):
        return float("nan")
    return float(pd_default) * float(loan_amount) * float(lgd)


def main():
    # 1) Load data
    df = pd.read_csv(DATA_PATH)

    # 2) Same split as training (so we evaluate on the same test set)
    target = "Status"
    X = df.drop(columns=[target], errors="ignore").drop(columns=["ID"], errors="ignore")
    y = df[target]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Load trained model pipeline
    clf = joblib.load(MODEL_PATH)

    # 4) Predict default probability (PD)
    pd_default = clf.predict_proba(X_test)[:, 1]

    # 5) Build an output table
    out = X_test.copy()
    out["pd_default"] = pd_default
    out["decision"] = out["pd_default"].apply(decision_from_pd)

    # 6) Expected Loss using loan_amount
    if "loan_amount" not in out.columns:
        raise ValueError("Column 'loan_amount' not found. Expected loss needs loan_amount.")

    out["expected_loss"] = [
        expected_loss(p, amt, LGD) for p, amt in zip(out["pd_default"], out["loan_amount"])
    ]

    # 7) Quick summary (this is what you show to “client”)
    print("\nDecision distribution:")
    print(out["decision"].value_counts())

    print("\nAverage PD by decision:")
    print(out.groupby("decision")["pd_default"].mean().sort_values())

    print("\nTotal Expected Loss by decision (on test set):")
    print(out.groupby("decision")["expected_loss"].sum().sort_values())

    # 8) Save a sample for reporting / dashboard
    out_sample = out.sample(20, random_state=42)[
        ["pd_default", "decision", "expected_loss", "loan_amount", "income", "Credit_Score", "LTV", "dtir1"]
    ]
    print("\nSample decisions (20 rows):")
    print(out_sample.to_string(index=False))

    out.to_csv("scored_with_policy.csv", index=False)
    print("\nSaved: scored_with_policy.csv")


if __name__ == "__main__":
    main()