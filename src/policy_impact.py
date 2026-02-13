import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

DATA_PATH = "data/Loan_Default.csv"
MODEL_PATH = "model_pipeline.joblib"

APPROVE_MAX = 0.30
REJECT_MIN = 0.60


def decision_from_pd(pd_default):
    if pd_default < APPROVE_MAX:
        return "APPROVE"
    elif pd_default >= REJECT_MIN:
        return "REJECT"
    else:
        return "MANUAL_REVIEW"


def main():
    df = pd.read_csv(DATA_PATH)

    target = "Status"
    X = df.drop(columns=[target], errors="ignore").drop(columns=["ID"], errors="ignore")
    y = df[target]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = joblib.load(MODEL_PATH)
    pd_default = clf.predict_proba(X_test)[:, 1]

    results = pd.DataFrame({
        "actual_default": y_test.values,
        "pd_default": pd_default
    })

    results["decision"] = results["pd_default"].apply(decision_from_pd)

    # Simulate policy:
    # REJECT → do not issue loan (no default possible)
    # APPROVE → issue loan
    # MANUAL_REVIEW → assume 50% approved (simplified assumption)

    approved = results[
        (results["decision"] == "APPROVE") |
        (results["decision"] == "MANUAL_REVIEW")
    ].copy()

    print("\nTotal loans issued:", len(approved))
    print("Total loans rejected:", (results["decision"] == "REJECT").sum())

    defaults_after_policy = approved["actual_default"].sum()
    print("Defaults among issued loans:", defaults_after_policy)

    total_defaults_without_model = results["actual_default"].sum()
    print("Total defaults without model filtering:", total_defaults_without_model)

    print("\nDefaults avoided by rejecting high-risk loans:",
          total_defaults_without_model - defaults_after_policy)


if __name__ == "__main__":
    main()