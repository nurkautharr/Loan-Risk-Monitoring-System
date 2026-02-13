import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

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

    results = X_test.copy()
    results["decision"] = [decision_from_pd(p) for p in pd_default]

    # Approval rate by Gender
    if "Gender" in results.columns:
        print("\nApproval rate by Gender:")
        gender_table = (
            results.groupby("Gender")["decision"]
            .apply(lambda x: (x == "APPROVE").mean())
        )
        print(gender_table)

    # Approval rate by Region
    if "Region" in results.columns:
        print("\nApproval rate by Region:")
        region_table = (
            results.groupby("Region")["decision"]
            .apply(lambda x: (x == "APPROVE").mean())
        )
        print(region_table)

    # Approval rate by Age
    if "age" in results.columns:
        print("\nApproval rate by Age group:")
        age_table = (
            results.groupby("age")["decision"]
            .apply(lambda x: (x == "APPROVE").mean())
        )
        print(age_table)


if __name__ == "__main__":
    main()