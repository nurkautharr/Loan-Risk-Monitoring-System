import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

DATA_PATH = "data/Loan_Default.csv"
MODEL_PATH = "model_pipeline.joblib"

def main():
    # 1) Load data
    df = pd.read_csv(DATA_PATH)

    target = "Status"
    X = df.drop(columns=[target], errors="ignore")
    y = df[target]

    # Drop ID (identifier)
    X = X.drop(columns=["ID"], errors="ignore")

    # 2) Same split as training (important: same random_state + stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Load trained pipeline and predict probabilities
    clf = joblib.load(MODEL_PATH)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))

    # 4) Evaluate different thresholds
    thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Metrics
        recall_default = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision_default = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # false positive rate

        rows.append({
            "threshold": round(float(t), 2),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall_default": round(recall_default, 3),
            "precision_default": round(precision_default, 3),
            "false_positive_rate": round(fpr, 3),
            "defaults_caught_%": round(100 * recall_default, 1),
        })

    results = pd.DataFrame(rows)

    # 5) Show top candidates depending on goal
    print("\nTop thresholds if you want HIGH default recall (catch more defaulters):")
    print(results.sort_values(["recall_default", "precision_default"], ascending=False).head(10).to_string(index=False))

    print("\nTop thresholds if you want LOW false positives (reject fewer good borrowers):")
    print(results.sort_values(["false_positive_rate", "recall_default"], ascending=True).head(10).to_string(index=False))

    # 6) Save results
    results.to_csv("threshold_results.csv", index=False)
    print("\nSaved: threshold_results.csv")

if __name__ == "__main__":
    main()