import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

DATA_PATH = "data/Loan_Default.csv"


def main():
    # 1) Load data
    df = pd.read_csv(DATA_PATH)

    # Basic checks
    print("Shape:", df.shape)
    print("Duplicates:", df.duplicated().sum())
    print("\nMissing values (top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    print("\nTarget distribution (Status):")
    print(df["Status"].value_counts(dropna=False))

    # 2) Define target + features
    target = "Status"
    X = df.drop(columns=[target], errors="ignore")
    y = df[target]

    # Drop ID (identifier only)
    X = X.drop(columns=["ID"], errors="ignore")

    # 3) Split data (stratified to keep the same 0/1 ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain rows:", X_train.shape[0], "| Test rows:", X_test.shape[0])

    # 4) Identify numeric vs categorical columns (based on TRAIN only)
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    print("\nNumeric cols:", numeric_cols)
    print("Categorical cols:", categorical_cols)

    # 5) Preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # 6) Baseline model
    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # 7) Train
    clf.fit(X_train, y_train)

    # 8) Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    auc = roc_auc_score(y_test, y_proba)
    print("\nROC-AUC:", round(auc, 4))

    # 9) Save model pipeline
    joblib.dump(clf, "model_pipeline.joblib")
    print("\nSaved: model_pipeline.joblib")

if __name__ == "__main__":
    main()