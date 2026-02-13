import numpy as np
import pandas as pd
import joblib

MODEL_PATH = "model_pipeline.joblib"

def main():
    # 1) Load trained pipeline
    clf = joblib.load(MODEL_PATH)

    # 2) Get the preprocessor and model
    preprocessor = clf.named_steps["preprocessor"]
    model = clf.named_steps["model"]

    # 3) Get feature names after preprocessing
    # Numeric features
    num_features = preprocessor.transformers_[0][2]  # ("num", pipeline, numeric_cols)

    # Categorical features (after one-hot)
    cat_transformer = preprocessor.transformers_[1][1]  # pipeline for cat
    cat_features = preprocessor.transformers_[1][2]     # categorical_cols
    ohe = cat_transformer.named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(cat_features)

    feature_names = np.concatenate([num_features, cat_feature_names])

    # 4) Logistic regression coefficients
    coefs = model.coef_.flatten()

    # 5) Put into a table
    importance = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs)
    }).sort_values("abs_coef", ascending=False)

    # 6) Show top drivers
    print("\nTop 15 most influential features (by absolute coefficient):")
    print(importance.head(15).to_string(index=False))

    print("\nTop 10 features that INCREASE default risk (positive coef):")
    print(importance[importance["coef"] > 0].head(10).to_string(index=False))

    print("\nTop 10 features that DECREASE default risk (negative coef):")
    print(importance[importance["coef"] < 0].head(10).to_string(index=False))

    # 7) Save for reporting
    importance.to_csv("feature_importance_logreg.csv", index=False)
    print("\nSaved: feature_importance_logreg.csv")


if __name__ == "__main__":
    main()