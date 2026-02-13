import pandas as pd

DATA_PATH = "data/Loan_Default.csv"

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())
print("\nColumns:", list(df.columns))
print("\nTarget distribution (Status):")
print(df["Status"].value_counts(dropna=False))