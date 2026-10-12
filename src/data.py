"""Data loading and cleaning for the IBM Telco Customer Churn dataset."""
import pandas as pd

URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
       "master/data/Telco-Customer-Churn.csv")


def load(path="data/telco_churn.csv") -> pd.DataFrame:
    """Load the raw CSV (falls back to the public URL) and fix dtypes."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.read_csv(URL)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)  # 11 brand-new customers (tenure 0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df.drop(columns=["customerID"])


NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]


def categorical(df):
    return [c for c in df.columns if c not in NUMERIC + ["Churn"]]
