"""
load_and_clean.py

Handles loading the raw dataset and performing initial cleaning steps.
"""

import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw JetBlue route dataset.
    """
    df = pd.read_csv("route_monthly_performance.csv")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial cleaning:
    - Strip column names
    - Standardize date formats
    - Ensure numeric columns are numeric
    """
    df.columns = df.columns.str.strip()

    # Convert month column to datetime if applicable
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"])

    # Convert numeric columns
    numeric_cols = df.select_dtypes(include=["object"]).columns
    for col in numeric_cols:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    return df
