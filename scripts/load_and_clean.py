"""
load_and_clean.py

Handles loading the raw dataset and performing initial cleaning steps.
"""

import pandas as pd
import numpy as np
import sqlite3

# Display formatting for readability (matches your notebook)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw JetBlue route dataset from a CSV file.

    Parameters:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial cleaning:
    - Strip column names
    - Standardize date formats
    - Convert numeric columns where appropriate

    Parameters:
        df (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Convert month column to datetime if present
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"])

    # Attempt numeric conversion for object columns
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

    return df
