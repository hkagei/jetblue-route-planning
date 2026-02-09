"""
financial_metrics.py

Calculates revenue, cost, profit, and profit margin for JetBlue route performance.
"""

import pandas as pd


def calculate_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate revenue using passengers * average fare.
    """
    df["revenue"] = df["passengers"] * df["avg_fare"]
    return df


def calculate_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cost model:
    - COST_PER_PAX: variable cost per passenger
    - COST_PER_FLIGHT: fixed cost per flight
    - FLIGHTS_PER_MONTH: assumed number of flights per route-month
    """
    COST_PER_PAX = 65
    COST_PER_FLIGHT = 18000
    FLIGHTS_PER_MONTH = 30

    df["cost"] = (
        df["passengers"] * COST_PER_PAX
        + COST_PER_FLIGHT * FLIGHTS_PER_MONTH
    )
    return df


def calculate_profit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate profit and profit margin.
    """
    df["profit"] = df["revenue"] - df["cost"]
    df["profit_margin"] = df["profit"] / df["revenue"]
    return df


def apply_financial_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full financial pipeline:
    - Revenue
    - Cost
    - Profit
    - Profit margin
    """
    df = calculate_revenue(df)
    df = calculate_cost(df)
    df = calculate_profit(df)
    return df
