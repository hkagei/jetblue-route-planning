"""
growth_metrics.py

Computes lag metrics, month-over-month growth, route-level summaries,
opportunity scores, and merges opportunity scores back into the monthly dataset.
"""

import pandas as pd


# -----------------------------
# 5.1 Lag Metrics
# -----------------------------
def add_lag_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged revenue, profit, and passenger values for each route.
    """
    df = df.sort_values(["route", "month"])

    df["lag_revenue"] = df.groupby("route")["revenue"].shift(1)
    df["lag_profit"] = df.groupby("route")["profit"].shift(1)
    df["lag_passengers"] = df.groupby("route")["passengers"].shift(1)

    return df


# -----------------------------
# 5.2 Month-over-Month Growth
# -----------------------------
def calculate_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate month-over-month revenue and profit growth.
    """
    df["revenue_growth"] = (
        (df["revenue"] - df["lag_revenue"]) / df["lag_revenue"]
    )

    df["profit_growth"] = (
        (df["profit"] - df["lag_profit"]) / df["lag_profit"]
    )

    return df


# -----------------------------
# 5.3 Route-Level Summary
# -----------------------------
def build_route_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a route-level summary with total revenue, total profit,
    and average growth metrics.
    """
    route_summary = (
        df.groupby("route")
          .agg(
              total_revenue=("revenue", "sum"),
              total_profit=("profit", "sum"),
              avg_revenue_growth=("revenue_growth", "mean"),
              avg_profit_growth=("profit_growth", "mean")
          )
          .reset_index()
    )

    return route_summary


# -----------------------------
# 5.4 Opportunity Score
# -----------------------------
def add_opportunity_score(route_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Add opportunity score as an equal-weight blend of
    avg revenue growth and avg profit growth.
    """
    route_summary["opportunity_score"] = (
        0.5 * route_summary["avg_profit_growth"] +
        0.5 * route_summary["avg_revenue_growth"]
    )

    return route_summary


# -----------------------------
# 5.5 Join Opportunity Score Back to Monthly Data
# -----------------------------
def merge_opportunity_score(df: pd.DataFrame, route_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Merge opportunity score back into the monthly dataset.
    """
    df = df.merge(
        route_summary[["route", "opportunity_score"]],
        on="route",
        how="left"
    )
    return df


# -----------------------------
# Full Pipeline
# -----------------------------
def apply_growth_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full growth metrics pipeline:
    - Lag metrics
    - MoM growth
    - Route summary
    - Opportunity score
    - Merge back into monthly data

    Returns:
        df (pd.DataFrame): Monthly dataset with growth + opportunity score
        route_summary (pd.DataFrame): Route-level summary table
    """
    df = add_lag_metrics(df)
    df = calculate_growth(df)

    route_summary = build_route_summary(df)
    route_summary = add_opportunity_score(route_summary)

    df = merge_opportunity_score(df, route_summary)

    return df, route_summary
