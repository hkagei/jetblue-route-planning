"""
export_master.py

Runs the full JetBlue route analytics pipeline:
- Load & clean raw data
- Enrich routes (aircraft type, seats, fleet utilization)
- Apply financial metrics (revenue, cost, profit, margin)
- Apply growth metrics (lag features, MoM growth, opportunity score)
- Export final master dataset for Tableau
"""

import pandas as pd

from load_and_clean import load_raw_data, clean_data
from enrich_routes import enrich_routes
from financial_metrics import apply_financial_metrics
from growth_metrics import apply_growth_metrics


def build_master_dataset(input_path: str, output_path: str):
    """
    Run the full pipeline and export the final enriched dataset.

    Parameters:
        input_path (str): Path to the raw CSV file.
        output_path (str): Path where the final master CSV will be saved.
    """

    # -----------------------------
    # 1. Load & Clean
    # -----------------------------
    df = load_raw_data(input_path)
    df = clean_data(df)

    # -----------------------------
    # 2. Route Enrichment
    # -----------------------------
    df = enrich_routes(df)

    # -----------------------------
    # 3. Financial Metrics
    # -----------------------------
    df = apply_financial_metrics(df)

    # -----------------------------
    # 4. Growth Metrics
    # -----------------------------
    df, route_summary = apply_growth_metrics(df)

    # -----------------------------
    # 5. Export Final Dataset
    # -----------------------------
    df.to_csv(output_path, index=False)
    print(f"Master dataset exported to: {output_path}")

    return df, route_summary


if __name__ == "__main__":
    build_master_dataset(
        input_path="route_monthly_performance.csv",
        output_path="jetblue_enriched_dataset.csv"
    )
