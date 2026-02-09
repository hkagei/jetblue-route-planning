"""
enrich_routes.py

Adds aircraft type assignment, seat configuration, and fleet utilization metrics
based on realistic JetBlue fleet deployment patterns.
"""

import pandas as pd


def prepare_route_identifier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a route identifier (e.g., JFK-LAX).
    """
    df["route"] = df["origin"] + "-" + df["destination"]
    return df


def assign_aircraft_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign aircraft types based on realistic JetBlue fleet usage patterns.
    """
    aircraft_map = {
        "JFK-LAX": "A321 Mint",
        "JFK-SFO": "A321 Mint",
        "JFK-SAN": "A321 Mint",
        "BOS-SEA": "A321",
        "BOS-DEN": "A320",
        "BOS-MCO": "A320",
        "BOS-CHS": "A220",
        "FLL-AUS": "A220",
        "FLL-EWR": "A220",
        "JFK-AUS": "A320"
    }

    df["aircraft_type"] = df["route"].map(aircraft_map)
    return df


def add_seat_configuration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map seat configuration to each aircraft type.
    """
    seat_map = {
        "A220": 140,
        "A320": 162,
        "A321": 200,
        "A321 Mint": 159  # 16 Mint + 143 economy
    }

    df["seats_configured"] = df["aircraft_type"].map(seat_map)
    return df


def compute_fleet_utilization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fleet utilization summary:
    - Total passengers per aircraft type
    - Average passengers per flight
    - Seat configuration
    - Load factor proxy
    """
    fleet_summary = (
        df.groupby("aircraft_type")
          .agg(
              total_passengers=("passengers", "sum"),
              avg_passengers_per_flight=("passengers", "mean"),
              seats_configured=("seats_configured", "first")
          )
          .reset_index()
    )

    fleet_summary["avg_load_factor_proxy"] = (
        fleet_summary["avg_passengers_per_flight"] /
        fleet_summary["seats_configured"]
    )

    return fleet_summary


def enrich_routes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full enrichment pipeline:
    - Sort by origin/destination/month
    - Create route identifier
    - Assign aircraft types
    - Add seat configuration
    """
    df = df.copy()

    # Ensure month is datetime and sorted
    df["month"] = pd.to_datetime(df["month"])
    df.sort_values(["origin", "destination", "month"], inplace=True)

    df = prepare_route_identifier(df)
    df = assign_aircraft_types(df)
    df = add_seat_configuration(df)

    return df
