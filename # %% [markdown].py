# %% [markdown]
# # JetBlue Route Planning – Python Analysis
# 
# # Executive Summary
# 
# This project models JetBlue route performance using a realistic, end-to-end analytics
# workflow that mirrors the responsibilities of a Route Planning Analyst. The analysis
# combines Python, SQL, and Tableau to evaluate financial performance, operational
# capacity, and month-over-month growth across a set of representative JetBlue routes.
# 
# Key enhancements include:
# - Realistic aircraft assignments (A220, A320, A321, A321 Mint) based on JetBlue’s
#   actual fleet deployment patterns.
# - Seat configuration mapping to support capacity and load factor-style analysis.
# - Financial metrics such as revenue, cost, profit, and profit margin.
# - Time-series metrics including lagged values and month-over-month growth.
# - SQL queries that replicate core analytical tasks used in airline planning.
# - Tableau dashboards that visualize route profitability, growth trends, and
#   aircraft-level performance.
# 
# This notebook demonstrates the ability to structure data, build analytical models,
# communicate insights, and create executive-ready visualizations—key skills for
# aviation analytics roles.

# %%
import pandas as pd
import numpy as np

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# %%
import sys
sys.executable

# %% [markdown]
# ## 1. Load raw dataset
# 
# We load `route_monthly_performance.csv` and do basic sanity checks.

# %%
# Load raw data
df = pd.read_csv("route_monthly_performance.csv")

# Quick inspection
df.head()

# %%
df.info()

# %%
df.describe()

# %%
df["month"] = pd.to_datetime(df["month"])
df.sort_values(["origin", "destination", "month"], inplace=True)
df.head()

# %% [markdown]
# ## 2. Create route identifier
# 
# We create a `route` column (e.g., `JFK-LAX`) to simplify grouping.

# %%
df["route"] = df["origin"] + "-" + df["destination"]
df[["origin", "destination", "route"]].head()

# %% [markdown]
# ## 3. Calculate revenue, cost, profit, and profit margin
# 
# Assumptions:
# - Revenue = passengers × avg_fare
# - CASM (cost per seat-mile) = 0.11 USD
# - Cost = seats × distance_miles × CASM
# - Profit = revenue − cost
# - Profit margin = profit / revenue

# %%
CASM = 0.11  # USD per seat-mile (assumption)

df["revenue"] = df["passengers"] * df["avg_fare"]
df["cost"] = df["seats"] * df["distance_miles"] * CASM
df["profit"] = df["revenue"] - df["cost"]
df["profit_margin"] = df["profit"] / df["revenue"]

df[["route", "month", "revenue", "cost", "profit", "profit_margin"]].head()

# %% [markdown]
# ## 4. Compute growth metrics
# 
# We calculate month-over-month growth in:
# - Passengers
# - Revenue
# 
# These are computed per route.

# %%
# Ensure sorted by route and month
df = df.sort_values(["route", "month"])

# Lagged values per route
df["passengers_prev"] = df.groupby("route")["passengers"].shift(1)
df["revenue_prev"] = df.groupby("route")["revenue"].shift(1)

# Growth rates
df["passenger_growth"] = (df["passengers"] - df["passengers_prev"]) / df["passengers_prev"]
df["revenue_growth"] = (df["revenue"] - df["revenue_prev"]) / df["revenue_prev"]

df[["route", "month", "passengers", "passengers_prev", "passenger_growth"]].head(10)

# %% [markdown]
# ## 5. Build route-level summary
# 
# We aggregate to one row per route with:
# - Average profit margin
# - Average passenger growth
# - Average competitor seats
# - Average load factor
# - Total revenue
# - Total profit

# %%
route_summary = df.groupby("route").agg(
    avg_profit_margin=("profit_margin", "mean"),
    avg_passenger_growth=("passenger_growth", "mean"),
    avg_competitor_seats=("competitor_seats", "mean"),
    avg_load_factor=("load_factor", "mean"),
    total_revenue=("revenue", "sum"),
    total_profit=("profit", "sum")
).reset_index()

route_summary

# %% [markdown]
# ## 6. Compute opportunity score
# 
# We normalize:
# - Profit margin (higher is better)
# - Passenger growth (higher is better)
# - Competitor seats (lower is better)
# 
# Then combine into a single opportunity score:
# - 0.4 × normalized profit margin
# - 0.4 × normalized passenger growth
# - 0.2 × (1 − normalized competitor seats)

# %%
def min_max_norm(series):
    return (series - series.min()) / (series.max() - series.min())

route_summary["norm_profit_margin"] = min_max_norm(route_summary["avg_profit_margin"])
route_summary["norm_passenger_growth"] = min_max_norm(route_summary["avg_passenger_growth"].fillna(0))
route_summary["norm_competitor_seats"] = min_max_norm(route_summary["avg_competitor_seats"])

route_summary["opportunity_score"] = (
    0.4 * route_summary["norm_profit_margin"] +
    0.4 * route_summary["norm_passenger_growth"] +
    0.2 * (1 - route_summary["norm_competitor_seats"])
)

route_summary.sort_values("opportunity_score", ascending=False)

# %% [markdown]
# ## 7. Join opportunity score back to monthly data
# 
# We attach the route-level opportunity score to each monthly row so Tableau can use it in all views.

# %%
df = df.merge(
    route_summary[["route", "opportunity_score"]],
    on="route",
    how="left"
)

df[["route", "month", "profit_margin", "passenger_growth", "opportunity_score"]].head()

# %% [markdown]
# ## 8. Export enriched datasets for Tableau
# 
# We export:
# - `route_monthly_enriched_for_tableau.csv` (monthly-level data)
# - `route_summary_for_tableau.csv` (route-level summary)

# %%
df.to_csv("route_monthly_enriched_for_tableau.csv", index=False)
route_summary.to_csv("route_summary_for_tableau.csv", index=False)

"Export complete."

# %% [markdown]
# ## 9. Sanity checks
# 
# We quickly inspect:
# - Top routes by total profit
# - Top routes by opportunity score
# - Distribution of profit margin

# %%
print("Top 5 routes by total profit:")
display(route_summary.sort_values("total_profit", ascending=False).head())

print("\nTop 5 routes by opportunity score:")
display(route_summary.sort_values("opportunity_score", ascending=False).head())

print("\nProfit margin distribution:")
df["profit_margin"].describe()

# %% [markdown]
# ### Aircraft Type and Seat Configuration
# 
# To add operational realism to the dataset, each route is assigned a JetBlue aircraft
# type based on typical fleet usage patterns (A220, A320, A321, A321 with Mint). Seat
# configuration is then mapped to each aircraft type to support capacity and load factor
# analysis.
# 
# This enhancement aligns the dataset with real-world airline planning practices and
# enables deeper insights in Tableau.

# %% [markdown]
# ### Aircraft Type Assignment and Fleet Realism
# 
# To add operational realism, each route is assigned a JetBlue aircraft type based on
# typical fleet usage patterns:
# 
# - **A321 Mint** on premium transcontinental routes (e.g., JFK–LAX, JFK–SFO, JFK–SAN)
# - **A321 / A320** on high-demand medium/long-haul domestic routes
# - **A220** on shorter and medium-haul routes where right-sizing capacity is important
# 
# These assignments reflect how JetBlue deploys its fleet in practice: A321 aircraft on
# premium, high-yield routes; A320s on core domestic markets; and A220s on thinner or
# developing routes. Seat configurations are mapped to each aircraft type to support
# capacity and load factor-style analysis.

# %%
# Aircraft assignment based on realistic JetBlue fleet usage
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
df.head()

# %%
seat_map = {
    "A220": 140,
    "A320": 162,
    "A321": 200,
    "A321 Mint": 159  # 16 Mint + 143 economy
}

df["seats_configured"] = df["aircraft_type"].map(seat_map)

# %%
df[["route", "aircraft_type", "seats_configured"]].drop_duplicates()

# %% [markdown]
# ### Fleet Utilization Overview
# 
# This section summarizes how each aircraft type is used across the network in this
# dataset. It highlights total flights, total passengers, and average load relative to
# configured seats, providing a simplified view of fleet utilization.

# %% [markdown]
# ## Fleet Utilization Summary
# 
# To complement the aircraft assignment and seat configuration enhancements, this section
# provides a high-level view of how each aircraft type is utilized across the network.
# Fleet utilization is a core component of airline planning, as it connects demand,
# capacity, and profitability.
# 
# This summary highlights:
# - Total passengers carried by each aircraft type  
# - Average passengers per flight (a proxy for demand)  
# - Configured seat capacity  
# - A simplified load factor proxy (average passengers ÷ seats configured)
# 
# While not a full operational utilization model, this provides a realistic snapshot of
# how JetBlue’s fleet performs across the selected routes.

# %%
fleet_summary = (
    df.groupby("aircraft_type")
      .agg(
          total_passengers=("passengers", "sum"),
          avg_passengers_per_flight=("passengers", "mean"),
          seats_configured=("seats_configured", "first")
      )
      .reset_index()
)

fleet_summary

# %%
fleet_summary["avg_load_factor_proxy"] = (
    fleet_summary["avg_passengers_per_flight"] / fleet_summary["seats_configured"]
)
fleet_summary

# %% [markdown]
# # Section 3 — Financial Metrics (Revenue, Cost, Profit, Margin)
# 
# In this section, I enrich the dataset with core financial metrics used in airline route
# profitability analysis. These metrics help quantify the financial performance of each
# route-month combination.
# 
# **Metrics added:**
# - **Revenue** = passengers × average fare  
# - **Cost** = passenger-related costs + fixed flight operating costs  
# - **Profit** = revenue − cost  
# - **Profit Margin** = profit ÷ revenue  
# 
# These metrics form the foundation for evaluating route performance and identifying
# high- and low-performing markets.

# %% [markdown]
# ### Revenue Calculation
# 
# Revenue represents the total amount generated from ticket sales for each route-month.
# It is calculated as:
# 
# **Revenue = Passengers × Average Fare**
# 
# This metric provides the top-line financial performance of each route and is essential
# for understanding demand and pricing effectiveness.

# %%
df["revenue"] = df["passengers"] * df["avg_fare"]
df[["route", "month", "passengers", "avg_fare", "revenue"]].head()

# %% [markdown]
# ### Cost Calculation
# 
# Cost represents the operational expenses associated with flying each route for a given
# month. For this project, I use a simplified cost model that includes:
# 
# - **Variable cost per passenger** (e.g., fuel, catering, handling)
# - **Fixed cost per flight** (e.g., crew, aircraft lease, maintenance)
# - **Assumed flights per month** (30 flights per route)
# 
# **Cost = (Passengers × Cost per Passenger) + (Flights per Month × Fixed Cost per Flight)**
# 
# This provides a realistic approximation of airline operating expenses.

# %%
COST_PER_PAX = 65
COST_PER_FLIGHT = 18000
FLIGHTS_PER_MONTH = 30

df["cost"] = (df["passengers"] * COST_PER_PAX) + (COST_PER_FLIGHT * FLIGHTS_PER_MONTH)
df[["route", "month", "cost"]].head()

# %% [markdown]
# ### Profit Calculation
# 
# Profit measures the financial outcome of each route-month after accounting for both
# revenue and operating costs.
# 
# **Profit = Revenue − Cost**
# 
# Positive profit indicates a financially strong month, while negative profit highlights
# periods where the route underperformed or experienced seasonal weakness.

# %%
df["profit"] = df["revenue"] - df["cost"]
df[["route", "month", "revenue", "cost", "profit"]].head()

# %% [markdown]
# ### Profit Margin Calculation
# 
# Profit margin expresses profitability as a percentage of revenue, allowing for easier
# comparison across routes and months with different scales.
# 
# **Profit Margin = Profit ÷ Revenue**
# 
# This metric helps identify which routes are most efficient at converting revenue into
# profit, regardless of absolute size.

# %%
df["profit_margin"] = df["profit"] / df["revenue"]
df[["route", "month", "profit_margin"]].head()

# %%
df[["revenue", "cost", "profit", "profit_margin"]].describe()

# %% [markdown]
# # Section 4 — Month-over-Month Growth Analysis
# 
# This section calculates month-over-month (MoM) growth metrics to understand how each
# route is trending over time. Growth metrics help identify momentum, seasonality, and
# routes that are accelerating or declining.
# 
# **Metrics added:**
# - **passengers_prev** — previous month's passenger count  
# - **revenue_prev** — previous month's revenue  
# - **passenger_growth** — MoM passenger growth rate  
# - **revenue_growth** — MoM revenue growth rate  
# 
# These metrics are essential for trend analysis and will be used later in Tableau to
# visualize route performance over time.

# %% [markdown]
# ### Sorting the Dataset for Time-Series Calculations
# 
# Before calculating month-over-month growth, the dataset must be sorted by **route** and
# **month**. This ensures that each row is aligned with the correct previous month when
# computing lagged values and growth metrics.
# 
# Sorting is essential for accurate time-series analysis and prevents incorrect or
# misaligned comparisons.

# %%
df = df.sort_values(["route", "month"]).reset_index(drop=True)
df.head()

# %% [markdown]
# ### Previous Month Values (Lagged Features)
# 
# To calculate month-over-month growth, we first need the previous month's values for
# each route. Using the `groupby().shift(1)` method, we create:
# 
# - **passengers_prev** — last month's passenger count  
# - **revenue_prev** — last month's revenue  
# 
# The first month of each route will naturally have `NaN` values because there is no
# prior month to reference. This is expected and correct.

# %%
df["passengers_prev"] = df.groupby("route")["passengers"].shift(1)
df["revenue_prev"] = df.groupby("route")["revenue"].shift(1)

# %% [markdown]
# ### Month-over-Month Passenger Growth
# 
# Passenger growth measures how demand changes from one month to the next for each route.
# 
# **Passenger Growth = (Passengers − Previous Month Passengers) ÷ Previous Month Passengers**
# 
# This metric highlights seasonal patterns, demand spikes, and declining routes. The
# first month of each route will have `NaN` because no previous month exists.

# %%
df["passenger_growth"] = (
    (df["passengers"] - df["passengers_prev"]) / df["passengers_prev"]
)

# %% [markdown]
# ### Month-over-Month Revenue Growth
# 
# Revenue growth measures how total revenue changes from one month to the next.
# 
# **Revenue Growth = (Revenue − Previous Month Revenue) ÷ Previous Month Revenue**
# 
# This metric captures pricing changes, demand shifts, and revenue momentum. As with
# passenger growth, the first month of each route will show `NaN` values.

# %%
df["revenue_growth"] = (
    (df["revenue"] - df["revenue_prev"]) / df["revenue_prev"]
)

# %% [markdown]
# ### Sanity Check: Inspect Growth Calculations
# 
# This quick check confirms that growth metrics were calculated correctly. The first
# month of each route should show `NaN` for previous-month and growth values, while all
# subsequent months should contain valid numbers.

# %%
df[["route", "month", "passengers", "passengers_prev", "passenger_growth"]].head(15)

# %%
df[[
    "route", "month",
    "passengers", "passengers_prev", "passenger_growth",
    "revenue", "revenue_prev", "revenue_growth"
]].head(20)

# %% [markdown]
# # Section 5 — Exporting the Enriched Dataset for Tableau
# 
# With all financial and growth metrics calculated, the next step is to export the
# dataset for visualization in Tableau. Tableau works best with a clean, flat file
# containing all relevant fields.
# 
# In this section, I export the fully enriched dataset to a CSV file that will be used
# to build the Route Profitability and Growth dashboards.

# %%
df.to_csv("jetblue_route_performance_enriched.csv", index=False)
df.head()

# %% [markdown]
# # Section 6 — Tableau Dashboards
# 
# With the enriched dataset exported, the next step is to build interactive dashboards in
# Tableau to visualize route performance. The goal is to create two clean, insight-driven
# dashboards:
# 
# 1. **Route Profitability Dashboard**  
#    - Highlights revenue, cost, profit, and profit margin by route and month  
#    - Helps identify high-performing and underperforming markets  
# 
# 2. **Month-over-Month Growth Dashboard**  
#    - Visualizes passenger and revenue growth trends  
#    - Highlights seasonal patterns and momentum shifts  
# 
# These dashboards provide a clear, executive-friendly view of route performance and
# support data-driven decision-making for network planning.

# %% [markdown]
# ### Key Insights
# 
# - Several routes show strong seasonality, with summer months outperforming winter.
# - Profitability varies significantly by route, with some markets consistently
#   generating losses despite stable demand.
# - Revenue growth and passenger growth do not always move together, indicating
#   pricing effects and yield management opportunities.
# - Certain routes demonstrate strong momentum, suggesting potential for increased
#   capacity or frequency adjustments.

# %% [markdown]
# ## Aircraft-Level Profitability Dashboards
# 
# To visualize aircraft-level performance, the enriched dataset is imported into Tableau
# and used to build a dedicated Fleet Profitability dashboard. This dashboard highlights
# how different aircraft types contribute to JetBlue’s financial and operational results.
# 
# ### Recommended Charts
# 
# #### 1. Total Profit by Aircraft Type (Bar Chart)
# - **Columns:** Aircraft Type  
# - **Rows:** SUM(Profit)  
# - **Color:** Aircraft Type  
# - **Insight:** Identifies which fleet types generate the strongest financial returns.
# 
# #### 2. Profit per Seat Proxy (Bar Chart)
# - Create a calculated field:  
#   `Profit per Seat Proxy = SUM([Profit]) / SUM([Seats Configured])`
# - **Columns:** Aircraft Type  
# - **Rows:** Profit per Seat Proxy  
# - **Insight:** Normalizes profitability by capacity to compare efficiency across aircraft.
# 
# #### 3. Monthly Profit Trend by Aircraft Type (Line Chart)
# - **Columns:** Month  
# - **Rows:** SUM(Profit)  
# - **Color:** Aircraft Type  
# - **Insight:** Reveals seasonality and performance patterns across the fleet.
# 
# #### 4. Route Profitability by Aircraft Type (Bar Chart)
# - **Columns:** Route  
# - **Rows:** SUM(Profit)  
# - **Color:** Aircraft Type  
# - **Insight:** Shows how aircraft assignment influences route-level performance.
# 
# ### Dashboard Layout
# - Use JetBlue-inspired colors (blue, light blue, orange accents).
# - Keep 3–4 charts per dashboard for clarity.
# - Add short captions summarizing key insights.
# - Arrange charts to tell a story: fleet → route → trend → efficiency.
# 
# This dashboard provides a clear, executive-friendly view of how JetBlue’s fleet
# contributes to overall route profitability.

# %% [markdown]
# # Section 7 — SQL Analysis
# 
# To demonstrate SQL proficiency, this section recreates key analytical steps from the
# Python workflow using SQL queries. This mirrors real-world analytics environments where
# data analysts frequently switch between SQL, Python, and BI tools.
# 
# The following examples use the enriched dataset to showcase:
# - Data filtering and route-level exploration
# - Aggregations and summary statistics
# - Window functions for time-series analysis
# - Profitability and growth calculations using SQL logic
# 
# These queries reflect the type of work performed in airline network planning, revenue
# management, and transit analytics.

# %% [markdown]
# ### Creating the SQL Table
# 
# The enriched dataset is loaded into a SQL table named `route_performance`. This allows
# SQL-based exploration of route profitability and growth metrics.

# %%
import sqlite3

conn = sqlite3.connect("jetblue.db")
df.to_sql("route_performance", conn, if_exists="replace", index=False)

# %% [markdown]
# ### Query 1 — Total Profit by Route
# 
# This query calculates total profit for each route, helping identify the strongest and
# weakest performers.

# %%
query = """
SELECT route,
         SUM(profit) AS total_profit
FROM route_performance
GROUP BY route
ORDER BY total_profit DESC
"""
pd.read_sql_query(query, conn)

# %% [markdown]
# ### Query 2 — Monthly Profit Trend for a Selected Route
# 
# This query retrieves month-by-month profit for a specific route (e.g., JFK-LAX, BOS-SEA),
# mirroring the trend analysis performed earlier in Python.

# %%
query = """
SELECT route, month, profit
FROM route_performance
WHERE route = 'JFK-LAX' OR route = 'BOS-SEA'
ORDER BY route;
"""
pd.read_sql_query(query, conn)

# %% [markdown]
# ### Query 3 — Using Window Functions to Compute Previous Month Revenue
# 
# This query demonstrates SQL window functions by calculating the previous month's
# revenue for each route, similar to the `shift()` logic used in Python.

# %%
query = """
SELECT
    route,
    month,
    revenue,
    LAG(revenue, 1) OVER (PARTITION BY route ORDER BY month) AS revenue_prev
FROM route_performance;
"""
pd.read_sql_query(query, conn)

# %% [markdown]
# ### Query 4 — Revenue Growth Calculation in SQL
# 
# This query computes month-over-month revenue growth using SQL expressions and window
# functions.

# %%
query = """
SELECT
    route,
    month,
    revenue,
    LAG(revenue, 1) OVER (PARTITION BY route ORDER BY month) AS revenue_prev,
    (revenue - LAG(revenue, 1) OVER (PARTITION BY route ORDER BY month))
        / LAG(revenue, 1) OVER (PARTITION BY route ORDER BY month) AS revenue_growth
FROM route_performance;
"""
pd.read_sql_query(query, conn)

# %% [markdown]
# ### Query 5 — Top 5 Most Profitable Route-Months
# 
# This query identifies the highest-profit route-month combinations, useful for spotting
# seasonal peaks or standout markets.

# %%
query = """
SELECT route, month, profit
FROM route_performance
ORDER BY profit DESC
LIMIT 5;
"""
pd.read_sql_query(query, conn)

# %% [markdown]
# ### Query 6 — Total Profit by Aircraft Type
# 
# This query aggregates total profit at the aircraft level to compare financial
# performance across JetBlue’s fleet types. It highlights which aircraft generate the
# strongest overall contribution to route profitability.

# %%
query = """
SELECT aircraft_type,
    SUM(profit) AS total_profit,
    AVG(profit) AS avg_profit_per_month
FROM route_performance
GROUP BY aircraft_type
ORDER BY total_profit DESC;
"""
pd.read_sql_query(query, conn)

# %% [markdown]
# ### Query 7 — Average Load Proxy by Aircraft Type
# 
# This query computes a simplified load factor proxy by comparing average passengers
# to the configured seat count for each aircraft type. While not a true load factor,
# it provides a useful approximation of demand relative to capacity.

# %%
query = """
SELECT
    aircraft_type,
    AVG(passengers) AS avg_passengers,
    AVG(seats_configured) AS avg_seats,
    AVG(passengers) * 1.0 / AVG(seats_configured) AS load_factor_proxy
FROM route_performance
GROUP BY aircraft_type
ORDER BY load_factor_proxy DESC;
"""
pd.read_sql_query(query, conn)

# %% [markdown]
# ### Query 8 — Profit per Seat by Aircraft Type
# 
# This query normalizes total profit by the number of configured seats for each aircraft
# type. Profit per seat provides a capacity-adjusted view of financial efficiency across
# the fleet.

# %%
query = """
SELECT
    aircraft_type,
    SUM(profit) AS total_profit,
    SUM(seats_configured) AS total_seats_configured,
    SUM(profit) * 1.0 / SUM(seats_configured) AS profit_per_seat_proxy
FROM route_performance
GROUP BY aircraft_type
ORDER BY profit_per_seat_proxy DESC;
"""
pd.read_sql_query(query, conn)

# %% [markdown]
# ### Query 9 — Monthly Profit Trend by Aircraft Type
# 
# This query shows month-by-month profit aggregated by aircraft type. It reveals
# seasonality patterns and helps compare how different fleet types perform throughout
# the year.

# %%
query = """
SELECT
    aircraft_type,
    month,
    SUM(profit) AS monthly_profit
FROM route_performance
GROUP BY aircraft_type, month
ORDER BY aircraft_type, month;
"""
pd.read_sql_query(query, conn)

# %% [markdown]
# ### Query 10 — Aircraft Type Mix by Route
# 
# This query identifies which aircraft types operate each route in the dataset. While
# each route in this project uses a single assigned aircraft type, this query mirrors
# real-world fleet assignment analysis where multiple aircraft may serve the same market.

# %%
query = """
SELECT
    route,
    aircraft_type,
    COUNT(*) AS months_operated
FROM route_performance
GROUP BY route, aircraft_type
ORDER BY route
"""
pd.read_sql_query(query, conn)

# %% [markdown]
# # Conclusion & Recommendations
# 
# This analysis provides a comprehensive view of JetBlue’s route performance across
# financial, operational, and growth dimensions. By integrating aircraft type,
# seat configuration, profitability metrics, and month-over-month trends, the project
# mirrors real-world airline network planning workflows.
# 
# ### Key Takeaways
# - Premium transcontinental routes operated with A321 Mint aircraft generate strong
#   revenue and competitive profit margins.
# - A220-operated routes show efficient right-sized capacity, with solid load factor
#   proxies despite smaller gauge.
# - Profitability varies significantly by route, with some markets showing strong
#   seasonality and others demonstrating consistent performance.
# - Growth metrics reveal momentum shifts that can inform schedule adjustments or
#   targeted pricing strategies.
# 
# ### Recommendations
# - **Increase focus on A321 Mint markets**, which show strong revenue and premium
#   demand characteristics.
# - **Monitor A220 routes for growth opportunities**, as these aircraft provide
#   flexibility and efficiency on thinner markets.
# - **Investigate underperforming routes** with persistently negative profit margins
#   to determine whether schedule, pricing, or aircraft assignment adjustments are
#   warranted.
# - **Leverage growth trends** to identify markets with accelerating demand for
#   potential frequency increases.


