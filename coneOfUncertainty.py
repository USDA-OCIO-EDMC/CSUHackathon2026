
"""
=============================================================================
CSU Hackathon — Geospatial AI Crop Yield Forecasting
Cone of Uncertainty Pipeline
=============================================================================
Data Sources (as per hackathon brief):
  - USDA NASS Historical Corn Yield Data (2005–2024) — Bu/Acre
  - Weather/Climate Features: GDD, Precipitation, PDSI, NDVI (proxy)
  - Cropland Data Layer (CDL) — corn field masking (simulated)
  - Prithvi / HLS Satellite signals (NDVI proxy used here)

Deliverables:
  1. Yield forecast at 4 time points: Aug1, Sep1, Oct1, Final
  2. Cone of Uncertainty (10th–90th percentile) per state per forecast date
  3. Analog year identification using Euclidean distance on weather features
  4. Results exported to CSV

States: Iowa, Colorado, Wisconsin, Missouri, Nebraska
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
STATES         = ["Iowa", "Colorado", "Wisconsin", "Missouri", "Nebraska"]
FORECAST_DATES = ["Aug1", "Sep1", "Oct1", "Final"]
HIST_YEARS     = list(range(2005, 2025))   # 2005–2024
TARGET_YEAR    = 2025
TOP_N_ANALOGS  = 7                          # number of analog years to select
LOWER_PCT      = 10                         # lower bound percentile
UPPER_PCT      = 90                         # upper bound percentile
RANDOM_SEED    = 42

np.random.seed(RANDOM_SEED)

# =============================================================================
# STEP 1 — LOAD / SIMULATE USDA NASS HISTORICAL YIELD DATA (2005–2024)
#
#  ✅ To use REAL data, replace the simulation block below with:
#       yield_df = pd.read_csv("usda_nass_corn_yield.csv")
#       yield_df = yield_df[yield_df["commodity_desc"] == "CORN"]
#       yield_df = yield_df[yield_df["statisticcat_desc"] == "YIELD"]
#       yield_df = yield_df[["year", "state_name", "Value"]]
#       yield_df.columns = ["year", "state", "yield_bu_acre"]
#       yield_df = yield_df[yield_df["state"].isin(STATES)]
# =============================================================================
print("=" * 65)
print("STEP 1: Loading Historical Yield Data (USDA NASS 2005–2024)")
print("=" * 65)

# Realistic base yields (Bu/Acre) with linear trend + stochastic noise
BASE_YIELDS = {
    "Iowa":      170,
    "Colorado":  145,
    "Wisconsin": 155,
    "Missouri":  140,
    "Nebraska":  160,
}
TREND_PER_YEAR = 1.2   # ~1.2 Bu/Acre improvement per year (national trend)

yield_records = []
for state in STATES:
    for i, year in enumerate(HIST_YEARS):
        yield_val = BASE_YIELDS[state] + TREND_PER_YEAR * i + np.random.normal(0, 12)
        yield_records.append({
            "year":          year,
            "state":         state,
            "yield_bu_acre": round(yield_val, 1),
        })

yield_df = pd.DataFrame(yield_records)
print(f"  Yield records loaded: {len(yield_df)} rows")
print(yield_df.pivot(index="year", columns="state", values="yield_bu_acre").round(1))

# =============================================================================
# STEP 2 — ENGINEER WEATHER & GEOSPATIAL FEATURES (2005–2025)
#
#  Features used for analog year matching:
#    GDD    — Growing Degree Days accumulated to forecast date
#    Precip — Cumulative precipitation (inches)
#    PDSI   — Palmer Drought Severity Index
#    NDVI   — Normalized Difference Vegetation Index (from HLS/Prithvi)
#
#  ✅ To use REAL data, replace the simulation block below with:
#       weather_df = pd.read_csv("noaa_weather_features.csv")
#       # Ensure columns: year, state, forecast_date, GDD, Precip, PDSI, NDVI
# =============================================================================
print(" " + "=" * 65)
print("STEP 2: Engineering Weather & Geospatial Features")
print("=" * 65)

weather_records = []
for state in STATES:
    for year in HIST_YEARS + [TARGET_YEAR]:
        for fd in FORECAST_DATES:
            weather_records.append({
                "year":          year,
                "state":         state,
                "forecast_date": fd,
                "GDD":           round(np.random.normal(1200, 150), 1),   # Growing Degree Days
                "Precip":        round(np.random.normal(18, 4),   2),     # Precipitation (inches)
                "PDSI":          round(np.random.normal(0, 1.5),  2),     # Drought Index
                "NDVI":          round(np.random.normal(0.65, 0.08), 3),  # Vegetation Index
            })

weather_df = pd.DataFrame(weather_records)
print(f"  Weather feature records: {len(weather_df)} rows")
print(f"  Features: {['GDD', 'Precip', 'PDSI', 'NDVI']}")
print(weather_df[weather_df["state"] == "Iowa"].head(8).to_string(index=False))

# =============================================================================
# STEP 3 — ANALOG YEAR IDENTIFICATION
#
#  For each state × forecast_date:
#    1. Normalize all features using StandardScaler
#    2. Compute Euclidean distance between 2025 and each historical year
#    3. Select TOP_N_ANALOGS closest (most similar) years
# =============================================================================
print(" " + "=" * 65)
print("STEP 3: Identifying Analog Years (Euclidean Distance Matching)")
print("=" * 65)

FEATURE_COLS  = ["GDD", "Precip", "PDSI", "NDVI"]
analog_records = []

for state in STATES:
    for fd in FORECAST_DATES:
        # Filter to this state + forecast date
        subset = weather_df[
            (weather_df["state"] == state) &
            (weather_df["forecast_date"] == fd)
        ].copy().reset_index(drop=True)

        # Normalize features
        scaler    = StandardScaler()
        scaled    = scaler.fit_transform(subset[FEATURE_COLS])
        scaled_df = pd.DataFrame(scaled, columns=FEATURE_COLS)
        scaled_df["year"] = subset["year"].values

        # Separate 2025 target vs historical years
        target_vec  = scaled_df[scaled_df["year"] == TARGET_YEAR][FEATURE_COLS].values[0]
        hist_scaled = scaled_df[scaled_df["year"] != TARGET_YEAR].copy()

        # Compute Euclidean distance for each historical year
        hist_scaled["distance"] = hist_scaled[FEATURE_COLS].apply(
            lambda row: euclidean(row.values, target_vec), axis=1
        )

        # Select top-N most similar analog years
        top_analogs = hist_scaled.nsmallest(TOP_N_ANALOGS, "distance")["year"].tolist()

        analog_records.append({
            "state":         state,
            "forecast_date": fd,
            "analog_years":  top_analogs,
        })

        print(f"  {state:10s} | {fd:6s} → Analog years: {top_analogs}")

analog_df = pd.DataFrame(analog_records)

# =============================================================================
# STEP 4 — CONE OF UNCERTAINTY CONSTRUCTION
#
#  For each state × forecast_date, using the analog year yields:
#    - Point Estimate = Mean of analog year yields
#    - Lower Bound    = 10th percentile  (LOWER_PCT)
#    - Upper Bound    = 90th percentile  (UPPER_PCT)
#    - Cone Width     = Upper − Lower
#
#  The cone should naturally NARROW as the season progresses:
#    Aug1 (widest) → Sep1 → Oct1 → Final (narrowest)
# =============================================================================
print(" " + "=" * 65)
print("STEP 4: Computing Cone of Uncertainty")
print("=" * 65)

cone_records = []
for _, row in analog_df.iterrows():
    state        = row["state"]
    fd           = row["forecast_date"]
    analog_years = row["analog_years"]

    # Retrieve actual USDA NASS yields for the analog years
    analog_yields = yield_df[
        (yield_df["state"] == state) &
        (yield_df["year"].isin(analog_years))
    ]["yield_bu_acre"].values

    point_est  = round(float(np.mean(analog_yields)),              1)
    lower_bnd  = round(float(np.percentile(analog_yields, LOWER_PCT)), 1)
    upper_bnd  = round(float(np.percentile(analog_yields, UPPER_PCT)), 1)
    cone_width = round(upper_bnd - lower_bnd,                      1)

    cone_records.append({
        "state":          state,
        "forecast_date":  fd,
        "analog_years":   str(analog_years),
        "point_estimate": point_est,
        "lower_10pct":    lower_bnd,
        "upper_90pct":    upper_bnd,
        "cone_width":     cone_width,
    })

cone_df = pd.DataFrame(cone_records)
cone_df["forecast_date"] = pd.Categorical(
    cone_df["forecast_date"], categories=FORECAST_DATES, ordered=True
)
cone_df = cone_df.sort_values(["state", "forecast_date"]).reset_index(drop=True)

# =============================================================================
# STEP 5 — DISPLAY RESULTS
# =============================================================================
print(" " + "=" * 65)
print("STEP 5: Final Cone of Uncertainty Results — 2025 Season")
print("=" * 65)

display_cols = [
    "state", "forecast_date", "point_estimate",
    "lower_10pct", "upper_90pct", "cone_width"
]
print(cone_df[display_cols].to_string(index=False))

# =============================================================================
# STEP 6 — EXPORT RESULTS TO CSV
# =============================================================================
output_path = "cone_of_uncertainty_2025.csv"
cone_df[display_cols].to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# =============================================================================
# STEP 7 — SUMMARY STATISTICS PER STATE
# =============================================================================
print(" " + "=" * 65)
print("STEP 7: Summary — Average Yield Forecast per State (2025)")
print("=" * 65)

summary = cone_df.groupby("state").agg(
    avg_yield      = ("point_estimate", "mean"),
    avg_lower      = ("lower_10pct",    "mean"),
    avg_upper      = ("upper_90pct",    "mean"),
    avg_cone_width = ("cone_width",     "mean"),
).round(1).reset_index()
summary.columns = ["State", "Avg Yield (Bu/Ac)", "Avg Lower", "Avg Upper", "Avg Cone Width"]
print(summary.to_string(index=False))

print(" " + "=" * 65)
print("Pipeline Complete.")
print("=" * 65)

