#!/usr/bin/env python3
"""
FruitGuard ML - country-origin fruit fly pathway forecasting.

This version is built around the hackathon datasets the user uploaded:
  - passenger_data.csv: origin_country -> us_port passenger volumes by month
  - trade_data.csv: country fruit import volume by month
  - pest_status.csv: country/year pest presence and fruit fly type
  - us_port.csv: observed U.S. port detections by month

The detection labels are port/month totals, not individual-origin detections.
To train a country-origin pathway model, the script uses weak supervision:
observed port detections are attributed across origin_country -> U.S. port
pathways according to passenger volume, fruit imports, and pest presence.

That makes the model useful for prioritizing high-pressure pathways while
remaining honest about what the source data can and cannot prove.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = Path(".")
OUTPUT_DIR = Path(".")
FORECAST_START = pd.Timestamp("2026-05-01")
FORECAST_END = pd.Timestamp("2026-12-01")
RANDOM_STATE = 42
FILTERED_TEMP_CLASS = "BELOW_40F_FILTERED_OUT"
LEGACY_FILTERED_TEMP_CLASS = "UNSUITABLE_BELOW_40F"
FILTERED_RISK_TIER = "FILTERED_OUT"

PORT_ALIASES = {
    "Atlanta": "ATL",
    "Chicago": "ORD",
    "Dallas": "DFW",
    "Houston": "IAH",
    "JFK": "JFK",
    "LAX": "LAX",
    "Miami": "MIA",
    "Seattle": "SEA",
    "ATL": "ATL",
    "ORD": "ORD",
    "DFW": "DFW",
    "IAH": "IAH",
    "MIA": "MIA",
    "SEA": "SEA",
}

PORT_COORDS = {
    "LAX": {"lat": 33.9425, "lng": -118.4081},
    "MIA": {"lat": 25.7959, "lng": -80.2870},
    "JFK": {"lat": 40.6413, "lng": -73.7781},
    "SFO": {"lat": 37.6213, "lng": -122.3790},
    "ORD": {"lat": 41.9742, "lng": -87.9073},
    "IAH": {"lat": 29.9902, "lng": -95.3368},
    "ATL": {"lat": 33.6407, "lng": -84.4277},
    "SEA": {"lat": 47.4502, "lng": -122.3088},
    "DFW": {"lat": 32.8998, "lng": -97.0403},
}

PORT_MONTHLY_MIN_TEMP_F = {
    # Average monthly low temperature proxy. Replace with raster-derived
    # monthly minimum temperature when a gridded climate layer is available.
    "ATL": [34, 37, 44, 52, 61, 69, 72, 71, 65, 53, 43, 36],
    "DFW": [37, 41, 49, 57, 66, 74, 78, 78, 70, 58, 47, 39],
    "IAH": [44, 47, 54, 61, 69, 74, 76, 76, 72, 62, 52, 45],
    "JFK": [26, 28, 35, 44, 54, 64, 70, 69, 62, 51, 42, 33],
    "LAX": [49, 50, 52, 55, 58, 61, 64, 65, 64, 60, 53, 48],
    "MIA": [61, 63, 66, 70, 74, 77, 78, 78, 77, 74, 69, 64],
    "ORD": [17, 21, 30, 40, 50, 60, 66, 64, 57, 45, 34, 23],
    "SEA": [37, 38, 41, 44, 49, 53, 56, 57, 53, 47, 41, 37],
}

COUNTRY_COORDS = {
    "Argentina": {"lat": -34.6037, "lng": -58.3816},
    "Australia": {"lat": -25.2744, "lng": 133.7751},
    "Brazil": {"lat": -14.2350, "lng": -51.9253},
    "Canada": {"lat": 56.1304, "lng": -106.3468},
    "Chile": {"lat": -35.6751, "lng": -71.5430},
    "China": {"lat": 35.8617, "lng": 104.1954},
    "Germany": {"lat": 51.1657, "lng": 10.4515},
    "India": {"lat": 20.5937, "lng": 78.9629},
    "Japan": {"lat": 36.2048, "lng": 138.2529},
    "Mexico": {"lat": 23.6345, "lng": -102.5528},
    "Peru": {"lat": -9.1900, "lng": -75.0152},
    "South Africa": {"lat": -30.5595, "lng": 22.9375},
    "Spain": {"lat": 40.4637, "lng": -3.7492},
    "Thailand": {"lat": 15.8700, "lng": 100.9925},
    "United Kingdom": {"lat": 55.3781, "lng": -3.4360},
}

COUNTRY_ALIASES = {
    "UK": "United Kingdom",
    "United Kingdom": "United Kingdom",
}

PEST_STATUS_SCORE = {
    "Present": 1.0,
    "Emerging": 0.65,
    "Absent": 0.05,
}

CATEGORICAL_FEATURES = [
    "origin_country",
    "us_port",
    "pest_status",
    "fruit_fly_type",
]

NUMERIC_FEATURES = [
    "year",
    "month_num",
    "quarter",
    "month_sin",
    "month_cos",
    "passengers",
    "fruit_imports",
    "pest_score",
    "pathway_pressure",
    "passenger_share",
    "trade_share",
    "port_lag_1",
    "port_lag_2",
    "port_lag_3",
    "port_rolling_3_mean",
    "port_rolling_6_sum",
    "port_historical_mean",
    "country_port_passenger_mean",
    "country_import_mean",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def normalize_country(country: str) -> str:
    value = str(country).strip()
    return COUNTRY_ALIASES.get(value, value)


def normalize_port(port: str) -> str:
    value = str(port).strip()
    return PORT_ALIASES.get(value, value)


def parse_month(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def risk_tier(score: float) -> str:
    if score >= 0.78:
        return "CRITICAL"
    if score >= 0.58:
        return "HIGH"
    if score >= 0.36:
        return "MEDIUM"
    return "LOW"


def risk_tier_label(tier: str) -> str:
    if tier == FILTERED_RISK_TIER:
        return "Below 40F filtered out"
    return tier


def recommended_action(tier: str) -> str:
    if tier == FILTERED_RISK_TIER:
        return "No Fruit & Fly deployment; below 40F filtered out by survival threshold"
    if tier == "CRITICAL":
        return "Deploy Fruit & Fly decoy layer immediately and intensify inspection"
    if tier == "HIGH":
        return "Stage Fruit & Fly decoy layer and increase targeted surveillance"
    if tier == "MEDIUM":
        return "Monitor with focused inspection; keep decoy layers ready"
    return "Routine inspection; no decoy layer deployment"


def monthly_min_temperature_f(port: str, month: int) -> float:
    values = PORT_MONTHLY_MIN_TEMP_F.get(port)
    if not values:
        return 55.0
    return float(values[int(month) - 1])


def temperature_suitability_score(temp_f: float) -> float:
    if pd.isna(temp_f):
        return 1.0
    if temp_f < 40:
        return 0.0
    if temp_f < 45:
        return 0.35
    return 1.0


def temperature_suitability_class(temp_f: float) -> str:
    if pd.isna(temp_f):
        return "NO_TEMPERATURE_DATA"
    if temp_f < 40:
        return FILTERED_TEMP_CLASS
    if temp_f < 45:
        return "MARGINAL_40_TO_45F"
    return "SUITABLE_ABOVE_45F"


def reclassify_temperature_array(temp_f_array) -> np.ndarray:
    """Raster-friendly elementwise reclassification for temperature arrays."""
    arr = np.asarray(temp_f_array, dtype=float)
    return np.where(
        arr < 40,
        0.0,
        np.where(arr < 45, 0.35, 1.0),
    )


def combine_pathway_and_temperature_rasters(pathway_score_array, temp_f_array) -> np.ndarray:
    pathway = np.asarray(pathway_score_array, dtype=float)
    suitability = reclassify_temperature_array(temp_f_array)
    return np.clip(pathway * suitability, 0, 1)


def temperature_adjusted_action(row: pd.Series) -> str:
    if row["temp_suitability_class"] in {FILTERED_TEMP_CLASS, LEGACY_FILTERED_TEMP_CLASS}:
        return "No Fruit & Fly deployment; below 40F temperature mask, retain inbound inspection only"
    if row["temp_suitability_class"] == "MARGINAL_40_TO_45F":
        return "Marginal 40-45F suitability; monitor pathway and stage decoy layer only if detections rise"
    return recommended_action(row["risk_tier"])


def pheromone_for_type(fly_type: str) -> str:
    s = str(fly_type).lower()
    if "bactrocera dorsalis" in s or "oriental" in s:
        return "Methyl eugenol"
    if "anastrepha" in s or "mexican" in s:
        return "Trimedlure"
    if "ceratitis" in s or "mediterranean" in s or "rhagoletis" in s:
        return "Trimedlure"
    return "Methyl eugenol / Trimedlure"


def load_hackathon_data(data_dir: Path):
    print("=" * 78)
    print("STEP 1  Loading uploaded hackathon datasets")
    print("=" * 78)

    trade = pd.read_csv(data_dir / "trade_data.csv")
    passengers = pd.read_csv(data_dir / "passenger_data.csv")
    pest = pd.read_csv(data_dir / "pest_status.csv")
    detections = pd.read_csv(data_dir / "us_port.csv")

    trade["country"] = trade["country"].map(normalize_country)
    trade["month_start"] = parse_month(trade["month"])
    trade["fruit_imports"] = pd.to_numeric(trade["fruit_imports"], errors="coerce").fillna(0)
    trade = trade.dropna(subset=["month_start"])

    passengers["origin_country"] = passengers["origin_country"].map(normalize_country)
    passengers["us_port"] = passengers["us_port"].map(normalize_port)
    passengers["month_start"] = parse_month(passengers["month"])
    passengers["passengers"] = pd.to_numeric(passengers["passengers"], errors="coerce").fillna(0)
    passengers = passengers.dropna(subset=["month_start"])

    pest["country"] = pest["country"].map(normalize_country)
    pest["year"] = pd.to_numeric(pest["year"], errors="coerce").astype("Int64")
    pest["pest_status"] = pest["pest_status"].fillna("Absent")
    pest["fruit_fly_type"] = pest["fruit_fly_type"].fillna("No regulated fruit fly reported")
    pest["pest_score"] = pest["pest_status"].map(PEST_STATUS_SCORE).fillna(0.25)
    pest = pest.dropna(subset=["year"])

    detections["us_port"] = detections["us_port"].map(normalize_port)
    detections["month_start"] = parse_month(detections["month"])
    detections["detections"] = pd.to_numeric(detections["detections"], errors="coerce").fillna(0)
    detections = detections.dropna(subset=["month_start"])

    print(f"  trade rows:       {len(trade):5d} ({trade['country'].nunique()} countries)")
    print(f"  passenger rows:   {len(passengers):5d} ({passengers['us_port'].nunique()} U.S. ports)")
    print(f"  pest rows:        {len(pest):5d} ({pest['country'].nunique()} countries)")
    print(f"  detection rows:   {len(detections):5d} ({detections['us_port'].nunique()} U.S. ports)")
    print(
        "  observed months:  "
        f"{detections['month_start'].min().strftime('%Y-%m')} to "
        f"{detections['month_start'].max().strftime('%Y-%m')}"
    )
    print()

    return trade, passengers, pest, detections


def complete_pathway_panel(
    trade: pd.DataFrame,
    passengers: pd.DataFrame,
    pest: pd.DataFrame,
    detections: pd.DataFrame,
) -> pd.DataFrame:
    print("=" * 78)
    print("STEP 2  Building country-origin pathway training panel")
    print("=" * 78)

    months = pd.date_range(
        min(trade["month_start"].min(), passengers["month_start"].min(), detections["month_start"].min()),
        max(trade["month_start"].max(), passengers["month_start"].max(), detections["month_start"].max()),
        freq="MS",
    )
    countries = sorted(
        set(trade["country"].unique())
        | set(passengers["origin_country"].unique())
        | set(pest["country"].unique())
    )
    ports = sorted(set(passengers["us_port"].unique()) | set(detections["us_port"].unique()))

    grid = pd.MultiIndex.from_product(
        [countries, ports, months],
        names=["origin_country", "us_port", "month_start"],
    ).to_frame(index=False)

    pax = (
        passengers.groupby(["origin_country", "us_port", "month_start"], dropna=False)
        .agg(passengers=("passengers", "sum"))
        .reset_index()
    )
    cargo = (
        trade.groupby(["country", "month_start"], dropna=False)
        .agg(fruit_imports=("fruit_imports", "sum"))
        .reset_index()
        .rename(columns={"country": "origin_country"})
    )
    det = (
        detections.groupby(["us_port", "month_start"], dropna=False)
        .agg(port_detections=("detections", "sum"))
        .reset_index()
    )
    pest_year = pest.rename(columns={"country": "origin_country"}).copy()

    panel = (
        grid.merge(pax, on=["origin_country", "us_port", "month_start"], how="left")
        .merge(cargo, on=["origin_country", "month_start"], how="left")
        .merge(det, on=["us_port", "month_start"], how="left")
    )
    panel["year"] = panel["month_start"].dt.year.astype(int)
    panel = panel.merge(
        pest_year[["origin_country", "year", "pest_status", "fruit_fly_type", "pest_score"]],
        on=["origin_country", "year"],
        how="left",
    )

    panel["passengers"] = panel["passengers"].fillna(0.0)
    panel["fruit_imports"] = panel["fruit_imports"].fillna(0.0)
    panel["port_detections"] = panel["port_detections"].fillna(0.0)
    panel["pest_status"] = panel["pest_status"].fillna("Absent")
    panel["fruit_fly_type"] = panel["fruit_fly_type"].fillna("No regulated fruit fly reported")
    panel["pest_score"] = panel["pest_score"].fillna(0.05)

    # Normalize cargo and passengers separately so neither dominates purely by units.
    panel["passenger_share"] = (
        panel["passengers"]
        / panel.groupby(["us_port", "month_start"])["passengers"].transform("sum").replace(0, np.nan)
    ).fillna(0)
    panel["trade_share"] = (
        panel["fruit_imports"]
        / panel.groupby(["month_start"])["fruit_imports"].transform("sum").replace(0, np.nan)
    ).fillna(0)

    passenger_norm = (
        panel["passengers"]
        / panel.groupby(["month_start"])["passengers"].transform("max").replace(0, np.nan)
    ).fillna(0)
    trade_norm = (
        panel["fruit_imports"]
        / panel.groupby(["month_start"])["fruit_imports"].transform("max").replace(0, np.nan)
    ).fillna(0)

    panel["pathway_pressure"] = (
        passenger_norm * 0.55
        + trade_norm * 0.30
        + panel["pest_score"] * 0.15
    )
    panel["weighted_pressure"] = panel["pathway_pressure"] * (0.30 + panel["pest_score"])
    pressure_total = panel.groupby(["us_port", "month_start"])["weighted_pressure"].transform("sum")
    panel["attribution_share"] = (panel["weighted_pressure"] / pressure_total.replace(0, np.nan)).fillna(0)
    panel["attributed_detections"] = panel["port_detections"] * panel["attribution_share"]

    panel = add_calendar_features(panel)
    panel = add_port_history_features(panel)
    panel["country_port_passenger_mean"] = panel.groupby(["origin_country", "us_port"])[
        "passengers"
    ].transform("mean")
    panel["country_import_mean"] = panel.groupby("origin_country")["fruit_imports"].transform("mean")

    threshold = max(0.25, float(panel["attributed_detections"].quantile(0.90)))
    panel["high_pathway_risk"] = (panel["attributed_detections"] >= threshold).astype(int)

    print(f"  pathway rows:          {len(panel):6d}")
    print(f"  countries:             {panel['origin_country'].nunique():6d}")
    print(f"  U.S. ports:            {panel['us_port'].nunique():6d}")
    print(f"  months:                {panel['month_start'].nunique():6d}")
    print(f"  high-risk threshold:   {threshold:.3f} attributed detections/path/month")
    print(f"  high-risk rows:        {panel['high_pathway_risk'].sum():6d}")
    print()
    return panel


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month_num"] = out["month_start"].dt.month.astype(int)
    out["quarter"] = ((out["month_num"] - 1) // 3 + 1).astype(int)
    out["month_sin"] = np.sin(2 * np.pi * out["month_num"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month_num"] / 12)
    return out


def add_port_history_features(df: pd.DataFrame) -> pd.DataFrame:
    port_month = (
        df[["us_port", "month_start", "port_detections"]]
        .drop_duplicates()
        .sort_values(["us_port", "month_start"])
        .copy()
    )
    grouped = port_month.groupby("us_port")["port_detections"]
    port_month["port_lag_1"] = grouped.shift(1)
    port_month["port_lag_2"] = grouped.shift(2)
    port_month["port_lag_3"] = grouped.shift(3)
    port_month["port_rolling_3_mean"] = grouped.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    port_month["port_rolling_6_sum"] = grouped.transform(
        lambda s: s.shift(1).rolling(6, min_periods=1).sum()
    )
    port_month["port_historical_mean"] = grouped.transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    fill_cols = [
        "port_lag_1",
        "port_lag_2",
        "port_lag_3",
        "port_rolling_3_mean",
        "port_rolling_6_sum",
        "port_historical_mean",
    ]
    port_month[fill_cols] = port_month[fill_cols].fillna(0.0)
    return df.merge(
        port_month[["us_port", "month_start", *fill_cols]],
        on=["us_port", "month_start"],
        how="left",
    )


def make_regressor() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("categorical", one_hot_encoder(), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=450,
        min_samples_leaf=3,
        max_features=0.75,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def make_classifier() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("categorical", one_hot_encoder(), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=450,
        min_samples_leaf=3,
        max_features=0.75,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def evaluate_and_train(panel: pd.DataFrame):
    print("=" * 78)
    print("STEP 3  Training and validating country-pathway ML models")
    print("=" * 78)

    holdout_start = panel["month_start"].max() - pd.DateOffset(months=11)
    train_df = panel[panel["month_start"] < holdout_start].copy()
    test_df = panel[panel["month_start"] >= holdout_start].copy()

    regressor = make_regressor()
    classifier = make_classifier()

    regressor.fit(train_df[FEATURES], train_df["attributed_detections"])
    classifier.fit(train_df[FEATURES], train_df["high_pathway_risk"])

    pred = np.clip(regressor.predict(test_df[FEATURES]), 0, None)
    pred_class = classifier.predict(test_df[FEATURES])
    pred_prob = classifier.predict_proba(test_df[FEATURES])[:, 1]

    mae = float(mean_absolute_error(test_df["attributed_detections"], pred))
    rmse = float(math.sqrt(mean_squared_error(test_df["attributed_detections"], pred)))
    r2 = float(r2_score(test_df["attributed_detections"], pred))
    accuracy = float(accuracy_score(test_df["high_pathway_risk"], pred_class))
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_df["high_pathway_risk"], pred_class, average="binary", zero_division=0
    )
    if test_df["high_pathway_risk"].nunique() > 1:
        roc_auc = float(roc_auc_score(test_df["high_pathway_risk"], pred_prob))
        avg_precision = float(average_precision_score(test_df["high_pathway_risk"], pred_prob))
    else:
        roc_auc = float("nan")
        avg_precision = float("nan")

    cv_mae = None
    try:
        groups = panel["month_start"].astype(str)
        splits = min(5, panel["month_start"].nunique())
        scores = cross_val_score(
            make_regressor(),
            panel[FEATURES],
            panel["attributed_detections"],
            cv=GroupKFold(n_splits=splits),
            groups=groups,
            scoring="neg_mean_absolute_error",
        )
        cv_mae = float((-scores).mean())
    except Exception:
        cv_mae = None

    class_report = classification_report(
        test_df["high_pathway_risk"],
        pred_class,
        target_names=["Normal pathway pressure", "High pathway pressure"],
        zero_division=0,
    )

    print(f"  holdout window:       {holdout_start.strftime('%Y-%m')} to {panel['month_start'].max().strftime('%Y-%m')}")
    print(f"  regressor MAE:        {mae:.3f} attributed detections")
    print(f"  regressor RMSE:       {rmse:.3f} attributed detections")
    print(f"  regressor R2:         {r2:.3f}")
    if cv_mae is not None:
        print(f"  month-group CV MAE:   {cv_mae:.3f} attributed detections")
    print(f"  classifier accuracy:  {accuracy:.3f}")
    print(f"  classifier precision: {float(precision):.3f}")
    print(f"  classifier recall:    {float(recall):.3f}")
    print(f"  classifier F1:        {float(f1):.3f}")
    if not math.isnan(roc_auc):
        print(f"  classifier ROC AUC:   {roc_auc:.3f}")
        print(f"  average precision:    {avg_precision:.3f}")
    print()

    # Refit on all observed data before forecasting.
    regressor.fit(panel[FEATURES], panel["attributed_detections"])
    classifier.fit(panel[FEATURES], panel["high_pathway_risk"])

    metrics = {
        "holdout_window": f"{holdout_start.strftime('%Y-%m')} to {panel['month_start'].max().strftime('%Y-%m')}",
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "month_group_cv_mae": cv_mae,
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "classification_report": class_report,
        "feature_importance": feature_importance(regressor)[:18],
    }
    return regressor, classifier, metrics


def feature_importance(model: Pipeline) -> list[tuple[str, float]]:
    names = model.named_steps["preprocess"].get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_
    cleaned = [name.replace("categorical__", "").replace("numeric__", "") for name in names]
    return sorted(zip(cleaned, importances), key=lambda pair: pair[1], reverse=True)


def same_month_mean(df: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    source = df.copy()
    source["month_num"] = source["month_start"].dt.month
    return (
        source.groupby([*keys, "month_num"])[value]
        .mean()
        .reset_index()
        .rename(columns={value: f"{value}_forecast"})
    )


def port_history_values(panel: pd.DataFrame) -> dict[str, list[float]]:
    port_month = (
        panel[["us_port", "month_start", "port_detections"]]
        .drop_duplicates()
        .sort_values(["us_port", "month_start"])
    )
    return {
        port: group["port_detections"].astype(float).tolist()
        for port, group in port_month.groupby("us_port")
    }


def history_features(values: list[float]) -> dict[str, float]:
    return {
        "port_lag_1": values[-1] if len(values) >= 1 else 0.0,
        "port_lag_2": values[-2] if len(values) >= 2 else 0.0,
        "port_lag_3": values[-3] if len(values) >= 3 else 0.0,
        "port_rolling_3_mean": float(np.mean(values[-3:])) if values else 0.0,
        "port_rolling_6_sum": float(np.sum(values[-6:])) if values else 0.0,
        "port_historical_mean": float(np.mean(values)) if values else 0.0,
    }


def build_future_features(panel: pd.DataFrame, pest: pd.DataFrame) -> pd.DataFrame:
    countries = sorted(panel["origin_country"].unique())
    ports = sorted(panel["us_port"].unique())
    months = pd.date_range(FORECAST_START, FORECAST_END, freq="MS")
    future = pd.MultiIndex.from_product(
        [countries, ports, months], names=["origin_country", "us_port", "month_start"]
    ).to_frame(index=False)
    future["month_num"] = future["month_start"].dt.month
    future["year"] = future["month_start"].dt.year

    pax_forecast = same_month_mean(panel, ["origin_country", "us_port"], "passengers")
    trade_forecast = same_month_mean(panel, ["origin_country"], "fruit_imports")
    future = future.merge(
        pax_forecast, on=["origin_country", "us_port", "month_num"], how="left"
    ).merge(trade_forecast, on=["origin_country", "month_num"], how="left")
    future["passengers"] = future["passengers_forecast"].fillna(
        future.groupby(["origin_country", "us_port"])["passengers_forecast"].transform("mean")
    ).fillna(0)
    future["fruit_imports"] = future["fruit_imports_forecast"].fillna(
        future.groupby("origin_country")["fruit_imports_forecast"].transform("mean")
    ).fillna(0)
    future = future.drop(columns=["passengers_forecast", "fruit_imports_forecast"])

    pest_future = pest.rename(columns={"country": "origin_country"}).copy()
    future = future.merge(
        pest_future[["origin_country", "year", "pest_status", "fruit_fly_type", "pest_score"]],
        on=["origin_country", "year"],
        how="left",
    )
    future["pest_status"] = future["pest_status"].fillna("Absent")
    future["fruit_fly_type"] = future["fruit_fly_type"].fillna("No regulated fruit fly reported")
    future["pest_score"] = future["pest_score"].fillna(0.05)

    future["passenger_share"] = (
        future["passengers"]
        / future.groupby(["us_port", "month_start"])["passengers"].transform("sum").replace(0, np.nan)
    ).fillna(0)
    future["trade_share"] = (
        future["fruit_imports"]
        / future.groupby("month_start")["fruit_imports"].transform("sum").replace(0, np.nan)
    ).fillna(0)

    passenger_norm = (
        future["passengers"]
        / future.groupby("month_start")["passengers"].transform("max").replace(0, np.nan)
    ).fillna(0)
    trade_norm = (
        future["fruit_imports"]
        / future.groupby("month_start")["fruit_imports"].transform("max").replace(0, np.nan)
    ).fillna(0)
    future["pathway_pressure"] = (
        passenger_norm * 0.55 + trade_norm * 0.30 + future["pest_score"] * 0.15
    )
    future["country_port_passenger_mean"] = future.groupby(["origin_country", "us_port"])[
        "passengers"
    ].transform("mean")
    future["country_import_mean"] = future.groupby("origin_country")["fruit_imports"].transform("mean")
    future = add_calendar_features(future)
    return future


def forecast_pathways(panel: pd.DataFrame, pest: pd.DataFrame, regressor: Pipeline, classifier: Pipeline) -> pd.DataFrame:
    print("=" * 78)
    print("STEP 4  Forecasting remaining 2026 country-origin entry pathways")
    print("=" * 78)

    future = build_future_features(panel, pest)
    histories = port_history_values(panel)
    month_results = []

    for month_start in pd.date_range(FORECAST_START, FORECAST_END, freq="MS"):
        rows = future[future["month_start"] == month_start].copy()
        for port, values in histories.items():
            mask = rows["us_port"] == port
            for key, val in history_features(values).items():
                rows.loc[mask, key] = val
        for key in [
            "port_lag_1",
            "port_lag_2",
            "port_lag_3",
            "port_rolling_3_mean",
            "port_rolling_6_sum",
            "port_historical_mean",
        ]:
            rows[key] = rows[key].fillna(0.0)

        rows["predicted_path_detections"] = np.clip(regressor.predict(rows[FEATURES]), 0, None)
        rows["high_pathway_probability"] = classifier.predict_proba(rows[FEATURES])[:, 1]
        max_pred = max(float(rows["predicted_path_detections"].max()), 1.0)
        rows["entry_priority_score"] = (
            rows["high_pathway_probability"] * 0.45
            + (rows["predicted_path_detections"] / max_pred) * 0.30
            + rows["pathway_pressure"].clip(0, 1) * 0.15
            + rows["pest_score"] * 0.10
        ).clip(0, 1)
        rows["entry_risk_tier"] = rows["entry_priority_score"].apply(risk_tier)

        port_totals = rows.groupby("us_port")["predicted_path_detections"].sum()
        for port, total in port_totals.items():
            histories.setdefault(port, []).append(float(total))

        month_results.append(rows)

    forecast = pd.concat(month_results, ignore_index=True)
    forecast["destination_temp_f"] = forecast.apply(
        lambda row: monthly_min_temperature_f(row["us_port"], int(row["month_num"])),
        axis=1,
    )
    forecast["temp_suitability_score"] = forecast["destination_temp_f"].apply(temperature_suitability_score)
    forecast["temp_suitability_class"] = forecast["destination_temp_f"].apply(temperature_suitability_class)
    forecast["temperature_adjusted_detections"] = (
        forecast["predicted_path_detections"] * forecast["temp_suitability_score"]
    )
    forecast["pathway_priority_score"] = (
        forecast["entry_priority_score"] * forecast["temp_suitability_score"]
    ).clip(0, 1)
    forecast["risk_tier"] = forecast["pathway_priority_score"].apply(risk_tier)
    forecast["temperature_filtered"] = forecast["temp_suitability_class"].isin(
        [FILTERED_TEMP_CLASS, LEGACY_FILTERED_TEMP_CLASS]
    )
    forecast.loc[forecast["temperature_filtered"], "high_pathway_probability"] = 0.0
    forecast.loc[forecast["temperature_filtered"], "temperature_adjusted_detections"] = 0.0
    forecast.loc[forecast["temperature_filtered"], "pathway_priority_score"] = 0.0
    forecast.loc[forecast["temperature_filtered"], "risk_tier"] = FILTERED_RISK_TIER
    forecast["risk_label"] = forecast["risk_tier"].apply(risk_tier_label)
    forecast["recommended_action"] = forecast["risk_tier"].apply(recommended_action)
    forecast["pheromone"] = forecast["fruit_fly_type"].apply(pheromone_for_type)
    forecast["origin_lat"] = forecast["origin_country"].map(lambda c: COUNTRY_COORDS.get(c, {"lat": 0})["lat"])
    forecast["origin_lng"] = forecast["origin_country"].map(lambda c: COUNTRY_COORDS.get(c, {"lng": 0})["lng"])
    forecast["port_lat"] = forecast["us_port"].map(lambda p: PORT_COORDS.get(p, {"lat": 0})["lat"])
    forecast["port_lng"] = forecast["us_port"].map(lambda p: PORT_COORDS.get(p, {"lng": 0})["lng"])
    forecast["attribution_method"] = "weak supervision: port detections attributed by passenger, cargo, and pest pressure"
    forecast["temperature_adjustment_method"] = "monthly destination temperature suitability mask: <40F=filtered out, 40-45F=0.35, >=45F=1"
    forecast["recommended_action"] = forecast.apply(temperature_adjusted_action, axis=1)

    print(f"  forecast rows:       {len(forecast):6d}")
    print(f"  high/critical rows:  {forecast['risk_tier'].isin(['HIGH', 'CRITICAL']).sum():6d}")
    print(f"  below-40F masked:    {(forecast['temp_suitability_score'] == 0).sum():6d}")
    print(f"  forecast months:     {FORECAST_START.strftime('%Y-%m')} to {FORECAST_END.strftime('%Y-%m')}")
    print()
    return forecast.sort_values(
        ["pathway_priority_score", "predicted_path_detections"], ascending=[False, False]
    ).reset_index(drop=True)


def build_hotspots(forecast: pd.DataFrame) -> list[dict]:
    tier_rank = {FILTERED_RISK_TIER: -1, "LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    hotspots = []
    for port, group in forecast.groupby("us_port"):
        peak = group.sort_values(
            ["pathway_priority_score", "predicted_path_detections"],
            ascending=[False, False],
        ).iloc[0]
        countries = (
            group.groupby("origin_country")["temperature_adjusted_detections"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        pests = (
            group.groupby("fruit_fly_type")["temperature_adjusted_detections"]
            .sum()
            .sort_values(ascending=False)
            .head(4)
        )
        max_tier = max(group["risk_tier"], key=lambda t: tier_rank.get(t, 0))
        hotspots.append(
            {
                "port": port,
                "lat": float(peak["port_lat"]),
                "lng": float(peak["port_lng"]),
                "risk_tier": max_tier,
                "risk_label": risk_tier_label(max_tier),
                "peak_month": int(peak["month_num"]),
                "peak_origin_country": peak["origin_country"],
                "peak_fruit_fly_type": peak["fruit_fly_type"],
                "peak_priority_score": round(float(peak["pathway_priority_score"]), 4),
                "peak_entry_priority_score": round(float(peak["entry_priority_score"]), 4),
                "peak_destination_temp_f": round(float(peak["destination_temp_f"]), 1),
                "peak_temp_suitability_class": peak["temp_suitability_class"],
                "avg_priority_score": round(float(group["pathway_priority_score"].mean()), 4),
                "total_predicted_path_detections": round(float(group["predicted_path_detections"].sum()), 3),
                "temperature_adjusted_detections": round(float(group["temperature_adjusted_detections"].sum()), 3),
                "top_origin_countries": [
                    {"country": idx, "temperature_adjusted_detections": round(float(val), 3)}
                    for idx, val in countries.items()
                ],
                "top_fruit_fly_types": [
                    {"fruit_fly_type": idx, "temperature_adjusted_detections": round(float(val), 3)}
                    for idx, val in pests.items()
                ],
                "recommended_action": peak["recommended_action"],
                "pheromone": peak["pheromone"],
            }
        )
    return sorted(hotspots, key=lambda item: item["peak_priority_score"], reverse=True)


def summarize_paths(forecast: pd.DataFrame) -> pd.DataFrame:
    summary = (
        forecast.groupby(["origin_country", "us_port", "fruit_fly_type"], dropna=False)
        .agg(
            avg_priority_score=("pathway_priority_score", "mean"),
            peak_priority_score=("pathway_priority_score", "max"),
            avg_entry_priority_score=("entry_priority_score", "mean"),
            peak_entry_priority_score=("entry_priority_score", "max"),
            predicted_path_detections=("predicted_path_detections", "sum"),
            temperature_adjusted_detections=("temperature_adjusted_detections", "sum"),
            avg_destination_temp_f=("destination_temp_f", "mean"),
            min_destination_temp_f=("destination_temp_f", "min"),
            high_or_critical_months=(
                "risk_tier",
                lambda values: sum(v in {"HIGH", "CRITICAL"} for v in values),
            ),
            filtered_months=("temperature_filtered", "sum"),
            pathway_months=("temperature_filtered", "size"),
            peak_month=("month_num", lambda s: int(s.iloc[np.argmax(forecast.loc[s.index, "pathway_priority_score"])])),
            passengers=("passengers", "sum"),
            fruit_imports=("fruit_imports", "sum"),
        )
        .reset_index()
    )
    summary["risk_tier"] = summary["peak_priority_score"].apply(risk_tier)
    summary.loc[summary["filtered_months"] == summary["pathway_months"], "risk_tier"] = FILTERED_RISK_TIER
    summary["risk_label"] = summary["risk_tier"].apply(risk_tier_label)
    return summary.sort_values(
        ["peak_priority_score", "predicted_path_detections"], ascending=[False, False]
    )


def save_outputs(forecast: pd.DataFrame, hotspots: list[dict], summary: pd.DataFrame, metrics: dict) -> None:
    print("=" * 78)
    print("STEP 5  Saving updated project outputs")
    print("=" * 78)

    out = forecast.copy()
    out["year"] = out["month_start"].dt.year.astype(int)
    out["month"] = out["month_num"].astype(int)
    out["risk_label"] = out["risk_tier"].apply(risk_tier_label)
    export_cols = [
        "year",
        "month",
        "origin_country",
        "origin_lat",
        "origin_lng",
        "us_port",
        "port_lat",
        "port_lng",
        "passengers",
        "fruit_imports",
        "pest_status",
        "fruit_fly_type",
        "pest_score",
        "pathway_pressure",
        "predicted_path_detections",
        "temperature_adjusted_detections",
        "high_pathway_probability",
        "entry_priority_score",
        "entry_risk_tier",
        "pathway_priority_score",
        "risk_tier",
        "risk_label",
        "destination_temp_f",
        "temp_suitability_score",
        "temp_suitability_class",
        "temperature_filtered",
        "recommended_action",
        "pheromone",
        "attribution_method",
        "temperature_adjustment_method",
    ]
    out[export_cols].to_csv(OUTPUT_DIR / "ml_predictions.csv", index=False)
    (OUTPUT_DIR / "ml_predictions.json").write_text(
        json.dumps(out[export_cols].to_dict(orient="records"), indent=2)
    )
    (OUTPUT_DIR / "ml_hotspots.json").write_text(json.dumps(hotspots, indent=2))
    temp_cols = [
        "us_port",
        "month_num",
        "destination_temp_f",
        "temp_suitability_score",
        "temp_suitability_class",
    ]
    forecast[temp_cols].drop_duplicates().sort_values(["us_port", "month_num"]).rename(
        columns={"month_num": "month"}
    ).to_csv(OUTPUT_DIR / "temperature_suitability.csv", index=False)

    lines = [
        "FruitGuard ML Model Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Model purpose",
        "-------------",
        "Predict remaining-2026 fruit fly entry pressure by origin_country -> U.S. port pathway.",
        "",
        "New uploaded data incorporated",
        "------------------------------",
        "passenger_data.csv: monthly origin-country passenger volume to U.S. ports.",
        "trade_data.csv: monthly fruit import volume by origin country.",
        "pest_status.csv: country/year pest presence and associated fruit fly type.",
        "us_port.csv: monthly fruit fly detections by U.S. port.",
        "",
        "Model structure",
        "---------------",
        "Regressor: RandomForestRegressor predicts attributed pathway detections.",
        "Classifier: RandomForestClassifier predicts high-pressure pathways.",
        "Weak supervision: port detections are attributed across country-port pathways using passenger volume, fruit imports, and pest presence.",
        "Temperature suitability: Fruit & Fly response priority is masked by destination-month temperature suitability (<40F=filtered out, 40-45F=0.35, >=45F=1).",
        "Raster method: the same reclassification is implemented as NumPy array overlay functions for gridded climate rasters.",
        "",
        "Validation",
        "----------",
        f"Holdout window: {metrics['holdout_window']}",
        f"MAE: {metrics['mae']:.3f} attributed detections/path/month",
        f"RMSE: {metrics['rmse']:.3f} attributed detections/path/month",
        f"R2: {metrics['r2']:.3f}",
    ]
    if metrics["month_group_cv_mae"] is not None:
        lines.append(f"Month-group CV MAE: {metrics['month_group_cv_mae']:.3f}")
    lines.extend(
        [
            f"Classifier accuracy: {metrics['accuracy']:.3f}",
            f"Classifier precision: {metrics['precision']:.3f}",
            f"Classifier recall: {metrics['recall']:.3f}",
            f"Classifier F1: {metrics['f1']:.3f}",
        ]
    )
    if not math.isnan(metrics["roc_auc"]):
        lines.append(f"Classifier ROC AUC: {metrics['roc_auc']:.3f}")
        lines.append(f"Average precision: {metrics['average_precision']:.3f}")
    lines.extend(["", "Classification report", "---------------------", metrics["classification_report"].rstrip(), ""])
    lines.extend(["Top model features", "------------------"])
    for feature, score in metrics["feature_importance"]:
        lines.append(f"{feature:<50} {score:.4f}")
    lines.extend(["", "Top predicted country-origin pathways", "-------------------------------------"])
    for _, row in summary.head(25).iterrows():
        lines.append(
            f"{row['origin_country']} -> {row['us_port']} | {row['fruit_fly_type']} | "
            f"tier={row['risk_label']} | climate_peak={row['peak_priority_score']:.3f} | "
            f"entry_peak={row['peak_entry_priority_score']:.3f} | "
            f"temp_avg={row['avg_destination_temp_f']:.1f}F | "
            f"temp_adjusted={row['temperature_adjusted_detections']:.2f} | "
            f"entry_predicted={row['predicted_path_detections']:.2f} | "
            f"high/critical months={int(row['high_or_critical_months'])}"
        )
    lines.extend(["", "Fruit & Fly response hotspots", "----------------------------"])
    for item in hotspots:
        countries = ", ".join(c["country"] for c in item["top_origin_countries"][:3])
        lines.append(
            f"{item['port']} | {item['risk_label']} | peak month={item['peak_month']} | "
            f"origin={item['peak_origin_country']} | pest={item['peak_fruit_fly_type']} | "
            f"temp={item['peak_destination_temp_f']:.1f}F ({item['peak_temp_suitability_class']}) | "
            f"top countries={countries} | action={item['recommended_action']}"
        )
    lines.extend(
        [
            "",
            "Caveat",
            "------",
            "The data does not identify the origin country of individual detections. Pathway risk is inferred by weak supervision from port detections plus passenger/cargo/pest pathway pressure.",
            "Temperature suitability currently uses a monthly airport-temperature proxy by port; if a raster temperature layer is provided, use reclassify_temperature_array() and combine_pathway_and_temperature_rasters() to apply the same threshold rules cell by cell.",
        ]
    )
    (OUTPUT_DIR / "model_report.txt").write_text("\n".join(lines) + "\n")

    print("  saved ml_predictions.csv")
    print("  saved ml_predictions.json")
    print("  saved ml_hotspots.json")
    print("  saved temperature_suitability.csv")
    print("  saved model_report.txt")
    print()


def generate_arcgis_overlay(summary: pd.DataFrame, hotspots: list[dict]) -> None:
    records = []
    for _, row in summary.head(80).iterrows():
        country = row["origin_country"]
        port = row["us_port"]
        c = COUNTRY_COORDS.get(country, {"lat": 0.0, "lng": 0.0})
        p = PORT_COORDS.get(port, {"lat": 0.0, "lng": 0.0})
        records.append(
            {
                "origin_country": country,
                "origin_lat": c["lat"],
                "origin_lng": c["lng"],
                "us_port": port,
                "port_lat": p["lat"],
                "port_lng": p["lng"],
                "fruit_fly_type": row["fruit_fly_type"],
                "risk_tier": row["risk_tier"],
                "risk_label": row["risk_label"],
                "priority_score": round(float(row["peak_priority_score"]), 4),
                "entry_priority_score": round(float(row["peak_entry_priority_score"]), 4),
                "predicted_path_detections": round(float(row["predicted_path_detections"]), 3),
                "temperature_adjusted_detections": round(float(row["temperature_adjusted_detections"]), 3),
                "avg_destination_temp_f": round(float(row["avg_destination_temp_f"]), 1),
                "min_destination_temp_f": round(float(row["min_destination_temp_f"]), 1),
                "high_or_critical_months": int(row["high_or_critical_months"]),
                "recommended_action": recommended_action(row["risk_tier"]),
            }
        )

    js = f"""// FruitGuard ML country-origin pathway overlay for ArcGIS Maps SDK.
// Auto-generated by fruitguard_ml.py.
const FRUITGUARD_ML_PATHWAYS = {json.dumps(records, indent=2)};

function renderFruitGuardMLPathways(map, view) {{
  require(["esri/layers/GraphicsLayer", "esri/Graphic"], function(GraphicsLayer, Graphic) {{
    const existing = map.layers.find(function(layer) {{ return layer.title === "ML Country-Origin Entry Pathways"; }});
    if (existing) map.remove(existing);

    const layer = new GraphicsLayer({{ title: "ML Country-Origin Entry Pathways" }});
    const colors = {{
      CRITICAL: [198, 40, 40, 0.88],
      HIGH: [230, 81, 0, 0.82],
      MEDIUM: [249, 168, 37, 0.72],
      LOW: [102, 187, 106, 0.62],
      FILTERED_OUT: [69, 90, 100, 0.72]
    }};

    FRUITGUARD_ML_PATHWAYS.forEach(function(path) {{
      const color = colors[path.risk_tier] || colors.MEDIUM;
      const midLng = (path.origin_lng + path.port_lng) / 2;
      const midLat = Math.max(path.origin_lat, path.port_lat) + 12;
      layer.add(new Graphic({{
        geometry: {{
          type: "polyline",
          paths: [[[path.origin_lng, path.origin_lat], [midLng, midLat], [path.port_lng, path.port_lat]]],
          spatialReference: {{ wkid: 4326 }}
        }},
        symbol: {{
          type: "simple-line",
          color: color,
          width: 1.5 + path.priority_score * 5,
          style: "dash"
        }},
        attributes: path,
        popupTemplate: {{
          title: "{{origin_country}} -> {{us_port}}",
          content:
            "<b>Fruit fly:</b> {{fruit_fly_type}}<br>" +
            "<b>Risk tier:</b> {{risk_label}}<br>" +
            "<b>Climate-adjusted priority:</b> {{priority_score}}<br>" +
            "<b>Entry priority:</b> {{entry_priority_score}}<br>" +
            "<b>Temperature-adjusted detections:</b> {{temperature_adjusted_detections}}<br>" +
            "<b>Raw predicted detections:</b> {{predicted_path_detections}}<br>" +
            "<b>Avg destination temp:</b> {{avg_destination_temp_f}}F<br>" +
            "<b>High/critical months:</b> {{high_or_critical_months}}<br>" +
            "<b>Response:</b> {{recommended_action}}"
        }}
      }}));
    }});
    map.add(layer);
    return layer;
  }});
}}
"""
    (OUTPUT_DIR / "ml_map_layer.js").write_text(js)
    print("  saved ml_map_layer.js")


def generate_ml_dashboard(forecast: pd.DataFrame, hotspots: list[dict], summary: pd.DataFrame) -> None:
    """Generate the standalone ArcGIS ML pathway dashboard."""
    forecast_sorted = forecast.sort_values(
        ["pathway_priority_score", "predicted_path_detections"], ascending=[False, False]
    ).reset_index(drop=True)

    pathway_records = []
    for idx, row in forecast_sorted.iterrows():
        pathway_records.append(
            {
                "id": int(idx),
                "rank": int(idx + 1),
                "year": int(row["month_start"].year),
                "month": int(row["month_num"]),
                "origin_country": row["origin_country"],
                "origin_lat": round(float(row["origin_lat"]), 5),
                "origin_lng": round(float(row["origin_lng"]), 5),
                "us_port": row["us_port"],
                "port_lat": round(float(row["port_lat"]), 5),
                "port_lng": round(float(row["port_lng"]), 5),
                "passengers": round(float(row["passengers"]), 2),
                "fruit_imports": round(float(row["fruit_imports"]), 2),
                "pest_status": row["pest_status"],
                "fruit_fly_type": row["fruit_fly_type"],
                "pest_score": round(float(row["pest_score"]), 4),
                "pathway_pressure": round(float(row["pathway_pressure"]), 4),
                "predicted_path_detections": round(float(row["predicted_path_detections"]), 4),
                "temperature_adjusted_detections": round(float(row["temperature_adjusted_detections"]), 4),
                "high_pathway_probability": round(float(row["high_pathway_probability"]), 4),
                "entry_priority_score": round(float(row["entry_priority_score"]), 4),
                "entry_risk_tier": row["entry_risk_tier"],
                "pathway_priority_score": round(float(row["pathway_priority_score"]), 4),
                "risk_tier": row["risk_tier"],
                "risk_label": row["risk_label"],
                "destination_temp_f": round(float(row["destination_temp_f"]), 1),
                "temp_suitability_score": round(float(row["temp_suitability_score"]), 4),
                "temp_suitability_class": row["temp_suitability_class"],
                "temperature_filtered": bool(row["temperature_filtered"]),
                "recommended_action": row["recommended_action"],
                "pheromone": row["pheromone"],
            }
        )

    aggregate_records = []
    for idx, row in summary.sort_values(
        ["peak_priority_score", "predicted_path_detections"], ascending=[False, False]
    ).head(120).reset_index(drop=True).iterrows():
        country = row["origin_country"]
        port = row["us_port"]
        c = COUNTRY_COORDS.get(country, {"lat": 0.0, "lng": 0.0})
        p = PORT_COORDS.get(port, {"lat": 0.0, "lng": 0.0})
        aggregate_records.append(
            {
                "rank": int(idx + 1),
                "origin_country": country,
                "origin_lat": round(float(c["lat"]), 5),
                "origin_lng": round(float(c["lng"]), 5),
                "us_port": port,
                "port_lat": round(float(p["lat"]), 5),
                "port_lng": round(float(p["lng"]), 5),
                "fruit_fly_type": row["fruit_fly_type"],
                "avg_priority_score": round(float(row["avg_priority_score"]), 4),
                "peak_priority_score": round(float(row["peak_priority_score"]), 4),
                "avg_entry_priority_score": round(float(row["avg_entry_priority_score"]), 4),
                "peak_entry_priority_score": round(float(row["peak_entry_priority_score"]), 4),
                "predicted_path_detections": round(float(row["predicted_path_detections"]), 4),
                "temperature_adjusted_detections": round(float(row["temperature_adjusted_detections"]), 4),
                "avg_destination_temp_f": round(float(row["avg_destination_temp_f"]), 1),
                "min_destination_temp_f": round(float(row["min_destination_temp_f"]), 1),
                "high_or_critical_months": int(row["high_or_critical_months"]),
                "peak_month": int(row["peak_month"]),
                "passengers": round(float(row["passengers"]), 2),
                "fruit_imports": round(float(row["fruit_imports"]), 2),
                "risk_tier": row["risk_tier"],
                "risk_label": row["risk_label"],
                "recommended_action": recommended_action(row["risk_tier"]),
            }
        )

    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>FruitGuard ML - Temperature-Adjusted Pathways</title>
  <link rel="stylesheet" href="https://js.arcgis.com/4.29/esri/themes/dark/main.css"/>
  <script src="https://js.arcgis.com/4.29/"></script>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:"Segoe UI",system-ui,-apple-system,sans-serif;background:#091018;color:#e4edf4}
    .topbar{height:64px;background:#111923;border-bottom:1px solid #263647;display:flex;align-items:center;justify-content:space-between;padding:0 22px;gap:16px}
    .brand h1{font-size:1.15rem;letter-spacing:.2px}.brand span{color:#4fc3f7}.brand p{font-size:.72rem;color:#90a4ae;margin-top:3px}
    .nav{display:flex;align-items:center;gap:10px}.nav a{color:#b7c7d6;text-decoration:none;border:1px solid #2b3e52;border-radius:6px;padding:7px 10px;font-size:.78rem}.nav a.active{color:#061018;background:#4fc3f7;border-color:#4fc3f7;font-weight:700}
    .shell{height:calc(100vh - 64px);display:grid;grid-template-columns:410px minmax(0,1fr)}
    .sidebar{background:#0d1621;border-right:1px solid #263647;display:flex;flex-direction:column;min-height:0}
    .brief{padding:13px 14px;border-bottom:1px solid #263647;background:#101b27}
    .brief h2{font-size:.78rem;color:#d6e4ef;text-transform:uppercase;letter-spacing:.65px;margin-bottom:8px}.brief p{font-size:.75rem;line-height:1.35;color:#aebfce}.brief-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:10px}.brief-chip{background:#152333;border:1px solid #2b3e52;border-radius:7px;padding:8px}.brief-chip b{display:block;color:#4fc3f7;font-size:.78rem;margin-bottom:2px}.brief-chip span{font-size:.68rem;color:#9fb0bf}
    .controls{padding:14px;border-bottom:1px solid #263647;display:grid;grid-template-columns:1fr 1fr;gap:8px}
    .controls select,.controls input{width:100%;background:#131f2b;color:#e4edf4;border:1px solid #2b3e52;border-radius:6px;padding:8px;font-size:.78rem}
    .controls input{grid-column:1/-1}
    .stats{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:14px;border-bottom:1px solid #263647}
    .stat{background:#131f2b;border:1px solid #2b3e52;border-radius:8px;padding:10px}.stat b{display:block;color:#4fc3f7;font-size:1.15rem}.stat span{font-size:.68rem;color:#90a4ae;text-transform:uppercase;letter-spacing:.5px}
    .insight{padding:14px;border-bottom:1px solid #263647;background:#0f1a25}.insight h2{font-size:.78rem;color:#b7c7d6;text-transform:uppercase;letter-spacing:.65px;margin-bottom:8px}.insight-route{font-weight:800;color:#f3f7fb;font-size:.94rem}.insight-action{margin-top:8px;padding:8px;border-radius:7px;background:#152333;border:1px solid #2b3e52;color:#d6e4ef;font-size:.76rem;line-height:1.35}.signal-list{display:grid;gap:5px;margin-top:8px}.signal{display:flex;justify-content:space-between;gap:8px;font-size:.72rem;color:#9fb0bf}.signal strong{color:#e4edf4;text-align:right}
    .endpoint-status{margin-top:8px;padding:8px;border-radius:7px;background:#07121d;border:1px solid #2b3e52;color:#9fb0bf;font-size:.72rem;line-height:1.35}.endpoint-status strong{color:#4fc3f7}
    .list-head{padding:12px 14px;border-bottom:1px solid #263647;display:flex;align-items:center;justify-content:space-between}.list-head h2{font-size:.82rem;text-transform:uppercase;color:#b7c7d6;letter-spacing:.6px}.list-head span{font-size:.72rem;color:#90a4ae}
    #pathList{overflow:auto;min-height:0}.path-card{padding:12px 14px;border-bottom:1px solid #1c2b3b;cursor:pointer;background:#0d1621;transition:background .15s,border-left .15s}
    .path-card:hover,.path-card.active{background:#142231}.path-card.active{border-left:3px solid #4fc3f7}
    .path-top{display:flex;justify-content:space-between;gap:8px;align-items:center}.route{font-size:.91rem;font-weight:700}.rank{color:#90a4ae;font-size:.74rem;margin-right:6px}
    .tier{font-size:.67rem;font-weight:800;border-radius:999px;padding:3px 8px;color:#111}.tier.CRITICAL{background:#ef5350}.tier.HIGH{background:#ff8a50}.tier.MEDIUM{background:#ffd54f}.tier.LOW{background:#81c784}.tier.FILTERED_OUT{background:#455a64;color:#e8eef3}
    .meta{display:grid;grid-template-columns:1fr 1fr;gap:5px 10px;margin-top:8px;font-size:.72rem;color:#90a4ae}.meta strong{color:#d6e4ef}
    .scorebar{height:5px;background:#203247;border-radius:999px;margin-top:9px;overflow:hidden}.scorefill{height:100%;border-radius:999px}
    .main{display:grid;grid-template-rows:minmax(0,1fr) 260px;min-width:0}.map-wrap{position:relative;min-height:0}#map{width:100%;height:100%}
    .map-panel{position:absolute;left:16px;bottom:16px;z-index:10;background:rgba(13,22,33,.94);border:1px solid #2b3e52;border-radius:8px;padding:11px 13px;min-width:220px}
    .legend-title{font-size:.68rem;text-transform:uppercase;color:#90a4ae;letter-spacing:.6px;margin-bottom:7px}.legend-row{display:flex;align-items:center;gap:7px;font-size:.72rem;color:#c8d7e4;margin:5px 0}.dot{width:10px;height:10px;border-radius:50%}
    .bottom{background:#0d1621;border-top:1px solid #263647;display:grid;grid-template-columns:1.15fr 1fr 1fr 1.15fr;min-height:0}
    .panel{padding:14px;border-right:1px solid #263647;overflow:auto}.panel:last-child{border-right:0}.panel h3{font-size:.75rem;color:#90a4ae;text-transform:uppercase;letter-spacing:.7px;margin-bottom:10px}
    .hotspot{display:grid;grid-template-columns:42px 1fr 66px;gap:8px;align-items:center;padding:7px 0;border-bottom:1px solid #1c2b3b;font-size:.76rem}.hotspot strong{color:#e4edf4}.hotspot span{color:#90a4ae}
    .bar-row{display:grid;grid-template-columns:92px 1fr 54px;gap:8px;align-items:center;margin:8px 0;font-size:.73rem;color:#b7c7d6}.bar-track{height:8px;background:#203247;border-radius:999px;overflow:hidden}.bar-fill{height:100%;border-radius:999px}
    .evidence-row{display:grid;grid-template-columns:88px 1fr;gap:8px;padding:7px 0;border-bottom:1px solid #1c2b3b;font-size:.72rem;color:#9fb0bf}.evidence-row strong{color:#d6e4ef}.evidence-row span{line-height:1.3}
    .empty{padding:20px;color:#90a4ae;font-size:.84rem}
    @media(max-width:940px){.shell{grid-template-columns:1fr}.sidebar{max-height:48vh}.main{min-height:720px}.bottom{grid-template-columns:1fr}.topbar{height:auto;align-items:flex-start;flex-direction:column;padding:12px 14px}.shell{height:auto}}
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <h1>FruitGuard <span>ML Pathway Forecast</span></h1>
      <p>Predicted origin country to U.S. port entry pressure with destination temperature suitability, remaining 2026</p>
    </div>
    <nav class="nav">
      <a href="fruitguard_live.html">Main Dashboard</a>
      <a class="active">ML Pathways</a>
    </nav>
  </header>

  <div class="shell">
    <aside class="sidebar">
      <section class="brief">
        <h2>PPQ Pathway Intelligence</h2>
        <p>Ranks foreign-origin fruit fly entry pathways by predicted detections, host movement pressure, pest status, U.S. port temperature suitability, and Fruit &amp; Fly decoy-layer response priority.</p>
        <div class="brief-grid">
          <div class="brief-chip"><b>&lt;40F masked</b><span>Unsuitable for survival</span></div>
          <div class="brief-chip"><b>40-45F marginal</b><span>Lower confidence habitat</span></div>
          <div class="brief-chip"><b>45F+ suitable</b><span>Fruit &amp; Fly response window</span></div>
          <div class="brief-chip"><b>PPQ focus</b><span>Early detection at ports</span></div>
        </div>
      </section>
      <div class="controls">
        <select id="riskFilter"><option value="ALL">All risk tiers</option></select>
        <select id="monthFilter"><option value="ALL">All forecast months</option></select>
        <select id="portFilter"><option value="ALL">All U.S. ports</option></select>
        <select id="tempFilter"><option value="ALL">All actionable temperature classes</option></select>
        <input id="countryFilter" type="search" placeholder="Filter origin country"/>
      </div>
      <div class="stats">
        <div class="stat"><b id="statPaths">0</b><span>Filtered Paths</span></div>
        <div class="stat"><b id="statHigh">0</b><span>High/Critical</span></div>
        <div class="stat"><b id="statTop">0%</b><span>Top Climate Score</span></div>
        <div class="stat"><b id="statDetect">0.0</b><span>Adjusted Detections</span></div>
      </div>
      <section class="insight" id="selectedInsight"></section>
      <div class="list-head"><h2>Ranked Pathways</h2><span id="listCaption">top 80</span></div>
      <div id="pathList"></div>
    </aside>

    <main class="main">
      <section class="map-wrap">
        <div id="map"></div>
        <div class="map-panel">
          <div class="legend-title">Path Risk</div>
          <div class="legend-row"><span class="dot" style="background:#ef5350"></span>Critical</div>
          <div class="legend-row"><span class="dot" style="background:#ff8a50"></span>High</div>
          <div class="legend-row"><span class="dot" style="background:#ffd54f"></span>Medium</div>
          <div class="legend-row"><span class="dot" style="background:#81c784"></span>Low</div>
          <div class="legend-row"><span class="dot" style="background:#455a64"></span>Below 40F filtered out</div>
        </div>
      </section>
      <section class="bottom">
        <div class="panel"><h3>Fruit &amp; Fly Hotspots</h3><div id="hotspotList"></div></div>
        <div class="panel"><h3>Origin Country Pressure</h3><div id="countryBars"></div></div>
        <div class="panel"><h3>Port Pressure</h3><div id="portBars"></div></div>
        <div class="panel"><h3>Evidence Framework</h3><div id="evidenceList"></div></div>
      </section>
    </main>
  </div>

<script>
const PATHWAYS = __PATHWAYS__;
const HOTSPOTS = __HOTSPOTS__;
const AGGREGATED_PATHWAYS = __AGGREGATED_PATHWAYS__;
const FILTERED_TEMP_CLASSES = new Set(["BELOW_40F_FILTERED_OUT", "UNSUITABLE_BELOW_40F"]);
const FILTERED_RISK_TIER = "FILTERED_OUT";
const RISK_ORDER = {CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1, FILTERED_OUT: 0};
const RISK_COLORS = {
  CRITICAL: [239, 83, 80, 0.88],
  HIGH: [255, 138, 80, 0.82],
  MEDIUM: [255, 213, 79, 0.76],
  LOW: [129, 199, 132, 0.62],
  FILTERED_OUT: [69, 90, 100, 0.72]
};
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
let map, view, pathLayer, pointLayer, selectedId = null;
let endpointRequestToken = 0;

function pct(value){ return `${Math.round(value * 1000) / 10}%`; }
function fmt(value, digits=1){ return Number(value || 0).toLocaleString(undefined,{maximumFractionDigits:digits}); }
function colorCss(tier){ const c = RISK_COLORS[tier] || RISK_COLORS.LOW; return `rgba(${c[0]},${c[1]},${c[2]},${c[3]})`; }
function isTemperatureFiltered(row){
  return Boolean(row.temperature_filtered) ||
    FILTERED_TEMP_CLASSES.has(row.temp_suitability_class) ||
    Number(row.destination_temp_f) < 40 ||
    Number(row.temp_suitability_score) === 0;
}
function displayTier(row){ return isTemperatureFiltered(row) ? FILTERED_RISK_TIER : row.risk_tier; }
function displayRiskLabel(row){ return isTemperatureFiltered(row) ? "Below 40F filtered out" : (row.risk_label || row.risk_tier); }
function tempClassLabel(tempClass){
  if (FILTERED_TEMP_CLASSES.has(tempClass)) return "Below 40F filtered out";
  return tempClass.replaceAll("_", " ");
}

function populateFilters(){
  ["CRITICAL","HIGH","MEDIUM","LOW"].forEach(tier => {
    const option = document.createElement("option");
    option.value = tier; option.textContent = tier;
    document.getElementById("riskFilter").appendChild(option);
  });
  [...new Set(PATHWAYS.map(p => p.month))].sort((a,b)=>a-b).forEach(month => {
    const option = document.createElement("option");
    option.value = String(month); option.textContent = `${MONTHS[month - 1]} 2026`;
    document.getElementById("monthFilter").appendChild(option);
  });
  [...new Set(PATHWAYS.map(p => p.us_port))].sort().forEach(port => {
    const option = document.createElement("option");
    option.value = port; option.textContent = port;
    document.getElementById("portFilter").appendChild(option);
  });
  [...new Set(PATHWAYS.map(p => p.temp_suitability_class))].sort().forEach(tempClass => {
    const option = document.createElement("option");
    option.value = tempClass; option.textContent = tempClassLabel(tempClass);
    document.getElementById("tempFilter").appendChild(option);
  });
}

function filteredRows(){
  const risk = document.getElementById("riskFilter").value;
  const month = document.getElementById("monthFilter").value;
  const port = document.getElementById("portFilter").value;
  const tempClass = document.getElementById("tempFilter").value;
  const country = document.getElementById("countryFilter").value.trim().toLowerCase();
  return PATHWAYS.filter(row => {
    if (risk !== "ALL" && displayTier(row) !== risk) return false;
    if (month !== "ALL" && row.month !== Number(month)) return false;
    if (port !== "ALL" && row.us_port !== port) return false;
    if (tempClass === "ALL" && isTemperatureFiltered(row)) return false;
    if (tempClass !== "ALL" && row.temp_suitability_class !== tempClass) return false;
    if (country && !row.origin_country.toLowerCase().includes(country)) return false;
    return true;
  }).sort((a,b) =>
    (b.pathway_priority_score - a.pathway_priority_score) ||
    (b.predicted_path_detections - a.predicted_path_detections)
  );
}

function aggregate(rows, key){
  const map = new Map();
  rows.forEach(row => map.set(key(row), (map.get(key(row)) || 0) + row.temperature_adjusted_detections));
  return [...map.entries()].sort((a,b)=>b[1]-a[1]).slice(0,8);
}

function renderStats(rows){
  document.getElementById("statPaths").textContent = rows.length.toLocaleString();
  document.getElementById("statHigh").textContent = rows.filter(r => ["HIGH","CRITICAL"].includes(displayTier(r))).length.toLocaleString();
  document.getElementById("statTop").textContent = rows.length ? pct(rows[0].pathway_priority_score) : "0%";
  document.getElementById("statDetect").textContent = fmt(rows.reduce((s,r)=>s+r.temperature_adjusted_detections,0), 1);
}

function pathwayWhy(row){
  if (!row) return "";
  if (isTemperatureFiltered(row)) return "filtered out by below-40F survival rule";
  const signals = [];
  if (row.pest_status === "Present") signals.push("foreign pest present");
  if (row.temp_suitability_score >= 1) signals.push("destination climate suitable");
  if (row.fruit_imports > 0) signals.push("host import pressure");
  if (row.passengers > 0) signals.push("passenger pathway pressure");
  if (row.high_pathway_probability >= 0.65) signals.push("high detection probability");
  return signals.length ? signals.join(" + ") : "ranked by combined model indicators";
}

function renderSelectedInsight(rows){
  const target = PATHWAYS.find(p => p.id === selectedId) || rows[0];
  const box = document.getElementById("selectedInsight");
  if (!target) {
    box.innerHTML = `<h2>Response Brief</h2><div class="empty" style="padding:0">No pathway selected.</div>`;
    return;
  }
  selectedId = target.id;
  box.innerHTML = `
    <h2>Response Brief</h2>
    <div class="insight-route">${target.origin_country} → ${target.us_port}</div>
    <div class="insight-action">${target.recommended_action}</div>
    <div class="signal-list">
      <div class="signal"><span>Risk tier</span><strong>${displayRiskLabel(target)} · ${pct(target.pathway_priority_score)}</strong></div>
      <div class="signal"><span>Forecast month</span><strong>${MONTHS[target.month - 1]} ${target.year}</strong></div>
      <div class="signal"><span>Temperature</span><strong>${target.destination_temp_f}F · ${tempClassLabel(target.temp_suitability_class)}</strong></div>
      <div class="signal"><span>Species</span><strong>${target.fruit_fly_type}</strong></div>
      <div class="signal"><span>Why ranked</span><strong>${pathwayWhy(target)}</strong></div>
    </div>
    <div class="endpoint-status" id="endpointScore">Endpoint model: checking live SageMaker score...</div>`;
  requestEndpointScore(target);
}

function endpointPayload(row){
  return {
    instances: [{
      origin_country: row.origin_country,
      us_port: row.us_port,
      passengers: row.passengers,
      month: row.month,
      year: row.year,
      fruit_imports: row.fruit_imports,
      pest_status: row.pest_status,
      detections: row.predicted_path_detections || 0,
      detection_type: "Unknown",
      temp_suitability_score: row.temp_suitability_score,
      destination_temp_f: row.destination_temp_f
    }]
  };
}

async function requestEndpointScore(row){
  const token = ++endpointRequestToken;
  const box = document.getElementById("endpointScore");
  if (!box) return;
  if (isTemperatureFiltered(row)) {
    box.innerHTML = "Endpoint model: <strong>below 40F filtered out</strong> by survival threshold.";
    return;
  }
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(endpointPayload(row))
    });
    if (token !== endpointRequestToken) return;
    if (!response.ok) {
      box.innerHTML = "Endpoint model: batch forecast mode. Start <strong>run_endpoint_app.sh</strong> with SAGEMAKER_ENDPOINT_NAME to call SageMaker live.";
      return;
    }
    const result = await response.json();
    const pred = result.predictions && result.predictions[0];
    if (!pred) throw new Error("No prediction returned");
    box.innerHTML = `Endpoint model: <strong>${Math.round(pred.future_detection_probability * 1000) / 10}% ${pred.risk_tier}</strong> future-detection probability`;
  } catch (error) {
    if (token === endpointRequestToken) {
      box.innerHTML = "Endpoint model: batch forecast mode. Endpoint proxy is not running on this server.";
    }
  }
}

function renderPathList(rows){
  const list = document.getElementById("pathList");
  const topRows = rows.slice(0, 80);
  document.getElementById("listCaption").textContent = `top ${topRows.length}`;
  if (!topRows.length) {
    list.innerHTML = `<div class="empty">No pathways match the current filters.</div>`;
    return;
  }
  list.innerHTML = topRows.map((row, idx) => `
    <article class="path-card ${row.id === selectedId ? "active" : ""}" onclick="selectPath(${row.id})">
      <div class="path-top">
        <div class="route"><span class="rank">#${idx + 1}</span>${row.origin_country} → ${row.us_port}</div>
        <div class="tier ${displayTier(row)}">${displayRiskLabel(row)}</div>
      </div>
      <div class="meta">
        <span>Month <strong>${MONTHS[row.month - 1]}</strong></span>
        <span>Climate score <strong>${pct(row.pathway_priority_score)}</strong></span>
        <span>Entry score <strong>${pct(row.entry_priority_score)}</strong></span>
        <span>Adj. detect <strong>${fmt(row.temperature_adjusted_detections,2)}</strong></span>
        <span>Temp <strong>${row.destination_temp_f}F</strong></span>
        <span>High-prob <strong>${pct(row.high_pathway_probability)}</strong></span>
        <span>Suitability <strong>${pct(row.temp_suitability_score)}</strong></span>
        <span>Response <strong>${row.recommended_action}</strong></span>
        <span style="grid-column:1/-1">Pest <strong>${row.fruit_fly_type}</strong></span>
      </div>
      <div class="scorebar"><div class="scorefill" style="width:${Math.max(3,row.pathway_priority_score * 100)}%;background:${colorCss(displayTier(row))}"></div></div>
    </article>
  `).join("");
}

function renderBars(rows){
  const countryData = aggregate(rows, row => row.origin_country);
  const portData = aggregate(rows, row => row.us_port);
  const maxCountry = Math.max(...countryData.map(x=>x[1]), 1);
  const maxPort = Math.max(...portData.map(x=>x[1]), 1);
  document.getElementById("countryBars").innerHTML = countryData.map(([label,val]) => `
    <div class="bar-row"><span>${label}</span><div class="bar-track"><div class="bar-fill" style="width:${val/maxCountry*100}%;background:#4fc3f7"></div></div><strong>${fmt(val,1)}</strong></div>
  `).join("");
  document.getElementById("portBars").innerHTML = portData.map(([label,val]) => `
    <div class="bar-row"><span>${label}</span><div class="bar-track"><div class="bar-fill" style="width:${val/maxPort*100}%;background:#ff8a50"></div></div><strong>${fmt(val,1)}</strong></div>
  `).join("");
}

function renderHotspots(){
  document.getElementById("hotspotList").innerHTML = HOTSPOTS.map((item, idx) => `
    <div class="hotspot">
      <strong>#${idx + 1}</strong>
      <div><strong>${item.port}</strong><br><span>${item.peak_origin_country} · ${MONTHS[item.peak_month - 1]} · ${item.peak_destination_temp_f}F</span></div>
      <div class="tier ${item.risk_tier}" style="text-align:center">${item.risk_tier === FILTERED_RISK_TIER ? item.risk_label : pct(item.peak_priority_score)}</div>
    </div>
  `).join("");
}

function renderEvidenceList(rows){
  const highRows = rows.filter(r => ["HIGH","CRITICAL"].includes(displayTier(r)));
  const countries = new Set(rows.map(r => r.origin_country)).size;
  const ports = new Set(rows.map(r => r.us_port)).size;
  const unsuitable = PATHWAYS.filter(r => isTemperatureFiltered(r)).length;
  const marginal = PATHWAYS.filter(r => r.temp_suitability_class === "MARGINAL_40_TO_45F").length;
  const adjustedTotal = rows.reduce((s,r)=>s+r.temperature_adjusted_detections,0);
  document.getElementById("evidenceList").innerHTML = `
    <div class="evidence-row"><strong>Geospatial</strong><span>Origin-country to U.S.-port arcs, port response hotspots, and filterable destination layers.</span></div>
    <div class="evidence-row"><strong>Prediction</strong><span>${highRows.length} high/critical pathways from ${countries} foreign sources into ${ports} U.S. ports.</span></div>
    <div class="evidence-row"><strong>Biology</strong><span>${unsuitable} below-40F pathway-months masked; ${marginal} 40-45F pathway-months kept as marginal.</span></div>
    <div class="evidence-row"><strong>Indicators</strong><span>Pest status, route pressure, passenger flow, fruit imports, prior detections, and temperature suitability.</span></div>
    <div class="evidence-row"><strong>Response</strong><span>${fmt(adjustedTotal,1)} climate-adjusted detections represented in the current filtered view for Fruit & Fly staging decisions.</span></div>
  `;
}

function initMap(){
  require([
    "esri/Map",
    "esri/views/MapView",
    "esri/layers/GraphicsLayer",
    "esri/Graphic",
    "esri/widgets/Home",
    "esri/widgets/LayerList",
    "esri/widgets/Expand"
  ], (Map, MapView, GraphicsLayer, Graphic, Home, LayerList, Expand) => {
    window.ArcGISGraphic = Graphic;
    map = new Map({ basemap: "dark-gray-vector" });
    pathLayer = new GraphicsLayer({ title: "Ranked ML Pathways" });
    pointLayer = new GraphicsLayer({ title: "Origins and U.S. Ports" });
    map.addMany([pathLayer, pointLayer]);
    view = new MapView({
      container: "map",
      map,
      center: [-50, 28],
      zoom: 2,
      popup: { dockEnabled: true, dockOptions: { position: "top-right", breakpoint: false } }
    });
    view.when(() => {
      view.ui.add(new Home({view}), "top-left");
      view.ui.add(new Expand({view, content: new LayerList({view}), expandIcon: "layers", expanded: false}), "top-right");
      renderMap(filteredRows());
    });
  });
}

function routePath(row){
  const midLng = (row.origin_lng + row.port_lng) / 2;
  const midLat = Math.max(row.origin_lat, row.port_lat) + 12;
  return [[row.origin_lng,row.origin_lat],[midLng,midLat],[row.port_lng,row.port_lat]];
}

function renderMap(rows){
  if (!pathLayer || !window.ArcGISGraphic) return;
  pathLayer.removeAll();
  pointLayer.removeAll();
  const topRows = rows.slice(0, 80);
  const pointKeys = new Set();
  topRows.forEach(row => {
    const color = RISK_COLORS[displayTier(row)] || RISK_COLORS.LOW;
    const selected = row.id === selectedId;
    pathLayer.add(new window.ArcGISGraphic({
      geometry: { type: "polyline", paths: [routePath(row)], spatialReference: { wkid: 4326 } },
      attributes: row,
      symbol: { type: "simple-line", color, width: selected ? 5 : 1.2 + row.pathway_priority_score * 5, style: selected ? "solid" : "short-dash" },
      popupTemplate: {
        title: "{origin_country} → {us_port}",
        content:
          "<b>Risk:</b> {risk_label}<br>" +
          "<b>Climate-adjusted priority:</b> {pathway_priority_score}<br>" +
          "<b>Entry priority:</b> {entry_priority_score}<br>" +
          "<b>Temperature:</b> {destination_temp_f}F ({temp_suitability_class})<br>" +
          "<b>Adjusted detections:</b> {temperature_adjusted_detections}<br>" +
          "<b>Raw predicted detections:</b> {predicted_path_detections}<br>" +
          "<b>Month:</b> {month}/2026<br>" +
          "<b>Pest:</b> {fruit_fly_type}<br>" +
          "<b>Response:</b> {recommended_action}"
      }
    }));
    const originKey = `o-${row.origin_country}`;
    if (!pointKeys.has(originKey)) {
      pointKeys.add(originKey);
      pointLayer.add(new window.ArcGISGraphic({
        geometry: { type: "point", longitude: row.origin_lng, latitude: row.origin_lat },
        attributes: { name: row.origin_country, type: "Origin country" },
        symbol: { type: "simple-marker", style: "diamond", color: [79,195,247,.86], size: 8, outline: { color: [7,16,24,1], width: 1 } },
        popupTemplate: { title: "{name}", content: "{type}" }
      }));
    }
    const portKey = `p-${row.us_port}`;
    if (!pointKeys.has(portKey)) {
      pointKeys.add(portKey);
      pointLayer.add(new window.ArcGISGraphic({
        geometry: { type: "point", longitude: row.port_lng, latitude: row.port_lat },
        attributes: { name: row.us_port, type: "U.S. port" },
        symbol: { type: "simple-marker", style: "circle", color: [255,255,255,.9], size: 10, outline: { color: [255,138,80,1], width: 2 } },
        popupTemplate: { title: "{name}", content: "{type}" }
      }));
    }
  });
}

function selectPath(id){
  selectedId = id;
  const row = PATHWAYS.find(p => p.id === id);
  renderAll();
  if (row && view) {
    view.goTo({ center: [(row.origin_lng + row.port_lng) / 2, (row.origin_lat + row.port_lat) / 2], zoom: 3 });
  }
}
window.selectPath = selectPath;

function renderAll(){
  const rows = filteredRows();
  renderStats(rows);
  renderSelectedInsight(rows);
  renderPathList(rows);
  renderBars(rows);
  renderEvidenceList(rows);
  renderMap(rows);
}

["riskFilter","monthFilter","portFilter","tempFilter"].forEach(id => document.getElementById(id).addEventListener("change", renderAll));
document.getElementById("countryFilter").addEventListener("input", renderAll);

populateFilters();
renderHotspots();
initMap();
renderAll();
</script>
</body>
</html>
"""

    html = (
        html.replace("__PATHWAYS__", json.dumps(pathway_records, indent=2))
        .replace("__HOTSPOTS__", json.dumps(hotspots, indent=2))
        .replace("__AGGREGATED_PATHWAYS__", json.dumps(aggregate_records, indent=2))
    )
    (OUTPUT_DIR / "ml_dashboard.html").write_text(html)
    print("  saved ml_dashboard.html")


def main() -> None:
    print()
    print("=" * 78)
    print("FruitGuard ML - country-origin fruit fly pathway forecasting")
    print("=" * 78)
    print()

    trade, passengers, pest, detections = load_hackathon_data(DATA_DIR)
    panel = complete_pathway_panel(trade, passengers, pest, detections)
    regressor, classifier, metrics = evaluate_and_train(panel)
    forecast = forecast_pathways(panel, pest, regressor, classifier)
    hotspots = build_hotspots(forecast)
    summary = summarize_paths(forecast)
    save_outputs(forecast, hotspots, summary, metrics)
    generate_arcgis_overlay(summary, hotspots)
    generate_ml_dashboard(forecast, hotspots, summary)

    print("=" * 78)
    print("Done. Updated ML outputs are ready.")
    print("=" * 78)
    print("Top Fruit & Fly response hotspots:")
    for item in hotspots[:8]:
        countries = ", ".join(c["country"] for c in item["top_origin_countries"][:3])
        print(
            f"  {item['port']:4s} {item['risk_label'][:18]:18s} "
            f"month {item['peak_month']:2d}  "
            f"{item['peak_origin_country']:<16s} {countries}"
        )
    print()


if __name__ == "__main__":
    main()
