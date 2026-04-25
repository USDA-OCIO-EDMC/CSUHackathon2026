"""Analog-year retrieval for the cone of uncertainty.

For each (state, year, checkpoint) we build a feature vector summarizing the
season-to-date weather + drought + vegetation signal:
  - cumulative GDD50_86 from May 1 to checkpoint
  - cumulative precip from May 1 to checkpoint
  - mean USDM DSCI from May 1 to checkpoint
  - month-by-month NDVI mean (months 5..checkpoint_month)

We z-score each feature using historical (2005..year-1) statistics, then take the
top-K nearest historical analogs (Euclidean distance over standardized features) and
report:
  - point forecast = model prediction (from Prithvi+head)
  - cone bounds    = empirical p10/p25/p75/p90 of analog-year actual NASS yields,
                     re-centered around the point forecast as relative deviations
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs" / "project.yaml").read_text())
K = CFG["uncertainty"]["analog_k"]
CHECKPOINTS = CFG["project"]["forecast_checkpoints"]


def load_features() -> pd.DataFrame:
    """Pre-built per-state-year-checkpoint feature table. Built by features pipeline."""
    return pd.read_parquet(ROOT / "data" / "processed" / "features" / "state_checkpoint_features.parquet")


def load_yields() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "raw" / "nass" / "yield_state.parquet")


def find_analogs(feats: pd.DataFrame, target_state: str, target_year: int, checkpoint: str) -> pd.DataFrame:
    pool = feats[(feats["state"] == target_state) & (feats["checkpoint"] == checkpoint)
                 & (feats["year"] < target_year)].copy()
    target = feats[(feats["state"] == target_state) & (feats["checkpoint"] == checkpoint)
                   & (feats["year"] == target_year)]
    if target.empty or pool.empty:
        return pd.DataFrame()

    cols = [c for c in pool.columns if c not in {"state", "year", "checkpoint", "fips"}]
    scaler = StandardScaler().fit(pool[cols].fillna(pool[cols].median()))
    P = scaler.transform(pool[cols].fillna(pool[cols].median()))
    T = scaler.transform(target[cols].fillna(pool[cols].median()))
    d = np.linalg.norm(P - T, axis=1)
    pool["distance"] = d
    return pool.nsmallest(K, "distance")


def cone(point_forecast: float, analog_years: list[int], state: str, yields: pd.DataFrame, target_year: int = 2025) -> dict:
    """Compute uncertainty cone from analog-year yield deviations from trend.

    Corn yields trend ~+2 bu/ac/yr in the corn belt, so analog years' raw values
    are not comparable to a 2025 forecast. We fit a linear trend over the full
    history and express each analog year as a *fractional deviation from trend*,
    then apply those fractional deviations to the model's point forecast.
    """
    sy = yields[yields["state_alpha"] == state].set_index("year")["Value"].astype(float).dropna()
    if len(sy) < 5:
        return {f"p{p}": point_forecast for p in (10, 25, 50, 75, 90)} | {"analog_years": analog_years}

    # Fit linear trend over history (excluding the target year)
    hist = sy[sy.index < target_year]
    coeffs = np.polyfit(hist.index.values, hist.values, deg=1)       # [slope, intercept]
    trend = lambda yr: coeffs[0] * yr + coeffs[1]                    # noqa: E731

    devs = []
    for y in analog_years:
        if y in sy.index:
            devs.append((sy.loc[y] - trend(y)) / trend(y))
    if not devs:
        return {f"p{p}": point_forecast for p in (10, 25, 50, 75, 90)} | {"analog_years": analog_years}
    devs = np.array(devs)
    return {
        "p10": float(point_forecast * (1 + np.quantile(devs, 0.10))),
        "p25": float(point_forecast * (1 + np.quantile(devs, 0.25))),
        "p50": float(point_forecast * (1 + np.quantile(devs, 0.50))),
        "p75": float(point_forecast * (1 + np.quantile(devs, 0.75))),
        "p90": float(point_forecast * (1 + np.quantile(devs, 0.90))),
        "analog_years": analog_years,
    }


def main() -> None:
    feats = load_features()
    yields = load_yields()
    preds = pd.read_parquet(ROOT / "reports" / "forecasts" / "point_forecasts_2025.parquet")

    out = []
    for r in preds.itertuples():
        analogs = find_analogs(feats, r.state, 2025, r.checkpoint)
        if analogs.empty:
            continue
        c = cone(r.point_forecast, analogs["year"].tolist(), r.state, yields, target_year=2025)
        out.append({"state": r.state, "checkpoint": r.checkpoint,
                    "point": r.point_forecast, **c})
    df = pd.DataFrame(out)
    path = ROOT / "reports" / "forecasts" / "yield_with_uncertainty_2025.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"→ {path}  ({len(df)} state-checkpoints)")


if __name__ == "__main__":
    main()
