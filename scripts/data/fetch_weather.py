"""Fetch growing-season weather for all county centroids via Open-Meteo (free, no key).

Pulls daily Tmax, Tmin, precip for 2022-2025 growing seasons (May–Oct).
Computes per county-year:
  - gdd_acc     : cumulative GDD (base 10°C / ceiling 30°C) May–Aug
  - precip_acc  : cumulative precip mm May–Aug
  - heat_days   : days Tmax > 32°C (90°F) in June–July (pollination stress)
  - vpd_jul_mean: mean VPD July (heat stress proxy from Tmax)
  - precip_jul  : July precip (critical grain-fill month)

Output: data/processed/features/weather_features.parquet
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "data" / "processed" / "features"
OUT.mkdir(parents=True, exist_ok=True)

YEARS       = [2022, 2023, 2024, 2025]
API_URL     = "https://archive-api.open-meteo.com/v1/archive"
RETRY_DELAY = 3
MAX_RETRIES = 4


def fetch_county(lat: float, lon: float) -> pd.DataFrame | None:
    """Fetch daily weather for a county centroid 2022-01-01 to 2025-10-31."""
    params = {
        "latitude":   round(lat, 4),
        "longitude":  round(lon, 4),
        "start_date": "2022-01-01",
        "end_date":   "2025-10-31",
        "daily":      "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone":   "America/Chicago",
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(API_URL, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()["daily"]
                return pd.DataFrame({
                    "date":   pd.to_datetime(data["time"]),
                    "tmax":   data["temperature_2m_max"],
                    "tmin":   data["temperature_2m_min"],
                    "precip": data["precipitation_sum"],
                })
            elif r.status_code == 429:
                time.sleep(RETRY_DELAY * (attempt + 1))
        except Exception as e:
            time.sleep(RETRY_DELAY)
    return None


def compute_features(daily: pd.DataFrame, year: int) -> dict:
    d = daily.copy()
    d["year"]  = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d = d[d["year"] == year].copy()

    # GDD base 10°C ceiling 30°C
    d["tmean"] = (d["tmax"].clip(upper=30) + d["tmin"].clip(lower=10)) / 2
    d["gdd"]   = (d["tmean"] - 10).clip(lower=0)

    gs = d[(d["month"] >= 5) & (d["month"] <= 8)]   # May–Aug growing season

    jul = d[d["month"] == 7]

    # Saturation vapor pressure approximation for VPD
    def svp(t): return 0.6108 * np.exp(17.27 * t / (t + 237.3))
    jul_vpd = (svp(jul["tmax"].fillna(25)) - svp(jul["tmin"].fillna(15))).mean()

    return {
        "gdd_acc":      float(gs["gdd"].sum()),
        "precip_acc":   float(gs["precip"].fillna(0).sum()),
        "heat_days":    int(d[(d["month"].isin([6, 7])) & (d["tmax"] > 32)].shape[0]),
        "vpd_jul_mean": float(jul_vpd) if not np.isnan(jul_vpd) else 1.5,
        "precip_jul":   float(jul["precip"].fillna(0).sum()),
    }


def main() -> None:
    centroids = pd.read_csv("/tmp/county_centroids.csv", dtype={"fips": str})
    centroids["fips"] = centroids["fips"].str.zfill(5)
    print(f"Fetching weather for {len(centroids)} counties × {len(YEARS)} years...")

    rows = []
    for i, row in enumerate(centroids.itertuples()):
        daily = fetch_county(row.lat, row.lon)
        if daily is None:
            print(f"  [{i+1}/{len(centroids)}] {row.fips} FAILED")
            continue
        for year in YEARS:
            feats = compute_features(daily, year)
            rows.append({"fips": row.fips, "year": year, **feats})
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(centroids)}] {row.fips} ok", flush=True)
        time.sleep(0.15)   # ~6 req/sec — well under free-tier limit

    df = pd.DataFrame(rows)
    path = OUT / "weather_features.parquet"
    df.to_parquet(path, index=False)
    print(f"\n→ {path}  ({len(df):,} county-years)")


if __name__ == "__main__":
    main()
