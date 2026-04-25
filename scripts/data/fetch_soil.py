"""Fetch soil properties for county centroids via SoilGrids REST API (free).

Properties pulled at 0-30cm depth:
  - awc      : available water capacity (cm³/cm³) — most important for yield
  - soc      : soil organic carbon (g/kg)
  - clay_pct : clay fraction (g/100g)
  - sand_pct : sand fraction
  - bdod     : bulk density (cg/cm³)

Output: data/processed/features/soil_features.parquet
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

API_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPS   = ["bdod", "cec", "clay", "sand", "soc", "phh2o"]
DEPTHS  = ["0-5cm", "5-15cm", "15-30cm"]


def fetch_soil(lat: float, lon: float) -> dict | None:
    params = {
        "lon":      round(lon, 4),
        "lat":      round(lat, 4),
        "property": PROPS,
        "depth":    DEPTHS,
        "value":    ["mean"],
    }
    for attempt in range(3):
        try:
            r = requests.get(API_URL, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                result = {}
                for layer in data.get("properties", {}).get("layers", []):
                    name = layer["name"]
                    vals = []
                    for depth_data in layer.get("depths", []):
                        v = depth_data.get("values", {}).get("mean")
                        if v is not None:
                            vals.append(v)
                    if vals:
                        result[name] = float(np.mean(vals))
                return result
            elif r.status_code == 429:
                time.sleep(5 * (attempt + 1))
        except Exception:
            time.sleep(3)
    return None


def main() -> None:
    centroids = pd.read_csv("/tmp/county_centroids.csv", dtype={"fips": str})
    centroids["fips"] = centroids["fips"].str.zfill(5)
    print(f"Fetching soil properties for {len(centroids)} counties...")

    rows = []
    for i, row in enumerate(centroids.itertuples()):
        soil = fetch_soil(row.lat, row.lon)
        if soil is None:
            print(f"  [{i+1}/{len(centroids)}] {row.fips} FAILED")
            # Fill with median estimates for corn belt
            soil = {"bdod": 130, "cec": 200, "clay": 250,
                    "sand": 400, "soc": 15, "phh2o": 65}
        rows.append({"fips": row.fips, **soil})
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(centroids)}] done", flush=True)
        time.sleep(0.5)   # SoilGrids is more rate-limited

    df = pd.DataFrame(rows)
    # Rename to clean names + convert units
    df = df.rename(columns={
        "bdod":  "bulk_density",
        "cec":   "cec",
        "clay":  "clay_pct",
        "sand":  "sand_pct",
        "soc":   "soil_organic_carbon",
        "phh2o": "soil_ph",
    })
    path = OUT / "soil_features.parquet"
    df.to_parquet(path, index=False)
    print(f"\n→ {path}  ({len(df):,} counties)")
    print(df.describe().round(1).to_string())


if __name__ == "__main__":
    main()
