"""Extract growing-season drought features from USDM county_weekly.parquet.

Output: data/processed/features/drought_features.parquet
Columns: fips, year, dsci_mean, d2plus_pct, d3plus_pct, dsci_peak
Covers May 1 – Aug 31 (weeks relevant to corn pollination / grain fill).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "data" / "processed" / "features"
OUT.mkdir(parents=True, exist_ok=True)

YEARS = [2022, 2023, 2024, 2025]


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "raw" / "usdm" / "county_weekly.parquet")
    df["date"] = pd.to_datetime(df["valid_start"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Growing season: May (5) through August (8)
    gs = df[(df["month"] >= 5) & (df["month"] <= 8) & (df["year"].isin(YEARS))].copy()
    gs["d2plus"] = gs["d2"] + gs["d3"] + gs["d4"]
    gs["d3plus"] = gs["d3"] + gs["d4"]

    feats = gs.groupby(["fips", "year"]).agg(
        dsci_mean  = ("dsci",   "mean"),
        dsci_peak  = ("dsci",   "max"),
        d2plus_pct = ("d2plus", "mean"),
        d3plus_pct = ("d3plus", "mean"),
    ).reset_index()

    feats["fips"] = feats["fips"].astype(str).str.zfill(5)
    path = OUT / "drought_features.parquet"
    feats.to_parquet(path, index=False)
    print(f"→ {path}  ({len(feats):,} county-years)")
    print(feats.groupby("year").size().to_string())


if __name__ == "__main__":
    main()
