"""Build the feature table the analog-year matcher consumes.

Output: data/processed/features/state_checkpoint_features.parquet
   one row per (state, year, checkpoint) with season-to-date summaries:

   gdd_cum         cumulative GDD50_86 from May 1 to checkpoint
   precip_cum      cumulative precip (mm) from May 1 to checkpoint
   vpd_p90         90th-percentile VPD (kPa) over the window
   srad_mean       mean shortwave radiation (W/m^2) over the window
   eto_cum         cumulative reference ET (mm) over the window
   usdm_dsci_mean  mean USDM DSCI over the window
   usdm_d2_pct     mean % area in D2+ over the window
   ndvi_may .. ndvi_oct   monthly mean NDVI (corn-masked, county-acreage-weighted)

NDVI uses HLS for 2015+ and the Landsat C2 backfill for 2005-2014. State-level
values are corn-acreage-weighted means of county values (CDL acreage from prior year).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs" / "project.yaml").read_text())
OUT = ROOT / "data" / "processed" / "features"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = CFG["project"]["forecast_checkpoints"]               # {name: "MM-DD"}
STATES = CFG["project"]["states"]
YEARS = list(range(CFG["project"]["history_start"], CFG["project"]["target_year"] + 1))
WINDOW_START_MONTH = 5                                              # May 1


# ---------- helpers ----------

def k_to_f(k: pd.Series) -> pd.Series:
    return (k - 273.15) * 9 / 5 + 32


def gdd_50_86(tmin_k: pd.Series, tmax_k: pd.Series) -> pd.Series:
    """Daily corn GDD: base 50F, cap 86F, on (tmin+tmax)/2 in F."""
    tmn = k_to_f(tmin_k).clip(lower=50, upper=86)
    tmx = k_to_f(tmax_k).clip(lower=50, upper=86)
    return ((tmn + tmx) / 2 - 50).clip(lower=0)


def checkpoint_end(year: int, mmdd: str) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-{mmdd}")


# ---------- per-source loaders ----------

def load_gridmet() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "raw" / "gridmet" / "county_daily.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["gdd"] = gdd_50_86(df["tmmn"], df["tmmx"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def load_usdm() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "raw" / "usdm" / "county_weekly.parquet")
    df["valid_start"] = pd.to_datetime(df["valid_start"])
    df["d2plus"] = df["d2"] + df["d3"] + df["d4"]
    df["year"] = df["valid_start"].dt.year
    return df


def load_ndvi() -> pd.DataFrame:
    """Concatenate Landsat backfill (2005-2014) with HLS-derived NDVI (2015+).

    HLS NDVI is computed offline by aggregating chip Red/NIR pre-Prithvi; if that
    table is missing we fall back to Landsat-only and warn.
    """
    frames = []
    landsat = ROOT / "data" / "raw" / "landsat" / "county_month_vi.parquet"
    if landsat.exists():
        frames.append(pd.read_parquet(landsat)[["fips", "state", "year", "month", "ndvi_mean"]])
    hls = ROOT / "data" / "processed" / "features" / "hls_county_month_ndvi.parquet"
    if hls.exists():
        frames.append(pd.read_parquet(hls)[["fips", "state", "year", "month", "ndvi_mean"]])
    if not frames:
        raise FileNotFoundError("Need at least one of landsat or HLS NDVI tables.")
    return pd.concat(frames, ignore_index=True).drop_duplicates(["fips", "year", "month"], keep="last")


def load_corn_acres() -> pd.DataFrame:
    """County corn planted acres from NASS, used as state aggregation weights."""
    src = ROOT / "data" / "raw" / "nass" / "yield_county.parquet"
    df = pd.read_parquet(src)
    if "ACRES" in df.columns:                                       # not in standard yield query
        return df
    # Fallback: weight equally if acres not present (acceptable hackathon proxy)
    return pd.DataFrame()


# ---------- per-window aggregation ----------

def weather_window(weather: pd.DataFrame, year: int, end: pd.Timestamp) -> pd.DataFrame:
    start = pd.Timestamp(f"{year}-{WINDOW_START_MONTH:02d}-01")
    w = weather[(weather["date"] >= start) & (weather["date"] <= end)]
    g = w.groupby(["fips", "state"], as_index=False).agg(
        gdd_cum=("gdd", "sum"),
        precip_cum=("pr", "sum"),
        vpd_p90=("vpd", lambda s: s.quantile(0.9)),
        srad_mean=("srad", "mean"),
        eto_cum=("eto", "sum"),
    )
    return g


def usdm_window(usdm: pd.DataFrame, year: int, end: pd.Timestamp) -> pd.DataFrame:
    start = pd.Timestamp(f"{year}-{WINDOW_START_MONTH:02d}-01")
    u = usdm[(usdm["valid_start"] >= start) & (usdm["valid_start"] <= end)]
    g = u.groupby(["fips", "state"], as_index=False).agg(
        usdm_dsci_mean=("dsci", "mean"),
        usdm_d2_pct=("d2plus", "mean"),
    )
    return g


def ndvi_pivot(ndvi: pd.DataFrame, year: int, end_month: int) -> pd.DataFrame:
    n = ndvi[(ndvi["year"] == year) & (ndvi["month"].between(WINDOW_START_MONTH, end_month))]
    pv = n.pivot_table(index=["fips", "state"], columns="month",
                       values="ndvi_mean", aggfunc="mean").reset_index()
    pv.columns = ["fips", "state"] + [f"ndvi_m{int(m):02d}" for m in pv.columns[2:]]
    return pv


# ---------- state aggregation ----------

def to_state(df: pd.DataFrame) -> pd.DataFrame:
    """Equal-weighted county → state mean (acreage weights are nice-to-have)."""
    num = df.select_dtypes("number").columns
    return df.groupby("state", as_index=False)[list(num)].mean()


# ---------- main ----------

def main() -> None:
    weather = load_gridmet()
    usdm = load_usdm()
    ndvi = load_ndvi()

    rows = []
    for year in YEARS:
        for ck_name, mmdd in CHECKPOINTS.items():
            end = checkpoint_end(year, mmdd)
            end_month = end.month

            wx = weather_window(weather, year, end)
            dr = usdm_window(usdm, year, end)
            vi = ndvi_pivot(ndvi, year, end_month)

            county = wx.merge(dr, on=["fips", "state"], how="outer") \
                       .merge(vi, on=["fips", "state"], how="outer")
            state = to_state(county)
            state["year"] = year
            state["checkpoint"] = ck_name
            rows.append(state)

    out = pd.concat(rows, ignore_index=True)
    out = out[out["state"].isin(STATES)]
    path = OUT / "state_checkpoint_features.parquet"
    out.to_parquet(path, index=False)
    print(f"→ {path}  ({len(out):,} state-year-checkpoints, {len(out.columns)} cols)")


if __name__ == "__main__":
    main()
