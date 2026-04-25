"""Compute HLS county-month mean NDVI (corn-masked) for 2015-2025.

Output: data/processed/features/hls_county_month_ndvi.parquet
   columns: fips, state, year, month, ndvi_mean, ndvi_p10, ndvi_p90, n_obs

Used by the analog-year matcher (build_features.py). This is a lightweight,
non-Prithvi NDVI signal — purely for analog matching, not for the model input.
HLS NIR is the Narrow NIR band (B05 on L30, B8A on S30), matching what we feed
Prithvi for consistency.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import ee
import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs" / "project.yaml").read_text())
OUT = ROOT / "data" / "processed" / "features"
OUT.mkdir(parents=True, exist_ok=True)

STATE_FP = {"IA": "19", "CO": "08", "WI": "55", "MO": "29", "NE": "31"}

# Default to full 2015–target range. Override with HLS_YEARS env var (comma-separated)
# to fetch a narrow slice — e.g. HLS_YEARS=2024,2025 trims wall time from ~1 hr to ~12 min.
_env_years = os.getenv("HLS_YEARS", "").strip()
if _env_years:
    YEARS = [int(y) for y in _env_years.split(",") if y.strip()]
    print(f"HLS_YEARS override: {YEARS}")
else:
    YEARS = list(range(CFG["project"]["hls_era_start"], CFG["project"]["target_year"] + 1))


def init_ee() -> None:
    proj = os.getenv("EE_PROJECT")
    if not proj:
        sys.exit("EE_PROJECT missing in .env")
    ee.Initialize(project=proj)


def mask_hls(img: ee.Image) -> ee.Image:
    fmask = img.select("Fmask")
    bad = fmask.bitwiseAnd(0b00111110).gt(0)
    return img.updateMask(bad.Not())


def add_ndvi_l30(img: ee.Image) -> ee.Image:
    s = mask_hls(img)
    red, nir = s.select("B04"), s.select("B05")                     # L30 narrow NIR
    return img.addBands(nir.subtract(red).divide(nir.add(red)).rename("NDVI"))


def add_ndvi_s30(img: ee.Image) -> ee.Image:
    s = mask_hls(img)
    red, nir = s.select("B04"), s.select("B8A")
    return img.addBands(nir.subtract(red).divide(nir.add(red)).rename("NDVI"))


def counties() -> ee.FeatureCollection:
    return (ee.FeatureCollection("TIGER/2018/Counties")
            .filter(ee.Filter.inList("STATEFP", list(STATE_FP.values()))))


def aggregate(year: int, month: int, fc: ee.FeatureCollection) -> pd.DataFrame:
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")

    l30 = (ee.ImageCollection("NASA/HLS/HLSL30/v002")
           .filterDate(start, end).map(add_ndvi_l30).select("NDVI"))
    s30 = (ee.ImageCollection("NASA/HLS/HLSS30/v002")
           .filterDate(start, end).map(add_ndvi_s30).select("NDVI"))
    coll = l30.merge(s30)

    cdl_year = min(year, 2024)
    cdl = (ee.ImageCollection("USDA/NASS/CDL")
           .filter(ee.Filter.calendarRange(cdl_year, cdl_year, "year"))
           .first().select("cropland"))
    corn = cdl.eq(1)

    composite = coll.median().updateMask(corn)
    means = composite.reduceRegions(
        collection=fc, reducer=ee.Reducer.mean().combine(ee.Reducer.percentile([10, 90]), sharedInputs=True),
        scale=30, tileScale=4,
    )
    rows = means.select(["GEOID", "STATEFP", "NDVI_mean", "NDVI_p10", "NDVI_p90"],
                        retainGeometry=False).getInfo()
    out = []
    for ft in rows["features"]:
        p = ft["properties"]
        out.append({
            "fips": p["GEOID"], "state_fp": p["STATEFP"],
            "year": year, "month": month,
            "ndvi_mean": p.get("NDVI_mean"),
            "ndvi_p10": p.get("NDVI_p10"),
            "ndvi_p90": p.get("NDVI_p90"),
        })
    return pd.DataFrame(out)


def main() -> None:
    init_ee()
    fc = counties()
    frames = []
    months = list(range(4, 11))                                     # Apr..Oct
    for y in tqdm(YEARS, desc="HLS NDVI years"):
        for m in months:
            try:
                frames.append(aggregate(y, m, fc))
            except Exception as e:
                print(f"  fail {y}-{m}: {e}", file=sys.stderr)
    df = pd.concat(frames, ignore_index=True)
    df["state"] = df["state_fp"].map({v: k for k, v in STATE_FP.items()})
    out = OUT / "hls_county_month_ndvi.parquet"
    df.to_parquet(out, index=False)
    print(f"→ {out}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
