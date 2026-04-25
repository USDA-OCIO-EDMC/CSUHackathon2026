"""Aggregate gridMET daily weather to county-day for the 5 target states, 2005-2025.

Variables: pr (precip mm), tmmn/tmmx (K), vpd (kPa), eto (mm), srad (W/m^2).
Derived: GDD50_86 (corn growing degree days, base 50F cap 86F), accumulated precip,
         VPD p90 per month, etc. — computed downstream in features pipeline.

Output: data/raw/gridmet/county_daily.parquet  (one row per county-day-variable, long format)

Uses Earth Engine reduceRegions for speed; one ee.Image per day per variable, batched by year.
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
OUT = ROOT / "data" / "raw" / "gridmet"
OUT.mkdir(parents=True, exist_ok=True)

VARS = ["pr", "tmmn", "tmmx", "vpd", "eto", "srad"]
YEARS = range(CFG["project"]["history_start"], CFG["project"]["target_year"] + 1)
STATE_FP = {"IA": "19", "CO": "08", "WI": "55", "MO": "29", "NE": "31"}


def init_ee() -> None:
    proj = os.getenv("EE_PROJECT")
    if not proj:
        sys.exit("EE_PROJECT missing in .env")
    ee.Initialize(project=proj)


def counties() -> ee.FeatureCollection:
    codes = list(STATE_FP.values())
    return (ee.FeatureCollection("TIGER/2018/Counties")
            .filter(ee.Filter.inList("STATEFP", codes)))


def aggregate_year(year: int, fc: ee.FeatureCollection) -> pd.DataFrame:
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)
    coll = (ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
            .filterDate(start, end)
            .select(VARS))

    def reduce_one(img):
        date = img.date().format("YYYY-MM-dd")
        means = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=4000)
        return means.map(lambda f: f.set("date", date))

    fc_out = coll.map(reduce_one).flatten()
    rows = fc_out.select(["GEOID", "STATEFP", "date"] + VARS, retainGeometry=False).getInfo()
    flat = []
    for ft in rows["features"]:
        p = ft["properties"]
        flat.append({"fips": p["GEOID"], "state_fp": p["STATEFP"], "date": p["date"],
                     **{v: p.get(v) for v in VARS}})
    return pd.DataFrame(flat)


def main() -> None:
    init_ee()
    fc = counties()
    frames = []
    for y in tqdm(list(YEARS), desc="gridMET years"):
        try:
            frames.append(aggregate_year(y, fc))
        except Exception as e:
            print(f"  fail {y}: {e}", file=sys.stderr)
    df = pd.concat(frames, ignore_index=True)
    df["state"] = df["state_fp"].map({v: k for k, v in STATE_FP.items()})
    out = OUT / "county_daily.parquet"
    df.to_parquet(out, index=False)
    print(f"→ {out}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
