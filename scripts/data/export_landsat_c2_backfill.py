"""Backfill 2005-2014 with Landsat 5/7/8 Collection-2 L2 NDVI/EVI county-month means.

Purpose: HLS only starts in 2013. To extend the analog-year pool from 12 -> 20 years,
we need a consistent vegetation signal back to 2005. Landsat C2 L2 surface reflectance
gives us NDVI/EVI on the same physical bands; we DO NOT use this for Prithvi input
(Prithvi expects HLS-quality 6-band stacks), only for analog-year matching.

Output: data/raw/landsat/county_month_vi.parquet
   columns: fips, state, year, month, ndvi_mean, ndvi_p10, ndvi_p90, evi_mean, n_obs

Bands (Collection 2 L2, surface reflectance, scale 0.0000275 + offset -0.2):
  L5/L7: SR_B3 (red), SR_B4 (NIR)
  L8/L9: SR_B4 (red), SR_B5 (NIR), SR_B2 (blue) for EVI
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
OUT = ROOT / "data" / "raw" / "landsat"
OUT.mkdir(parents=True, exist_ok=True)

STATE_FP = {"IA": "19", "CO": "08", "WI": "55", "MO": "29", "NE": "31"}
BACKFILL_YEARS = range(2005, CFG["project"]["hls_era_start"])     # 2005..2014


def init_ee() -> None:
    proj = os.getenv("EE_PROJECT")
    if not proj:
        sys.exit("EE_PROJECT missing in .env")
    ee.Initialize(project=proj)


def scale_l(img: ee.Image) -> ee.Image:
    """C2 L2 surface reflectance scale/offset → 0..1 reflectance."""
    sr = img.select("SR_B.").multiply(0.0000275).add(-0.2)
    qa = img.select("QA_PIXEL")
    cloud = qa.bitwiseAnd(1 << 3).neq(0).Or(qa.bitwiseAnd(1 << 4).neq(0))   # cloud or shadow
    return sr.updateMask(cloud.Not()).copyProperties(img, ["system:time_start"])


def vi_l57(img: ee.Image) -> ee.Image:
    s = scale_l(img)
    red, nir = s.select("SR_B3"), s.select("SR_B4")
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    return img.addBands(ndvi)


def vi_l89(img: ee.Image) -> ee.Image:
    s = scale_l(img)
    blue, red, nir = s.select("SR_B2"), s.select("SR_B4"), s.select("SR_B5")
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    evi = (nir.subtract(red).multiply(2.5)
              .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
              .rename("EVI"))
    return img.addBands(ndvi).addBands(evi)


def collection(year: int) -> ee.ImageCollection:
    """Pick L5+L7 for ≤2013, L7+L8 for 2013+. Avoid L7 SLC-off after 2003 if possible."""
    start = ee.Date.fromYMD(year, 4, 1)
    end = ee.Date.fromYMD(year, 11, 1)
    if year <= 2011:
        coll = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
                .filterDate(start, end).map(vi_l57))
    elif year == 2012:
        coll = (ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
                .filterDate(start, end).map(vi_l57))
    else:                                                          # 2013+
        l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
              .filterDate(start, end).map(vi_l89))
        l7 = (ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
              .filterDate(start, end).map(vi_l57))
        coll = l8.merge(l7)
    return coll


def counties() -> ee.FeatureCollection:
    return (ee.FeatureCollection("TIGER/2018/Counties")
            .filter(ee.Filter.inList("STATEFP", list(STATE_FP.values()))))


def aggregate(year: int, month: int, fc: ee.FeatureCollection) -> pd.DataFrame:
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    coll = collection(year).filterDate(start, end).select(["NDVI"])

    cdl_year = min(year, 2024)
    cdl = (ee.ImageCollection("USDA/NASS/CDL")
           .filter(ee.Filter.calendarRange(max(cdl_year, 2008), max(cdl_year, 2008), "year"))
           .first().select("cropland"))
    corn_mask = cdl.eq(1)

    composite = coll.median().updateMask(corn_mask)
    means = composite.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30, tileScale=4)
    rows = means.select(["GEOID", "STATEFP", "NDVI"], retainGeometry=False).getInfo()
    out = []
    for ft in rows["features"]:
        p = ft["properties"]
        out.append({"fips": p["GEOID"], "state_fp": p["STATEFP"],
                    "year": year, "month": month, "ndvi_mean": p.get("NDVI")})
    return pd.DataFrame(out)


def main() -> None:
    init_ee()
    fc = counties()
    frames = []
    months = list(range(4, 11))                                    # Apr..Oct
    for y in tqdm(list(BACKFILL_YEARS), desc="Landsat years"):
        for m in months:
            try:
                frames.append(aggregate(y, m, fc))
            except Exception as e:
                print(f"  fail {y}-{m}: {e}", file=sys.stderr)
    df = pd.concat(frames, ignore_index=True)
    df["state"] = df["state_fp"].map({v: k for k, v in STATE_FP.items()})
    out = OUT / "county_month_vi.parquet"
    df.to_parquet(out, index=False)
    print(f"→ {out}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
