"""Aggregate gNATSGO-derived soil properties to county means for the 5 target states.

Uses the OpenLandMap / gNATSGO mirror on Earth Engine (CONUS Soils) which exposes
pre-derived rasters of available water capacity, organic matter, clay %, drainage,
slope at 30 m. This avoids wrangling raw SSURGO MUKEY tables.

Output: data/raw/gnatsgo/county_soil.parquet
   columns: fips, state, awc_0_25cm, om_pct, clay_pct, drainage_class, slope_pct
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import ee
import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs" / "project.yaml").read_text())
OUT = ROOT / "data" / "raw" / "gnatsgo"
OUT.mkdir(parents=True, exist_ok=True)

STATE_FP = {"IA": "19", "CO": "08", "WI": "55", "MO": "29", "NE": "31"}


def init_ee() -> None:
    proj = os.getenv("EE_PROJECT")
    if not proj:
        sys.exit("EE_PROJECT missing in .env")
    ee.Initialize(project=proj)


def soil_image() -> ee.Image:
    """Stack the soil property bands we want from gNATSGO on EE."""
    base = ee.Image("projects/sat-io/open-datasets/CSRL_soil_properties/physical/awc_0to25")
    awc = base.rename("awc_0_25cm")
    om = ee.Image("projects/sat-io/open-datasets/CSRL_soil_properties/chemical/om_kg_sq_m").rename("om_kg_m2")
    clay = ee.Image("projects/sat-io/open-datasets/CSRL_soil_properties/physical/clay_0_25").rename("clay_pct")
    drain = ee.Image("projects/sat-io/open-datasets/CSRL_soil_properties/physical/drainage_class").rename("drainage_class")
    slope = ee.Terrain.slope(ee.Image("USGS/3DEP/10m")).rename("slope_pct")
    return ee.Image.cat([awc, om, clay, drain, slope])


def main() -> None:
    init_ee()
    fc = (ee.FeatureCollection("TIGER/2018/Counties")
          .filter(ee.Filter.inList("STATEFP", list(STATE_FP.values()))))
    img = soil_image()
    means = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=90, tileScale=4)
    rows = means.select(["GEOID", "STATEFP", "awc_0_25cm", "om_kg_m2",
                         "clay_pct", "drainage_class", "slope_pct"],
                        retainGeometry=False).getInfo()
    flat = []
    for ft in rows["features"]:
        p = ft["properties"]
        flat.append({"fips": p["GEOID"], "state_fp": p["STATEFP"],
                     "awc_0_25cm": p.get("awc_0_25cm"),
                     "om_kg_m2": p.get("om_kg_m2"),
                     "clay_pct": p.get("clay_pct"),
                     "drainage_class": p.get("drainage_class"),
                     "slope_pct": p.get("slope_pct")})
    df = pd.DataFrame(flat)
    df["state"] = df["state_fp"].map({v: k for k, v in STATE_FP.items()})
    out = OUT / "county_soil.parquet"
    df.to_parquet(out, index=False)
    print(f"→ {out}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
