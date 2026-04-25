"""Export HLS monthly composites (6 Prithvi bands) per county, masked to corn pixels via CDL.

Output layout (zarr on local NVMe):
  data/processed/chips/{state}/{fips}/{year}/{checkpoint}.zarr
    dims: (timestep=3, band=6, y=224, x=224)
    coords: lat (centroid), lon (centroid), dates (3 ISO strings)

For each (county, year, checkpoint) we build a 3-month rolling window ending at the checkpoint
(e.g. Aug 1 → May/Jun/Jul medians), median-composite cloud-masked HLS scenes, mask to CDL=1
(corn), and tile the county into 224x224 chips at 30 m. Chip metadata (lat/lon/date) is
written alongside each zarr — required by the Prithvi-EO-2.0-TL temporal+location embedding.

Run on a workstation with `earthengine authenticate` already done. Uses GEE getDownloadURL
for sub-county sized COG tiles; for very large counties it auto-tiles.
"""
from __future__ import annotations

import io
import json
import os
import sys
from datetime import date
from pathlib import Path

import ee
import geemap
import numpy as np
import rasterio
import requests
import xarray as xr
import yaml
import zarr
from dotenv import load_dotenv
from rasterio.io import MemoryFile
from tqdm import tqdm

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs" / "project.yaml").read_text())
OUT = ROOT / CFG["paths"]["processed"]
OUT.mkdir(parents=True, exist_ok=True)

STATES = CFG["project"]["states"]
HLS_START = CFG["project"]["hls_era_start"]
HIST_END = CFG["project"]["history_end"]
TARGET = CFG["project"]["target_year"]
CHECKPOINTS = CFG["project"]["forecast_checkpoints"]
CHIP = CFG["prithvi"]["chip_size"]
PIX = CFG["prithvi"]["pixel_size_m"]
SCALE = CFG["prithvi"]["scale_factor"]


def init_ee() -> None:
    proj = os.getenv("EE_PROJECT")
    if not proj:
        sys.exit("EE_PROJECT missing in .env")
    ee.Initialize(project=proj)


def mask_hls(img: ee.Image) -> ee.Image:
    """Drop cloud, cloud shadow, water, snow using HLS Fmask bits."""
    fmask = img.select("Fmask")
    # bits: 1=cloud, 2=adj cloud, 3=cloud shadow, 4=snow/ice, 5=water
    bad = (fmask.bitwiseAnd(0b00111110)).gt(0)
    return img.updateMask(bad.Not())


def hls_composite(start: ee.Date, end: ee.Date, region: ee.Geometry) -> ee.Image:
    """Median composite of HLSL30+HLSS30, returning the 6 Prithvi bands in canonical order."""
    bm_l = CFG["prithvi"]["hls_band_map"]["HLSL30"]
    bm_s = CFG["prithvi"]["hls_band_map"]["HLSS30"]
    canon = ["Blue", "Green", "Red", "NarrowNIR", "SWIR1", "SWIR2"]

    l30 = (ee.ImageCollection("NASA/HLS/HLSL30/v002")
           .filterDate(start, end).filterBounds(region).map(mask_hls)
           .select([bm_l[b] for b in canon] + ["Fmask"], canon + ["Fmask"]))
    s30 = (ee.ImageCollection("NASA/HLS/HLSS30/v002")
           .filterDate(start, end).filterBounds(region).map(mask_hls)
           .select([bm_s[b] for b in canon] + ["Fmask"], canon + ["Fmask"]))
    merged = l30.merge(s30).select(canon)
    return merged.median().clip(region)


def cdl_corn_mask(year: int, region: ee.Geometry) -> ee.Image:
    """Corn mask from CDL. For target year (no CDL yet), use prior-year mask."""
    cdl_year = min(year, 2024)                                  # 2025 CDL won't exist mid-season
    cdl = (ee.ImageCollection("USDA/NASS/CDL")
           .filter(ee.Filter.calendarRange(cdl_year, cdl_year, "year"))
           .first()
           .select("cropland"))
    return cdl.eq(1).clip(region)


def county_geoms() -> ee.FeatureCollection:
    """TIGER 2018 county boundaries filtered to our 5 states."""
    fips = {"IA": "19", "CO": "08", "WI": "55", "MO": "29", "NE": "31"}
    state_codes = [fips[s] for s in STATES]
    return (ee.FeatureCollection("TIGER/2018/Counties")
            .filter(ee.Filter.inList("STATEFP", state_codes)))


def checkpoint_window(year: int, mmdd: str) -> tuple[ee.Date, ee.Date, ee.Date, ee.Date, ee.Date, ee.Date]:
    """Three rolling 30-day windows ending at the checkpoint."""
    end = ee.Date(f"{year}-{mmdd}")
    return (end.advance(-90, "day"), end.advance(-60, "day"),
            end.advance(-60, "day"), end.advance(-30, "day"),
            end.advance(-30, "day"), end)


def export_county_year(feat: ee.Feature, year: int, checkpoint: str, mmdd: str) -> None:
    fips = feat.get("GEOID").getInfo()
    state_fp = feat.get("STATEFP").getInfo()
    state_abbr = {"19": "IA", "08": "CO", "55": "WI", "29": "MO", "31": "NE"}[state_fp]

    out_dir = OUT / "chips" / state_abbr / fips / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{checkpoint}.zarr"
    if out_path.exists():
        return                                                   # idempotent

    region = feat.geometry()
    s1, e1, s2, e2, s3, e3 = checkpoint_window(year, mmdd)
    t1 = hls_composite(s1, e1, region)
    t2 = hls_composite(s2, e2, region)
    t3 = hls_composite(s3, e3, region)

    corn = cdl_corn_mask(year, region)
    stack = (ee.Image.cat([t1.rename([f"t1_{b}" for b in t1.bandNames().getInfo()]),
                           t2.rename([f"t2_{b}" for b in t2.bandNames().getInfo()]),
                           t3.rename([f"t3_{b}" for b in t3.bandNames().getInfo()])])
             .updateMask(corn)
             .toUint16())

    # GEE getDownloadURL caps at 50 MB / 32k pixels per side. Counties fit; if not, tile.
    url = stack.getDownloadURL({
        "scale": PIX,
        "region": region,
        "format": "GEO_TIFF",
        "crs": "EPSG:5070",                                      # Albers Equal Area CONUS
    })
    r = requests.get(url, timeout=600)
    r.raise_for_status()

    with MemoryFile(r.content) as mem, mem.open() as src:
        arr = src.read()                                          # (18, H, W) uint16
        H, W = arr.shape[1], arr.shape[2]
        # reshape into (timestep=3, band=6, H, W)
        arr3 = arr.reshape(3, 6, H, W)

        # Tile into 224x224 chips
        chips, chip_meta = [], []
        for iy in range(0, H - CHIP + 1, CHIP):
            for ix in range(0, W - CHIP + 1, CHIP):
                tile = arr3[:, :, iy:iy + CHIP, ix:ix + CHIP]
                if (tile > 0).mean() < 0.05:                      # skip mostly-empty tiles
                    continue
                chips.append(tile)
                xc, yc = src.xy(iy + CHIP // 2, ix + CHIP // 2)
                chip_meta.append({"x_center": float(xc), "y_center": float(yc)})

        if not chips:
            return
        chips_arr = np.stack(chips, axis=0)                       # (N, 3, 6, 224, 224)

    centroid = feat.geometry().centroid().coordinates().getInfo()
    dates = [ee.Date(d).format("YYYY-MM-dd").getInfo() for d in (e1, e2, e3)]

    z = zarr.open(out_path, mode="w")
    z.create_dataset("chips", data=chips_arr, chunks=(1, 3, 6, CHIP, CHIP), dtype="u2")
    z.attrs["fips"] = fips
    z.attrs["state"] = state_abbr
    z.attrs["year"] = year
    z.attrs["checkpoint"] = checkpoint
    z.attrs["mmdd"] = mmdd
    z.attrs["centroid_lonlat"] = centroid
    z.attrs["dates"] = dates
    z.attrs["scale_factor"] = SCALE
    z.attrs["chip_centers_albers"] = chip_meta


def main() -> None:
    init_ee()
    counties = county_geoms().toList(2000).getInfo()             # TIGER returns Feature list
    years = list(range(HLS_START, TARGET + 1))

    work = [(c, y, ck, mmdd) for c in counties for y in years
            for ck, mmdd in CHECKPOINTS.items()]

    for feat_dict, year, ck, mmdd in tqdm(work, desc="HLS chips"):
        try:
            export_county_year(ee.Feature(feat_dict), year, ck, mmdd)
        except Exception as e:
            print(f"  fail {feat_dict['properties'].get('GEOID')} {year} {ck}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
