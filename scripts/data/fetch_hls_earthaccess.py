"""Fetch HLS L30/S30 satellite imagery directly from NASA LP DAAC.

No Google Earth Engine needed. Uses the `earthaccess` library against NASA's
Land Processes DAAC (the same data GEE would serve, just accessed at the source).

Signup is faster than GEE: free NASA Earthdata account at
  https://urs.earthdata.nasa.gov/users/new
First run will prompt for credentials and persist them to ~/.netrc.

Output:
  data/raw/hls/scenes/<scene_id>/B0X.tif    individual band GeoTIFFs
  data/raw/hls/manifest.parquet              one row per scene (id, date, cloud, bbox, files)

Tunable via env vars:
  HLS_BBOX        "lon_min,lat_min,lon_max,lat_max"  (default: Iowa corn belt)
  HLS_DATE_START  "YYYY-MM-DD"                       (default: 2024-08-01, peak grain fill)
  HLS_DATE_END    "YYYY-MM-DD"                       (default: 2024-08-31)
  HLS_MAX_SCENES  int                                (default: 10, caps download size)
  HLS_MAX_CLOUD   int 0-100                          (default: 30, percent)
  HLS_PRODUCT     "HLSL30" | "HLSS30" | "both"       (default: both)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import earthaccess
import pandas as pd
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "hls"
SCENES_DIR = OUT / "scenes"
SCENES_DIR.mkdir(parents=True, exist_ok=True)

BBOX = tuple(float(x) for x in os.getenv("HLS_BBOX", "-96.6,40.4,-90.1,43.5").split(","))
DATE_START = os.getenv("HLS_DATE_START", "2024-08-01")
DATE_END = os.getenv("HLS_DATE_END", "2024-08-31")
MAX_SCENES = int(os.getenv("HLS_MAX_SCENES", "10"))
MAX_CLOUD = int(os.getenv("HLS_MAX_CLOUD", "30"))
PRODUCT = os.getenv("HLS_PRODUCT", "both").lower()
DOWNLOAD_THREADS = int(os.getenv("HLS_DOWNLOAD_THREADS", "8"))

PRODUCTS = ["HLSL30", "HLSS30"] if PRODUCT == "both" else [PRODUCT.upper()]

# Only the 6 model bands + cloud mask — skips thermal, view-angle, SAA/SZA files
NEEDED_BANDS = {"B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12", "Fmask"}


def band_links(granules: list) -> list[str]:
    """Return only the download URLs for the bands we actually need."""
    urls = []
    for g in granules:
        for link in g.data_links():
            # HLS filenames end in .<BAND>.tif  e.g.  ...v2.0.B02.tif
            stem = link.rsplit(".", 2)[-2]  # second-to-last extension
            if stem in NEEDED_BANDS:
                urls.append(link)
    return urls


def search(short_name: str) -> list:
    print(f"  searching {short_name} v2.0  bbox={BBOX}  {DATE_START}..{DATE_END}  cloud<{MAX_CLOUD}%")
    return earthaccess.search_data(
        short_name=short_name,
        version="2.0",
        bounding_box=BBOX,
        temporal=(DATE_START, DATE_END),
        cloud_cover=(0, MAX_CLOUD),
        count=MAX_SCENES,
    )


def main() -> None:
    print("NASA Earthdata login (using EARTHDATA_USERNAME / EARTHDATA_PASSWORD from .env)")
    auth = earthaccess.login(strategy="environment", persist=True)
    if not auth.authenticated:
        sys.exit("Earthdata login failed — check EARTHDATA_USERNAME/PASSWORD in .env")

    all_results = []
    for sn in PRODUCTS:
        rs = search(sn)
        print(f"  found {len(rs)} {sn} scenes")
        all_results.extend(rs)

    if not all_results:
        sys.exit("No scenes matched. Widen date range, bbox, or cloud cover threshold.")

    urls = band_links(all_results)
    print(f"\nDownloading {len(urls)} band files from {len(all_results)} scenes "
          f"-> {SCENES_DIR}  (threads={DOWNLOAD_THREADS})")
    files = earthaccess.download(urls, str(SCENES_DIR), threads=DOWNLOAD_THREADS)
    print(f"  wrote {len(files)} files")

    rows = []
    for r in all_results:
        try:
            umm = r["umm"]
            scene_id = umm["GranuleUR"]
            date = umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
            cloud = next((a["Values"][0] for a in umm.get("AdditionalAttributes", [])
                          if a.get("Name") == "CLOUD_COVERAGE"), None)
            rows.append({
                "scene_id": scene_id,
                "date": date,
                "cloud_cover_pct": float(cloud) if cloud is not None else None,
                "product": "HLSL30" if "HLS.L30" in scene_id else "HLSS30",
            })
        except (KeyError, TypeError, ValueError) as e:
            print(f"  skip manifest row: {e}")

    manifest = pd.DataFrame(rows)
    manifest.to_parquet(OUT / "manifest.parquet", index=False)
    print(f"\nManifest -> {OUT / 'manifest.parquet'}  ({len(manifest)} rows)")
    print(f"Total disk usage:")
    os.system(f"du -sh {OUT}")


if __name__ == "__main__":
    main()
