"""
Phase 1.3 — HLS (Harmonized Landsat Sentinel-2) Cloud-Direct Access

Reads HLS tiles directly from NASA's S3 via earthaccess open() —
no local download or re-upload needed. Granule data is streamed
directly into rasterio/numpy for feature extraction.

Required env vars:
    EARTHDATA_USERNAME   — NASA EarthData username
    EARTHDATA_PASSWORD   — NASA EarthData password

Install:
    pip install earthaccess rasterio numpy boto3
"""

import os
from pathlib import Path

import numpy as np
import rasterio
import earthaccess

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# 6 bands Prithvi-100M was trained on (HLS band names)
HLS_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07"]

# Approximate bounding boxes (min_lon, min_lat, max_lon, max_lat) per state
STATE_BBOX = {
    "IA": (-96.64, 40.38, -90.14, 43.50),
    "CO": (-109.06, 36.99, -102.04, 41.00),
    "WI": (-92.89, 42.49, -86.25, 47.08),
    "MO": (-95.77, 35.99, -89.10, 40.61),
    "NE": (-104.05, 39.99, -95.31, 43.00),
}

# Time windows for each forecast date (inclusive)
FORECAST_WINDOWS = {
    "aug1":  ("05-01", "07-31"),
    "sep1":  ("05-01", "08-31"),
    "oct1":  ("05-01", "09-30"),
    "final": ("05-01", "10-31"),
}

# HLS product short names — use both Landsat and Sentinel-2 for better coverage
HLS_PRODUCTS = [
    ("HLSL30", "2.0"),  # Landsat 30 m
    ("HLSS30", "2.0"),  # Sentinel-2 30 m
]


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def login():
    """Login to NASA EarthData using env vars EARTHDATA_USERNAME / EARTHDATA_PASSWORD."""
    user = os.environ.get("EARTHDATA_USERNAME", "")
    pwd = os.environ.get("EARTHDATA_PASSWORD", "")
    if not user or not pwd:
        raise EnvironmentError(
            "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables.\n"
            "Register free at https://urs.earthdata.nasa.gov/"
        )
    earthaccess.login(strategy="environment")
    print("EarthData login successful.")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_hls(state_abbr, year, forecast_date="final"):
    """
    Search CMR for HLS granules (Landsat + Sentinel-2) covering a state
    for a given year/forecast window.

    Returns a list of earthaccess DataGranule objects.
    """
    bbox = STATE_BBOX[state_abbr]
    start_mmdd, end_mmdd = FORECAST_WINDOWS[forecast_date]
    temporal = (f"{year}-{start_mmdd}", f"{year}-{end_mmdd}")

    results = []
    for short_name, version in HLS_PRODUCTS:
        hits = earthaccess.search_data(
            short_name=short_name,
            version=version,
            temporal=temporal,
            bounding_box=bbox,
        )
        results.extend(hits)

    print(f"  Found {len(results)} granules for {state_abbr} {year} [{forecast_date}]")
    return results


# ---------------------------------------------------------------------------
# Cloud-direct band stacking (no download, no re-upload)
# ---------------------------------------------------------------------------


def stack_granule_cloud(granule):
    """
    Stream a single HLS granule's 6 bands directly from NASA S3
    without downloading to disk. Uses earthaccess.open() which
    handles S3 credential negotiation automatically.

    Returns
    -------
    stacked : np.ndarray shape (6, H, W), float32
    profile : rasterio profile dict
    granule_id : str
    """
    granule_id = granule["umm"]["GranuleUR"]

    # earthaccess.open() returns file-like objects for every asset in the granule
    file_objs = earthaccess.open([granule])

    # Match file objects to the 6 required bands by filename
    band_files = {}
    for fobj in file_objs:
        name = Path(fobj.path).name if hasattr(fobj, "path") else str(fobj)
        for b in HLS_BANDS:
            if f".{b}." in name:
                band_files[b] = fobj
                break

    missing = [b for b in HLS_BANDS if b not in band_files]
    if missing:
        raise ValueError(f"Granule {granule_id} missing bands: {missing}")

    arrays = []
    profile = None
    for band in HLS_BANDS:
        with rasterio.open(band_files[band]) as src:
            arrays.append(src.read(1).astype(np.float32))
            if profile is None:
                profile = src.profile.copy()

    stacked = np.stack(arrays, axis=0)  # (6, H, W)
    profile.update(count=6, dtype=rasterio.float32)
    return stacked, profile, granule_id


def iter_granules_cloud(state_abbr, year, forecast_date="final", max_granules=None):
    """
    Iterator that yields (stacked_array, profile, granule_id) for each
    HLS granule covering a state/year/forecast_date, reading directly
    from NASA S3 — no local storage required.

    Parameters
    ----------
    state_abbr    : e.g. "IA"
    year          : int, e.g. 2023
    forecast_date : "aug1" | "sep1" | "oct1" | "final"
    max_granules  : limit number of granules (useful for testing)

    Example
    -------
    for stacked, profile, gid in iter_granules_cloud("IA", 2023, "final", max_granules=5):
        features = extract_features_from_array(model, stacked)
    """
    granules = search_hls(state_abbr, year, forecast_date)
    if max_granules:
        granules = granules[:max_granules]

    for granule in granules:
        try:
            stacked, profile, gid = stack_granule_cloud(granule)
            yield stacked, profile, gid
        except Exception as e:
            gid = granule["umm"].get("GranuleUR", "unknown")
            print(f"  WARN skipping {gid}: {e}")


