import os
import io
import xml.etree.ElementTree as ET
import requests
import pandas as pd
import numpy as np
import boto3
import rasterio
from rasterio.io import MemoryFile
import os

NASS_KEY = os.environ.get("NASS_API_KEY", "")  # Register at quickstats.nass.usda.gov
STATES = {"IA":"19","CO":"08","WI":"55","MO":"29","NE":"31"}

def get_nass_yields(state_fips, year_start=2010, year_end=2024):
    params = {
        "key": NASS_KEY,
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "commodity_desc": "CORN",
        "statisticcat_desc": "YIELD",
        "unit_desc": "BU / ACRE",
        "agg_level_desc": "COUNTY",
        "state_fips_code": state_fips,
        "year__GE": year_start,
        "year__LE": year_end,
        "format": "JSON"
    }
    r = requests.get("https://quickstats.nass.usda.gov/api/api_GET/", params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return pd.DataFrame(columns=["year", "county_name", "yield_bu_acre"])
    return pd.DataFrame(data)[["year","county_name","Value"]] \
             .rename(columns={"Value":"yield_bu_acre"}) \
             .assign(yield_bu_acre=lambda d: pd.to_numeric(d.yield_bu_acre, errors="coerce"))

def save_yields_to_s3(df, bucket, state_fips):
    s3 = boto3.client('s3')
    key = f"processed/yields/state_{state_fips}.csv"
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

def load_yields_from_s3(bucket, state_fips):
    s3 = boto3.client('s3')
    key = f"processed/yields/state_{state_fips}.csv"
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))


def fetch_all_states(bucket, year_start=2005, year_end=2024):
    """
    Fetch NASS county-level corn yields for all 5 target states,
    cache each to S3, and return a combined DataFrame with a 'state' column.
    """
    all_dfs = []
    for abbr, fips in STATES.items():
        print(f"Fetching {abbr} (FIPS {fips})...")
        df = get_nass_yields(fips, year_start=year_start, year_end=year_end)
        if df.empty:
            print(f"  WARNING: no data returned for {abbr}")
            continue
        df["state"] = abbr
        df["state_fips"] = fips
        save_yields_to_s3(df, bucket, fips)
        print(f"  {len(df)} rows saved to s3://{bucket}/processed/yields/state_{fips}.csv")
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def load_all_states(bucket):
    """Load all 5 states' cached yield CSVs from S3 into a combined DataFrame."""
    all_dfs = []
    for abbr, fips in STATES.items():
        df = load_yields_from_s3(bucket, fips)
        df["state"] = abbr
        df["state_fips"] = fips
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Phase 1.2 — Cropland Data Layer (CDL)
# ---------------------------------------------------------------------------

CDL_API = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"
CORN_CLASS = 1  # CDL class 1 = Corn


def _get_cdl_download_url(state_fips, year):
    """Call CropScape API and return the GeoTIFF download URL for a state/year."""
    r = requests.get(CDL_API, params={"year": year, "fips": state_fips}, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    # The response XML contains a returnURL element (namespace-agnostic search)
    url_el = root.find(".//{*}returnURL") or root.find(".//returnURL")
    if url_el is None or not url_el.text:
        raise ValueError(f"No returnURL in CDL response for FIPS {state_fips}, year {year}")
    return url_el.text.strip()


def download_cdl_state(state_fips, year, bucket):
    """
    Download the CDL GeoTIFF for one state/year from CropScape,
    mask to corn pixels only (class 1), and upload to S3.

    S3 key: raw/cdl/{state_fips}/{year}.tif
    """
    print(f"  Fetching CDL URL for FIPS {state_fips}, {year}...")
    tif_url = _get_cdl_download_url(state_fips, year)

    print(f"  Downloading: {tif_url}")
    tif_r = requests.get(tif_url, timeout=300, stream=True)
    tif_r.raise_for_status()
    raw_bytes = tif_r.content

    # Mask to corn pixels (class 1 = corn)
    with MemoryFile(raw_bytes) as memfile:
        with memfile.open() as src:
            data = src.read(1)
            profile = src.profile.copy()

    corn_mask = (data == CORN_CLASS).astype(np.uint8)
    profile.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=0)

    out_buf = io.BytesIO()
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(corn_mask, 1)
        out_buf.write(memfile.read())
    out_buf.seek(0)

    s3 = boto3.client("s3")
    key = f"raw/cdl/{state_fips}/{year}.tif"
    s3.put_object(Bucket=bucket, Key=key, Body=out_buf.read())
    corn_pixels = int(corn_mask.sum())
    print(f"  Uploaded to s3://{bucket}/{key}  ({corn_pixels:,} corn pixels)")
    return corn_pixels


def fetch_all_cdl(bucket, year_start=2005, year_end=2024):
    """
    Download and cache CDL corn masks for all 5 states across all years.
    Skips files already present in S3.
    """
    s3 = boto3.client("s3")
    for abbr, fips in STATES.items():
        for year in range(year_start, year_end + 1):
            key = f"raw/cdl/{fips}/{year}.tif"
            # Skip if already uploaded
            try:
                s3.head_object(Bucket=bucket, Key=key)
                print(f"  Skipping {abbr} {year} (already in S3)")
                continue
            except s3.exceptions.ClientError:
                pass
            try:
                download_cdl_state(fips, year, bucket)
            except Exception as e:
                print(f"  ERROR {abbr} {year}: {e}")


def load_cdl_from_s3(bucket, state_fips, year):
    """Load a corn mask GeoTIFF from S3. Returns (numpy array, rasterio profile)."""
    s3 = boto3.client("s3")
    key = f"raw/cdl/{state_fips}/{year}.tif"
    obj = s3.get_object(Bucket=bucket, Key=key)
    with MemoryFile(obj["Body"].read()) as memfile:
        with memfile.open() as src:
            return src.read(1), src.profile.copy()