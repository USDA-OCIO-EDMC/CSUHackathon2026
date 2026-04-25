"""On-demand data API: query each source over the network, return in-memory.

Replaces the bulk-download flow (`scripts/download_all.py` → parquet/zarr on disk)
with thin REST/STAC clients that fetch only the slice you ask for. No Google
Earth Engine, no overnight HLS export — just a function call.

Sources & endpoints
    nass_yield / nass_crop_progress / nass_in_season_forecast
        → https://quickstats.nass.usda.gov/api  (needs NASS_API_KEY)
    usdm_county_weekly
        → https://usdmdataservices.unl.edu/api  (no auth)
    cdl_corn_mask
        → https://nassgeodata.gmu.edu/CropScape REST WMS  (no auth)
    weather_point   (gridMET-equivalent)
        → https://power.larc.nasa.gov/api/temporal/daily/point  (no auth)
    soil_point      (gNATSGO-equivalent)
        → https://rest.isric.org/soilgrids/v2.0  (no auth)
    hls_search / hls_open_band
        → https://cmr.earthdata.nasa.gov/stac  (search is no-auth; data download
          needs Earthdata netrc, but COG byte-range reads via rioxarray work fine)
    landsat_search
        → https://planetarycomputer.microsoft.com/api/stac/v1  (no auth, signed
          URLs returned)

Quick start
    from scripts.data_api import nass_yield, usdm_county_weekly, weather_point
    df = nass_yield("IOWA", 2020, 2024)
    drought = usdm_county_weekly("IA", "2024-04-01", "2024-10-31")
    wx = weather_point(42.03, -93.62, "2024-04-01", "2024-10-31",
                       params=("T2M","PRECTOTCORR","ALLSKY_SFC_SW_DWN"))
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

STATE_FP = {"IA": "19", "CO": "08", "WI": "55", "MO": "29", "NE": "31"}
STATE_NAME = {"IA": "IOWA", "CO": "COLORADO", "WI": "WISCONSIN",
              "MO": "MISSOURI", "NE": "NEBRASKA"}


# ───────────────────────── NASS QuickStats ─────────────────────────

_NASS_API = "https://quickstats.nass.usda.gov/api/api_GET/"


def _nass(params: dict) -> list[dict]:
    key = os.getenv("NASS_API_KEY")
    if not key:
        raise RuntimeError("NASS_API_KEY missing — see https://quickstats.nass.usda.gov/api")
    r = requests.get(_NASS_API, params={"key": key, "format": "JSON", **params}, timeout=120)
    if r.status_code == 413:
        raise RuntimeError(f"NASS row-cap hit, narrow your query: {params}")
    r.raise_for_status()
    return r.json().get("data", [])


def _nass_to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "Value" in df:
        df["Value"] = pd.to_numeric(df["Value"].str.replace(",", ""), errors="coerce")
        df = df[df["Value"].notna()]
    if "year" in df:
        df["year"] = df["year"].astype(int)
    return df


def nass_yield(state: str, year_start: int = 2005, year_end: int = 2024,
               agg_level: str = "COUNTY") -> pd.DataFrame:
    """Corn-grain yield (bu/acre) for one state, agg_level=COUNTY|STATE."""
    return _nass_to_df(_nass({
        "source_desc": "SURVEY", "commodity_desc": "CORN",
        "statisticcat_desc": "YIELD", "unit_desc": "BU / ACRE",
        "util_practice_desc": "GRAIN", "agg_level_desc": agg_level,
        "state_name": STATE_NAME.get(state, state).upper(),
        "year__GE": year_start, "year__LE": year_end,
    }))


def nass_crop_progress(state: str, year_start: int = 2005,
                       year_end: int = 2025) -> pd.DataFrame:
    """Weekly corn progress %s (planted/silking/dough/dent/mature/harvested)."""
    return _nass_to_df(_nass({
        "source_desc": "SURVEY", "sector_desc": "CROPS", "commodity_desc": "CORN",
        "statisticcat_desc": "PROGRESS", "agg_level_desc": "STATE",
        "state_name": STATE_NAME.get(state, state).upper(),
        "year__GE": year_start, "year__LE": year_end,
    }))


def nass_in_season_forecast(state: str, year_start: int = 2005,
                            year_end: int = 2024) -> pd.DataFrame:
    """Monthly NASS in-season state yield forecasts (Aug/Sep/Oct/Nov reports)."""
    return _nass_to_df(_nass({
        "source_desc": "SURVEY", "commodity_desc": "CORN",
        "statisticcat_desc": "YIELD", "unit_desc": "BU / ACRE",
        "util_practice_desc": "GRAIN", "agg_level_desc": "STATE",
        "freq_desc": "MONTHLY",
        "state_name": STATE_NAME.get(state, state).upper(),
        "year__GE": year_start, "year__LE": year_end,
    }))


# ───────────────────────── US Drought Monitor ─────────────────────────

_USDM_API = ("https://usdmdataservices.unl.edu/api/CountyStatistics/"
             "GetDroughtSeverityStatisticsByAreaPercent")


def usdm_county_weekly(state_abbr: str, start: str | date, end: str | date) -> pd.DataFrame:
    """County-week drought severity & coverage for one state.

    Adds `dsci` = 1*D0+2*D1+3*D2+4*D3+5*D4 — the single drought scalar used
    everywhere downstream.
    """
    s = pd.Timestamp(start).strftime("%-m/%-d/%Y")
    e = pd.Timestamp(end).strftime("%-m/%-d/%Y")
    r = requests.get(_USDM_API, params={
        "aoi": state_abbr, "startdate": s, "enddate": e, "statisticsType": "1",
    }, headers={"Accept": "application/json"}, timeout=300)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        return df
    # API returns camelCase keys; normalize to the project's snake_case schema.
    df = df.rename(columns={"validStart": "valid_start", "validEnd": "valid_end"})
    for c in ("none", "d0", "d1", "d2", "d3", "d4"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["dsci"] = df["d0"] + 2 * df["d1"] + 3 * df["d2"] + 4 * df["d3"] + 5 * df["d4"]
    df["state"] = state_abbr
    df["valid_start"] = pd.to_datetime(df["valid_start"])
    df["valid_end"] = pd.to_datetime(df["valid_end"])
    return df[["fips", "state", "valid_start", "valid_end",
               "none", "d0", "d1", "d2", "d3", "d4", "dsci"]]


# ───────────────────────── CDL corn mask ─────────────────────────
# Two backends:
#   cdl_corn_mask_url   →  CropScape REST (covers all years incl. latest, but the
#                         GMU server is often slow / times out — wrap in try/except
#                         and fall back).
#   cdl_search_pc       →  Microsoft Planetary Computer STAC (years 2008–2021,
#                         reliable, returns COG URLs you can window-read).

_CDL_BBOX_API = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"
_PC_CDL_COLLECTION = "usda-cdl"


def cdl_corn_mask_url(year: int, bbox: tuple[float, float, float, float],
                      timeout: int = 60) -> str:
    """CropScape returns a TIF URL for CDL clipped to bbox (lon_min,lat_min,lon_max,lat_max).

    Caller can `rioxarray.open_rasterio(url)` and `(arr == 1)` to get the corn mask.
    Note: nassgeodata.gmu.edu is intermittently slow — use cdl_search_pc as a
    fallback for years 2008-2021.
    """
    minx, miny, maxx, maxy = bbox
    r = requests.get(_CDL_BBOX_API, params={
        "year": year, "bbox": f"{minx},{miny},{maxx},{maxy}",
    }, timeout=timeout)
    r.raise_for_status()
    text = r.text
    start = text.find("<returnURL>") + len("<returnURL>")
    end = text.find("</returnURL>")
    if start < len("<returnURL>") or end < 0:
        raise RuntimeError(f"CropScape returned no URL: {text[:300]}")
    return text[start:end]


def cdl_search_pc(year: int, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """Search USDA CDL on Planetary Computer (covers 2008-2021).

    Returns one row per matching tile with the `cropland` COG URL (already signed).
    """
    body = {
        "collections": [_PC_CDL_COLLECTION],
        "bbox": list(bbox),
        "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
        "limit": 50,
    }
    r = requests.post(_PC_STAC, json=body, timeout=60)
    r.raise_for_status()
    feats = r.json().get("features", [])
    rows = []
    for f in feats:
        rows.append({
            "id": f.get("id"),
            "datetime": f.get("properties", {}).get("datetime"),
            "cropland_href": f.get("assets", {}).get("cropland", {}).get("href"),
            "bbox": f.get("bbox"),
        })
    return pd.DataFrame(rows)


# ───────────────────────── Weather (NASA POWER, gridMET-equiv) ─────────────────────────

_POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point"

# POWER variables roughly equivalent to gridMET features:
#   T2M       daily mean 2m air temp °C       (gridMET tmmn/tmmx)
#   T2M_MAX   daily max
#   T2M_MIN   daily min
#   PRECTOTCORR  precip mm/day                  (gridMET pr)
#   ALLSKY_SFC_SW_DWN  shortwave MJ/m²/day      (gridMET srad)
#   RH2M      2m relative humidity %            (gridMET rmin/rmax)
#   WS2M      2m wind speed m/s                 (gridMET vs)
#   VPD via T2M+RH2M (computed below)
POWER_DEFAULT = ("T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR",
                 "ALLSKY_SFC_SW_DWN", "RH2M", "WS2M")


def weather_point(lat: float, lon: float, start: str | date, end: str | date,
                  params: Iterable[str] = POWER_DEFAULT,
                  community: str = "AG") -> pd.DataFrame:
    """Daily weather at a point from NASA POWER. No auth.

    Returns one row per day with columns matching the requested POWER variables,
    plus a derived `vpd_kpa` if T2M and RH2M are present. Use the county centroid
    as the point — this is the same idea as gridMET reduceRegions(mean).
    """
    s = pd.Timestamp(start).strftime("%Y%m%d")
    e = pd.Timestamp(end).strftime("%Y%m%d")
    r = requests.get(_POWER_API, params={
        "parameters": ",".join(params), "community": community,
        "longitude": lon, "latitude": lat,
        "start": s, "end": e, "format": "JSON",
    }, timeout=120)
    r.raise_for_status()
    payload = r.json()["properties"]["parameter"]
    df = pd.DataFrame(payload)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.replace(-999.0, pd.NA).astype("Float64")
    if "T2M" in df and "RH2M" in df:
        es = 0.6108 * (17.27 * df["T2M"] / (df["T2M"] + 237.3)).map(_exp)
        df["vpd_kpa"] = es * (1 - df["RH2M"] / 100.0)
    return df.reset_index()


def _exp(x):
    import math
    return math.exp(float(x)) if pd.notna(x) else pd.NA


# ───────────────────────── Soils (SoilGrids, gNATSGO-equiv) ─────────────────────────

_SOILGRIDS_API = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# SoilGrids properties roughly aligned with the gNATSGO bands the project uses:
#   wv0033 (water content @ -33 kPa)         ~ awc_0_25cm
#   soc    (soil organic carbon)             ~ om_pct (g/kg → divide by 10 for %)
#   clay                                     ~ clay_pct
#   bdod   (bulk density)                    bonus
SOILGRIDS_DEFAULT = ("wv0033", "soc", "clay", "bdod")


SOILGRIDS_DEPTHS = ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm")


def soil_point(lat: float, lon: float,
               properties: Iterable[str] = SOILGRIDS_DEFAULT,
               depths: Iterable[str] = ("0-5cm", "5-15cm", "15-30cm"),
               value: str = "mean") -> dict:
    """Soil properties at a point from SoilGrids v2 (250 m). No auth.

    Returns a flat dict like {"clay_0-5cm_mean": 23.1, ...}. Values are in
    SoilGrids native units (clay g/kg, soc dg/kg, wv0033 0.1 vol%, bdod cg/cm³)
    — divide clay by 10 for %, soc by 100 for %.

    Default depths cover the agronomically relevant 0-30 cm rooting zone. Valid
    depth labels: see SOILGRIDS_DEPTHS.
    """
    params = [("lat", lat), ("lon", lon), ("value", value)]
    params += [("property", p) for p in properties]
    params += [("depth", d) for d in depths]
    r = requests.get(_SOILGRIDS_API, params=params, timeout=120)
    r.raise_for_status()
    layers = r.json().get("properties", {}).get("layers", [])
    out = {}
    for layer in layers:
        name = layer["name"]
        for d in layer.get("depths", []):
            label = d["label"]
            for k, v in d.get("values", {}).items():
                out[f"{name}_{label}_{k}"] = v
    return out


# ───────────────────────── HLS via NASA CMR-STAC ─────────────────────────

_CMR_STAC = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search"


def hls_search(bbox: tuple[float, float, float, float],
               start: str | date, end: str | date,
               max_cloud: int = 30, limit: int = 50,
               product: str = "both") -> pd.DataFrame:
    """Search HLS L30 + S30 scenes via NASA CMR-STAC. Returns one row per scene
    with a `cog_assets` dict mapping band → signed COG URL.

    Search itself is anonymous; downloading the COG bytes needs a `~/.netrc`
    Earthdata entry (one-time, free signup).
    """
    collections = {"HLSL30": "HLSL30_2.0", "HLSS30": "HLSS30_2.0"}
    if product == "both":
        cids = list(collections.values())
    else:
        cids = [collections[product.upper()]]

    body = {
        "collections": cids,
        "bbox": list(bbox),
        "datetime": f"{pd.Timestamp(start).date()}T00:00:00Z/{pd.Timestamp(end).date()}T23:59:59Z",
        "limit": limit,
        "query": {"eo:cloud_cover": {"lte": max_cloud}},
    }
    r = requests.post(_CMR_STAC, json=body, timeout=120)
    r.raise_for_status()
    feats = r.json().get("features", [])
    rows = []
    for f in feats:
        props = f.get("properties", {})
        rows.append({
            "scene_id": f.get("id"),
            "collection": f.get("collection"),
            "datetime": props.get("datetime"),
            "cloud_cover": props.get("eo:cloud_cover"),
            "cog_assets": {k: v.get("href") for k, v in f.get("assets", {}).items()
                           if v.get("href", "").endswith(".tif")},
            "bbox": f.get("bbox"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def hls_open_band(scene_row: pd.Series, band: str,
                  bbox: tuple[float, float, float, float] | None = None):
    """Lazily open one HLS band as an xarray DataArray, optionally clipped to bbox.

    Reads only the bytes needed via rasterio's HTTP COG driver — no full-scene
    download. Caller is responsible for `.load()` or `.compute()` if eager.
    """
    import rioxarray  # noqa: F401
    import xarray as xr
    url = scene_row["cog_assets"].get(band)
    if not url:
        raise KeyError(f"Band {band!r} not in scene {scene_row['scene_id']}; "
                       f"have {list(scene_row['cog_assets'])}")
    da = xr.open_dataarray(url, engine="rasterio", chunks={"x": 512, "y": 512})
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        da = da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs="EPSG:4326")
    return da


# ───────────────────────── Landsat via Planetary Computer ─────────────────────────

_PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"


def landsat_search(bbox: tuple[float, float, float, float],
                   start: str | date, end: str | date,
                   max_cloud: int = 30, limit: int = 50) -> pd.DataFrame:
    """Search Landsat C2 L2 scenes via Microsoft Planetary Computer STAC.

    PC returns *unsigned* asset URLs; sign them with `planetary_computer.sign`
    before opening (do that lazily so we don't import the dep here).
    """
    body = {
        "collections": ["landsat-c2-l2"],
        "bbox": list(bbox),
        "datetime": f"{pd.Timestamp(start).date()}T00:00:00Z/{pd.Timestamp(end).date()}T23:59:59Z",
        "limit": limit,
        "query": {"eo:cloud_cover": {"lte": max_cloud}},
    }
    r = requests.post(_PC_STAC, json=body, timeout=120)
    r.raise_for_status()
    feats = r.json().get("features", [])
    rows = []
    for f in feats:
        props = f.get("properties", {})
        rows.append({
            "scene_id": f.get("id"),
            "platform": props.get("platform"),
            "datetime": props.get("datetime"),
            "cloud_cover": props.get("eo:cloud_cover"),
            "assets": {k: v.get("href") for k, v in f.get("assets", {}).items()},
            "bbox": f.get("bbox"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


# ───────────────────────── self-check ─────────────────────────
if __name__ == "__main__":
    print("smoke-testing data_api.py …")
    if os.getenv("NASS_API_KEY"):
        df = nass_yield("IA", 2022, 2024, agg_level="STATE")
        print(f"  NASS Iowa state yield 2022-2024: {len(df)} rows")
    else:
        print("  NASS skipped (no NASS_API_KEY)")
    df = usdm_county_weekly("IA", "2024-08-01", "2024-08-15")
    print(f"  USDM Iowa 2024-08-01..15: {len(df)} rows")
    wx = weather_point(42.03, -93.62, "2024-08-01", "2024-08-07")
    print(f"  POWER Ames IA 2024-08-01..07: {len(wx)} days, cols={list(wx.columns)}")
    soils = soil_point(42.03, -93.62)
    print(f"  SoilGrids Ames IA: {len(soils)} fields")
    hls = hls_search((-93.7, 41.9, -93.5, 42.1), "2024-08-01", "2024-08-31",
                     max_cloud=20, limit=5)
    print(f"  HLS STAC search: {len(hls)} scenes")
    if not hls.empty:
        print(f"    first scene bands: {list(hls.iloc[0]['cog_assets'])}")
    cdl = cdl_search_pc(2021, (-93.7, 41.9, -93.5, 42.1))
    print(f"  CDL (Planetary Computer) 2021: {len(cdl)} tiles")
    ls = landsat_search((-93.7, 41.9, -93.5, 42.1), "2024-08-01", "2024-08-31",
                        max_cloud=30, limit=3)
    print(f"  Landsat C2 (Planetary Computer): {len(ls)} scenes")
