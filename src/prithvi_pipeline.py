"""
Phase 2.5 — Complete Prithvi Extraction Pipeline

Orchestrates the full workflow:
1. Search HLS granules for a state/year/forecast_date
2. Extract Prithvi embeddings from each granule
3. Map embeddings to counties/months using spatial overlap
4. Store results: [year, month, county, fips, prithvi_embedding]

Usage:
    python prithvi_pipeline.py --state IA --year 2024 --forecast-date final
    
Or in code:
    from prithvi_pipeline import build_prithvi_embeddings
    df = build_prithvi_embeddings(state_abbr="IA", year=2024, forecast_date="final", device="cuda")
    df.to_parquet("prithvi_embeddings_IA.parquet")
"""

import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import torch
import rasterio
from rasterio.features import geometry_mask

# Local imports
from hls_downloader import login as hls_login, search_hls, iter_granules_cloud
from privthi_extractor import load_prithvi, extract_features_from_array
from data_utils import STATES

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "data" / "processed")

# ---------------------------------------------------------------------------
# County Geometry Setup
# ---------------------------------------------------------------------------

# County centroids (FIPS, Name, Lat, Lon) for each state
# In production, these would be loaded from a full shapefile
COUNTY_CENTROIDS = {
    "IA": [
        ("19001", "ADAIR", 41.3306, -94.4686),
        ("19013", "BLACK HAWK", 42.3333, -92.3333),
        ("19027", "BUENA VISTA", 42.7500, -95.3000),
        ("19049", "GRUNDY", 42.3833, -92.7667),
        ("19065", "JASPER", 41.9000, -93.4667),
        ("19193", "STORY", 42.0083, -93.6167),
    ],
    "CO": [
        ("08087", "YUMA", 40.0667, -102.6167),
        ("08125", "WELD", 40.5000, -104.6833),
        ("08031", "DENVER", 39.7392, -104.9903),
    ],
    "WI": [
        ("55011", "CHIPPEWA", 44.8333, -91.3667),
        ("55095", "OUTAGAMIE", 44.2500, -88.6667),
        ("55115", "SAUK", 43.4833, -89.7333),
    ],
    "MO": [
        ("29087", "HOWARD", 38.5500, -92.3667),
        ("29227", "VERNON", 37.8000, -94.2667),
    ],
    "NE": [
        ("31193", "YORK", 40.8500, -97.5833),
        ("31079", "HAMILTON", 40.5333, -97.9833),
    ],
}

# Buffer (degrees) around county centroid to check for HLS coverage
# ~0.5 degree = ~56 km at these latitudes, reasonable for county-size areas
COUNTY_BUFFER_DEG = 0.5


# ---------------------------------------------------------------------------
# Spatial Utilities
# ---------------------------------------------------------------------------

def bbox_from_centroid(lat, lon, buffer_deg=COUNTY_BUFFER_DEG):
    """Create a bounding box around a county centroid."""
    return (lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg)


def is_granule_over_county(granule_bbox, county_lat, county_lon, buffer_deg=COUNTY_BUFFER_DEG):
    """
    Check if HLS granule bbox overlaps with county buffer zone.
    
    Parameters
    ----------
    granule_bbox : tuple (minx, miny, maxx, maxy) from rasterio profile
    county_lat, county_lon : county centroid
    buffer_deg : buffer around centroid
    
    Returns
    -------
    bool : True if granule covers county
    """
    county_bbox = bbox_from_centroid(county_lat, county_lon, buffer_deg)
    
    # Check overlap: granule (g) and county (c)
    minx_g, miny_g, maxx_g, maxy_g = granule_bbox
    minx_c, miny_c, maxx_c, maxy_c = county_bbox
    
    # No overlap if one is completely to the left/right/above/below the other
    if maxx_g < minx_c or minx_g > maxx_c:
        return False
    if maxy_g < miny_c or miny_g > maxy_c:
        return False
    
    return True


def get_granule_bbox(profile):
    """Extract bbox from rasterio profile."""
    transform = profile['transform']
    width = profile['width']
    height = profile['height']
    
    # Top-left corner
    minx = transform.c
    maxy = transform.f
    
    # Bottom-right corner
    maxx = transform.c + transform.a * width
    miny = transform.f + transform.e * height
    
    return (minx, miny, maxx, maxy)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def extract_monthly_embeddings(
    state_abbr: str,
    year: int,
    forecast_date: str,
    model,
    device: str = "cpu",
) -> dict:
    """
    Extract Prithvi embeddings for all HLS granules covering a state,
    for a specific year/forecast_date window.
    
    Returns dict mapping (county_fips, month) -> [embedding_vectors]
    where multiple embeddings per (county, month) can occur if multiple
    granules cover that area on different dates.
    
    Parameters
    ----------
    state_abbr : "IA", "CO", etc.
    year : int
    forecast_date : "aug1", "sep1", "oct1", "final"
    model : loaded Prithvi model
    device : "cpu" or "cuda"
    
    Returns
    -------
    dict:
        {
            (county_fips, month): [vec1, vec2, ...],  # Multiple granules
            ...
        }
    """
    
    results = {}
    
    print(f"\n{'='*70}")
    print(f"Extracting Prithvi embeddings for {state_abbr} {year} [{forecast_date}]")
    print(f"{'='*70}")
    
    # Search HLS granules
    print(f"\n[1/3] Searching HLS granules...")
    try:
        granules = search_hls(state_abbr, year, forecast_date)
    except Exception as e:
        print(f"  ✗ HLS search failed: {e}")
        return results
    
    if not granules:
        print(f"  ✗ No granules found for {state_abbr} {year} [{forecast_date}]")
        return results
    
    print(f"  ✓ Found {len(granules)} granules")
    
    # Get county centroids for this state
    counties = COUNTY_CENTROIDS.get(state_abbr, [])
    if not counties:
        print(f"  ✗ No county centroids configured for {state_abbr}")
        return results
    
    print(f"  ✓ {len(counties)} counties to match")
    
    # Extract embeddings from each granule
    print(f"\n[2/3] Extracting embeddings from granules...")
# HLS GranuleUR format: HLS.{L30|S30}.T{tile}.{YYYY}{DDD}T{HHMMSS}.v2.0
_GRANULE_DATE_RE = re.compile(r"\.(\d{4})(\d{3})T\d{6}\.")


def _month_from_granule_id(granule_id: str, fallback: int) -> int:
    """Extract acquisition month from HLS GranuleUR. Returns fallback on parse failure."""
    m = _GRANULE_DATE_RE.search(granule_id or "")
    if not m:
        return fallback
    try:
        year = int(m.group(1))
        doy = int(m.group(2))
        return (datetime(year, 1, 1) + timedelta(days=doy - 1)).month
    except (ValueError, OverflowError):
        return fallback


    
    for stacked, profile, granule_id in iter_granules_cloud(
        state_abbr, year, forecast_date, max_granules=None
    ):
        try:
            # Get granule bbox and date info
            granule_bbox = get_granule_bbox(profile)

            # Extract acquisition month from granule_id; fall back to mid-window if unparseable
            forecast_months = {
                "aug1": [5, 6, 7],
                "sep1": [5, 6, 7, 8],
                "oct1": [5, 6, 7, 8, 9],
                "final": [5, 6, 7, 8, 9, 10],
            }
            window = forecast_months.get(forecast_date, [5, 6, 7, 8, 9, 10])
            month = _month_from_granule_id(granule_id, fallback=window[len(window) // 2])
            # Drop granules acquired outside the forecast window (no temporal leakage)
            if month not in window:
                continue
            
            # Extract Prithvi embedding
            try:
                embedding = extract_features_from_array(model, stacked, max_tiles=4)
            except Exception as e:
                print(f"    ⚠ Could not extract features from {granule_id}: {e}")
                continue
            
            # Match to counties
            matched_counties = []
            for county_fips, county_name, lat, lon in counties:
                if is_granule_over_county(granule_bbox, lat, lon):
                    key = (county_fips, month)
                    if key not in results:
                        results[key] = []
                    results[key].append(embedding)
                    matched_counties.append(county_name)
            
            if matched_counties:
                print(f"  ✓ {granule_id}: {len(embedding)}-dim → {', '.join(matched_counties[:2])}")
            
        except Exception as e:
            print(f"  ✗ Error processing {granule_id}: {e}")
            continue
    
    print(f"\n[3/3] Aggregating results...")
    print(f"  ✓ Extracted embeddings for {len(results)} (county, month) pairs")
    
    return results


def build_prithvi_embeddings(
    state_abbr: str,
    year: int,
    forecast_date: str = "final",
    device: str = "cpu",
    output_dir: str = None,
) -> pd.DataFrame:
    """
    Build a complete Prithvi embeddings DataFrame for a state/year.
    
    Parameters
    ----------
    state_abbr : "IA", "CO", "WI", "MO", "NE"
    year : int
    forecast_date : "aug1" | "sep1" | "oct1" | "final"
    device : "cpu" or "cuda"
    output_dir : optional directory to save parquet
    
    Returns
    -------
    pd.DataFrame
        Columns: [year, month, county, fips, prithvi_embedding]
    """
    
    print(f"\n{'#'*70}")
    print(f"# PRITHVI EXTRACTION PIPELINE")
    print(f"# State: {state_abbr}, Year: {year}, Forecast: {forecast_date}")
    print(f"{'#'*70}")
    
    # Authenticate with NASA EarthData
    print(f"\n[0/2] Authenticating with NASA EarthData...")
    try:
        hls_login()
        print(f"  ✓ Authentication successful")
    except Exception as e:
        print(f"  ✗ Authentication failed: {e}")
        print(f"  → Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables")
        print(f"  → Register at https://urs.earthdata.nasa.gov/")
        return pd.DataFrame()
    
    # Load Prithvi model
    print(f"\n[1/2] Loading Prithvi-100M model...")
    try:
        model = load_prithvi(device=device)
        print(f"  ✓ Model loaded on {device}")
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        return pd.DataFrame()
    
    # Extract embeddings
    print(f"\n[2/2] Extracting embeddings...")
    embeddings_dict = extract_monthly_embeddings(
        state_abbr, year, forecast_date, model, device
    )
    
    # Convert to DataFrame
    records = []
    for (county_fips, month), embeddings in embeddings_dict.items():
        # Average multiple embeddings if multiple granules cover the county
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            
            # Find county name
            county_name = None
            for fips, name, lat, lon in COUNTY_CENTROIDS.get(state_abbr, []):
                if fips == county_fips:
                    county_name = name
                    break
            
            records.append({
                'year': year,
                'month': month,
                'county': county_name or county_fips,
                'fips': county_fips,
                'state': state_abbr,
                'prithvi_embedding': avg_embedding,
            })
    
    df = pd.DataFrame(records)
    
    print(f"\n{'='*70}")
    print(f"✓ Complete! {len(df)} records extracted")
    if not df.empty:
        print(f"  Embedding dimension: {df['prithvi_embedding'].iloc[0].shape}")
    print(f"{'='*70}")
    
    # Save if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/prithvi_embeddings_{state_abbr}_{year}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"\n✓ Saved to: {output_path}")
    
    return df


def build_all_states(
    year: int,
    forecast_date: str = "final",
    device: str = "cpu",
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Build Prithvi embeddings for all 5 target states, then combine.
    
    Parameters
    ----------
    year : int
    forecast_date : str
    device : str
    output_dir : str
    
    Returns
    -------
    pd.DataFrame : combined embeddings for all states
    """
    
    all_dfs = []
    
    for state_abbr in STATES.keys():
        print(f"\n\n{'*'*70}")
        print(f"PROCESSING STATE: {state_abbr}")
        print(f"{'*'*70}")
        
        try:
            df = build_prithvi_embeddings(
                state_abbr=state_abbr,
                year=year,
                forecast_date=forecast_date,
                device=device,
                output_dir=output_dir,
            )
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            print(f"✗ Error processing {state_abbr}: {e}")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        output_path = f"{output_dir}/prithvi_embeddings_all_{year}.parquet"
        combined.to_parquet(output_path, index=False)
        print(f"\n✓ Combined file: {output_path}")
        print(f"  Total records: {len(combined)}")
        return combined
    else:
        print(f"✗ No data extracted for any state")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Prithvi embeddings from HLS satellite imagery"
    )
    parser.add_argument(
        "--state",
        type=str,
        default="IA",
        help="State code: IA, CO, WI, MO, NE (default: IA)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Year (default: 2024)",
    )
    parser.add_argument(
        "--forecast-date",
        type=str,
        default="final",
        choices=["aug1", "sep1", "oct1", "final"],
        help="Forecast date (default: final)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu or cuda (default: cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--all-states",
        action="store_true",
        help="Process all 5 states and combine",
    )

    args = parser.parse_args()

    if args.all_states:
        combined_df = build_all_states(
            year=args.year,
            forecast_date=args.forecast_date,
            device=args.device,
            output_dir=args.output_dir,
        )
    else:
        df = build_prithvi_embeddings(
            state_abbr=args.state,
            year=args.year,
            forecast_date=args.forecast_date,
            device=args.device,
            output_dir=args.output_dir,
        )
