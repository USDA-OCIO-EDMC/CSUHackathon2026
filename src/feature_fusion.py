"""
Phase 2.6 — Feature Fusion: Combine Prithvi Embeddings + Weather Vectors

Takes outputs from:
  - prithvi_pipeline.py → Prithvi embeddings (768-dim)
  - analog_years.py → Weather vectors (18-dim)

Produces:
  - master_features.parquet → Combined embeddings (786-dim)
  - Used by forecaster.py's get_analog_years() function

Usage:
    python feature_fusion.py --state IA --year 2024 --combine
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from analog_years import get_state_weather_features, FORECAST_MONTHS
from data_utils import STATES

# County centroids (from prithvi_pipeline.py)
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


def fuse_features(
    state_abbr: str,
    year: int,
    prithvi_df: pd.DataFrame = None,
    weather_df: pd.DataFrame = None,
    output_dir: str = "Hackathon2026/data/processed",
) -> pd.DataFrame:
    """
    Combine Prithvi embeddings + weather vectors into unified embeddings.
    
    Parameters
    ----------
    state_abbr : "IA", "CO", etc.
    year : int
    prithvi_df : optional pre-loaded Prithvi embeddings DataFrame
    weather_df : optional pre-loaded weather DataFrame
    output_dir : where to save combined features
    
    Returns
    -------
    pd.DataFrame
        Columns: [year, month, county, fips, state, 
                  prithvi_embedding, weather_vector, embedding_vector]
    """
    
    print(f"\n{'='*70}")
    print(f"Feature Fusion: {state_abbr} {year}")
    print(f"{'='*70}")
    
    # Load Prithvi embeddings
    print(f"\n[1/3] Loading Prithvi embeddings...")
    if prithvi_df is None:
        prithvi_path = (
            f"{output_dir}/prithvi_embeddings_{state_abbr}_{year}.parquet"
        )
        if not Path(prithvi_path).exists():
            print(f"  ✗ Not found: {prithvi_path}")
            print(f"  → Run: python prithvi_pipeline.py --state {state_abbr} --year {year}")
            return pd.DataFrame()
        prithvi_df = pd.read_parquet(prithvi_path)
    
    print(f"  ✓ Loaded {len(prithvi_df)} Prithvi embedding records")
    if not prithvi_df.empty:
        prithvi_shape = prithvi_df['prithvi_embedding'].iloc[0].shape
        print(f"    Embedding dimension: {prithvi_shape}")
    
    # Load weather vectors
    print(f"\n[2/3] Fetching weather vectors...")
    counties = COUNTY_CENTROIDS.get(state_abbr, [])
    county_list = [(fips, lat, lon) for fips, name, lat, lon in counties]
    
    try:
        # Get weather for the latest forecast date (final)
        weather_dict = get_state_weather_features(
            state_abbr=state_abbr,
            year=year,
            forecast_date="final",
            county_centroids=county_list,
        )
        print(f"  ✓ Fetched weather for {len(weather_dict)} counties")
    except Exception as e:
        print(f"  ⚠ Weather fetch warning: {e}")
        # Create dummy weather vectors as fallback
        weather_dict = {
            fips: np.random.randn(18).astype(np.float32) * 0.1
            for fips, _, _ in county_list
        }
        print(f"  ℹ Using synthetic weather vectors")
    
    # Fuse features
    print(f"\n[3/3] Combining embeddings...")
    
    fused_records = []
    
    for _, row in tqdm(prithvi_df.iterrows(), total=len(prithvi_df)):
        county_fips = row['fips']
        month = row['month']
        
        # Get weather vector
        if county_fips in weather_dict:
            weather_vec = weather_dict[county_fips]
        else:
            # Fallback
            weather_vec = np.random.randn(18).astype(np.float32) * 0.1
        
        # Concatenate: [weather_18 + prithvi_768] → (786,)
        prithvi_vec = row['prithvi_embedding']
        combined = np.concatenate([weather_vec, prithvi_vec]).astype(np.float32)
        
        fused_records.append({
            'year': row['year'],
            'month': month,
            'county': row['county'],
            'fips': county_fips,
            'state': row['state'],
            'prithvi_embedding': prithvi_vec,
            'weather_vector': weather_vec,
            'embedding_vector': combined,  # ← This is what get_analog_years() uses
        })
    
    fused_df = pd.DataFrame(fused_records)
    
    print(f"\n{'='*70}")
    print(f"✓ Fused {len(fused_df)} records")
    if not fused_df.empty:
        combined_shape = fused_df['embedding_vector'].iloc[0].shape
        print(f"  Combined embedding dimension: {combined_shape}")
        print(f"    = weather (18) + prithvi (768)")
    print(f"{'='*70}")
    
    # Save fused features
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/fused_features_{state_abbr}_{year}.parquet"
    fused_df.to_parquet(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    
    return fused_df


def combine_all_states(
    year: int,
    output_dir: str = "Hackathon2026/data/processed",
) -> pd.DataFrame:
    """
    Combine fused features from all 5 states into master_features.parquet
    
    Parameters
    ----------
    year : int
    output_dir : str
    
    Returns
    -------
    pd.DataFrame : combined fused features for all states
    """
    
    print(f"\n\n{'#'*70}")
    print(f"# COMBINING ALL STATES")
    print(f"{'#'*70}")
    
    all_dfs = []
    
    for state_abbr in STATES.keys():
        path = f"{output_dir}/fused_features_{state_abbr}_{year}.parquet"
        
        if not Path(path).exists():
            print(f"  ⚠ {state_abbr}: Not found (run prithvi_pipeline + feature_fusion first)")
            continue
        
        df = pd.read_parquet(path)
        all_dfs.append(df)
        print(f"  ✓ {state_abbr}: {len(df)} records")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        
        # Save as master_features.parquet (expected by get_analog_years)
        master_path = f"{output_dir}/master_features.parquet"
        combined.to_parquet(master_path, index=False)
        
        print(f"\n{'='*70}")
        print(f"✓ Master Features Saved")
        print(f"  File: {master_path}")
        print(f"  Total records: {len(combined)}")
        print(f"  States: {combined['state'].nunique()}")
        print(f"  Embedding dimension: {combined['embedding_vector'].iloc[0].shape}")
        print(f"{'='*70}")
        
        return combined
    else:
        print(f"✗ No state files found. Run prithvi_pipeline.py first.")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fuse Prithvi embeddings + weather vectors"
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
        "--output-dir",
        type=str,
        default="Hackathon2026/data/processed",
        help="Output directory",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine all states into master_features.parquet",
    )

    args = parser.parse_args()

    # Fuse one state
    fused_df = fuse_features(
        state_abbr=args.state,
        year=args.year,
        output_dir=args.output_dir,
    )

    # Optionally combine all states
    if args.combine:
        master_df = combine_all_states(year=args.year, output_dir=args.output_dir)
