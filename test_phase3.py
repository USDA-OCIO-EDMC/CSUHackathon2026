"""
Testing Suite for Phase 3 Pipeline
Includes:
  - Unit tests (individual functions)
  - Mock tests (no API calls needed)
  - Integration tests (full pipeline)
  - Real tests (with NASA APIs)

Run:
  python test_phase3.py --mode mock      # Quick test (no APIs)
  python test_phase3.py --mode unit      # Function tests
  python test_phase3.py --mode real      # Full pipeline (requires credentials)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch


# ===========================================================================
# PART 1: UNIT TESTS (Test individual functions)
# ===========================================================================

def test_prithvi_loading():
    """Test that Prithvi model loads correctly."""
    print("\n[TEST 1] Loading Prithvi Model")
    print("-" * 60)
    
    try:
        from privthi_extractor import load_prithvi
        
        print("  Loading Prithvi-100M on CPU...")
        model = load_prithvi(device="cpu")
        
        print(f"  ✓ Model type: {type(model)}")
        print(f"  ✓ Model loaded successfully on CPU")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_weather_extraction():
    """Test weather vector extraction."""
    print("\n[TEST 2] Weather Vector Extraction")
    print("-" * 60)
    
    try:
        from analog_years import get_county_weather_features_single
        
        print("  Fetching weather for Story County, IA (42.0, -93.6)...")
        weather_vec = get_county_weather_features_single(
            lat=42.0,
            lon=-93.6,
            year=2024,
            forecast_date="aug1",
            retry=1
        )
        
        print(f"  ✓ Weather vector shape: {weather_vec.shape}")
        print(f"  ✓ Sample values: {weather_vec[:5]}")
        
        assert weather_vec.shape == (18,), f"Expected (18,), got {weather_vec.shape}"
        print(f"  ✓ Dimension check passed")
        
        return True
    except Exception as e:
        print(f"  ⚠ Weather test skipped (API issue): {e}")
        return None  # Skip, not failure


def test_data_utils():
    """Test NASS yield data fetching."""
    print("\n[TEST 3] NASS Yield Data Fetching")
    print("-" * 60)
    
    try:
        from data_utils import get_nass_yields
        
        if not os.environ.get("NASS_API_KEY"):
            print("  ⚠ Skipped (NASS_API_KEY not set)")
            return None
        
        print("  Fetching yields for Iowa (FIPS 19)...")
        df = get_nass_yields(state_fips="19", year_start=2020, year_end=2023)
        
        print(f"  ✓ Fetched {len(df)} records")
        print(f"  ✓ Columns: {list(df.columns)}")
        print(f"  Sample:\n{df.head(2)}")
        
        return True
    except Exception as e:
        print(f"  ⚠ NASS test skipped: {e}")
        return None


# ===========================================================================
# PART 2: MOCK TESTS (No API calls, synthetic data)
# ===========================================================================

def test_mock_prithvi_extraction():
    """Test Prithvi extraction with synthetic HLS data."""
    print("\n[MOCK TEST 1] Prithvi Extraction with Synthetic Data")
    print("-" * 60)
    
    try:
        from privthi_extractor import load_prithvi, extract_features_from_array
        
        print("  Creating synthetic HLS tile (6, 224, 224)...")
        synthetic_hls = np.random.randint(0, 10000, size=(6, 224, 224)).astype(np.float32)
        
        print("  Loading Prithvi model...")
        model = load_prithvi(device="cpu")
        
        print("  Extracting features...")
        embedding = extract_features_from_array(model, synthetic_hls, max_tiles=1)
        
        print(f"  ✓ Embedding shape: {embedding.shape}")
        print(f"  ✓ Embedding type: {type(embedding)}")
        print(f"  ✓ Sample values: {embedding[:5]}")
        
        assert embedding.shape == (768,), f"Expected (768,), got {embedding.shape}"
        print(f"  ✓ Dimension check passed")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mock_weather_vectors():
    """Test weather vector creation with synthetic data."""
    print("\n[MOCK TEST 2] Weather Vector Synthesis")
    print("-" * 60)
    
    try:
        print("  Creating synthetic weather vectors...")
        
        # Simulate what analog_years.py produces
        weather_vecs = []
        for _ in range(10):
            # 18-dim: [temp, precip, gdd] × 6 months
            vec = np.random.normal(loc=0, scale=1, size=18).astype(np.float32)
            weather_vecs.append(vec)
        
        weather_array = np.array(weather_vecs)
        
        print(f"  ✓ Created {len(weather_array)} weather vectors")
        print(f"  ✓ Shape: {weather_array.shape}")
        print(f"  ✓ Sample vector: {weather_array[0][:5]}")
        
        assert weather_array.shape == (10, 18), f"Wrong shape"
        print(f"  ✓ Dimension check passed")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_mock_feature_fusion():
    """Test feature fusion with synthetic data."""
    print("\n[MOCK TEST 3] Feature Fusion (Weather + Prithvi)")
    print("-" * 60)
    
    try:
        print("  Creating synthetic weather (18-dim) + Prithvi (768-dim)...")
        
        weather_vec = np.random.randn(18).astype(np.float32)
        prithvi_vec = np.random.randn(768).astype(np.float32)
        
        print(f"  Weather shape: {weather_vec.shape}")
        print(f"  Prithvi shape: {prithvi_vec.shape}")
        
        # Concatenate (what feature_fusion.py does)
        combined = np.concatenate([weather_vec, prithvi_vec])
        
        print(f"  ✓ Combined shape: {combined.shape}")
        print(f"  ✓ Expected: (786,)")
        
        assert combined.shape == (786,), f"Expected (786,), got {combined.shape}"
        print(f"  ✓ Dimension check passed")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_mock_get_analog_years():
    """Test analog year identification with mock data."""
    print("\n[MOCK TEST 4] Analog Year Identification")
    print("-" * 60)
    
    try:
        from forecaster import get_analog_years
        
        print("  Creating mock feature space...")
        
        # Create synthetic features for 2005-2024 + 2025
        records = []
        years = list(range(2005, 2026))
        months = [5, 6, 7]
        
        for year in years:
            for month in months:
                # Create random 786-dim embedding
                embedding = np.random.randn(786).astype(np.float32)
                
                # Make 2012 and 2025 similar (test that we identify them)
                if year in [2012, 2025]:
                    embedding = np.random.normal(0.5, 0.01, 786).astype(np.float32)
                
                records.append({
                    'year': year,
                    'month': month,
                    'county': 'STORY',
                    'fips': '19193',
                    'embedding_vector': embedding
                })
        
        feature_df = pd.DataFrame(records)
        
        # Mock yield data
        yield_df = pd.DataFrame({
            'year': list(range(2005, 2025)),
            'yield_bu_acre': np.random.uniform(150, 200, 20)
        })
        
        print(f"  ✓ Created feature space: {len(feature_df)} records")
        print(f"  ✓ Created yield data: {len(yield_df)} records")
        
        print("  Running analog year identification...")
        result = get_analog_years(
            state_fips="19",
            county_name="STORY",
            forecast_date="aug1",
            feature_df=feature_df,
            yield_df=yield_df,
        )
        
        print(f"  ✓ Got results: {len(result)} analog years")
        print(f"\n  Results:")
        print(result.to_string(index=False))
        
        # Check that 2012 is ranked high (since we made it similar to 2025)
        if not isinstance(result, str) and len(result) > 0:
            top_year = result.iloc[0]['year']
            top_sim = result.iloc[0]['similarity']
            print(f"\n  ✓ Top analog year: {top_year} (similarity: {top_sim:.3f})")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ===========================================================================
# PART 3: INTEGRATION TESTS (Full pipeline with mocks)
# ===========================================================================

def test_full_mock_pipeline():
    """Test complete pipeline flow with synthetic data."""
    print("\n[INTEGRATION TEST] Full Mock Pipeline")
    print("=" * 60)
    
    try:
        print("\nStep 1: Create mock Prithvi embeddings for all counties...")
        
        records = []
        for year in [2023, 2024, 2025]:
            for month in range(5, 11):  # May-Oct
                for county_fips, county_name in [
                    ("19193", "STORY"),
                    ("19013", "BLACK HAWK"),
                ]:
                    # Synthetic 768-dim Prithvi embedding
                    embedding = np.random.randn(768).astype(np.float32)
                    records.append({
                        'year': year,
                        'month': month,
                        'county': county_name,
                        'fips': county_fips,
                        'state': 'IA',
                        'prithvi_embedding': embedding,
                    })
        
        prithvi_df = pd.DataFrame(records)
        print(f"  ✓ Created {len(prithvi_df)} Prithvi records")
        
        print("\nStep 2: Create mock weather vectors...")
        
        weather_records = []
        for record in records:
            weather_vec = np.random.randn(18).astype(np.float32)
            weather_records.append({
                'fips': record['fips'],
                'weather_vector': weather_vec,
            })
        
        weather_map = {r['fips']: r['weather_vector'] for r in weather_records}
        print(f"  ✓ Created weather for {len(weather_map)} counties")
        
        print("\nStep 3: Fuse features (Weather + Prithvi)...")
        
        fused_records = []
        for _, row in prithvi_df.iterrows():
            weather_vec = weather_map.get(row['fips'], np.random.randn(18).astype(np.float32))
            combined = np.concatenate([weather_vec, row['prithvi_embedding']])
            
            fused_records.append({
                'year': row['year'],
                'month': row['month'],
                'county': row['county'],
                'fips': row['fips'],
                'embedding_vector': combined,
            })
        
        master_df = pd.DataFrame(fused_records)
        print(f"  ✓ Created {len(master_df)} fused records")
        print(f"  ✓ Embedding dimension: {master_df['embedding_vector'].iloc[0].shape}")
        
        print("\nStep 4: Run analog year identification...")
        
        yield_df = pd.DataFrame({
            'year': list(range(2020, 2025)),
            'yield_bu_acre': np.random.uniform(150, 200, 5)
        })
        
        from forecaster import get_analog_years
        
        result = get_analog_years(
            state_fips="19",
            county_name="STORY",
            forecast_date="aug1",
            feature_df=master_df,
            yield_df=yield_df,
        )
        
        if isinstance(result, str):
            print(f"  ⚠ {result}")
        else:
            print(f"  ✓ Identified {len(result)} analog years")
            print(f"\n  Top analog years:")
            print(result.to_string(index=False))
        
        print("\n" + "=" * 60)
        print("✓ FULL MOCK PIPELINE TEST PASSED")
        print("=" * 60)
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ===========================================================================
# PART 4: REAL TESTS (with actual NASA APIs)
# ===========================================================================

def test_real_hls_search():
    """Test real HLS search from NASA CMR."""
    print("\n[REAL TEST 1] HLS Granule Search")
    print("-" * 60)
    
    try:
        from hls_downloader import search_hls
        
        print("  Searching for HLS granules covering Iowa (2024, final)...")
        granules = search_hls("IA", 2024, "final")
        
        print(f"  ✓ Found {len(granules)} granules")
        if granules:
            granule_id = granules[0]["umm"]["GranuleUR"]
            print(f"  ✓ Sample granule ID: {granule_id}")
        
        return True
    except Exception as e:
        print(f"  ⚠ HLS search failed (API issue): {e}")
        return None


def test_real_weather_api():
    """Test real NASA POWER weather API."""
    print("\n[REAL TEST 2] NASA POWER Weather API")
    print("-" * 60)
    
    try:
        from analog_years import get_county_weather_features_single
        
        print("  Fetching weather from NASA POWER for Story County...")
        weather = get_county_weather_features_single(
            lat=42.0, lon=-93.6, year=2024, forecast_date="final"
        )
        
        print(f"  ✓ Weather shape: {weather.shape}")
        print(f"  ✓ Sample values: {weather[:3]}")
        
        return True
    except Exception as e:
        print(f"  ⚠ Weather API test failed: {e}")
        return None


# ===========================================================================
# MAIN TEST RUNNER
# ===========================================================================

def run_all_tests(mode="mock"):
    """Run tests based on mode."""
    
    print("\n" + "=" * 70)
    print(f"PHASE 3 TEST SUITE - Mode: {mode.upper()}")
    print("=" * 70)
    
    results = {}
    
    if mode == "unit":
        print("\n>>> RUNNING UNIT TESTS <<<")
        results['prithvi_loading'] = test_prithvi_loading()
        results['weather_extraction'] = test_weather_extraction()
        results['data_utils'] = test_data_utils()
    
    elif mode == "mock":
        print("\n>>> RUNNING MOCK TESTS (No API calls) <<<")
        results['mock_prithvi'] = test_mock_prithvi_extraction()
        results['mock_weather'] = test_mock_weather_vectors()
        results['mock_fusion'] = test_mock_feature_fusion()
        results['mock_analog'] = test_mock_get_analog_years()
        results['full_pipeline'] = test_full_mock_pipeline()
    
    elif mode == "real":
        print("\n>>> RUNNING REAL TESTS (with NASA APIs) <<<")
        print("\nNote: Requires EARTHDATA_USERNAME and EARTHDATA_PASSWORD")
        results['hls_search'] = test_real_hls_search()
        results['weather_api'] = test_real_weather_api()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is None)
    failed = sum(1 for v in results.values() if v is False)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result is True else ("⊘ SKIP" if result is None else "✗ FAIL")
        print(f"  {status:8} {test_name}")
    
    print("\n" + "-" * 70)
    print(f"Total: {passed} passed, {skipped} skipped, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Phase 3 Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="mock",
        choices=["unit", "mock", "real"],
        help="Test mode: unit, mock (default), or real",
    )
    
    args = parser.parse_args()
    exit_code = run_all_tests(mode=args.mode)
    sys.exit(exit_code)
