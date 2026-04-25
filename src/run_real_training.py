
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_utils import get_nass_yields
from train_model import prepare_matrices, train_and_validate

def run_training_with_real_data(state_fips, start_year, end_year):
    """
    Runs the full training pipeline with real data.
    
    1. Fetches historical yield data from the CSV file.
    2. Runs the Prithvi feature extraction pipeline (this is a placeholder
       and would typically involve significant data processing).
    3. Prepares the data matrices for XGBoost.
    4. Trains and validates the model using Leave-One-Out Cross-Validation.
    """
    print("="*60)
    print("  STARTING REAL DATA TRAINING PIPELINE")
    print("="*60)

    # --- 1. Load Real Yield Data ---
    print("\n[STEP 1] Loading real NASS yield data...")
    try:
        # In a real scenario, you might use get_nass_yields, but for this
        # example, we'll load directly from the provided CSV.
        yield_df = pd.read_csv("data/raw/corn_yield_2005_2024.csv")
        # The CSV does not contain state fips, so we cannot filter by it.
        # We will use all data present in the file.
        print(f"  ✓ Loaded {len(yield_df)} records.")
    except FileNotFoundError:
        print("  ✗ ERROR: data/raw/corn_yield_2005_2024.csv not found.")
        return

    # --- 2. Generate Seasonal Signatures (Prithvi Embeddings) ---
    # NOTE: This is a simplified stand-in for the complex feature extraction process.
    # The real `run_feature_extraction` would download HLS data, process it,
    # and extract embeddings, which is a long and resource-intensive task.
    # Here, we'll create mock embeddings that align with the yield data years.
    '''
    print("\n[STEP 2] Generating seasonal signatures (Prithvi embeddings)...")
    
    # The real data does not have fips codes, so we will mock them for alignment.
    # A more robust solution would involve mapping county names to fips codes.

    #county_map = {name: f"mock_fips_{i}" for i, name in enumerate(yield_df['county_name'].unique())}
    #yield_df['county_fips'] = yield_df['county_name'].map(county_map)
    #yield_df.rename(columns={'county_name': 'county'}, inplace=True)
    #yield_df['state_fips'] = '19' # Add mock state_fips for merging


    counties = yield_df[['county_fips', 'county']].drop_duplicates()
    seasonal_signatures_list = []
    
    for year in range(start_year, end_year + 1):
        for _, row in counties.iterrows():
            # MOCK EMBEDDING: Replace with real feature extraction
            embedding = np.random.rand(768) 
            seasonal_signatures_list.append({
                'year': year,
                'state_fips': '19', # Mock state fips
                'county_fips': row['county_fips'],
                'county': row['county'],
                'embedding_vector': embedding
            })
            
    seasonal_signatures = pd.DataFrame(seasonal_signatures_list)
    print(f"  ✓ Generated {len(seasonal_signatures)} mock seasonal signatures for alignment.")
    print("  (NOTE: Using mock embeddings as the real process is resource-intensive)")
'''

    print("\n[STEP 2] Loading real Prithvi embeddings...")
    

    features_path = Path("data/processed/prithvi_embeddings_2005_2024.parquet")
    
    if features_path.exists():

        seasonal_signatures = pd.read_parquet(features_path)
        

        if isinstance(seasonal_signatures['embedding_vector'].iloc[0], list):
            seasonal_signatures['embedding_vector'] = seasonal_signatures['embedding_vector'].apply(np.array)
            
        print(f"  ✓ Loaded {len(seasonal_signatures)} REAL satellite signatures from disk.")
    else:
 
        print("  ! Feature file not found. Running real extraction (Warning: This is slow)...")
        

        print("  ✗ ERROR: Please run the feature extraction script first.")
        return
        
    # --- 3. Prepare Data Matrices ---
    print("\n[STEP 3] Preparing data matrices for the model...")
    X, y, years = prepare_matrices(seasonal_signatures, yield_df)
    print(f"  ✓ Feature matrix X shape: {X.shape}")
    print(f"  ✓ Target vector y shape: {y.shape}")

    # --- 4. Train and Validate Model ---
    print("\n[STEP 4] Training and validating the model with LOOCV...")
# Update this call as well:
    mae, actuals, preds = train_and_validate(X, y, years)
    
    print("\n" + "="*60)
    print("  REAL DATA TRAINING PIPELINE COMPLETE")
    print(f"  Final Validation MAE: {mae:.2f} bu/acre")
    print("="*60)

    return mae

if __name__ == "__main__":
    # Example: Run for Iowa (fips 19) for years 2022-2024
    run_training_with_real_data(state_fips="19", start_year=2024, end_year=2025)
