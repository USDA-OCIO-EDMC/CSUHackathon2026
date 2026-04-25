import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_analog_years(state_fips, county_name, forecast_date,feature_df,yield_df):
    """
    Revised Phase 3: Actively matches the growth window (months) 
    to ensure we only compare data known BEFORE the forecast_date.
    """
    # 1. Define the "Knowledge Window" based on the forecast_date
    # If it's Aug 1, we only know what happened in May, June, July.
    date_map = {
        "aug1": [5, 6, 7],
        "sep1": [5, 6, 7, 8],
        "oct1": [5, 6, 7, 8, 9],
        "final": [5, 6, 7, 8, 9, 10]
    }
    
    target_months = date_map.get(forecast_date.lower())
    
    # 2. Load the Feature Space (Assuming a master file with a 'month' column)
    # This file has columns: [year, month, county, fips, embedding_vector]
    if feature_df is None:
        feature_df = pd.read_parquet("Hackathon2026/data/processed/master_features.parquet")
    
    # 3. FILTER BY TIME: Only keep data from the months we care about
    # This prevents "cheating" by using future data.
    filtered_df = feature_df[feature_df['month'].isin(target_months)]
    
    # 4. AGGREGATE: Group by Year/County to get ONE vector for the whole period
    def combine_vectors(group):
        return np.mean(np.stack(group.values), axis=0)

    # group by year&county and only take embedding_vector and get its average
    seasonal_signatures = filtered_df.groupby(['year', 'state_flip','county'])['embedding_vector'].apply(combine_vectors).reset_index()

    # 5. DIVIDE: 2025 vs. History
    current_sig = seasonal_signatures[
        (seasonal_signatures['year'] == 2025) & 
        (seasonal_signatures['state_fips'] == state_fips) &
        (seasonal_signatures['county'] == county_name.upper())
    ]
    
    if current_sig.empty:
        return "No 2025 data found for this window."

    current_vector = current_sig['embedding_vector'].values[0].reshape(1, -1)
    
    history_sigs = seasonal_signatures[
        (seasonal_signatures['year'] < 2025)].copy()

    # 6. COMPARE: Calculate Similarity
    history_vectors = np.stack(history_sigs['embedding_vector'].values)
    history_sigs['similarity'] = cosine_similarity(current_vector, history_vectors)[0]

    # 7. MATCH: Join with your Yield CSV
    if yield_df is None:
        yield_df = pd.read_csv(f"Hackathon2026/data/raw/all_states_corn_yield_2005_2024.csv")
    
    result = history_sigs.sort_values('similarity', ascending=False).head(5)
    final_output = result.merge(yield_df[['year', 'state_fips', 'county', 'yield_bu_acre']], 
                               on=['year', 'state_fips', 'county'])

    return final_output[['year', 'similarity', 'yield_bu_acre']]