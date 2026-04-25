import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Knowledge windows: which months are observable BEFORE the forecast date
DATE_MAP = {
    "aug1":  [5, 6, 7],
    "sep1":  [5, 6, 7, 8],
    "oct1":  [5, 6, 7, 8, 9],
    "final": [5, 6, 7, 8, 9, 10],
}

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _normalize_feature_df(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    feature_fusion.py emits columns: [year, month, county, fips, state, embedding_vector, ...]
    NASS yield_df uses:              [year, county_name, state_fips, yield_bu_acre, ...]
    Coerce feature_df to the NASS schema so they can be merged.
    """
    df = feature_df.copy()
    if 'county_name' not in df.columns and 'county' in df.columns:
        df = df.rename(columns={'county': 'county_name'})
    if 'state_fips' not in df.columns and 'fips' in df.columns:
        df['state_fips'] = df['fips'].astype(str).str.zfill(5).str[:2]
    return df


def _normalize_yield_df(yield_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure yield_df has canonical, string-typed join columns."""
    df = yield_df.copy()
    if 'state_fips' in df.columns:
        df['state_fips'] = df['state_fips'].astype(str).str.zfill(2)
    if 'county_name' in df.columns:
        df['county_name'] = df['county_name'].astype(str).str.upper()
    return df


def get_analog_years(
    state_fips,
    county_name,
    forecast_date,
    feature_df=None,
    yield_df=None,
    query_year=2025,
    top_k=5,
):
    """
    Find historical years whose pre-forecast feature signature most resembles
    the query_year for a given county.

    Returns DataFrame: [year, similarity, yield_bu_acre]
    """
    # 1. Knowledge window
    target_months = DATE_MAP.get(str(forecast_date).lower())
    if target_months is None:
        raise ValueError(
            f"forecast_date must be one of {list(DATE_MAP)}, got {forecast_date!r}"
        )

    # 2. Load feature space if not provided
    if feature_df is None:
        feature_df = pd.read_parquet(
            PROJECT_ROOT / "data" / "processed" / "master_features.parquet"
        )
    feature_df = _normalize_feature_df(feature_df)

    # Normalize join keys
    state_fips = str(state_fips).zfill(2)
    county_name = str(county_name).upper()
    feature_df['state_fips'] = feature_df['state_fips'].astype(str).str.zfill(2)
    feature_df['county_name'] = feature_df['county_name'].astype(str).str.upper()

    # 3. Filter by knowledge window (no temporal leakage)
    filtered_df = feature_df[feature_df['month'].isin(target_months)]
    if filtered_df.empty:
        raise ValueError(
            f"No feature rows match months {target_months}. "
            f"Available months: {sorted(feature_df['month'].unique())}"
        )

    # 4. One vector per (year, county) — average across window months
    def _combine(group):
        return np.mean(np.stack(group.values), axis=0)

    seasonal_signatures = (
        filtered_df
        .groupby(['year', 'state_fips', 'county_name'])['embedding_vector']
        .apply(_combine)
        .reset_index()
    )

    # 5. Split current vs history
    current_sig = seasonal_signatures[
        (seasonal_signatures['year'] == query_year) &
        (seasonal_signatures['state_fips'] == state_fips) &
        (seasonal_signatures['county_name'] == county_name)
    ]
    if current_sig.empty:
        return f"No {query_year} data found for {county_name}, FIPS {state_fips}, window {forecast_date}."

    current_vector = current_sig['embedding_vector'].values[0].reshape(1, -1)

    history_sigs = seasonal_signatures[
        (seasonal_signatures['year'] < query_year) &
        (seasonal_signatures['state_fips'] == state_fips)
    ].copy()
    if history_sigs.empty:
        return f"No history (year < {query_year}) available for state {state_fips}."

    # 6. Cosine similarity
    history_vectors = np.stack(history_sigs['embedding_vector'].values)
    history_sigs['similarity'] = cosine_similarity(current_vector, history_vectors)[0]

    # 7. Join with yields
    if yield_df is None:
        yield_path = PROJECT_ROOT / "data" / "processed" / "all_states_corn_yield.csv"
        yield_df = pd.read_csv(yield_path)
    yield_df = _normalize_yield_df(yield_df)

    top = history_sigs.sort_values('similarity', ascending=False).head(top_k)
    final_output = top.merge(
        yield_df[['year', 'state_fips', 'county_name', 'yield_bu_acre']],
        on=['year', 'state_fips', 'county_name'],
        how='left',
    )

    return final_output[['year', 'similarity', 'yield_bu_acre']]
