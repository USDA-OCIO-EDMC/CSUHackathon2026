all_features = pd.read_parquet("Hackathon2026/data/processed/master_features.parquet")
all_yields = pd.read_csv("Hackathon2026/data/raw/all_states_corn_yield_2005_2024.csv")

target_fips = 19
target_county = "STORY"
prediction_date = "aug1"


analog_results = get_analog_years(
    state_fips=target_fips,
    county_name=target_county,
    forecast_date=prediction_date,
    feature_df=all_features,
    yield_df=all_yields
)

print(f"--- 2025 {target_county} county ({prediction_date} similiatry years ---")
print(analog_results)