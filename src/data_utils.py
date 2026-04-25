import requests, pandas as pd
import boto3, io

NASS_KEY = "YOUR_API_KEY"  # Register at quickstats.nass.usda.gov
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