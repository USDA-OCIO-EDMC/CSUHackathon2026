"""
Weather + CDL corn yield prediction pipeline. No Prithvi, no SageMaker.
Runs in ~20 minutes.

Feature vector per county/year: [weather_18 + cdl_corn_fraction_1] = 19-dim

Steps:
  1. Load 2005-2024 yield data from S3
  2. Fetch all county centroids from Census Bureau Gazetteer
  3. Fetch 18-dim weather in parallel via NASA POWER point API
  4. Extract CDL corn fraction per county from S3 tiffs
  5. Train XGBoost on 2015-2022, validate on 2023-2024
  6. Predict 2025 — county level and state aggregate
  7. Save to S3
"""

import io
import sys
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import numpy as np
import pandas as pd
import xgboost as xgb
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform
from rasterio.io import MemoryFile
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics import mean_absolute_error

load_dotenv(find_dotenv(usecwd=False), override=True)
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_utils import load_all_states, STATES

GSOD_BUCKET_NAME = "noaa-gsod-pds"   # AWS Open Data — no rate limits

BUCKET         = "cornsight-data"
TARGET_FIPS    = {"IA": "19", "CO": "08", "WI": "55", "MO": "29", "NE": "31"}
FORECAST_DATES = ["aug1", "sep1", "oct1", "final"]

# Month indices within the 18-dim weather vector (ALL_MONTHS = [5,6,7,8,9,10])
_ALL_MONTHS = [5, 6, 7, 8, 9, 10]
_ACTIVE_MONTHS = {
    "aug1":  {5, 6, 7},
    "sep1":  {5, 6, 7, 8},
    "oct1":  {5, 6, 7, 8, 9},
    "final": {5, 6, 7, 8, 9, 10},
}


# ---------------------------------------------------------------------------
# County centroids — Census Bureau Gazetteer
# ---------------------------------------------------------------------------

def fetch_county_centroids(yield_df):
    """
    Download the Census Gazetteer county file, filter to our 5 states,
    and keep only counties that appear in the yield data.

    Returns dict: { state_abbr: [(fips5, county_name, lat, lon), ...] }
    """
    url = (
        "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
        "2023_Gazetteer/2023_Gaz_counties_national.zip"
    )
    print("  Downloading Census Gazetteer...", end=" ", flush=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        fname = next(n for n in zf.namelist() if n.endswith(".txt"))
        with zf.open(fname) as f:
            gaz = pd.read_csv(f, sep="\t", dtype={"GEOID": str})

    # Column names sometimes have trailing whitespace
    gaz.columns = gaz.columns.str.strip()

    print(f"{len(gaz)} counties loaded")

    # Counties that actually appear in historical yield data (per state)
    yield_counties = (
        yield_df
        .assign(
            state_fips=yield_df["state_fips"].astype(str).str.zfill(2),
            county_upper=yield_df["county_name"].str.upper().str.strip(),
        )
        .groupby("state_fips")["county_upper"]
        .apply(set)
        .to_dict()
    )

    fips2_to_abbr = {v: k for k, v in TARGET_FIPS.items()}
    centroids = {abbr: [] for abbr in TARGET_FIPS}

    for _, row in gaz.iterrows():
        fips5  = str(row["GEOID"]).zfill(5)
        fips2  = fips5[:2]
        if fips2 not in fips2_to_abbr:
            continue
        abbr = fips2_to_abbr[fips2]

        # Gazetteer has "Story County" — strip suffix to match yield data "STORY"
        raw_name    = str(row["NAME"]).strip()
        county_name = raw_name.upper().removesuffix(" COUNTY").strip()

        # Only include counties present in yield data
        if county_name not in yield_counties.get(fips2, set()):
            continue

        centroids[abbr].append((
            fips5,
            county_name,
            float(row["INTPTLAT"]),
            float(row["INTPTLONG"]),
        ))

    for abbr, counties in centroids.items():
        print(f"  {abbr}: {len(counties)} counties")

    return centroids


# ---------------------------------------------------------------------------
# Weather — NOAA GSOD on AWS Open Data (s3://noaa-gsod-pds)
# No rate limits; direct S3 reads; TMAX, TMIN, PRCP per station/day
# ---------------------------------------------------------------------------

GSOD_BUCKET = "noaa-gsod-pds"
_GSOD_STATIONS = None   # loaded once, cached in memory


def _load_gsod_stations():
    """
    Download NOAA ISD station inventory from NOAA HTTPS (small CSV, ~14 MB).
    Returns DataFrame indexed for fast nearest-station lookup.
    """
    global _GSOD_STATIONS
    if _GSOD_STATIONS is not None:
        return _GSOD_STATIONS

    cache = Path("/tmp/isd_history.csv")
    if not cache.exists():
        print("  Downloading ISD station inventory...", end=" ", flush=True)
        url  = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
        print("done")

    df = pd.read_csv(cache, dtype={"USAF": str, "WBAN": str})
    df = df.dropna(subset=["LAT", "LON"])
    df["lat"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["lon"] = pd.to_numeric(df["LON"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    # S3 key = USAF (6 chars) + WBAN (5 chars), no separator
    df["s3_stem"] = df["USAF"].str.zfill(6) + df["WBAN"].str.zfill(5)
    _GSOD_STATIONS = df.reset_index(drop=True)
    return _GSOD_STATIONS


def _nearest_station_key(lat, lon, stations, year):
    """Return the S3 key stem for the nearest station that has data for year."""
    lats  = stations["lat"].values
    lons  = stations["lon"].values
    dists = (lats - lat) ** 2 + (lons - lon) ** 2
    # Walk outward until we find a station file that exists for this year
    order = np.argsort(dists)
    s3    = boto3.client("s3", region_name="us-east-1")
    for idx in order[:20]:           # check at most 20 nearest
        stem = stations.iloc[idx]["s3_stem"]
        key  = f"{year}/{stem}.csv"
        try:
            s3.head_object(Bucket=GSOD_BUCKET, Key=key)
            return key
        except Exception:
            continue
    return None


def _gsod_to_vec(df_station):
    """Convert a GSOD station DataFrame (May–Oct) to an 18-dim weather vector."""
    df = df_station.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df[df["DATE"].dt.month.between(5, 10)]

    for col in ["MAX", "MIN", "PRCP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # GSOD missing-data flags
    df["MAX"]  = df["MAX"].where(df["MAX"]  < 9000.0)
    df["MIN"]  = df["MIN"].where(df["MIN"]  < 9000.0)
    df["PRCP"] = df["PRCP"].where(df["PRCP"] < 99.0)

    # °F → °C, inches → mm
    df["TMAX_C"]  = (df["MAX"]  - 32.0) * 5.0 / 9.0
    df["TMIN_C"]  = (df["MIN"]  - 32.0) * 5.0 / 9.0
    df["PRCP_MM"] = df["PRCP"]  * 25.4
    df["month"]   = df["DATE"].dt.month

    vector = []
    for mo in _ALL_MONTHS:
        sub = df[df["month"] == mo]
        txv = sub["TMAX_C"].dropna().values
        tnv = sub["TMIN_C"].dropna().values
        prv = sub["PRCP_MM"].dropna().values
        n   = min(len(txv), len(tnv))
        if n > 0:
            txv, tnv  = txv[:n], tnv[:n]
            mean_temp = (np.mean(txv) + np.mean(tnv)) / 2.0
            gdd_sum   = sum(max(0.0, (tx + tn) / 2.0 - 10.0)
                            for tx, tn in zip(txv, tnv))
        else:
            mean_temp = gdd_sum = 0.0
        total_prec = float(np.sum(prv)) if len(prv) > 0 else 0.0
        vector.extend([float(mean_temp), total_prec, float(gdd_sum)])

    return np.array(vector, dtype=np.float32)


def _fetch_gsod(lat, lon, year):
    """Return an 18-dim weather vector for (lat, lon, year) using NOAA GSOD on S3."""
    stations = _load_gsod_stations()
    key      = _nearest_station_key(lat, lon, stations, year)
    if key is None:
        raise RuntimeError(f"No GSOD station found near ({lat:.2f}, {lon:.2f}) for {year}")

    s3  = boto3.client("s3", region_name="us-east-1")
    obj = s3.get_object(Bucket=GSOD_BUCKET, Key=key)
    df  = pd.read_csv(io.BytesIO(obj["Body"].read()))
    return _gsod_to_vec(df)


def _worker(args):
    fips, name, state_abbr, state_fips2, lat, lon, year = args
    try:
        vec = _fetch_gsod(lat, lon, year)
        return (fips, name, state_abbr, str(state_fips2).zfill(2), int(year), vec)
    except Exception as e:
        print(f"\n    WARNING: {name} {state_abbr} {year}: {e}")
        return None


def fetch_weather_features(county_centroids, years, max_workers=20):
    """
    Parallel Open-Meteo fetch — 18-dim weather vector per county/year.
    Returns DataFrame: [year, county, fips, state, state_fips, weather_vec]
    """
    years = list(years)
    tasks = [
        (fips, name, state_abbr, TARGET_FIPS[state_abbr], lat, lon, year)
        for state_abbr, counties in county_centroids.items()
        for fips, name, lat, lon in counties
        for year in years
    ]
    total   = len(tasks)
    done    = 0
    records = []
    print(f"  {total} calls ({max_workers} workers)...", end=" ", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            result = fut.result()
            done  += 1
            if done % 200 == 0:
                print(f"{done}/{total}", end=" ", flush=True)
            if result is not None:
                fips, name, state_abbr, state_fips2, year, vec = result
                records.append({
                    "year":        year,
                    "county":      name,
                    "fips":        fips,
                    "state":       state_abbr,
                    "state_fips":  state_fips2,
                    "weather_vec": vec,
                })
    print("done")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CDL helpers
# ---------------------------------------------------------------------------

def _load_cdl_from_s3(state_fips2, year):
    s3  = boto3.client("s3")
    key = f"raw/cdl/{state_fips2}/{year}.tif"
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        with MemoryFile(obj["Body"].read()) as mf:
            with mf.open() as src:
                return src.read(1), src.profile.copy()
    except Exception:
        return None, None


def _corn_fraction(corn_mask, profile, lat, lon, buffer_deg=0.5):
    dst_crs = profile["crs"]
    src_crs = CRS.from_epsg(4326)
    xs, ys  = warp_transform(src_crs, dst_crs, [lon], [lat])
    x_c, y_c = xs[0], ys[0]

    res_m  = abs(profile["transform"].a)
    buf_px = max(1, int(buffer_deg * 111_000 / res_m))
    tfm    = profile["transform"]
    col_c  = int((x_c - tfm.c) / tfm.a)
    row_c  = int((y_c - tfm.f) / tfm.e)
    h, w   = corn_mask.shape

    r0, r1 = max(0, row_c - buf_px), min(h, row_c + buf_px)
    c0, c1 = max(0, col_c - buf_px), min(w, col_c + buf_px)
    if r0 >= r1 or c0 >= c1:
        return 0.0

    window = corn_mask[r0:r1, c0:c1]
    return float(window.sum()) / window.size


def fetch_cdl_features(county_centroids, years):
    """
    Load CDL tiffs from S3 and extract per-county corn fraction.
    Returns dict: {(state_abbr, year, fips): corn_fraction}
    """
    results = {}
    for state_abbr, counties in county_centroids.items():
        fips2 = TARGET_FIPS[state_abbr]
        for year in years:
            cdl_year = min(year, 2024)          # 2025 uses 2024 CDL as proxy
            mask, profile = _load_cdl_from_s3(fips2, cdl_year)
            if mask is None:
                for fips, name, lat, lon in counties:
                    results[(state_abbr, year, fips)] = 0.0
                continue
            for fips, name, lat, lon in counties:
                results[(state_abbr, year, fips)] = _corn_fraction(
                    mask, profile, lat, lon
                )
    return results


# ---------------------------------------------------------------------------
# Merge / train / predict
# ---------------------------------------------------------------------------

def zero_mask_vec(vec_18, forecast_date):
    """Return a copy of an 18-dim weather vector with future months zeroed out."""
    active = _ACTIVE_MONTHS[forecast_date]
    out = vec_18.copy()
    for i, m in enumerate(_ALL_MONTHS):
        if m not in active:
            out[i * 3:(i + 1) * 3] = 0.0
    return out


def build_embedding_df(weather_df, cdl_dict, forecast_date="final"):
    """Append CDL corn fraction to (masked) weather vector → 19-dim embedding_vector."""
    rows = []
    for _, row in weather_df.iterrows():
        cf  = cdl_dict.get((row["state"], row["year"], row["fips"]), 0.0)
        vec = zero_mask_vec(row["weather_vec"], forecast_date)
        vec = np.append(vec, cf).astype(np.float32)
        rows.append({**row.to_dict(), "embedding_vector": vec})
    return pd.DataFrame(rows)


def merge_with_yields(feature_df, yield_df):
    feat = feature_df.copy()
    yld  = yield_df.copy()

    feat["year"]       = feat["year"].astype(int)
    feat["state_fips"] = feat["state_fips"].astype(str).str.zfill(2)
    feat["county"]     = feat["county"].str.upper().str.strip()

    yld["year"]       = yld["year"].astype(int)
    yld["state_fips"] = yld["state_fips"].astype(str).str.zfill(2)
    yld["county"]     = yld["county_name"].str.upper().str.strip()

    return feat.merge(
        yld[["year", "county", "state_fips", "yield_bu_acre"]],
        on=["year", "county", "state_fips"],
        how="inner",
    )


def train_xgb(X, y, seed=42):
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        objective="reg:squarederror", random_state=seed,
    )
    model.fit(X, y)
    return model


def bootstrap_predictions(X_train, y_train, X_pred, n=100):
    """
    Bootstrap n models and return array of shape (n, n_pred_rows).
    Used to compute prediction intervals (cone of uncertainty).
    """
    rng   = np.random.default_rng(42)
    preds = np.zeros((n, len(X_pred)), dtype=np.float32)
    for i in range(n):
        idx = rng.integers(0, len(X_train), size=len(X_train))
        m   = train_xgb(X_train[idx], y_train[idx], seed=int(i))
        preds[i] = m.predict(X_pred)
    return preds


def state_aggregate_with_ci(pred_df, boot_preds, cdl_dict, year=2025):
    """
    CDL-weighted state averages for mean, p10, p25, p75, p90.
    boot_preds: (n_bootstrap, n_counties)
    """
    pred_df = pred_df.reset_index(drop=True).copy()
    pred_df["corn_frac"] = pred_df.apply(
        lambda r: cdl_dict.get((r["state"], year, r["fips"]), 1.0), axis=1
    )

    rows = []
    for state, grp in pred_df.groupby("state"):
        idx  = grp.index.tolist()
        w    = grp["corn_frac"].clip(lower=1e-6).values
        if w.sum() == 0:
            w = np.ones_like(w)

        # Point estimate
        mean_pred = np.average(grp["predicted_yield_bu_acre"].values, weights=w)

        # Bootstrap distribution of weighted averages
        boot_state = np.average(boot_preds[:, idx], weights=w, axis=1)
        p10, p25, p75, p90 = np.percentile(boot_state, [10, 25, 75, 90])

        rows.append({
            "state":       state,
            "predicted_bu_acre":      round(mean_pred, 1),
            "p10_bu_acre": round(p10, 1),
            "p25_bu_acre": round(p25, 1),
            "p75_bu_acre": round(p75, 1),
            "p90_bu_acre": round(p90, 1),
        })

    return pd.DataFrame(rows).sort_values("state").reset_index(drop=True)


def save_all_results(county_all, state_all):
    s3 = boto3.client("s3")

    # Parquet of full county detail
    buf = io.BytesIO()
    county_all.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key="processed/predictions/2025_all_dates_county.parquet", Body=buf.read())

    for df, name in [
        (county_all, "2025_all_dates_county.csv"),
        (state_all,  "2025_all_dates_state.csv"),
    ]:
        buf2 = io.StringIO()
        df.to_csv(buf2, index=False)
        s3.put_object(Bucket=BUCKET, Key=f"processed/predictions/{name}", Body=buf2.getvalue())

    print(f"  s3://{BUCKET}/processed/predictions/2025_all_dates_county.{{parquet,csv}}")
    print(f"  s3://{BUCKET}/processed/predictions/2025_all_dates_state.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Weather + CDL Corn Yield Prediction — 2025 (all dates)")
    print("=" * 60)

    # 1. Yield data
    print("\n[1/7] Loading yield data from S3...")
    yield_df = load_all_states(BUCKET)
    yield_df["year"] = yield_df["year"].astype(int)
    print(f"  {len(yield_df):,} records  ({yield_df['year'].min()}–{yield_df['year'].max()})")

    # 2. County centroids from Census Gazetteer
    print("\n[2/7] Fetching county centroids...")
    county_centroids = fetch_county_centroids(yield_df)
    total_counties = sum(len(v) for v in county_centroids.values())
    print(f"  {total_counties} total counties across 5 states")

    CACHE_HIST = Path("/tmp/weather_hist_cache.parquet")
    CACHE_PRED = Path("/tmp/weather_pred_cache.parquet")

    # 3. Historical weather (2015-2024) — always fetch "final" (all months)
    #    Other forecast dates are derived by zero-masking in build_embedding_df.
    if CACHE_HIST.exists():
        print("\n[3/7] Loading historical weather from cache...")
        hist_weather = pd.read_parquet(CACHE_HIST)
        hist_weather["weather_vec"] = hist_weather["weather_vec"].apply(np.array)
    else:
        print("\n[3/7] Fetching historical weather (2015–2024, final)...")
        hist_weather = fetch_weather_features(county_centroids, range(2015, 2025))
        _save = hist_weather.copy()
        _save["weather_vec"] = _save["weather_vec"].apply(lambda v: v.tolist())
        _save.to_parquet(CACHE_HIST, index=False)
    print(f"  {len(hist_weather)} records")

    # 4. 2025 weather
    if CACHE_PRED.exists():
        print("\n[4/7] Loading 2025 weather from cache...")
        pred_weather = pd.read_parquet(CACHE_PRED)
        pred_weather["weather_vec"] = pred_weather["weather_vec"].apply(np.array)
    else:
        print("\n[4/7] Fetching 2025 weather (final)...")
        pred_weather = fetch_weather_features(county_centroids, [2025])
        _save = pred_weather.copy()
        _save["weather_vec"] = _save["weather_vec"].apply(lambda v: v.tolist())
        _save.to_parquet(CACHE_PRED, index=False)
    if pred_weather.empty:
        print("ERROR: No 2025 weather data. Exiting.")
        sys.exit(1)
    print(f"  {len(pred_weather)} records")

    # 5. CDL corn fraction
    all_years = list(range(2015, 2025)) + [2025]
    print("\n[5/7] Loading CDL corn fractions from S3...")
    cdl = fetch_cdl_features(county_centroids, all_years)
    print(f"  {len(cdl)} county/year CDL values")

    # 6. Train one XGBoost per forecast date + bootstrap uncertainty
    print("\n[6/7] Training models for all 4 forecast dates (100-sample bootstrap)...")
    county_rows = []
    state_rows  = []

    for fdate in FORECAST_DATES:
        print(f"\n  [{fdate}]", end=" ", flush=True)

        hist_df = build_embedding_df(hist_weather, cdl, forecast_date=fdate)
        pred_df = build_embedding_df(pred_weather, cdl, forecast_date=fdate)

        merged   = merge_with_yields(hist_df, yield_df)
        if merged.empty:
            print("  WARNING: merge empty — skipping.")
            continue

        train_df = merged[merged["year"] <= 2022]
        val_df   = merged[merged["year"].isin([2023, 2024])]
        print(f"train={len(train_df)} val={len(val_df)}", end=" ", flush=True)

        X_train = np.stack(train_df["embedding_vector"].values)
        y_train = train_df["yield_bu_acre"].astype(float).values
        X_pred  = np.stack(pred_df["embedding_vector"].values)

        # Point model
        model = train_xgb(X_train, y_train)
        if not val_df.empty:
            X_val = np.stack(val_df["embedding_vector"].values)
            mae   = mean_absolute_error(val_df["yield_bu_acre"].astype(float).values,
                                        model.predict(X_val))
            print(f"MAE={mae:.1f}", end=" ", flush=True)

        # Bootstrap
        print("bootstrap...", end=" ", flush=True)
        boot = bootstrap_predictions(X_train, y_train, X_pred, n=100)

        pred_df = pred_df.copy()
        pred_df["predicted_yield_bu_acre"] = np.round(model.predict(X_pred), 1)
        pred_df["p10_yield"]               = np.round(np.percentile(boot, 10, axis=0), 1)
        pred_df["p25_yield"]               = np.round(np.percentile(boot, 25, axis=0), 1)
        pred_df["p75_yield"]               = np.round(np.percentile(boot, 75, axis=0), 1)
        pred_df["p90_yield"]               = np.round(np.percentile(boot, 90, axis=0), 1)
        pred_df["forecast_date"]           = fdate

        county_rows.append(pred_df[[
            "forecast_date", "state", "county", "fips",
            "predicted_yield_bu_acre", "p10_yield", "p25_yield", "p75_yield", "p90_yield",
        ]])

        s_df = state_aggregate_with_ci(pred_df, boot, cdl)
        s_df["forecast_date"] = fdate
        state_rows.append(s_df)
        print("done")

    county_all = pd.concat(county_rows, ignore_index=True).sort_values(
        ["forecast_date", "state", "county"]
    )
    state_all = pd.concat(state_rows, ignore_index=True).sort_values(
        ["forecast_date", "state"]
    )

    # 7. Display and save
    print("\n[7/7] Results:")
    print("\n── 2025 State Forecasts with Cone of Uncertainty (bu/acre) ──")
    for fdate in FORECAST_DATES:
        sub = state_all[state_all["forecast_date"] == fdate]
        print(f"\n  {fdate.upper()}")
        print(sub[["state", "predicted_bu_acre",
                    "p10_bu_acre", "p25_bu_acre",
                    "p75_bu_acre", "p90_bu_acre"]].to_string(index=False))

    print("\n  Saving to S3...")
    save_all_results(county_all, state_all)
    print("\nDone.")
