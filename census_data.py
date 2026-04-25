from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
from datetime import date

BASE_URL = "https://api.census.gov/data/timeseries/intltrade/imports/porths"

HS6_CODES = [
    "070200", "070700", "070930", "070960",
    "080450", "080510", "080550", "080711", "080719",
    "080720", "080810", "080830", "080910", "080930", "080940",
]

FIELDS = [
    "I_COMMODITY",
    "I_COMMODITY_SDESC",
    "CTY_CODE",
    "CTY_NAME",
    "PORT",
    "PORT_NAME",
    "GEN_VAL_MO",
    "AIR_VAL_MO",
    "AIR_WGT_MO",
    "VES_VAL_MO",
    "VES_WGT_MO",
    "CNT_VAL_MO",
    "CNT_WGT_MO",
    "YEAR",
    "MONTH",
]


def month_range(start_year, start_month, end_year, end_month):
    year, month = start_year, start_month

    while (year, month) <= (end_year, end_month):
        yield f"{year}-{month:02d}"

        month += 1
        if month == 13:
            month = 1
            year += 1


def fetch_one(args):
    period, hs_code, api_key = args

    params = {
        "get": ",".join(FIELDS),
        "time": period,
        "COMM_LVL": "HS6",
        "I_COMMODITY": hs_code,
    }

    if api_key:
        params["key"] = api_key

    try:
        r = requests.get(BASE_URL, params=params, timeout=60)

        if r.status_code == 204:
            return pd.DataFrame()

        if r.status_code != 200:
            print(f"Failed {period}, {hs_code}: {r.status_code} {r.text[:200]}")
            return pd.DataFrame()

        data = r.json()

        if len(data) <= 1:
            return pd.DataFrame()

        return pd.DataFrame(data[1:], columns=data[0])

    except Exception as e:
        print(f"Error {period}, {hs_code}: {e}")
        return pd.DataFrame()


def download_parallel(
    start_year=2020,
    start_month=1,
    end_year=2024,
    end_month=12,
    api_key=None,
    output_csv="us_imports_fruitfly_hs6_by_port_parallel.csv",
    workers=8,
):
    tasks = [
        (period, hs_code, api_key)
        for period in month_range(start_year, start_month, end_year, end_month)
        for hs_code in HS6_CODES
    ]

    frames = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_one, task) for task in tasks]

        for i, future in enumerate(as_completed(futures), start=1):
            df = future.result()

            if not df.empty:
                frames.append(df)

            if i % 25 == 0:
                print(f"Completed {i}/{len(tasks)} requests")

    if not frames:
        raise RuntimeError("No data downloaded.")

    out = pd.concat(frames, ignore_index=True)

    numeric_cols = [
        "GEN_VAL_MO",
        "AIR_VAL_MO",
        "AIR_WGT_MO",
        "VES_VAL_MO",
        "VES_WGT_MO",
        "CNT_VAL_MO",
        "CNT_WGT_MO",
        "YEAR",
        "MONTH",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Saved {len(out):,} rows to {output_csv}")
    return out


if __name__ == "__main__":
    df = download_parallel(
        start_year=2020,
        start_month=1,
        end_year=2025,
        end_month=12,
        workers=8,
        api_key=None,
    )

    print(df.head())