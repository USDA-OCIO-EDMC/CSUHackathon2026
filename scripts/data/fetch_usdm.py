"""Pull US Drought Monitor county-week percent-area-by-category, 2005-2025.

USDM publishes a free CSV API for county-level drought stats:
  https://usdmdataservices.unl.edu/api/CountyStatistics/GetDroughtSeverityStatisticsByAreaPercent

Output: data/raw/usdm/county_weekly.parquet
   columns: fips, state, valid_start, valid_end, none, d0, d1, d2, d3, d4, dsci

`dsci` (Drought Severity & Coverage Index) = 1*D0 + 2*D1 + 3*D2 + 4*D3 + 5*D4 — single
scalar drought feature that's the workhorse for analog-year matching.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "usdm"
OUT.mkdir(parents=True, exist_ok=True)

API = ("https://usdmdataservices.unl.edu/api/CountyStatistics/"
       "GetDroughtSeverityStatisticsByAreaPercent")
STATE_FP = {"IA": "19", "CO": "08", "WI": "55", "MO": "29", "NE": "31"}
START = date(2005, 1, 4)                                          # USDM weeks start Tuesday
END = date(2025, 12, 30)


def fetch_state(state_abbr: str) -> pd.DataFrame:
    params = {
        "aoi": state_abbr,
        "startdate": START.strftime("%-m/%-d/%Y"),
        "enddate": END.strftime("%-m/%-d/%Y"),
        "statisticsType": "1",                                     # 1 = percent area
    }
    r = requests.get(API, params=params, timeout=300)
    r.raise_for_status()
    if not r.text.strip():
        print(f"  USDM empty response for {state_abbr} — skipping")
        return pd.DataFrame()
    import io
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        return df
    df = df.rename(columns={
        "FIPS": "fips", "ValidStart": "valid_start", "ValidEnd": "valid_end",
        "None": "none", "D0": "d0", "D1": "d1", "D2": "d2", "D3": "d3", "D4": "d4",
    })
    for c in ["none", "d0", "d1", "d2", "d3", "d4"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["dsci"] = df["d0"] + 2 * df["d1"] + 3 * df["d2"] + 4 * df["d3"] + 5 * df["d4"]
    df["state"] = state_abbr
    df["valid_start"] = pd.to_datetime(df["valid_start"])
    df["valid_end"] = pd.to_datetime(df["valid_end"])
    return df[["fips", "state", "valid_start", "valid_end",
               "none", "d0", "d1", "d2", "d3", "d4", "dsci"]]


def main() -> None:
    frames = []
    for s in tqdm(list(STATE_FP), desc="USDM states"):
        frames.append(fetch_state(s))
    frames = [f for f in frames if not f.empty]
    if not frames:
        sys.exit("USDM: all states returned empty — API may be down")
    df = pd.concat(frames, ignore_index=True)
    out = OUT / "county_weekly.parquet"
    df.to_parquet(out, index=False)
    print(f"→ {out}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
