"""Pull USDA NASS QuickStats: county+state corn-grain yield and weekly crop progress.

Outputs (parquet under data/raw/nass/):
  yield_county.parquet      county-year corn grain yield, bu/acre, 2005-2024
  yield_state.parquet       state-year corn grain yield, bu/acre, 2005-2024
  crop_progress.parquet     weekly corn progress percentages, 2005-2025
  in_season_forecast.parquet  monthly NASS in-season state yield forecasts, 2005-2024 (benchmark)

Requires: NASS_API_KEY env var (free, instant: https://quickstats.nass.usda.gov/api).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API = "https://quickstats.nass.usda.gov/api/api_GET/"
STATES = ["IOWA", "COLORADO", "WISCONSIN", "MISSOURI", "NEBRASKA"]
OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "nass"
OUT.mkdir(parents=True, exist_ok=True)


def _key() -> str:
    k = os.getenv("NASS_API_KEY")
    if not k:
        sys.exit("NASS_API_KEY missing — request at https://quickstats.nass.usda.gov/api")
    return k


def _get(params: dict) -> list[dict]:
    """Paginated GET. NASS caps at 50k rows; we filter by state and year so this is rarely hit."""
    p = {"key": _key(), "format": "JSON", **params}
    r = requests.get(API, params=p, timeout=120)
    if r.status_code == 413:
        raise RuntimeError(f"NASS row limit hit, narrow the query: {params}")
    if r.status_code == 400:
        print(f"    NASS 400 (no data for these params) — returning empty")
        return []
    r.raise_for_status()
    return r.json().get("data", [])


def yield_query(agg_level: str, state: str) -> list[dict]:
    return _get({
        "source_desc": "SURVEY",
        "commodity_desc": "CORN",
        "statisticcat_desc": "YIELD",
        "unit_desc": "BU / ACRE",
        "util_practice_desc": "GRAIN",
        "agg_level_desc": agg_level,           # COUNTY or STATE
        "state_name": state,
        "year__GE": 2005,
        "year__LE": 2024,
    })


def crop_progress_query(state: str) -> list[dict]:
    """Weekly CORN progress: planted/silking/dough/dent/mature/harvested %."""
    return _get({
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "commodity_desc": "CORN",
        "statisticcat_desc": "PROGRESS",
        "agg_level_desc": "STATE",
        "state_name": state,
        "year__GE": 2005,
        "year__LE": 2025,
    })


def in_season_forecast_query(state: str) -> list[dict]:
    """Monthly NASS in-season state yield forecasts (Aug/Sep/Oct/Nov reports)."""
    return _get({
        "source_desc": "SURVEY",
        "commodity_desc": "CORN",
        "statisticcat_desc": "YIELD",
        "unit_desc": "BU / ACRE",
        "util_practice_desc": "GRAIN",
        "agg_level_desc": "STATE",
        "state_name": state,
        "freq_desc": "MONTHLY",
        "year__GE": 2005,
        "year__LE": 2024,
    })


def to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "Value" in df:
        df["Value"] = pd.to_numeric(df["Value"].str.replace(",", ""), errors="coerce")
    df = df[df["Value"].notna()]                # drops "(D)" suppressed and "(Z)" rounded-to-zero
    if "year" in df:
        df["year"] = df["year"].astype(int)
    return df


def main() -> None:
    print(f"Pulling NASS data for {len(STATES)} states → {OUT}")

    blocks = {
        "yield_county.parquet": ("COUNTY", yield_query),
        "yield_state.parquet":  ("STATE",  yield_query),
        "in_season_forecast.parquet": (None, in_season_forecast_query),
        "crop_progress.parquet": (None, crop_progress_query),
    }

    for fname, (agg, fn) in blocks.items():
        all_rows: list[dict] = []
        for s in STATES:
            print(f"  {fname} :: {s}")
            rows = fn(agg, s) if agg is not None else fn(s)
            all_rows.extend(rows)
            time.sleep(0.3)                     # be polite
        df = to_df(all_rows)
        path = OUT / fname
        df.to_parquet(path, index=False)
        print(f"  → {path}  ({len(df):,} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
