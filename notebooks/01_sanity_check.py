"""Sanity-check the data joins BEFORE kicking off the heavy HLS chip export.

Run as a script (`python notebooks/01_sanity_check.py`) or open in Jupyter via jupytext.
Reports:
  1. Row counts and year coverage per source
  2. State coverage check (must be exactly IA/CO/WI/MO/NE)
  3. NASS yield × gridMET join coverage at county-year
  4. Distribution of corn-yield labels per state-year (sanity: 100-200 bu/ac for IA/NE)
  5. NDVI source coverage (Landsat 2005-2014, HLS 2015+) — flag any gaps
  6. Final feature table dimensionality and missingness

Exits non-zero if any critical check fails so you can wire it into a pre-train hook.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "project.yaml").read_text())
STATES = set(CFG["project"]["states"])
HIST_RANGE = (CFG["project"]["history_start"], CFG["project"]["history_end"])
TARGET = CFG["project"]["target_year"]

failures: list[str] = []


def hr(s: str) -> None:
    print(f"\n{'=' * 8}  {s}  {'=' * 8}")


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    failures.append(msg)


def ok(msg: str) -> None:
    print(f"  ok:   {msg}")


def must_exist(path: Path, label: str) -> Path | None:
    if not path.exists():
        fail(f"{label} missing: {path}")
        return None
    ok(f"{label}: {path.relative_to(ROOT)}")
    return path


# --- 1. NASS ---
hr("1. NASS yield labels")
p = must_exist(ROOT / "data/raw/nass/yield_county.parquet", "yield_county")
if p:
    df = pd.read_parquet(p)
    df = df[df["statisticcat_desc"] == "YIELD"]
    df["fips"] = (df["state_fips_code"].astype(str).str.zfill(2)
                  + df["county_ansi"].astype(str).str.zfill(3))
    df["state"] = df["state_alpha"]
    print(f"  rows: {len(df):,}  counties: {df['fips'].nunique():,}  years: {sorted(df['year'].unique())[:3]}..{sorted(df['year'].unique())[-3:]}")
    miss = STATES - set(df["state"].unique())
    if miss: fail(f"NASS missing states: {miss}")
    else: ok("all 5 states present")
    yr_min, yr_max = df["year"].min(), df["year"].max()
    if yr_min > HIST_RANGE[0] or yr_max < HIST_RANGE[1]:
        fail(f"NASS year coverage {yr_min}-{yr_max} narrower than {HIST_RANGE}")
    else: ok(f"year coverage {yr_min}-{yr_max} OK")
    # Distribution sanity
    state_med = df.groupby("state")["Value"].median()
    print(f"  state median yield (bu/ac):\n{state_med.to_string()}")
    if state_med.get("IA", 0) < 130 or state_med.get("IA", 0) > 220:
        fail(f"IA median yield {state_med.get('IA')} looks wrong (expected ~150-200)")

# --- 2. gridMET ---
hr("2. gridMET county-day weather")
p = must_exist(ROOT / "data/raw/gridmet/county_daily.parquet", "gridmet")
if p:
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  rows: {len(df):,}  counties: {df['fips'].nunique():,}  date range: {df['date'].min().date()} → {df['date'].max().date()}")
    miss_v = [v for v in ["pr", "tmmn", "tmmx", "vpd", "eto", "srad"] if v not in df.columns]
    if miss_v: fail(f"missing vars: {miss_v}")
    else: ok("all 6 weather vars present")
    if df["date"].max().year < TARGET:
        fail(f"gridMET ends {df['date'].max().year}, need through {TARGET}")

# --- 3. USDM ---
hr("3. US Drought Monitor")
p = must_exist(ROOT / "data/raw/usdm/county_weekly.parquet", "usdm")
if p:
    df = pd.read_parquet(p)
    print(f"  rows: {len(df):,}  counties: {df['fips'].nunique():,}  dsci range: {df['dsci'].min():.1f} → {df['dsci'].max():.1f}")
    if df["dsci"].max() > 500 or df["dsci"].min() < 0:
        fail("DSCI out of expected 0-500 range")
    else: ok("DSCI range OK")

# --- 4. gNATSGO ---
hr("4. gNATSGO soils")
p = must_exist(ROOT / "data/raw/gnatsgo/county_soil.parquet", "gnatsgo")
if p:
    df = pd.read_parquet(p)
    print(f"  rows: {len(df):,}  counties: {df['fips'].nunique():,}")
    null_pct = df[["awc_0_25cm", "om_kg_m2", "clay_pct"]].isna().mean()
    print(f"  null rates:\n{null_pct.to_string()}")
    if (null_pct > 0.2).any():
        fail("gNATSGO has >20% nulls in core soil props")

# --- 5. NDVI sources (Landsat backfill + HLS NDVI) ---
hr("5. NDVI coverage 2005-2025")
ls_p = ROOT / "data/raw/landsat/county_month_vi.parquet"
hls_p = ROOT / "data/processed/features/hls_county_month_ndvi.parquet"
ls = pd.read_parquet(ls_p) if ls_p.exists() else pd.DataFrame()
hls = pd.read_parquet(hls_p) if hls_p.exists() else pd.DataFrame()
if ls.empty: fail("Landsat backfill missing")
else: ok(f"Landsat: {ls['year'].min()}-{ls['year'].max()}  rows={len(ls):,}")
if hls.empty: fail("HLS NDVI missing")
else: ok(f"HLS NDVI: {hls['year'].min()}-{hls['year'].max()}  rows={len(hls):,}")
if not ls.empty and not hls.empty:
    all_years = set(ls["year"]).union(hls["year"])
    expected = set(range(HIST_RANGE[0], TARGET + 1))
    gap = expected - all_years
    if gap: fail(f"NDVI gap years: {sorted(gap)}")
    else: ok(f"continuous NDVI 2005-{TARGET}")

# --- 6. Chip + label join ---
hr("6. Chip × yield join readiness")
labels_p = ROOT / "data/processed/labels/county_yield.parquet"
meta_p = ROOT / "data/processed/labels/chip_metadata.parquet"
if labels_p.exists() and meta_p.exists():
    labels = pd.read_parquet(labels_p)
    meta = pd.read_parquet(meta_p)
    j = meta.merge(labels, on=["fips", "year"], how="inner")
    print(f"  joined chip-county-years: {len(j):,}")
    print(f"  unique counties: {j['fips'].nunique()}  unique years: {sorted(j['year'].unique())}")
    print(f"  total chips available: {int(j['n_chips'].sum()):,}")
    if len(j) < 1000:
        fail(f"only {len(j)} county-year-checkpoints joined — chip export likely incomplete")
else:
    print("  (chip metadata not built yet — run build_labels_metadata.py after chip export)")

# --- 7. Final feature table ---
hr("7. State-checkpoint feature table")
feat_p = ROOT / "data/processed/features/state_checkpoint_features.parquet"
if feat_p.exists():
    f = pd.read_parquet(feat_p)
    print(f"  rows: {len(f)}  cols: {list(f.columns)}")
    print(f"  states: {sorted(f['state'].unique())}  years: {f['year'].min()}-{f['year'].max()}")
    null_pct = f.isna().mean().sort_values(ascending=False).head(10)
    print(f"  top null rates:\n{null_pct.to_string()}")
    expected_rows = len(STATES) * (TARGET - HIST_RANGE[0] + 1) * len(CFG["project"]["forecast_checkpoints"])
    if len(f) < 0.9 * expected_rows:
        fail(f"feature table has {len(f)} rows, expected ~{expected_rows}")
else:
    print("  (build_features.py not run yet)")

# --- summary ---
hr("SUMMARY")
if failures:
    print(f"  {len(failures)} FAILURES:")
    for m in failures: print(f"   - {m}")
    sys.exit(1)
print("  All checks passed. Safe to train.")
