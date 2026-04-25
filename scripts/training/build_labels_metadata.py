"""Build the two parquet files the training DataModule reads.

  data/processed/labels/county_yield.parquet     fips, year, yield_bu_acre
  data/processed/labels/chip_metadata.parquet    fips, year, checkpoint, lat, lon, dates, n_chips, chip_path

Run after fetch_nass.py and export_hls_chips.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
import zarr

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs" / "project.yaml").read_text())
OUT = ROOT / "data" / "processed" / "labels"
OUT.mkdir(parents=True, exist_ok=True)


def build_labels() -> Path:
    src = ROOT / "data" / "raw" / "nass" / "yield_county.parquet"
    df = pd.read_parquet(src)
    df = df[(df["statisticcat_desc"] == "YIELD") & (df["unit_desc"] == "BU / ACRE")]
    fips_col = "county_ansi"
    state_col = "state_fips_code"
    df["fips"] = df[state_col].astype(str).str.zfill(2) + df[fips_col].astype(str).str.zfill(3)
    out = df[["fips", "year", "Value"]].rename(columns={"Value": "yield_bu"})
    out = out.dropna().drop_duplicates(["fips", "year"])
    path = OUT / "county_yield.parquet"
    out.to_parquet(path, index=False)
    print(f"→ {path}  ({len(out):,} rows)")
    return path


def build_metadata() -> Path:
    chip_root = ROOT / "data" / "processed" / "chips"
    rows = []
    for zpath in chip_root.rglob("*.zarr"):
        try:
            z = zarr.open(zpath, mode="r")
            attrs = dict(z.attrs)
            n = z["chips"].shape[0]
            lon, lat = attrs["centroid_lonlat"]
            rows.append({
                "fips": attrs["fips"],
                "state": attrs["state"],
                "year": int(attrs["year"]),
                "checkpoint": attrs["checkpoint"],
                "lat": float(lat),
                "lon": float(lon),
                "dates": attrs["dates"],
                "n_chips": int(n),
                "chip_path": str(zpath),
            })
        except Exception as e:
            print(f"  skip {zpath}: {e}")
    df = pd.DataFrame(rows)
    path = OUT / "chip_metadata.parquet"
    df.to_parquet(path, index=False)
    print(f"→ {path}  ({len(df):,} county-year-checkpoints)")
    return path


def main() -> None:
    build_labels()
    build_metadata()


if __name__ == "__main__":
    main()
