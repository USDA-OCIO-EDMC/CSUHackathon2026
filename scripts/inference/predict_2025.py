"""Run the trained Prithvi-EO-2.0-600M-TL + LoRA head on 2025 county chips,
aggregate to state with CDL-corn-acreage weights, and write point forecasts.

Output: reports/forecasts/point_forecasts_2025.parquet
   columns: state, checkpoint, point_forecast (bu/ac), n_counties, n_chips

The analog-year cone is added downstream by scripts/training/analog_year_uncertainty.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import zarr
from torch.utils.data import DataLoader

from scripts.training.dataset import CornYieldChipDataset, Sample, _date_triplet

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs" / "project.yaml").read_text())
CHECKPOINTS = CFG["project"]["forecast_checkpoints"]
STATES = CFG["project"]["states"]
TARGET = CFG["project"]["target_year"]


def load_model(ckpt_path: Path):
    """Load the TerraTorch-trained Lightning checkpoint with LoRA head."""
    from terratorch.tasks import PixelwiseRegressionTask
    model = PixelwiseRegressionTask.load_from_checkpoint(str(ckpt_path), map_location="cuda")
    model.eval()
    return model


def gather_samples(year: int, checkpoint: str) -> list[Sample]:
    chip_root = ROOT / "data" / "processed" / "chips"
    samples: list[Sample] = []
    for zpath in chip_root.rglob(f"{year}/{checkpoint}.zarr"):
        z = zarr.open(zpath, mode="r")
        attrs = dict(z.attrs)
        n = z["chips"].shape[0]
        lon, lat = attrs["centroid_lonlat"]
        for ci in range(n):
            samples.append(Sample(
                chip_path=zpath, chip_idx=ci,
                lat=float(lat), lon=float(lon),
                dates=attrs["dates"],
                yield_bu=float("nan"),
                fips=attrs["fips"],
                year=int(attrs["year"]),
            ))
    return samples


def load_acreage_weights() -> dict[tuple[str, int], float]:
    """County-level corn planted acres for the most recent year as state-aggregation weights."""
    path = ROOT / "data" / "raw" / "nass" / "yield_county.parquet"
    df = pd.read_parquet(path)
    if "ACRES" not in df.columns:
        return {}                                                    # equal weights downstream
    latest = df.sort_values("year").drop_duplicates(["state_fips_code", "county_ansi"], keep="last")
    latest["fips"] = (latest["state_fips_code"].astype(str).str.zfill(2)
                      + latest["county_ansi"].astype(str).str.zfill(3))
    return dict(zip(latest["fips"], pd.to_numeric(latest["ACRES"], errors="coerce").fillna(0)))


@torch.no_grad()
def predict_chips(model, samples: list[Sample], batch_size: int = 32) -> pd.DataFrame:
    ds = CornYieldChipDataset(samples, chips_per_county=10**9, augment=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    rows = []
    device = next(model.parameters()).device
    for batch in dl:
        out = model({
            "image": batch["image"].to(device, non_blocking=True),
            "location": batch["location"].to(device, non_blocking=True),
            "time": batch["time"].to(device, non_blocking=True),
        })
        preds = out.detach().float().cpu().numpy().reshape(-1)
        for fips, year, p in zip(batch["fips"], batch["year"].tolist(), preds):
            rows.append({"fips": str(fips), "year": int(year), "chip_pred": float(p)})
    return pd.DataFrame(rows)


def aggregate_to_state(chip_df: pd.DataFrame, weights: dict[tuple[str, int], float]) -> pd.DataFrame:
    chip_df["state"] = chip_df["fips"].str[:2].map({
        "19": "IA", "08": "CO", "55": "WI", "29": "MO", "31": "NE",
    })
    county = chip_df.groupby(["state", "fips"], as_index=False)["chip_pred"].mean() \
                    .rename(columns={"chip_pred": "county_pred"})
    if weights:
        county["w"] = county["fips"].map(weights).fillna(0)
        county.loc[county["w"] == 0, "w"] = 1.0                      # don't drop weightless counties
    else:
        county["w"] = 1.0

    def _wmean(g: pd.DataFrame) -> float:
        return float(np.average(g["county_pred"], weights=g["w"]))

    state = county.groupby("state").apply(_wmean).reset_index(name="point_forecast")
    state["n_counties"] = county.groupby("state")["fips"].nunique().values
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to Lightning .ckpt")
    ap.add_argument("--year", type=int, default=TARGET)
    ap.add_argument("--out", default=str(ROOT / "reports" / "forecasts" / "point_forecasts_2025.parquet"))
    args = ap.parse_args()

    model = load_model(Path(args.ckpt))
    weights = load_acreage_weights()

    all_state = []
    for ck_name in CHECKPOINTS:
        samples = gather_samples(args.year, ck_name)
        if not samples:
            print(f"  no chips for {args.year}/{ck_name}, skipping")
            continue
        chip_df = predict_chips(model, samples)
        state_df = aggregate_to_state(chip_df, weights)
        state_df["checkpoint"] = ck_name
        state_df["n_chips"] = len(samples)
        all_state.append(state_df)
        print(f"  {ck_name}: {len(samples)} chips → {len(state_df)} states")

    out = pd.concat(all_state, ignore_index=True)
    out = out[out["state"].isin(STATES)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"→ {args.out}  ({len(out)} state-checkpoints)")


if __name__ == "__main__":
    main()
