"""Ensemble inference: average predictions from v3 + v3b multimodal models on 2025 chips,
apply per-state bias correction from 2022-2024 hindcast, write final forecast.

Output: reports/forecasts/yield_with_uncertainty_2025.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import zarr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.train_v3 import PrithviMultiModal, load_tabular, TAB_COLS

PRITHVI_CKPT = (Path.home() / ".cache/huggingface/hub"
                / "models--ibm-nasa-geospatial--Prithvi-EO-2.0-600M-TL"
                / "snapshots/3d72adc6dfc4cc3862bf4f41da8c83db267193c9"
                / "Prithvi_EO_V2_600M_TL.pt")

CKPT_PATHS = [
    ROOT / "models/checkpoints/prithvi_yield_v3/best.pt",
    ROOT / "models/checkpoints/prithvi_yield_v3b/best.pt",
]

OUT_PATH   = ROOT / "reports/forecasts/yield_with_uncertainty_2025.parquet"
BATCH_SIZE = 24
STATES_MAP = {"19": "IA", "08": "CO", "55": "WI", "29": "MO", "31": "NE"}
NASS_ACTUALS = {
    2022: {"IA": 202, "CO": 155, "WI": 172, "MO": 135, "NE": 181},
    2023: {"IA": 191, "CO": 158, "WI": 176, "MO": 168, "NE": 183},
    2024: {"IA": 201, "CO": 162, "WI": 179, "MO": 164, "NE": 195},
}


def load_models(device: torch.device) -> list[nn.Module]:
    models = []
    for ckpt_path in CKPT_PATHS:
        if not ckpt_path.exists():
            print(f"  skip {ckpt_path} (not found)")
            continue
        m = PrithviMultiModal(n_tab=len(TAB_COLS)).to(device)
        ckpt = torch.load(str(ckpt_path), map_location=device)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        models.append(m)
        print(f"  loaded {ckpt_path.name}: epoch {ckpt['epoch']}, val_rmse={ckpt['val_rmse']:.2f}")
    return models


def gather_chips(year: int, ck: str = "aug_01") -> list[dict]:
    rows = []
    for zpath in (ROOT / "data/processed/chips").rglob(f"{year}/{ck}.zarr"):
        try:
            z    = zarr.open(str(zpath), mode="r")
            n    = z["chips"].shape[0]
            fips = dict(z.attrs)["fips"]
            for ci in range(n):
                rows.append({"zpath": str(zpath), "idx": ci, "fips": fips})
        except Exception:
            pass
    return rows


@torch.no_grad()
def ensemble_predict(models: list[nn.Module], chip_rows: list[dict],
                     tab_lookup: dict, device: torch.device, year: int) -> pd.DataFrame:
    county_preds: dict[str, list[float]] = {}
    n_batches = (len(chip_rows) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(chip_rows), BATCH_SIZE):
        batch = chip_rows[i : i + BATCH_SIZE]
        imgs, tabs = [], []
        for r in batch:
            z    = zarr.open(r["zpath"], mode="r")
            chip = z["chips"][r["idx"]].astype(np.float32) / 10000.0
            chip = np.transpose(chip, (1, 0, 2, 3))
            imgs.append(torch.from_numpy(chip.copy()))
            tab_vec = tab_lookup.get((r["fips"], year), np.zeros(len(TAB_COLS), dtype=np.float32))
            tabs.append(torch.from_numpy(np.asarray(tab_vec, dtype=np.float32)))
        x   = torch.stack(imgs).to(device)
        t   = torch.stack(tabs).to(device)
        # ensemble = average of model predictions
        preds_list = [m(x, t).float().cpu().numpy() for m in models]
        preds      = np.mean(preds_list, axis=0)
        for r, p in zip(batch, preds):
            county_preds.setdefault(r["fips"], []).append(float(p))
        if (i // BATCH_SIZE + 1) % 20 == 0:
            print(f"    batch {i // BATCH_SIZE + 1}/{n_batches}", flush=True)

    state_buckets: dict[str, list[float]] = {}
    for fips, vals in county_preds.items():
        state = STATES_MAP.get(fips[:2])
        if state:
            state_buckets.setdefault(state, []).append(np.mean(vals))
    return {st: float(np.mean(v)) for st, v in state_buckets.items()}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading models...")
    models = load_models(device)
    if not models:
        raise RuntimeError("No checkpoints found")
    print(f"Ensemble size: {len(models)}")

    tab = load_tabular()
    tab_lookup = {(r["fips"], r["year"]): np.array([r[c] for c in TAB_COLS], dtype=np.float32)
                  for _, r in tab.iterrows()}

    # --- hindcast 2022-2024 for bias correction ---
    print("\n=== HINDCAST 2022-2024 (per-state bias) ===")
    errors: dict[str, list[float]] = {}
    for year in [2022, 2023, 2024]:
        print(f"\nYear {year}...")
        chips = gather_chips(year)
        if not chips:
            continue
        print(f"  {len(chips)} chips")
        state_preds = ensemble_predict(models, chips, tab_lookup, device, year)
        actuals     = NASS_ACTUALS[year]
        for st in sorted(state_preds):
            if st in actuals:
                err = state_preds[st] - actuals[st]
                errors.setdefault(st, []).append(err)
                print(f"  {st}: pred={state_preds[st]:.1f}  actual={actuals[st]}  err={err:+.1f}")

    # median bias for robustness
    bias = {st: float(np.median(errs)) for st, errs in errors.items()}
    print("\n── Per-state bias (median) ──")
    for st, b in bias.items():
        print(f"  {st}: {b:+.1f}  → correction {-b:+.1f}")

    # --- forecast 2025 ---
    print("\n=== 2025 FORECAST ===")
    chips_25 = gather_chips(2025)
    print(f"  {len(chips_25)} chips")
    state_preds_25 = ensemble_predict(models, chips_25, tab_lookup, device, 2025)

    # --- uncertainty from analog years (NASS deviation from trend) ---
    sy = pd.read_parquet(ROOT / "data/raw/nass/yield_state.parquet")
    sy = sy[(sy["statisticcat_desc"] == "YIELD") & (sy["unit_desc"] == "BU / ACRE")]
    sy["value"] = pd.to_numeric(sy["Value"], errors="coerce")

    rows = []
    print(f"\n  {'State':5} {'Raw':>7} {'Corrected':>10} {'p10':>7} {'p90':>7}")
    print("  " + "-" * 50)
    for st, raw in sorted(state_preds_25.items()):
        corrected = raw - bias.get(st, 0.0)
        # cone from historical fractional deviations from trend
        hist = (sy[sy["state_alpha"] == st].set_index("year")["value"]
                .sort_index().dropna())
        hist = hist[hist.index < 2025]
        cone = {}
        if len(hist) >= 5:
            xs   = hist.index.values.astype(float)
            cf   = np.polyfit(xs, hist.values, 1)
            tr   = lambda y: cf[0] * y + cf[1]                # noqa: E731
            devs = np.array([(v - tr(y)) / tr(y) for y, v in hist.items()])
            for p in (10, 25, 50, 75, 90):
                cone[f"p{p}"] = float(corrected * (1 + np.quantile(devs, p / 100.0)))
        else:
            cone = {f"p{p}": corrected for p in (10, 25, 50, 75, 90)}

        rows.append({
            "state":      st,
            "checkpoint": "aug_01",
            "point":      corrected,
            "raw_model":  raw,
            "bias":       bias.get(st, 0.0),
            **cone,
        })
        print(f"  {st:5} {raw:7.1f} {corrected:10.1f} {cone['p10']:7.1f} {cone['p90']:7.1f}")

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
