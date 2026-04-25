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
    # v3 / v3b dropped — trained on old May-padded chips, out-of-distribution for new multi-temporal chips
    ROOT / "models/checkpoints/prithvi_yield_v4/best.pt",
    ROOT / "models/checkpoints/prithvi_yield_v4_restart/best.pt",
]
USE_TTA = False           # disabled for speed
HINDCAST_SAMPLE = 200     # subsample chips per hindcast year for fast calibration

OUT_PATH   = ROOT / "reports/forecasts/yield_with_uncertainty_2025.parquet"
BATCH_SIZE = 24
STATES_MAP = {"19": "IA", "08": "CO", "55": "WI", "29": "MO", "31": "NE"}
NASS_ACTUALS = {
    2021: {"IA": 199, "CO": 132, "WI": 173, "MO": 165, "NE": 190},   # for prior-year feature
    2022: {"IA": 201, "CO": 122, "WI": 182, "MO": 154, "NE": 172},
    2023: {"IA": 201, "CO": 127, "WI": 169, "MO": 146, "NE": 178},
    2024: {"IA": 212, "CO": 122, "WI": 180, "MO": 182, "NE": 193},
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
        # TTA: average predictions over original + 3 flipped/rotated versions
        if USE_TTA:
            tta_views = [
                x,                                # original
                torch.flip(x, dims=[3]),          # hflip (W axis)
                torch.flip(x, dims=[4]),          # vflip (H axis... actually wait, dims [3,4] map to H,W)
                torch.flip(x, dims=[3, 4]),       # rot180
            ]
            all_preds = []
            for view in tta_views:
                for m in models:
                    all_preds.append(m(view, t).float().cpu().numpy())
            preds = np.mean(all_preds, axis=0)
        else:
            preds_list = [m(x, t).float().cpu().numpy() for m in models]
            preds = np.mean(preds_list, axis=0)
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

    # --- hindcast 2022-2024 to learn calibration ---
    print("\n=== HINDCAST 2022-2024 (calibration with prior-year-yield) ===")
    # Collect (model_pred, prior_year_yield, actual) for each (state, year)
    cal_data: list[tuple[str, int, float, float, float]] = []   # (state, year, pred, prior, actual)
    rng = np.random.default_rng(42)
    for year in [2022, 2023, 2024]:
        print(f"\nYear {year}...")
        chips = gather_chips(year)
        if not chips:
            continue
        # Subsample stratified by state for fast hindcast calibration
        if HINDCAST_SAMPLE and len(chips) > HINDCAST_SAMPLE * 5:
            by_state: dict[str, list] = {}
            for c in chips:
                by_state.setdefault(c["fips"][:2], []).append(c)
            sampled = []
            for st_fips, sts in by_state.items():
                k = min(HINDCAST_SAMPLE, len(sts))
                idxs = rng.choice(len(sts), size=k, replace=False)
                sampled.extend(sts[i] for i in idxs)
            chips = sampled
        print(f"  {len(chips)} chips (sampled)")
        state_preds = ensemble_predict(models, chips, tab_lookup, device, year)
        actuals     = NASS_ACTUALS[year]
        priors      = NASS_ACTUALS[year - 1]
        for st in sorted(state_preds):
            if st in actuals and st in priors:
                p, a, pr = state_preds[st], actuals[st], priors[st]
                cal_data.append((st, year, p, pr, a))
                print(f"  {st}: pred={p:.1f}  prior_yr={pr}  actual={a}  err={p-a:+.1f}")

    # Fit GLOBAL linear calibration: actual = w_p * pred + w_y * prior + b
    # (15 hindcast points, 3 free params — well-conditioned)
    X = np.array([[p, pr] for (_, _, p, pr, _) in cal_data])
    y = np.array([a for (_, _, _, _, a) in cal_data])
    X_aug = np.column_stack([X, np.ones(len(X))])                 # add intercept
    coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    w_pred, w_prior, b_intercept = coeffs
    print(f"\n── Global calibration ──")
    print(f"  actual ≈ {w_pred:.3f} * pred + {w_prior:.3f} * prior_year + {b_intercept:+.2f}")

    # Per-state residual after global calibration (small adjustment)
    state_residuals: dict[str, list[float]] = {}
    for st, _, p, pr, a in cal_data:
        cal_pred = w_pred * p + w_prior * pr + b_intercept
        state_residuals.setdefault(st, []).append(a - cal_pred)
    state_offsets = {st: float(np.median(rs)) for st, rs in state_residuals.items()}
    print(f"\n── Per-state residual offset (after global) ──")
    for st, off in state_offsets.items():
        print(f"  {st}: {off:+.1f}")

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
    print(f"\n  {'State':5} {'Raw':>7} {'Calibrated':>11} {'p10':>7} {'p90':>7}")
    print("  " + "-" * 55)
    priors_25 = NASS_ACTUALS[2024]   # prior year for 2025 = 2024
    for st, raw in sorted(state_preds_25.items()):
        prior = priors_25.get(st, raw)
        # Apply global calibration + per-state residual
        corrected = (w_pred * raw + w_prior * prior + b_intercept
                     + state_offsets.get(st, 0.0))
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
            "state":       st,
            "checkpoint":  "aug_01",
            "point":       corrected,
            "raw_model":   raw,
            "prior_year":  prior,
            "calibration": float(w_pred * raw + w_prior * prior + b_intercept),
            "state_offset": state_offsets.get(st, 0.0),
            **cone,
        })
        print(f"  {st:5} {raw:7.1f} {corrected:10.1f} {cone['p10']:7.1f} {cone['p90']:7.1f}")

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
