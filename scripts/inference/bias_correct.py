"""Compute per-state bias from 2022-2024 hindcast, apply to 2025 predictions.

Steps:
1. Run model on all 2022, 2023, 2024 chips → state-level predictions
2. Compare vs NASS actuals → per-state mean error
3. Apply correction to yield_with_uncertainty_2025.parquet
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

CKPT_PATH    = ROOT / "models" / "checkpoints" / "prithvi_yield" / "best.pt"
PRITHVI_CKPT = (Path.home()
                / ".cache/huggingface/hub"
                / "models--ibm-nasa-geospatial--Prithvi-EO-2.0-600M-TL"
                / "snapshots/3d72adc6dfc4cc3862bf4f41da8c83db267193c9"
                / "Prithvi_EO_V2_600M_TL.pt")
FORECAST_PATH = ROOT / "reports" / "forecasts" / "yield_with_uncertainty_2025.parquet"

EMBED_DIM  = 1280
HIDDEN_DIM = 512
BATCH_SIZE = 16
STATES_MAP = {"19": "IA", "08": "CO", "55": "WI", "29": "MO", "31": "NE"}

# NASS state-level actuals bu/ac
NASS_ACTUALS = {
    2022: {"IA": 202, "CO": 155, "WI": 172, "MO": 135, "NE": 181},
    2023: {"IA": 191, "CO": 158, "WI": 176, "MO": 168, "NE": 183},
    2024: {"IA": 201, "CO": 162, "WI": 179, "MO": 164, "NE": 195},
}


class PrithviYieldModel(nn.Module):
    UNFREEZE_BLOCKS = 4

    def __init__(self):
        super().__init__()
        import terratorch.models.backbones.prithvi_vit as pv
        self.backbone = pv.prithvi_eo_v2_600_tl(
            pretrained=True, num_frames=3, in_chans=6,
            ckpt_path=str(PRITHVI_CKPT),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats  = self.backbone(x)
        tokens = feats[-1][:, 1:, :]
        pooled = tokens.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def load_model(device: torch.device) -> PrithviYieldModel:
    model = PrithviYieldModel().to(device)
    ckpt  = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_rmse={ckpt['val_rmse']:.2f})")
    return model


def gather_chips(year: int, checkpoint: str = "aug_01") -> list[dict]:
    chip_root = ROOT / "data" / "processed" / "chips"
    rows = []
    for zpath in chip_root.rglob(f"{year}/{checkpoint}.zarr"):
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
def predict_state(model: PrithviYieldModel, chip_rows: list[dict],
                  device: torch.device) -> dict[str, float]:
    county_preds: dict[str, list[float]] = {}
    n_batches = (len(chip_rows) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(chip_rows), BATCH_SIZE):
        batch = chip_rows[i : i + BATCH_SIZE]
        tensors = []
        for r in batch:
            z    = zarr.open(r["zpath"], mode="r")
            chip = z["chips"][r["idx"]].astype(np.float32) / 10000.0
            chip = np.transpose(chip, (1, 0, 2, 3))
            tensors.append(torch.from_numpy(chip.copy()))
        x     = torch.stack(tensors).to(device)
        preds = model(x).cpu().numpy()
        for r, p in zip(batch, preds):
            county_preds.setdefault(r["fips"], []).append(float(p))
        if (i // BATCH_SIZE + 1) % 20 == 0:
            print(f"    batch {i // BATCH_SIZE + 1}/{n_batches}", flush=True)

    # county mean → state weighted mean
    state_buckets: dict[str, list[float]] = {}
    for fips, vals in county_preds.items():
        state = STATES_MAP.get(fips[:2])
        if state:
            state_buckets.setdefault(state, []).append(np.mean(vals))
    return {st: float(np.mean(v)) for st, v in state_buckets.items()}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)

    # ── hindcast 2022–2024 ─────────────────────────────────────────────────
    errors: dict[str, list[float]] = {}   # state → list of (pred - actual)
    for year in [2022, 2023, 2024]:
        print(f"\nHindcast {year}...")
        chips = gather_chips(year)
        print(f"  {len(chips)} chips")
        state_preds = predict_state(model, chips, device)
        actuals     = NASS_ACTUALS[year]
        print(f"  {'State':6} {'Pred':>8} {'Actual':>8} {'Error':>8}")
        for st in sorted(state_preds):
            if st in actuals:
                err = state_preds[st] - actuals[st]
                errors.setdefault(st, []).append(err)
                print(f"  {st:6} {state_preds[st]:8.1f} {actuals[st]:8.0f} {err:+8.1f}")

    # ── per-state bias ─────────────────────────────────────────────────────
    print("\n── Bias summary (mean error over 2022–2024) ──")
    bias = {}
    for st, errs in errors.items():
        bias[st] = float(np.mean(errs))
        print(f"  {st}: mean error = {bias[st]:+.1f} bu/ac  "
              f"(correction = {-bias[st]:+.1f})")

    # ── apply correction to 2025 ───────────────────────────────────────────
    df = pd.read_parquet(FORECAST_PATH)
    print("\n── 2025 predictions before/after correction ──")
    print(f"  {'State':6} {'Before':>8} {'Correction':>12} {'After':>8}")
    for col in ["point", "p10", "p25", "p50", "p75", "p90"]:
        if col in df.columns:
            df[col] = df.apply(
                lambda r: r[col] - bias.get(r["state"], 0.0), axis=1
            )
    for r in df.itertuples():
        corr = -bias.get(r.state, 0.0)
        before = r.point + bias.get(r.state, 0.0)   # undo to show original
        print(f"  {r.state:6} {before:8.1f} {corr:+12.1f} {r.point:8.1f}")

    df.to_parquet(FORECAST_PATH, index=False)
    print(f"\n→ Updated {FORECAST_PATH}")


if __name__ == "__main__":
    main()
