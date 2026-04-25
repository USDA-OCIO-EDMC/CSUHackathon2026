"""Inference + uncertainty for 2025 corn yield forecast.

1. Loads our trained PrithviYieldModel checkpoint
2. Runs it on all available 2025 county chips
3. Aggregates chip predictions to state level (acreage-weighted)
4. Builds a yield uncertainty cone from NASS historical deviations from trend
5. Writes reports/forecasts/yield_with_uncertainty_2025.parquet

Columns: state, checkpoint, point, p10, p25, p50, p75, p90, analog_years
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

CKPT_PATH    = ROOT / "models" / "checkpoints" / "prithvi_yield_v2" / "best.pt"
PRITHVI_CKPT = (Path.home()
                / ".cache/huggingface/hub"
                / "models--ibm-nasa-geospatial--Prithvi-EO-2.0-600M-TL"
                / "snapshots/3d72adc6dfc4cc3862bf4f41da8c83db267193c9"
                / "Prithvi_EO_V2_600M_TL.pt")
OUT_PATH   = ROOT / "reports" / "forecasts" / "yield_with_uncertainty_2025.parquet"
EMBED_DIM  = 1280
HIDDEN_DIM = 512
BATCH_SIZE = 16
ANALOG_K   = 5
STATES_MAP = {"19": "IA", "08": "CO", "55": "WI", "29": "MO", "31": "NE"}


# ── model (must match train.py exactly) ───────────────────────────────────────

class PrithviYieldModel(nn.Module):
    UNFREEZE_BLOCKS = 8

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


# ── inference ─────────────────────────────────────────────────────────────────

def gather_chips(year: int, checkpoint: str) -> list[dict]:
    chip_root = ROOT / "data" / "processed" / "chips"
    rows = []
    for zpath in chip_root.rglob(f"{year}/{checkpoint}.zarr"):
        try:
            z     = zarr.open(str(zpath), mode="r")
            chips = z["chips"]
            attrs = dict(z.attrs)
            fips  = attrs["fips"]
            for ci in range(chips.shape[0]):
                rows.append({"zpath": str(zpath), "idx": ci, "fips": fips})
        except Exception as e:
            print(f"  skip {zpath}: {e}")
    return rows


@torch.no_grad()
def predict(model: PrithviYieldModel, chip_rows: list[dict], device: torch.device) -> pd.DataFrame:
    results = []
    n_batches = (len(chip_rows) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(chip_rows), BATCH_SIZE):
        batch = chip_rows[i : i + BATCH_SIZE]
        print(f"  batch {i // BATCH_SIZE + 1}/{n_batches}", flush=True)
        tensors = []
        for r in batch:
            z    = zarr.open(r["zpath"], mode="r")
            chip = z["chips"][r["idx"]].astype(np.float32) / 10000.0  # (T, C, H, W)
            chip = np.transpose(chip, (1, 0, 2, 3))                   # (C, T, H, W)
            tensors.append(torch.from_numpy(chip.copy()))
        x    = torch.stack(tensors).to(device)
        preds = model(x).cpu().numpy()
        for r, p in zip(batch, preds):
            results.append({"fips": r["fips"], "pred": float(p)})
    return pd.DataFrame(results)


def aggregate_state(pred_df: pd.DataFrame, acreage: dict[str, float]) -> pd.DataFrame:
    pred_df["state"] = pred_df["fips"].str[:2].map(STATES_MAP)
    pred_df = pred_df.dropna(subset=["state"])
    county  = pred_df.groupby(["state", "fips"])["pred"].mean().reset_index()
    county["w"] = county["fips"].map(acreage).fillna(1.0)
    county.loc[county["w"] <= 0, "w"] = 1.0

    rows = []
    for state, g in county.groupby("state"):
        rows.append({
            "state":          state,
            "point_forecast": float(np.average(g["pred"], weights=g["w"])),
            "n_counties":     len(g),
        })
    return pd.DataFrame(rows)


def load_acreage() -> dict[str, float]:
    df = pd.read_parquet(ROOT / "data" / "raw" / "nass" / "yield_county.parquet")
    if "ACRES HARVESTED" in df.columns:
        acre_col = "ACRES HARVESTED"
    else:
        candidates = [c for c in df.columns if "acre" in c.lower() or "ACRE" in c]
        acre_col = candidates[0] if candidates else None

    if acre_col is None:
        return {}
    latest = (df.sort_values("year")
                .drop_duplicates(["state_fips_code", "county_ansi"], keep="last"))
    latest["fips"] = (latest["state_fips_code"].astype(str).str.zfill(2)
                      + latest["county_ansi"].astype(str).str.zfill(3))
    return dict(zip(latest["fips"],
                    pd.to_numeric(latest[acre_col], errors="coerce").fillna(0)))


# ── uncertainty cone ──────────────────────────────────────────────────────────

def load_nass_state_yields() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "raw" / "nass" / "yield_state.parquet")
    df = df[(df["statisticcat_desc"] == "YIELD") & (df["unit_desc"] == "BU / ACRE")]
    df["value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df[["state_alpha", "year", "value"]].dropna()


def uncertainty_cone(point: float, state: str, yields: pd.DataFrame,
                     target_year: int = 2025, k: int = ANALOG_K) -> dict:
    sy = (yields[yields["state_alpha"] == state]
          .set_index("year")["value"]
          .sort_index())
    hist = sy[sy.index < target_year].dropna()
    if len(hist) < 5:
        return {"p10": point, "p25": point, "p50": point,
                "p75": point, "p90": point, "analog_years": []}

    # Linear trend over history
    xs = hist.index.values.astype(float)
    coeffs = np.polyfit(xs, hist.values, 1)
    trend  = lambda y: coeffs[0] * y + coeffs[1]  # noqa: E731

    # Fractional deviation from trend per year
    deviations = {yr: (val - trend(yr)) / trend(yr)
                  for yr, val in hist.items()}

    # Analog years = K closest to current trend deviation (use random subset for hackathon)
    # Simple: pick K years with smallest absolute deviation (near-average years) as base pool
    sorted_yrs = sorted(deviations, key=lambda y: abs(deviations[y]))
    analog_yrs = sorted_yrs[:k]
    devs = np.array([deviations[y] for y in analog_yrs])

    return {
        "p10": float(point * (1 + np.quantile(devs, 0.10))),
        "p25": float(point * (1 + np.quantile(devs, 0.25))),
        "p50": float(point * (1 + np.quantile(devs, 0.50))),
        "p75": float(point * (1 + np.quantile(devs, 0.75))),
        "p90": float(point * (1 + np.quantile(devs, 0.90))),
        "analog_years": analog_yrs,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = load_model(device)
    acreage = load_acreage()
    yields  = load_nass_state_yields()

    all_rows = []
    for checkpoint in ["aug_01"]:          # extend when more checkpoints have chips
        print(f"\nCheckpoint: {checkpoint}")
        chip_rows = gather_chips(2025, checkpoint)
        print(f"  {len(chip_rows)} chips across "
              f"{len({r['fips'] for r in chip_rows})} counties")
        if not chip_rows:
            continue

        pred_df  = predict(model, chip_rows, device)
        state_df = aggregate_state(pred_df, acreage)

        for r in state_df.itertuples():
            cone = uncertainty_cone(r.point_forecast, r.state, yields)
            all_rows.append({
                "state":      r.state,
                "checkpoint": checkpoint,
                "point":      r.point_forecast,
                **cone,
                "n_counties": r.n_counties,
            })
            print(f"  {r.state}: {r.point_forecast:.1f} bu/ac  "
                  f"[{cone['p10']:.0f}–{cone['p90']:.0f}]")

    out = pd.DataFrame(all_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\n→ {OUT_PATH}  ({len(out)} state-checkpoints)")


if __name__ == "__main__":
    main()
