"""Optimized Prithvi-EO-2.0-600M-TL fine-tune for corn yield prediction.

Key improvements over v1:
  - All 3 labelled years (2022+2023+2024) in training, no holdout
  - Stratified 80/20 county split within each state (val sees all states)
  - Unfreeze last 8 backbone blocks (more capacity, more data to support it)
  - Huber loss (delta=10) — robust to cloud/bad-chip outliers
  - LR warmup (3 epochs) then cosine decay
  - Chip quality filter (skip >50% zero pixels = cloud mask)
  - Brightness jitter augmentation
  - Tighter gradient clipping (0.5)

Run:
    python scripts/training/train_v2.py
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import zarr
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
PRITHVI_CKPT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--ibm-nasa-geospatial--Prithvi-EO-2.0-600M-TL"
    / "snapshots/3d72adc6dfc4cc3862bf4f41da8c83db267193c9"
    / "Prithvi_EO_V2_600M_TL.pt"
)

EMBED_DIM      = 1280
HIDDEN_DIM     = 512
LR_HEAD        = 5e-4
LR_BACKBONE    = 5e-5
WEIGHT_DECAY   = 1e-3
EPOCHS         = 35
WARMUP_EPOCHS  = 3
BATCH_SIZE     = 16
NUM_WORKERS    = 4
UNFREEZE_BLOCKS = 8
VAL_FRAC       = 0.20
CHIPS_TRAIN    = 32
CHIPS_VAL      = 64
HUBER_DELTA    = 10.0
ZERO_THRESH    = 0.50   # skip chip if >50% pixels are zero
SEED           = 42
OUT_DIR        = ROOT / "models" / "checkpoints" / "prithvi_yield_v2"
LOG_PATH       = ROOT / "reports" / "training_logs" / "train_v2_log.csv"

STATES_MAP = {"19": "IA", "08": "CO", "55": "WI", "29": "MO", "31": "NE"}


# ── data ──────────────────────────────────────────────────────────────────────

def chip_is_valid(chip: np.ndarray) -> bool:
    """Return False if >ZERO_THRESH fraction of spatial pixels are all-zero."""
    # chip: (T, C, H, W) — check fraction of (H,W) positions that are 0 across all bands
    spatial_max = chip.max(axis=(0, 1))          # (H, W)
    zero_frac   = (spatial_max == 0).mean()
    return float(zero_frac) < ZERO_THRESH


class YieldDataset(Dataset):
    def __init__(self, rows: list[dict], augment: bool):
        self.rows    = rows
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        r    = self.rows[i]
        z    = zarr.open(r["chip_path"], mode="r")
        chip = z["chips"][r["chip_idx"]].astype(np.float32) / 10000.0  # (T, C, H, W)

        if self.augment:
            # Spatial flips + rotation
            if random.random() < 0.5:
                chip = chip[:, :, :, ::-1].copy()
            if random.random() < 0.5:
                chip = chip[:, :, ::-1, :].copy()
            k = random.randint(0, 3)
            if k:
                chip = np.rot90(chip, k=k, axes=(2, 3)).copy()
            # Brightness jitter: scale each band by U(0.85, 1.15)
            scale = np.random.uniform(0.85, 1.15, size=(1, chip.shape[1], 1, 1)).astype(np.float32)
            chip  = np.clip(chip * scale, 0, 1)

        chip = np.transpose(chip, (1, 0, 2, 3))   # → (C, T, H, W)
        return (
            torch.from_numpy(chip.copy()),
            torch.tensor(r["yield_bu"], dtype=torch.float32),
        )


def stratified_split(meta: pd.DataFrame, labels: pd.DataFrame,
                     val_frac: float, chips_train: int, chips_val: int,
                     seed: int) -> tuple[list[dict], list[dict]]:
    """Split counties 80/20 within each state, then expand to chip rows."""
    rng = np.random.default_rng(seed)
    df  = meta.merge(labels, on=["fips", "year"], how="inner")
    df["state"] = df["fips"].str[:2].map(STATES_MAP)
    df = df.dropna(subset=["state"])

    train_fips, val_fips = set(), set()
    for state, grp in df.groupby("state"):
        fips_unique = grp["fips"].unique()
        rng.shuffle(fips_unique)
        cut = max(1, int(len(fips_unique) * val_frac))
        val_fips.update(fips_unique[:cut])
        train_fips.update(fips_unique[cut:])

    def expand(sub: pd.DataFrame, chips_per: int) -> list[dict]:
        rows: list[dict] = []
        for r in sub.itertuples():
            n    = min(int(r.n_chips), chips_per)
            idxs = rng.choice(int(r.n_chips), size=n, replace=False) if n > 0 else []
            for ci in idxs:
                rows.append({
                    "chip_path": r.chip_path,
                    "chip_idx":  int(ci),
                    "yield_bu":  float(r.yield_bu),
                    "fips":      r.fips,
                    "year":      r.year,
                    "state":     r.state,
                })
        return rows

    train_rows = expand(df[df["fips"].isin(train_fips)], chips_train)
    val_rows   = expand(df[df["fips"].isin(val_fips)],   chips_val)

    # Filter bad chips from train (quality check on a sample)
    print("  Filtering low-quality chips...", flush=True)
    clean_train = []
    for r in train_rows:
        try:
            z    = zarr.open(r["chip_path"], mode="r")
            chip = z["chips"][r["chip_idx"]]
            if chip_is_valid(chip):
                clean_train.append(r)
        except Exception:
            pass
    print(f"  Quality filter: {len(train_rows)} → {len(clean_train)} train chips "
          f"({len(train_rows) - len(clean_train)} removed)")
    return clean_train, val_rows


# ── model ─────────────────────────────────────────────────────────────────────

class PrithviYieldModel(nn.Module):
    def __init__(self):
        super().__init__()
        import terratorch.models.backbones.prithvi_vit as pv
        self.backbone = pv.prithvi_eo_v2_600_tl(
            pretrained=True, num_frames=3, in_chans=6,
            ckpt_path=str(PRITHVI_CKPT),
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        n = len(self.backbone.blocks)
        for blk in self.backbone.blocks[n - UNFREEZE_BLOCKS:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for p in self.backbone.norm.parameters():
            p.requires_grad_(True)

        self.head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats  = self.backbone(x)
        tokens = feats[-1][:, 1:, :]
        pooled = tokens.mean(dim=1)
        return self.head(pooled).squeeze(-1)


# ── lr schedule: linear warmup then cosine ────────────────────────────────────

def get_lr_scale(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ── training loop ─────────────────────────────────────────────────────────────

def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return math.sqrt(((pred - target) ** 2).mean().item())


def run_epoch(model, loader, opt, device, train: bool):
    model.train(train)
    total_loss, all_pred, all_tgt = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = nn.functional.huber_loss(pred, y, delta=HUBER_DELTA)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
            total_loss += loss.item() * len(y)
            all_pred.append(pred.detach().cpu())
            all_tgt.append(y.cpu())
    all_pred = torch.cat(all_pred)
    all_tgt  = torch.cat(all_tgt)
    return total_loss / len(all_tgt), rmse(all_pred, all_tgt)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    meta   = pd.read_parquet(ROOT / "data/processed/labels/chip_metadata.parquet")
    labels = pd.read_parquet(ROOT / "data/processed/labels/county_yield.parquet")

    print("Building stratified train/val split...")
    train_rows, val_rows = stratified_split(
        meta, labels, VAL_FRAC, CHIPS_TRAIN, CHIPS_VAL, SEED
    )

    # Print split summary
    train_df = pd.DataFrame(train_rows)
    val_df   = pd.DataFrame(val_rows)
    print(f"\nSplit → train={len(train_rows)} chips  val={len(val_rows)} chips")
    for st in sorted(STATES_MAP.values()):
        tr = (train_df["state"] == st).sum() if "state" in train_df else 0
        vl = (val_df["state"] == st).sum()   if "state" in val_df   else 0
        print(f"  {st}: train={tr}  val={vl}")

    train_loader = DataLoader(YieldDataset(train_rows, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True)
    val_loader   = DataLoader(YieldDataset(val_rows, augment=False),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = PrithviYieldModel().to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable:,}  (backbone blocks {32-UNFREEZE_BLOCKS}–31 + head)")

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = list(model.head.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params,     "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    log_rows      = []
    best_val_rmse = float("inf")
    best_ckpt     = OUT_DIR / "best.pt"

    for epoch in range(EPOCHS):
        # Warmup + cosine LR scaling
        scale = get_lr_scale(epoch, WARMUP_EPOCHS, EPOCHS)
        opt.param_groups[0]["lr"] = LR_BACKBONE * scale
        opt.param_groups[1]["lr"] = LR_HEAD      * scale

        tr_loss, tr_rmse_ = run_epoch(model, train_loader, opt, device, train=True)
        va_loss, va_rmse_ = run_epoch(model, val_loader,   opt, device, train=False)

        lr_now = opt.param_groups[1]["lr"]
        print(f"Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train_rmse={tr_rmse_:.2f}  val_rmse={va_rmse_:.2f}  lr={lr_now:.2e}")

        log_rows.append({"epoch": epoch + 1, "train_rmse": tr_rmse_, "val_rmse": va_rmse_})
        pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)

        if va_rmse_ < best_val_rmse:
            best_val_rmse = va_rmse_
            torch.save({
                "epoch":       epoch + 1,
                "model_state": model.state_dict(),
                "val_rmse":    va_rmse_,
                "config": {
                    "unfreeze_blocks": UNFREEZE_BLOCKS,
                    "embed_dim":       EMBED_DIM,
                    "hidden_dim":      HIDDEN_DIM,
                },
            }, best_ckpt)
            print(f"  ↑ new best ({va_rmse_:.2f} bu/ac) → {best_ckpt}")

    print(f"\nBest val RMSE: {best_val_rmse:.2f} bu/ac")
    print(f"Checkpoint saved → {best_ckpt}")


if __name__ == "__main__":
    main()
