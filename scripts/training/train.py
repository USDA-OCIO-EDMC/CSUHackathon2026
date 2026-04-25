"""Train Prithvi-EO-2.0-600M-TL + regression head for county-level corn yield.

Backbone: frozen Prithvi ViT (CLS token, last layer, dim=1280)
Head:     trainable 2-layer MLP → scalar bu/acre

Run:
    python scripts/training/train.py
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
CKPT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--ibm-nasa-geospatial--Prithvi-EO-2.0-600M-TL"
    / "snapshots/3d72adc6dfc4cc3862bf4f41da8c83db267193c9"
    / "Prithvi_EO_V2_600M_TL.pt"
)
EMBED_DIM   = 1280
HIDDEN_DIM  = 512
LR          = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS      = 20
BATCH_SIZE  = 16
NUM_WORKERS = 4
HOLDOUT_YEAR = 2024
VAL_YEAR     = 2023
CHIPS_TRAIN  = 16
CHIPS_VAL    = 32
SEED         = 42
OUT_DIR      = ROOT / "models" / "checkpoints" / "prithvi_yield"
LOG_PATH     = ROOT / "reports" / "training_logs" / "train_log.csv"


# ── data ──────────────────────────────────────────────────────────────────────

class YieldDataset(Dataset):
    def __init__(self, rows: list[dict], augment: bool):
        self.rows    = rows
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.rows[i]
        z = zarr.open(r["chip_path"], mode="r")
        chip = z["chips"][r["chip_idx"]].astype(np.float32) / 10000.0  # (T, C, H, W)
        if self.augment:
            if random.random() < 0.5:
                chip = chip[:, :, :, ::-1].copy()
            if random.random() < 0.5:
                chip = chip[:, :, ::-1, :].copy()
            k = random.randint(0, 3)
            if k:
                chip = np.rot90(chip, k=k, axes=(2, 3)).copy()
        # Backbone expects (C, T, H, W)
        chip = np.transpose(chip, (1, 0, 2, 3))
        return (
            torch.from_numpy(chip.copy()),                     # (6, 3, 224, 224)
            torch.tensor(r["yield_bu"], dtype=torch.float32),
        )


def build_rows(meta: pd.DataFrame, labels: pd.DataFrame, year_filter, chips_per: int) -> list[dict]:
    df = meta.merge(labels, on=["fips", "year"], how="inner")
    if callable(year_filter):
        df = df[df["year"].apply(year_filter)]
    else:
        df = df[df["year"] == year_filter]
    rows: list[dict] = []
    for r in df.itertuples():
        n = min(int(r.n_chips), chips_per)
        idxs = np.random.choice(int(r.n_chips), size=n, replace=False) if n > 0 else []
        for ci in idxs:
            rows.append({
                "chip_path": r.chip_path,
                "chip_idx":  int(ci),
                "yield_bu":  float(r.yield_bu),
                "fips":      r.fips,
                "year":      r.year,
            })
    return rows


# ── model ─────────────────────────────────────────────────────────────────────

class PrithviYieldModel(nn.Module):
    UNFREEZE_BLOCKS = 4  # last N transformer blocks get LR_BACKBONE

    def __init__(self, backbone_ckpt: Path):
        super().__init__()
        import terratorch.models.backbones.prithvi_vit as pv
        self.backbone = pv.prithvi_eo_v2_600_tl(
            pretrained=True,
            num_frames=3,
            in_chans=6,
            ckpt_path=str(backbone_ckpt),
        )
        # Freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        # Unfreeze last 4 blocks + norm
        n = len(self.backbone.blocks)
        for blk in self.backbone.blocks[n - self.UNFREEZE_BLOCKS:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for p in self.backbone.norm.parameters():
            p.requires_grad_(True)

        self.head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, 3, 224, 224)
        feats = self.backbone(x)  # list of (B, 769, 1280)
        tokens = feats[-1][:, 1:, :]  # drop CLS, keep spatial: (B, 768, 1280)
        pooled = tokens.mean(dim=1)    # mean over all patch tokens: (B, 1280)
        return self.head(pooled).squeeze(-1)  # (B,)


# ── training loop ─────────────────────────────────────────────────────────────

def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return math.sqrt(((pred - target) ** 2).mean().item())


def run_epoch(model, loader, opt, device, train: bool):
    model.train(train)
    total_loss = 0.0
    all_pred, all_tgt = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += loss.item() * len(y)
            all_pred.append(pred.detach().cpu())
            all_tgt.append(y.cpu())
    all_pred = torch.cat(all_pred)
    all_tgt  = torch.cat(all_tgt)
    n = len(all_tgt)
    return math.sqrt(total_loss / n), rmse(all_pred, all_tgt)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    meta   = pd.read_parquet(ROOT / "data/processed/labels/chip_metadata.parquet")
    labels = pd.read_parquet(ROOT / "data/processed/labels/county_yield.parquet")

    train_rows = build_rows(meta, labels,
                            lambda y: y not in (HOLDOUT_YEAR, VAL_YEAR),
                            CHIPS_TRAIN)
    val_rows   = build_rows(meta, labels, VAL_YEAR,    CHIPS_VAL)
    test_rows  = build_rows(meta, labels, HOLDOUT_YEAR, CHIPS_VAL)

    print(f"Split → train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)} chips")

    train_loader = DataLoader(YieldDataset(train_rows, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(YieldDataset(val_rows, augment=False),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PrithviYieldModel(CKPT).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable:,}")

    # Differential LR: backbone blocks at 1/10th the head LR
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = list(model.head.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR / 10},
            {"params": head_params,     "lr": LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    log_rows = []
    best_val_rmse = float("inf")
    best_ckpt = OUT_DIR / "best.pt"

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_rmse = run_epoch(model, train_loader, opt, device, train=True)
        va_loss, va_rmse = run_epoch(model, val_loader,   opt, device, train=False)
        sched.step()
        lr_now = sched.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_rmse={tr_rmse:.2f}  val_rmse={va_rmse:.2f}  lr={lr_now:.2e}")

        log_rows.append({"epoch": epoch, "train_rmse": tr_rmse, "val_rmse": va_rmse})
        pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)

        if va_rmse < best_val_rmse:
            best_val_rmse = va_rmse
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_rmse": va_rmse,
            }, best_ckpt)
            print(f"  ↑ new best ({va_rmse:.2f} bu/ac) → {best_ckpt}")

    print(f"\nBest val RMSE: {best_val_rmse:.2f} bu/ac")
    print(f"Checkpoint: {best_ckpt}")

    # Evaluate on holdout
    if test_rows:
        test_loader = DataLoader(YieldDataset(test_rows, augment=False),
                                 batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        _, test_rmse = run_epoch(model, test_loader, None, device, train=False)
        print(f"Holdout {HOLDOUT_YEAR} RMSE: {test_rmse:.2f} bu/ac")


if __name__ == "__main__":
    main()
