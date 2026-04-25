"""Multi-modal corn yield: Prithvi satellite + weather + soil + drought.

Architecture (late fusion):
  Prithvi backbone → mean pool → 1280-dim
  Tabular features → BatchNorm → 64-dim MLP
  Concat (1344-dim) → regression head → yield bu/ac

Training data: all 3 years (2022-2024), stratified 80/20 county split.
Run:
    python scripts/training/train_v3.py
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

EMBED_DIM       = 1280
TAB_HIDDEN      = 64
HEAD_HIDDEN     = 512
LR_HEAD         = 5e-4
LR_BACKBONE     = 5e-5
WEIGHT_DECAY    = 1e-3
EPOCHS          = 3           # warm restart — short fresh cosine cycle
WARMUP_EPOCHS   = 0           # no warmup — model is already trained
BATCH_SIZE      = 24
NUM_WORKERS     = 2
UNFREEZE_BLOCKS = 4
VAL_FRAC        = 0.20
CHIPS_TRAIN     = 6
CHIPS_VAL       = 12
USE_BF16        = True
HUBER_DELTA     = 10.0
ZERO_THRESH     = 0.50
SEED            = 73          # different seed for restart diversity
OUT_DIR         = ROOT / "models" / "checkpoints" / "prithvi_yield_v4_restart"
LOG_PATH        = ROOT / "reports" / "training_logs" / "train_v4_restart_log.csv"
INIT_CKPT       = ROOT / "models" / "checkpoints" / "prithvi_yield_v4" / "best.pt"

STATES_MAP = {"19": "IA", "08": "CO", "55": "WI", "29": "MO", "31": "NE"}

# Tabular feature columns (filled with 0 if missing)
WEATHER_COLS = ["gdd_acc", "precip_acc", "heat_days", "vpd_jul_mean", "precip_jul"]
DROUGHT_COLS = ["dsci_mean", "dsci_peak", "d2plus_pct", "d3plus_pct"]
SOIL_COLS    = ["bulk_density", "cec", "clay_pct", "sand_pct",
                "soil_organic_carbon", "soil_ph"]
TAB_COLS     = WEATHER_COLS + DROUGHT_COLS + SOIL_COLS   # 15 features total


# ── feature loading ────────────────────────────────────────────────────────────

def load_tabular() -> pd.DataFrame:
    feat_dir = ROOT / "data" / "processed" / "features"

    weather = pd.read_parquet(feat_dir / "weather_features.parquet")
    drought = pd.read_parquet(feat_dir / "drought_features.parquet")
    soil    = pd.read_parquet(feat_dir / "soil_features.parquet")

    weather["fips"] = weather["fips"].astype(str).str.zfill(5)
    drought["fips"] = drought["fips"].astype(str).str.zfill(5)
    soil["fips"]    = soil["fips"].astype(str).str.zfill(5)

    # Weather + drought are per county-year; soil is per county (static)
    tab = weather.merge(drought, on=["fips", "year"], how="outer")
    tab = tab.merge(soil, on="fips", how="left")

    # Z-score normalize per column (fit on 2022-2024 training data)
    for col in TAB_COLS:
        if col not in tab.columns:
            tab[col] = 0.0
        mu  = tab.loc[tab["year"].isin([2022, 2023, 2024]), col].mean()
        std = tab.loc[tab["year"].isin([2022, 2023, 2024]), col].std()
        tab[col] = (tab[col].fillna(mu) - mu) / (std + 1e-6)

    return tab


# ── data ──────────────────────────────────────────────────────────────────────

def chip_is_valid(chip: np.ndarray) -> bool:
    return float((chip.max(axis=(0, 1)) == 0).mean()) < ZERO_THRESH


class YieldDataset(Dataset):
    def __init__(self, rows: list[dict], augment: bool):
        self.rows    = rows
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r    = self.rows[i]
        z    = zarr.open(r["chip_path"], mode="r")
        chip = z["chips"][r["chip_idx"]].astype(np.float32) / 10000.0

        if self.augment:
            if random.random() < 0.5:
                chip = chip[:, :, :, ::-1].copy()
            if random.random() < 0.5:
                chip = chip[:, :, ::-1, :].copy()
            k = random.randint(0, 3)
            if k:
                chip = np.rot90(chip, k=k, axes=(2, 3)).copy()
            scale = np.random.uniform(0.85, 1.15, (1, chip.shape[1], 1, 1)).astype(np.float32)
            chip  = np.clip(chip * scale, 0, 1)

        chip = np.transpose(chip, (1, 0, 2, 3))
        tab  = np.array(r["tab_feats"], dtype=np.float32)
        return (
            torch.from_numpy(chip.copy()),
            torch.from_numpy(tab),
            torch.tensor(r["yield_bu"], dtype=torch.float32),
        )


def stratified_split(meta: pd.DataFrame, labels: pd.DataFrame, tab: pd.DataFrame,
                     val_frac: float, chips_train: int, chips_val: int,
                     seed: int) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    df  = meta.merge(labels, on=["fips", "year"], how="inner")
    df["state"] = df["fips"].str[:2].map(STATES_MAP)
    df = df.dropna(subset=["state"])

    # Merge tabular features
    df = df.merge(tab[["fips", "year"] + TAB_COLS], on=["fips", "year"], how="left")
    for col in TAB_COLS:
        df[col] = df[col].fillna(0.0)

    train_fips, val_fips = set(), set()
    for state, grp in df.groupby("state"):
        fips_unique = grp["fips"].unique()
        rng.shuffle(fips_unique)
        cut = max(1, int(len(fips_unique) * val_frac))
        val_fips.update(fips_unique[:cut])
        train_fips.update(fips_unique[cut:])

    def expand(sub: pd.DataFrame, chips_per: int, quality_check: bool) -> list[dict]:
        rows: list[dict] = []
        for r in sub.itertuples():
            n    = min(int(r.n_chips), chips_per)
            idxs = rng.choice(int(r.n_chips), size=n, replace=False) if n > 0 else []
            tab_vals = [getattr(r, c, 0.0) for c in TAB_COLS]
            for ci in idxs:
                if quality_check:
                    try:
                        z = zarr.open(r.chip_path, mode="r")
                        if not chip_is_valid(z["chips"][ci]):
                            continue
                    except Exception:
                        continue
                rows.append({
                    "chip_path": r.chip_path,
                    "chip_idx":  int(ci),
                    "yield_bu":  float(r.yield_bu),
                    "tab_feats": tab_vals,
                    "fips":      r.fips,
                    "year":      r.year,
                    "state":     r.state,
                })
        return rows

    print("  Building train rows (with quality filter)...", flush=True)
    train_rows = expand(df[df["fips"].isin(train_fips)], chips_train, quality_check=True)
    print("  Building val rows...", flush=True)
    val_rows   = expand(df[df["fips"].isin(val_fips)], chips_val, quality_check=False)
    return train_rows, val_rows


# ── model ─────────────────────────────────────────────────────────────────────

class PrithviMultiModal(nn.Module):
    def __init__(self, n_tab: int):
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

        # Tabular branch
        self.tab_encoder = nn.Sequential(
            nn.BatchNorm1d(n_tab),
            nn.Linear(n_tab, TAB_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(TAB_HIDDEN, TAB_HIDDEN),
            nn.GELU(),
        )

        # Fusion head
        fusion_dim = EMBED_DIM + TAB_HIDDEN
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, HEAD_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(HEAD_HIDDEN // 2, 1),
        )

    def forward(self, img: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        feats  = self.backbone(img)
        tokens = feats[-1][:, 1:, :]
        img_emb = tokens.mean(dim=1)        # (B, 1280)
        tab_emb = self.tab_encoder(tab)     # (B, 64)
        fused   = torch.cat([img_emb, tab_emb], dim=1)
        return self.head(fused).squeeze(-1)


# ── lr schedule ───────────────────────────────────────────────────────────────

def get_lr_scale(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ── training loop ─────────────────────────────────────────────────────────────

def rmse(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    return math.sqrt(((pred - tgt) ** 2).mean().item())


def run_epoch(model, loader, opt, device, train: bool):
    model.train(train)
    total_loss, all_pred, all_tgt = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    autocast_dtype = torch.bfloat16 if USE_BF16 and device.type == "cuda" else torch.float32
    n_batches = len(loader)
    with ctx:
        for batch_i, (img, tab, y) in enumerate(loader):
            if train and batch_i % 20 == 0:
                print(f"    train batch {batch_i}/{n_batches}", flush=True)
            img, tab, y = img.to(device), tab.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=USE_BF16):
                pred = model(img, tab)
                loss = nn.functional.huber_loss(pred, y, delta=HUBER_DELTA)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
            total_loss += loss.item() * len(y)
            all_pred.append(pred.float().detach().cpu())
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
    tab    = load_tabular()
    print(f"Tabular features: {TAB_COLS}")

    print("\nBuilding stratified train/val split...")
    train_rows, val_rows = stratified_split(
        meta, labels, tab, VAL_FRAC, CHIPS_TRAIN, CHIPS_VAL, SEED
    )
    train_df = pd.DataFrame(train_rows)
    val_df   = pd.DataFrame(val_rows)
    print(f"\nSplit → train={len(train_rows)} chips  val={len(val_rows)} chips")
    for st in sorted(STATES_MAP.values()):
        tr = (train_df["state"] == st).sum()
        vl = (val_df["state"] == st).sum()
        print(f"  {st}: train={tr}  val={vl}")

    train_loader = DataLoader(YieldDataset(train_rows, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=False)
    val_loader   = DataLoader(YieldDataset(val_rows, augment=False),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = PrithviMultiModal(n_tab=len(TAB_COLS)).to(device)

    # Warm restart: load v4 best.pt and resume training with fresh LR cycle
    if INIT_CKPT.exists():
        ckpt = torch.load(str(INIT_CKPT), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded warm-start checkpoint: epoch {ckpt['epoch']}, "
              f"val_rmse={ckpt['val_rmse']:.2f}")
    else:
        print(f"⚠ {INIT_CKPT} not found — training from scratch")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable:,}")

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    other_params    = list(model.tab_encoder.parameters()) + list(model.head.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": other_params,    "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    log_rows      = []
    best_val_rmse = float("inf")
    best_ckpt     = OUT_DIR / "best.pt"

    for epoch in range(EPOCHS):
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
                "tab_cols":    TAB_COLS,
                "config": {
                    "unfreeze_blocks": UNFREEZE_BLOCKS,
                    "n_tab":           len(TAB_COLS),
                },
            }, best_ckpt)
            print(f"  ↑ new best ({va_rmse_:.2f} bu/ac) → {best_ckpt}")

    print(f"\nBest val RMSE: {best_val_rmse:.2f} bu/ac → {best_ckpt}")


if __name__ == "__main__":
    main()
