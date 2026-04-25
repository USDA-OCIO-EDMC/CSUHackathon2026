"""Lightning DataModule for county-year HLS chip stacks → corn yield regression.

Each training sample = (chip[3, 6, 224, 224], lat, lon, date_ints, yield_bu_acre).
Chips are sampled randomly within each county for training (so one county-year supplies
many training samples per epoch); for validation we use a fixed deterministic subset.

The TL variant of Prithvi expects:
  - imagery:  (B, T=3, C=6, H=224, W=224)  uint16/float (we divide by 10000)
  - location: (B, 2)  lat, lon in degrees
  - time:     (B, T=3, 3)  year, day_of_year, hour (hour=0 for daily comp)

Yield labels: county-year corn-grain yield in bu/acre, joined from NASS QuickStats.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset


def _date_triplet(iso: str) -> tuple[int, int, int]:
    d = datetime.fromisoformat(iso)
    return d.year, d.timetuple().tm_yday, 0


@dataclass
class Sample:
    chip_path: Path
    chip_idx: int
    lat: float
    lon: float
    dates: list[str]
    yield_bu: float
    fips: str
    year: int


class CornYieldChipDataset(Dataset):
    def __init__(self, samples: list[Sample], chips_per_county: int, augment: bool):
        self.samples = samples
        self.chips_per_county = chips_per_county
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        z = zarr.open(s.chip_path, mode="r")
        chip = z["chips"][s.chip_idx].astype(np.float32) / 10000.0   # (3, 6, 224, 224)
        if self.augment:
            if np.random.rand() < 0.5:
                chip = chip[:, :, :, ::-1].copy()
            if np.random.rand() < 0.5:
                chip = chip[:, :, ::-1, :].copy()
            k = np.random.randint(0, 4)
            if k:
                chip = np.rot90(chip, k=k, axes=(2, 3)).copy()
        date_arr = np.array([_date_triplet(d) for d in s.dates], dtype=np.int32)  # (3, 3)
        return {
            "image": torch.from_numpy(chip),                          # (T, C, H, W)
            "location": torch.tensor([s.lat, s.lon], dtype=torch.float32),
            "time": torch.from_numpy(date_arr),                       # (T, 3)
            "label": torch.tensor(s.yield_bu, dtype=torch.float32),
            "fips": s.fips,
            "year": s.year,
        }


class CornYieldChipDataModule(LightningDataModule):
    def __init__(
        self,
        chip_root: str,
        labels_path: str,
        metadata_path: str,
        holdout_year: int = 2024,
        val_year: int = 2023,
        batch_size: int = 16,
        num_workers: int = 8,
        chips_per_county_train: int = 8,
        chips_per_county_val: int = 16,
        augmentations: list[str] | None = None,
        normalize: dict | None = None,
        use_temporal_location: bool = True,
    ):
        super().__init__()
        self.chip_root = Path(chip_root)
        self.labels_path = Path(labels_path)
        self.metadata_path = Path(metadata_path)
        self.holdout_year = holdout_year
        self.val_year = val_year
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chips_per_train = chips_per_county_train
        self.chips_per_val = chips_per_county_val
        self.augment = bool(augmentations)

    def setup(self, stage: str | None = None) -> None:
        labels = pd.read_parquet(self.labels_path)                    # fips, year, yield_bu
        meta = pd.read_parquet(self.metadata_path)                    # fips, year, checkpoint, lat, lon, dates, n_chips, chip_path

        df = meta.merge(labels, on=["fips", "year"], how="inner")

        train_df = df[~df["year"].isin([self.holdout_year, self.val_year])]
        val_df = df[df["year"] == self.val_year]
        test_df = df[df["year"] == self.holdout_year]

        self.train = self._expand(train_df, self.chips_per_train)
        self.val = self._expand(val_df, self.chips_per_val)
        self.test = self._expand(test_df, self.chips_per_val)

    @staticmethod
    def _expand(df: pd.DataFrame, chips_per: int) -> list[Sample]:
        out: list[Sample] = []
        for r in df.itertuples():
            n = min(int(r.n_chips), chips_per)
            idxs = np.random.choice(int(r.n_chips), size=n, replace=False) if r.n_chips > 0 else []
            for ci in idxs:
                out.append(Sample(
                    chip_path=Path(r.chip_path),
                    chip_idx=int(ci),
                    lat=float(r.lat),
                    lon=float(r.lon),
                    dates=list(r.dates),
                    yield_bu=float(r.yield_bu),
                    fips=str(r.fips),
                    year=int(r.year),
                ))
        return out

    def _loader(self, ds_samples: list[Sample], shuffle: bool, augment: bool) -> DataLoader:
        ds = CornYieldChipDataset(ds_samples, self.chips_per_train, augment)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def train_dataloader(self): return self._loader(self.train, True, self.augment)
    def val_dataloader(self):   return self._loader(self.val, False, False)
    def test_dataloader(self):  return self._loader(self.test, False, False)
