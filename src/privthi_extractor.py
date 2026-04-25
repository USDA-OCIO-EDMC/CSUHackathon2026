"""
Phase 2 — Prithvi-100M Feature Extraction

Extracts 768-dim spatial embeddings from HLS 6-band tiles using the
Prithvi-EO-1.0-100M temporal Vision Transformer (NASA/IBM).

Input from hls_downloader.iter_granules_cloud():
    stacked array: (6, H, W) float32, values in raw HLS units (0-10000)

Prithvi expected input:
    tensor: (B, C, T, H, W)  ->  B=batch, C=6, T=time_steps, H=W=224

Output:
    feature vector: (768,) float32 -- mean-pooled patch embeddings
"""

import io
import sys
import json
from pathlib import Path
import numpy as np
import torch
import boto3
import rasterio

# ---------------------------------------------------------------------------
# Prithvi-100M normalization constants (raw HLS DN units, NOT reflectance!)
# Source: config.json -> pretrained_cfg.mean / .std in the model snapshot.
# Band order: B02 Blue, B03 Green, B04 Red, B05 NIR, B06 SWIR1, B07 SWIR2.
# ---------------------------------------------------------------------------
PRITHVI_MEANS = np.array(
    [775.23, 1080.99, 1228.59, 2497.20, 2204.21, 1610.83],
    dtype=np.float32,
)
PRITHVI_STDS = np.array(
    [1281.53, 1270.03, 1399.48, 1368.34, 1291.68, 1154.51],
    dtype=np.float32,
)

TILE_SIZE = 224     # Prithvi patch grid assumes 224x224 input


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_prithvi(device="cpu"):
    """
    Load Prithvi-EO-1.0-100M.
    Use device='cuda' on EC2 GPU instance.

    The model isn't a HuggingFace transformers model -- it's a custom
    PyTorch nn.Module distributed via HF Hub. We:
      1. snapshot_download the repo (gets prithvi_mae.py + Prithvi_EO_V1_100M.pt)
      2. Import PrithviViT from the bundled prithvi_mae.py
      3. Instantiate it from config.json's pretrained_cfg
      4. Load the .pt state dict (stripping the 'encoder.' prefix from the
         full PrithviMAE checkpoint)
    """
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download("ibm-nasa-geospatial/Prithvi-EO-1.0-100M")

    # Make prithvi_mae.py importable
    if local_dir not in sys.path:
        sys.path.insert(0, local_dir)
    import prithvi_mae  # noqa: E402

    # Read architecture config
    cfg = json.loads((Path(local_dir) / "config.json").read_text(encoding="utf-8"))
    pcfg = cfg["pretrained_cfg"]

    model = prithvi_mae.PrithviViT(
        img_size=pcfg["img_size"],
        patch_size=tuple(pcfg["patch_size"]),
        num_frames=pcfg["num_frames"],
        in_chans=pcfg["in_chans"],
        embed_dim=pcfg["embed_dim"],
        depth=pcfg["depth"],
        num_heads=pcfg["num_heads"],
        mlp_ratio=pcfg["mlp_ratio"],
        encoder_only=True,
    )

    # Load weights -- the .pt file contains the full PrithviMAE state dict.
    # Encoder weights are prefixed with "encoder." -- strip that prefix.
    weights_path = Path(local_dir) / "Prithvi_EO_V1_100M.pt"
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    encoder_state = {
        k[len("encoder."):]: v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    if not encoder_state:
        # Already an encoder-only checkpoint
        encoder_state = state_dict
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")

    model = model.to(device).eval()
    print(f"Prithvi loaded on {device} (PrithviViT, embed_dim={pcfg['embed_dim']})")
    return model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_tile(arr_6hw):
    """
    Normalize a (6, H, W) HLS array for Prithvi input.

    Prithvi was trained on raw HLS DN values (0-10000 range), not on
    [0,1] reflectance, so DO NOT divide by 10000. Just (x - mean) / std
    using the means/stds shipped in the model config.

    Returns (6, H, W) float32.
    """
    arr = arr_6hw.astype(np.float32)
    arr = (arr - PRITHVI_MEANS[:, None, None]) / (PRITHVI_STDS[:, None, None] + 1e-6)
    return arr


def _crop_tiles(arr_6hw, tile_size=TILE_SIZE):
    """
    Slice a (6, H, W) array into non-overlapping (6, tile_size, tile_size) patches.
    Drops partial tiles at the edges.
    Returns list of (6, tile_size, tile_size) arrays.
    """
    _, H, W = arr_6hw.shape
    tiles = []
    for r in range(0, H - tile_size + 1, tile_size):
        for c in range(0, W - tile_size + 1, tile_size):
            tiles.append(arr_6hw[:, r:r + tile_size, c:c + tile_size])
    return tiles


def _crop_tiles_with_mask(arr_6hw, mask_hw, tile_size=TILE_SIZE, min_corn_frac=0.05):
    """
    Like _crop_tiles, but also takes a (H, W) boolean corn mask and:
      - drops tiles whose corn fraction < min_corn_frac
      - zeros out non-corn pixels inside the kept tiles (post-normalization)

    Returns
    -------
    kept_tiles : list of (6, tile_size, tile_size) arrays
    n_total    : int, total candidate tiles before filtering
    """
    _, H, W = arr_6hw.shape
    kept = []
    n_total = 0
    threshold = int(min_corn_frac * tile_size * tile_size)
    for r in range(0, H - tile_size + 1, tile_size):
        for c in range(0, W - tile_size + 1, tile_size):
            n_total += 1
            tile_mask = mask_hw[r:r + tile_size, c:c + tile_size]
            if tile_mask.sum() < threshold:
                continue
            tile = arr_6hw[:, r:r + tile_size, c:c + tile_size].copy()
            # Zero out non-corn pixels (post-normalization, so 0 == band mean)
            tile[:, ~tile_mask] = 0.0
            kept.append(tile)
    return kept, n_total


# ---------------------------------------------------------------------------
# CDL corn mask -> granule grid
# ---------------------------------------------------------------------------

_STATE_FIPS = {"IA": "19", "CO": "08", "WI": "55", "MO": "29", "NE": "31"}
_CORN_CLASS = 1


def load_corn_mask_for_grid(bucket, state_abbr, year, profile):
    """
    Pull the CDL corn raster for (state, year) from S3 and reproject it onto the
    grid described by `profile` (a rasterio profile dict from the HLS read).

    Returns a (H, W) bool array, True == corn.  Returns None if no CDL is
    available (e.g. Colorado pre-2009).  Falls back to most recent prior year
    if the exact year is missing.
    """
    from rasterio.io import MemoryFile
    from rasterio.warp import Resampling, reproject
    import boto3

    fips = _STATE_FIPS[state_abbr]
    s3 = boto3.client("s3")

    # Pick the most recent CDL <= requested year
    candidates = [year] + list(range(year - 1, 2007, -1))
    key = None
    for y in candidates:
        k = f"raw/cdl/{fips}/{y}.tif"
        try:
            s3.head_object(Bucket=bucket, Key=k)
            key = k
            break
        except Exception:
            continue
    if key is None:
        return None

    H, W = profile["height"], profile["width"]
    out = np.zeros((H, W), dtype=np.uint8)
    obj = s3.get_object(Bucket=bucket, Key=key)
    with MemoryFile(obj["Body"].read()) as mf, mf.open() as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            resampling=Resampling.nearest,
        )
    return out == _CORN_CLASS


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features_from_array(model, arr_6hw, max_tiles=None):
    """
    Extract a single 768-dim feature vector from a (6, H, W) HLS granule array.

    The granule is sliced into 224x224 tiles, each run through Prithvi,
    and the patch embeddings are mean-pooled across both spatial patches
    and tiles to produce one vector per granule.

    Parameters
    ----------
    model      : loaded Prithvi model (from load_prithvi)
    arr_6hw    : np.ndarray (6, H, W), raw HLS values 0-10000
    max_tiles  : limit tiles processed (useful for quick testing)

    Returns
    -------
    np.ndarray (768,) -- granule-level feature embedding
    """
    device = next(model.parameters()).device

    arr = preprocess_tile(arr_6hw)
    tiles = _crop_tiles(arr)

    if not tiles:
        raise ValueError(
            f"Array shape {arr_6hw.shape} too small to produce any 224x224 tiles."
        )
    if max_tiles:
        tiles = tiles[:max_tiles]

    all_embeddings = []
    for tile in tiles:
        # PrithviViT.forward expects (B, C, T, H, W) -- T=1 here.
        tensor = (
            torch.from_numpy(tile)
            .unsqueeze(0)   # (1, 6, 224, 224)
            .unsqueeze(2)   # (1, 6, 1, 224, 224)
            .to(device)
        )
        with torch.no_grad():
            # PrithviViT.forward returns (latent, mask, ids_restore) when
            # encoder_only=True. We want the latent tokens.
            out = model(tensor)
        if isinstance(out, tuple):
            latent = out[0]  # (1, num_patches+1, embed_dim) -- includes CLS
        else:
            latent = out

        # Mean-pool patch tokens (skip CLS at index 0) -> (embed_dim,)
        embedding = latent[:, 1:, :].mean(dim=1).squeeze(0).cpu().numpy()
        all_embeddings.append(embedding)

    # Mean-pool across all tiles -> single (embed_dim,) granule vector
    return np.stack(all_embeddings, axis=0).mean(axis=0)


def _embed_tiles(model, tiles):
    """Run a list of (6, 224, 224) preprocessed tiles through Prithvi and mean-pool."""
    device = next(model.parameters()).device
    all_embeddings = []
    for tile in tiles:
        tensor = (
            torch.from_numpy(tile)
            .unsqueeze(0)
            .unsqueeze(2)
            .to(device)
        )
        with torch.no_grad():
            out = model(tensor)
        latent = out[0] if isinstance(out, tuple) else out
        embedding = latent[:, 1:, :].mean(dim=1).squeeze(0).cpu().numpy()
        all_embeddings.append(embedding)
    return np.stack(all_embeddings, axis=0).mean(axis=0)


def extract_features_corn_only(
    model,
    arr_6hw,
    profile,
    state_abbr,
    year,
    bucket,
    min_corn_frac=0.05,
    max_tiles=None,
    verbose=False,
):
    """
    Same as extract_features_from_array, but skips tiles with no corn coverage
    using the cached CDL corn mask in S3.

    Parameters
    ----------
    arr_6hw       : (6, H, W) HLS granule array (raw 0-10000)
    profile       : rasterio profile from stack_granule_cloud (gives CRS+transform)
    state_abbr    : e.g. "IA"
    year          : e.g. 2023 (used to pick CDL year)
    bucket        : S3 bucket holding raw/cdl/{fips}/{year}.tif
    min_corn_frac : drop tiles with corn fraction below this (0.0-1.0).
                    Default 0.05 keeps tiles with >= 5% corn pixels.

    Returns
    -------
    (features (768,), kept_tiles, total_tiles)

    Falls back to no-mask extraction if no CDL is available for that state/year.
    """
    mask = load_corn_mask_for_grid(bucket, state_abbr, year, profile)
    if mask is None:
        if verbose:
            print(f"    no CDL mask for {state_abbr} {year} — using full granule")
        feats = extract_features_from_array(model, arr_6hw, max_tiles=max_tiles)
        n = len(_crop_tiles(preprocess_tile(arr_6hw)))
        return feats, n, n

    arr = preprocess_tile(arr_6hw)
    tiles, n_total = _crop_tiles_with_mask(arr, mask, min_corn_frac=min_corn_frac)

    if verbose:
        print(f"    corn-filtered tiles: {len(tiles)} / {n_total} "
              f"(min_corn_frac={min_corn_frac:.2f})")

    if not tiles:
        # No tile met the corn threshold — granule is essentially non-corn.
        # Return zeros so the caller can decide to drop it.
        return np.zeros(768, dtype=np.float32), 0, n_total

    if max_tiles:
        tiles = tiles[:max_tiles]

    feats = _embed_tiles(model, tiles)
    return feats, len(tiles), n_total



def extract_features_batch(model, arrays, max_tiles_per_granule=16):
    """
    Extract features from a list of (6, H, W) arrays.
    Returns np.ndarray (N, 768).
    """
    return np.stack(
        [extract_features_from_array(model, arr, max_tiles=max_tiles_per_granule)
         for arr in arrays],
        axis=0,
    )


# ---------------------------------------------------------------------------
# S3 persistence -- save/load extracted feature vectors
# ---------------------------------------------------------------------------

def save_features_to_s3(features, granule_id, bucket, state_abbr, year, forecast_date):
    """
    Save a (768,) feature array to S3 as a .npy file.
    S3 key: processed/features/{state_abbr}/{year}/{forecast_date}/{granule_id}.npy
    """
    s3 = boto3.client("s3")
    key = f"processed/features/{state_abbr}/{year}/{forecast_date}/{granule_id}.npy"
    buf = io.BytesIO()
    np.save(buf, features)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())


def load_features_from_s3(bucket, state_abbr, year, forecast_date, granule_id):
    """Load a feature .npy from S3. Returns np.ndarray."""
    s3 = boto3.client("s3")
    key = f"processed/features/{state_abbr}/{year}/{forecast_date}/{granule_id}.npy"
    obj = s3.get_object(Bucket=bucket, Key=key)
    return np.load(io.BytesIO(obj["Body"].read()))


def list_features_in_s3(bucket, state_abbr, year, forecast_date):
    """List all granule feature keys in S3 for a state/year/forecast_date."""
    s3 = boto3.client("s3")
    prefix = f"processed/features/{state_abbr}/{year}/{forecast_date}/"
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys