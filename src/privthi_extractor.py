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