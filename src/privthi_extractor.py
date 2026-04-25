import transformers
import torch, rasterio
import numpy as np
import boto3
from io import BytesIO

def load_prithvi(device="cpu"):
    model = transformers.AutoModel.from_pretrained(
        "ibm-nasa-geospatial/Prithvi-100M",
        trust_remote_code=True
    ).to(device).eval()
    return model

def extract_features(model, naip_tile_path, device="cpu"):
    """Extract spatial embeddings from a single NAIP tile."""
    with rasterio.open(naip_tile_path) as src:
        # NAIP: bands 1=R, 2=G, 3=B, 4=NIR
        img = src.read().astype(np.float32)  # (4, H, W)
    # Normalize per-band
    img = (img - img.mean(axis=(1,2), keepdims=True)) / \
          (img.std(axis=(1,2), keepdims=True) + 1e-6)
    tensor = torch.tensor(img).unsqueeze(0).to(device)  # (1,4,H,W)
    with torch.no_grad():
        out = model(pixel_values=tensor)
    # Mean-pool patch embeddings → fixed-dim feature vector
    return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def extract_features_s3(model, bucket, key, device="cpu"):
    """Extract Prithvi embeddings from a NAIP tile stored in S3."""
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    with rasterio.open(BytesIO(obj['Body'].read())) as src:
        img = src.read().astype(np.float32)
    img = (img - img.mean(axis=(1,2), keepdims=True)) / \
          (img.std(axis=(1,2), keepdims=True) + 1e-6)
    tensor = torch.tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(pixel_values=tensor)
    return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()