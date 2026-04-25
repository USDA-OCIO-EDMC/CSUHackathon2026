"""
Phase 1.5 — Cloud-Sorted Monthly HLS Composites

Adapted from notebooks/landsat_merge.ipynb (Carpentries) into our pipeline:
  1. Search HLS granules for (state, year, month).
  2. Sort by per-scene cloud cover, keep the top-K cleanest.
  3. Reproject each scene's 6-band stack onto a common state-level grid
     (EPSG:5070, 30 m).
  4. Pixel-wise nanmean across scenes -> one denoised (6, H, W) mosaic.
  5. (Optional) Mask non-corn pixels using the CDL corn raster cached in S3
     by `data_utils.fetch_all_cdl`.
  6. Write composite to s3://{bucket}/processed/composites/{state}/{year}/{month}.tif

Downstream:
  privthi_extractor.extract_features_from_array() takes the (6, H, W) array
  unchanged, so feature extraction now produces ONE embedding per
  (state, year, month) instead of one per granule -- matching the `month`
  axis already expected by feature_fusion + forecaster.DATE_MAP.
"""

from __future__ import annotations

import io
from calendar import monthrange
from datetime import datetime

import boto3
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, calculate_default_transform, reproject

import earthaccess

from hls_downloader import HLS_BANDS, HLS_PRODUCTS, STATE_BBOX, stack_granule_cloud

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Albers Equal Area (CONUS) -- single CRS for all 5 states, units = metres.
TARGET_CRS = "EPSG:5070"

# Native HLS res is 30 m, but a state-wide 30 m grid is ~6.6 GB per band-stack.
# 120 m (16x less memory) keeps each state under ~500 MB and is plenty for
# producing one Prithvi embedding per (state, year, month) for analog-year
# similarity search. Override per-call if you have lots of RAM.
TARGET_RES = 120.0    # metres

CORN_CLASS = 1        # CDL class for Corn (matches data_utils.CORN_CLASS)


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def _month_window(year: int, month: int) -> tuple[str, str]:
    last = monthrange(year, month)[1]
    return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last:02d}"


def search_hls_month(state_abbr: str, year: int, month: int) -> list:
    """Search HLS L30+S30 granules covering one (state, month)."""
    bbox = STATE_BBOX[state_abbr]
    temporal = _month_window(year, month)
    results = []
    for short_name, version in HLS_PRODUCTS:
        results.extend(earthaccess.search_data(
            short_name=short_name,
            version=version,
            temporal=temporal,
            bounding_box=bbox,
        ))
    return results


def _granule_cloud_cover(g) -> float:
    """Read scene-level cloud cover from CMR UMM metadata. Returns 100.0 if missing."""
    try:
        for attr in g["umm"].get("AdditionalAttributes", []):
            if attr.get("Name") in ("CLOUD_COVERAGE", "CloudCover", "PERCENT_CLOUD_COVER"):
                vals = attr.get("Values", [])
                if vals:
                    return float(vals[0])
    except Exception:
        pass
    return 100.0


def _granule_tile_id(g) -> str:
    """
    Extract the MGRS tile id (e.g. 'T15TWH') from an HLS granule's native id
    so we can group and pick top-K cleanest per tile.
    """
    try:
        gid = g["umm"]["GranuleUR"]
    except Exception:
        gid = str(g)
    # e.g. HLS.L30.T15TWH.2023121T164646.v2.0  ->  T15TWH
    parts = gid.split(".")
    for p in parts:
        if len(p) == 6 and p.startswith("T") and p[1:3].isdigit():
            return p
    return "UNKNOWN"


def _select_topk_per_tile(granules: list, top_k_per_tile: int) -> list:
    """
    Group granules by MGRS tile, sort each group by cloud cover ascending,
    keep the top_k_per_tile cleanest within each group. Returns the union.
    """
    by_tile: dict[str, list] = {}
    for g in granules:
        by_tile.setdefault(_granule_tile_id(g), []).append(g)
    picked = []
    for tile, group in by_tile.items():
        group.sort(key=_granule_cloud_cover)
        picked.extend(group[:top_k_per_tile])
    return picked


# ---------------------------------------------------------------------------
# Reprojection / merging
# ---------------------------------------------------------------------------

def _state_target_grid(state_abbr: str, target_res: float = TARGET_RES) -> tuple[rasterio.Affine, int, int]:
    """Compute a fixed Albers grid spanning the state's bbox at target_res metres."""
    min_lon, min_lat, max_lon, max_lat = STATE_BBOX[state_abbr]
    transform, width, height = calculate_default_transform(
        src_crs="EPSG:4326",
        dst_crs=TARGET_CRS,
        width=int((max_lon - min_lon) * 3000),     # rough pixel guess at 30 m
        height=int((max_lat - min_lat) * 3000),
        left=min_lon, bottom=min_lat, right=max_lon, top=max_lat,
        resolution=target_res,
    )
    return transform, int(width), int(height)


def _reproject_scene_to_grid(
    stacked: np.ndarray,
    src_profile: dict,
    dst_transform: rasterio.Affine,
    dst_width: int,
    dst_height: int,
) -> np.ndarray:
    """
    Reproject a (6, H, W) granule onto the state target grid.
    Pixels outside the source coverage become NaN.
    """
    out = np.full((6, dst_height, dst_width), np.nan, dtype=np.float32)
    src_nodata = src_profile.get("nodata")

    for b in range(6):
        reproject(
            source=stacked[b],
            destination=out[b],
            src_transform=src_profile["transform"],
            src_crs=src_profile["crs"],
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    if src_nodata is not None:
        out[out == src_nodata] = np.nan
    # HLS encodes "no data" as -9999 even when nodata isn't set in the profile
    out[out <= -9999] = np.nan
    return out


# ---------------------------------------------------------------------------
# CDL corn mask
# ---------------------------------------------------------------------------

_STATE_FIPS = {"IA": "19", "CO": "08", "WI": "55", "MO": "29", "NE": "31"}


def _load_corn_mask_on_grid(
    bucket: str,
    state_abbr: str,
    year: int,
    dst_transform: rasterio.Affine,
    dst_width: int,
    dst_height: int,
) -> np.ndarray | None:
    """
    Pull the CDL corn raster for (state, year) from S3 and reproject onto the
    composite grid. Returns a boolean (H, W) mask (True = corn). Falls back to
    the most recent available year if the exact year is missing.
    """
    s3 = boto3.client("s3")
    fips = _STATE_FIPS[state_abbr]

    # Pick best available year
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

    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    out = np.zeros((dst_height, dst_width), dtype=np.uint8)
    with MemoryFile(data) as mf, mf.open() as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.nearest,
        )
    return out == CORN_CLASS


# ---------------------------------------------------------------------------
# S3 persistence
# ---------------------------------------------------------------------------

def composite_s3_key(state_abbr: str, year: int, month: int) -> str:
    return f"processed/composites/{state_abbr}/{year}/{month:02d}.tif"


def _composite_exists(bucket: str, key: str) -> bool:
    try:
        boto3.client("s3").head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _write_composite_to_s3(
    arr: np.ndarray,
    transform: rasterio.Affine,
    width: int,
    height: int,
    bucket: str,
    key: str,
) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 6,
        "width": width,
        "height": height,
        "crs": TARGET_CRS,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "tiled": True,
    }
    with MemoryFile() as mf:
        with mf.open(**profile) as dst:
            dst.write(arr.astype(np.float32))
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=mf.read())


def load_composite_from_s3(bucket: str, state_abbr: str, year: int, month: int):
    """Read a composite GeoTIFF from S3. Returns (arr (6,H,W), profile)."""
    key = composite_s3_key(state_abbr, year, month)
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    with MemoryFile(obj["Body"].read()) as mf, mf.open() as src:
        return src.read().astype(np.float32), src.profile.copy()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_monthly_composite(
    state_abbr: str,
    year: int,
    month: int,
    bucket: str,
    top_k: int = 5,
    top_k_per_tile: int | None = None,
    mask_corn: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
    target_res: float = TARGET_RES,
) -> str | None:
    """
    Build a cloud-sorted, corn-masked monthly composite for one (state, year, month)
    and upload it to S3.

    Selection strategy:
      - If `top_k_per_tile` is set, group granules by MGRS tile and keep the
        `top_k_per_tile` cleanest per tile (recommended for full-state coverage).
      - Otherwise fall back to global top-K (`top_k`).

    Returns the S3 key on success, or None if no usable scenes were found.
    """
    key = composite_s3_key(state_abbr, year, month)
    if not overwrite and _composite_exists(bucket, key):
        if verbose:
            print(f"  [{state_abbr} {year}-{month:02d}] already in S3 ({key}) — skipping")
        return key

    granules = search_hls_month(state_abbr, year, month)
    if not granules:
        if verbose:
            print(f"  [{state_abbr} {year}-{month:02d}] no granules found")
        return None

    if top_k_per_tile is not None:
        cleanest = _select_topk_per_tile(granules, top_k_per_tile)
        if verbose:
            tiles = sorted({_granule_tile_id(g) for g in cleanest})
            print(f"  [{state_abbr} {year}-{month:02d}] using {len(cleanest)}/{len(granules)} "
                  f"granules across {len(tiles)} tiles (top {top_k_per_tile} per tile): {tiles}")
    else:
        granules.sort(key=_granule_cloud_cover)
        cleanest = granules[:top_k]
        if verbose:
            print(f"  [{state_abbr} {year}-{month:02d}] using {len(cleanest)}/{len(granules)} "
                  f"granules (cloud cover: "
                  f"{[f'{_granule_cloud_cover(g):.0f}%' for g in cleanest]})")

    transform, width, height = _state_target_grid(state_abbr, target_res=target_res)
    if verbose:
        mb = (6 * height * width * 4) / (1024 ** 2)
        print(f"    target grid {width}x{height} px @ {target_res:.0f} m  (~{mb:.0f} MB / band-stack)")

    accum = np.zeros((6, height, width), dtype=np.float32)
    counts = np.zeros((6, height, width), dtype=np.int16)

    for i, g in enumerate(cleanest, 1):
        try:
            stacked, profile, gid = stack_granule_cloud(g)
        except Exception as e:
            if verbose:
                print(f"    skip granule {i}: {e}")
            continue
        warped = _reproject_scene_to_grid(stacked, profile, transform, width, height)
        valid = ~np.isnan(warped)
        accum = np.where(valid, accum + np.nan_to_num(warped), accum)
        counts += valid.astype(np.int16)

    if counts.max() == 0:
        if verbose:
            print(f"  [{state_abbr} {year}-{month:02d}] all scenes failed to reproject")
        return None

    with np.errstate(invalid="ignore", divide="ignore"):
        composite = np.where(counts > 0, accum / counts, np.nan).astype(np.float32)

    if mask_corn:
        mask = _load_corn_mask_on_grid(bucket, state_abbr, year, transform, width, height)
        if mask is None:
            if verbose:
                print(f"    no CDL mask available — saving unmasked composite")
        else:
            corn_pct = 100.0 * mask.mean()
            if verbose:
                print(f"    applying CDL corn mask ({corn_pct:.1f}% corn pixels)")
            composite[:, ~mask] = np.nan

    _write_composite_to_s3(composite, transform, width, height, bucket, key)
    if verbose:
        print(f"    wrote s3://{bucket}/{key}  shape={composite.shape}")
    return key


def build_composites_for_year(
    state_abbr: str,
    year: int,
    bucket: str,
    months: tuple[int, ...] = (5, 6, 7, 8, 9, 10),
    top_k: int = 5,
    top_k_per_tile: int | None = None,
    mask_corn: bool = True,
    overwrite: bool = False,
    target_res: float = TARGET_RES,
) -> list[str]:
    """Convenience: build composites for the standard May–Oct growing season."""
    keys = []
    for m in months:
        k = build_monthly_composite(
            state_abbr, year, m, bucket,
            top_k=top_k, top_k_per_tile=top_k_per_tile,
            mask_corn=mask_corn, overwrite=overwrite,
            target_res=target_res,
        )
        if k:
            keys.append(k)
    return keys
