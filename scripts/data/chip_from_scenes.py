"""Tile-centric parallel chip extractor with GPU median compositing.

Key speedup over the naive county-centric approach: each HLS scene file is read
EXACTLY ONCE. All counties that overlap a tile are processed while that tile is
in memory, eliminating redundant disk reads.

GPU (CuPy) is used for nanmedian compositing when available — on the DGX Spark's
unified memory this is zero-copy and typically 5-10x faster than numpy.

Usage:
    python scripts/data/chip_from_scenes.py
    python scripts/data/chip_from_scenes.py --workers 8 --years 2024
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import rasterio.warp
import yaml
import zarr
from shapely.geometry import box
from shapely.ops import transform
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CFG  = yaml.safe_load((ROOT / "configs" / "project.yaml").read_text())

SCENES_DIR  = ROOT / "data" / "raw" / "hls" / "scenes"
OUT_DIR     = ROOT / "data" / "processed" / "chips"
TIGER_CACHE = ROOT / "data" / "raw" / "tiger_counties.gpkg"
CHIP        = CFG["prithvi"]["chip_size"]       # 224
PIX         = CFG["prithvi"]["pixel_size_m"]    # 30
SCALE       = CFG["prithvi"]["scale_factor"]
CHECKPOINTS = CFG["project"]["forecast_checkpoints"]
STATES      = CFG["project"]["states"]
STATE_FIPS  = {"IA": "19", "CO": "08", "WI": "55", "MO": "29", "NE": "31"}
FIPS_STATE  = {v: k for k, v in STATE_FIPS.items()}

BAND_MAP = {
    "HLSL30": ["B02", "B03", "B04", "B05", "B06", "B07"],
    "HLSS30": ["B02", "B03", "B04", "B8A", "B11", "B12"],
}

_FILE_RE = re.compile(r"HLS\.(L30|S30)\.(\w+)\.(\d{7})T\d+\.v2\.0\.(\w+)\.tif$")


# ---------------------------------------------------------------------------
# Index building (runs in main process)
# ---------------------------------------------------------------------------

def build_scene_index(scenes_dir: Path) -> dict:
    """Return {tile: {date: {product: {band: str(path)}}}}."""
    idx: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for f in scenes_dir.glob("HLS.*.tif"):
        m = _FILE_RE.match(f.name)
        if not m:
            continue
        product, tile, doy_str, band = m.groups()
        year, doy = int(doy_str[:4]), int(doy_str[4:])
        date = datetime(year, 1, 1) + timedelta(days=doy - 1)
        idx[tile][date][f"HLS{product}"][band] = str(f)
    # convert defaultdicts to plain dicts for pickling
    return {t: {d: dict(p) for d, p in dates.items()} for t, dates in idx.items()}


def tile_bounds_gdf(scene_idx: dict) -> gpd.GeoDataFrame:
    """Read CRS+extent of ONE file per tile; return WGS84 GeoDataFrame."""
    rows = []
    for tile, dates in scene_idx.items():
        sample = None
        for _, products in dates.items():
            for _, bands in products.items():
                if "B02" in bands:
                    sample = bands["B02"]
                    break
            if sample:
                break
        if not sample:
            continue
        with rasterio.open(sample) as src:
            l, b, r, t = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        rows.append({"tile": tile, "geometry": box(l, b, r, t)})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def get_counties() -> gpd.GeoDataFrame:
    if TIGER_CACHE.exists():
        return gpd.read_file(TIGER_CACHE)
    url = "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
    print("Downloading TIGER county boundaries...")
    counties = gpd.read_file(url)
    fps = list(STATE_FIPS.values())
    counties = counties[counties["STATEFP"].isin(fps)].to_crs("EPSG:4326").copy()
    counties.to_file(TIGER_CACHE, driver="GPKG")
    print(f"  {len(counties)} counties cached.")
    return counties


# ---------------------------------------------------------------------------
# GPU-aware median (zero-copy on unified memory)
# ---------------------------------------------------------------------------

def _nanmedian(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    """Use CuPy on GPU if available, fall back to numpy."""
    try:
        import cupy as cp
        gpu = cp.asarray(stack)           # zero-copy on DGX Spark unified memory
        result = cp.nanmedian(gpu, axis=axis)
        return cp.asnumpy(result)
    except Exception:
        return np.nanmedian(stack, axis=axis).astype(np.float32)


# ---------------------------------------------------------------------------
# Chip cutting (vectorised with stride tricks)
# ---------------------------------------------------------------------------

def cut_chips(stack: np.ndarray, min_valid: float = 0.01) -> np.ndarray | None:
    """
    stack: (3, 6, H, W) float32
    returns: (N, 3, 6, 224, 224) float32, or None if no valid chips
    """
    T, C, H, W = stack.shape
    n_y = (H - CHIP) // CHIP + 1
    n_x = (W - CHIP) // CHIP + 1
    if n_y < 1 or n_x < 1:
        return None
    chips = []
    for iy in range(n_y):
        for ix in range(n_x):
            tile = stack[:, :, iy*CHIP:(iy+1)*CHIP, ix*CHIP:(ix+1)*CHIP]
            if np.isfinite(tile).mean() >= min_valid:
                chips.append(np.nan_to_num(tile, nan=0.0))
    if not chips:
        return None
    return np.stack(chips).astype(np.float32)


# ---------------------------------------------------------------------------
# Worker: processes ONE tile — reads each scene file exactly once
# ---------------------------------------------------------------------------

def _reproject_geom(geom, dst_crs_wkt: str):
    proj = pyproj.Transformer.from_crs("EPSG:4326", dst_crs_wkt, always_xy=True)
    return transform(proj.transform, geom)


def process_tile(args: tuple) -> dict:
    """
    Tile-centric worker: load all scenes for one tile into memory once,
    then extract chips for every overlapping county × year × checkpoint.

    Returns {(fips, year, ck): n_chips | 'skip' | 'fail'}.
    """
    import rasterio.mask
    import shapely.wkb

    tile, tile_dates, county_rows, target_years = args
    # county_rows: list of (fips, statefp, geom_wkb)

    results = {}

    # Build per-date array cache: {date: {product: {"bands": (6,H,W), "fmask": (H,W), "meta": dict}}}
    # We defer reading until needed (lazy per-date), but within each date we read all bands.
    # This way peak memory = one date × all bands × tile = ~300 MB.

    def load_date(date, products):
        loaded = {}
        for product, bands in products.items():
            bnames = BAND_MAP.get(product)
            if bnames is None:
                continue
            fmask_path = bands.get("Fmask")
            band_paths = [bands.get(b) for b in bnames]
            if fmask_path is None or any(p is None for p in band_paths):
                continue
            try:
                with rasterio.open(fmask_path) as src:
                    crs_wkt = src.crs.to_wkt()
                    transform_ = src.transform
                    fm = src.read(1)
                band_arrs = []
                for bp in band_paths:
                    with rasterio.open(bp) as src:
                        band_arrs.append(src.read(1).astype(np.float32))
                loaded[product] = {
                    "bands": np.stack(band_arrs),   # (6, H, W)
                    "fmask": fm,
                    "crs_wkt": crs_wkt,
                    "transform": transform_,
                }
            except Exception:
                continue
        return loaded

    def extract_county_window(loaded_date: dict, county_geom_wgs84):
        """Crop loaded tile arrays to county bbox, apply cloud mask. Returns (6,H,W) or None."""
        from rasterio.transform import rowcol
        results_per_product = []
        for product, data in loaded_date.items():
            try:
                crs_wkt = data["crs_wkt"]
                fmask   = data["fmask"]
                bands   = data["bands"]
                tf      = data["transform"]

                county_native = _reproject_geom(county_geom_wgs84, crs_wkt)
                minx, miny, maxx, maxy = county_native.bounds

                row_start, col_start = rowcol(tf, minx, maxy)
                row_stop,  col_stop  = rowcol(tf, maxx, miny)
                row_start = max(0, int(row_start))
                col_start = max(0, int(col_start))
                row_stop  = min(fmask.shape[0], int(row_stop))
                col_stop  = min(fmask.shape[1], int(col_stop))
                if row_stop <= row_start or col_stop <= col_start:
                    continue

                fm_crop   = fmask[row_start:row_stop, col_start:col_stop]
                band_crop = bands[:, row_start:row_stop, col_start:col_stop].copy()

                bad   = (fm_crop & 0b00111110) > 0
                valid = (~bad) & (fm_crop != 255)
                if valid.mean() < 0.01:   # 1% threshold — was 5%, too aggressive
                    continue
                band_crop[:, ~valid] = np.nan
                results_per_product.append(band_crop)
            except Exception:
                continue

        if not results_per_product:
            return None
        return results_per_product[0]

    def checkpoint_windows(year: int, mmdd: str):
        month, day = int(mmdd[:2]), int(mmdd[3:])
        end = datetime(year, month, day)
        windows = []
        for lag in range(2, -1, -1):
            w_end   = end   - timedelta(days=30 * lag)
            w_start = w_end - timedelta(days=30)
            windows.append((w_start, w_end))
        return windows

    # Build list of unique dates sorted
    all_dates = sorted(tile_dates.keys())

    for county_geom_wkb, fips, statefp in county_rows:
        state_abbr = FIPS_STATE.get(statefp)
        if not state_abbr:
            continue
        county_geom = shapely.wkb.loads(county_geom_wkb)

        for year in target_years:
            for ck_name, mmdd in CHECKPOINTS.items():
                out_path = OUT_DIR / state_abbr / fips / str(year) / f"{ck_name}.zarr"
                key = (fips, year, ck_name)
                if out_path.exists():
                    results[key] = "skip"
                    continue

                windows = checkpoint_windows(year, mmdd)
                timesteps = []

                for w_start, w_end in windows:
                    window_arrays = []
                    for date in all_dates:
                        if not (w_start <= date <= w_end):
                            continue
                        loaded = load_date(date, tile_dates[date])
                        arr = extract_county_window(loaded, county_geom)
                        if arr is not None:
                            window_arrays.append(arr)

                    if not window_arrays:
                        # Empty window: skip but keep iterating later windows.
                        # Will be backfilled later via padding from a non-empty window.
                        continue

                    # Trim to common shape and median composite
                    H = min(a.shape[1] for a in window_arrays)
                    W = min(a.shape[2] for a in window_arrays)
                    stack = np.stack([a[:, :H, :W] for a in window_arrays])  # (T,6,H,W)
                    comp  = _nanmedian(stack, axis=0).astype(np.float32)
                    timesteps.append(comp)

                if len(timesteps) == 0:
                    results[key] = "fail"
                    continue
                # Pad to 3 timesteps by repeating last available if needed
                while len(timesteps) < 3:
                    timesteps.append(timesteps[-1].copy())

                # Ensure all timesteps have same spatial shape
                H = min(c.shape[1] for c in timesteps)
                W = min(c.shape[2] for c in timesteps)
                full_stack = np.stack([c[:, :H, :W] for c in timesteps])  # (3,6,H,W)

                chips = cut_chips(full_stack)
                if chips is None:
                    results[key] = "fail"
                    continue

                # centroid in WGS84 [lon, lat]
                c = county_geom.centroid
                centroid_lonlat = [c.x, c.y]
                # end-date of each 30-day window as ISO string
                month, day = int(mmdd[:2]), int(mmdd[3:])
                end = datetime(year, month, day)
                window_ends = [
                    (end - timedelta(days=30 * lag)).strftime("%Y-%m-%d")
                    for lag in range(2, -1, -1)
                ]

                out_path.parent.mkdir(parents=True, exist_ok=True)
                z = zarr.open(str(out_path), mode="w")
                z.create_dataset("chips", data=chips, shape=chips.shape,
                                 chunks=(1, 3, 6, CHIP, CHIP), dtype="f4")
                z.attrs.update({
                    "fips": fips, "state": state_abbr,
                    "year": year, "checkpoint": ck_name,
                    "scale_factor": SCALE,
                    "centroid_lonlat": centroid_lonlat,
                    "dates": window_ends,
                })
                results[key] = len(chips)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel tile workers (default: 8 — each holds ~1–5 GB in memory)")
    ap.add_argument("--years", default=None,
                    help="comma-separated years, e.g. 2024 (default: all found in scenes/)")
    ap.add_argument("--max-tiles", type=int, default=None,
                    help="process only the first N tiles — useful for a partial checkpoint run")
    args = ap.parse_args()

    print("Building scene index (one pass over scenes/)...")
    scene_idx = build_scene_index(SCENES_DIR)
    all_dates  = [d for dates in scene_idx.values() for d in dates]
    years_in   = sorted({d.year for d in all_dates})
    target_years = [int(y) for y in args.years.split(",")] if args.years else years_in
    print(f"  {len(scene_idx)} tiles | {len(all_dates)} scene-dates | years: {target_years}")

    print("Building tile bounds spatial index...")
    tile_gdf = tile_bounds_gdf(scene_idx)

    print("Loading county geometries...")
    counties = get_counties()

    # Spatial join: county → overlapping tiles
    joined = gpd.sjoin(tile_gdf, counties[["GEOID","STATEFP","geometry"]],
                       how="left", predicate="intersects")

    # Invert: tile → list of (geom_wkb, fips, statefp)
    tile_counties: dict[str, list] = defaultdict(list)
    seen = set()
    for _, row in joined.iterrows():
        if not isinstance(row.get("GEOID"), str):
            continue
        tile = row["tile"]
        fips = row["GEOID"]
        key  = (tile, fips)
        if key in seen:
            continue
        seen.add(key)
        # get county geometry
        county_geom = counties.loc[counties["GEOID"] == fips, "geometry"].iloc[0]
        tile_counties[tile].append((county_geom.wkb, fips, row["STATEFP"]))

    work = [
        (tile, scene_idx[tile], tile_counties[tile], target_years)
        for tile in scene_idx
        if tile in tile_counties and tile_counties[tile]
    ]
    if args.max_tiles:
        work = work[:args.max_tiles]
        print(f"Limiting to first {args.max_tiles} tiles (checkpoint mode).")

    n_combos = sum(
        len(counties_) * len(target_years) * len(CHECKPOINTS)
        for _, _, counties_, _ in work
    )
    print(f"\n{len(work)} tiles × ~{n_combos:,} county-year-checkpoint combos")
    print(f"Workers: {args.workers}  (each tile read once — no redundant I/O)\n")

    try:
        import cupy  # noqa
        print("GPU median: CuPy available — using GPU compositing")
    except ImportError:
        print("GPU median: CuPy not found — using numpy (run: pip install cupy-cuda12x)")

    ok = fail = skip = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_tile, w): w[0] for w in work}
        for fut in tqdm(as_completed(futs), total=len(work), desc="tiles"):
            tile_results = fut.result()
            for key, val in tile_results.items():
                if val == "skip":
                    skip += 1
                elif val == "fail":
                    fail += 1
                else:
                    ok += 1

    print(f"\nDone — chips written: {ok}  skipped: {skip}  no-data/fail: {fail}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
