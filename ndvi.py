import rasterio
import numpy as np
import os
import glob
import pandas as pd

def computeNDVI():
    paths = ("Iowa", "Colorado", "Missouri", "Nebraska", "Wisconsin")
    records = []

    for state in paths:
        hls_dir = os.path.join("cornYield", state)
        corn_dir = os.path.join("output", state)

        if not os.path.exists(hls_dir) or not os.path.exists(corn_dir):
            print(f"Skipping {state}: missing directories")
            continue

        # Find all B04 files, then match with B05 and corn mask
        b04_files = glob.glob(os.path.join(hls_dir, "*B04.tif"))

        for b04_path in b04_files:
            # Derive matching B05 and corn mask paths
            b05_path = b04_path.replace("B04.tif", "B05.tif")
            base_name = os.path.splitext(os.path.basename(b04_path))[0]  # Remove .tif extension
            # Remove B04 suffix to get base identifier
            base_identifier = base_name.replace("B04", "")
            corn_mask_path = os.path.join(corn_dir, f"corn_only_{base_identifier}.tif")

            if not os.path.exists(b05_path):
                print(f"  Missing B05 for {base_name}, skipping.")
                continue
            if not os.path.exists(corn_mask_path):
                print(f"  Missing corn mask for {base_name}, skipping.")
                continue

            print(f"Computing NDVI: {base_name}")
            try:
                with rasterio.open(b04_path) as src:
                    red = src.read(1).astype(np.float32)
                    profile = src.profile
                    nodata = src.nodata

                with rasterio.open(b05_path) as src:
                    nir = src.read(1).astype(np.float32)

                with rasterio.open(corn_mask_path) as src:
                    corn_mask = src.read(1)  # 255 = corn, 0 = not corn

                # HLS reflectance scaling: values are typically scaled by 10000
                # Scale once at the beginning
                if np.nanmax(red) > 1.0:  # Check if values need scaling
                    red = red / 10000.0
                    nir = nir / 10000.0

                # Mask out nodata and non-corn pixels
                valid = (corn_mask == 255)
                if nodata is not None:
                    valid &= (red != nodata/10000.0 if np.nanmax(red) <= 1.0 else red != nodata) & \
                             (nir != nodata/10000.0 if np.nanmax(nir) <= 1.0 else nir != nodata)

                # Compute NDVI: (NIR - Red) / (NIR + Red)
                denominator = nir + red
                # Avoid division by zero
                valid_calc = valid & (denominator != 0) & ~np.isnan(denominator)
                
                ndvi = np.full_like(red, np.nan)
                ndvi[valid_calc] = (nir[valid_calc] - red[valid_calc]) / denominator[valid_calc]

                # Save NDVI raster
                ndvi_dir = os.path.join("ndvi_output", state)
                os.makedirs(ndvi_dir, exist_ok=True)
                ndvi_path = os.path.join(ndvi_dir, f"ndvi_{base_name}.tif")

                profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
                with rasterio.open(ndvi_path, "w", **profile) as dst:
                    dst.write(ndvi.astype(np.float32), 1)

                # Extract date from HLS filename (format: HLS.L30.TXXXXX.YYYYDDD.vXX)
                parts = base_name.split(".")
                date_str = parts[3] if len(parts) > 3 else "unknown"  # e.g. 2020001 = year+DOY

                mean_ndvi = float(np.nanmean(ndvi))
                corn_count = int(valid.sum())

                print(f"  Mean NDVI: {mean_ndvi:.4f} | Corn pixels: {corn_count:,}")

                records.append({
                    "state": state,
                    "file": base_name,
                    "date": date_str,
                    "mean_ndvi": round(mean_ndvi, 4),
                    "corn_pixels": corn_count
                })

            except Exception as ex:
                print(f"  Error processing {base_name}: {ex}")

    # Save summary CSV
    if records:
        df = pd.DataFrame(records)
        df.to_csv("ndvi_summary.csv", index=False)
        print(f"NDVI summary saved to ndvi_summary.csv")
        print(df.head(10))

    return df

df = computeNDVI()

import rasterio
import numpy as np
import os
import glob
import pandas as pd

def computeNDVI():
    paths = ("Iowa", "Colorado", "Missouri", "Nebraska", "Wisconsin")
    records = []

    for state in paths:
        hls_dir = os.path.join("cornYield", state)
        corn_dir = os.path.join("output", state)

        if not os.path.exists(hls_dir) or not os.path.exists(corn_dir):
            print(f"Skipping {state}: missing directories")
            continue

        # Find all B04 files, then match with B05 and corn mask
        b04_files = glob.glob(os.path.join(hls_dir, "*B04.tif"))

        for b04_path in b04_files:
            # Derive matching B05 and corn mask paths
            b05_path = b04_path.replace("B04.tif", "B05.tif")
            base_name = os.path.splitext(os.path.basename(b04_path))[0]  # Remove .tif extension
            # Remove B04 suffix to get base identifier
            base_identifier = base_name.replace("B04", "")
            corn_mask_path = os.path.join(corn_dir, f"corn_only_{base_identifier}.tif")

            if not os.path.exists(b05_path):
                print(f"  Missing B05 for {base_name}, skipping.")
                continue
            if not os.path.exists(corn_mask_path):
                print(f"  Missing corn mask for {base_name}, skipping.")
                continue

            print(f"Computing NDVI: {base_name}")
            try:
                with rasterio.open(b04_path) as src:
                    red = src.read(1).astype(np.float32)
                    profile = src.profile
                    nodata = src.nodata

                with rasterio.open(b05_path) as src:
                    nir = src.read(1).astype(np.float32)

                with rasterio.open(corn_mask_path) as src:
                    corn_mask = src.read(1)  # 255 = corn, 0 = not corn

                # HLS reflectance scaling: values are typically scaled by 10000
                # Scale once at the beginning
                if np.nanmax(red) > 1.0:  # Check if values need scaling
                    red = red / 10000.0
                    nir = nir / 10000.0

                # Mask out nodata and non-corn pixels
                valid = (corn_mask == 255)
                if nodata is not None:
                    valid &= (red != nodata/10000.0 if np.nanmax(red) <= 1.0 else red != nodata) & \
                             (nir != nodata/10000.0 if np.nanmax(nir) <= 1.0 else nir != nodata)

                # Compute NDVI: (NIR - Red) / (NIR + Red)
                denominator = nir + red
                # Avoid division by zero
                valid_calc = valid & (denominator != 0) & ~np.isnan(denominator)
                
                ndvi = np.full_like(red, np.nan)
                ndvi[valid_calc] = (nir[valid_calc] - red[valid_calc]) / denominator[valid_calc]

                # Save NDVI raster
                ndvi_dir = os.path.join("ndvi_output", state)
                os.makedirs(ndvi_dir, exist_ok=True)
                ndvi_path = os.path.join(ndvi_dir, f"ndvi_{base_name}.tif")

                profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
                with rasterio.open(ndvi_path, "w", **profile) as dst:
                    dst.write(ndvi.astype(np.float32), 1)

                # Extract date from HLS filename (format: HLS.L30.TXXXXX.YYYYDDD.vXX)
                parts = base_name.split(".")
                date_str = parts[3] if len(parts) > 3 else "unknown"  # e.g. 2020001 = year+DOY

                mean_ndvi = float(np.nanmean(ndvi))
                corn_count = int(valid.sum())

                print(f"  Mean NDVI: {mean_ndvi:.4f} | Corn pixels: {corn_count:,}")

                records.append({
                    "state": state,
                    "file": base_name,
                    "date": date_str,
                    "mean_ndvi": round(mean_ndvi, 4),
                    "corn_pixels": corn_count
                })

            except Exception as ex:
                print(f"  Error processing {base_name}: {ex}")

    # Save summary CSV
    if records:
        df = pd.DataFrame(records)
        df.to_csv("ndvi_summary.csv", index=False)
        print(f"NDVI summary saved to ndvi_summary.csv")
        print(df.head(10))

    return df

df = computeNDVI()

