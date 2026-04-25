
import rasterio
import numpy as np
import os

# USDA CDL crop codes
CORN_VALUE = 1  # Corn = 5 in USDA Cropland Data Layer

def findCorn():
    paths = ("Iowa", "Colorado", "Missouri", "Nebraska", "Wisconsin")

    for p in paths:
        dir_path = os.path.join("cornYield", p)

        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue

        for e in os.scandir(dir_path):
            if not (e.is_file() and e.name.lower().endswith(".tif")):
                continue

            print(f"Processing: {e.name}")
            try:
                with rasterio.open(e.path) as src:
                    data = src.read(1)
                    profile = src.profile

                # Debug: Check data values
                unique_values = np.unique(data[data != src.nodata] if src.nodata is not None else data)
                print(f"  Unique values in raster: {unique_values[:10]}...")  # Show first 10
                print(f"  Data range: {data.min()} to {data.max()}")
                print(f"  Contains corn value {CORN_VALUE}: {CORN_VALUE in unique_values}")

                corn_mask = np.where(data == CORN_VALUE, 255, 0).astype(np.uint8)  # Use 255 for visibility

                profile.update(
                    dtype=rasterio.uint8,
                    count=1,
                    nodata=0,
                    compress='lzw',  # Add compression to reduce file size
                )

                output_dir = os.path.join("output", p)
                os.makedirs(output_dir, exist_ok=True)

                # [0] extracts just the filename string from the tuple
                base_name = os.path.splitext(e.name)[0]
                output_path = os.path.join(output_dir, f"corn_only_{base_name}.tif")

                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(corn_mask, 1)

                corn_pixel_count = np.sum(corn_mask == 255)
                if corn_pixel_count > 0:
                    print(f"  Corn pixels: {corn_pixel_count:,} → saved to {output_path}")
                else:
                    os.remove(output_path)
            except Exception as ex:
                print(f"  Error processing {e.name}: {ex}")

            

findCorn()

