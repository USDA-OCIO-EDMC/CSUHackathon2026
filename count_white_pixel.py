import rasterio
import numpy as np
import os

def count_white_pixels():
    output_dir = "output"
    
    if not os.path.exists(output_dir):
        print(f"Directory not found: {output_dir}")
        return
    
    for folder in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder)
        if os.path.isdir(folder_path):
            print(f"\nChecking folder: {folder}")
            has_white_pixels = False
            
            for file in os.listdir(folder_path):
                if file.lower().endswith('.tif'):
                    file_path = os.path.join(folder_path, file)
                    try:
                        with rasterio.open(file_path) as src:
                            data = src.read(1)
                            white_pixels = np.sum(data == 255)
                            if white_pixels > 0:
                                has_white_pixels = True
                                print(f"  {file}: {white_pixels:,} white pixels")
                    except Exception as e:
                        print(f"  Error processing {file}: {e}")
            
            if not has_white_pixels:
                print(f"  No white pixels found in {folder}")

count_white_pixels()