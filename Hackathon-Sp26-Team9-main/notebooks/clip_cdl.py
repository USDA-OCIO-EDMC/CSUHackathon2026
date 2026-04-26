cat > clip_cdl.py << 'PYEOF'
import rioxarray as rxr
import os
from pyproj import Transformer
from pathlib import Path

TIF_PATH = os.path.expanduser('~/Downloads/2022_30m_cdls/2022_30m_cdls.tif')
OUT_DIR  = os.path.expanduser('~/cdl_masks')

STATE_BBOX_WGS84 = {
    'Iowa':      (-96.64, 40.37, -90.14, 43.50),
    'Colorado':  (-109.06, 36.99, -102.04, 41.00),
    'Wisconsin': (-92.89, 42.49, -86.25, 47.08),
    'Missouri':  (-95.77, 35.99, -89.10, 40.61),
    'Nebraska':  (-104.05, 39.99, -95.31, 43.00),
}

CORN_VALUE = 1
os.makedirs(OUT_DIR, exist_ok=True)

# Transform bounding boxes from WGS84 to EPSG:5070
transformer = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)

print('Opening national CDL (~30 seconds)...')
cdl = rxr.open_rasterio(TIF_PATH, masked=True).squeeze()
print(f'CRS: {cdl.rio.crs}')

for state, (west, south, east, north) in STATE_BBOX_WGS84.items():
    print(f'Clipping {state}...', end=' ', flush=True)
    # Transform all 4 corners to Albers
    x_min, y_min = transformer.transform(west, south)
    x_max, y_max = transformer.transform(east, north)
    clipped  = cdl.rio.clip_box(x_min, y_min, x_max, y_max)
    n_corn   = int((clipped == CORN_VALUE).sum().values)
    out_path = f'{OUT_DIR}/{state}_2022.tif'
    clipped.rio.to_raster(out_path)
    print(f'OK — {n_corn:,} corn pixels → {out_path}')

print('\nDone. Zip and upload cdl_masks/ to SageMaker.')
PYEOF
pip install pyproj -q && python3 clip_cdl.py