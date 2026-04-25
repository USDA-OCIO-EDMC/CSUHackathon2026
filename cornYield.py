import os
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gp
from skimage import io
import matplotlib.pyplot as plt
from osgeo import gdal
import xarray as xr
import rioxarray as rxr
import hvplot.xarray
import hvplot.pandas
import earthaccess

#login to earthaccess
earthaccess.login(persist=True)

#times of which to find satellite images
temporal = ("2023-07-01T00:00:00", "2023-10-31T23:59:59")

#state bounding boxes
state_bboxes = {
    'Iowa':      (-96.6, 40.4, -90.1, 43.5),
    'Colorado':  (-109.1, 37.0, -102.0, 41.0),
    'Wisconsin': (-92.9, 42.5, -86.8, 47.1),
    'Missouri':  (-95.8, 36.0, -89.1, 40.6),
    'Nebraska':  (-104.1, 40.0, -95.3, 43.0),
}

states = ['Iowa', 'Colorado', 'Wisconsin', 'Missouri', 'Nebraska']


for state in states:
    results = earthaccess.search_data(
    short_name=['HLSL30','HLSS30'],
    bounding_box=state_bboxes[state],
    temporal=temporal,
    count=100
    )

    file_path = "./cornYeild/" + state + "/"

    earthaccess.download(results, local_path=file_path)