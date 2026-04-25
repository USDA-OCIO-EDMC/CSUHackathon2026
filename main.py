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

import findCorn
import cornYield
import stateYield

earthaccess.login(persist=True)

field = gp.read_file('data/Field_Boundary.geojson')
print(field)

if not os.path.exists("cornYield"):
    loadData()

if not os.path.exists("output"):
    findCorn()
    computeNDVI()

states = [("Colorado", 12, 1), ("Iowa", 11, 17),
          ("Missouri", 12, 1), ("Nebraska", 12, 1),
          ("Wisconsin", 11, 30)]

for state in states:
    stateYield(state[0], state[1], state[2])

