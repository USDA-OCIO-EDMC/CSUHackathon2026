Group 22 Spring 26 Hackathon

Python Environment:
python -m venv Hackathon2026
source Hackathon2026/bin/activate

Project structure
cornsight/
├── data/
│   ├── raw/          # NAIP tiles, NASS CSV, NOAA weather
│   ├── processed/    # Aligned, masked, normalized
│   └── features/     # Feature matrices per state/date
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_prithvi_features.ipynb
│   ├── 03_train_forecast.ipynb
│   └── 04_cone_uncertainty.ipynb
├── src/
│   ├── data_utils.py
│   ├── prithvi_extractor.py
│   ├── forecaster.py
│   ├── analog_years.py
│   └── visualize.py
├── outputs/
│   └── maps/, charts/, forecasts/
└── requirements.txt

Pip installs:
pip install torch transformers huggingface_hub
pip install geopandas rasterio earthpy shapely pyproj
pip install xgboost scikit-learn pandas numpy
pip install folium matplotlib seaborn tqdm requests