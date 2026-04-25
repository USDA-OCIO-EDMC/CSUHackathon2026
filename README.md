Group 22 Spring 26 Hackathon

Python Environment:
conda create -n Hackathon2026 python=3.14 -y
conda deactivate
conda activate Hackathon2026

Then make sure that the interpreter is to Hackathon2026 -> conda
CTRL+SHIFT+P -> Python: Select Interpreter 

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
conda install -c conda-forge geopandas rasterio shapely pyproj earthpy folium -y
conda install -c conda-forge numpy pandas matplotlib seaborn tqdm scikit-learn xgboost -y
pip install torch transformers huggingface_hub requests
pip install awscli