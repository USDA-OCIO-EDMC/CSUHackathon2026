# CSU Hackathon 2026 — Geospatial AI Crop Yield Forecasting

**Team:** Dev Lucas, Juan, Haven, Sophia

---

## Challenge

Predict corn grain yield (bushels per acre) for **Iowa, Colorado, Wisconsin, Missouri, and Nebraska** at four forecast dates during the 2025 growing season, with a quantified uncertainty range per forecast.

Ground truth: 2025 actuals already published by USDA NASS.

| Forecast Date | Crop Stage | Data Available |
|---|---|---|
| August 1 | Early grain fill | Weather May–Jul + early NDVI |
| September 1 | Late grain fill / dough stage | Weather May–Aug + NDVI through Sep |
| October 1 | Maturity / early harvest | Weather May–Sep + NDVI through Oct |
| End of Season | Post-harvest reconciliation | Full season weather + NDVI |

---

## Results (model1.4 — current best)

| Forecast Date | Validation RMSE |
|---|---|
| August 1 | ~9.7 bu/acre |
| September 1 | ~9.3 bu/acre |
| October 1 | ~9.7 bu/acre |
| End of Season | ~10.3 bu/acre |

State-level predictions for 2025 with 90% bootstrap confidence intervals available in `outputs/predictions.05_model1.4.csv`.

---

## Pipeline Overview

```
USDA QuickStats → Yield labels (2005–2024)
NOAA GSOM API  → Monthly weather features (2005–2025)
NASA HLS S30   → Satellite NDVI, corn-field masked (2015–2024)
                          │
                    04_merge_features
                          │
                  05_model → Random Forest + bias correction
                          │
              outputs/predictions.csv
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline diagram and [METHODOLOGY.md](METHODOLOGY.md) for a plain-language explanation of how every piece fits together.

---

## Quickstart

```bash
git clone <repo-url>
cd top_CSUHackathon26
conda activate geospatial-python-crash-course
cp .env.example .env   # add your NOAA_API_KEY
jupyter notebook
```

---

## Notebook Execution Order

| Notebook | Output | Status | Notes |
|---|---|---|---|
| `01_quickstats` | `data/processed/quickstats_yield.csv` | ✅ Complete | |
| `02_weather` | `data/processed/weather_features.csv` | ✅ Complete | Requires `NOAA_API_KEY` in `.env` |
| `03_satellite_fixed` | `data/raw/ndvi_by_state_date.csv` | ✅ Complete | **SageMaker only** — CSV committed to repo, do not re-run locally |
| `04_merge_features` | `data/processed/training_features.csv` | ✅ Complete | |
| `05_model1.4` | `outputs/predictions.05_model1.4.csv` | ✅ Current best | RF + per-state bias correction |
| `06_viz` | `outputs/*.png` | ✅ Complete | |

> **Note on notebook 03:** `03_satellite_fixed.ipynb` runs on SageMaker and requires NASA Earthdata credentials via IAM. It includes CDL corn-field masking reprojected to the HLS pixel grid. The output CSV is committed to the repo — **do not re-run locally**.

---

## Model Version History

| Version | Key Change | Aug1 RMSE | Notes |
|---|---|---|---|
| 1.0 | Baseline RF, no year feature | ~17.8 | Original |
| 1.1 | Added `year` as feature | ~15.0 | Captures long-term yield trend |
| 1.2 | Same features as 1.0, re-run | ~17.8 | Parity check |
| **1.4** | **RF + per-state bias correction** | **~9.7** | **Current best** |
| 1.5 | Intermediate experiment | ~12.4 | Superseded by 1.4 |

---

## Stack

- Python 3.9 — `conda activate geospatial-python-crash-course`
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `geopandas`, `rasterio`
- `earthaccess`, `rioxarray`, `pystac-client` (satellite notebooks only)
- `ipynbname` (auto-versioned output filenames)

---

## Contacts in the Room

| Person | Role |
|---|---|
| Andrew Dau | USDA NASS — crop yield track prompt owner |
| Kevin (NASA) | HLS data and Prithvi setup questions |
| AWS Solutions Architects | SageMaker environment, GPU instances |
| David Bartels | USDA — fruit fly track |
| Esri reps | Visualization and GIS |
