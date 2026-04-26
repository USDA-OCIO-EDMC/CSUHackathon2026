# Architecture

## Mental Model

> "Given what satellites and weather stations observed about corn fields in Iowa, Colorado,
> Wisconsin, Missouri, and Nebraska as of a given date in 2025 — how many bushels per acre
> will each state produce at end of season?"

Answered by training on 20 years of historical data (2005–2024), then predicting 2025.

---

## MVP Definition

Guaranteed deliverables within the hackathon window:

1. End-to-end pipeline executes without errors (notebooks 01 → 06)
2. 2025 yield predictions for all 5 states at all 4 forecast dates
3. Bootstrap confidence interval (cone of uncertainty) per state per date
4. Analog year identification per state per date
5. Presentation-ready charts

**Beyond MVP (priority order):**
1. Per-state bias correction on top of Random Forest predictions ✅ (model1.4)
2. Prithvi-100M replacing Random Forest (GPU required)
3. NAIP / higher-resolution field masking

---

## Pipeline

```
RAW DATA SOURCES
      │
      ▼
┌─────────────────────────────────────┐
│  01_quickstats.ipynb                │
│  USDA NASS yield data (2005–2024)   │
│  5 states × 20 years = 100 rows     │
│  → data/processed/quickstats_yield  │
│  STATUS: ✅ Complete                │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  02_weather.ipynb                   │
│  NOAA GSOM — monthly TAVG + PRCP    │
│  5 states × May–Oct × 2005–2025     │
│  → data/processed/weather_features  │
│  STATUS: ✅ Complete                │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  03_satellite_fixed.ipynb (SageMaker)│
│  NASA HLS S30 — NDVI per state      │
│  CDL corn-field masking applied     │
│  per forecast date, 2015–2024       │
│  → data/raw/ndvi_by_state_date.csv  │
│  STATUS: ✅ Complete (CSV committed) │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  04_merge_features.ipynb            │
│  Joins QuickStats + NOAA + NDVI     │
│  on year + state                    │
│  → data/processed/training_features │
│  STATUS: ✅ Complete                │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  05_model1.4.ipynb                  │
│  Random Forest Regressor (baseline) │
│  + per-state bias correction        │
│  → outputs/predictions.*.csv        │
│  STATUS: ✅ Current best            │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  06_viz.ipynb                       │
│  Yield trajectories, uncertainty    │
│  cones, analog year overlays        │
│  → outputs/*.png                    │
│  STATUS: ⏳ Pending                 │
└─────────────────────────────────────┘
```

---

## Data Flow: Shapes and Keys

```
quickstats_yield.csv       weather_features.csv      ndvi_by_state_date.csv
(100 rows)                 (105 rows — incl. 2025)   (45 rows — 2016–2024)
year | state | yield   +   year | state | tavg_may…  +  year | state | ndvi_aug1…
                 │                        │                           │
                 └────────────────────────┴───────────────────────────┘
                                          │  JOIN on (year, state)
                                          ▼
                              training_features.csv
                              (105 rows × 19 columns)
                              year | state | yield | weather[12] | ndvi[4]
```

---

## Model Architecture

### Current Best: Random Forest + Bias Correction (model1.4)

Four `RandomForestRegressor` instances (scikit-learn), one per forecast date.
Feature sets are strictly bounded by data available at each calendar date — no future leakage.

| Model | Features Used |
|---|---|
| `aug1` | Weather May–Jul + `ndvi_aug1` |
| `sep1` | Weather May–Aug + `ndvi_aug1`, `ndvi_sep1` |
| `oct1` | Weather May–Sep + `ndvi_aug1`–`oct1` |
| `final` | Weather May–Oct + `ndvi_aug1`–`ndvi_final` |

**Train:** 2005–2020 (80 state-year pairs)
**Validate:** 2021–2024 (20 state-year pairs)
**Predict:** 2025 (5 states × 4 dates)

**Bias correction:** A per-state offset is computed from validation residuals and applied to raw RF predictions. This accounts for systematic under- or over-prediction in individual states (e.g., a state where the RF consistently predicts low due to limited training examples in a yield range).

**Uncertainty:** 500-iteration bootstrap (5th–95th percentile CI).

**Analog years:** Top 3 historical years by Euclidean distance on z-scored features.

### Model Version Comparison

| Version | Feature Set | Val RMSE (aug1) | Key Change |
|---|---|---|---|
| 1.0 | Weather + NDVI + state dummies | ~17.8 bu/acre | Baseline |
| 1.1 | + `year` as numeric feature | ~15.0 bu/acre | Captures trend |
| 1.2 | Same as 1.0, re-run | ~17.8 bu/acre | Parity check |
| **1.4** | **1.0 features + bias correction** | **~9.7 bu/acre** | **Current best** |
| 1.5 | Intermediate experiment | ~12.4 bu/acre | Superseded |

### Future: Prithvi-100M (nasa-ibm/prithvi-100m)

The hackathon prompt specifies Prithvi — NASA/IBM's open-source geospatial foundation model
available on Hugging Face — as the intended model backbone.

Prithvi replaces raw NDVI scalars with rich temporal-spectral embeddings extracted from HLS
tile stacks. A lightweight regression head (MLP or RF) trains on top of the frozen encoder.
The pipeline architecture does not change — only `05_model.ipynb` is modified. Output schema
(predictions.csv) is identical. Runs on SageMaker GPU (ml.g4dn.xlarge or larger).

---

## Training Data Schema

| Column | Source | Coverage | Notes |
|---|---|---|---|
| `year` | QuickStats | 2005–2025 | 2025 has no yield label (prediction target) |
| `state` | QuickStats | IA, CO, WI, MO, NE | One-hot encoded for model input |
| `yield_bu_acre` | QuickStats | **Y variable** | 2005–2024 only |
| `ndvi_aug1` | HLS S30 (CDL-masked) | 2016–2024 | Mean NDVI over corn pixels, ±7 day window |
| `ndvi_sep1` | HLS S30 (CDL-masked) | 2016–2024 | Same methodology |
| `ndvi_oct1` | HLS S30 (CDL-masked) | 2016–2024 | Same methodology |
| `ndvi_final` | HLS S30 (CDL-masked) | 2016–2024 | Same methodology |
| `tavg_may` – `tavg_oct` | NOAA GSOM | 2005–2025 | State-level station average, °F |
| `prcp_may` – `prcp_oct` | NOAA GSOM | 2005–2025 | State-level station average, inches |

> **Note on NDVI coverage:** HLS S30 (30m resolution) is available from 2015–2024. Years with <2 cloud-free tiles in a ±7-day window were filled by interpolation from adjacent forecast dates. 2015 had insufficient coverage across all states and was dropped, leaving 2016–2024 (9 years × 5 states = 45 rows).

---

## Data Sources

| Dataset | API / Access | Purpose | Status |
|---|---|---|---|
| USDA NASS QuickStats | `quickstats.nass.usda.gov/api` | Yield labels (Y variable) | ✅ Complete |
| NOAA Global Summary of the Month | `ncei.noaa.gov/cdo-web/api/v2` | Weather features | ✅ Complete |
| NASA HLS S30 | NASA CMR STAC + Earthdata auth | Satellite NDVI features | ✅ Complete |
| USDA Cropland Data Layer | Pre-clipped local TIF | Corn pixel mask | ✅ Integrated |
| Prithvi-100M | `huggingface.co/nasa-ibm/prithvi-100m` | Geospatial embeddings (future) | Planned |

---

## Infrastructure

### Local (most notebooks)
- Environment: `geospatial-python-crash-course` conda env
- Hardware: standard laptop — no GPU required
- Notebooks: 01, 02, 04, 05, 06

### SageMaker (satellite notebook only)
- Instance: ml.g4dn.xlarge or larger (for Prithvi); ml.t3.medium sufficient for HLS NDVI
- Auth: NASA Earthdata credentials via IAM role → `.netrc` in home directory
- Notebook: 03_satellite_fixed only
- **Output CSV is committed to repo** — teammates do not need SageMaker access

---

## Output Schema: predictions.csv

| Column | Type | Description |
|---|---|---|
| `state` | string | State name |
| `forecast_date` | string | `aug1`, `sep1`, `oct1`, `final` |
| `predicted_yield` | float | Point estimate (bu/acre), bias-corrected |
| `predicted_yield_raw` | float | RF prediction before bias correction |
| `bias_correction` | float | Per-state offset applied |
| `ci_lower` | float | 5th percentile bootstrap CI |
| `ci_upper` | float | 95th percentile bootstrap CI |
| `analog_years` | string | Top 3 nearest historical years |
| `val_rmse_pre` | float | Validation RMSE before bias correction |
