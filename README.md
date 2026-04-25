# Geospatial AI Crop Yield Forecasting — Prithvi-EO-2.0-600M-TL

CSU Hackathon 2026, Prompt 2. State-level corn-grain yield forecasts for **Iowa,
Colorado, Wisconsin, Missouri, Nebraska** at four 2025-season checkpoints
(Aug 1 / Sep 1 / Oct 1 / end-of-season), with a cone of uncertainty derived
from analog-year retrieval over historical weather + drought + vegetation signals.

## Pipeline at a glance

```
HLS (2015-2025) ─┐
CDL corn mask ───┼─→ Prithvi-EO-2.0-600M-TL (LoRA fine-tune) ──→ point yield (bu/ac)
TIGER counties ──┘                                                   │
                                                                     ▼
Landsat C2 (2005-2014) ─┐                                    cone of uncertainty
gridMET ────────────────┼─→ analog-year features ──→ k-NN ──→ p10/p25/p50/p75/p90
USDM ───────────────────┤
gNATSGO soils ──────────┘
NASS QuickStats yield ─→ training labels + benchmark
```

Trained locally on **DGX Spark** (128 GB unified memory, Blackwell, BF16). With LoRA
(r=16) we update ~0.3% of the 600M params, trains in ~4 hours per fold.

## Data sources (verified)

| Source | Use | Years | Access |
|---|---|---|---|
| [Prithvi-EO-2.0-600M-TL](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL) | Backbone | – | Hugging Face |
| [HLS L30/S30 v2.0](https://lpdaac.usgs.gov/products/hlsl30v002/) | 6-band Prithvi input | 2015–2025 | GEE `NASA/HLS/HLSL30/v002`, `HLSS30/v002` |
| [USDA CDL](https://nassgeodata.gmu.edu/CropScape/) | Corn pixel mask | 2008–2024 | GEE `USDA/NASS/CDL` |
| [USDA NASS QuickStats](https://quickstats.nass.usda.gov/api) | Yield labels + crop progress | 2005–2025 | REST API (free key) |
| [gridMET](https://www.climatologylab.org/gridmet.html) | Weather features | 2005–2025 | GEE `IDAHO_EPSCOR/GRIDMET` |
| [Landsat C2 L2](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products) | Pre-2015 NDVI backfill | 2005–2014 | GEE `LANDSAT/{LT05,LE07,LC08}/C02/T1_L2` |
| [US Drought Monitor](https://droughtmonitor.unl.edu/DmData/GISData.aspx) | Drought severity (DSCI) | 2005–2025 | USDM CountyStatistics REST |
| [gNATSGO](https://www.nrcs.usda.gov/resources/data-and-reports/gridded-national-soil-survey-geographic-database-gnatsgo) | Soil features (AWC, OM, clay) | static | GEE `projects/sat-io/open-datasets/CSRL_soil_properties/*` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # fill in NASS_API_KEY, EE_PROJECT, HF_TOKEN
earthengine authenticate
huggingface-cli login
```

## Run order

### One command to download all data
```bash
python scripts/download_all.py                  # everything not yet present
python scripts/download_all.py --skip hls       # everything except the overnight chip export
python scripts/download_all.py --only nass usdm # just specific steps
python scripts/download_all.py --dry-run        # see the plan without running
```
The orchestrator is idempotent (skips steps whose output already exists), validates
env vars and GEE auth up front, and reports per-step wall time.

### Or step-by-step
```bash
# Data
python scripts/data/fetch_nass.py
python scripts/data/fetch_usdm.py
python scripts/data/fetch_gnatsgo.py
python scripts/data/export_gridmet.py
python scripts/data/export_landsat_c2_backfill.py
python scripts/data/export_hls_ndvi.py
python scripts/data/export_hls_chips.py        # overnight

# Tables + sanity check
python scripts/training/build_labels_metadata.py
python scripts/training/build_features.py
python notebooks/01_sanity_check.py

# Train Prithvi + LoRA on DGX Spark
terratorch fit --config configs/terratorch_lora.yaml

# Inference + uncertainty cone for 2025
python scripts/inference/predict_2025.py --ckpt models/checkpoints/prithvi_yield/best.ckpt
python scripts/training/analog_year_uncertainty.py
```

## Forecast outputs

`reports/forecasts/yield_with_uncertainty_2025.parquet` — one row per
(state × checkpoint) with `point`, `p10`, `p25`, `p50`, `p75`, `p90`, `analog_years`.

## Project layout

```
configs/                project.yaml, terratorch_lora.yaml
scripts/
  data/                 fetch_*.py, export_*.py    # one script per source
  training/             dataset.py, build_labels_metadata.py, analog_year_uncertainty.py
  inference/            predict_2025.py
data/
  raw/                  one subdir per source (nass/, cdl/, hls/, ...)
  processed/
    chips/{state}/{fips}/{year}/{checkpoint}.zarr
    labels/             county_yield.parquet, chip_metadata.parquet
    features/           state_checkpoint_features.parquet
models/
  checkpoints/          PyTorch Lightning checkpoints
  lora_adapters/        exported LoRA weights for re-use
reports/
  forecasts/            *.parquet
  figures/              presentation plots
```

## Design notes

- **County-level training, state-level reporting.** State-level training has only
  100 samples (5 states × 20 years) — too few for ML. County-level gives ~9.4k
  rows (~470 corn counties × 20 years) for the regression head. Aggregate to state
  via CDL-acreage-weighted means at inference time.
- **LoRA over full fine-tune.** 4.7k HLS-era county-years would overfit a 600M
  backbone. LoRA r=16 on `qkv` and `proj` updates ~2M params — generalization-friendly.
- **Pre-2015 backfill is for analog matching only.** Landsat C2 NDVI extends the
  analog-year pool from 10 → 20 years. Prithvi never sees Landsat — keeps the
  6-band HLS input distribution intact.
- **2025 corn mask uses CDL 2024.** 2025 CDL is released after the season ends.
  Year-over-year corn pixel agreement is ~85%.
- **TL (Temporal-Location) variant.** Prithvi-2.0-TL takes lat/lon + acquisition
  dates as auxiliary inputs. Plumbing in [scripts/training/dataset.py](scripts/training/dataset.py)
  passes county centroid + median composite date for each of the 3 timesteps.
- **Cone of uncertainty.** k-NN (k=5) over standardized weather+USDM+NDVI features
  retrieves the closest historical seasons; the empirical p10–p90 of those years'
  actual NASS yield deviations becomes the cone, re-anchored to the model's point
  forecast.
