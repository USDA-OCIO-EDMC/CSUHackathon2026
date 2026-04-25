# Phase 3 Workflow: Prithvi Embeddings + Feature Fusion

## Pipeline Overview

```
                    HLS Satellite Imagery (NASA S3)
                            ↓
                  hls_downloader.py
                  (search + download)
                            ↓
        ┌─────────────────────────────────────┐
        │   prithvi_pipeline.py               │
        │  (Extract 768-dim embeddings)       │
        │   - Download HLS granules           │
        │   - Extract Prithvi features        │
        │   - Map to counties/months          │
        │   - Output: prithvi_embeddings_*.   │
        │             parquet                 │
        └─────────────────────────────────────┘
                            ↓
        ┌─────────────────────────────────────┐
        │   feature_fusion.py                 │
        │  (Combine with weather)             │
        │   - Load Prithvi embeddings         │
        │   - Fetch weather vectors           │
        │   - Concatenate: [W|P]              │
        │   - Output: master_features.        │
        │             parquet                 │
        └─────────────────────────────────────┘
                            ↓
        ┌─────────────────────────────────────┐
        │   forecaster.py                     │
        │  (Identify analog years)            │
        │   - Load master_features            │
        │   - Compare with 2025               │
        │   - Return top 5 analog years       │
        │   - Generate yield uncertainty cone │
        └─────────────────────────────────────┘
```

---

## Step-by-Step Usage

### **Step 1: Extract Prithvi Embeddings**

Extracts 768-dimensional embeddings from HLS satellite imagery for each county/month combination.

```bash
# Single state
python src/prithvi_pipeline.py --state IA --year 2024 --device cpu

# All 5 states (combine flag will be added after)
python src/prithvi_pipeline.py --state IA --year 2024 --device cpu
python src/prithvi_pipeline.py --state CO --year 2024 --device cpu
python src/prithvi_pipeline.py --state WI --year 2024 --device cpu
python src/prithvi_pipeline.py --state MO --year 2024 --device cpu
python src/prithvi_pipeline.py --state NE --year 2024 --device cpu
```

**Output:** `data/processed/prithvi_embeddings_IA_2024.parquet` (one per state)

**Format:**
```
year | month | county      | fips  | state | prithvi_embedding
2024 |   5   | STORY       | 19193 | IA    | [768-dim array]
2024 |   6   | STORY       | 19193 | IA    | [768-dim array]
...
```

---

### **Step 2: Fuse with Weather Vectors**

Combines Prithvi embeddings (768-dim) with weather vectors (18-dim) → 786-dim unified embeddings.

```bash
# Single state
python src/feature_fusion.py --state IA --year 2024

# Combine all states into master_features.parquet
python src/feature_fusion.py --state IA --year 2024 --combine
```

**Output:** 
- Per-state: `data/processed/fused_features_IA_2024.parquet`
- Combined: `data/processed/master_features.parquet` ← **Used by forecaster.py**

**Format:**
```
year | month | county | fips  | state | prithvi_embedding | weather_vector | embedding_vector
2024 |   5   | STORY  | 19193 | IA    | [768-d]          | [18-d]         | [786-d] ← concatenated
```

---

### **Step 3: Run Analog Year Identification**

Uses `master_features.parquet` to identify analog years and generate yield uncertainty cones.

```python
import pandas as pd
from src.forecaster import get_analog_years

# Load data
feature_df = pd.read_parquet("Hackathon2026/data/processed/master_features.parquet")
yield_df = pd.read_csv("Hackathon2026/data/raw/corn_yield_19_2005_2024.csv")

# Get analog years for Story County, IA on Aug 1 forecast
result = get_analog_years(
    state_fips="19",
    county_name="STORY",
    forecast_date="aug1",
    feature_df=feature_df,
    yield_df=yield_df,
)

print(result)
# Output:
#    year  similarity  yield_bu_acre
# 0  2012      0.9543          172.5
# 1  2008      0.9234          168.3
# 2  2015      0.9102          175.1
# 3  2019      0.8987          170.2
# 4  2011      0.8765          169.8

# Uncertainty cone: [168.3, 175.1] bu/acre
```

---

## How It Works

### **Prithvi Pipeline** (`prithvi_pipeline.py`)

1. **Search HLS Granules** 
   - Queries NASA CMR for satellite tiles covering the state
   - For specific year + forecast_date window

2. **Download via Cloud Direct**
   - Streams HLS 6 bands directly from NASA S3
   - No local storage needed (uses `hls_downloader.py`)

3. **Extract Embeddings**
   - Loads Prithvi-100M model
   - Processes each granule through Prithvi
   - Returns 768-dim vector per granule

4. **Map to Counties**
   - Checks if granule bbox overlaps with county buffer
   - Averages multiple granules per county/month
   - Stores with `[year, month, county]` keys

### **Feature Fusion** (`feature_fusion.py`)

1. **Load Prithvi Embeddings**
   - Reads `prithvi_embeddings_IA_2024.parquet`

2. **Fetch Weather Vectors**
   - Calls `analog_years.py`'s `get_state_weather_features()`
   - Gets 18-dim vectors: [temp, precip, GDD] × 6 months

3. **Concatenate**
   - Combines: `[weather_18 + prithvi_768]` → 786-dim embedding

4. **Save Master Features**
   - Stores all counties/months/states
   - Format expected by `get_analog_years()`

---

## Data Flow Example

### Input (Year 2024, Iowa)
- **HLS Granules:** 50 satellite tiles covering Iowa (May-Oct 2024)
- **Counties:** 6 test counties (Story, Black Hawk, Grundy, etc.)

### Processing
1. **Prithvi Pipeline:**
   - Extract embeddings from 50 granules → 768-dim each
   - Match to 6 counties × 6 months = 36 (county, month) pairs
   - Output: 36 records with 768-dim embeddings

2. **Feature Fusion:**
   - Load 36 Prithvi embeddings (768-dim)
   - Fetch weather for all 6 counties (18-dim)
   - Combine: 36 records × 786-dim embeddings
   - Save to `master_features.parquet`

3. **Analog Years:**
   - Load 2024 data: Story County, Aug 1 → 786-dim vector
   - Compare with historical (2005-2023) → cosine similarity
   - Return top 5 years with highest similarity
   - Extract their yields → uncertainty cone

### Output
```
Top 5 Analog Years for Story County, IA (Aug 1, 2024):
Year 2012 (sim=0.954): 172.5 bu/acre
Year 2008 (sim=0.923): 168.3 bu/acre
Year 2015 (sim=0.910): 175.1 bu/acre
Year 2019 (sim=0.899): 170.2 bu/acre
Year 2011 (sim=0.877): 169.8 bu/acre

FORECAST for 2024: 170.2 ± 3.4 bu/acre (90% confidence cone)
```

---

## Key Dimensions

| Component | Dimensions | Notes |
|-----------|-----------|-------|
| Weather Vector | 18 | Mean temp, precip, GDD × 6 months |
| Prithvi Embedding | 768 | Spatial features from satellite imagery |
| Combined Embedding | **786** | What `get_analog_years()` uses |
| Top Analog Years | 5 | Configurable in forecaster.py |

---

## Common Issues & Solutions

### Issue: "No granules found"
- **Cause:** HLS coverage gap or wrong forecast_date window
- **Solution:** Use `--all-states` flag or check NASA CMR directly

### Issue: "EARTHDATA_USERNAME not set"
- **Cause:** Environment variables not configured
- **Solution:**
  ```bash
  export EARTHDATA_USERNAME=your_username
  export EARTHDATA_PASSWORD=your_password
  ```
  Register at: https://urs.earthdata.nasa.gov/

### Issue: Out of memory (OOM) on GPU
- **Solution:** Use `--device cpu` or reduce `max_tiles` in `prithvi_pipeline.py`

---

## Next Steps

1. **Run the full pipeline:**
   ```bash
   # Extract Prithvi for all states
   for state in IA CO WI MO NE; do
     python src/prithvi_pipeline.py --state $state --year 2024 --device cuda
   done
   
   # Fuse with weather for all states
   python src/feature_fusion.py --state IA --year 2024 --combine
   
   # Run analog year identification
   python src/forecaster.py  # or use jupyter notebook
   ```

2. **Generate yield forecasts** for all 4 forecast dates:
   - Aug 1, Sep 1, Oct 1, Final

3. **Build interactive dashboard:**
   - Use outputs for `corn_yield_hackathon_dashboard.html`
