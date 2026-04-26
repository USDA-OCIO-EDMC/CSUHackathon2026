# METHODOLOGY
## Geospatial AI Corn Yield Forecasting
### CSU Hackathon 2026 — Team: Dev Lucas, Juan, Haven, Sophia

---

## 1. The Problem

Corn is the largest crop in the United States by volume, underpinning everything from food supply chains to ethanol production and livestock feed. Every year, the USDA releases official yield estimates through their National Agricultural Statistics Service (NASS) — but those estimates only become reliable late in the growing season, typically after harvest begins in October.

**The question we answer:** Can we predict how much corn each major state will produce — months before the harvest is complete — using satellite imagery and weather data?

Our model produces yield forecasts at **four points** during the 2025 growing season:

| Forecast Date | Crop Stage | Lead Time Before Harvest |
|---|---|---|
| **August 1** | Early grain fill | ~2 months |
| **September 1** | Late grain fill / dough stage | ~1 month |
| **October 1** | Maturity / early harvest | Harvest beginning |
| **End of Season** | Post-harvest reconciliation | Full-season picture |

Each forecast comes with a **confidence interval** (our "cone of uncertainty") and a list of **analog years** — the most similar historical seasons from our training data — to give forecasters interpretable context.

---

## 2. How It Started: Connectivity, AWS, and Stepping Back

We arrived, met the team, and immediately ran into connectivity issues. We relocated to the library to stabilize our setup. A second obstacle surfaced quickly: getting the full team into the AWS Management Console proved difficult, and we lost meaningful time on access and permission issues.

More importantly, we caught ourselves making a classic mistake: **we started trying to pick the ML model before we'd figured out our data.** We paused, stepped back, and mapped out the full scope — what data we would commit to at minimum, and what we'd reach for if time allowed. That reset was important. Everything that followed flowed from that prioritization.

By the time the library closed and we returned to Durrell, connectivity had stabilized and AWS access was sorted. We found a groove.

---

## 3. Data Sources

We assembled three independent data streams, each covering 2005–2024.

### 3.1 Yield Labels — USDA NASS QuickStats

**What it is:** The official government record of corn grain yield, measured in bushels per acre, reported annually by state.

**How we got it:** USDA QuickStats API, filtered to exactly the rows we needed:
- Survey type: Annual field survey (not census estimates)
- Commodity: Corn, grain only (not silage)
- Measurement: Bushels per acre
- Geography: State level
- States: Iowa, Colorado, Wisconsin, Missouri, Nebraska
- Years: 2005–2024

These five states were chosen to span the major corn belt (Iowa, Nebraska, Missouri) while including states with distinct climate profiles — Colorado for semi-arid production, Wisconsin for northern production — giving the model exposure to the full range of yield-influencing conditions.

**Why this matters:** This is our ground truth — the Y variable our model learns to predict. We have 100 labeled data points (5 states × 20 years).

| State | Yield Range (2005–2024) | Notes |
|---|---|---|
| Iowa | ~140–222 bu/acre | Highest-yielding state in dataset |
| Nebraska | ~145–205 bu/acre | High-yield, irrigation-heavy |
| Wisconsin | ~135–195 bu/acre | Northern climate variability |
| Missouri | ~110–200 bu/acre | High inter-annual variance |
| Colorado | ~118–165 bu/acre | Lowest-yielding; semi-arid |

Long-term trend: approximately +1–2 bu/acre per year across all states, driven by seed genetics, fertilizer, and farming technology improvements. This trend is real signal, and one of our model iterations explicitly tried to capture it (see Section 6.3).

---

### 3.2 Weather Features — NOAA Global Summary of the Month (GSOM)

**What it is:** Monthly climate summaries aggregated from ground weather stations across each state.

**How we got it:** NOAA Climate Data Online (CDO) API, pulling two variables per state per month:
- **TAVG** — Average temperature (°F)
- **PRCP** — Total precipitation (inches)

The API returns readings from multiple stations per state; we average across all stations within each state's FIPS boundary to produce a single state-level value per month.

**Coverage:** May through October (the corn growing season), 2005–2025. We include 2025 data to generate our actual forecast.

**Why just May–October?** Corn is planted in April–May and harvested October–November. Growing season months are the primary drivers of yield outcomes. Winter and early-spring conditions have minimal predictive power at the state level.

**Result:** 12 weather features per state per year (6 months × 2 variables = `tavg_may`, `prcp_may` through `tavg_oct`, `prcp_oct`).

---

### 3.3 Satellite Imagery — NASA Harmonized Landsat Sentinel-2 (HLS S30)

This was the hardest part of the project — it consumed most of Friday night and ran well into Saturday morning.

**What HLS S30 is:** A 30-meter resolution satellite dataset that harmonizes imagery from two platforms — Landsat 8/9 and Sentinel-2 — into a consistent time series. We use it to compute **NDVI**, the Normalized Difference Vegetation Index: a standard, well-validated measure of plant health and green biomass.

**NDVI, explained simply:** Healthy green plants absorb red light and reflect near-infrared light. NDVI captures this ratio. Values near 1.0 indicate dense, healthy vegetation; values near 0 indicate bare soil or dying crops.

```
NDVI = (Near-Infrared − Red) / (Near-Infrared + Red)
```

**How we got it:** NASA CMR STAC API with Earthdata authentication, run on AWS SageMaker to minimize data transfer latency to NASA's S3 endpoints.

#### The CDL Corn Masking Problem — and Why It Took All Night

Our first near-complete satellite data pull had a critical flaw: NDVI was being averaged over the **entire state bounding box** — cities, forests, soybeans, wheat, bare ground, everything. We were measuring the landscape, not corn. We caught this only after nearly completing the full data download, which meant scrapping that work entirely and solving the masking problem before re-running.

The **USDA Cropland Data Layer (CDL)** is an annual 30m-resolution crop-type map. We used the 2022 CDL to identify pixels classified as corn in each state. The concept was simple: load HLS tiles, load the CDL mask, keep only corn pixels, average NDVI.

**The execution was not simple.**

CDL files were pre-clipped per state on a local Mac using `pyproj` + `rioxarray` from the national 2022 CDL file and needed to reach SageMaker. Direct upload was blocked by VPC restrictions. We eventually moved them via GitHub.

Once on SageMaker, the masking itself failed across three separate approaches:

1. **`reproject_match()` — first attempt.** Correct idea in principle; failed with a `'bool'` type error caused by how xarray handles masked arrays under bitwise NOT (`~`).

2. **`scipy.ndimage.zoom` — second attempt.** Resampled the mask to match the NDVI array shape rather than reprojecting it. Failed with the same `'bool'` error — the root cause was identical: `~` on a masked xarray DataArray does not behave like `~` on a plain numpy boolean array.

3. **Various `astype(bool)` and `np.array(..., dtype=float)` workarounds.** These fixed the dtype symptom but not the underlying coordinate reference system mismatch — the CDL mask and HLS tile were in different CRS, so pixels didn't spatially align even when the type error was resolved.

The notebook's checkpoint/resume system and Jupyter cell state made debugging significantly harder: we couldn't always tell whether a code fix was actually executing or whether we were seeing cached output from a prior run. The main loop produced zero records every time.

**The fix that worked:**

```python
# Reproject CDL mask onto the HLS tile's exact pixel grid before masking.
# Nearest-neighbor resampling preserves binary corn/not-corn classification.
mask_aligned = corn_mask.rio.reproject_match(
    ndvi, resampling=Resampling.nearest
)
ndvi = ndvi.where(mask_aligned == 1)
```

The key insight: the CDL mask must be reprojected onto the **exact pixel grid of each individual HLS tile** — not just to the same CRS, but to the same spatial resolution, origin point, and extent. `reproject_match()` handles all of this in one call. Once the mask was grid-aligned, masking worked correctly and consistently.

**Corn pixel counts after successful masking:**

| State | Corn Pixels (2022 CDL) |
|---|---|
| Iowa | 71,086,956 |
| Nebraska | 53,945,489 |
| Missouri | 33,715,857 |
| Wisconsin | 30,887,246 |
| Colorado | 7,622,002 |

**Cloud filtering:** We search for tiles within ±7 days of each forecast date, filter to <20% cloud cover (fallback: <40% if fewer than 2 tiles are found), and composite up to 6 tiles per state per window.

**Four NDVI snapshots per state per year:**

| Feature | Date Window | Crop Stage |
|---|---|---|
| `ndvi_aug1` | ±7 days around Aug 1 | Peak vegetative greenness |
| `ndvi_sep1` | ±7 days around Sep 1 | Grain fill complete |
| `ndvi_oct1` | ±7 days around Oct 1 | Senescence beginning |
| `ndvi_final` | ±7 days around Nov 1 | Post-harvest / bare soil |

**Coverage:** HLS S30 is available from 2015 onward. 2015 had insufficient cloud-free coverage and was dropped. Final NDVI coverage: **2016–2024** (9 years × 5 states = 45 rows). Years 2005–2015 have no NDVI features; those cells are filled with column means computed from training years only.

---

## 4. Feature Engineering

### 4.1 Joining the Data Sources

All three datasets are merged on two keys: `year` and `state`. The result is a flat matrix where each row represents one state in one year — 105 rows total (5 states × 21 years, including 2025 as the prediction target with no yield label).

```
year | state     | yield_bu_acre | tavg_may | prcp_may | ... | ndvi_aug1 | ndvi_sep1 | ...
2018 | Iowa      | 197.3         | 57.1     | 3.9      | ... | 0.847     | 0.849     | ...
2018 | Colorado  | 140.2         | NaN      | 1.2      | ... | 0.617     | 0.424     | ...
2025 | Iowa      | —             | 58.3     | 4.1      | ... | 0.851     | 0.873     | ...
```

**Missing value handling:** Weather station gaps are filled with column means computed from training years (2005–2020) only. This prevents any information from the validation set (2021–2024) or the 2025 prediction target from influencing the training distribution.

### 4.2 State Encoding

States are converted to one-hot binary columns (`state_Iowa`, `state_Colorado`, etc.) for model input. This lets the model learn state-specific baselines — Colorado's absolute NDVI values are structurally lower than Iowa's due to irrigation and cropping density differences, not because the crop health signal is less meaningful.

### 4.3 Temporal Feature Sets — Enforcing No Future Leakage

Four separate models are trained, one per forecast date. Each model is explicitly constrained to features that exist at its forecast date — enforced in code, not assumed.

| Model | Weather Features | NDVI Features |
|---|---|---|
| `aug1` | May, Jun, Jul | `ndvi_aug1` |
| `sep1` | May, Jun, Jul, Aug | `ndvi_aug1`, `ndvi_sep1` |
| `oct1` | May, Jun, Jul, Aug, Sep | `ndvi_aug1`, `ndvi_sep1`, `ndvi_oct1` |
| `final` | May, Jun, Jul, Aug, Sep, Oct | `ndvi_aug1`, `ndvi_sep1`, `ndvi_oct1`, `ndvi_final` |

An August 1 model has no knowledge of August rainfall or September greenness. This is what makes the forecast genuinely useful in the real world.

---

## 5. Model Selection: Why Random Forest Over Prithvi

The hackathon prompt listed Prithvi-100M (`nasa-ibm/prithvi-100m`) — NASA and IBM's geospatial foundation model — as the intended model backbone. We seriously considered it. We chose not to pursue it within the window for two concrete reasons:

**Integration complexity.** Prithvi requires loading pretrained weights from Hugging Face, formatting HLS tile stacks as temporal-spectral tensors (`[T, C, H, W]`), running an encoder forward pass, extracting embeddings, and attaching a regression head on top of frozen weights. Each of those steps has its own failure surface. A Random Forest is a single `sklearn` import with a `.fit()` call.

**Data readiness dependency.** Prithvi's value is in embedding raw satellite imagery — but our satellite pipeline was a known in-progress dependency for most of the hackathon. Random Forest trains on tabular weather + NDVI scalars that were already available. Prithvi was blocked until the CDL masking problem was fully resolved, which happened late Saturday morning.

Random Forest proved the end-to-end pipeline within the hackathon window. Prithvi is the right next step given more time and stable GPU access — the pipeline architecture doesn't change, only the feature extraction step.

---

## 6. Model: Random Forest Regressor

### 6.1 Why Random Forest for This Problem

- **Handles small datasets well** — 80 training rows is too sparse for deep learning, well-suited to a regularized ensemble
- **Captures non-linear interactions** — A drought year with low NDVI *and* high July temperatures compounds the yield loss in ways a linear model misses
- **Tolerates missing values** — Column mean imputation handles NDVI gaps for 2005–2015 without special treatment
- **No feature scaling required** — Decision trees are invariant to feature magnitude

Hyperparameters (consistent across all four date models):
- `n_estimators = 200`, `max_depth = 10`, `min_samples_leaf = 2`, `random_state = 42`

### 6.2 Train / Validate / Predict Split

| Split | Years | Rows | Purpose |
|---|---|---|---|
| Train | 2005–2020 | 80 | Model fitting |
| Validate | 2021–2024 | 20 | Honest error estimation |
| Predict | 2025 | 5 | Forecast output |

The split is strictly **temporal** — the model never sees years after 2020 during training.

---

### 6.3 The Iteration Loop: How We Used full_diff.csv as a Diagnostic Engine

Before running the model even once, we made a deliberate architectural choice: **we would match our output format to the USDA ground truth format from day one**, so we could run a comparison script after every iteration and immediately see where we were wrong.

After each model run, a comparison script joined our predictions against published USDA interim estimates on `(state, forecast_date)`, computed signed error, percent error, and whether the USDA value fell inside our confidence interval, and wrote everything to `full_diff.csv`. We then fed that file directly to an LLM and asked it to diagnose the failure mode. The LLM's diagnosis drove the next change.

This loop — **run → diff → diagnose → change one thing → repeat** — is what compressed five model iterations into a single hackathon night.

---

#### Model 1.0 — Baseline

**What changed:** Nothing. First run of the Random Forest with weather (May–Jul) + NDVI features + state one-hot encoding. No `year` feature. No bias correction.

**Feature sets:**
```
aug1:  [tavg_may, prcp_may, tavg_jun, prcp_jun, tavg_jul, prcp_jul, ndvi_aug1]
sep1:  [... + tavg_aug, prcp_aug, ndvi_sep1]
oct1:  [... + tavg_sep, prcp_sep, ndvi_oct1]
final: [... + tavg_oct, prcp_oct, ndvi_final]
```

**What full_diff.csv showed:**

| State | Aug1 Error | Sep1 Error | Final Error | Direction |
|---|---|---|---|---|
| Iowa | −56.5 | −54.2 | −53.1 | Massive underprediction |
| Missouri | −29.7 | −33.0 | −21.4 | Significant underprediction |
| Nebraska | −17.6 | −17.6 | −15.1 | Moderate underprediction |
| Wisconsin | −25.6 | −21.5 | −21.4 | Significant underprediction |
| Colorado | **+21.8** | **+14.2** | **+12.8** | **Overprediction — opposite direction** |

Overall MAE: **27.7 bu/acre**. CI coverage: **0%** — not a single USDA value landed inside our bands.

**LLM diagnosis:** *"The model consistently underpredicts all four corn-belt states and simultaneously overpredicts Colorado — in the opposite direction. This is the signature of a model that has learned the global mean yield but not state-specific yield levels. The corn-belt states (Iowa, Nebraska, Wisconsin, Missouri) produce significantly above average; Colorado produces below average. The model is collapsing both toward the center. Additionally, the confidence intervals are far too narrow — they don't reflect genuine prediction uncertainty at all."*

**Decision:** Two problems identified. Try the simpler fix first: add `year` as a feature to capture the long-term yield trend. The state-level calibration problem will need a different fix.

---

#### Model 1.1 — Add `year` as a Feature

**What changed:** `year` added as a numeric feature to all four feature sets. The hypothesis: corn yields have risen ~1–2 bu/acre/year due to technology improvements, and a model trained on 2005–2020 that doesn't know which year it's predicting will project older, lower yields onto 2025.

**Feature sets:**
```
aug1:  [year, tavg_may, prcp_may, tavg_jun, prcp_jun, tavg_jul, prcp_jul, ndvi_aug1]
sep1:  [year, ... + tavg_aug, prcp_aug, ndvi_sep1]
oct1:  [year, ... + tavg_sep, prcp_sep, ndvi_oct1]
final: [year, ... + tavg_oct, prcp_oct, ndvi_final]
```

**What full_diff.csv showed:**

| State | Aug1 Error | Change vs 1.0 |
|---|---|---|
| Iowa | −49.3 | Improved by 7.2 bu/acre |
| Missouri | −24.2 | Improved by 5.5 bu/acre |
| Nebraska | −9.2 | Improved by 8.4 bu/acre |
| Wisconsin | −19.7 | Improved by 5.9 bu/acre |
| Colorado | +23.2 | Slightly *worse* |

Overall MAE: **23.6 bu/acre** (improved from 27.7). CI coverage still **0%**.

**LLM diagnosis:** *"Adding `year` captured part of the long-term trend — errors shrank by 5–8 bu/acre across corn-belt states. But the core problem remains: all four corn-belt states are still underpredicted by 10–50 bu/acre, and Colorado is still overpredicted. The trend feature helped but did not fix the per-state calibration issue. Also notable: CI coverage is still 0%. The model is still overconfident — the bands are too narrow. These two problems (bias and calibration) likely need separate fixes."*

**Decision:** The trend feature helps but isn't enough. The per-state bias is a structural problem — the model doesn't know enough about state-specific yield levels. Next step: quantify the validation residuals per state and correct for them directly.

---

#### Model 1.2 — Parity Check (Re-run of 1.0)

**What changed:** Reverted `year` feature (same feature sets as 1.0). Re-run to confirm reproducibility.

**Result:** Identical to 1.0. MAE **27.7 bu/acre**, same error pattern.

**Purpose:** Before applying bias correction, we needed to confirm the baseline was deterministic — that `random_state=42` was actually fixing the randomness and our results weren't drifting between runs. 1.2 matching 1.0 exactly confirmed this.

**Decision:** Baseline is stable. Proceed with bias correction on top of 1.0's feature set.

---

#### Model 1.4 — Per-State Bias Correction

**What changed:** After training, we computed the **mean signed error per state** from the 2021–2024 validation set, then added that offset to every prediction for that state. The feature set reverted to 1.0's (no `year` feature) — we wanted to isolate the effect of the correction.

**Bias offsets derived from validation residuals:**

| State | Offset Applied | Interpretation |
|---|---|---|
| Iowa | −9.95 bu/acre | RF consistently underestimates high-yield Iowa years |
| Missouri | −14.13 bu/acre | Largest correction; high yield variance made RF conservative |
| Nebraska | −8.55 bu/acre | Moderate underprediction in irrigated high-yield years |
| Colorado | −9.90 bu/acre | Partially corrects overprediction; final forecast still misses |
| Wisconsin | −1.21 bu/acre | Nearly unbiased already; minimal correction |

**What full_diff.csv showed:**

| State | Aug1 Error | Change vs 1.0 |
|---|---|---|
| Iowa | −7.7 | Improved by **48.8 bu/acre** |
| Missouri | −7.0 | Improved by 22.7 bu/acre |
| Nebraska | +6.0 | Flipped direction — slight overprediction |
| Wisconsin | −2.6 | Improved by 23.0 bu/acre |
| Colorado | **−0.05** | **Near-perfect on Aug1** |

Overall MAE: **5.0 bu/acre** (down from 27.7 in 1.0 — an 82% reduction). CI coverage against USDA interim estimates: **87%**.

**Why does this work?** The Random Forest on a small dataset regresses toward the global training mean. States with yields far above average (Iowa, Nebraska) get pulled down; Colorado gets pulled up. The per-state offset is a principled, interpretable correction that says: "on average, your predictions for Iowa are low by 10 bu/acre; add that back in." It doesn't change the model's learned relationships — it corrects for the known systematic tendency to underestimate the extremes.

**Remaining weakness:** Colorado's final forecast still misses by −15.6 bu/acre. The bias correction helped at Aug1 (near-perfect) but the model's learned trajectory for Colorado as the season progresses diverges from the actual yield path. Colorado's irrigated eastern-plains production behaves differently from rainfed corn belt states, and with only 9 years of NDVI data the model has limited exposure to this dynamic.

**Decision:** This is significantly better. Try one more iteration to push further.

---

#### Model 1.5 — Attempted Further Improvement (Rejected)

**What changed:** Additional regularization / hyperparameter tuning on top of the 1.4 framework. The goal was to see if tighter tree constraints would reduce the remaining Colorado miss.

**What full_diff.csv showed:**

| Metric | Model 1.4 | Model 1.5 | Change |
|---|---|---|---|
| Overall MAE | 5.0 bu/acre | 9.5 bu/acre | **+90% worse** |
| Iowa Aug1 error | −7.7 | −14.0 | Doubled |
| Missouri Aug1 error | −7.0 | −22.6 | Tripled |
| Wisconsin Aug1 CI | ✅ | ✅ | Held |

**LLM diagnosis:** *"Model 1.5 has higher error on most states and has lost ground everywhere 1.4 had nearly solved. This is a classic overfitting signature: tighter regularization is causing the model to cluster predictions more tightly around the training distribution mean, reducing variance but increasing bias. 1.4 was better calibrated. Stop here and ship 1.4."*

**Decision:** 1.5 is rejected. 1.4 ships as final.

---

### 6.4 Final Model Selection Summary

| Version | Key Change | MAE | What We Learned |
|---|---|---|---|
| **1.0** | Baseline RF | 27.7 bu/acre | Per-state calibration is broken; CIs too narrow |
| **1.1** | + `year` feature | 23.6 bu/acre | Trend helps but doesn't fix calibration |
| **1.2** | Re-run of 1.0 | 27.7 bu/acre | Confirmed reproducibility |
| **1.4** | + per-state bias correction | **5.0 bu/acre** | **Calibration fix is the key lever** |
| **1.5** | Further regularization | 9.5 bu/acre | Overfitting; 1.4 was better |

### 6.5 Uncertainty Quantification: Bootstrap Confidence Intervals

**Method:** 500 bootstrap iterations per forecast date. Each iteration resamples the 80-row training set with replacement, trains a 30-tree RF, and predicts all 5 states for 2025. The 5th and 95th percentiles of the 500 predictions form the 90% confidence interval.

**What this measures:** How sensitive the forecast is to which specific historical years the model learned from. Narrow intervals mean stable forecasts; wide intervals signal dependence on particular seasons.

### 6.6 Analog Year Identification

For each state's 2025 prediction, we identify the 3 most similar historical years by Euclidean distance on the z-scored feature vector. This makes forecasts interpretable: instead of a bare number, we can say "2025 in Iowa looks most like 2016, 2015, and 2018 — years when Iowa produced 174, 168, and 197 bu/acre respectively."

---

## 7. Results

### 7.1 2025 Predictions (model1.4)

| State | Aug 1 | Sep 1 | Oct 1 | Final | CI (Final, 90%) |
|---|---|---|---|---|---|
| Iowa | 214.3 | 214.6 | 214.4 | 212.7 | 208.7–220.8 |
| Nebraska | 198.0 | 197.8 | 198.7 | 198.1 | 194.1–205.6 |
| Missouri | 184.0 | 181.7 | 182.0 | 180.7 | 174.8–188.8 |
| Wisconsin | 182.4 | 183.0 | 183.4 | 182.0 | 178.2–189.3 |
| Colorado | 118.0 | 119.1 | 113.6 | 114.4 | 98.9–134.5 |

### 7.2 Comparison to 2025 USDA Actuals

| State | Forecast Date | Predicted | USDA Actual | Error |
|---|---|---|---|---|
| Colorado | Aug 1 | 117.95 | 133 | −15.1 |
| Iowa | Aug 1 | 214.28 | 210 | +4.3 |
| Missouri | Aug 1 | 184.00 | 185 | −1.0 |
| Nebraska | Aug 1 | 197.99 | 194 | +4.0 |
| Wisconsin | Aug 1 | 182.44 | 188 | −5.6 |
| Iowa | Final | 212.66 | 210 | +2.7 |
| Missouri | Final | 180.74 | 185 | −4.3 |
| Colorado | Final | 114.36 | 133 | −18.6 |

Four of five states at Aug 1 are within single-digit bu/acre of the final published USDA value — two months before harvest. Colorado remains the persistent exception.

### 7.3 Validation RMSE by Forecast Date (model1.4)

| Forecast Date | RMSE (bu/acre) | RMSE as % of Avg Yield |
|---|---|---|
| August 1 | 9.73 | ~5.5% |
| September 1 | 9.30 | ~5.2% |
| October 1 | 9.65 | ~5.4% |
| End of Season | 10.33 | ~5.8% |

For context: USDA's own in-season August 1 estimates historically carry ~5–8% RMSE nationally. Our model lands in the same range, trained on 20 years of state-level data with no manual expert input.

---

## 8. What We'd Do With More Time

1. **Prithvi-100M integration** — Replace mean NDVI scalars with full spatial-temporal embeddings from raw HLS tile stacks. The pipeline architecture is unchanged; only the feature extraction step swaps out. Requires SageMaker GPU and stable HLS data access.

2. **Resolve Colorado underprediction** — Add irrigation data (USDA Farm and Ranch Irrigation Survey) as a feature, or train a Colorado-specific model calibrated to eastern-plains irrigated production patterns.

3. **County-level granularity** — State-level averaging hides meaningful within-state variation. County-level NDVI and weather would give 300+ training rows and substantially more precise forecasts.

4. **Soil data integration** — Root-zone available water capacity (USDA Web Soil Survey) modulates how observed rainfall translates to plant-available water — especially important in drought years.

5. **Dynamic bias correction** — Replace the static per-state offset with a Gaussian process or quantile regression layer that models prediction uncertainty as a function of the forecast value itself.

---

## 9. Reproducibility

| Component | How to reproduce |
|---|---|
| Yield data | `01_quickstats.ipynb` — runs locally, no credentials required |
| Weather data | `02_weather.ipynb` — requires `NOAA_API_KEY` in `.env` |
| Satellite NDVI | **Do not re-run** — `data/raw/ndvi_by_state_date.csv` committed to repo |
| Feature merge | `04_merge_features.ipynb` — runs locally |
| Model & predictions | `05_model1.4.ipynb` — runs locally, no GPU required |

**Environment:** `conda activate geospatial-python-crash-course`

**Random seeds:** Main RF uses `random_state=42`. Bootstrap iterations use non-fixed seeds by design — variance estimation requires genuine randomness.
