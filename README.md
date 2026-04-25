# Fruit Fly Pathway Risk Dashboard — USDA APHIS PPQ
CSU Hackathon 2026, Prompt 1. Interactive geospatial risk dashboard for fruit fly introduction into the contiguous U.S. via foreign air passenger and cargo pathways. Monthly composite risk scores by origin country × U.S. port of entry, with species-level filtering for *Ceratitis capitata* (Medfly), *Bactrocera dorsalis* (Oriental FF), and *Anastrepha ludens* (Mexican FF).

## Pipeline at a glance

```
BTS T-100 (2015–2025) ──┐
USDA FAS GATS imports ──┼──→ Pathway volume (pax + cargo) ──→ composite risk score
EPPO pest presence ──────┤       per (origin_country × dest_port × month)
APHIS interceptions ─────┘                    │
                                              ▼
WorldClim tavg/prec ──┐             scored risk table
USDA CDL (2025) ──────┼──→ Establishment risk index ──→ app/data/app_data.js
Natural Earth GeoJSON ─┘                                        │
                                                                ▼
                                              Leaflet.js interactive dashboard
                                              (choropleth + flight arcs + port bubbles)
```

Risk score = `(passengers + freight / 5 000) × pest_presence_score`, averaged over available years and indexed monthly. Pest presence scored from EPPO distribution records: 3 = widespread, 2 = restricted distribution, 1 = few occurrences / transient.

## Data sources (verified)

| Source | Use | Years | Access |
|---|---|---|---|
| BTS T-100 International Segment | Passenger + freight volumes by route | 2015–2025 | BTS TranStats download |
| USDA FAS GATS | Host commodity imports (kg) by partner country | 2015–2026 | GATS Standard Query |
| EPPO Global Database | Pest presence status per country, 3 target species | current | gd.eppo.int distribution export |
| APHIS PPQ Program Data | Interception / detection validation records | 2015–2026 | APHIS public bulletins |
| IPPC Pest Reports | Phytosanitary event feed (2025–2026) | 2025–2026 | IPPC REST API |
| OurAirports | Airport metadata — IATA, lat/lon | current | davidmegginson.github.io |
| Natural Earth 50m | Country boundaries GeoJSON | current | nvkelso/natural-earth-vector |
| WorldClim v2.1 | Monthly avg temperature + precipitation rasters | 1970–2000 normals | worldclim.org |
| USDA NASS CDL | U.S. host crop acreage mask (2025 clip) | 2025 | CropScape / GEE |
| ISO 3166-1 country codes | Country name ↔ ISO2 reconciliation | current | datasets/country-codes |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys required for the dashboard itself. T-100 and GATS data require manual export from BTS TranStats and USDA FAS GATS portals respectively (see [`DATA_PLAN.md`](DATA_PLAN.md) for step-by-step instructions).

## Run order

```bash
# 1. Check which raw data files are present
python scripts/01_acquire_data.py --check

# 2. Auto-download the files that can be fetched directly (airports, country codes, GeoJSON)
python scripts/01_acquire_data.py --auto

# 3. Build the app data bundle (processes all CSVs → app/data/app_data.js)
python scripts/05_build_app_data.py

# 4. Serve and open the dashboard
cd app && python -m http.server 8765
# then open http://localhost:8765
```

`05_build_app_data.py` is idempotent — re-running it regenerates the bundle from whatever raw files are present. Expected runtime: ~15 seconds for the full dataset (2015–2025 T-100, 11 years × ~37k rows/year).

## Dashboard features

**Map layers (all toggleable)**
- Country choropleth in two modes: *Pest Presence* (EPPO score categories) or *Pathway Risk* (gradient heatmap — flight volume × pest score)
- Great-circle flight arcs for top 15 / 30 / 50 risk routes, colored and weighted by risk tier
- U.S. port-of-entry bubbles sized by total inbound risk score; click for full origin breakdown

**Controls**
- Month slider (Jan–Dec) with play/pause animation — all layers and charts update live
- Species filter: All Species / Medfly / Oriental FF / Mexican FF
- Include/exclude cargo from risk calculation
- Port IATA labels overlay
- Focus CONUS / World view buttons

**Sidebar panels** (each collapsible; sidebar itself collapsible)
- Risk summary stats: origin country count, total passengers, cargo tonnage, ports exposed
- Top 12 risk pathways ranked with species badges and bar chart
- U.S. port exposure: top 10 ports with stacked species-mix chart and APHIS state detection grid
- Monthly passenger flow trend (Chart.js line, all 3 species overlaid)
- Host commodity imports by partner country (GATS, horizontal bar)
- Recent detections & IPPC alerts feed (combined, sorted by date)
- Risk model methodology callout

## Project layout

```
scripts/
  01_acquire_data.py          auto-download + manual instructions for every source
  05_build_app_data.py        process raw CSVs → app/data/app_data.js (5.5 MB bundle)
data/
  raw/                        one file per source — gitignored
    t100_international_*.csv  BTS T-100, 2015–2025
    eppo_ceratitis_capitata.csv
    eppo_bactrocera_dorsalis.csv
    eppo_anastrepha_ludens.csv
    gats_host_imports.csv
    aphis_validation.csv
    ippc_fruit_fly_pest_reports_2025_2026.csv
    airports.csv
    countries.geojson
    wc2.1_10m_tavg_*.tif      WorldClim temperature normals (12 months)
    wc2.1_10m_prec_*.tif      WorldClim precipitation normals (12 months)
    cdl_2025_clipped.tif      USDA CDL host crop mask
  processed/                  (generated by pipeline scripts)
app/
  index.html                  single-file Leaflet.js + Chart.js dashboard
  data/
    app_data.js               generated JS bundle (pest presence + routes + GeoJSON)
DATA_PLAN.md                  full data acquisition plan and source rationale
```

## Design notes

**Risk score is pathway-weighted, not model-predicted.** This dashboard computes a deterministic composite index — `volume × pest_score` — rather than a trained ML model. This is appropriate for the data volume available (152 origin countries, 11 years of monthly T-100, 3 species) and produces interpretable, auditable scores that program managers can act on directly.

**Cargo scaling.** T-100 FREIGHT is in pounds; passengers are integer headcounts. Raw freight values are ~5,000× larger than passenger values. The `freight / 5 000` scaling factor approximately equalises their contributions before multiplying by pest score. The "Include Cargo Risk" toggle lets analysts compare passenger-only vs. combined pathway risk.

**Choropleth opacity encodes pathway volume.** Countries with the same EPPO pest score (e.g., two "widespread" nations) are distinguished by opacity — higher-volume routes render darker, making Mexico visually dominant over a low-traffic widespread-presence country like Ghana.

**Pre-computation at build time.** `05_build_app_data.py` pre-aggregates risk by `(species × month × country)` and `(species × month × US_port)`, writing the results into `app_data.js`. The browser performs no heavy aggregation at runtime — all slider and filter updates are O(n_countries) lookups.

**Hardware.** Data processing and raster clipping (WorldClim, CDL) run locally. The DGX Spark (128 GB unified memory, Blackwell architecture, BF16) was used for any computationally intensive raster operations and to accelerate the data pipeline for the full 11-year T-100 corpus.

**Validation.** Predicted high-risk corridors (Mexico → TX/CA for *A. ludens*; Thailand / Kenya → West Coast for *B. dorsalis* / *C. capitata*) align with APHIS 2025–2026 detection bulletins and IPPC phytosanitary reports included in the dataset.
