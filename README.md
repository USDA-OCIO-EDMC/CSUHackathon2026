# FruitGuard

FruitGuard is a USDA/PPQ hackathon prototype for identifying high-risk fruit fly entry pathways into the United States. It combines PPQ ArcGIS feature layers, trade and passenger movement data, port detections, origin pest pressure, and a temperature suitability rule so researchers can prioritize pathways where response work is most likely to matter.

## What the demo shows

- `fruitguard_live.html` is the operational dashboard: port risk, ArcGIS-backed pathway data, detections, and decision-ready filters.
- `ml_dashboard.html` is the predictive pathway view: origin country to U.S. port routes, ranked by modeled risk.
- `fruitguard_ml.py` builds the batch ML pathway outputs used by the static dashboard.
- `train_fruitfly.py` trains the SageMaker XGBoost workflow: regression for detection counts, 4-tier risk classification, and a binary model with the temperature mask.
- `inference.py` is the SageMaker endpoint inference handler used by the app server proxy.

## Response Concept

The response concept is now `Fruit & Fly`: a cargo-box decoy apple layer. Pheromone-baited decoy apples hang above a protected catch layer so any captured flies remain isolated from marketable fruit. Side emitters provide male wing-frequency cues and acoustic activity estimates; once the model indicates flies are clustered near the decoys, the layer closes to contain them. The dashboards use this concept as the recommended response for high-risk port and pathway rankings.

## Run Locally

```bash
./run.sh
```

This regenerates ArcGIS/dashboard data, rebuilds the ML pathway view, and serves the app at:

- Main dashboard: `http://127.0.0.1:8765/fruitguard_live.html`
- ML pathways: `http://127.0.0.1:8765/ml_dashboard.html`

If port `8765` is already running, the script refreshes the generated files and opens the dashboard.

## Run With SageMaker

From a SageMaker Studio terminal configured for the hackathon AWS account:

```bash
./run_all_sagemaker.sh
```

This validates scripts, launches training from `s3://bucket-for-xgboost/data/`, deploys or updates `fruitfly-risk-endpoint`, regenerates dashboards, and starts the endpoint-backed app server.

For rapid demo iteration after the endpoint already exists:

```bash
SKIP_TRAINING=1 SKIP_DEPLOY=1 ./run_all_sagemaker.sh
```

To serve the browser app against an existing endpoint:

```bash
SAGEMAKER_ENDPOINT_NAME=fruitfly-risk-endpoint AWS_REGION=us-west-2 ./run_endpoint_app.sh
```

SageMaker endpoints are billable while running, so stop or delete the endpoint after the demo if it is no longer needed.

## Current Model Snapshot

The current promoted model artifacts live in `model_artifacts/`.

- Samples: 133
- Features: 37
- Risk tier classification accuracy: 54.9% leave-one-out, 52.6% k-fold
- Regression R2: 0.25 k-fold, 0.22 leave-one-out
- Regression MAE: about 2.6 to 2.7 detections
- Temperature-masked binary model accuracy: 54.3%
- Temperature dead zone: below 40F risk is forced to zero
- Marginal suitability band: 40F to 45F
- Peak season months: June through September

That model quality is realistic for a small, imbalanced hackathon dataset. The value of the prototype is the transparent risk framework: foreign pest pressure times temperature suitability times movement volume, with low-temperature dead zones removed before ranking pathways.

## Data And Sources

Root-level CSV/JSON files are the curated demo inputs and outputs. The `sagemakerenv/` folder is the downloaded SageMaker mirror and should be treated as a source archive, not the presentation surface.

Key promoted files:

- `raw_international.csv`, `raw_detections.csv`, `raw_domestic.csv`: ArcGIS layer snapshots.
- `port_risk_data.json`: risk table consumed by the live dashboard.
- `ml_predictions.json`, `ml_predictions.csv`, `ml_hotspots.json`: predictive pathway outputs.
- `arcgis_bridge.py`, `run_pipeline.py`: ArcGIS/PPQ integration and dashboard data build.
- `model_artifacts/`: latest trained model metadata and serialized estimators.
