# Fruit Fly Origin & Seasonal Hotspot Explorer

A multi-file Django dashboard for comparing fruit fly detection outbreaks against inbound international airplane traffic and annual freight-port context.

## What it does

- Imports `fruit_fly_detections.csv` detections by month, county, state, and approximate map centroid.
- Imports `international_segmentation.csv` and keeps **inbound international traffic to the United States** where `DEST_COUNTRY` / `DEST_COUNTRY_NAME` is US / United States.
- Computes monthly lagged Pearson correlations between fruit fly detections and inbound traffic by origin country.
- Ranks origin countries with a risk-style score based on positive correlation, traffic volume, and detection volume.
- Displays seasonal hotspot maps using Leaflet.
- Shows annual port freight context from `2020Ports.csv`, `2021Ports.csv`, and `2022Ports.csv`.

This is exploratory correlation tooling, not proof of biological origin. Treat high-ranking countries as hypotheses that need domain validation.

## Project layout

```text
website/
  manage.py
  requirements.txt
  README.md
  data/
    put your CSV files here
  outbreak_origin_site/
    settings.py
    urls.py
    wsgi.py
  outbreaks/
    models.py
    views.py
    urls.py
    utils.py
    admin.py
    management/commands/import_data.py
    templates/outbreaks/dashboard.html
    static/outbreaks/dashboard.js
    static/outbreaks/styles.css
```

## Setup

From inside the `website` folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy these files into the `data/` folder with exactly these names:

```text
fruit_fly_detections.csv
international_segmentation.csv
2020Ports.csv
2021Ports.csv
2022Ports.csv
```

Create the database tables:

```bash
python manage.py makemigrations outbreaks
python manage.py migrate
```

Import the data:

```bash
python manage.py import_data --data-dir data --clear
```

For a faster import that skips estimating county centroids from WKT polygons:

```bash
python manage.py import_data --data-dir data --clear --skip-geometry
```

Run the site:

```bash
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

## How the lag works

A lag of `2` means detections in a given month are compared against traffic from two months earlier. This is often more useful than same-month correlation when the causal chain may involve shipping, inspection, establishment, trapping, and reporting delay.

## Notes for improving scientific accuracy

- Replace the rough WKT centroid estimator with Shapely if exact county centroids are required.
- Add actual maritime country-of-origin data if available; the provided port files are annual port totals, not country-by-country import origins.
- Add climate variables and host-crop seasonality to reduce false attribution from traffic alone.
- Consider negative-binomial or Poisson regression for count modeling after the dashboard helps identify candidate predictors.
