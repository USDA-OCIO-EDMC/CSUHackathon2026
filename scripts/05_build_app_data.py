#!/usr/bin/env python3
"""
Fruit Fly Risk Dashboard — Data Builder
Processes raw CSVs into a single JS data file for the web app.

Run from project root:
    python scripts/05_build_app_data.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "app" / "data"
OUT.mkdir(parents=True, exist_ok=True)

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_COLS = [m+"_Qty" for m in ["January","February","March","April","May","June",
                                  "July","August","September","October","November","December"]]

SPECIES_META = {
    "certca": {"name": "Mediterranean Fruit Fly", "latin": "Ceratitis capitata",
               "file": "eppo_ceratitis_capitata.csv", "color": "#ff6b6b",
               "peak_months": [4,5,6,7,8,9,10]},
    "dacudo": {"name": "Oriental Fruit Fly", "latin": "Bactrocera dorsalis",
               "file": "eppo_bactrocera_dorsalis.csv", "color": "#ffd166",
               "peak_months": [5,6,7,8,9,10]},
    "anstlu": {"name": "Mexican Fruit Fly", "latin": "Anastrepha ludens",
               "file": "eppo_anastrepha_ludens.csv", "color": "#06d6a0",
               "peak_months": [4,5,6,7,8,9]},
}

PEST_SCORE = {
    "present, widespread": 3,
    "present, restricted distribution": 2,
    "present, no details": 1,
    "present, few occurrences": 1,
    "transient": 0.5,
}


def score_status(status: str) -> float:
    s = status.lower().strip()
    for key, val in PEST_SCORE.items():
        if s.startswith(key):
            return val
    return 0.0


# ---------------------------------------------------------------------------
# 1. Pest presence by country
# ---------------------------------------------------------------------------

def build_pest_presence() -> dict:
    presence: dict = {}
    for code, meta in SPECIES_META.items():
        path = RAW / meta["file"]
        if not path.exists():
            print(f"  WARN: {path.name} not found, skipping")
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                iso2 = row.get("country code", "").strip()
                if not iso2:
                    continue
                s = score_status(row.get("Status", ""))
                if iso2 not in presence:
                    presence[iso2] = {}
                presence[iso2][code] = {
                    "score": s,
                    "status": row.get("Status", "").strip(),
                }
    # add max_score for quick choropleth coloring
    for iso2, sp in presence.items():
        sp["max_score"] = max((v["score"] for v in sp.values()), default=0)
    return presence


# ---------------------------------------------------------------------------
# 2. Country centroids from GeoJSON
# ---------------------------------------------------------------------------

def build_centroids() -> dict:
    path = RAW / "countries.geojson"
    if not path.exists():
        return {}
    with open(path) as f:
        gj = json.load(f)
    centroids = {}
    for feat in gj["features"]:
        p = feat["properties"]
        iso2 = p.get("ISO_A2", "")
        if iso2 and iso2 != "-99":
            centroids[iso2] = {
                "lat": round(p.get("LABEL_Y", 0), 4),
                "lon": round(p.get("LABEL_X", 0), 4),
                "name": p.get("NAME_EN") or p.get("NAME", ""),
                "continent": p.get("CONTINENT", ""),
            }
    return centroids


# ---------------------------------------------------------------------------
# 3. US airports with IATA codes
# ---------------------------------------------------------------------------

def build_us_airports() -> dict:
    path = RAW / "airports.csv"
    if not path.exists():
        return {}
    airports = {}
    large_types = {"large_airport", "medium_airport"}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("iso_country") != "US":
                continue
            if row.get("type") not in large_types:
                continue
            iata = row.get("iata_code", "").strip()
            if not iata:
                continue
            airports[iata] = {
                "name": row.get("name", ""),
                "city": row.get("municipality", ""),
                "lat": round(float(row["latitude_deg"]), 4),
                "lon": round(float(row["longitude_deg"]), 4),
            }
    return airports


# ---------------------------------------------------------------------------
# 4. T-100 flight data — aggregate to (origin_country, dest_iata, month)
# ---------------------------------------------------------------------------

def build_flight_data(us_airports: dict) -> list:
    agg: dict = defaultdict(lambda: {"passengers": 0.0, "freight": 0.0, "count": 0})

    years_loaded = 0
    for year in range(2022, 2026):
        path = RAW / f"t100_international_{year}.csv"
        if not path.exists():
            continue
        years_loaded += 1
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("DEST_COUNTRY") != "US":
                    continue
                if row.get("ORIGIN_COUNTRY") == "US":
                    continue
                dest = row.get("DEST", "").strip()
                if dest not in us_airports:
                    continue
                origin_iso2 = row.get("ORIGIN_COUNTRY", "").strip()
                month = int(row.get("MONTH", 0))
                if not (1 <= month <= 12):
                    continue
                key = (origin_iso2, dest, month)
                agg[key]["passengers"] += float(row.get("PASSENGERS") or 0)
                agg[key]["freight"] += float(row.get("FREIGHT") or 0)
                agg[key]["count"] += 1

    if years_loaded == 0:
        print("  WARN: no T-100 files found")
        return []

    # Average over years loaded
    routes = []
    for (origin_iso2, dest_iata, month), vals in agg.items():
        routes.append({
            "origin_iso2": origin_iso2,
            "dest_iata": dest_iata,
            "month": month,
            "passengers": round(vals["passengers"] / years_loaded),
            "freight": round(vals["freight"] / years_loaded),
        })
    return routes


# ---------------------------------------------------------------------------
# 5. GATS host commodity imports — melt wide → long
# ---------------------------------------------------------------------------

def build_imports() -> list:
    path = RAW / "gats_host_imports.csv"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for row in csv.DictReader(f):
            partner = row.get("Partner", "").strip().rstrip("(*)")
            year_str = row.get("Year", "")
            m = re.match(r"(\d{4})", year_str)
            if not m:
                continue
            year = int(m.group(1))
            if year < 2022:
                continue
            product = row.get("Product", "").strip()
            hs_code = row.get("HS Code", "").strip()
            for i, col in enumerate(MONTH_COLS, start=1):
                raw_val = row.get(col, "").replace(",", "").strip()
                if not raw_val:
                    continue
                try:
                    qty = float(raw_val)
                except ValueError:
                    continue
                if qty > 0:
                    records.append({
                        "partner": partner,
                        "hs_code": hs_code,
                        "product": product,
                        "year": year,
                        "month": i,
                        "qty_kg": round(qty),
                    })
    return records


# ---------------------------------------------------------------------------
# 6. APHIS detections
# ---------------------------------------------------------------------------

def build_detections() -> list:
    path = RAW / "aphis_validation.csv"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for row in csv.DictReader(f):
            records.append({
                "year": int(row["year"]),
                "month": int(row["month"]),
                "state": row["state_or_port"].strip(),
                "species": row["species"].strip(),
                "count": int(row.get("count", 1) or 1),
                "notes": row.get("notes", "").strip(),
            })
    return sorted(records, key=lambda x: (x["year"], x["month"]), reverse=True)


# ---------------------------------------------------------------------------
# 7. IPPC pest reports
# ---------------------------------------------------------------------------

def build_ippc_reports() -> list:
    path = RAW / "ippc_fruit_fly_pest_reports_2025_2026.csv"
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for row in csv.DictReader(f):
            records.append({
                "date": row.get("git date", "").strip().strip('"'),
                "country": row.get("country", "").strip(),
                "species": row.get("species", "").strip(),
                "event_type": row.get("event_type", "").strip(),
                "description": row.get("description", "").strip(),
            })
    return records


# ---------------------------------------------------------------------------
# 8. Build composite risk table (join flights + pest + locations)
# ---------------------------------------------------------------------------

def build_risk_table(routes: list, pest: dict, centroids: dict, us_airports: dict) -> list:
    risk_rows = []
    for r in routes:
        iso2 = r["origin_iso2"]
        dest = r["dest_iata"]
        p_data = pest.get(iso2, {})
        max_score = p_data.get("max_score", 0)
        if max_score == 0:
            continue
        centroid = centroids.get(iso2, {})
        airport = us_airports.get(dest, {})
        if not centroid or not airport:
            continue
        pax = r["passengers"]
        frt = r["freight"]
        # normalised risk: passengers dominate, freight adds weight
        risk = round((pax + frt / 5000) * max_score)
        risk_rows.append({
            "origin_iso2": iso2,
            "origin_name": centroid.get("name", iso2),
            "origin_lat": centroid["lat"],
            "origin_lon": centroid["lon"],
            "dest_iata": dest,
            "dest_name": airport.get("name", dest),
            "dest_city": airport.get("city", ""),
            "dest_lat": airport["lat"],
            "dest_lon": airport["lon"],
            "month": r["month"],
            "passengers": pax,
            "freight": frt,
            "certca": p_data.get("certca", {}).get("score", 0),
            "dacudo": p_data.get("dacudo", {}).get("score", 0),
            "anstlu": p_data.get("anstlu", {}).get("score", 0),
            "max_pest_score": max_score,
            "risk_score": risk,
        })
    return sorted(risk_rows, key=lambda x: -x["risk_score"])


# ---------------------------------------------------------------------------
# 9. Aggregate country-month risk for choropleth
# ---------------------------------------------------------------------------

def build_country_risk(risk_rows: list) -> dict:
    # Sum risk by (origin_iso2, month) across all dest airports
    by_country_month: dict = defaultdict(float)
    for r in risk_rows:
        by_country_month[(r["origin_iso2"], r["month"])] += r["risk_score"]
    # Per country: list of monthly totals (index 0 = Jan)
    country_monthly: dict = defaultdict(lambda: [0]*12)
    for (iso2, month), total in by_country_month.items():
        country_monthly[iso2][month - 1] = round(total)
    return dict(country_monthly)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_slim_geojson() -> dict:
    """Strip all but ISO_A2, NAME, and geometry from countries.geojson for choropleth."""
    path = RAW / "countries.geojson"
    if not path.exists():
        return {"type": "FeatureCollection", "features": []}
    with open(path) as f:
        gj = json.load(f)
    slim_features = []
    for feat in gj["features"]:
        p = feat["properties"]
        slim_features.append({
            "type": "Feature",
            "properties": {
                "ISO_A2": p.get("ISO_A2", ""),
                "NAME": p.get("NAME_EN") or p.get("NAME", ""),
            },
            "geometry": feat["geometry"],
        })
    return {"type": "FeatureCollection", "features": slim_features}


def main() -> None:
    print("Building app data...")

    print("  Loading pest presence...")
    pest = build_pest_presence()
    print(f"    {len(pest)} countries with pest data")

    print("  Loading country centroids...")
    centroids = build_centroids()
    print(f"    {len(centroids)} countries")

    print("  Loading US airports...")
    us_airports = build_us_airports()
    print(f"    {len(us_airports)} US airports")

    print("  Loading T-100 flight data...")
    routes = build_flight_data(us_airports)
    print(f"    {len(routes)} route-month records")

    print("  Building risk table...")
    risk_rows = build_risk_table(routes, pest, centroids, us_airports)
    print(f"    {len(risk_rows)} risk records")

    print("  Aggregating country risk...")
    country_risk = build_country_risk(risk_rows)

    print("  Loading GATS imports...")
    imports = build_imports()
    print(f"    {len(imports)} import records")

    print("  Loading APHIS detections...")
    detections = build_detections()
    print(f"    {len(detections)} detection records")

    print("  Loading IPPC reports...")
    ippc = build_ippc_reports()
    print(f"    {len(ippc)} IPPC reports")

    print("  Building slim GeoJSON...")
    geojson = build_slim_geojson()
    print(f"    {len(geojson['features'])} country features")

    # Write individual JSON files
    print("  Writing JSON files...")
    (OUT / "pest_presence.json").write_text(json.dumps(pest, separators=(",",":")))
    (OUT / "country_risk.json").write_text(json.dumps(country_risk, separators=(",",":")))
    (OUT / "risk_routes.json").write_text(json.dumps(risk_rows[:5000], separators=(",",":"))); # top 5000
    (OUT / "us_airports.json").write_text(json.dumps(us_airports, separators=(",",":")))
    (OUT / "centroids.json").write_text(json.dumps(centroids, separators=(",",":")))
    (OUT / "imports.json").write_text(json.dumps(imports, separators=(",",":")))
    (OUT / "detections.json").write_text(json.dumps(detections, separators=(",",":")))
    (OUT / "ippc_reports.json").write_text(json.dumps(ippc, separators=(",",":")))
    (OUT / "species_meta.json").write_text(json.dumps(SPECIES_META, separators=(",",":")))

    # Single combined JS file — works with file:// and http://
    js_out = OUT / "app_data.js"
    with open(js_out, "w") as f:
        f.write("// Auto-generated by scripts/05_build_app_data.py\n")
        f.write(f"const PEST_PRESENCE = {json.dumps(pest, separators=(',',':'))};\n")
        f.write(f"const COUNTRY_RISK = {json.dumps(country_risk, separators=(',',':'))};\n")
        f.write(f"const RISK_ROUTES = {json.dumps(risk_rows[:5000], separators=(',',':'))};\n")
        f.write(f"const US_AIRPORTS = {json.dumps(us_airports, separators=(',',':'))};\n")
        f.write(f"const CENTROIDS = {json.dumps(centroids, separators=(',',':'))};\n")
        f.write(f"const IMPORTS = {json.dumps(imports, separators=(',',':'))};\n")
        f.write(f"const DETECTIONS = {json.dumps(detections, separators=(',',':'))};\n")
        f.write(f"const IPPC_REPORTS = {json.dumps(ippc, separators=(',',':'))};\n")
        f.write(f"const SPECIES_META = {json.dumps(SPECIES_META, separators=(',',':'))};\n")
        f.write(f"const COUNTRIES_GJ = {json.dumps(geojson, separators=(',',':'))};\n")

    print(f"  Done. Output: {OUT}")
    print(f"  JS bundle: {js_out} ({js_out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
