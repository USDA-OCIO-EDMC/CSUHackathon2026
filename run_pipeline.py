import requests
import pandas as pd
import json
import math
import os
from pathlib import Path

# ==============================================================================
# FRUITGUARD — ArcGIS Live Data Pipeline
# Run this script locally or in ArcGIS Notebook / Google Colab
# Requirements: pip install requests pandas
# ==============================================================================

SERVICES = {
    "international": {
        "title": "PPQ International Segments",
        "url": "https://services1.arcgis.com/KNdRU5cN6ENqCTjk/arcgis/rest/services/PPQ%20International%20Segments%20Feature%20Layer/FeatureServer",
        "item_id": "ffc2e851a5fd4ef7a1e2581072b5c46d",
    },
    "detections": {
        "title": "PPQ Fruit Fly Detections Summary",
        "url": "https://services1.arcgis.com/KNdRU5cN6ENqCTjk/arcgis/rest/services/PPQ%20Fruit%20Fly%20Detections%20Summary%20Feature%20Layer/FeatureServer",
        "item_id": "24ed5194234f4d0a824430b745b2b8f4",
    },
    "domestic": {
        "title": "PPQ Domestic Segments",
        "url": "https://services1.arcgis.com/KNdRU5cN6ENqCTjk/arcgis/rest/services/PPQ%20Domestic%20Segments%20Feature%20Layer/FeatureServer",
        "item_id": "",
    },
}

LAYERS = {key: f"{service['url']}/0" for key, service in SERVICES.items()}

BASE_DIR = Path(__file__).resolve().parent
ARCGIS_TOKEN = os.getenv("ARCGIS_TOKEN", "").strip()
USING_DEMO_DATA = False
# CSU hackathon guest accounts authenticate through the CSU ArcGIS Online org.
ARCGIS_PORTAL_URL = os.getenv("ARCGIS_PORTAL_URL", "https://csurams.maps.arcgis.com").rstrip("/")
ARCGIS_TOKEN_URL = f"{ARCGIS_PORTAL_URL}/sharing/rest/generateToken"
ARCGIS_REFERER = os.getenv("ARCGIS_REFERER", "http://127.0.0.1:8765")
ARCGIS_TOKEN_EXPIRATION = int(os.getenv("ARCGIS_TOKEN_EXPIRATION", "720"))
DEFAULT_ARCGIS_USERNAME = "csuguest19"
DEFAULT_ARCGIS_PASSWORD = "CSUguest19!"
USING_LOCAL_HACKATHON_DATA = False

PORT_ALIASES = {
    "Atlanta": "ATL",
    "Chicago": "ORD",
    "Dallas": "DFW",
    "Houston": "IAH",
    "JFK": "JFK",
    "LAX": "LAX",
    "Miami": "MIA",
    "Seattle": "SEA",
    "ATL": "ATL",
    "ORD": "ORD",
    "DFW": "DFW",
    "IAH": "IAH",
    "MIA": "MIA",
    "SEA": "SEA",
}

# -- Known U.S. port coordinates (fallback if geometry missing) ----------------
PORT_COORDS = {
    "LAX": (33.9425, -118.4081), "MIA": (25.7959, -80.2870),
    "JFK": (40.6413, -73.7781),  "HNL": (21.3245, -157.9251),
    "SFO": (37.6213, -122.3790), "ORD": (41.9742, -87.9073),
    "IAH": (29.9902, -95.3368),  "ATL": (33.6407, -84.4277),
    "SEA": (47.4502, -122.3088), "DFW": (32.8998, -97.0403),
    "BOS": (42.3656, -71.0096),  "EWR": (40.6895, -74.1745),
    "SAN": (32.7338, -117.1933), "PHX": (33.4373, -112.0078),
    "IAD": (38.9531, -77.4565),  "MCO": (28.4312, -81.3081),
    "DTW": (42.2162, -83.3554),  "MSP": (44.8848, -93.2223),
    "DEN": (39.8561, -104.6737), "LAS": (36.0840, -115.1537),
}

# -- Species pheromone mapping -------------------------------------------------
SPECIES_PHEROMONE = {
    "Bactrocera dorsalis":  "Methyl eugenol",
    "Ceratitis capitata":   "Trimedlure",
    "Anastrepha ludens":    "Trimedlure",
    "Anastrepha suspensa":  "Trimedlure",
    "Anastrepha obliqua":   "Trimedlure",
    "Bactrocera cucurbitae":"Cuelure",
    "Bactrocera zonata":    "Methyl eugenol",
}

# -- High-risk origin regions --------------------------------------------------
REGION_RISK = {
    "Asia": 88, "Asia-Pacific": 88, "Southeast Asia": 90,
    "Latin America": 82, "Central America": 84, "South America": 80,
    "Caribbean": 79, "Mexico": 85,
    "Africa": 72, "Sub-Saharan Africa": 74,
    "Europe": 55, "Middle East": 60,
    "North America": 40,
    "Brazil": 82, "Chile": 78, "China": 88, "India": 84,
    "Japan": 75, "Australia": 70, "Spain": 62, "Germany": 60,
    "Argentina": 80, "Peru": 82, "South Africa": 72, "Thailand": 90,
    "Canada": 25, "UK": 30, "United Kingdom": 30,
}

# ==============================================================================
# STEP 1 — QUERY ARCGIS FEATURE LAYERS
# ==============================================================================

def load_arcgis_token():
    """Use an existing token or request one from ArcGIS env credentials."""
    if os.getenv("ARCGIS_SKIP_AUTH", "").lower() in {"1", "true", "yes"}:
        return ""
    if ARCGIS_TOKEN:
        return ARCGIS_TOKEN

    username = os.getenv("ARCGIS_USERNAME", DEFAULT_ARCGIS_USERNAME).strip()
    password = os.getenv("ARCGIS_PASSWORD", DEFAULT_ARCGIS_PASSWORD)
    if not username or not password:
        return ""

    print("🔐 Requesting ArcGIS token for configured username...")
    try:
        response = requests.post(
            ARCGIS_TOKEN_URL,
            data={
                "username": username,
                "password": password,
                "client": "referer",
                "referer": ARCGIS_REFERER,
                "expiration": ARCGIS_TOKEN_EXPIRATION,
                "f": "json",
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        token = data.get("token", "")
        if token:
            print("  ArcGIS token acquired.")
            return token

        error = data.get("error", {})
        print(f"  Could not acquire token: {error.get('message', 'unknown ArcGIS error')}")
    except Exception as exc:
        print(f"  Could not acquire token: {exc}")
    return ""


def arcgis_request_headers():
    """Headers required when a token is bound to a referer."""
    return {"Referer": ARCGIS_REFERER} if ARCGIS_TOKEN else {}


def discover_feature_layers():
    """Find every layer exposed by each FeatureServer for the ArcGIS map view."""
    discovered = []

    for service_key, service in SERVICES.items():
        params = {"f": "json"}
        if ARCGIS_TOKEN:
            params["token"] = ARCGIS_TOKEN

        try:
            response = requests.get(
                service["url"],
                params=params,
                headers=arcgis_request_headers(),
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                raise RuntimeError(data["error"].get("message", data["error"]))

            layers = data.get("layers", [])
            if not layers:
                layers = [{"id": 0, "name": service["title"]}]

            for layer in layers:
                layer_id = layer.get("id", 0)
                layer_name = layer.get("name") or f"{service['title']} {layer_id}"
                discovered.append({
                    "serviceKey": service_key,
                    "serviceTitle": service["title"],
                    "title": layer_name,
                    "url": f"{service['url']}/{layer_id}",
                    "itemId": service["item_id"],
                    "available": True,
                })
        except Exception as exc:
            print(f"  Could not discover layers for {service['title']}: {exc}")
            discovered.append({
                "serviceKey": service_key,
                "serviceTitle": service["title"],
                "title": service["title"],
                "url": f"{service['url']}/0",
                "itemId": service["item_id"],
                "available": False,
            })

    return discovered


def get_layer_fields(url):
    """Fetch and print field names from a FeatureServer layer."""
    try:
        params = {"f": "json"}
        if ARCGIS_TOKEN:
            params["token"] = ARCGIS_TOKEN
        r = requests.get(url, params=params, headers=arcgis_request_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            print(f"  Could not fetch fields: {data['error'].get('message', data['error'])}")
            return []
        fields = [f["name"] for f in data.get("fields", [])]
        print(f"  Fields: {fields}")
        return fields
    except Exception as e:
        print(f"  Could not fetch fields: {e}")
        return []


def query_layer(name, url, where="1=1", max_records=5000):
    """
    Query an ArcGIS FeatureServer layer with pagination support.
    Returns a pandas DataFrame with attributes + extracted geometry columns.
    """
    all_records = []
    offset = 0
    page_size = 1000

    print(f"📡 Querying: {name}")
    get_layer_fields(url)

    while True:
        params = {
            "where":             where,
            "outFields":         "*",
            "returnGeometry":    "true",
            "outSR":             "4326",
            "f":                 "json",
            "resultRecordCount": page_size,
            "resultOffset":      offset,
        }
        if ARCGIS_TOKEN:
            params["token"] = ARCGIS_TOKEN
        try:
            r = requests.get(f"{url}/query", params=params, headers=arcgis_request_headers(), timeout=30)
            r.raise_for_status()
            data = r.json()

            if "error" in data:
                print(f"  API Error: {data['error'].get('message', data['error'])}")
                break

            features = data.get("features", [])
            if not features:
                break

            for f in features:
                row = dict(f.get("attributes", {}))
                geom = f.get("geometry", {})

                if "x" in geom and "y" in geom:
                    row["_lng"], row["_lat"] = geom["x"], geom["y"]
                elif "paths" in geom and geom["paths"]:
                    path = geom["paths"][0]
                    mid = path[len(path) // 2]
                    row["_lng"], row["_lat"] = mid[0], mid[1]
                elif "rings" in geom and geom["rings"]:
                    ring = geom["rings"][0]
                    row["_lng"] = sum(c[0] for c in ring) / len(ring)
                    row["_lat"] = sum(c[1] for c in ring) / len(ring)
                else:
                    row["_lng"], row["_lat"] = None, None

                all_records.append(row)

            print(f"  Page offset={offset}: {len(features)} records")
            offset += page_size

            if len(features) < page_size or offset >= max_records:
                break

        except Exception as e:
            print(f"  Exception at offset {offset}: {e}")
            break

    df = pd.DataFrame(all_records)
    print(f"  Total records: {len(df)}")
    return df


def build_demo_frames():
    """Seed enough representative data for the dashboard when private layers need a token."""
    demo_ports = [
        {"port": "LAX", "lat": 33.9425, "lng": -118.4081, "origin": "Southeast Asia", "detections": 78, "routes": 132, "species": (54, 26, 20)},
        {"port": "MIA", "lat": 25.7959, "lng": -80.2870, "origin": "Caribbean", "detections": 64, "routes": 108, "species": (18, 30, 52)},
        {"port": "JFK", "lat": 40.6413, "lng": -73.7781, "origin": "Latin America", "detections": 59, "routes": 116, "species": (28, 40, 32)},
        {"port": "HNL", "lat": 21.3245, "lng": -157.9251, "origin": "Asia-Pacific", "detections": 52, "routes": 74, "species": (62, 24, 14)},
        {"port": "SFO", "lat": 37.6213, "lng": -122.3790, "origin": "Asia", "detections": 47, "routes": 92, "species": (58, 22, 20)},
        {"port": "ORD", "lat": 41.9742, "lng": -87.9073, "origin": "Europe", "detections": 31, "routes": 88, "species": (20, 56, 24)},
        {"port": "DFW", "lat": 32.8998, "lng": -97.0403, "origin": "Mexico", "detections": 42, "routes": 81, "species": (16, 34, 50)},
        {"port": "ATL", "lat": 33.6407, "lng": -84.4277, "origin": "Central America", "detections": 39, "routes": 85, "species": (18, 28, 54)},
        {"port": "SEA", "lat": 47.4502, "lng": -122.3088, "origin": "Asia-Pacific", "detections": 34, "routes": 63, "species": (52, 30, 18)},
        {"port": "IAH", "lat": 29.9902, "lng": -95.3368, "origin": "Mexico", "detections": 36, "routes": 72, "species": (20, 25, 55)},
    ]
    species_names = [
        "Bactrocera dorsalis",
        "Ceratitis capitata",
        "Anastrepha ludens",
    ]
    seasonality = [4, 4, 6, 7, 9, 12, 16, 17, 12, 7, 4, 2]
    season_total = sum(seasonality)

    detection_rows = []
    international_rows = []
    domestic_rows = []

    for port in demo_ports:
        for month_idx, month_weight in enumerate(seasonality, start=1):
            month_total = max(1, round(port["detections"] * month_weight / season_total))
            for species_name, pct in zip(species_names, port["species"]):
                count = max(0, round(month_total * pct / 100))
                if count:
                    detection_rows.append({
                        "Port": port["port"],
                        "DetectionCount": count,
                        "Species": species_name,
                        "DetectionDate": f"2025-{month_idx:02d}-15",
                        "_lat": port["lat"],
                        "_lng": port["lng"],
                    })

        for route_idx in range(port["routes"]):
            international_rows.append({
                "DestinationPort": port["port"],
                "OriginRegion": port["origin"],
                "RouteName": f"{port['origin']}-{port['port']}-{route_idx + 1:03d}",
                "_lat": port["lat"],
                "_lng": port["lng"],
            })

        domestic_rows.append({
            "Port": port["port"],
            "DomesticSegment": f"{port['port']} inspection corridor",
            "SegmentCount": max(4, port["routes"] // 12),
            "_lat": port["lat"],
            "_lng": port["lng"],
        })

    return (
        pd.DataFrame(international_rows),
        pd.DataFrame(detection_rows),
        pd.DataFrame(domestic_rows),
    )


def normalize_port(port):
    return PORT_ALIASES.get(str(port).strip(), str(port).strip().upper())


def normalize_country(country):
    value = str(country).strip()
    return {"UK": "United Kingdom"}.get(value, value)


def local_hackathon_files_available():
    required = [
        "passenger_data.csv",
        "trade_data.csv",
        "pest_status.csv",
        "us_port.csv",
    ]
    return all((BASE_DIR / name).exists() for name in required)


def build_hackathon_frames():
    """Build dashboard source frames from the uploaded hackathon CSV datasets."""
    passenger = pd.read_csv(BASE_DIR / "passenger_data.csv")
    trade = pd.read_csv(BASE_DIR / "trade_data.csv")
    pest = pd.read_csv(BASE_DIR / "pest_status.csv")
    detections = pd.read_csv(BASE_DIR / "us_port.csv")

    passenger["origin_country"] = passenger["origin_country"].map(normalize_country)
    passenger["DestinationPort"] = passenger["us_port"].map(normalize_port)
    passenger["DetectionDate"] = pd.to_datetime(passenger["month"], errors="coerce")
    passenger["PassengerVolume"] = pd.to_numeric(passenger["passengers"], errors="coerce").fillna(0)

    trade["country"] = trade["country"].map(normalize_country)
    trade["DetectionDate"] = pd.to_datetime(trade["month"], errors="coerce")
    trade["FruitImports"] = pd.to_numeric(trade["fruit_imports"], errors="coerce").fillna(0)

    pest["country"] = pest["country"].map(normalize_country)
    pest["fruit_fly_type"] = pest["fruit_fly_type"].fillna("No regulated fruit fly reported")
    pest["pest_status"] = pest["pest_status"].fillna("Absent")

    detections["Port"] = detections["us_port"].map(normalize_port)
    detections["DetectionDate"] = pd.to_datetime(detections["month"], errors="coerce")
    detections["DetectionCount"] = pd.to_numeric(detections["detections"], errors="coerce").fillna(0)

    passenger["year"] = passenger["DetectionDate"].dt.year
    trade["year"] = trade["DetectionDate"].dt.year
    detections["year"] = detections["DetectionDate"].dt.year

    pest_lookup = pest.rename(columns={"country": "origin_country"})
    passenger = passenger.merge(
        pest_lookup[["origin_country", "year", "pest_status", "fruit_fly_type"]],
        on=["origin_country", "year"],
        how="left",
    )
    passenger["pest_status"] = passenger["pest_status"].fillna("Absent")
    passenger["fruit_fly_type"] = passenger["fruit_fly_type"].fillna("No regulated fruit fly reported")

    trade_monthly = trade.rename(columns={"country": "origin_country"})[
        ["origin_country", "DetectionDate", "FruitImports"]
    ]
    passenger = passenger.merge(
        trade_monthly,
        on=["origin_country", "DetectionDate"],
        how="left",
    )
    passenger["FruitImports"] = passenger["FruitImports"].fillna(0)

    intl_rows = []
    for idx, row in passenger.dropna(subset=["DetectionDate"]).iterrows():
        port = row["DestinationPort"]
        coords = PORT_COORDS.get(port, {"lat": None, "lng": None})
        if isinstance(coords, tuple):
            lat, lng = coords
        else:
            lat, lng = coords.get("lat"), coords.get("lng")
        intl_rows.append({
            "DestinationPort": port,
            "OriginCountry": row["origin_country"],
            "OriginRegion": row["origin_country"],
            "RouteName": f"{row['origin_country']}-{port}-{row['DetectionDate'].strftime('%Y-%m')}-{idx}",
            "PassengerVolume": row["PassengerVolume"],
            "FruitImports": row["FruitImports"],
            "PestStatus": row["pest_status"],
            "FruitFlyType": row["fruit_fly_type"],
            "DetectionDate": row["DetectionDate"].strftime("%Y-%m-%d"),
            "_lat": lat,
            "_lng": lng,
        })

    # Assign port/month detections to the highest-pressure fruit fly type for that port/month.
    pressure = passenger.copy()
    pressure["pressure"] = (
        pressure["PassengerVolume"].rank(pct=True)
        + pressure["FruitImports"].rank(pct=True)
        + pressure["pest_status"].map({"Present": 1.0, "Emerging": 0.65, "Absent": 0.05}).fillna(0.05)
    )
    pressure_top = (
        pressure.sort_values("pressure", ascending=False)
        .drop_duplicates(["DestinationPort", "DetectionDate"])
        .set_index(["DestinationPort", "DetectionDate"])["fruit_fly_type"]
        .to_dict()
    )

    det_rows = []
    for _, row in detections.dropna(subset=["DetectionDate"]).iterrows():
        port = row["Port"]
        coords = PORT_COORDS.get(port, {"lat": None, "lng": None})
        if isinstance(coords, tuple):
            lat, lng = coords
        else:
            lat, lng = coords.get("lat"), coords.get("lng")
        species = pressure_top.get((port, row["DetectionDate"]), "Exotic fruit fly")
        det_rows.append({
            "Port": port,
            "DetectionCount": row["DetectionCount"],
            "Species": species,
            "DetectionDate": row["DetectionDate"].strftime("%Y-%m-%d"),
            "_lat": lat,
            "_lng": lng,
        })

    dom_rows = []
    for port, group in detections.groupby("Port"):
        coords = PORT_COORDS.get(port, {"lat": None, "lng": None})
        if isinstance(coords, tuple):
            lat, lng = coords
        else:
            lat, lng = coords.get("lat"), coords.get("lng")
        dom_rows.append({
            "Port": port,
            "DomesticSegment": f"{port} detection network",
            "SegmentCount": max(4, int(group["DetectionCount"].mean()) + 1),
            "_lat": lat,
            "_lng": lng,
        })

    return pd.DataFrame(intl_rows), pd.DataFrame(det_rows), pd.DataFrame(dom_rows)


# -- Run all three queries -----------------------------------------------------
print("=" * 60)
print("FRUITGUARD — Live ArcGIS Data Pipeline")
print("=" * 60)

if local_hackathon_files_available() and not os.getenv("ARCGIS_FORCE_LIVE"):
    USING_LOCAL_HACKATHON_DATA = True
    print("📁 Using uploaded hackathon CSV datasets as the primary source.")
    df_intl, df_detect, df_domestic = build_hackathon_frames()
    ARCGIS_TOKEN = load_arcgis_token()
    if not ARCGIS_TOKEN:
        print("   ArcGIS auth unavailable; dashboard will show computed CSV/ML layers only.")
else:
    ARCGIS_TOKEN = load_arcgis_token()
    df_intl     = query_layer("PPQ International Segments", LAYERS["international"])
    df_detect   = query_layer("PPQ Fruit Fly Detections",   LAYERS["detections"])
    df_domestic = query_layer("PPQ Domestic Segments",      LAYERS["domestic"])

if df_detect.empty or df_intl.empty:
    USING_DEMO_DATA = True
    print("⚠️  Live ArcGIS layers returned no usable rows. Using seeded demo data.")
    if not ARCGIS_TOKEN:
        print("   Set ARCGIS_TOKEN to query the private PPQ layers directly.")
    df_intl, df_detect, df_domestic = build_demo_frames()

# Save raw CSVs for inspection
df_intl.to_csv(BASE_DIR / "raw_international.csv", index=False)
df_detect.to_csv(BASE_DIR / "raw_detections.csv", index=False)
df_domestic.to_csv(BASE_DIR / "raw_domestic.csv", index=False)
print("💾 Raw CSVs saved: raw_international.csv, raw_detections.csv, raw_domestic.csv")

# ==============================================================================
# STEP 2 — INSPECT & NORMALIZE COLUMNS
# ==============================================================================

def find_col(df, candidates):
    """Find the first matching column name from a list of candidates."""
    for c in candidates:
        matches = [col for col in df.columns if c.lower() in col.lower()]
        if matches:
            return matches[0]
    return None

print("" + "=" * 60)
print("STEP 2: Column inspection")
print(f"International columns: {list(df_intl.columns)}")
print(f"Detections columns:    {list(df_detect.columns)}")
print(f"Domestic columns:      {list(df_domestic.columns)}")

# ==============================================================================
# STEP 3 — CPRI SCORING FUNCTIONS
# ==============================================================================

def compute_cpri(detection_count, route_count, region_risk_score,
                 passenger_vol=0, cargo_vol=0):
    """
    Composite Pathway Risk Index (CPRI) — 0 to 100
    Weights:
      Detection history:  35%
      Route volume:       25%
      Origin region risk: 25%
      Passenger volume:   10%
      Cargo volume:        5%
    """
    det_score    = min((detection_count / 50.0) * 100, 100)
    route_score  = min((route_count / 120.0) * 100, 100)
    region_score = region_risk_score
    pass_score   = min((passenger_vol / 1000000.0) * 100, 100)
    cargo_score  = min((cargo_vol / 500000.0) * 100, 100)

    cpri = (
        det_score    * 0.35 +
        route_score  * 0.25 +
        region_score * 0.25 +
        pass_score   * 0.10 +
        cargo_score  * 0.05
    )
    return round(min(cpri, 100), 1)


def risk_tier(cpri):
    if cpri >= 85: return "CRITICAL"
    if cpri >= 70: return "HIGH"
    if cpri >= 50: return "MEDIUM"
    return "LOW"


def decoy_status(cpri):
    if cpri >= 70: return "deploy"
    if cpri >= 50: return "monitor"
    return "standby"


def get_pheromone(species_str):
    if not species_str:
        return "Methyl eugenol / Trimedlure"
    s = str(species_str).lower()
    if "dorsalis" in s or "zonata" in s:
        return "Methyl eugenol"
    if "capitata" in s or "anastrepha" in s or "ludens" in s:
        return "Trimedlure"
    if "cucurbitae" in s:
        return "Cuelure"
    return "Methyl eugenol / Trimedlure"


# ==============================================================================
# STEP 4 — BUILD PORT-LEVEL RISK TABLE
# ==============================================================================

# Identify key columns dynamically
port_col_det  = find_col(df_detect,  ["port", "airport", "location", "city", "name"])
count_col_det = find_col(df_detect,  ["count", "total", "detections", "number", "num"])
species_col   = find_col(df_detect,  ["species", "pest", "fly", "insect"])
month_col     = find_col(df_detect,  ["month", "date", "year", "time"])
# Explicit column mapping for live ArcGIS PPQ International Segments layer
# DEST = US destination airport code (LAX, MIA, JFK, etc.)
# ORIGIN_COUNTRY_NAME = where the flight originated (the pathway origin)
if "DEST" in df_intl.columns:
    port_col_intl = "DEST"
elif "DEST_AIRPORT_ID" in df_intl.columns:
    port_col_intl = "DEST_AIRPORT_ID"
else:
    port_col_intl = find_col(df_intl, ["dest", "airport", "port"])

if "ORIGIN_COUNTRY_NAME" in df_intl.columns:
    origin_col = "ORIGIN_COUNTRY_NAME"
elif "ORIGIN_COUNTRY" in df_intl.columns:
    origin_col = "ORIGIN_COUNTRY"
else:
    origin_col = find_col(df_intl, ["origin", "country", "source"])

route_col_intl = find_col(df_intl, ["route", "segment", "flight", "count", "total"])


print(f"Detections  — port: {port_col_det} | count: {count_col_det} | species: {species_col} | month: {month_col}")
print(f"International — port: {port_col_intl} | route: {route_col_intl} | origin: {origin_col}")


def build_port_risk_table(df_detect, df_intl, df_domestic):
    if "DEST" in df_intl.columns:
        port_col_intl = "DEST"
    elif "DestinationPort" in df_intl.columns:
        port_col_intl = "DestinationPort"
    elif "DEST_AIRPORT_ID" in df_intl.columns:
        port_col_intl = "DEST_AIRPORT_ID"
    else:
        port_col_intl = find_col(df_intl, ["destination", "dest", "airport", "port"])

    if "ORIGIN_COUNTRY_NAME" in df_intl.columns:
        origin_col = "ORIGIN_COUNTRY_NAME"
    elif "OriginCountry" in df_intl.columns:
        origin_col = "OriginCountry"
    elif "ORIGIN_COUNTRY" in df_intl.columns:
        origin_col = "ORIGIN_COUNTRY"
    else:
        origin_col = find_col(df_intl, ["origin", "country", "source"])
    ports = {}

    # -- Aggregate detections by port -----------------------------------------
    if port_col_det and count_col_det:
        det_grp = df_detect.groupby(port_col_det).agg(
            total_detections=(count_col_det, "sum"),
            lat=("_lat", "first"),
            lng=("_lng", "first"),
        ).reset_index()

        for _, row in det_grp.iterrows():
            port_id = str(row[port_col_det]).strip().upper()
            if not port_id or port_id == "NAN":
                continue
            ports[port_id] = {
                "id":          port_id,
                "name":        port_id,
                "detections":  int(row["total_detections"]) if pd.notna(row["total_detections"]) else 0,
                "lat":         row["lat"],
                "lng":         row["lng"],
                "routes":      0,
                "passengerVolume": 0,
                "cargoVolume": 0,
                "topOrigin":   "Unknown",
                "species":     {"bdorsalis": 33, "ccapitata": 34, "anastrepha": 33},
                "pathway":     {"passenger": 54, "cargo": 35, "courier": 11},
                "pheromone":   "Methyl eugenol / Trimedlure",
                "monthlyRisk": [50] * 12,
            }

    # -- Species breakdown per port -------------------------------------------
    if port_col_det and species_col and count_col_det:
        spec_grp = df_detect.groupby([port_col_det, species_col])[count_col_det].sum().reset_index()
        for port_id in ports:
            port_specs = spec_grp[spec_grp[port_col_det].str.upper().str.strip() == port_id]
            if not port_specs.empty:
                total = port_specs[count_col_det].sum()
                if total > 0:
                    bd = port_specs[port_specs[species_col].str.contains("dorsalis",  case=False, na=False)][count_col_det].sum()
                    cc = port_specs[port_specs[species_col].str.contains("capitata",  case=False, na=False)][count_col_det].sum()
                    an = port_specs[port_specs[species_col].str.contains("anastrepha",case=False, na=False)][count_col_det].sum()
                    ports[port_id]["species"] = {
                        "bdorsalis":  round(bd / total * 100),
                        "ccapitata":  round(cc / total * 100),
                        "anastrepha": round(an / total * 100),
                    }
                    dominant = max(ports[port_id]["species"], key=ports[port_id]["species"].get)
                    dominant_species = {
                        "bdorsalis": "Bactrocera dorsalis",
                        "ccapitata": "Ceratitis capitata",
                        "anastrepha": "Anastrepha ludens",
                    }
                    ports[port_id]["pheromone"] = get_pheromone(dominant_species.get(dominant, dominant))

    # -- Monthly risk per port ------------------------------------------------
    if port_col_det and month_col and count_col_det:
        try:
            df_detect["_month_num"] = pd.to_datetime(df_detect[month_col], errors="coerce").dt.month
            month_grp = df_detect.groupby([port_col_det, "_month_num"])[count_col_det].sum().reset_index()
            for port_id in ports:
                pm = month_grp[month_grp[port_col_det].str.upper().str.strip() == port_id]
                if not pm.empty:
                    max_val = pm[count_col_det].max()
                    monthly = []
                    for m in range(1, 13):
                        val = pm[pm["_month_num"] == m][count_col_det].sum()
                        monthly.append(round(min((val / max(max_val, 1)) * 100, 100)))
                    ports[port_id]["monthlyRisk"] = monthly
        except Exception as e:
            print(f"  Monthly risk error: {e}")

    # -- International routes per port ----------------------------------------
    if port_col_intl:
        passenger_col = find_col(df_intl, ["passenger"])
        cargo_col = find_col(df_intl, ["fruitimports", "fruit_imports", "cargo", "import"])
        route_grp = df_intl.groupby(port_col_intl).size().reset_index(name="route_count")
        for _, row in route_grp.iterrows():
            port_id = str(row[port_col_intl]).strip().upper()
            if port_id in ports:
                ports[port_id]["routes"] = int(row["route_count"])

        if passenger_col:
            pass_grp = df_intl.groupby(port_col_intl)[passenger_col].sum().reset_index(name="passenger_volume")
            for _, row in pass_grp.iterrows():
                port_id = str(row[port_col_intl]).strip().upper()
                if port_id in ports:
                    ports[port_id]["passengerVolume"] = int(row["passenger_volume"])

        if cargo_col:
            cargo_grp = df_intl.groupby(port_col_intl)[cargo_col].sum().reset_index(name="cargo_volume")
            for _, row in cargo_grp.iterrows():
                port_id = str(row[port_col_intl]).strip().upper()
                if port_id in ports:
                    ports[port_id]["cargoVolume"] = int(row["cargo_volume"])

        if origin_col:
            df_origin = df_intl.copy()
            df_origin["_origin_weight"] = 1
            if passenger_col:
                df_origin["_origin_weight"] += pd.to_numeric(df_origin[passenger_col], errors="coerce").fillna(0) / 100000
            if cargo_col:
                df_origin["_origin_weight"] += pd.to_numeric(df_origin[cargo_col], errors="coerce").fillna(0) / 50000
            
            if port_col_intl != origin_col:
                orig_grp = df_intl.groupby([port_col_intl, origin_col]).size().reset_index(name="cnt")
            else:
                # Fallback: if columns collided, use DEST + ORIGIN_COUNTRY_NAME
                _pcol = "DEST" if "DEST" in df_intl.columns else port_col_intl
                _ocol = "ORIGIN_COUNTRY_NAME" if "ORIGIN_COUNTRY_NAME" in df_intl.columns else origin_col
                orig_grp = df_intl.groupby([_pcol, _ocol]).size().reset_index(name="cnt")
                port_col_intl, origin_col = _pcol, _ocol



            for port_id in ports:
                pm = orig_grp[orig_grp[port_col_intl].str.upper().str.strip() == port_id]
                if not pm.empty:
                    top = pm.sort_values("cnt", ascending=False).iloc[0][origin_col]
                    ports[port_id]["topOrigin"] = str(top)

    # -- Fill coordinates from lookup if missing ------------------------------
    for port_id, p in ports.items():
        try:
            if (p["lat"] is None or p["lng"] is None or
                    math.isnan(float(p["lat"])) or math.isnan(float(p["lng"]))):
                if port_id in PORT_COORDS:
                    p["lat"], p["lng"] = PORT_COORDS[port_id]
        except Exception:
            if port_id in PORT_COORDS:
                p["lat"], p["lng"] = PORT_COORDS[port_id]

    # -- Compute CPRI ---------------------------------------------------------
    for port_id, p in ports.items():
        region_score = REGION_RISK.get(p["topOrigin"], 60)
        p["cpri"]       = compute_cpri(
            p["detections"],
            p["routes"],
            region_score,
            passenger_vol=p.get("passengerVolume", 0),
            cargo_vol=p.get("cargoVolume", 0),
        )
        p["risk"]       = risk_tier(p["cpri"])
        p["decoyStatus"] = decoy_status(p["cpri"])

    # -- Drop ports with no valid coordinates ---------------------------------
    valid_ports = {}
    for k, v in ports.items():
        try:
            if v["lat"] is not None and v["lng"] is not None:
                if not (math.isnan(float(v["lat"])) or math.isnan(float(v["lng"]))):
                    valid_ports[k] = v
        except Exception:
            pass

    print(f"✅ Built risk table for {len(valid_ports)} ports")
    return list(valid_ports.values())


port_data = build_port_risk_table(df_detect, df_intl, df_domestic)

# Save processed port data
with open(BASE_DIR / "port_risk_data.json", "w") as f:
    json.dump(port_data, f, indent=2)
print("💾 Saved: port_risk_data.json")

# ==============================================================================
# STEP 5 — GENERATE FRUITGUARD HTML APP WITH LIVE DATA
# ==============================================================================

arcgis_layers = discover_feature_layers()
print(f"🗺️  ArcGIS map layers configured: {len(arcgis_layers)}")

ports_json = json.dumps(port_data, indent=2)
arcgis_layers_json = json.dumps(arcgis_layers, indent=2)
arcgis_token_json = json.dumps(ARCGIS_TOKEN if ARCGIS_TOKEN and not USING_DEMO_DATA else "")
arcgis_referer_json = json.dumps(ARCGIS_REFERER)
arcgis_token_minutes_json = json.dumps(max(5, ARCGIS_TOKEN_EXPIRATION - 5))
if USING_LOCAL_HACKATHON_DATA:
    data_badge = "HACKATHON CSV DATA"
elif USING_DEMO_DATA:
    data_badge = "SEEDED DEMO DATA"
else:
    data_badge = "LIVE ARCGIS DATA"

html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>FruitGuard — Fruit Fly Incursion Risk Intelligence System</title>
  <link rel="stylesheet" href="https://js.arcgis.com/4.29/esri/themes/dark/main.css"/>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <script src="https://js.arcgis.com/4.29/"></script>
  <style>
    *{margin:0;padding:0;box-sizing:border-box;}
    body{font-family:'Segoe UI',sans-serif;background:#0a0f1e;color:#e0e6f0;}
    header{background:linear-gradient(135deg,#0d1b2a,#1b3a5c);padding:16px 28px;display:flex;align-items:center;justify-content:space-between;border-bottom:2px solid #2a6496;box-shadow:0 2px 12px rgba(0,0,0,0.5);gap:16px;}
    header h1{font-size:1.5rem;color:#4fc3f7;letter-spacing:1px;}
    header h1 span{color:#ff7043;}
    header p{font-size:0.75rem;color:#90a4ae;margin-top:2px;}
    .header-actions{display:flex;align-items:center;justify-content:flex-end;gap:9px;flex-wrap:wrap;}
    .header-nav{display:flex;gap:8px;align-items:center;}
    .header-nav a{color:#cfd8dc;text-decoration:none;border:1px solid #2a6496;border-radius:6px;padding:6px 9px;font-size:.72rem;background:#102238;}
    .header-nav a.active{background:#4fc3f7;color:#07111c;font-weight:bold;border-color:#4fc3f7;}
    .badge{background:#ff7043;color:white;font-size:0.65rem;padding:3px 10px;border-radius:12px;font-weight:bold;letter-spacing:1px;}
    .live-badge{background:#1b5e20;color:#a5d6a7;font-size:0.65rem;padding:3px 10px;border-radius:12px;font-weight:bold;margin-left:8px;}
    .app-body{display:flex;height:calc(100vh - 70px);}
    .sidebar{width:320px;min-width:280px;background:#0d1b2a;border-right:1px solid #1e3a5f;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:14px;}
    .sidebar h2{font-size:0.8rem;color:#4fc3f7;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;}
    .control-group{background:#112240;border-radius:8px;padding:12px;border:1px solid #1e3a5f;}
    .control-group label{font-size:0.75rem;color:#90a4ae;display:block;margin-bottom:6px;}
    select{width:100%;background:#0a1628;color:#e0e6f0;border:1px solid #2a6496;border-radius:6px;padding:6px 10px;font-size:0.8rem;cursor:pointer;}
    .ops-brief{background:#112240;border-radius:8px;padding:12px;border:1px solid #1e3a5f;}
    .ops-brief p{font-size:.74rem;color:#aebfce;line-height:1.35;margin-top:7px;}
    .brief-metrics{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:10px;}
    .brief-metric{background:#0a1628;border:1px solid #1e3a5f;border-radius:7px;padding:8px;}
    .brief-metric b{display:block;color:#4fc3f7;font-size:.82rem;margin-bottom:2px;}
    .brief-metric span{font-size:.65rem;color:#90a4ae;}
    .risk-meter{background:#112240;border-radius:8px;padding:12px;border:1px solid #1e3a5f;}
    .risk-bar-wrap{background:#0a1628;border-radius:6px;height:12px;overflow:hidden;margin:8px 0;}
    .risk-bar{height:100%;border-radius:6px;transition:width 0.6s ease,background 0.6s ease;}
    .risk-label{font-size:1.1rem;font-weight:bold;text-align:center;}
    .port-list{background:#112240;border-radius:8px;padding:12px;border:1px solid #1e3a5f;flex:1;}
    .port-item{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid #1e3a5f;font-size:0.78rem;cursor:pointer;transition:background 0.2s;}
    .port-item:hover{background:#1b3a5c;border-radius:4px;padding-left:4px;}
    .port-item:last-child{border-bottom:none;}
    .port-name{color:#cfd8dc;}
    .risk-chip{font-size:0.65rem;font-weight:bold;padding:2px 8px;border-radius:10px;color:white;}
    .risk-CRITICAL{background:#c62828;}.risk-HIGH{background:#e65100;}.risk-MEDIUM{background:#f9a825;color:#111;}.risk-LOW{background:#2e7d32;}
    .source-layers{background:#112240;border-radius:8px;padding:12px;border:1px solid #1e3a5f;}
    .source-layer-row{display:flex;justify-content:space-between;gap:8px;padding:5px 0;border-bottom:1px solid #1e3a5f;font-size:0.7rem;color:#b0bec5;}
    .source-layer-row:last-child{border-bottom:none;}
    .source-layer-row span:first-child{color:#e0e6f0;}
    .source-layer-row span:last-child{color:#607d8b;text-align:right;}
    .action-queue{background:#112240;border-radius:8px;padding:12px;border:1px solid #1e3a5f;}
    .queue-item{display:grid;grid-template-columns:36px 1fr 56px;gap:8px;align-items:center;padding:8px 0;border-bottom:1px solid #1e3a5f;font-size:.73rem;}
    .queue-item:last-child{border-bottom:none;}
    .queue-item strong{color:#e0e6f0}.queue-item span{color:#90a4ae;}
    .decoy-toggle{background:linear-gradient(135deg,#1a237e,#283593);border-radius:8px;padding:12px;border:1px solid #3949ab;}
    .decoy-toggle h2{color:#7986cb;}
    .toggle-btn{width:100%;margin-top:8px;padding:8px;background:#3949ab;color:white;border:none;border-radius:6px;cursor:pointer;font-size:0.8rem;font-weight:bold;transition:background 0.3s;}
    .toggle-btn.active{background:#ff7043;}
    .decoy-signals{margin-top:8px;display:none;}
    .decoy-signals.visible{display:block;}
    .signal-row{display:flex;align-items:center;gap:8px;font-size:0.72rem;color:#b0bec5;margin:4px 0;}
    .signal-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
    .main-panel{flex:1;display:flex;flex-direction:column;}
    .tabs{display:flex;background:#0d1b2a;border-bottom:1px solid #1e3a5f;}
    .tab{padding:10px 20px;font-size:0.78rem;cursor:pointer;color:#607d8b;border-bottom:2px solid transparent;transition:all 0.2s;user-select:none;}
    .tab.active{color:#4fc3f7;border-bottom-color:#4fc3f7;background:#112240;}
    .tab:hover:not(.active){color:#90a4ae;background:#0f1f35;}
    .tab-content{display:none;flex:1;}
    .tab-content.active{display:flex;flex-direction:column;flex:1;}
    #map{flex:1;min-height:400px;}
    .charts-panel{flex:1;overflow-y:auto;padding:20px;display:grid;grid-template-columns:1fr 1fr;gap:16px;}
    .chart-card{background:#112240;border-radius:10px;padding:16px;border:1px solid #1e3a5f;}
    .chart-card h3{font-size:0.78rem;color:#4fc3f7;margin-bottom:12px;text-transform:uppercase;letter-spacing:0.5px;}
    .chart-card canvas{max-height:220px;}
    .chart-card.full-width{grid-column:1/-1;}
    .deploy-panel{flex:1;overflow-y:auto;padding:20px;}
    .deploy-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;}
    .deploy-card{background:#112240;border-radius:10px;padding:16px;border:1px solid #1e3a5f;transition:border-color 0.3s;}
    .deploy-card:hover{border-color:#4fc3f7;}
    .deploy-card h3{font-size:0.9rem;color:#e0e6f0;margin-bottom:6px;}
    .signal-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-top:8px;}
    .signal-card{background:#0a1628;border-radius:6px;padding:8px;text-align:center;font-size:0.65rem;color:#90a4ae;border:1px solid #1e3a5f;}
    .signal-card .signal-icon{font-size:1.2rem;display:block;margin-bottom:4px;}
    .signal-card .signal-val{color:#4fc3f7;font-weight:bold;font-size:0.7rem;}
    .deploy-status{margin-top:10px;padding:6px 10px;border-radius:6px;font-size:0.7rem;font-weight:bold;text-align:center;}
    .status-deploy{background:#1b5e20;color:#a5d6a7;}.status-monitor{background:#e65100;color:#ffccbc;}.status-standby{background:#1a237e;color:#c5cae9;}
    .esri-ui .esri-widget{background:#112240;color:#e0e6f0;}
    .esri-ui .esri-widget__heading{color:#4fc3f7;}
    .esri-popup__main-container{background:#112240;color:#e0e6f0;border:1px solid #2a6496;}
    .popup-title{font-size:0.9rem;font-weight:bold;color:#4fc3f7;margin-bottom:6px;}
    .popup-row{font-size:0.75rem;margin:3px 0;color:#b0bec5;}
    .popup-row span{color:#e0e6f0;font-weight:bold;}
    .popup-decoy{margin-top:8px;padding:6px;background:#1a237e;border-radius:6px;font-size:0.7rem;color:#c5cae9;}
    .stats-bar{display:flex;background:#0d1b2a;border-top:1px solid #1e3a5f;}
    .stat-item{flex:1;padding:8px 12px;text-align:center;border-right:1px solid #1e3a5f;font-size:0.7rem;color:#607d8b;}
    .stat-item:last-child{border-right:none;}
    .stat-item .stat-val{font-size:1rem;font-weight:bold;color:#4fc3f7;display:block;}
    .stat-item .stat-val.red{color:#ef5350;}.stat-item .stat-val.orange{color:#ff7043;}.stat-item .stat-val.green{color:#66bb6a;}
    .evidence-tab{flex:1;overflow-y:auto;padding:20px;}
    .framework-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;}
    .framework-panel{background:#112240;border:1px solid #1e3a5f;border-radius:8px;padding:14px;}
    .framework-panel h3{font-size:.82rem;color:#4fc3f7;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;}
    .framework-panel p,.framework-panel li{font-size:.74rem;line-height:1.42;color:#b0bec5;}
    .framework-panel ul{padding-left:16px;margin-top:7px;}
    .source-pill{display:inline-flex;margin:4px 4px 0 0;padding:5px 8px;border-radius:999px;border:1px solid #2a6496;background:#0a1628;color:#d7e4ee;font-size:.68rem;}
    @media (max-width:900px){
      header{padding:12px 16px;align-items:flex-start;gap:8px;flex-direction:column;}
      header h1{font-size:1.15rem;}
      .app-body{height:auto;min-height:calc(100vh - 96px);flex-direction:column;}
      .sidebar{width:100%;min-width:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));}
      .port-list{min-height:160px;}
      .main-panel{min-height:620px;}
      .tabs{overflow-x:auto;}
      .tab{white-space:nowrap;flex:0 0 auto;}
      .charts-panel{grid-template-columns:1fr;padding:12px;}
      .stats-bar{flex-wrap:wrap;}
      .stat-item{flex:1 1 33%;min-width:120px;}
    }
    @media (max-width:560px){
      .sidebar{display:flex;}
      .stats-bar{display:grid;grid-template-columns:1fr 1fr;}
      .stat-item{border-bottom:1px solid #1e3a5f;}
      .signal-grid{grid-template-columns:1fr;}
    }
  </style>
</head>
<body>
<header>
  <div>
    <h1>&#x1FAB0; Fruit<span>Guard</span> + Fruit &amp; Fly</h1>
    <p>USDA PPQ &middot; Invasive Fruit Fly Incursion Risk Intelligence System</p>
  </div>
  <div class="header-actions">
    <nav class="header-nav">
      <a class="active" href="fruitguard_live.html">Port Risk</a>
      <a href="ml_dashboard.html">ML Pathways</a>
    </nav>
    <div>
      <span class="badge">CPRI RISK MODEL</span>
      <span class="live-badge">&#x25CF; """ + data_badge + """</span>
    </div>
  </div>
</header>

<div class="app-body">
  <div class="sidebar">
    <div class="control-group">
      <h2>&#x1F5D3; Filters</h2>
      <label>Month</label>
      <select id="monthSelect">
        <option value="0">All Year</option>
        <option value="1">January</option><option value="2">February</option>
        <option value="3">March</option><option value="4">April</option>
        <option value="5">May</option><option value="6">June</option>
        <option value="7">July</option><option value="8">August</option>
        <option value="9">September</option><option value="10">October</option>
        <option value="11">November</option><option value="12">December</option>
      </select>
      <label style="margin-top:10px">Species</label>
      <select id="speciesSelect">
        <option value="all">All Species</option>
        <option value="bdorsalis">Bactrocera dorsalis (Oriental FF)</option>
        <option value="ccapitata">Ceratitis capitata (Medfly)</option>
        <option value="anastrepha">Anastrepha spp.</option>
      </select>
      <label style="margin-top:10px">Pathway</label>
      <select id="pathwaySelect">
        <option value="all">All Pathways</option>
        <option value="passenger">Air Passenger</option>
        <option value="cargo">Cargo / Freight</option>
        <option value="courier">Express Courier</option>
      </select>
      <label style="margin-top:10px">Risk Tier</label>
      <select id="riskTierSelect">
        <option value="all">All Risk Tiers</option>
        <option value="CRITICAL">Critical</option>
        <option value="HIGH">High</option>
        <option value="MEDIUM">Medium</option>
        <option value="LOW">Low</option>
      </select>
    </div>
    <div class="ops-brief">
      <h2>Operational Focus</h2>
      <p>Prioritizes PPQ inspection, Fruit &amp; Fly decoy-layer staging, and early detection using port detections, international routes, domestic context, pest biology, and source-country pressure.</p>
      <div class="brief-metrics">
        <div class="brief-metric"><b id="briefTopPort">--</b><span>Highest filtered port</span></div>
        <div class="brief-metric"><b id="briefTopOrigin">--</b><span>Foreign source signal</span></div>
        <div class="brief-metric"><b id="briefDeploy">0</b><span>Deploy recommendations</span></div>
        <div class="brief-metric"><b id="briefCoverage">0</b><span>Provided sources used</span></div>
      </div>
    </div>
    <div class="risk-meter">
      <h2>&#x26A1; National Risk Level</h2>
      <div class="risk-bar-wrap"><div class="risk-bar" id="nationalRiskBar"></div></div>
      <div class="risk-label" id="nationalRiskLabel"></div>
      <div style="font-size:0.65rem;color:#607d8b;text-align:center;margin-top:4px;">Composite Pathway Risk Index (CPRI)</div>
    </div>
    <div class="source-layers">
      <h2>&#x1F5FA; ArcGIS Source Layers</h2>
      <div id="sourceLayerList"></div>
    </div>
    <div class="action-queue">
      <h2>Action Queue</h2>
      <div id="actionQueue"></div>
    </div>
    <div class="port-list">
      <h2>&#x1F3DB; Top Risk Ports</h2>
      <div id="portList"></div>
    </div>
    <div class="decoy-toggle">
      <h2>&#x1F34E; Fruit &amp; Fly Decoy Layer</h2>
      <button class="toggle-btn" id="decoyBtn" onclick="toggleDecoy()">&#x25B6; ACTIVATE DECOY LAYER</button>
      <div class="decoy-signals" id="decoySignals">
        <div class="signal-row"><div class="signal-dot" style="background:#7986cb"></div>&#x1F50A; Acoustic emitter &mdash; male wing-frequency cue plus activity count</div>
        <div class="signal-row"><div class="signal-dot" style="background:#ef9a9a"></div>&#x1F34E; Decoy apples &mdash; pheromone-baited host mimic with approved immobilizing coating</div>
        <div class="signal-row"><div class="signal-dot" style="background:#a5d6a7"></div>&#x1F512; Containment &mdash; protected catch layer closes once flies cluster near decoys</div>
        <div style="font-size:0.65rem;color:#7986cb;margin-top:6px;">&#x2605; Keeps captured flies isolated from marketable fruit inside the cargo box</div>
      </div>
    </div>
  </div>

  <div class="main-panel">
    <div class="tabs">
      <div class="tab active" onclick="switchTab('map')">&#x1F5FA; Risk Map</div>
      <div class="tab" onclick="switchTab('charts')">&#x1F4CA; Analytics</div>
      <div class="tab" onclick="switchTab('deploy')">&#x1F34E; Fruit &amp; Fly Response</div>
      <div class="tab" onclick="switchTab('evidence')">Evidence</div>
    </div>
    <div class="tab-content active" id="tab-map"><div id="map"></div></div>
    <div class="tab-content" id="tab-charts">
      <div class="charts-panel">
        <div class="chart-card"><h3>&#x1F4C5; Monthly CPRI Risk Score</h3><canvas id="monthlyChart"></canvas></div>
        <div class="chart-card"><h3>&#x1F30D; Risk by Origin Region</h3><canvas id="originChart"></canvas></div>
        <div class="chart-card"><h3>&#x2708;&#xFE0F; Pathway Breakdown</h3><canvas id="pathwayChart"></canvas></div>
        <div class="chart-card"><h3>&#x1FAB0; Species Distribution</h3><canvas id="speciesChart"></canvas></div>
        <div class="chart-card full-width"><h3>&#x1F3DB; Port-Level Risk Scores (Top 10)</h3><canvas id="portChart"></canvas></div>
      </div>
    </div>
    <div class="tab-content" id="tab-deploy">
      <div class="deploy-panel">
        <div style="margin-bottom:16px;">
          <h2 style="color:#4fc3f7;font-size:1rem;">&#x1F34E; Fruit &amp; Fly &mdash; Cargo-Box Decoy Apple Layer</h2>
          <p style="font-size:0.75rem;color:#90a4ae;margin-top:4px;">
            FruitGuard's CPRI model identifies high-risk ports of entry using live PPQ ArcGIS data.
            Fruit &amp; Fly stages pheromone-baited decoy apples above a protected catch layer inside
            cargo boxes. Side emitters broadcast male wing-frequency cues and use acoustic returns to
            estimate fly activity. When flies cluster near the decoys, the layer closes to isolate them
            from marketable fruit.
          </p>
        </div>
        <div class="deploy-grid" id="deployGrid"></div>
      </div>
    </div>
    <div class="tab-content" id="tab-evidence">
      <div class="evidence-tab">
        <div class="framework-grid">
          <section class="framework-panel">
            <h3>Risk Framework</h3>
            <p>CPRI combines detection history, route volume, pathway type, source-country pest pressure, species profile, and seasonal port risk into a transparent 0-100 score.</p>
            <ul>
              <li>Critical and high ports feed the deployment queue.</li>
              <li>Filters recompute the operational map and metrics.</li>
              <li>Port popups expose detections, routes, source country, and decoy-layer status.</li>
            </ul>
          </section>
          <section class="framework-panel">
            <h3>Provided Sources</h3>
            <div id="providedSourcePills"></div>
            <p style="margin-top:8px;">ArcGIS feature services are displayed when credentials are available; generated CSVs preserve the same analytical fields for repeatable judging demos.</p>
          </section>
          <section class="framework-panel">
            <h3>Fruit Fly Biology</h3>
            <p>Tephritid survival and establishment depend on host availability, transport pathway, and destination climate. Temperature suitability is emphasized in the ML pathway view, where below-40F pathway-months are treated as unsuitable.</p>
          </section>
          <section class="framework-panel">
            <h3>Action Logic</h3>
            <p>Deploy recommendations target ports where Fruit &amp; Fly decoy layers should be staged for high-risk cargo-box pathways. Monitor recommendations keep inspection active and decoy layers ready where risk is elevated but not critical.</p>
          </section>
        </div>
      </div>
    </div>
    <div class="stats-bar">
      <div class="stat-item"><span class="stat-val red" id="statCritical">--</span>Critical Ports</div>
      <div class="stat-item"><span class="stat-val orange" id="statHigh">--</span>High Risk Ports</div>
      <div class="stat-item"><span class="stat-val" id="statRoutes">--</span>Routes Monitored</div>
      <div class="stat-item"><span class="stat-val" id="statLayers">--</span>ArcGIS Layers</div>
      <div class="stat-item"><span class="stat-val green" id="statTraps">0</span>Fruit &amp; Fly Units Active</div>
      <div class="stat-item"><span class="stat-val" id="statDetections">--</span>Total Detections</div>
    </div>
  </div>
</div>

<script>
const PORTS = """ + ports_json + """;
const ARCGIS_LAYERS = """ + arcgis_layers_json + """;
const ARCGIS_TOKEN = """ + arcgis_token_json + """;
const ARCGIS_REFERER = """ + arcgis_referer_json + """;
const ARCGIS_TOKEN_MINUTES = """ + arcgis_token_minutes_json + """;
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
let decoyActive = false, map, view, riskLayer, decoyLayer, charts = {};

function hasPorts() {
  return Array.isArray(PORTS) && PORTS.length > 0;
}

function riskColor(c) {
  if (c >= 85) return "#c62828";
  if (c >= 70) return "#e65100";
  if (c >= 50) return "#f9a825";
  return "#2e7d32";
}
function riskLabel(c) {
  if (c >= 85) return "CRITICAL";
  if (c >= 70) return "HIGH";
  if (c >= 50) return "MEDIUM";
  return "LOW";
}

function hexToRgba(hex, alpha) {
  const raw = hex.replace("#", "");
  const value = parseInt(raw, 16);
  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;
  return [r, g, b, alpha];
}

function filteredCpri(port) {
  const month = parseInt(document.getElementById("monthSelect").value);
  const species = document.getElementById("speciesSelect").value;
  const pathway = document.getElementById("pathwaySelect").value;
  let cpri = month === 0 ? port.cpri : (port.monthlyRisk[month - 1] || port.cpri);

  if (species !== "all") {
    const sm = {bdorsalis: port.species.bdorsalis, ccapitata: port.species.ccapitata, anastrepha: port.species.anastrepha};
    cpri = Math.min(Math.round(cpri * (sm[species] / 100) * 1.8), 100);
  }
  if (pathway !== "all") {
    const pm = {passenger: port.pathway.passenger, cargo: port.pathway.cargo, courier: port.pathway.courier};
    cpri = Math.min(Math.round(cpri * (pm[pathway] / 100) * 2.2), 100);
  }
  return cpri;
}

function visiblePorts() {
  const tier = document.getElementById("riskTierSelect").value;
  return PORTS.map(port => {
    const cpri = filteredCpri(port);
    return {...port, filteredCpri: cpri, filteredRisk: riskLabel(cpri)};
  }).filter(port => tier === "all" || port.filteredRisk === tier);
}

function circleRing(lng, lat, radiusKm = 80, segments = 72) {
  const ring = [];
  const latRad = lat * Math.PI / 180;
  const kmPerLat = 110.574;
  const kmPerLng = Math.max(1, 111.320 * Math.cos(latRad));
  for (let i = 0; i <= segments; i++) {
    const angle = (i / segments) * Math.PI * 2;
    ring.push([
      lng + (Math.cos(angle) * radiusKm) / kmPerLng,
      lat + (Math.sin(angle) * radiusKm) / kmPerLat
    ]);
  }
  return ring;
}

function initMap() {
  require([
    "esri/Map",
    "esri/views/MapView",
    "esri/layers/FeatureLayer",
    "esri/layers/GraphicsLayer",
    "esri/Graphic",
    "esri/widgets/LayerList",
    "esri/widgets/Legend",
    "esri/widgets/Expand",
    "esri/widgets/Home",
    "esri/config",
    "esri/identity/IdentityManager"
  ], (Map, MapView, FeatureLayer, GraphicsLayer, Graphic, LayerList, Legend, Expand, Home, esriConfig, IdentityManager) => {
    window.ArcGISGraphic = Graphic;

    if (ARCGIS_TOKEN) {
      esriConfig.request.interceptors.push({
        urls: "https://services1.arcgis.com",
        before: (params) => {
          params.requestOptions.query = {
            ...(params.requestOptions.query || {}),
            token: ARCGIS_TOKEN
          };
        }
      });
      IdentityManager.registerToken({
        server: "https://services1.arcgis.com/KNdRU5cN6ENqCTjk/arcgis",
        token: ARCGIS_TOKEN,
        ssl: true,
        expires: Date.now() + ARCGIS_TOKEN_MINUTES * 60 * 1000
      });
    }

    map = new Map({ basemap: "dark-gray-vector" });

    if (ARCGIS_TOKEN) {
      ARCGIS_LAYERS
        .filter(info => info.available !== false)
        .forEach((info, index) => {
          map.add(new FeatureLayer({
            url: info.url,
            title: info.title,
            outFields: ["*"],
            opacity: info.serviceKey === "detections" ? 0.85 : 0.55,
            visible: index < 6
          }));
        });
    }

    decoyLayer = new GraphicsLayer({ title: "Fruit & Fly Decoy Layer Zones", visible: false });
    riskLayer = new GraphicsLayer({ title: "Computed CPRI Port Risk" });
    map.addMany([decoyLayer, riskLayer]);

    view = new MapView({
      container: "map",
      map,
      center: [-96, 37.5],
      zoom: 4,
      popup: { dockEnabled: true, dockOptions: { position: "top-right", breakpoint: false } }
    });

    view.when(() => {
      view.ui.add(new Home({ view }), "top-left");
      view.ui.add(new Expand({ view, content: new LayerList({ view }), expandIcon: "layers", expanded: false }), "top-right");
      view.ui.add(new Expand({ view, content: new Legend({ view }), expandIcon: "legend", expanded: false }), "top-right");
      renderMapMarkers();
    });
  });
}

function renderMapMarkers() {
  if (!riskLayer || !window.ArcGISGraphic) return;
  riskLayer.removeAll();
  if (decoyLayer) decoyLayer.removeAll();

  visiblePorts().forEach(port => {
    const cpri = port.filteredCpri;
    const color = riskColor(cpri);
    const size = 10 + (cpri / 100) * 32;
    riskLayer.add(new window.ArcGISGraphic({
      geometry: { type: "point", longitude: port.lng, latitude: port.lat },
      attributes: { ...port, filteredCpri: cpri, filteredRisk: riskLabel(cpri) },
      symbol: {
        type: "simple-marker",
        style: "circle",
        color: hexToRgba(color, 0.42),
        size,
        outline: { color, width: 2 }
      },
      popupTemplate: {
        title: "{name}",
        content: `
        <div class="popup-title">${port.name}</div>
        <div class="popup-row">CPRI Score: <span style="color:${color}">${cpri}/100 - ${riskLabel(cpri)}</span></div>
        <div class="popup-row">Detections: <span>${port.detections}</span></div>
        <div class="popup-row">Routes: <span>${port.routes}</span></div>
        <div class="popup-row">Top Origin: <span>${port.topOrigin}</span></div>
        <div class="popup-decoy"><b>Fruit &amp; Fly: ${port.decoyStatus.toUpperCase()}</b><br>Emitter: wing-frequency cue + acoustic count | Bait: ${port.pheromone || "ME/TML"} decoy apples | Containment: closing catch layer</div>
        `
      }
    }));
  });

  if (decoyActive) renderDecoyLayer();
  updateNationalRisk();
}

function renderDecoyLayer() {
  if (!decoyLayer || !window.ArcGISGraphic) return;
  decoyLayer.removeAll();
  decoyLayer.visible = true;
  visiblePorts().filter(p => p.decoyStatus === "deploy").forEach(port => {
    decoyLayer.add(new window.ArcGISGraphic({
      geometry: {
        type: "polygon",
        rings: [circleRing(port.lng, port.lat)],
        spatialReference: { wkid: 4326 }
      },
      attributes: { name: port.name },
      symbol: {
        type: "simple-fill",
        color: [57, 73, 171, 0.15],
        outline: { color: [121, 134, 203, 0.9], width: 2, style: "dash" }
      },
      popupTemplate: { title: "Fruit & Fly Zone: {name}", content: "Recommended cargo-box decoy apple layer deployment zone." }
    }));
  });
}

function updateNationalRisk() {
  const ports = visiblePorts();
  if (!hasPorts() || !ports.length) {
    document.getElementById("nationalRiskBar").style.cssText = "width:0%;background:#2a6496";
    document.getElementById("nationalRiskLabel").style.color = "#90a4ae";
    document.getElementById("nationalRiskLabel").textContent = "NO DATA";
    return;
  }
  const avg = Math.round(ports.reduce((s, p) => s + p.filteredCpri, 0) / ports.length);
  const color = riskColor(avg);
  document.getElementById("nationalRiskBar").style.cssText = `width:${avg}%;background:linear-gradient(90deg,${color}99,${color})`;
  document.getElementById("nationalRiskLabel").style.color = color;
  document.getElementById("nationalRiskLabel").textContent = `${riskLabel(avg)} - ${avg} / 100`;
}

function renderPortList() {
  const ports = visiblePorts();
  if (!hasPorts() || !ports.length) {
    document.getElementById("portList").innerHTML = `<div style="font-size:0.75rem;color:#90a4ae;padding-top:8px;">No port data available.</div>`;
    return;
  }
  const sorted = ports.sort((a, b) => b.filteredCpri - a.filteredCpri).slice(0, 10);
  document.getElementById("portList").innerHTML = sorted.map(p => `
    <div class="port-item" onclick="focusPort('${p.id}')">
      <span class="port-name">${p.name.length > 20 ? p.name.substring(0, 20) + "..." : p.name}</span>
      <span class="risk-chip risk-${p.filteredRisk}">${p.filteredCpri}</span>
    </div>`).join("");
}

function renderSourceLayerList() {
  const grouped = ARCGIS_LAYERS.reduce((acc, layer) => {
    if (!acc[layer.serviceTitle]) acc[layer.serviceTitle] = {total: 0, available: 0};
    acc[layer.serviceTitle].total += 1;
    if (layer.available !== false) acc[layer.serviceTitle].available += 1;
    return acc;
  }, {});
  document.getElementById("sourceLayerList").innerHTML = Object.entries(grouped).map(([title, counts]) => `
    <div class="source-layer-row"><span>${title}</span><span>${ARCGIS_TOKEN ? counts.available : 0}/${counts.total} enabled</span></div>
  `).join("");
}

function renderActionQueue() {
  const ports = visiblePorts().sort((a,b) => b.filteredCpri - a.filteredCpri).slice(0, 5);
  document.getElementById("actionQueue").innerHTML = ports.length ? ports.map((p, idx) => `
    <div class="queue-item" onclick="focusPort('${p.id}')">
      <strong>#${idx + 1}</strong>
      <div><strong>${p.name}</strong><br><span>${p.topOrigin} · ${p.detections} detections · ${p.routes} routes</span></div>
      <span class="risk-chip risk-${p.filteredRisk}" style="text-align:center">${p.filteredCpri}</span>
    </div>
  `).join("") : `<div style="font-size:0.75rem;color:#90a4ae;padding-top:8px;">No ports match the current filters.</div>`;
}

function renderBriefMetrics() {
  const ports = visiblePorts().sort((a,b) => b.filteredCpri - a.filteredCpri);
  const top = ports[0];
  const sourceCount = new Set(["Final.csv","PPQ detections","PPQ international segments","PPQ domestic segments","passenger data","trade data","pest status"]).size;
  document.getElementById("briefTopPort").textContent = top ? `${top.name} ${top.filteredCpri}` : "--";
  document.getElementById("briefTopOrigin").textContent = top ? top.topOrigin : "--";
  document.getElementById("briefDeploy").textContent = ports.filter(p => p.decoyStatus === "deploy").length;
  document.getElementById("briefCoverage").textContent = sourceCount;
}

function renderEvidenceTab() {
  const sources = [
    "PPQ Fruit Fly Detections",
    "PPQ International Segments",
    "PPQ Domestic Segments",
    "Final.csv",
    "Passenger Data",
    "Trade Data",
    "Pest Status",
    "Temperature Suitability"
  ];
  document.getElementById("providedSourcePills").innerHTML = sources.map(source => `<span class="source-pill">${source}</span>`).join("");
}

function focusPort(id) {
  const p = PORTS.find(x => x.id === id);
  if (p && view) { switchTab("map"); view.goTo({ center: [p.lng, p.lat], zoom: 7 }); }
}

function initCharts() {
  if (!hasPorts()) return;
  const cd = {
    plugins: {legend: {labels: {color: "#90a4ae", font: {size: 11}}}},
    scales: {
      x: {ticks: {color: "#607d8b"}, grid: {color: "#1e3a5f"}},
      y: {ticks: {color: "#607d8b"}, grid: {color: "#1e3a5f"}}
    }
  };
  const avgMonthly = MONTHS.map((_, i) =>
    Math.round(PORTS.reduce((s, p) => s + (p.monthlyRisk[i] || 0), 0) / PORTS.length)
  );
  charts.monthly = new Chart(document.getElementById("monthlyChart"), {
    type: "line",
    data: {labels: MONTHS, datasets: [{label: "Avg CPRI", data: avgMonthly, borderColor: "#4fc3f7", backgroundColor: "rgba(79,195,247,0.1)", fill: true, tension: 0.4, pointBackgroundColor: "#4fc3f7"}]},
    options: cd
  });
  charts.origin = new Chart(document.getElementById("originChart"), {
    type: "bar",
    data: {labels: ["Asia-Pacific","Latin America","Caribbean","Europe","Africa","Middle East"], datasets: [{label: "Risk Score", data: [88,82,79,65,58,52], backgroundColor: ["#c62828","#e65100","#e65100","#f9a825","#f9a825","#2e7d32"]}]},
    options: {...cd, indexAxis: "y"}
  });
  const bdTotal = PORTS.reduce((s, p) => s + p.species.bdorsalis, 0);
  const ccTotal = PORTS.reduce((s, p) => s + p.species.ccapitata, 0);
  const anTotal = PORTS.reduce((s, p) => s + p.species.anastrepha, 0);
  charts.species = new Chart(document.getElementById("speciesChart"), {
    type: "doughnut",
    data: {labels: ["B. dorsalis","C. capitata","Anastrepha spp."], datasets: [{data: [Math.round(bdTotal/PORTS.length), Math.round(ccTotal/PORTS.length), Math.round(anTotal/PORTS.length)], backgroundColor: ["#ef5350","#ff7043","#ffa726"], borderColor: "#0d1b2a", borderWidth: 2}]},
    options: {plugins: {legend: {labels: {color: "#90a4ae"}}}}
  });
  charts.pathway = new Chart(document.getElementById("pathwayChart"), {
    type: "doughnut",
    data: {labels: ["Air Passenger","Cargo / Freight","Express Courier"], datasets: [{data: [54,35,11], backgroundColor: ["#4fc3f7","#ff7043","#7986cb"], borderColor: "#0d1b2a", borderWidth: 2}]},
    options: {plugins: {legend: {labels: {color: "#90a4ae"}}}}
  });
  const top10 = [...PORTS].sort((a, b) => b.cpri - a.cpri).slice(0, 10);
  charts.port = new Chart(document.getElementById("portChart"), {
    type: "bar",
    data: {labels: top10.map(p => p.name.length > 15 ? p.name.substring(0,15)+"..." : p.name), datasets: [{label: "CPRI Score", data: top10.map(p => p.cpri), backgroundColor: top10.map(p => riskColor(p.cpri)+"cc"), borderColor: top10.map(p => riskColor(p.cpri)), borderWidth: 1}]},
    options: {...cd, plugins: {...cd.plugins, legend: {display: false}}}
  });
}

function renderDeployGrid() {
  const ports = visiblePorts();
  if (!hasPorts() || !ports.length) {
    document.getElementById("deployGrid").innerHTML = "";
    return;
  }
  const sorted = ports.sort((a, b) => b.filteredCpri - a.filteredCpri);
  document.getElementById("deployGrid").innerHTML = sorted.map(p => {
    const color = riskColor(p.filteredCpri);
    const statusText = p.decoyStatus === "deploy" ? "DEPLOY FRUIT & FLY LAYER"
                     : p.decoyStatus === "monitor" ? "MONITOR - STAGE DECOY LAYER"
                     : "STANDBY - INSPECTION ONLY";
    return `<div class="deploy-card">
      <h3>${p.name}</h3>
      <div style="font-size:0.75rem;margin-bottom:6px">CPRI: <span style="color:${color};font-weight:bold">${p.filteredCpri}/100 - ${p.filteredRisk}</span></div>
      <div style="font-size:0.7rem;color:#607d8b;margin-bottom:6px">${p.detections} detections | ${p.routes} routes | ${p.topOrigin}</div>
      <div class="signal-grid">
        <div class="signal-card"><span class="signal-icon">&#x1F50A;</span><div>Emitter</div><div class="signal-val">Cue + count</div></div>
        <div class="signal-card"><span class="signal-icon">&#x1F34E;</span><div>Decoy apple</div><div class="signal-val" style="font-size:0.6rem">${p.pheromone || "ME/TML"} bait</div></div>
        <div class="signal-card"><span class="signal-icon">&#x1F512;</span><div>Containment</div><div class="signal-val">Closing layer</div></div>
      </div>
      <div class="deploy-status status-${p.decoyStatus}">${statusText}</div>
    </div>`;
  }).join("");
}

function updateStats() {
  const ports = visiblePorts();
  if (!hasPorts() || !ports.length) {
    ["statCritical","statHigh","statRoutes","statLayers","statDetections"].forEach(id => {
      document.getElementById(id).textContent = "0";
    });
    return;
  }
  document.getElementById("statCritical").textContent  = ports.filter(p => p.filteredRisk === "CRITICAL").length;
  document.getElementById("statHigh").textContent      = ports.filter(p => p.filteredRisk === "HIGH").length;
  document.getElementById("statRoutes").textContent    = ports.reduce((s, p) => s + p.routes, 0).toLocaleString();
  document.getElementById("statLayers").textContent    = (ARCGIS_TOKEN ? ARCGIS_LAYERS.filter(l => l.available !== false).length : 0).toLocaleString();
  document.getElementById("statDetections").textContent = ports.reduce((s, p) => s + p.detections, 0).toLocaleString();
}

function toggleDecoy() {
  decoyActive = !decoyActive;
  const btn = document.getElementById("decoyBtn");
  const sig = document.getElementById("decoySignals");
  if (decoyActive) {
    btn.textContent = "DEACTIVATE DECOY LAYER";
    btn.classList.add("active");
    sig.classList.add("visible");
    renderDecoyLayer();
    document.getElementById("statTraps").textContent = visiblePorts().filter(p => p.decoyStatus === "deploy").length;
  } else {
    btn.textContent = "ACTIVATE DECOY LAYER";
    btn.classList.remove("active");
    sig.classList.remove("visible");
    if (decoyLayer) {
      decoyLayer.removeAll();
      decoyLayer.visible = false;
    }
    document.getElementById("statTraps").textContent = "0";
  }
}

function switchTab(name) {
  ["map","charts","deploy","evidence"].forEach(n => {
    document.getElementById("tab-" + n).classList.toggle("active", n === name);
  });
  document.querySelectorAll(".tab").forEach((t, i) =>
    t.classList.toggle("active", ["map","charts","deploy","evidence"][i] === name)
  );
  if (name === "map" && view) setTimeout(() => view.resize(), 100);
}

function refreshFilteredViews() {
  renderMapMarkers();
  renderPortList();
  renderDeployGrid();
  renderActionQueue();
  renderBriefMetrics();
  updateStats();
  updateNationalRisk();
  if (decoyActive) document.getElementById("statTraps").textContent = visiblePorts().filter(p => p.decoyStatus === "deploy").length;
}

["monthSelect","speciesSelect","pathwaySelect","riskTierSelect"].forEach(id =>
  document.getElementById(id).addEventListener("change", refreshFilteredViews)
);

window.onload = () => { initMap(); renderSourceLayerList(); renderPortList(); initCharts(); renderDeployGrid(); renderActionQueue(); renderBriefMetrics(); renderEvidenceTab(); updateStats(); updateNationalRisk(); };
</script>
</body>
</html>"""

with open(BASE_DIR / "fruitguard_live.html", "w") as f:
    f.write(html)

print("" + "=" * 60)
print("SUCCESS! Files generated:")
print("  fruitguard_live.html  — Open in any browser")
print("  port_risk_data.json   — Processed port risk data")
print("  raw_international.csv — Raw ArcGIS layer data")
print("  raw_detections.csv    — Raw ArcGIS layer data")
print("  raw_domestic.csv      — Raw ArcGIS layer data")
print("=" * 60)
