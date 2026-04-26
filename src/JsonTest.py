# pip install requests numpy matplotlib geopandas shapely pyogrio
import requests
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point

PORTAL_URL = "https://csurams.maps.arcgis.com/sharing/rest"
SERVICE_URL = "https://services1.arcgis.com/KNdRU5cN6ENqCTjk/arcgis/rest/services/PPQ Fruit Fly Detections Summary Feature Layer/FeatureServer"
LAYER_ID = 0

# This is a feature for later ;)
#USERNAME = os.environ["ARCGIS_USERNAME"]
#PASSWORD = os.environ["ARCGIS_PASSWORD"]
# Security ++
USERNAME = "csuguest28"
PASSWORD = "IEatChildren7!"

TOKEN_URL = f"{PORTAL_URL}/generateToken"
LAYER_URL = f"{SERVICE_URL}/{LAYER_ID}"
QUERY_URL = f"{LAYER_URL}/query"


def generate_token():
    params = {
        "username": USERNAME,
        "password": PASSWORD,
        "client": "referer",
        "referer": "https://csurams.maps.arcgis.com",
        "expiration": 60,
        "f": "json",
    }
    r = requests.post(TOKEN_URL, data=params)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data["token"]


def request_json(url, params, token):
    params = dict(params)
    params["token"] = token
    params["f"] = "json"

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    if "error" in data:
        raise RuntimeError(data["error"])

    return data


def download_sample_features(token, sample_size=300):
    data = request_json(
        QUERY_URL,
        {
            "where": "1=1",
            "outFields": "OBJECTID,NAME,STATE_NAME,monthyear,CommonName,Count_",
            "returnGeometry": "true",
            "outSR": "4326",
            "resultRecordCount": sample_size,
            "orderByFields": "OBJECTID",
        },
        token,
    )
    return data.get("features", [])


def arcgis_polygon_to_shapely(geometry):
    if not geometry or "rings" not in geometry:
        return None

    rings = geometry["rings"]
    polys = []

    for ring in rings:
        if len(ring) >= 3:
            try:
                poly = Polygon(ring)
                if poly.is_valid and not poly.is_empty:
                    polys.append(poly)
            except Exception:
                pass

    if not polys:
        return None

    if len(polys) == 1:
        return polys[0]
    return MultiPolygon(polys)


# --- main ---
token = generate_token()
features = download_sample_features(token, sample_size=500)

geoms = []
weights = []

for feature in features:
    geom = arcgis_polygon_to_shapely(feature.get("geometry"))
    if geom is None:
        continue

    count_val = feature.get("attributes", {}).get("Count_", 1)
    if count_val is None:
        count_val = 1

    geoms.append(geom.centroid)
    weights.append(float(count_val))

if not geoms:
    raise RuntimeError("No usable geometries found.")

gdf = gpd.GeoDataFrame({"weight": weights}, geometry=geoms, crs="EPSG:4326")

# world dataset; keep USA only
natural_earth_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"

world = gpd.read_file(natural_earth_url)
usa = world[world["ADMIN"] == "United States of America"]

fig, ax = plt.subplots(figsize=(14, 8))

# plot US background
usa.plot(ax=ax, edgecolor="black", facecolor="white", linewidth=1)

# overlay weighted hexbin
x = gdf.geometry.x.to_numpy()
y = gdf.geometry.y.to_numpy()
w = gdf["weight"].to_numpy()

hb = ax.hexbin(
    x,
    y,
    C=w,
    reduce_C_function=np.sum,
    gridsize=35,
    mincnt=1,
)

plt.colorbar(hb, ax=ax, label="Weighted detections (Count_)")

ax.set_title("Fruit Fly Detections Overlay on US Map")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# zoom roughly to continental US
ax.set_xlim(-125, -66)
ax.set_ylim(24, 50)

plt.tight_layout()
plt.show()
