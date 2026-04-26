import os
import json
import time
import requests

PORTAL_URL = "https://csurams.maps.arcgis.com/sharing/rest"
SERVICE_URL = "https://services1.arcgis.com/KNdRU5cN6ENqCTjk/arcgis/rest/services/PPQ%20International%20Segments%20Feature%20Layer/FeatureServer"
LAYER_ID = 0

TOKEN_URL = f"{PORTAL_URL}/generateToken"
LAYER_URL = f"{SERVICE_URL}/{LAYER_ID}"
QUERY_URL = f"{LAYER_URL}/query"

OUTPUT_FILE = "international_segmentation.geojson"

USERNAME = "csuguest28"
PASSWORD = "IEatChildren7!"

def generate_token():
    params = {
        "username": USERNAME,
        "password": PASSWORD,
        "client": "referer",
        "referer": "https://csurams.maps.arcgis.com",
        "expiration": 120,
        "f": "json",
    }

    response = requests.post(TOKEN_URL, data=params)
    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(data["error"])

    return data["token"]


def request_json(url, params, token):
    params = dict(params)
    params["token"] = token
    params["f"] = "json"

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(data["error"])

    return data


def get_layer_info(token):
    return request_json(LAYER_URL, {}, token)


def get_feature_count(token):
    data = request_json(
        QUERY_URL,
        {
            "where": "1=1",
            "returnCountOnly": "true",
        },
        token,
    )

    return data["count"]


def arcgis_ring_to_geojson_ring(ring):
    """
    GeoJSON polygons need closed rings.
    ArcGIS usually returns closed rings already, but this makes it safe.
    """
    if not ring:
        return ring

    if ring[0] != ring[-1]:
        ring = ring + [ring[0]]

    return ring


def arcgis_polygon_to_geojson_geometry(geometry):
    """
    Converts ArcGIS polygon JSON to basic GeoJSON Polygon.

    For this dataset, county polygons should usually be simple enough.
    This keeps all rings in one Polygon. For many county datasets, this is fine.
    """
    if not geometry or "rings" not in geometry:
        return None

    rings = geometry["rings"]
    geojson_rings = [arcgis_ring_to_geojson_ring(ring) for ring in rings if len(ring) >= 4]

    if not geojson_rings:
        return None

    return {
        "type": "Polygon",
        "coordinates": geojson_rings,
    }


def arcgis_feature_to_geojson_feature(feature):
    geometry = arcgis_polygon_to_geojson_geometry(feature.get("geometry"))
    attributes = feature.get("attributes", {})

    if geometry is None:
        return None

    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": attributes,
    }


def download_all_features(token, page_size=1000):
    all_geojson_features = []

    result_offset = 0
    total_downloaded = 0

    total_count = get_feature_count(token)
    print(f"Server reports {total_count} total features.")

    while True:
        print(f"Downloading records {result_offset} to {result_offset + page_size - 1}...")

        data = request_json(
            QUERY_URL,
            {
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "4326",
                "resultOffset": result_offset,
                "resultRecordCount": page_size,
                "orderByFields": "OBJECTID",
            },
            token,
        )

        features = data.get("features", [])

        if not features:
            break

        for feature in features:
            geojson_feature = arcgis_feature_to_geojson_feature(feature)

            if geojson_feature is not None:
                all_geojson_features.append(geojson_feature)

        total_downloaded += len(features)
        print(f"Downloaded {total_downloaded}/{total_count}")

        if len(features) < page_size:
            break

        result_offset += page_size

    return all_geojson_features


def save_geojson(features, output_file):
    collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(collection, f)

    print(f"\nSaved {len(features)} features to {output_file}")


def main():
    token = generate_token()

    layer_info = get_layer_info(token)

    print("Layer name:", layer_info.get("name"))
    print("Geometry type:", layer_info.get("geometryType"))

    print("\nFields:")
    for field in layer_info.get("fields", []):
        print(f"- {field['name']} ({field['type']})")

    features = download_all_features(token, page_size=10000)

    save_geojson(features, OUTPUT_FILE)


if __name__ == "__main__":
    main()
