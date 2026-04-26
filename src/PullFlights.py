import os
import json
import time
import requests


PORTAL_URL = "https://csurams.maps.arcgis.com/sharing/rest"
SERVICE_URL = (
    "https://services1.arcgis.com/KNdRU5cN6ENqCTjk/arcgis/rest/services/"
    "PPQ%20International%20Segments%20Feature%20Layer/FeatureServer"
)

LAYER_ID = 0

TOKEN_URL = f"{PORTAL_URL}/generateToken"
LAYER_URL = f"{SERVICE_URL}/{LAYER_ID}"
QUERY_URL = f"{LAYER_URL}/query"

OUTPUT_FILE = "international_segmentation.geojson"

USERNAME = "csuguest28"
PASSWORD = "IEatChildren7!"

# Keep this modest. Huge objectIds lists can fail.
CHUNK_SIZE = 10000


def generate_token():
    if not USERNAME or not PASSWORD:
        raise RuntimeError(
            "Missing credentials.\n"
            "PowerShell:\n"
            '  $env:ARCGIS_USERNAME="your_username"\n'
            '  $env:ARCGIS_PASSWORD="your_password"\n'
            "Then run:\n"
            "  python PullFlights.py"
        )

    params = {
        "username": USERNAME,
        "password": PASSWORD,
        "client": "referer",
        "referer": "https://csurams.maps.arcgis.com",
        "expiration": 120,
        "f": "json",
    }

    response = requests.post(TOKEN_URL, data=params, timeout=60)
    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(f"Token error: {data['error']}")

    return data["token"]


def request_json(url, params=None, token=None, method="POST", retries=3):
    params = dict(params or {})

    if token:
        params["token"] = token

    if "f" not in params:
        params["f"] = "json"

    for attempt in range(1, retries + 1):
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, timeout=120)
            else:
                response = requests.post(url, data=params, timeout=120)

            try:
                response.raise_for_status()
            except requests.HTTPError as e:
                print("\nHTTP error:")
                print("  status:", response.status_code)
                print("  url:", url)
                print("  method:", method)
                print("  response text:", response.text[:1000])
                raise e

            data = response.json()

            if "error" in data:
                print("\nArcGIS error:")
                print("  URL:", url)
                print("  Method:", method)
                print("  Params:")
                for k, v in params.items():
                    if k.lower() == "token":
                        print(f"    {k}: <hidden>")
                    elif k.lower() == "password":
                        print(f"    {k}: <hidden>")
                    elif k == "objectIds":
                        ids = str(v).split(",")
                        print(f"    objectIds: {len(ids)} ids")
                    else:
                        print(f"    {k}: {v}")

                print("  Error:", data["error"])
                return None

            return data

        except Exception as e:
            print(f"Request failed attempt {attempt}/{retries}: {e}")

            if attempt < retries:
                time.sleep(2)

    return None


def get_layer_info(token):
    data = request_json(
        LAYER_URL,
        {
            "f": "json",
        },
        token,
        method="GET",
    )

    if data is None:
        raise RuntimeError("Could not read layer metadata.")

    return data


def get_feature_count(token):
    data = request_json(
        QUERY_URL,
        {
            "f": "json",
            "where": "1=1",
            "returnCountOnly": "true",
        },
        token,
        method="POST",
    )

    if data is None:
        raise RuntimeError("Could not get feature count.")

    return data["count"]


def get_object_ids(token):
    data = request_json(
        QUERY_URL,
        {
            "f": "json",
            "where": "1=1",
            "returnIdsOnly": "true",
        },
        token,
        method="POST",
    )

    if data is None:
        raise RuntimeError("Could not get ObjectIDs.")

    object_ids = data.get("objectIds") or []
    object_id_field = data.get("objectIdFieldName")

    if not object_ids:
        raise RuntimeError("No ObjectIDs returned.")

    object_ids = sorted(object_ids)

    print(f"ObjectID field: {object_id_field}")
    print(f"Total ObjectIDs: {len(object_ids):,}")

    return object_ids


def download_geojson_features(token, object_ids):
    all_features = []
    total = len(object_ids)

    for start in range(0, total, CHUNK_SIZE):
        chunk = object_ids[start:start + CHUNK_SIZE]

        print(f"Downloading {start + 1:,} to {min(start + CHUNK_SIZE, total):,} of {total:,}...")

        data = request_json(
            QUERY_URL,
            {
                "f": "geojson",
                "where": "1=1",
                "objectIds": ",".join(map(str, chunk)),
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "4326",
            },
            token,
            method="POST",
        )

        if data is None:
            print("Chunk failed. Trying smaller emergency chunks...")

            emergency_features = download_emergency_chunks(token, chunk)

            if not emergency_features:
                print("Emergency chunks failed too. Stopping early.")
                break

            all_features.extend(emergency_features)
        else:
            features = data.get("features", [])
            all_features.extend(features)

        print(f"Downloaded {len(all_features):,}/{total:,}")

    return all_features


def download_emergency_chunks(token, object_ids):
    """
    If a 250-id chunk fails, try tiny chunks.
    """
    features = []

    for start in range(0, len(object_ids), 25):
        chunk = object_ids[start:start + 25]

        data = request_json(
            QUERY_URL,
            {
                "f": "geojson",
                "where": "1=1",
                "objectIds": ",".join(map(str, chunk)),
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "4326",
            },
            token,
            method="POST",
        )

        if data is None:
            print(f"Failed tiny chunk starting with ObjectID {chunk[0]}. Skipping it.")
            continue

        features.extend(data.get("features", []))

    return features


def save_geojson(features, output_file):
    collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(collection, f, ensure_ascii=False)

    print(f"\nSaved {len(features):,} features to {output_file}")


def main():
    print("Generating token...")
    token = generate_token()

    print("Reading layer info...")
    layer_info = get_layer_info(token)

    print("Layer name:", layer_info.get("name"))
    print("Geometry type:", layer_info.get("geometryType"))
    print("Max record count:", layer_info.get("maxRecordCount"))

    count = get_feature_count(token)
    print(f"\nServer reports {count:,} total features.")

    object_ids = get_object_ids(token)

    features = download_geojson_features(token, object_ids)

    if not features:
        print("No features downloaded. Exiting.")
        return

    save_geojson(features, OUTPUT_FILE)


if __name__ == "__main__":
    main()
