import csv
import json
import os
import sys


INPUT_GEOJSON = "international_segmentation.geojson"
OUTPUT_CSV = "international_segmentation.csv"


def geometry_to_columns(geometry):
    """
    Converts GeoJSON geometry into simple CSV columns.

    For Point:
        geometry_lon, geometry_lat

    For other geometry types:
        geometry_type, geometry_json
    """
    if not geometry:
        return {
            "geometry_type": None,
            "geometry_lon": None,
            "geometry_lat": None,
            "geometry_json": None,
        }

    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")

    row = {
        "geometry_type": geom_type,
        "geometry_lon": None,
        "geometry_lat": None,
        "geometry_json": None,
    }

    if geom_type == "Point" and isinstance(coords, list) and len(coords) >= 2:
        # GeoJSON order is [longitude, latitude]
        row["geometry_lon"] = coords[0]
        row["geometry_lat"] = coords[1]
    else:
        # Keep complex geometry safely as JSON text.
        row["geometry_json"] = json.dumps(geometry, ensure_ascii=False)

    return row


def collect_fieldnames(features):
    """
    Builds a stable CSV header from:
    - GeoJSON id
    - geometry columns
    - every property key found in the file
    """
    fieldnames = [
        "feature_id",
        "geometry_type",
        "geometry_lon",
        "geometry_lat",
        "geometry_json",
    ]

    property_keys = []

    for feature in features:
        props = feature.get("properties") or {}

        for key in props.keys():
            if key not in property_keys:
                property_keys.append(key)

    return fieldnames + property_keys


def feature_to_row(feature, fieldnames):
    row = {key: None for key in fieldnames}

    row["feature_id"] = feature.get("id")

    geometry_cols = geometry_to_columns(feature.get("geometry"))
    row.update(geometry_cols)

    props = feature.get("properties") or {}
    row.update(props)

    return row


def convert_geojson_to_csv(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input GeoJSON not found: {input_path}")

    print(f"Reading GeoJSON: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") != "FeatureCollection":
        raise ValueError("Expected a GeoJSON FeatureCollection.")

    features = data.get("features", [])

    if not features:
        raise ValueError("GeoJSON has no features.")

    print(f"Found {len(features):,} features.")

    fieldnames = collect_fieldnames(features)

    print(f"Writing CSV: {output_path}")
    print(f"Columns: {len(fieldnames):,}")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for feature in features:
            row = feature_to_row(feature, fieldnames)
            writer.writerow(row)

    print(f"Done. Saved CSV to: {output_path}")


def main():
    input_path = INPUT_GEOJSON
    output_path = OUTPUT_CSV

    # Optional command line usage:
    # python geojson_to_csv.py input.geojson output.csv
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]

    convert_geojson_to_csv(input_path, output_path)


if __name__ == "__main__":
    main()
