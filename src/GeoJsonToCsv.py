import json
import csv
from pathlib import Path


def geometry_to_wkt(geometry):
    """
    Converts basic GeoJSON geometry into WKT-like text.
    Supports Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon.
    """
    if geometry is None:
        return ""

    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")

    if geom_type == "Point":
        return f"POINT ({coords[0]} {coords[1]})"

    elif geom_type == "LineString":
        points = ", ".join(f"{x} {y}" for x, y in coords)
        return f"LINESTRING ({points})"

    elif geom_type == "Polygon":
        rings = []
        for ring in coords:
            points = ", ".join(f"{x} {y}" for x, y in ring)
            rings.append(f"({points})")
        return f"POLYGON ({', '.join(rings)})"

    elif geom_type == "MultiPoint":
        points = ", ".join(f"({x} {y})" for x, y in coords)
        return f"MULTIPOINT ({points})"

    elif geom_type == "MultiLineString":
        lines = []
        for line in coords:
            points = ", ".join(f"{x} {y}" for x, y in line)
            lines.append(f"({points})")
        return f"MULTILINESTRING ({', '.join(lines)})"

    elif geom_type == "MultiPolygon":
        polygons = []
        for polygon in coords:
            rings = []
            for ring in polygon:
                points = ", ".join(f"{x} {y}" for x, y in ring)
                rings.append(f"({points})")
            polygons.append(f"({', '.join(rings)})")
        return f"MULTIPOLYGON ({', '.join(polygons)})"

    return json.dumps(geometry)


def geojson_to_csv(input_file, output_file):
    input_file = Path(input_file)
    output_file = Path(output_file)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])

    if not features:
        print("No features found in GeoJSON.")
        return

    rows = []
    all_columns = set()

    for feature in features:
        properties = feature.get("properties", {}) or {}
        geometry = feature.get("geometry", {})

        row = dict(properties)
        row["geometry_type"] = geometry.get("type", "")
        row["geometry_wkt"] = geometry_to_wkt(geometry)

        all_columns.update(row.keys())
        rows.append(row)

    columns = sorted(all_columns)

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Converted {len(rows)} features.")
    print(f"Saved CSV to: {output_file}")


if __name__ == "__main__":
    geojson_to_csv(
        input_file="domestic_flights_100k.geojson",
        output_file="domestic_flights_100k.geojson"
    )
