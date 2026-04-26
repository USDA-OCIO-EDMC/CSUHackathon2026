from pathlib import Path
import re

import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats


BASE_DIR = Path(__file__).parent

COUNTRY_FOLDER = BASE_DIR / "ne_10m_admin_0_countries"

TMIN_DIR = BASE_DIR / "wc2.1_cruts4.09_10m_tmin_2020-2024"
TMAX_DIR = BASE_DIR / "wc2.1_cruts4.09_10m_tmax_2020-2024"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "country_monthly_tmin_tmax_2020_2024.csv"


def find_country_shapefile():
    matches = list(COUNTRY_FOLDER.rglob("*.shp"))
    if not matches:
        raise FileNotFoundError(f"No .shp file found in {COUNTRY_FOLDER}")
    return matches[0]


def parse_year_month(path: Path):
    """
    Works with common WorldClim historical monthly names like:
    wc2.1_10m_tmin_2020-01.tif
    wc2.1_cruts4.09_10m_tmin_2020-01.tif
    """
    match = re.search(r"(\d{4})[-_](\d{1,2})", path.name)
    if not match:
        raise ValueError(f"Could not parse year/month from filename: {path.name}")

    year = int(match.group(1))
    month = int(match.group(2))
    return year, month


def build_file_index(folder: Path, variable: str):
    files = sorted(folder.rglob(f"*{variable}*.tif"))

    if not files:
        raise FileNotFoundError(f"No {variable} .tif files found in {folder}")

    index = {}
    for f in files:
        year, month = parse_year_month(f)
        index[(year, month)] = f

    return index


def maybe_scale_temperature(value):
    """
    Some WorldClim temperature rasters are stored as °C * 10.
    If values look like 250, convert to 25.0.
    If values already look like 25.0, leave them alone.
    """
    if value is None:
        return None

    if abs(value) > 100:
        return value / 10

    return value


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    country_path = find_country_shapefile()
    print(f"Using country shapefile: {country_path}")

    print("Loading country boundaries...")
    countries = gpd.read_file(country_path)
    countries = countries.to_crs("EPSG:4326")

    useful_cols = [
        "ADMIN",
        "NAME",
        "SOVEREIGNT",
        "ISO_A3",
        "CONTINENT",
        "REGION_UN",
        "geometry",
    ]
    keep_cols = [c for c in useful_cols if c in countries.columns]
    countries = countries[keep_cols].copy()

    if "ADMIN" in countries.columns:
        countries["country"] = countries["ADMIN"]
    elif "NAME" in countries.columns:
        countries["country"] = countries["NAME"]
    else:
        countries["country"] = countries.index.astype(str)

    print("Indexing WorldClim files...")
    tmin_files = build_file_index(TMIN_DIR, "tmin")
    tmax_files = build_file_index(TMAX_DIR, "tmax")

    common_dates = sorted(set(tmin_files.keys()) & set(tmax_files.keys()))

    if not common_dates:
        raise ValueError(
            "No matching year/month pairs found between tmin and tmax folders. "
            "Make sure you downloaded the matching tmax 2020-2024 folder."
        )

    print(f"Found {len(common_dates)} matching monthly tmin/tmax files.")

    all_rows = []

    for year, month in common_dates:
        print(f"Processing {year}-{month:02d}...")

        tmin_stats = zonal_stats(
            countries,
            str(tmin_files[(year, month)]),
            stats=["mean"],
            nodata=-3.4e38,
            all_touched=True,
        )

        tmax_stats = zonal_stats(
            countries,
            str(tmax_files[(year, month)]),
            stats=["mean"],
            nodata=-3.4e38,
            all_touched=True,
        )

        for idx, country_row in countries.iterrows():
            avg_tmin = maybe_scale_temperature(tmin_stats[idx]["mean"])
            avg_tmax = maybe_scale_temperature(tmax_stats[idx]["mean"])

            all_rows.append({
                "country": country_row.get("country"),
                "iso_a3": country_row.get("ISO_A3", None),
                "continent": country_row.get("CONTINENT", None),
                "region_un": country_row.get("REGION_UN", None),
                "year": year,
                "month": month,
                "avg_tmin_c": avg_tmin,
                "avg_tmax_c": avg_tmax,
                "avg_temp_c": None if avg_tmin is None or avg_tmax is None else (avg_tmin + avg_tmax) / 2,
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDone. Saved to: {OUTPUT_PATH}")
    print(df.head(20))


if __name__ == "__main__":
    main()