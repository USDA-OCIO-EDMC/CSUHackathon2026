"""
analog_years.py — Weather feature extraction using NASA POWER regional API.
"""

import time
import requests
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POWER_REGIONAL_URL = "https://power.larc.nasa.gov/api/temporal/daily/regional"

# Approximate bounding boxes (south, west, north, east) for our 5 states.
# All are within POWER's 10°×10° regional limit.
STATE_BBOXES = {
    "IA": (40.4, -96.7, 43.5, -90.1),
    "CO": (37.0, -109.1, 41.0, -102.0),
    "WI": (42.5, -92.9, 47.1, -86.8),
    "MO": (36.0, -95.8, 40.6, -89.1),
    "NE": (40.0, -104.1, 43.0, -95.3),
}

# Which months are "active" for each forecast window (May=5 … Oct=10)
FORECAST_MONTHS = {
    "aug1":  [5, 6, 7],
    "sep1":  [5, 6, 7, 8],
    "oct1":  [5, 6, 7, 8, 9],
    "final": [5, 6, 7, 8, 9, 10],
}
ALL_MONTHS = [5, 6, 7, 8, 9, 10]
WEATHER_DIM = len(ALL_MONTHS) * 3  # 18


# ---------------------------------------------------------------------------
# Method 1 — fetch raw regional grid from NASA POWER
# ---------------------------------------------------------------------------

def _fetch_power_regional(
    south: float,
    west: float,
    north: float,
    east: float,
    year: int,
    retry: int = 3,
) -> dict:
    """
    Fetch daily T2M_MAX, T2M_MIN, PRECTOTCORR for a bounding box,
    May–Oct of *year*, from the NASA POWER regional endpoint.

    NASA POWER returns a GeoJSON Feature where:
      - geometry.coordinates  : list of [lon, lat] grid points (~0.5° grid)
      - properties.parameter  : { variable: { YYYYMMDD: [value_per_grid_point] } }

    Returns
    -------
    dict with shape:
      {
        variable: {
          YYYYMMDD: { (lat, lon): value }
        }
      }
    Missing-data sentinel (-999.0) entries are dropped.
    """
    params = {
        "parameters": "T2M_MAX,T2M_MIN,PRECTOTCORR",
        "community":  "AG",
        "bbox":       f"{south},{west},{north},{east}",
        "start":      f"{year}0501",
        "end":        f"{year}1031",
        "format":     "JSON",
    }

    for attempt in range(retry):
        try:
            resp = requests.get(POWER_REGIONAL_URL, params=params, timeout=120)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == retry - 1:
                raise RuntimeError(
                    f"NASA POWER regional fetch failed after {retry} attempts "
                    f"(bbox={south},{west},{north},{east}, year={year})"
                ) from exc
            time.sleep(2 ** attempt)

    payload    = resp.json()
    coords     = payload["geometry"]["coordinates"]   # [[lon, lat], ...]
    param_data = payload["properties"]["parameter"]   # {var: {date: [values]}}

    # Re-index so callers can look up by (lat, lon) tuple
    result: dict = {}
    for var, daily in param_data.items():
        result[var] = {}
        for date_str, values in daily.items():
            result[var][date_str] = {
                (float(coords[i][1]), float(coords[i][0])): v
                for i, v in enumerate(values)
                if v != -999.0
            }

    return result


# ---------------------------------------------------------------------------
# Method 2 — nearest-grid-cell lookup
# ---------------------------------------------------------------------------

def _nearest_grid_value(
    grid: dict[tuple[float, float], float],
    lat: float,
    lon: float,
) -> float | None:
    """
    Given a {(lat, lon): value} dict for a single variable on a single day,
    return the value from the closest grid cell to (lat, lon).

    Uses squared Euclidean distance in degree-space — sufficient at 0.5°
    resolution for county-level matching (error < 1 km at these latitudes).

    Returns None if *grid* is empty.
    """
    if not grid:
        return None

    grid_pts  = np.array(list(grid.keys()), dtype=np.float32)   # (N, 2)  [lat, lon]
    target    = np.array([lat, lon], dtype=np.float32)           # (2,)

    diffs     = grid_pts - target                                 # (N, 2)
    sq_dists  = (diffs ** 2).sum(axis=1)                         # (N,)
    nearest_i = int(sq_dists.argmin())

    nearest_key = tuple(grid_pts[nearest_i])
    return grid[nearest_key]


# ---------------------------------------------------------------------------
# Method 3 — build (18,) weather vector from regional grid + county centroid
# ---------------------------------------------------------------------------

def _calc_gdd(tmax_c: float, tmin_c: float, base: float = 10.0) -> float:
    """Daily GDD (base 10 °C, max capped at 30 °C)."""
    tmax_c = min(tmax_c, 30.0)
    tmin_c = max(tmin_c, base)
    return max((tmax_c + tmin_c) / 2.0 - base, 0.0)


def _build_county_weather_vector(
    regional_data: dict,
    lat: float,
    lon: float,
    year: int,
    forecast_date: str,
) -> np.ndarray:
    """
    Produce a (18,) float32 weather feature vector for one county centroid.

    Parameters
    ----------
    regional_data : output of _fetch_power_regional — already indexed as
                    { variable: { YYYYMMDD: { (lat, lon): value } } }
    lat, lon      : county centroid
    year          : YYYY (used to filter date keys by month prefix)
    forecast_date : "aug1" | "sep1" | "oct1" | "final"

    Returns
    -------
    np.ndarray of shape (18,) — [mean_temp_C, total_precip_mm, gdd_sum] × 6 months
    Months beyond the forecast_date cutoff are zero-filled.
    """
    active_months = set(FORECAST_MONTHS.get(forecast_date.lower(), ALL_MONTHS))

    tmax_grid = regional_data["T2M_MAX"]       # {YYYYMMDD: {(lat,lon): val}}
    tmin_grid = regional_data["T2M_MIN"]
    prec_grid = regional_data["PRECTOTCORR"]

    vector: list[float] = []

    for m in ALL_MONTHS:
        if m not in active_months:
            vector.extend([0.0, 0.0, 0.0])
            continue

        prefix = f"{year}{m:02d}"

        tmax_vals, tmin_vals, prec_vals = [], [], []

        for date_str, grid in tmax_grid.items():
            if date_str.startswith(prefix):
                v = _nearest_grid_value(grid, lat, lon)
                if v is not None:
                    tmax_vals.append(v)

        for date_str, grid in tmin_grid.items():
            if date_str.startswith(prefix):
                v = _nearest_grid_value(grid, lat, lon)
                if v is not None:
                    tmin_vals.append(v)

        for date_str, grid in prec_grid.items():
            if date_str.startswith(prefix):
                v = _nearest_grid_value(grid, lat, lon)
                if v is not None:
                    prec_vals.append(v)

        if tmax_vals and tmin_vals:
            mean_temp = (np.mean(tmax_vals) + np.mean(tmin_vals)) / 2.0
            gdd_sum   = sum(_calc_gdd(tx, tn) for tx, tn in zip(tmax_vals, tmin_vals))
        else:
            mean_temp = 0.0
            gdd_sum   = 0.0

        total_prec = float(np.sum(prec_vals)) if prec_vals else 0.0

        vector.extend([float(mean_temp), total_prec, float(gdd_sum)])

    return np.array(vector, dtype=np.float32)


# ---------------------------------------------------------------------------
# Method 4 — public API: fetch weather for all counties in one state
# ---------------------------------------------------------------------------

def get_state_weather_features(
    state_abbr: str,
    year: int,
    forecast_date: str,
    county_centroids: list[tuple[str, float, float]],
) -> dict[str, np.ndarray]:
    """
    Fetch weather features for every county in a state using a single
    regional API call, then extract per-county (18,) vectors.

    Parameters
    ----------
    state_abbr        : "IA" | "CO" | "WI" | "MO" | "NE"
    year              : YYYY
    forecast_date     : "aug1" | "sep1" | "oct1" | "final"
    county_centroids  : list of (county_fips, lat, lon)
                        county_fips is a string, e.g. "19001"

    Returns
    -------
    dict  { county_fips: np.ndarray(18,) }

    Raises
    ------
    KeyError  if state_abbr is not in STATE_BBOXES
    """
    if state_abbr not in STATE_BBOXES:
        raise KeyError(
            f"Unknown state '{state_abbr}'. "
            f"Valid options: {list(STATE_BBOXES.keys())}"
        )

    south, west, north, east = STATE_BBOXES[state_abbr]

    # One API call covers all counties in the state for the entire season
    regional_data = _fetch_power_regional(south, west, north, east, year)

    results: dict[str, np.ndarray] = {}
    for county_fips, lat, lon in county_centroids:
        results[county_fips] = _build_county_weather_vector(
            regional_data, lat, lon, year, forecast_date
        )

    return results


# ---------------------------------------------------------------------------
# Method 5 — real-time single-county query (2025 at demo time)
# ---------------------------------------------------------------------------

POWER_POINT_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def get_county_weather_features_single(
    lat: float,
    lon: float,
    year: int,
    forecast_date: str,
    retry: int = 3,
) -> np.ndarray:
    """
    Fetch a (18,) weather vector for a single county centroid using the
    POWER point endpoint.  Use this at query time for the current year
    (e.g. 2025) where only one county is needed.

    For historical bulk builds use get_state_weather_features instead.

    Parameters
    ----------
    lat, lon      : county centroid (decimal degrees)
    year          : YYYY
    forecast_date : "aug1" | "sep1" | "oct1" | "final"
    retry         : HTTP retry attempts on transient failure

    Returns
    -------
    np.ndarray of shape (18,) float32
    """
    active_months = set(FORECAST_MONTHS.get(forecast_date.lower(), ALL_MONTHS))

    params = {
        "parameters": "T2M_MAX,T2M_MIN,PRECTOTCORR",
        "community":  "AG",
        "longitude":  round(lon, 4),
        "latitude":   round(lat, 4),
        "start":      f"{year}0501",
        "end":        f"{year}1031",
        "format":     "JSON",
    }

    for attempt in range(retry):
        try:
            resp = requests.get(POWER_POINT_URL, params=params, timeout=90)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == retry - 1:
                raise RuntimeError(
                    f"NASA POWER point fetch failed after {retry} attempts "
                    f"for lat={lat}, lon={lon}, year={year}"
                ) from exc
            time.sleep(2 ** attempt)

    raw    = resp.json()["properties"]["parameter"]
    tmax_d = raw["T2M_MAX"]       # {YYYYMMDD: scalar}
    tmin_d = raw["T2M_MIN"]
    prec_d = raw["PRECTOTCORR"]

    vector: list[float] = []

    for m in ALL_MONTHS:
        if m not in active_months:
            vector.extend([0.0, 0.0, 0.0])
            continue

        prefix = f"{year}{m:02d}"

        tmax_vals = [v for k, v in tmax_d.items() if k.startswith(prefix) and v != -999.0]
        tmin_vals = [v for k, v in tmin_d.items() if k.startswith(prefix) and v != -999.0]
        prec_vals = [v for k, v in prec_d.items() if k.startswith(prefix) and v != -999.0]

        if tmax_vals and tmin_vals:
            mean_temp = (np.mean(tmax_vals) + np.mean(tmin_vals)) / 2.0
            gdd_sum   = sum(_calc_gdd(tx, tn) for tx, tn in zip(tmax_vals, tmin_vals))
        else:
            mean_temp = 0.0
            gdd_sum   = 0.0

        total_prec = float(np.sum(prec_vals)) if prec_vals else 0.0
        vector.extend([float(mean_temp), total_prec, float(gdd_sum)])

    return np.array(vector, dtype=np.float32)
