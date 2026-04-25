from arcgis.gis import GIS
from arcgis.geocoding import geocode
from copy import deepcopy
from threading import Lock
from collections import defaultdict
import os
import pickle
import atexit
import warnings
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings(
    "ignore",
    message=r"Unverified HTTPS request is being made to host 'geocode\.arcgis\.com'.*",
    category=InsecureRequestWarning,
)

# Keep an active GIS for geocoding, but use a verified public endpoint.
gis = GIS(url="https://www.arcgis.com", verify_cert=True)

# INTL_FLIGHTS_ITEM_ID = "ffc2e851a5fd4ef7a1e2581072b5c46d"

import arcpy

intl_flights_project = arcpy.mp.ArcGISProject(
    r"C:\Users\Ryan\Documents\arcgis_data\intl_flight_data\intl_flight_data.aprx"
)
fruitfly_project = arcpy.mp.ArcGISProject(
    r"C:\Users\Ryan\Documents\arcgis_data\fruitfly_data\fruitfly_data.aprx"
)

FRUITFLY_LAYER_NAME = "Fruit Fly Detections"
INTL_FLIGHTS_LAYER_NAME = "InternationalSegments"

_nearest_counties_cache = {}
_nearest_counties_cache_lock = Lock()
_fruitfly_data_cache = {}
_fruitfly_data_cache_lock = Lock()
_preloaded_intl_flights_cache = {}
_preloaded_city_fruitfly_cache = {}
_preloaded_nearby_counties_cache = {}
_preloaded_lock = Lock()

_CACHE_VERSION = 1
_PERSISTENT_CACHE_FILE = os.path.join(os.path.dirname(__file__), "arcgis_query_cache.pkl")


def _load_persistent_cache():
    if not os.path.exists(_PERSISTENT_CACHE_FILE):
        return

    try:
        with open(_PERSISTENT_CACHE_FILE, "rb") as infile:
            payload = pickle.load(infile)

        if not isinstance(payload, dict) or payload.get("version") != _CACHE_VERSION:
            return

        nearest = payload.get("nearest_counties_cache", {})
        fruitfly = payload.get("fruitfly_data_cache", {})
        preloaded_intl = payload.get("preloaded_intl_flights_cache", {})
        preloaded_city_fruitfly = payload.get("preloaded_city_fruitfly_cache", {})
        preloaded_nearby = payload.get("preloaded_nearby_counties_cache", {})

        if isinstance(nearest, dict):
            with _nearest_counties_cache_lock:
                _nearest_counties_cache.update(nearest)

        if isinstance(fruitfly, dict):
            with _fruitfly_data_cache_lock:
                _fruitfly_data_cache.update(fruitfly)

        if isinstance(preloaded_intl, dict) and isinstance(preloaded_city_fruitfly, dict) and isinstance(preloaded_nearby, dict):
            with _preloaded_lock:
                _preloaded_intl_flights_cache.update(preloaded_intl)
                _preloaded_city_fruitfly_cache.update(preloaded_city_fruitfly)
                _preloaded_nearby_counties_cache.update(preloaded_nearby)
    except Exception as exc:
        print(f"Warning: failed to load ArcGIS persistent cache: {exc}")


def save_persistent_cache():
    try:
        with _nearest_counties_cache_lock:
            nearest = deepcopy(_nearest_counties_cache)
        with _fruitfly_data_cache_lock:
            fruitfly = deepcopy(_fruitfly_data_cache)
        with _preloaded_lock:
            preloaded_intl = deepcopy(_preloaded_intl_flights_cache)
            preloaded_city_fruitfly = deepcopy(_preloaded_city_fruitfly_cache)
            preloaded_nearby = deepcopy(_preloaded_nearby_counties_cache)

        payload = {
            "version": _CACHE_VERSION,
            "nearest_counties_cache": nearest,
            "fruitfly_data_cache": fruitfly,
            "preloaded_intl_flights_cache": preloaded_intl,
            "preloaded_city_fruitfly_cache": preloaded_city_fruitfly,
            "preloaded_nearby_counties_cache": preloaded_nearby,
        }

        with open(_PERSISTENT_CACHE_FILE, "wb") as outfile:
            pickle.dump(payload, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        print(f"Warning: failed to save ArcGIS persistent cache: {exc}")


_load_persistent_cache()
atexit.register(save_persistent_cache)


def _escape_sql_string(value):
    return str(value).replace("'", "''")


def _build_in_or_clause(field_name, values, quote_strings=True, chunk_size=200):
    values = list(values)
    if not values:
        return None

    chunks = [values[i:i + chunk_size] for i in range(0, len(values), chunk_size)]
    parts = []
    for chunk in chunks:
        if quote_strings:
            serialized = ", ".join(f"'{_escape_sql_string(v)}'" for v in chunk)
        else:
            serialized = ", ".join(str(v) for v in chunk)
        parts.append(f"{field_name} IN ({serialized})")

    if len(parts) == 1:
        return parts[0]
    return "(" + " OR ".join(parts) + ")"

def get_layer_by_name(project, layer_name):
    """
    Finds a layer in the currently open ArcGIS Pro map by name.
    """
    active_map = my_map = project.listMaps("Map")[0]

    for layer in active_map.listLayers():
        if layer.name == layer_name:
            return layer

    raise ValueError(f"Could not find layer named: {layer_name}")

fruitfly_layer = get_layer_by_name(fruitfly_project, FRUITFLY_LAYER_NAME)
intl_flights_layer = get_layer_by_name(intl_flights_project, INTL_FLIGHTS_LAYER_NAME)

# ArcPy layer objects can be unreliable when shared across threads.
# Use datasource paths for SearchCursor inputs instead.
fruitfly_data_source = fruitfly_layer.dataSource
intl_flights_data_source = intl_flights_layer.dataSource

def get_fruitfly_data(county, monthyear):
    """
    Gets fruit fly counts for a county and month/year from a local ArcGIS Pro layer.

    county example: "Hidalgo County"
    monthyear example: "01/2024"
    """

    county_safe = county
    monthyear_safe = monthyear

    cache_key = (county_safe.lower(), monthyear_safe)
    with _fruitfly_data_cache_lock:
        cached = _fruitfly_data_cache.get(cache_key)
    if cached is not None:
        return dict(cached)

    where_clause = f"NAME = '{county_safe}' AND monthyear = '{monthyear_safe}'"

    out_data = {}

    fields = ["CommonName", "Count_"]

    with arcpy.da.SearchCursor(fruitfly_data_source, fields, where_clause) as cursor:
        for common_name, count in cursor:
            out_data[common_name] = count

    with _fruitfly_data_cache_lock:
        _fruitfly_data_cache[cache_key] = dict(out_data)

    return out_data

def get_intl_flights_data(city, monthyear):
    """
    Gets total international passenger volume for a city and month/year
    from a local ArcGIS Pro layer.

    city example: "Dallas"
    monthyear example: "01/2024"
    """

    preload_key = (city.lower(), monthyear)
    with _preloaded_lock:
        preloaded = _preloaded_intl_flights_cache.get(preload_key)
    if preloaded is not None:
        return {
            "PassengerVolume": preloaded["PassengerVolume"],
            "PassengerData": dict(preloaded["PassengerData"]),
        }

    month, year = monthyear.split("/")
    month = int(month)
    year = int(year)

    where_clause = (
        f"DEST_CITY_NAME = '{city}' "
        f"AND YEAR_WADS = {year} "
        f"AND MONTH_WADS = {month}"
    )

    passenger_volume = 0

    fields = ["PASSENGERS", "ORIGIN_COUNTRY_NAME"]

    byOriginCountry = {}

    with arcpy.da.SearchCursor(intl_flights_data_source, fields, where_clause) as cursor:
        for (passengers, origin_country) in cursor:
            if passengers is not None:
                passenger_volume += passengers
                origin_country = origin_country.lower()
                byOriginCountry[origin_country] = byOriginCountry.get(origin_country, 0) + passengers

    computed = {
        "PassengerVolume": passenger_volume,
        "PassengerData": byOriginCountry,
    }

    # Cache direct query results so future runs/calls can reuse them.
    with _preloaded_lock:
        _preloaded_intl_flights_cache[preload_key] = {
            "PassengerVolume": computed["PassengerVolume"],
            "PassengerData": dict(computed["PassengerData"]),
        }

    return computed


fruitfly_fl = None
intl_flights_fl = None

def get_nearest_counties(city, country="USA", max_counties=5):
    """
    Get nearby counties for a given city using ArcGIS geocoding only.

    This does not require city or county layers.
    It uses the ArcGIS World Geocoding Service.

    Parameters:
        city (str): City name, e.g. "Dallas"
        state (str): Optional state name, e.g. "Texas"
        country (str): Country name, default "USA"
        max_counties (int): Number of nearby counties to return

    Returns:
        list[dict]: Nearby county matches
    """

    cache_key = (city.strip().lower(), country.strip().lower(), int(max_counties))
    with _nearest_counties_cache_lock:
        cached = _nearest_counties_cache.get(cache_key)
    if cached is not None:
        return deepcopy(cached)

    # Build city query
    city_query = ", ".join(part for part in [city, country] if part)

    city_results = geocode(
        address=city_query,
        max_locations=1,
        as_featureset=False
    )

    if not city_results:
        raise ValueError(f"Could not geocode city: {city_query}")

    city_match = city_results[0]
    city_location = city_match["location"]

    # Search for nearby subregions.
    # In the United States, Subregion usually means county.
    county_results = geocode(
        address="",
        location=city_location,
        category="Subregion",
        max_locations=max_counties,
        as_featureset=False
    )

    counties = []

    for result in county_results:
        attrs = result.get("attributes", {})

        counties.append({
            "county": attrs.get("Subregion") or result.get("address"),
            "state": attrs.get("Region"),
            "country": attrs.get("Country"),
            "matched_address": result.get("address"),
            "score": result.get("score"),
            "x": result.get("location", {}).get("x"),
            "y": result.get("location", {}).get("y"),
        })

    with _nearest_counties_cache_lock:
        _nearest_counties_cache[cache_key] = deepcopy(counties)

    return counties

# def load_fruitfly_data():
# 	global fruitfly_fl
# 	fruitfly_item = gis.content.get(FRUITFLY_ITEM_ID)
# 	fruitfly_fl = fruitfly_item.layers[0]

# def get_fruitfly_data(county, monthyear):
# 	"""
# 	:param county: Name of the location (county) to query
# 	:param monthyear: Month and year to query in the format "MM/YYYY"
# 	:return: A dictionary of fruit fly counts for the specified location and month/year
# 	"""

# 	if fruitfly_fl is None:
# 		load_fruitfly_data()

# 	# county_geocode = geocode(county)
# 	# highest_score = max(result['score'] for result in county_geocode)
# 	# county_geocode = [result for result in county_geocode if result['score'] == highest_score]
# 	# for result in county_geocode:
# 	# 	print(f"Geocoded location for {county}: {result['attributes']['City']} (Score: {result['score']})")
# 	# print(f"Best geocoded location for {county}: {county_geocode[0]['attributes']['City']} (Score: {county_geocode[0]['score']})")

# 	fruitFlyData = fruitfly_fl.query(where=f"NAME='{county}' AND monthyear='{monthyear}'", return_geometry=False)

# 	outData = {}
# 	for feature in fruitFlyData.features:
# 		outData[feature.attributes["CommonName"]] = feature.attributes["Count_"]

# 	return outData

# def load_intl_flights_data():
# 	global intl_flights_fl
# 	intl_flights_item = gis.content.get(INTL_FLIGHTS_ITEM_ID)
# 	intl_flights_fl = intl_flights_item.layers[0]

# def get_intl_flights_data(city, monthyear):
# 	"""
# 	:param city: Name of the location (city) to query
# 	:param monthyear: Month and year to query in the format "MM/YYYY"
# 	:return: A dictionary of international flight counts for the specified location and month/year
# 	"""

# 	if intl_flights_fl is None:
# 		load_intl_flights_data()

# 	# print(intl_flights_fl.query(where=f"OBJECTID=429520", return_geometry=False))

# 	intlFlightsData = intl_flights_fl.query(where=f"DEST_CITY_NAME='{city}' AND YEAR_WADS={monthyear.split('/')[1]} AND MONTH_WADS={monthyear.split('/')[0]}", return_geometry=False)

# 	# print(intlFlightsData)

# 	outData = {
# 		"PassengerVolume": 0,
# 	}
# 	for feature in intlFlightsData.features:
# 		outData["PassengerVolume"] += feature.attributes["PASSENGERS"]

# 	return outData

def get_nearby_counties_with_fruitfly_data(city, country="USA", monthyear="01/2024"):
    preload_key = (city.lower(), monthyear)
    with _preloaded_lock:
        preloaded_city = _preloaded_city_fruitfly_cache.get(preload_key)
        preloaded_counties = _preloaded_nearby_counties_cache.get(preload_key)
    if preloaded_city is not None and preloaded_counties is not None:
        return (dict(preloaded_city), deepcopy(preloaded_counties))

    nearby_counties = get_nearest_counties(city, country)

    city_fruitfly_data = {}
    for county_info in nearby_counties:
        county_name = county_info["county"]
        try:
            fruitfly_data = get_fruitfly_data(county_name, monthyear)
            for key, value in fruitfly_data.items():
                city_fruitfly_data[key] = city_fruitfly_data.get(key, 0) + value
            county_info["fruitfly_data"] = fruitfly_data
        except Exception as e:
            print(f"Error fetching fruit fly data for {county_name}: {e}")
            county_info["fruitfly_data"] = None

    # Cache computed city/month aggregate so it can be reused directly.
    with _preloaded_lock:
        _preloaded_city_fruitfly_cache[preload_key] = dict(city_fruitfly_data)
        _preloaded_nearby_counties_cache[preload_key] = deepcopy(nearby_counties)

    return (city_fruitfly_data, nearby_counties)


def preload_city_month_data(cities, monthyears, country="USA", max_counties=5):
    """
    Preload all GIS-backed city/month inputs up front.

    This performs bulk reads and stores results in in-memory caches so
    repeated risk calculations do not hit GIS systems per city/month call.
    """

    unique_cities = list(dict.fromkeys(city for city in cities if city))
    unique_monthyears = list(dict.fromkeys(monthyear for monthyear in monthyears if monthyear))

    if not unique_cities or not unique_monthyears:
        return

    requested_keys = {
        (city.strip().lower(), monthyear)
        for city in unique_cities
        for monthyear in unique_monthyears
    }

    with _preloaded_lock:
        missing_keys = [
            key for key in requested_keys
            if key not in _preloaded_intl_flights_cache
            or key not in _preloaded_city_fruitfly_cache
            or key not in _preloaded_nearby_counties_cache
        ]

    # Everything requested is already cached; skip GIS queries entirely.
    if not missing_keys:
        return

    missing_city_keys = {}
    for city in unique_cities:
        normalized = city.strip().lower()
        missing_months = {monthyear for key_city, monthyear in missing_keys if key_city == normalized}
        if missing_months:
            missing_city_keys[city] = missing_months

    if not missing_city_keys:
        return

    city_to_nearby = {}
    county_name_set = set()
    for city in missing_city_keys:
        nearby = get_nearest_counties(city, country=country, max_counties=max_counties)
        city_to_nearby[city] = nearby
        for county_info in nearby:
            county_name = county_info.get("county")
            if county_name:
                county_name_set.add(county_name)

    missing_monthyears = sorted({monthyear for _, monthyear in missing_keys})

    month_clause = _build_in_or_clause("monthyear", missing_monthyears, quote_strings=True)
    county_clause = _build_in_or_clause("NAME", county_name_set, quote_strings=True)
    fruitfly_where = f"({month_clause}) AND ({county_clause})" if county_clause else month_clause

    county_month_fruitfly = defaultdict(dict)
    fruitfly_fields = ["NAME", "monthyear", "CommonName", "Count_"]
    if fruitfly_where:
        with arcpy.da.SearchCursor(fruitfly_data_source, fruitfly_fields, fruitfly_where) as cursor:
            for county_name, monthyear, common_name, count in cursor:
                if not county_name or not monthyear or not common_name or count is None:
                    continue
                key = (county_name.lower(), monthyear)
                county_month_fruitfly[key][common_name] = county_month_fruitfly[key].get(common_name, 0) + count

    for key, fly_data in county_month_fruitfly.items():
        with _fruitfly_data_cache_lock:
            _fruitfly_data_cache[key] = dict(fly_data)

    city_month_fruitfly = {}
    city_month_nearby = {}
    for city, months_for_city in missing_city_keys.items():
        nearby_counties = city_to_nearby[city]
        for monthyear in sorted(months_for_city):
            city_key = (city.lower(), monthyear)
            aggregate = {}
            enriched_nearby = deepcopy(nearby_counties)
            for county_info in enriched_nearby:
                county_name = county_info.get("county")
                if not county_name:
                    county_info["fruitfly_data"] = None
                    continue
                fly_data = county_month_fruitfly.get((county_name.lower(), monthyear), {})
                county_info["fruitfly_data"] = dict(fly_data)
                for fly_name, count in fly_data.items():
                    aggregate[fly_name] = aggregate.get(fly_name, 0) + count

            city_month_fruitfly[city_key] = dict(aggregate)
            city_month_nearby[city_key] = enriched_nearby

    months = sorted({int(monthyear.split("/")[0]) for monthyear in missing_monthyears})
    years = sorted({int(monthyear.split("/")[1]) for monthyear in missing_monthyears})
    month_numbers_clause = _build_in_or_clause("MONTH_WADS", months, quote_strings=False)
    year_numbers_clause = _build_in_or_clause("YEAR_WADS", years, quote_strings=False)
    cities_clause = _build_in_or_clause("DEST_CITY_NAME", list(missing_city_keys.keys()), quote_strings=True)

    intl_where_parts = [part for part in [month_numbers_clause, year_numbers_clause, cities_clause] if part]
    intl_where = " AND ".join(f"({part})" for part in intl_where_parts)

    city_month_intl = {}
    intl_fields = ["DEST_CITY_NAME", "MONTH_WADS", "YEAR_WADS", "PASSENGERS", "ORIGIN_COUNTRY_NAME"]
    if intl_where:
        with arcpy.da.SearchCursor(intl_flights_data_source, intl_fields, intl_where) as cursor:
            for dest_city, month, year, passengers, origin_country in cursor:
                if not dest_city or month is None or year is None:
                    continue

                monthyear = f"{int(month):02d}/{int(year)}"
                city_key = (dest_city.lower(), monthyear)
                if city_key not in city_month_intl:
                    city_month_intl[city_key] = {
                        "PassengerVolume": 0,
                        "PassengerData": {},
                    }

                if passengers is not None:
                    city_month_intl[city_key]["PassengerVolume"] += passengers
                    if origin_country:
                        origin_country = origin_country.lower()
                        origin_map = city_month_intl[city_key]["PassengerData"]
                        origin_map[origin_country] = origin_map.get(origin_country, 0) + passengers

    for city, months_for_city in missing_city_keys.items():
        for monthyear in sorted(months_for_city):
            city_key = (city.lower(), monthyear)
            if city_key not in city_month_intl:
                city_month_intl[city_key] = {
                    "PassengerVolume": 0,
                    "PassengerData": {},
                }

    with _preloaded_lock:
        _preloaded_intl_flights_cache.update(city_month_intl)
        _preloaded_city_fruitfly_cache.update(city_month_fruitfly)
        _preloaded_nearby_counties_cache.update(city_month_nearby)

    save_persistent_cache()

# print(fruitFlyData)

# print()