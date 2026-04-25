import requests, pandas as pd
import os

NASS_KEY = "8DA01D1A-3B97-3C01-8F19-AEDB794053C3"  # Register at quickstats.nass.usda.gov
STATES = {"IA":"19","CO":"08","WI":"55","MO":"29","NE":"31"}

def get_nass_yields(state_fips, year_start=2010, year_end=2023):
    params = {
        "key": NASS_KEY,
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "commodity_desc": "CORN",
        "statisticcat_desc": "YIELD",
        "unit_desc": "BU / ACRE",
        "agg_level_desc": "COUNTY",
        "state_fips_code": state_fips,
        "year__GE": year_start,
        "year__LE": year_end,
        "format": "JSON"
    }
    r = requests.get("https://quickstats.nass.usda.gov/api/api_GET/", params=params)
    data = r.json().get("data", [])
    return pd.DataFrame(data)[["year","county_name","Value"]] \
             .rename(columns={"Value":"yield_bu_acre"}) \
             .assign(yield_bu_acre=lambda d: pd.to_numeric(d.yield_bu_acre, errors="coerce"))
