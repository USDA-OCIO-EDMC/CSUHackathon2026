print("Loading ArcGIS data...")
import arcgis_data
print("Done.")

# import weather_data

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
print("Loading imports data...")
imports_df = pd.read_csv("us_imports_2020_2025.csv")
print("Done.")

print("Loading Mediterranean fruit fly presence data...")
_mediterranean_df = pd.read_csv("distribution_mediterranean.csv")
print("Done.")

print("Loading Oriental fruit fly presence data...")
_oriental_df = pd.read_csv("distribution_oriental.csv")
print("Done.")

print("Loading Mexican fruit fly presence data...")
_mexican_df = pd.read_csv("distribution_mexican.csv")
print("Done.")

def _build_country_status_lookup(target_df):
	lookup = {}

	for _, row in target_df.iterrows():
		country_raw = row.get("country")
		status_raw = row.get("Status")

		if pd.isna(country_raw) or pd.isna(status_raw):
			continue

		country = str(country_raw).strip().lower()
		status = str(status_raw).strip()
		if not country or not status:
			continue

		state_raw = row.get("state", "")
		state = "" if pd.isna(state_raw) else str(state_raw).strip()
		is_country_level = (state == "")

		existing = lookup.get(country)
		if existing is None or (is_country_level and not existing["is_country_level"]):
			lookup[country] = {
				"status": status,
				"is_country_level": is_country_level,
			}

	return {country: info["status"] for country, info in lookup.items()}


_mediterranean_status_by_country = _build_country_status_lookup(_mediterranean_df)
_oriental_status_by_country = _build_country_status_lookup(_oriental_df)
_mexican_status_by_country = _build_country_status_lookup(_mexican_df)

import json
print("Loading fruit flies data...")
with open("fly_data.json", "r") as jsonfile: 
    fly_data = json.load(jsonfile)
print("Done.")

def risk_data(city_name, monthyear):
	# print(f"Getting risk data for {city_name} in {monthyear}...")

	intl_flights_data = arcgis_data.get_intl_flights_data(city_name, monthyear)
	city_fruitfly_data, _ = arcgis_data.get_nearby_counties_with_fruitfly_data(city_name, monthyear=monthyear)

	imports_data = imports_df[
		(imports_df["MONTH"] == int(monthyear.split("/")[0])) & 
		(imports_df["YEAR"] == int(monthyear.split("/")[1])) & 
		(imports_df["PORT_NAME"].str.lower() == city_name.lower()) &
		(imports_df["CTY_CODE"].str.isnumeric()) &
		(imports_df["CTY_CODE"].str[:1] != "0")
	]

	by_country = (
		imports_data
		.groupby("CTY_NAME", as_index=False)["GEN_VAL_MO"]
		.sum()
		.sort_values("GEN_VAL_MO", ascending=False)
	)

	cargo_value = {
		row.CTY_NAME.lower(): row.GEN_VAL_MO
		for row in by_country.itertuples(index=False)
	}

	risk_countries = {}

	def get_fly_status(country, status_by_country, species_name):
		if species_name in risk_countries[country]:
			return

		status = status_by_country.get(country)
		if status is not None:
			risk_countries[country][species_name] = status

	for country in by_country["CTY_NAME"]:
		risk_countries[country.lower()] = {}
		get_fly_status(country.lower(), _mediterranean_status_by_country, "Mediterranean fruit fly")
		get_fly_status(country.lower(), _oriental_status_by_country, "Oriental fruit fly")
		get_fly_status(country.lower(), _mexican_status_by_country, "Mexican fruit fly")
		
		if country.lower() in risk_countries and not risk_countries[country.lower()]:
			del risk_countries[country.lower()]

	for country in intl_flights_data["PassengerData"]:
		if not country.lower() in risk_countries:
			risk_countries[country.lower()] = {}
		get_fly_status(country.lower(), _mediterranean_status_by_country, "Mediterranean fruit fly")
		get_fly_status(country.lower(), _oriental_status_by_country, "Oriental fruit fly")
		get_fly_status(country.lower(), _mexican_status_by_country, "Mexican fruit fly")

		if country.lower() in risk_countries and not risk_countries[country.lower()]:
			del risk_countries[country.lower()]

	# weather_info = weather_data.get_average_temperature_for_city_month(city_name, monthyear, units="metric")
	# fly_env = {
	# 	"Mediterranean fruit fly": 0,
	# 	"Oriental fruit fly": 0,
	# 	"Mexican fruit fly": 0,
	# }

	# for species in fly_env.keys():
	# 	if "average_temperature" in weather_info:
	# 		avg_temp = weather_info["average_temperature"]
	# 		species_max = fly_data[species]["Temperature Maximum"]
	# 		species_min = fly_data[species]["Temperature Minimum"]
	# 		if species_max > avg_temp > species_min:
	# 			fly_env[species] = 1
				

	data = {
		"PassengerVolume": intl_flights_data["PassengerVolume"],
		"PassengerData": intl_flights_data["PassengerData"],
		"CargoValue": cargo_value,
		"FruitFlyCount": city_fruitfly_data,
		"RiskCountries": risk_countries,
		# "Weather": weather_info,
		# "FlyEnvironmentSuitability": fly_env,
	}

	return data


if __name__ == "__main__":
	# name = "Los Angeles, CA"
	# monthyear = "04/2020"
	default_max_workers = 16

	# print(risk_data(name, monthyear))
	
	with open("track_cities.json", "r") as jsonfile: 
		tracked_cities = json.load(jsonfile)

	monthyears = [f"{month:02d}/{year}" for year in range(2020, 2026) for month in range(1, 13)]
	print("Preloading GIS city/month data...")
	arcgis_data.preload_city_month_data(tracked_cities, monthyears)
	print("Done preloading GIS data.")

	city_risk_data = {}
	for city in tracked_cities:
		city_risk_data[city] = {}

	futures = {}
	total_jobs = len(tracked_cities) * len(monthyears)
	completed_jobs = 0
	with ThreadPoolExecutor(max_workers=default_max_workers) as executor:
		for city in tracked_cities:
			for monthyear in monthyears:
				future = executor.submit(risk_data, city, monthyear)
				futures[future] = (city, monthyear)

		for future in as_completed(futures):
			city, monthyear = futures[future]
			completed_jobs += 1
			print(f"Completed {completed_jobs}/{total_jobs}: {city} {monthyear}")
			try:
				this_city_data = future.result()
				city_risk_data[city][monthyear] = this_city_data
			except Exception as exc:
				print(f"Error getting risk data for {city} in {monthyear}: {exc}")
				city_risk_data[city][monthyear] = {
					"Error": str(exc)
				}

	with open("city_risk_data.json", "w") as jsonfile:
		json.dump(city_risk_data, jsonfile, indent=2)
			

	# intl_flights_data = arcgis_data.get_intl_flights_data(name, monthyear)

	# print(f"Flight Data for {name} in {monthyear}: {intl_flights_data}")

	# city_fruitfly_data, nearby_counties = arcgis_data.get_nearby_counties_with_fruitfly_data(name, monthyear)	

	# fruitfly_data = arcgis_data.get_fruitfly_data(name, monthyear)

	# print(f"Fruit Fly Data for {name} in {monthyear}: {city_fruitfly_data}")
	# print(f"Nearby Counties with Fruit Fly Data for {name} in {monthyear}: {nearby_counties}")