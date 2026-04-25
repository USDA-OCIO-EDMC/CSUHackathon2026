import requests
import pandas as pd
import time
import findNearestStation

def weather(lat, lon):
    TOKEN = "BUQUVsWSPNRvsxrOKGHYVIqxrNhcGPSv"
    BASE_URL = "https://ncdc.noaa.gov/cdo-web/api/v2"
    headers = {"token": TOKEN}

    STATION_ID = findStation(lat, lon)

    def fetch_weather(station_id, start_date, end_date):
        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "startdate": start_date,
            "enddate": end_date,
            "datatypeid": "TMAX,TMIN,PRCP",  # Max temp, Min temp, Precipitation
            "units": "standard",
            "limit": 1000
        }
        response = requests.get(f"{BASE_URL}/data", headers=headers, params=params)
        if response.status_code == 200:
            return response.json().get("results", [])
        else:
            print(f"Error {response.status_code}: {response.text}")
            return []

    # Define year ranges (API max = 1 year per request)
    date_ranges = [
        ("2005-08-01", "2005-12-31"),
        ("2006-01-01", "2006-12-31"),
        ("2007-01-01", "2007-12-31"),
        # ... continue for each year ...
        ("2024-01-01", "2024-12-31"),
    ]

    all_records = []

    for start, end in date_ranges:
        print(f"Fetching {start} to {end}...")
        records = fetch_weather(STATION_ID, start, end)
        all_records.extend(records)
        time.sleep(0.3)  # Respect rate limit (5 req/sec)

    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.pivot_table(index="date", columns="datatype", values="value").reset_index()

    print(df.head())
    print(f"Total records: {len(df)}")

weather(40.581347061192425, -105.09176511543451)