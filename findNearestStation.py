import requests

def findStation(lat, lon):
    TOKEN = "BUQUVsWSPNRvsxrOKGHYVIqxrNhcGPSv"
    BASE_URL = "https://ncdc.noaa.gov/cdo-web/api/v2"
    headers = {"token": TOKEN}

    # Find the nearest station that has data for your date range
    station_params = {
        "datasetid": "GHCND",
        "startdate": "2005-08-01",
        "enddate": "2024-12-31",
        "extent": f"{lat - 0.5},{lon - 0.5},{lat + 0.5},{lon + 0.5}",  # Bounding box
        "limit": 10
    }

    station_response = requests.get(f"{BASE_URL}/stations", headers=headers, params=station_params)
    stations = station_response.json()

    for s in stations.get("results", []):
        print(f"ID: {s['id']}, Name: {s['name']}, Lat: {s['latitude']}, Lon: {s['longitude']}")