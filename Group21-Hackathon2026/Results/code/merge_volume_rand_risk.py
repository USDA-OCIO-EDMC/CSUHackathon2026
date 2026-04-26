from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent

RISK_FILE = BASE_DIR / "timeline" / "origin_country_risk_month_FINAL.csv"
PASSENGER_FILE = BASE_DIR / "climate_data" / "output" / "country_month_passenger_volume.csv"

OUTPUT_FILE = BASE_DIR / "climate_data" / "output" / "country_month_route_risk.csv"


def main():
    risk_df = pd.read_csv(RISK_FILE)
    passenger_df = pd.read_csv(PASSENGER_FILE)

    risk_df["country"] = risk_df["country"].str.strip()
    passenger_df["country"] = passenger_df["country"].str.strip()

    risk_df["month"] = pd.to_numeric(risk_df["month"], errors="coerce").astype("Int64")
    passenger_df["month"] = pd.to_numeric(passenger_df["month"], errors="coerce").astype("Int64")

    merged = risk_df.merge(
        passenger_df,
        on=["country", "month"],
        how="left"
    )

    merged["total_passengers"] = merged["total_passengers"].fillna(0)
    merged["passenger_score"] = merged["passenger_score"].fillna(0)

    merged["route_risk"] = (
        merged["origin_risk"] * merged["passenger_score"]
    )

    merged = merged.sort_values(
        ["route_risk"],
        ascending=False
    )

    merged.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", OUTPUT_FILE)
    print()
    print(merged.head(30))
    print()
    print("Rows:", len(merged))
    print("Countries:", merged["country"].nunique())


if __name__ == "__main__":
    main()