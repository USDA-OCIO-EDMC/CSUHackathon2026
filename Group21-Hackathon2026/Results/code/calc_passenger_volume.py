from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "international_inbound_segments.csv"
OUTPUT_FILE = BASE_DIR / "climate_data" / "output" / "country_month_passenger_volume.csv"

THRESHOLD = 1e-4


def main():
    df = pd.read_csv(INPUT_FILE)

    # Keep only inbound to US
    df = df[df["DEST_COUNTRY_NAME"] == "United States"]

    # Clean passengers
    df["PASSENGERS"] = pd.to_numeric(df["PASSENGERS"], errors="coerce").fillna(0)

    # Aggregate to country-month
    agg = (
        df.groupby(["ORIGIN_COUNTRY_NAME", "MONTH_WADS"])
        .agg(
            total_passengers=("PASSENGERS", "sum")
        )
        .reset_index()
    )

    # Rename to match your other datasets
    agg = agg.rename(columns={
        "ORIGIN_COUNTRY_NAME": "country",
        "MONTH_WADS": "month"
    })

    # Normalize (0–1)
    max_val = agg["total_passengers"].max()
    agg["passenger_score"] = agg["total_passengers"] / max_val

        # Step 1: power transform to reduce skew
    agg["passenger_score"] = agg["passenger_score"] ** 0.5   # sqrt transform

    # Step 2: rescale to 0–1 again
    max_val = agg["passenger_score"].max()
    agg["passenger_score"] = agg["passenger_score"] / max_val

    # Step 3:  — shift mean to ~0.25
    target_mean = 0.25
    current_mean = agg["passenger_score"].mean()
    scale_factor = target_mean / current_mean
    agg["passenger_score"] = agg["passenger_score"] * scale_factor

    # cap at 1
    agg["passenger_score"] = agg["passenger_score"].clip(upper=1)


    #  Zero out tiny values
    agg.loc[agg["passenger_score"] < THRESHOLD, "passenger_score"] = 0


    # Sort nicely
    agg = agg.sort_values(["country", "month"])

    agg.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", OUTPUT_FILE)
    print()
    print(agg.head(20))
    print()
    print("Countries:", agg["country"].nunique())


if __name__ == "__main__":
    main()