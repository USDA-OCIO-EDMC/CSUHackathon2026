from pathlib import Path
import pandas as pd
import numpy as np
import unicodedata

BASE_DIR = Path(__file__).parent

RISK_FILE = BASE_DIR / "climate_data" / "output" / "origin_country_risk_month_with_iso_min_temp.csv"
FRUIT_FILE = BASE_DIR / "fruit_production_by_country.csv"

OUTPUT_FILE = BASE_DIR / "climate_data" / "output" / "origin_country_risk_month_with_production.csv"

PRODUCTION_WEIGHT = 0.25


def normalize_country(text):
    if pd.isna(text):
        return text
    text = str(text).strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    return text.lower()


def main():
    risk_df = pd.read_csv(RISK_FILE)
    fruit_df = pd.read_csv(FRUIT_FILE)

    # Rename columns
    fruit_df = fruit_df.rename(columns={
        "Country/Region": "country",
        "Fruit production\n(tonnes)": "fruit_production_tonnes",
        "Fruit production (tonnes)": "fruit_production_tonnes",
        '"Fruit production\n(tonnes)"': "fruit_production_tonnes",
    })

    # Clean production numbers
    fruit_df["fruit_production_tonnes"] = (
        fruit_df["fruit_production_tonnes"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    fruit_df["fruit_production_tonnes"] = pd.to_numeric(
        fruit_df["fruit_production_tonnes"],
        errors="coerce"
    )

    # Keep only needed cols
    fruit_df = fruit_df[["country", "fruit_production_tonnes"]]
    fruit_df = fruit_df.drop_duplicates(subset=["country"])

    # 🔧 Normalize names
    risk_df["country_clean"] = risk_df["country"].apply(normalize_country)
    fruit_df["country_clean"] = fruit_df["country"].apply(normalize_country)

    # 🔁 Manual fixes (add as needed)
    COUNTRY_FIXES = {
        "united states of america": "united states",
        "russian federation": "russia",
        "korea, republic of": "south korea",
        "iran (islamic republic of)": "iran",
        "viet nam": "vietnam",
        "turkiye": "turkey",
    }

    risk_df["country_clean"] = risk_df["country_clean"].replace(COUNTRY_FIXES)
    fruit_df["country_clean"] = fruit_df["country_clean"].replace(COUNTRY_FIXES)

    # 🔍 DEBUG BEFORE MERGE
    risk_countries = set(risk_df["country_clean"].unique())
    fruit_countries = set(fruit_df["country_clean"].unique())

    missing_in_fruit = sorted(risk_countries - fruit_countries)

    print("\nCountries missing from fruit dataset:")
    for c in missing_in_fruit[:30]:
        print(c)
    print("Total missing:", len(missing_in_fruit))

    # Merge
    merged = risk_df.merge(
        fruit_df[["country_clean", "fruit_production_tonnes"]],
        on="country_clean",
        how="left"
    )

    # Fill missing with mean
    mean_production = merged["fruit_production_tonnes"].mean()
    merged["fruit_production_tonnes"] = merged["fruit_production_tonnes"].fillna(mean_production)

    # Log transform
    merged["fruit_production_log"] = np.log1p(merged["fruit_production_tonnes"])

    # Normalize 0–1
    min_val = merged["fruit_production_log"].min()
    max_val = merged["fruit_production_log"].max()

    merged["fruit_production_score"] = (
        (merged["fruit_production_log"] - min_val) /
        (max_val - min_val)
    )

    # Apply boost
    merged["production_boost"] = (
        1 + PRODUCTION_WEIGHT * merged["fruit_production_score"]
    )

    merged["origin_country_risk_month_with_production"] = (
        merged["origin_country_risk_month"] * merged["production_boost"]
    )

    # Clean up
    merged = merged.sort_values(["country", "month"])

    merged.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved:", OUTPUT_FILE)
    print(merged.head(20))

    print("\nStats:")
    print("Total rows:", len(merged))
    print("Countries:", merged["country"].nunique())
    print("Mean production used:", mean_production)


if __name__ == "__main__":
    main()