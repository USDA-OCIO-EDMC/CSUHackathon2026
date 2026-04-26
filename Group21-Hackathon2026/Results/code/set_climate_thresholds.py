from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent

SOM_FILE = BASE_DIR / "output" / "clean_target_fly_country_associations.csv"
CLIMATE_FILE = BASE_DIR / "output" / "country_monthly_min_max_risk_score.csv"

OUTPUT_FILE = BASE_DIR / "output" / "species_origin_risk.csv"

CLIMATE_YEAR = 2020


def survival_factor(avg_tmin_c):
    if pd.isna(avg_tmin_c):
        return 0

    if avg_tmin_c < 5:
        return 0.05
    elif avg_tmin_c < 10:
        return 0.2
    elif avg_tmin_c < 15:
        return 0.5
    elif avg_tmin_c < 25:
        return 1.0
    else:
        return 0.8


def main():
    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    som_df = pd.read_csv(SOM_FILE)
    climate_df = pd.read_csv(CLIMATE_FILE)

    COUNTRY_NAME_FIXES = {
        "Bosnia-Hercegovina": "Bosnia and Herzegovina",
        "Cape Verde": "Cabo Verde",
        "Congo": "Republic of the Congo",
        "Czech Republic": "Czechia",
        "EI Salvador": "El Salvador",
        "Georgia(Republic of)": "Georgia",
        "Janpan": "Japan",
        "Korea, Republic of": "South Korea",
        "USA": "United States of America",
        "United States": "United States of America",
        "Yugoslavia(former)": "Republic of Serbia",
        "Serbia": "Republic of Serbia",
        "Russian Federation": "Russia",
    }

    som_df["country"] = som_df["country"].replace(COUNTRY_NAME_FIXES)

    som_df["som_index_clean"] = pd.to_numeric(
        som_df["som_index_clean"],
        errors="coerce"
    )

    climate_df["avg_tmin_c"] = pd.to_numeric(
        climate_df["avg_tmin_c"],
        errors="coerce"
    )

    climate_df = climate_df[climate_df["year"] == CLIMATE_YEAR]

    climate_df = climate_df[[
        "country",
        "month",
        "avg_tmin_c"
    ]]

    df = som_df.merge(
        climate_df,
        on="country",
        how="left"
    )

    missing = df[df["avg_tmin_c"].isna()]["country"].unique()

    print("\nCountries missing climate:")
    for country in missing:
        print(country)
    print("Total missing:", len(missing))

    df["climate_factor"] = df["avg_tmin_c"].apply(survival_factor)

    df["species_origin_risk"] = (
        df["som_index_clean"] * df["climate_factor"]
    )

    df = df[[
        "country",
        "month",
        "common_name",
        "scientific_name",
        "som_index_clean",
        "avg_tmin_c",
        "climate_factor",
        "species_origin_risk"
    ]]

    df = df.sort_values([
        "country",
        "month",
        "species_origin_risk"
    ], ascending=[True, True, False])

    df.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", OUTPUT_FILE)
    print()
    print(df.head(30))
    print()
    print("Rows:", len(df))
    print("Countries:", df["country"].nunique())
    print("Species:", df["scientific_name"].nunique())
    print("Missing climate rows:", df["avg_tmin_c"].isna().sum())


if __name__ == "__main__":
    main()