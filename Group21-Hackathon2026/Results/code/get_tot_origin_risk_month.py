from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "output" / "species_origin_risk.csv"
OUTPUT_FILE = BASE_DIR / "output" / "origin_country_risk_month.csv"

HIGH_RISK_THRESHOLD = 0.4
MODERATE_RISK_THRESHOLD = 0.2
MODERATE_SPECIES_BOOST = 0.10


def top_k_mean(values, k=3):
    return values.nlargest(k).mean()


def main():
    df = pd.read_csv(INPUT_FILE)

    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df["species_origin_risk"] = pd.to_numeric(
        df["species_origin_risk"],
        errors="coerce"
    )

    df = df.dropna(subset=["country", "month", "species_origin_risk"])

    agg = (
        df.groupby(["country", "month"])
        .agg(
            max_risk=("species_origin_risk", "max"),
            mean_risk=("species_origin_risk", "mean"),
            total_risk=("species_origin_risk", "sum"),
            top3_mean=("species_origin_risk", lambda x: top_k_mean(x, 3)),
            high_species_count=(
                "species_origin_risk",
                lambda x: (x >= HIGH_RISK_THRESHOLD).sum()
            ),
            moderate_species_count=(
                "species_origin_risk",
                lambda x: (x >= MODERATE_RISK_THRESHOLD).sum()
            ),
        )
        .reset_index()
    )

    # Base risk captures both worst-case species and top-species pressure
    agg["base_origin_risk"] = (
        0.6 * agg["max_risk"] +
        0.4 * agg["top3_mean"]
    )

    # Boost countries/months with multiple moderate-or-higher risk species
    agg["multi_species_boost"] = (
        1 + MODERATE_SPECIES_BOOST * agg["moderate_species_count"]
    )

    agg["origin_country_risk_month"] = (
        agg["base_origin_risk"] * agg["multi_species_boost"]
    )

    agg = agg.sort_values(
        ["country", "month"],
        ascending=[True, True]
    )

    agg.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", OUTPUT_FILE)
    print()
    print(agg.head(30))
    print()
    print("Rows:", len(agg))
    print("Countries:", agg["country"].nunique())


if __name__ == "__main__":
    main()