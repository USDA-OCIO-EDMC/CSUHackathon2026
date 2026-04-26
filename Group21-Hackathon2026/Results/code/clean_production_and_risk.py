from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "climate_data" / "output" / "origin_country_risk_month_with_production.csv"
OUTPUT_FILE = BASE_DIR / "climate_data" / "output" / "origin_country_risk_month_FINAL.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    # Select only important columns
    keep_cols = [
        "iso_a3",   # keep for mapping
        "country",
        "month",
        "origin_country_risk_month_with_production"
    ]

    # Optional: include this if you want visibility
    if "fruit_production_score" in df.columns:
        keep_cols.append("fruit_production_score")

    df = df[keep_cols]

    # Rename final risk column (cleaner name)
    df = df.rename(columns={
        "origin_country_risk_month_with_production": "origin_risk"
    })

    # 📊 Distribution analysis
    print("\n=== ORIGIN RISK DISTRIBUTION ===")

    print("\nBasic stats:")
    print(df["origin_risk"].describe())

    # Count > 1
    above_one = (df["origin_risk"] > 1).sum()
    print(f"\nValues > 1: {above_one} / {len(df)} ({above_one / len(df):.2%})")

    # Percentiles
    print("\nPercentiles:")
    for p in [50, 75, 90, 95, 99]:
        val = df["origin_risk"].quantile(p / 100)
        print(f"{p}th percentile: {val:.4f}")

    # Bucket distribution
    bins = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1, float("inf")]
    labels = [
        "0–0.01 (very low)",
        "0.01–0.05 (low)",
        "0.05–0.1 (moderate)",
        "0.1–0.25 (elevated)",
        "0.25–0.5 (high)",
        "0.5–1 (very high)",
        ">1 (extreme)"
    ]

    df["risk_bucket"] = pd.cut(df["origin_risk"], bins=bins, labels=labels)

    print("\nBucket counts:")
    print(df["risk_bucket"].value_counts().sort_index())

    # Sort nicely
    df = df.sort_values(["country", "month"])

    df.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", OUTPUT_FILE)
    print(df.head(20))
    print("Columns:", df.columns.tolist())


if __name__ == "__main__":
    main()