from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "fly_country_associations.xlsx"
OUTPUT_FILE = BASE_DIR / "output" / "target_fly_country_associations.csv"

COMMON_TO_SCIENTIFIC = {
    "Mediterranean Fruit Fly": "Ceratitis capitata",
    "Oriental Fruit Fly": "Bactrocera dorsalis",
    "Mexican Fruit Fly": "Anastrepha ludens",
    "Peach Fruit Fly": "Bactrocera zonata",
    "Guava Fruit Fly": "Bactrocera correcta",
    "Caribbean Fruit Fly": "Anastrepha suspensa",
    "Melon Fruit Fly": "Bactrocera cucurbitae",
    "Sapote Fruit Fly": "Anastrepha serpentina",
    "Zeugodacus Tau": "Bactrocera tau",
    "Queensland Fruit Fly": "Bactrocera tryoni",
}

SCIENTIFIC_TO_COMMON = {v: k for k, v in COMMON_TO_SCIENTIFIC.items()}
TARGET_SPECIES = set(COMMON_TO_SCIENTIFIC.values())


def clean_name(value):
    if pd.isna(value):
        return None
    return str(value).strip()


def normalize(value):
    cleaned = clean_name(value)
    if cleaned is None:
        return None
    return cleaned.lower().strip()


def main():
    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    raw = pd.read_excel(INPUT_FILE, sheet_name="Sheet2", header=None, skiprows=1)

    rows = []

    country_row = raw.iloc[0]
    subheader_row = raw.iloc[1]

    print("Detected subheader row:")
    print(subheader_row.tolist())
    print()

    for col in range(raw.shape[1] - 1):
        country = clean_name(country_row.iloc[col])

        subheader_1 = normalize(subheader_row.iloc[col])
        subheader_2 = normalize(subheader_row.iloc[col + 1])

        if subheader_1 == "som index" and subheader_2 == "scientific name":
            if country is None:
                print(f"Skipped column {col}: no country name found")
                continue

            print(f"Found country block: {country}")

            for r in range(2, raw.shape[0]):
                som_index = raw.iloc[r, col]
                scientific_name = clean_name(raw.iloc[r, col + 1])

                if scientific_name is None:
                    continue

                scientific_name = scientific_name.strip()

                if scientific_name in TARGET_SPECIES:
                    rows.append(
                        {
                            "country": country,
                            "common_name": SCIENTIFIC_TO_COMMON[scientific_name],
                            "scientific_name": scientific_name,
                            "som_index": som_index,
                        }
                    )

    df = pd.DataFrame(rows)

    if df.empty:
        print("No matching species found.")
        print("This usually means the Excel structure or species names do not match.")
        return

    df["som_index"] = pd.to_numeric(df["som_index"], errors="coerce")

    df = df.dropna(subset=["country", "scientific_name", "som_index"])

    df = df.sort_values(["country", "common_name"])

    df.to_csv(OUTPUT_FILE, index=False)

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print()
    print(df.head(30))
    print()
    print("Rows found:", len(df))
    print("Countries found:", df["country"].nunique())
    print("Species found:", df["scientific_name"].nunique())


if __name__ == "__main__":
    main()