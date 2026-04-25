import argparse
import json
import city_map_generator


def load_city_risk_data(input_path):
	with open(input_path, "r", encoding="utf-8") as infile:
		data = json.load(infile)

	if not isinstance(data, dict):
		raise ValueError("city_risk_data JSON must be an object keyed by city.")

	return data


def build_risk_table_template(city_risk_data):
	risk_table = {}

	for city, month_data in city_risk_data.items():
		if not isinstance(month_data, dict):
			continue

		risk_table[city] = {}
		for monthyear, metrics in month_data.items():
			species_template = {}

			risk_countries = {}
			if isinstance(metrics, dict):
				risk_countries = metrics.get("RiskCountries", {})

			if isinstance(risk_countries, dict):
				for species_by_country in risk_countries.values():
					if not isinstance(species_by_country, dict):
						continue
					for species in species_by_country.keys():
						if species not in species_template:
							species_template[species] = 0
						species_template[species] += 1

			risk_table[city][monthyear] = species_template

	return risk_table


def write_json(data, output_path):
	with open(output_path, "w", encoding="utf-8") as outfile:
		json.dump(data, outfile, indent=2)


def parse_args():
	parser = argparse.ArgumentParser(
		description="Build a City -> Month/Year -> Species risk table template from city_risk_data.json."
	)
	parser.add_argument(
		"--input",
		default="city_risk_data.json",
		help="Path to city risk data JSON file.",
	)
	parser.add_argument(
		"--output",
		default="city_risk_table.json",
		help="Path to output template JSON file.",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	city_risk_data = load_city_risk_data(args.input)
	risk_table = build_risk_table_template(city_risk_data)
	write_json(risk_table, args.output)
	print(f"Wrote risk table template for {len(risk_table)} cities to {args.output}")
	city_map_generator.main(risk_table)


if __name__ == "__main__":
	main()
