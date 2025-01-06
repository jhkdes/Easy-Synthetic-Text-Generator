import csv
import json
from collections import defaultdict
import argparse
import sys

def read_and_combine_csv(input_filename, output_filename):
    # Dictionary to hold combined JSON objects
    combined_data = defaultdict(lambda: {
        "description": None,
        "seeds": []
    })

    try:
        # Read the input CSV file
        with open(input_filename, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                try:
                    # Parse the "Chat" column as JSON
                    json_obj = json.loads(row["Chat"])
                    
                    # Extract label and category to form a unique key
                    label = json_obj["label"]
                    category = json_obj["category"]
                    key = (label, category)

                    # Combine seeds and use the first description
                    if combined_data[key]["description"] is None:
                        combined_data[key]["description"] = json_obj["description"]
                    combined_data[key]["seeds"].extend(json_obj["seeds"])
                except (json.JSONDecodeError, KeyError):
                    # Skip rows where "Chat" is not a valid JSON or required keys are missing
                    continue

        # Prepare the output format
        output_data = {"categories": []}
        for (label, category), values in combined_data.items():
            output_data["categories"].append({
                "label": label,
                "category": category,
                "description": values["description"],
                "seeds": list(set(values["seeds"]))  # Remove duplicate seeds
            })

        # Write the combined data to the output JSON file
        with open(output_filename, mode='w', encoding='utf-8') as jsonfile:
            json.dump(output_data, jsonfile, indent=4)

        print(f"Combined JSON data has been written to {output_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage: python script.py input.csv output.json
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input CSV file by combining Chat columns to create prompt_categories.json with seeds in the output JSON file.")
    parser.add_argument("--input_filename", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_filename", required=True, help="Path to the output JSON file.")

    args = parser.parse_args()

    input_filename = args.input_filename
    output_filename = args.output_filename

    try:
        read_and_combine_csv(input_filename, output_filename)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
