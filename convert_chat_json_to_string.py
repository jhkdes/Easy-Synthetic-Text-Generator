import csv
import json
import sys
import argparse

def convert_chat_column(input_file, output_file):
    """
    Reads an input CSV file, processes the "Chat" column by converting JSON chat objects
    into concatenated sender-message strings, and writes the output to a new CSV file.
    """
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        for row in reader:
            if 'Chat' in row and row['Chat']:
                try:
                    chat_data = json.loads(row['Chat'])
                    # Process the chat data if it contains the expected structure
                    if isinstance(chat_data, dict) and 'chat' in chat_data:
                        chat_entries = chat_data['chat']
                        row['Chat'] = "\n".join(f"{entry['sender']}: {entry['msg']}" for entry in chat_entries)
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Keep the Chat column unchanged if parsing fails
                    pass
            # Write the processed row to the output file
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reads an input CSV file, processes the 'Chat' column by converting JSON chat objects into concatenated sender-message strings, and writes the output to a new CSV file.")
    parser.add_argument("-i", "--input_filename", required=True, help="Path to the input CSV file with 'Chat' column as JSON objects.")
    parser.add_argument("-o", "--output_filename", required=True, help="Path to the output CSV file with 'Chat' column as sender-message strings.")

    args = parser.parse_args()

    input_filename = args.input_filename
    output_filename = args.output_filename

    try:
        convert_chat_column(input_filename, output_filename)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
