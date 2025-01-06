import csv
import json
import sys
import argparse

def process_chat_csv(input_file, output_file):
    """
    Reads an input CSV file, processes the "Chat" column by converting JSON chat objects
    into individually separate messages, and writes the output to a new CSV file.

    For example, 
    gpt-4o-mini,Depression,Exhaustion,'{"chat": [
        {"sender": "A", "msg": "I'm tired" },
        {"sender": "A", "msg": "Exhausting." },
        {"sender": "A", "msg": "Need to sleep..." }
    ]}',1

    Would be converted to

    gpt-4o-mini,Depression,Exhaustion,"I'm tired",1
    gpt-4o-mini,Depression,Exhaustion,"Exhausting.",1
    gpt-4o-mini,Depression,Exhaustion,"Need to sleep...",1
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        # Ensure we maintain the same headers
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            try:
                # Parse the Chat column as JSON
                chat_data = json.loads(row['Chat'])
                for entry in chat_data.get('chat', []):
                    new_row = row.copy()
                    new_row['Chat'] = entry['msg']  # Replace Chat with individual messages
                    writer.writerow(new_row)
            except json.JSONDecodeError:
                # Echo the original line to the output file if JSON parsing fails
                writer.writerow(row)
            except KeyError as e:
                print(f"Skipping row due to missing key: {e}")
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reads an input CSV file, processes the 'Chat' column by converting JSON chat objects into individual message strings, and writes the output to a new CSV file.")
    parser.add_argument("-i", "--input_filename", required=True, help="Path to the input CSV file with 'Chat' column as JSON objects.")
    parser.add_argument("-o", "--output_filename", required=True, help="Path to the output CSV file with 'Chat' column as sender-message strings.")

    args = parser.parse_args()

    input_filename = args.input_filename
    output_filename = args.output_filename

    try:
        process_chat_csv(input_filename, output_filename)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
