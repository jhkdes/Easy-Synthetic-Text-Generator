import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from util import select_prompt_config_file, load_json_file, configure_logger
import numpy as np
import argparse
import os
import sys
import logging
import csv
import statistics
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_csv(file_path):
    # Initialize variables
    string_lengths = []
    chats = []

    # Read the CSV file
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            chat = row['Chat']
            if chat:  # Ensure the 'Chat' field is not empty
                chats.append(chat)
                string_lengths.append(len(chat))
    
    # Calculate statistics
    if string_lengths:
        avg_length = sum(string_lengths) / len(string_lengths)
        std_dev = statistics.stdev(string_lengths) if len(string_lengths) > 1 else 0
        median_length = statistics.median(string_lengths)
        min_length = min(string_lengths)
        max_length = max(string_lengths)

        logging.info(f"Average string length: {avg_length:.2f}")
        logging.info(f"Standard deviation: {std_dev:.2f}")
        logging.info(f"Median string length: {median_length}")
        logging.info(f"Minimum string length: {min_length}")
        logging.info(f"Maximum string length: {max_length}")

        # Find chats with length closest to the median
        chats_with_median_length = [
            chat for chat in chats 
            if abs(len(chat) - median_length) <= 1
        ]

        logging.info("\nChat(s) with length closest to median:")
        for chat in chats_with_median_length[:5]:  # Limiting to first 5 to prevent overwhelming output
            logging.info(f"- Length {len(chat)}: {chat}")

        # Find the top 5 shortest-length chats
        sorted_chats_by_length = sorted(chats, key=len)
        logging.info("\nTop 5 shortest-length chats:")
        for chat in sorted_chats_by_length[:5]:
            logging.info(f"- Length {len(chat)}: {chat}")

        # Find the top 3 longest-length chats
        sorted_chats_by_length_desc = sorted(chats, key=len, reverse=True)
        logging.info("\nTop 3 longest-length chats:")
        for chat in sorted_chats_by_length_desc[:3]:
            logging.info(f"- Length {len(chat)}: {chat}")

        # Plot the distribution
        # commenting it out so that it doesn't pause the execution with popup
        #
        # plt.figure(figsize=(10, 6))
        # plt.hist(string_lengths, bins=range(0, max_length + 10, 10), color='blue', alpha=0.7, edgecolor='black')
        # plt.title('Distribution of String Lengths in Chat Column')
        # plt.xlabel('String Length')
        # plt.ylabel('Frequency')
        # plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.show()
    else:
        logging.error("No data found in the 'Chat' column.")

def compute_embeddings_cache(df):
    # Load SBERT model
    logging.info("Loading SBERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Precompute embeddings and cache them
    logging.info("Precomputing embeddings for all chats...")
    embeddings_cache = {chat: model.encode(chat, show_progress_bar=False) for chat in df['Chat']}
    logging.info("Precomputed embeddings for %d chats.", len(embeddings_cache))

    return embeddings_cache

def select_distinct_chats(chat_list, embeddings_cache, similarity_threshold):
    # logging.info("Starting selection of distinct chats with similarity_threshold=%f", similarity_threshold)

    # Compute cosine similarity matrix using precomputed embeddings
    # logging.info("Calculating cosine similarity matrix...")
    embeddings = [embeddings_cache[chat] for chat in chat_list]
    cosine_sim_matrix = cosine_similarity(embeddings)

    # Calculate aggregate similarity for each chat
    aggregate_similarity = cosine_sim_matrix.sum(axis=1)
    # logging.info("Computed aggregate similarity for each chat.")

    # Sort chats by ascending aggregate similarity
    sorted_indices = np.argsort(aggregate_similarity)
    sorted_chats = [chat_list[i] for i in sorted_indices]
    # logging.info("Chats sorted by aggregate similarity.")

    # Select representative chats
    selected_chats = []
    selected_indices = []
    for idx, chat in enumerate(sorted_chats):
        chat_embedding = embeddings_cache[chat]
        if all(cosine_similarity([chat_embedding], [embeddings_cache[selected]]) < similarity_threshold for selected in selected_chats):
            selected_chats.append(chat)
            selected_indices.append(sorted_indices[idx])  # Use original index to retain all columns
    # logging.info("Selected %d representative chats out of %d total chats.", len(selected_chats), len(chat_list))

    return selected_indices

def filter_chat_by_category_group(df, embeddings_cache, similarity_threshold=0.7):
    # Filter chats by Category groups
    filtered_indices = []
    for category, group_df in df.groupby('Category'):
        logging.info("Processing category: %s with %d chats...", category, len(group_df))
        chat_list = group_df['Chat'].tolist()
        group_selected_indices = select_distinct_chats(chat_list, embeddings_cache, similarity_threshold)
        # Convert local indices to global indices
        global_indices = group_df.index[group_selected_indices]
        filtered_indices.extend(global_indices)
        logging.info("Processing category: %s complete. Selected %d chats.", category, len(group_selected_indices))

    # Filter the original DataFrame
    df_filtered = df.loc[filtered_indices].reset_index(drop=True)
    logging.info("Filtered DataFrame created with %d rows.", len(df_filtered))

    return df_filtered

def print_chat_summary_table(df, embeddings_cache):
    """
    Prints a table showing the number of chats filtered at different similarity thresholds for each category.

    Parameters:
    df (pd.DataFrame): DataFrame containing chats with a 'Category' column and 'Chat' column.
    embeddings_cache (dict): Precomputed embeddings for chats.
    """
    thresholds = [0.9, 0.8, 0.75, 0.7, 0.65, 0.6]
    summary_data = []

    for category, group_df in df.groupby('Category'):
        row = {
            "Category": category,
            "Total Chats": len(group_df)
        }
        for threshold in thresholds:
            filtered_indices = select_distinct_chats(group_df['Chat'].tolist(), embeddings_cache, threshold)
            row[f"Chats @ {threshold}"] = len(filtered_indices)
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    # Add a row for totals
    total_row = {"Category": "Total", "Total Chats": summary_df["Total Chats"].sum()}
    for threshold in thresholds:
        total_row[f"Chats @ {threshold}"] = summary_df[f"Chats @ {threshold}"].sum()
    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)

    # Add a row for percentages
    percentage_row = {"Category": "Percentage", "Total Chats": ""}
    for threshold in thresholds:
        percentage_row[f"Chats @ {threshold}"] = f"{(total_row[f'Chats @ {threshold}'] / total_row['Total Chats']) * 100:.2f}%" if total_row['Total Chats'] > 0 else "0.00%"
    summary_df = pd.concat([summary_df, pd.DataFrame([percentage_row])], ignore_index=True)

    logging.info(f"\n{summary_df.to_string(index=False)}")

def filter_data(input_file, similarity_threshold, output_file):
    # Create a tqdm object with a total of 100 iterations
    pbar = tqdm(total=100, desc="Filtering data - loading data")

    # Load the CSV file
    df = pd.read_csv(input_file)
    logging.info("Input file loaded with %d rows and columns: %s", len(df), df.columns.tolist())

    pbar.update(10)
    pbar.set_description("Filtering data - computing embedding")  

    # Ensure the required columns exist
    if 'Chat' not in df.columns or 'Category' not in df.columns:
        logging.error("Missing required columns 'Chat' and 'Category' in the input CSV file.")
        raise ValueError("The input CSV file must contain 'Chat' and 'Category' columns.")

    embeddings_cache = compute_embeddings_cache(df)

    pbar.update(40)
    pbar.set_description("Filtering data - filtering by group")  

    df_filtered = filter_chat_by_category_group(df, embeddings_cache, similarity_threshold)

    pbar.update(20)
    pbar.set_description("Filtering data - writing filtered data")  

    # Save the filtered DataFrame to a new CSV file
    df_filtered.to_csv(output_file, index=False)
    logging.info("Filtered DataFrame saved to: %s with %f threshold", output_file, similarity_threshold)

    pbar.update(10)
    pbar.set_description("Filtering data - logging data summary")  

    print_chat_summary_table(df, embeddings_cache)

    pbar.update(10)
    pbar.set_description("Filtering data - logging sample data")  

    analyze_csv(output_file)

    pbar.update(10)
    pbar.set_description("Filtering data - completed")

    pbar.close()

def main_old():
    # Get filename from command line argument if provided
    prompt_config_filename = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Select and print the selected file
    prompt_config_filename = select_prompt_config_file(prompt_config_filename)
    print(f"Selected prompt config file: {prompt_config_filename}")

    # read in prompt_config
    prompt_config_json = load_json_file(prompt_config_filename)
    working_path = prompt_config_json.get("working_path") or "."
    output_data_filename = prompt_config_json.get("output_data_filename") or "output_data.csv"
    output_data_filename_fullpath = os.path.join(working_path, output_data_filename)
    output_file = f"{output_data_filename_fullpath}.filtered.csv"

    filter_data(output_data_filename_fullpath, 0.7, output_file)

def main():
    """
    Main function to parse arguments and run the filtering process.
    """
    parser = argparse.ArgumentParser(description='Filter CSV based on cosine similarity of Chat column')
    
    parser.add_argument('-i', '--input_filename', 
                        required=True, 
                        help='Input CSV file path')
    parser.add_argument('-o', '--output_filename', 
                        help='Output CSV file path')
    parser.add_argument('-l', '--log_filename', 
                        help='Log file path')
    parser.add_argument('-v', '--log_level', 
                        help='Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    #parser.add_argument('-p', '--print_report', 
    #                    action='store_true', 
    #                    help='Print distribution report after filtering')
    parser.add_argument('-t', '--threshold', 
                        type=float, 
                        default=0.7, 
                        help='Maximum cosine similarity threshold (default: 0.7)')

    # Parse arguments
    args = parser.parse_args()

    # Set default output file if not specified
    output_file = args.output_filename or f"{args.input_filename}.filtered.csv"
    log_file = args.log_filename or f"{args.input_filename}.filter_data_log.txt"
    log_level = args.log_level or "INFO"

    # configure the logger
    configure_logger(log_file, log_level)

    # Run filtering
    filter_data(args.input_filename, args.threshold, output_file)

    #filter_by_cosine_similarity(
    #    input_file=args.input, 
    #    output_file=output_file, 
    #    threshold=args.threshold, 
    #    print_report=args.print_report
    #)

if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    main()