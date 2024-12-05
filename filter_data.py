import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from util import select_prompt_config_file, load_json_file
import numpy as np
import json
import os
import sys
import logging
import csv
import statistics
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

        print(f"Average string length: {avg_length:.2f}")
        print(f"Standard deviation: {std_dev:.2f}")
        print(f"Median string length: {median_length}")
        print(f"Minimum string length: {min_length}")
        print(f"Maximum string length: {max_length}")

        # Find chats with length closest to the median
        chats_with_median_length = [
            chat for chat in chats 
            if abs(len(chat) - median_length) <= 1
        ]

        print("\nChat(s) with length closest to median:")
        for chat in chats_with_median_length[:5]:  # Limiting to first 5 to prevent overwhelming output
            print(f"- Length {len(chat)}: {chat}")

        # Find the top 5 shortest-length chats
        sorted_chats_by_length = sorted(chats, key=len)
        print("\nTop 5 shortest-length chats:")
        for chat in sorted_chats_by_length[:5]:
            print(f"- Length {len(chat)}: {chat}")

        # Find the top 3 longest-length chats
        sorted_chats_by_length_desc = sorted(chats, key=len, reverse=True)
        print("\nTop 3 longest-length chats:")
        for chat in sorted_chats_by_length_desc[:3]:
            print(f"- Length {len(chat)}: {chat}")

        # Plot the distribution
        plt.figure(figsize=(10, 6))
        plt.hist(string_lengths, bins=range(0, max_length + 10, 10), color='blue', alpha=0.7, edgecolor='black')
        plt.title('Distribution of String Lengths in Chat Column')
        plt.xlabel('String Length')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    else:
        print("No data found in the 'Chat' column.")

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
    logging.info("Starting selection of distinct chats with similarity_threshold=%f", similarity_threshold)

    # Compute cosine similarity matrix using precomputed embeddings
    logging.info("Calculating cosine similarity matrix...")
    embeddings = [embeddings_cache[chat] for chat in chat_list]
    cosine_sim_matrix = cosine_similarity(embeddings)

    # Calculate aggregate similarity for each chat
    aggregate_similarity = cosine_sim_matrix.sum(axis=1)
    logging.info("Computed aggregate similarity for each chat.")

    # Sort chats by ascending aggregate similarity
    sorted_indices = np.argsort(aggregate_similarity)
    sorted_chats = [chat_list[i] for i in sorted_indices]
    logging.info("Chats sorted by aggregate similarity.")

    # Select representative chats
    selected_chats = []
    selected_indices = []
    for idx, chat in enumerate(sorted_chats):
        chat_embedding = embeddings_cache[chat]
        if all(cosine_similarity([chat_embedding], [embeddings_cache[selected]]) < similarity_threshold for selected in selected_chats):
            selected_chats.append(chat)
            selected_indices.append(sorted_indices[idx])  # Use original index to retain all columns
    logging.info("Selected %d representative chats out of %d total chats.", len(selected_chats), len(chat_list))

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
        logging.info("Category %s processed. Selected %d chats.", category, len(group_selected_indices))

    # Filter the original DataFrame
    df_filtered = df.loc[filtered_indices].reset_index(drop=True)
    logging.info("Filtered DataFrame created with %d rows.", len(df_filtered))

    return df_filtered

def main():
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

    # Load the CSV file
    df = pd.read_csv(output_data_filename_fullpath)
    logging.info("Input file loaded with %d rows and columns: %s", len(df), df.columns.tolist())

    # Ensure the required columns exist
    if 'Chat' not in df.columns or 'Category' not in df.columns:
        logging.error("Missing required columns 'Chat' and 'Category' in the input CSV file.")
        raise ValueError("The input CSV file must contain 'Chat' and 'Category' columns.")

    embeddings_cache = compute_embeddings_cache(df)
    df_filtered = filter_chat_by_category_group(df, embeddings_cache, 0.7)

    # Save the filtered DataFrame to a new CSV file
    df_filtered.to_csv(output_file, index=False)
    logging.info("Filtered DataFrame saved to: %s", output_file)

    analyze_csv(output_file)

if __name__ == "__main__":
    main()