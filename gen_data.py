from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from language_model import LanguageModel
from util import select_prompt_config_file, load_json_file
import csv
import re
import json
import os
import pandas as pd
import sys
import time

def parse_input_file(file_path):
    """
    Parse the input file to extract system prompt, labels, categories, and prompts.
    Returns a tuple of (system_prompt, list of (label, category, prompt, prompt_index) tuples)
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Extract system prompt - updated pattern to match the exact format
    system_match = re.search(r'System:\s*\n(.*?)(\n\s*)+\nLabel:', content, re.DOTALL)
    if not system_match:
        raise ValueError("Could not find system prompt in the expected format")
    system_prompt = system_match.group(1).strip()

    # Extract label, category, and prompts
    prompts_data = []
    
    # Find all sections starting with "Label:"
    sections = re.finditer(
        r'Label: (.*?)\nCategory: (.*?)\nPrompt (\d+):\n(.*?)(?=(?:\nLabel:|$))',
        content,
        re.DOTALL
    )

    for section in sections:
        label = section.group(1).strip()
        category = section.group(2).strip()
        prompt_index = section.group(3).strip()
        prompt = section.group(4).strip()
        prompts_data.append((label, category, prompt, prompt_index))

    return system_prompt, prompts_data

def process_with_lang_model(model, prompts_data, input_file, output_file):
    """
    Process prompts using LangChain and GPT-4-mini, saving results to CSV
    Only processes prompts not already in the output file
    Writes each response immediately to ensure partial results are saved
    """
    model_name = model.get_current_model_name()

    # Read existing output file if it exists
    try:
        existing_output = pd.read_csv(output_file)
        processed_indices = set(existing_output['PromptIndex'].astype(str))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        processed_indices = set()
        # Create the file with headers if it doesn't exist
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Model', 'Label', 'Category', 'Chat', 'PromptIndex'])

    # Process each prompt
    for label, category, prompt, prompt_index in prompts_data:
        # Skip if prompt_index already processed
        if prompt_index in processed_indices:
            print(f"Skipping already processed prompt index: {prompt_index}")
            continue

        try:
            # Get response
            response = model.send_request(prompt)
            
            # Immediately write to CSV to preserve partial results
            with open(output_file, 'a', newline='', encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    model_name,   
                    label,
                    category,
                    response,
                    prompt_index
                ])
                # Flush to ensure immediate write to disk
                csvfile.flush()
                os.fsync(csvfile.fileno())
            
            print(f"Processed prompt index ({prompt_index}) for Label: {label}, Category: {category}")
            
        except Exception as e:
            print(f"Error processing prompt index ({prompt_index}) for Label: {label}, Category: {category}")
            print(f"Error: {str(e)}")
            if 'too many messages' in str(e) and 'Try again later' in str(e):
                print("Sleeping 5 mins due to rate limit detection")
                time.sleep(300)  # Sleep for 390 seconds, which is 5 minutes

def main():
    # Get filename from command line argument if provided
    prompt_config_filename = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Select and print the selected file
    prompt_config_filename = select_prompt_config_file(prompt_config_filename)
    print(f"Selected prompt config file: {prompt_config_filename}")

    # read in prompt_config
    prompt_config_json = load_json_file(prompt_config_filename)
    working_path = prompt_config_json.get("working_path") or "."
    working_prompt_list_filename = prompt_config_json.get("working_prompt_list_filename") or "working_prompt_list.txt"
    working_prompt_list_filename_fullpath = os.path.join(working_path, working_prompt_list_filename)
    output_data_filename = prompt_config_json.get("output_data_filename") or "output_data.csv"
    output_data_filename_fullpath = os.path.join(working_path, output_data_filename)

    model_name = prompt_config_json.get("model")
    if model_name is None:
        raise ValueError(f"Error: 'model' is undefined in prompt_config_json: {prompt_config_filename}")

    api_key = prompt_config_json.get("api_key")
    if api_key is None:
        raise ValueError(f"Error: 'api_key' is undefined in prompt_config_json: {prompt_config_filename}")

    login_email = prompt_config_json.get("login_email")
    if login_email is None:
        raise ValueError(f"Error: 'login_email' is undefined in prompt_config_json: {prompt_config_filename}")

    login_passwd = prompt_config_json.get("login_passwd")
    if login_passwd is None:
        raise ValueError(f"Error: 'login_passwd' is undefined in prompt_config_json: {prompt_config_filename}")

    temperature = int(prompt_config_json.get("temperature", 1))
    huggingchat_model_index = int(prompt_config_json.get("huggingchat_model_index", 1)) # 1 is 'meta-llama/Meta-Llama-3.1-70B-Instruct' on HuggingChat

    # Configuration
    INPUT_FILE = working_prompt_list_filename_fullpath
    OUTPUT_FILE = output_data_filename_fullpath
    MODEL = model_name
    API_KEY = api_key
    
    try:
        # Parse input file
        system_prompt, prompts_data = parse_input_file(INPUT_FILE)
        print(f"system_prompt = {system_prompt}")

        model = LanguageModel(MODEL, API_KEY, system_prompt, temperature, login_email, login_passwd, huggingchat_model_index)

        # Process with LanguageModel
        process_with_lang_model(model, prompts_data, INPUT_FILE, OUTPUT_FILE)
        
        print(f"Processing complete. Results saved to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()