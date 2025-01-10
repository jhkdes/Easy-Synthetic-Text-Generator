from language_model import LanguageModel
from util import select_prompt_config_file, load_json_file, configure_logger
import csv
import re
import os
import pandas as pd
import sys
import time
import logging
from tqdm import tqdm
import hashlib
import re

def cache_prompt_response(log_filename):
    """
    Reads a log file, processes prompt, model_name, and response pairs, and returns a mapping of
    MD5 hashes of prompts to their corresponding responses and model names.

    Args:
        log_filename (str): The path to the log file.

    Returns:
        dict: A mapping of MD5(prompt_str) to a dictionary with response and model_name.
    """
    hash_mapping = {}

    # Define regex patterns for prompts, model names, responses, and timestamps
    prompt_pattern = re.compile(r" - prompt: (.+)")
    model_name_pattern = re.compile(r" - model_name: (.+)")
    response_start_pattern = re.compile(r" - response: (.+)")
    timestamp_pattern = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

    current_prompt = None
    current_response = None
    current_model_name = "Unknown_model"
    in_response = False

    with open(log_filename, 'r', encoding='utf-8') as log_file:
        for line in log_file:
            # Check if the line matches a prompt
            prompt_match = prompt_pattern.search(line)
            if prompt_match:
                current_prompt = prompt_match.group(1)
                current_model_name = "Unknown_model"  # Reset model name for new prompt
                in_response = False
                current_response = None
                continue

            # Check if the line specifies a model_name
            model_name_match = model_name_pattern.search(line)
            if model_name_match:
                current_model_name = model_name_match.group(1)
                continue

            # Check if the line starts a response
            response_match = response_start_pattern.search(line)
            if response_match:
                current_response = response_match.group(1) + "\n"
                in_response = True
                continue

            # If in response, continue capturing lines until the next timestamp
            if in_response:
                if timestamp_pattern.match(line):
                    # Process the complete response
                    if current_prompt and current_response:
                        prompt_hash = hashlib.md5(current_prompt.encode('utf-8')).hexdigest()
                        hash_mapping[prompt_hash] = {
                            "response": current_response.strip(),
                            "model_name": current_model_name
                        }

                    # Reset for the next sequence
                    current_prompt = None
                    current_response = None
                    current_model_name = "Unknown_model"
                    in_response = False
                else:
                    current_response += line

    # Finalize any pending response at the end of the file
    if current_prompt and current_response:
        prompt_hash = hashlib.md5(current_prompt.encode('utf-8')).hexdigest()
        hash_mapping[prompt_hash] = {
            "response": current_response.strip(),
            "model_name": current_model_name
        }

    return hash_mapping

# Global variable to store the LanguageModel instance
_global_language_model = None

def get_language_model(model_name=None, api_key=None, system_prompt=None, 
                       temperature=None, login_email=None, login_passwd=None, 
                       huggingchat_model_index=None, progress_bar=None):
    global _global_language_model
    
    # If model doesn't exist, create it
    if _global_language_model is None:
        # Require all parameters for first-time initialization
        if all([model_name, api_key, system_prompt, temperature is not None, 
                login_email, login_passwd, huggingchat_model_index is not None]):
            _global_language_model = LanguageModel(
                model_name, api_key, system_prompt, temperature, 
                login_email, login_passwd, huggingchat_model_index, progress_bar
            )
        else:
            raise ValueError("Must provide all parameters for first-time model initialization")
    else:
        # change the language model
        _global_language_model.update_model(model_name, api_key, system_prompt, temperature,
                                            login_email, login_passwd, huggingchat_model_index, progress_bar)

    return _global_language_model

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

def process_with_lang_model(model, prompts_data, output_file, prompt_response_type, prompt_response_array_constraint=None, prompt_response_cache={}):
    """
    Process prompts using LangChain and GPT-4-mini, saving results to CSV
    Only processes prompts not already in the output file
    Writes each response immediately to ensure partial results are saved
    """
    # model_name = model.get_current_model_name()

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
    for label, category, prompt, prompt_index in tqdm(prompts_data, desc="Processing Prompts", unit="prompt"):
        # Skip if prompt_index already processed
        if prompt_index in processed_indices:
            logging.info(f"Skipping already processed prompt index: {prompt_index}")
            # print(f"Skipping already processed prompt index: {prompt_index}")
            continue

        try:
            # start a new conversation if requested            
            marker = "#### START NEW CONVERSATION ####"
            if marker in prompt:
                prompt = prompt.replace(marker, "").strip()
                model.create_new_conversation()

            # Get response
            response_dict = model.send_request(prompt, prompt_response_type, prompt_response_array_constraint, prompt_response_cache)
            response = response_dict.get("response", "Internal error: No response found")
            model_name = response_dict.get("model_name", "Internal error: No model_name found")

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
            
            logging.info(f"Processed prompt index ({prompt_index}) for Label: {label}, Category: {category}")
            
        except Exception as e:
            logging.error(f"Error processing prompt index ({prompt_index}) for Label: {label}, Category: {category}")
            logging.error(f"Error: {str(e)}")
            if 'too many messages' in str(e) and 'Try again later' in str(e):
                logging.info("Sleeping 5 mins due to rate limit detection")
                time.sleep(300)  # Sleep for 390 seconds, which is 5 minutes

def gen_data(model_name, api_key, temperature, login_email, login_passwd, 
             huggingchat_model_index, input_file, output_file, prompt_response_type = "string", prompt_response_array_constraint = None,
             log_file = "gen_data_log.txt", log_level_str = "INFO"):    
    # Create a tqdm object with a total of 100 iterations
    pbar = tqdm(total=100, desc="Setting up gen_data - building cache")

    # cache the previous prompt responses
    prompt_response_cache = {}
    if os.path.exists(log_file):
        prompt_response_cache = cache_prompt_response(log_file)

    # Configure logging
    configure_logger(log_file, log_level_str)
    pbar.update(10)
    pbar.set_description("Setting up gen_data - parsing input_file")

    try:
        # Parse input file
        system_prompt, prompts_data = parse_input_file(input_file)
        logging.debug(f"system_prompt = {system_prompt}")
        #print(f"system_prompt = {system_prompt}")

        pbar.update(20)
        pbar.set_description("Setting up gen_data - lang model setup")

        # Get or create the global LanguageModel instance
        model = get_language_model(
            model_name=model_name, 
            api_key=api_key, 
            system_prompt=system_prompt, 
            temperature=temperature, 
            login_email=login_email, 
            login_passwd=login_passwd, 
            huggingchat_model_index=huggingchat_model_index,
            progress_bar=pbar
        )
        # model = LanguageModel(model_name, api_key, system_prompt, temperature, login_email, login_passwd, huggingchat_model_index)

        pbar.update(30)
        pbar.set_description("Setting up gen_data - completed")

        pbar.close()

        # Process with LanguageModel
        process_with_lang_model(model, prompts_data, output_file, prompt_response_type, prompt_response_array_constraint, prompt_response_cache)
        
        logging.info(f"Processing complete. Results saved to {output_file}")
        # print(f"Processing complete. Results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        # print(f"An error occurred: {str(e)}")


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
    
    gen_data(MODEL, API_KEY, temperature, login_email, login_passwd, huggingchat_model_index, INPUT_FILE, OUTPUT_FILE)

if __name__ == "__main__":
    main()