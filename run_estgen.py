from util import select_prompt_config_file, load_json_file
from gen_prompts import gen_prompts
from gen_data import gen_data
import sys
import os
import subprocess
import shlex
import subprocess
import hashlib
import time
from datetime import datetime
from datetime import timedelta
from typing import Dict, Any

def calculate_md5(filename: str) -> str:
    """Calculate MD5 hash of a file."""
    try:
        with open(filename, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        return "FILE_NOT_FOUND"

def format_log_entry(key: str, value: str) -> str:
    """Format log entry in a consistent, parseable format."""
    return f"[{key}]: {value}"

def convert_to_filepath(step: Dict[str, Any], path: str) -> None:
    for key, value in step.items():
        if key.endswith('_filename'):
            step[key] = os.path.join(path, value)

def log_step_info(step_num: int, total_steps: int, step: Dict[str, Any]) -> None:
    """Log step information before execution."""
    print(format_log_entry("STEP", f"{step_num}/{total_steps}"))
    print(format_log_entry("SCRIPT", step["script"]))
    print(format_log_entry("DESCRIPTION", step["description"]))
    
    # Calculate MD5 for input files
    for key, value in step.items():
        if key.endswith('_filename') and key != 'output_filename' and key != 'log_filename':
            md5_hash = calculate_md5(value)
            print(format_log_entry(f"MD5_{key.upper()}", md5_hash))

def execute_steps(config: Dict[str, Any]) -> None:
    """Execute a series of steps defined in a JSON configuration."""
    steps = config.get("steps", [])
    total_steps = len(steps)
    
    for step_num, step in enumerate(steps, 1):
        script_name = step["script"]
        convert_to_filepath(step, config.get("working_path") or ".")
        
        # Log step information
        log_step_info(step_num, total_steps, step)
        
        # Record start time
        start_time = time.time()
        start_datetime = datetime.now()
        print(format_log_entry("START_TIME", start_datetime.isoformat()))
        
        try:
            if script_name == "gen_prompts":
                # Import and call gen_prompts function
                invoke_gen_prompts(config, step)
            
            elif script_name == "gen_data":
                # Import and call gen_data function
                invoke_gen_data(config, step)
            
            else:
                # Execute as Python script with command line arguments
                invoke_script(script_name, step)
            
            # Calculate end time and duration
            end_time = time.time()
            end_datetime = datetime.now()
            duration = end_time - start_time
            duration_str = str(timedelta(seconds=int(duration)))
            
            # Log completion information
            print(format_log_entry("END_TIME", end_datetime.isoformat()))
            print(format_log_entry("DURATION", f"{duration_str} - ({duration:.2f} seconds)"))
            
            # Calculate MD5 of output file
            output_md5 = calculate_md5(step["output_filename"])
            print(format_log_entry("MD5_OUTPUT_FILENAME", output_md5))
            print(format_log_entry("STATUS", "SUCCESS"))
            
        except Exception as e:
            print(format_log_entry("ERROR", str(e)))
            print(format_log_entry("STATUS", "FAILED"))
            raise
        
        finally:
            print("-" * 80)  # Separator between steps

def process_prompts(json_data):
    # Extract prompts from JSON
    prompts = json_data.get("prompts", [])

    # Sort prompts by 'prompt_index' in increasing order
    sorted_prompts = sorted(prompts, key=lambda x: x.get("prompt_index", float('inf')))

    # Iterate and print the 'prompt_name' and 'prompt_index'
    for prompt in sorted_prompts:
        prompt_name = prompt.get("prompt_name", "Unknown")
        prompt_index = prompt.get("prompt_index", "Unknown")
        print(f"Prompt Index: {prompt_index}, Prompt Name: {prompt_name}")

        # Ask user whether to continue
        #user_input = input("Do you want to continue processing? (yes/NO): ").strip().lower()
        #if user_input not in ["yes", "y"]:
        #    print("Stopping the process.")
        #    break

        invoke_gen_prompts(json_data, prompt)
        invoke_gen_data(json_data, prompt)
        invoke_processor(json_data, prompt)

def check_version(version: str, required_major: int, required_minor: int, required_release: int):
    try:
        # Split the version string into components
        major, minor, release = map(int, version.split('.'))

        # Check if major and minor versions match
        if major != required_major or minor != required_minor:
            raise ValueError(f"Mismatched version: Expected {required_major}.{required_minor}.{required_release}, got {major}.{minor}.{release}.")

        # Check if release version is lower than required
        if release < required_release:
            raise ValueError(f"Release version too low: Required at least {required_major}.{required_minor}.{required_release}, got {major}.{minor}.{release}.")

        print("Version check passed.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception:
        print("Invalid version format. Ensure it is in '<major_version>.<minor_version>.<release_version>' format.")

def invoke_gen_prompts(prompt_config_json, prompt_config_json_local):
    working_path = prompt_config_json.get("working_path") or "."
    prompt_template = prompt_config_json_local["prompt_template"]  # required
    system_prompt = prompt_config_json_local["system_prompt"]  # required
    prompt_repeat = int(prompt_config_json_local.get("prompt_repeat", "1"))
    #prompt_categories_filename = prompt_config_json_local["prompt_categories_filename"] # required
    #prompt_categories_filename_fullpath = os.path.join(working_path, prompt_categories_filename)
    prompt_expansions = prompt_config_json_local.get("prompt_expansions")  # prompt_expansions is optional, may be null
    #output_filename = prompt_config_json_local.get("output_filename") or "working_prompt_list.txt"
    #output_filename_fullpath = os.path.join(working_path, output_filename)

    prompt_categories_filename_fullpath = prompt_config_json_local["prompt_categories_filename"]
    output_filename_fullpath = prompt_config_json_local["output_filename"] or os.path.join(working_path, "working_prompt_list.txt")
    
    # Check if the file exists and is already populated
    #if os.path.exists(output_filename_fullpath) and os.path.getsize(output_filename_fullpath) > 0:
    #    overwrite = input(f"gen_prompts: Overwrite the file ('{output_filename_fullpath}')? (yes/no): ").strip().lower()
    #    if overwrite not in ["yes", "y"]:
    #        print("Aborting operation as per user request.")
    #        return
        
    gen_prompts(prompt_expansions, prompt_template, system_prompt, prompt_categories_filename_fullpath, output_filename_fullpath, prompt_repeat)

def get_value_from_prompt_config_json(key, prompt_config_json, prompt_config_json_local):
    value = prompt_config_json_local.get(key, prompt_config_json.get(key))
    if value is None:
        raise ValueError(f"Error: '{key}' is undefined in prompt_config.json")
    return value

def invoke_gen_data(prompt_config_json, prompt_config_json_local):
    working_path = prompt_config_json.get("working_path") or "."
    #working_prompt_list_filename = prompt_config_json_local.get("working_prompt_list_filename") or "working_prompt_list.txt"
    #working_prompt_list_filename_fullpath = os.path.join(working_path, working_prompt_list_filename)
    #output_filename = prompt_config_json_local.get("output_filename") or "output_data.csv"
    #output_filename_fullpath = os.path.join(working_path, output_filename)
    prompt_response_type = prompt_config_json_local.get("prompt_response_type") or "string"
    prompt_response_array_constraint = prompt_config_json_local.get("prompt_response_array_constraint") or None 
    #log_filename = prompt_config_json_local.get("log_filename") or "gen_data_log.txt"
    #log_filename_fullpath = os.path.join(working_path, log_filename)
    log_level = prompt_config_json_local.get("log_level") or "INFO"

    working_prompt_list_filename_fullpath = prompt_config_json_local.get("working_prompt_list_filename") or os.path.join(working_path, "working_prompt_list.txt")
    output_filename_fullpath = prompt_config_json_local.get("output_filename") or os.path.join(working_path, "output_data.csv")
    log_filename_fullpath = prompt_config_json_local.get("log_filename") or os.path.join(working_path, "gen_data_log.txt")

    print(format_log_entry("LOGGING_FILENAME", log_filename_fullpath))

    model_name = get_value_from_prompt_config_json("model", prompt_config_json, prompt_config_json_local)
    api_key = get_value_from_prompt_config_json("api_key", prompt_config_json, prompt_config_json_local)
    login_email = get_value_from_prompt_config_json("login_email", prompt_config_json, prompt_config_json_local)
    login_passwd = get_value_from_prompt_config_json("login_passwd", prompt_config_json, prompt_config_json_local)

    temperature = int(get_value_from_prompt_config_json("temperature", prompt_config_json, prompt_config_json_local))
    huggingchat_model_index = int(get_value_from_prompt_config_json("huggingchat_model_index", prompt_config_json, prompt_config_json_local))

    gen_data(model_name, api_key, temperature, login_email, login_passwd, huggingchat_model_index, 
             working_prompt_list_filename_fullpath, output_filename_fullpath, prompt_response_type, prompt_response_array_constraint,
             log_filename_fullpath, log_level)
    
def invoke_script(script_name: str, step: dict):
    """
    Execute a Python script with command line arguments derived from the `step` dictionary.

    :param script_name: Name of the script to be executed.
    :param step: Dictionary containing arguments for the script.
    :param path: Base path to prepend to arguments ending with '_filename'.
    """
    # Construct the command
    cmd = ["python", script_name]
    for key, value in step.items():
        if key != "script" and key != "description":
            # Prepend path if the key ends with '_filename'
            #if key.endswith("_filename"):
            #    value = os.path.join(path, value)
            cmd.extend([f"--{key}", str(value)])
    
    # Execute the script
    subprocess.run(cmd, check=True)

def replace_token(json_data, json_data_global, token, working_path):
    # Remove < and > from the token
    clean_token = token.strip('<>')
    
    # Check if the token exists in the JSON data
    if clean_token not in json_data:
        if clean_token not in json_data_global:
            raise ValueError(f"Token '{clean_token}' not found in JSON data")
        
        # Special handling for filename tokens
        if clean_token.endswith('filename'):
            # Join working path with the filename
            return str(os.path.join(working_path, json_data_global[clean_token]))

        return str(json_data_global[clean_token])
    
    # Convert the value to string to ensure it can be used in command
    if clean_token.endswith('filename'):
        return str(os.path.join(working_path, json_data[clean_token]))

    return str(json_data[clean_token])
    
def invoke_processor(prompt_config_json, prompt_config_json_local):
    working_path = prompt_config_json.get("working_path") or "."
    processor = prompt_config_json_local["processor"] # required

    # Split the script command into parts
    parts = shlex.split(processor)
    
    # Replace tokens with actual values
    processed_parts = [
        replace_token(prompt_config_json_local, prompt_config_json, part, working_path) if part.startswith('<') and part.endswith('>') else part
        for part in parts
    ]
    
    try:
        # Execute the script
        result = subprocess.run(['python'] + processed_parts, capture_output=True, text=True)
        
        # Return a tuple with return code, stdout, and stderr
        print(f"Processed: {processed_parts}")
        print(f"  {result.stdout.strip()}")
        print(f"  {result.returncode}")
        print(f"  {result.stderr.strip()}") 
    
    except Exception as e:
        print(f"Error: {e}")
        raise e

def main():
    # Get filename from command line argument if provided
    prompt_config_filename = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Select and print the selected file
    prompt_config_filename = select_prompt_config_file(prompt_config_filename)
    print(f"Selected prompt config file: {prompt_config_filename}")

    prompt_config_json = load_json_file(prompt_config_filename)
    check_version(prompt_config_json.get("version", "missing version in prompt_config.json - min version required: 0.1.0"), 0, 1, 0)
              
    # Call the function with the example JSON
    #process_prompts(prompt_config_json)
    execute_steps(prompt_config_json)

if __name__ == "__main__":
    main()