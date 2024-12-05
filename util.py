import os
import sys
import json

def load_json_file(prompt_categories_json):
    with open(prompt_categories_json, 'r', encoding='utf-8') as f:
        prompt_categories = json.load(f)
    return prompt_categories

def select_prompt_config_file(initial_filename=None):
    """
    Select a prompt configuration JSON file.
    
    Args:
        initial_filename (str, optional): Filename provided as an argument.
    
    Returns:
        str: Path to the selected prompt config file.
    """
    # If filename is provided as an argument, use it directly
    if initial_filename:
        if os.path.exists(initial_filename):
            print(f"Using provided file: {initial_filename}")
            return initial_filename
        else:
            print(f"Error: File {initial_filename} does not exist.")
            sys.exit(1)
    
    # Search for prompt_config*.json files recursively
    prompt_configs = []
    for root, dirs, files in os.walk('.'):
        prompt_configs.extend([
            os.path.join(root, file) 
            for file in files 
            if file.startswith('prompt_config') and file.endswith('.json')
        ])
    
    # If no files found
    if not prompt_configs:
        print("No prompt_config*.json files found in the current directory or subdirectories.")
        sys.exit(1)
    
    # Display files with numbering
    print("Available prompt config files:")
    for i, file in enumerate(prompt_configs, 1):
        print(f"{i}. {file}")
    
    # User selection
    while True:
        try:
            selection = input("Enter the number of the file you want to select: ")
            index = int(selection) - 1
            
            if 0 <= index < len(prompt_configs):
                selected_file = prompt_configs[index]
                print(f"Selected file: {selected_file}")
                return selected_file
            else:
                print("Invalid selection. Please enter a number from the list.")
        except ValueError:
            print("Please enter a valid number.")
