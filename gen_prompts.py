import json
import re
import itertools
from typing import Dict, List
from util import select_prompt_config_file, load_json_file
import os
import sys

def find_expansion_keys(prompt_template: Dict) -> List[str]:
    """
    Dynamically find expansion keys in the prompt template.
    
    Args:
        prompt_template (Dict): Template containing expansion keys
    
    Returns:
        List[str]: List of found expansion keys
    """
    # Use regex to find all tokens between <* and *>
    template_str = json.dumps(prompt_template)
    expansion_keys = re.findall(r'<\*(\w+)\*>', template_str)
    return expansion_keys  # Return just the key names without the markers

def generate_prompts(prompt_categories: List[Dict], 
                     prompt_expansions: Dict, 
                     prompt_template: Dict) -> List[str]:
    """
    Generate prompts based on the input parameters.
    
    Args:
        prompt_categories (List[Dict]): Categories with descriptions and seeds
        prompt_expansions (Dict): Expansion options for tones and styles
        prompt_template (Dict): Template for prompt generation
    
    Returns:
        List[str]: Generated prompts
    """
    # Dynamically find expansion keys
    expansion_keys = find_expansion_keys(prompt_template)

    # Generate base prompts
    base_prompts = []
    for category_info in prompt_categories:
        seeds = category_info.get('seeds', None)
        if seeds and '<seeds>' in prompt_template['template'] :  # If seeds exist in prompt_config AND <seeds> is contained in the template, replace <seeds>
            for seed in seeds:
                current_prompt = prompt_template['template'].replace('<category>', category_info['category'])
                current_prompt = current_prompt.replace('<description>', category_info['description'])
                current_prompt = current_prompt.replace('<seeds>', seed)
                base_prompts.append(current_prompt)
        else:  # If seeds are missing, do not replace <seeds>
            current_prompt = prompt_template['template'].replace('<category>', category_info['category'])
            current_prompt = current_prompt.replace('<description>', category_info['description'])
            base_prompts.append(current_prompt)
    
    # Generate all permutations of expansions
    final_prompts = []
    for base_prompt in base_prompts:
        # Get expansion options for each key
        # expansion_options = [prompt_expansions[key] for key in expansion_keys]
        expansion_options = [prompt_expansions[key] for key in expansion_keys if key in prompt_expansions] if prompt_expansions else []

        
        # Create all possible combinations
        for combination in itertools.product(*expansion_options):
            current_prompt = base_prompt
            # Replace each expansion token with its value
            for key, value in zip(expansion_keys, combination):
                current_prompt = current_prompt.replace(f'<*{key}*>', str(value))
            final_prompts.append(current_prompt)
    
    return final_prompts

def output_prompts(prompts: List[str], prompt_categories: List[Dict], sys_prompt: str, output_file: str):
    """
    Write prompts to an output file with specified formatting.
    
    Args:
        prompts (List[str]): List of generated prompts
        prompt_categories (List[Dict]): Categories with descriptions
        sys_prompt (str): System prompt to use
        output_file (str): Path to the output file
    """    
    with open(output_file, 'w', encoding='utf-8') as f:
        if sys_prompt:
            f.write(f"System: \n")
            f.write(f"{sys_prompt}\n\n")

        for i, prompt in enumerate(prompts, 1):
            # Determine the category (extract from the prompt)
            category_match = [[cat['label'], cat['category']] for cat in prompt_categories 
                              if cat['category'] in prompt]
            category = category_match[0][1] if category_match else 'Unknown'
            label = category_match[0][0] if category_match else 'Unknown'

            f.write(f"Label: {label}\n")
            f.write(f"Category: {category}\n")
            f.write(f"Prompt {i}:\n")
            f.write(f"{prompt}\n\n")
    
    print(f"Prompts have been written to {output_file}")

def main():
    # Get filename from command line argument if provided
    prompt_config_filename = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Select and print the selected file
    prompt_config_filename = select_prompt_config_file(prompt_config_filename)
    print(f"Selected prompt config file: {prompt_config_filename}")

    # read in prompt_config
    prompt_config_json = load_json_file(prompt_config_filename)
    working_path = prompt_config_json.get("working_path") or "."
    prompt_template = prompt_config_json["prompt_template"]  # required
    system_prompt = prompt_config_json["system_prompt"]  # required
    prompt_categories_seeded_filename = prompt_config_json["prompt_categories_seeded_filename"] # required
    prompt_categories_seeded_filename_fullpath = os.path.join(working_path, prompt_categories_seeded_filename)
    prompt_expansions = prompt_config_json.get("prompt_expansions")  # prompt_expansions is optional, may be null
    working_prompt_list_filename = prompt_config_json.get("working_prompt_list_filename") or "working_prompt_list.txt"
    working_prompt_list_filename_fullpath = os.path.join(working_path, working_prompt_list_filename)

    # read in prompt_categories file
    prompt_categories = load_json_file(prompt_categories_seeded_filename_fullpath)

    # Generate and output prompts
    generated_prompts = generate_prompts(prompt_categories["categories"], prompt_expansions, prompt_template)
    output_prompts(generated_prompts, prompt_categories["categories"], system_prompt, working_prompt_list_filename_fullpath)

if __name__ == "__main__":
    main()