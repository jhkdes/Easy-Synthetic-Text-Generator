# Easy Synthetic Text Generator (aka ESTGen)

Easy Synthetic Text Generator (ESTGen) helps you create quality synthetic text data. Think of it as a toolkit for generating synthetic text data using LLM based on categories provided.

## What is it?
ESTGen is a simple script execution framework that comes with three python scripts:
1. **gen_prompts.py**: Generate prompts using prompt_template and prompt_categories.json so that gen_data can generate synthetic data using LLM
2. **gen_data.py**: Generate data by executing pre-generated prompts against HuggingChat model or ChatGPT
3. **filter_data.py**: Filter the dataset by cosine similarity and analyze the data generated

ESTGen execution is configured using estgen_config.json. While gen_prompts and gen_data are supported natively, other python scripts can be added in the execution steps. When a new python script is added, the script is invoked using JSON parameters as arguments.

For example, 
```
        {
            "script": "convert_to_prompt_categories.py",
            "description": "Convert scenarios to category seeds in JSON format",
            "input_filename": "scenarios.csv",
            "output_filename": "prompt_categories_seeded.json"
        },
```
would be executed as
```
python convert_to_prompt_categories.py --input_filename scenarios.csv --output_filename prompt_categories_seeded.json
```

## Why should I care?
If you are looking to train an AI model to classify text communication data, then ESTGen can help you obtain your first set of labeled synthetic data fast. It will save you time and manual processing by automating many mandane tasks (like generating prompts, monitoring whether all prompts have been executed, re-running data generation process).

It offers the following features:

1. **Prompt generation**: generate prompts using template, using variable replacement and seeding
2. **Data generation**: multiple models, pick up where it left off, using earlier prompt response cache
3. **Multiple language model support**: Out-of-the-box support for free Hugging Chat APIs which offers 10 different LLMs including meta-llama/Llama-3.3-70B-Instruct and CohereForAI/c4ai-command-r-plus-08-2024; and Chat GPT APIs which offers GPT-4o and GPT-4o-Mini
4. **Anti-hallucination support**: Built-in 3 retry logic to recover from invalid JSON and incorrect number of elements in the JSON array
5. **Cosine similarity filtering and data analysis reporting**: A script to filter dataset by cosine similarity threshold so that dataset consists of sufficiently distinct data, and report to show shorted, median, and longest data along with cosine similarity threshold sensitivity analysis
6. **Custom scripts with input and output**: A place holder to introduce any custom python script that takes input and generates output

## How do I use it?
You can run ESTGen in three easy steps:

- Step 1: Clone the Repo https://github.com/jhkdes/Easy-Synthetic-Text-Generator
- Step 2: Obtain API Keys from HuggingChat and ChatGPT, and populate them in estgen_config file
  - login_email: Login email to HuggingChat
  - login_passwd: Login passwd to HuggingChat
  - api_key: API key to ChatGPT
- Step 3: Install required packages, and run
```
pip install -r requirements
python run_estgen.py
```
It will start generating synthetic text conversations using 9 depression categories outlined in the prompt_categories.json file. My synthetic dataset copy which was generated using the same configs is [available on HuggingFace](https://huggingface.co/datasets/jaeho9kim/estgen-depression-chat/blob/main/texts_messages_regen_2_messages_filtered.csv)
