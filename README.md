Easy Synthetic Text Generator (aka EST Gen)

Easy Synthetic Text Generator helps you create high quality synthetic text data. Think of it as a toolkit for generating text message data using LLM based on categories provided.

What is it?
It comes with three python scripts:
1. gen_prompts.py: Generate prompts using prompt_config.json and prompt_categories.json
2. gen_data.py: Generate data by executing prompts against HuggingChat or ChatGPT
3. filter_data.py: Filter and analyze the data generated

Why should I care?
If you are looking to train an AI model to classify text communication data, then ESTGen can help you obtain your first set of labeled synthetic data fast. It will save you time and manual processing by automating many mandane tasks (like generating prompts, monitoring whether all prompts have been executed, re-running data generation process)

How do I use it?
You can check out [this post on Substack]([url](https://substack.com/home/post/p-152629522)) to get a rundown on how to generate your first synthetic dataset.
