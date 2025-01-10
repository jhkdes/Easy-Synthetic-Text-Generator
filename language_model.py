from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from hugchat import hugchat
from hugchat.login import Login
import time
import json
import logging
import hashlib

def print_current_model(chatbot):
    """Print current model information"""

    """Print current model information using available methods"""
    try:
        # Try get_available_llm_models first
        models = chatbot.get_available_llm_models()
        logging.debug("Available models:")
        for i, model in enumerate(models):
            logging.debug(f"  [{i}] {model}")
        
        # Try to get current model id
        try:
            current_id = chatbot.active_model
            logging.info(f"Current model ID: {current_id}")
        except:
            logging.error("Could not get current_model ID")
                
    except Exception as e:
        logging.error(f"Error getting model information: {str(e)}")

class LanguageModel:
    def __init__(self, model_name, api_key, system_prompt=None, temperature=0.7, email="", passwd="", model_index=1, progress_bar=None):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.request_count = 0  # Track number of send_request() calls
        self.api_key = api_key
        self.email = email
        self.passwd = passwd
        self.model_index = model_index

        if progress_bar:
            progress_bar.set_description("Language Model: initializing model")

        # Initialize the appropriate API client based on the model name
        self.chat = self.initialize_chat(model_name)

        if progress_bar:
            progress_bar.update(50)
            progress_bar.set_description("Language Model: initializing conversation")

        # Set up the initial conversation with the system prompt, if provided
        if system_prompt:
            self.initial_conversation = self.send_request(system_prompt)

    def initialize_chat(self, model_name):
        """Helper to initialize chat based on model name."""
        if "gpt-4o-mini" in model_name.lower():
            return ChatOpenAI(model="gpt-4o-mini", openai_api_key=self.api_key, temperature=self.temperature)
        elif "huggingchat" in model_name.lower():
            return self.setup_huggingchat(self.email, self.passwd, self.model_index, self.system_prompt)
        else:
            raise ValueError("Unsupported language model")

    def clean_up(self):
        """Cleanup method for the previous model."""
        # Add any necessary cleanup logic here
        if hasattr(self.chat, "close"):
            self.chat.close()
        self.chat = None

    def update_model(self, new_model_name, api_key, system_prompt=None, temperature=0.7, email="", passwd="", model_index=1, progress_bar=None):
        """
        Update the model name and reinitialize the underlying chat object.

        Args:
            new_model_name (str): The name of the new language model.
        """
        if progress_bar:
            progress_bar.set_description("Language Model: cleaning up")

        # Clean up resources tied to the current model
        self.clean_up()

        if progress_bar:
            progress_bar.update(10)
            progress_bar.set_description("Language Model: initializing model")

        # Update the model name
        self.model_name = new_model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.request_count = 0  # Track number of send_request() calls
        self.api_key = api_key
        self.email = email
        self.passwd = passwd
        self.model_index = model_index

        # Reinitialize the chat object with the new model name
        self.chat = self.initialize_chat(new_model_name)

        if progress_bar:
            progress_bar.update(40)
            progress_bar.set_description("Language Model: initializing conversation")

        # Optionally, reinitialize the initial conversation with the system prompt
        if self.system_prompt:
            self.initial_conversation = self.send_request(self.system_prompt)

    def get_current_model_name(self):
        if "gpt-4o-mini" in self.model_name.lower():
            return self.model_name
        elif "huggingchat" in self.model_name.lower():
            # huggingchat has multiple models available
            return self.chat.active_model
        else:
            raise ValueError("Unsupported language model")

    def setup_huggingchat(self, email, password, model_index, system_prompt):
        """
        This function logs in to Hugging Face and creates a ChatBot instance.

        Args:
            email (str): User email for Hugging Face account.
            password (str): User password for Hugging Face account.
            model_index (int): Index of model to use from Huggingchat
            system_prompt (str): System prompt to initialze the chat session

        Returns:
            ChatBot: A ChatBot instance for interacting with the Hugging Face API.
        """
        logging.info("Logging in to HuggingChat")
        sign = Login(email, password)
        logging.debug("Signing in with sign.login()...")
        cookies = sign.login()

        logging.debug("Inspecting cookies object...")
        if cookies:
            logging.debug(f"Cookies object type: {type(cookies)}")
            logging.debug(f"Cookies content (if dict-like): {cookies.get_dict() if hasattr(cookies, 'get_dict') else 'Not a dict-like object'}")
        else:
            logging.debug("Cookies object is None. Login might have failed.")

        # Create ChatBot
        logging.info("Creating ChatBot instance")
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

        try:
            # Select model by index
            logging.info("Switching model by index")
            chatbot.switch_llm(model_index)  # Assuming the model might be at index 1
            logging.info(f"Successfully switched to model index = {model_index}")
            print_current_model(chatbot)  # Print updated model info
        except:
            logging.error(f"Warning: Could not switch to model index = {model_index}. Using default model")
            print_current_model(chatbot)  # Print current model info

        # Create a new conversation with the system prompt
        logging.info("Creating new conversation with system prompt")
        conversation_id = chatbot.new_conversation(
            system_prompt=system_prompt,
            switch_to=True  # Automatically switch to this conversation
        )

        logging.info(f"conversation_id created: {conversation_id}")

        return chatbot

    def create_new_conversation(self):
        # For HuggingChat, start a new conversation every 13 requests
        if "huggingchat" in self.model_name.lower():
            logging.info("Creating a new conversation")
            # Create a new conversation with the system prompt
            conversation_id = self.chat.new_conversation(
                system_prompt=self.system_prompt,
                switch_to=True  # Automatically switch to this conversation
            )
            logging.info(f"New conversation_id created: {conversation_id}")
            
            # Reset request count
            self.request_count = 1

    def send_request(self, prompt, prompt_response_type="string", prompt_response_array_constraint=None, prompt_response_cache=None):
        def execute_send_request(prompt):
            # first check whether prompt response has been hashed
            if prompt_response_cache: 
                logging.debug(f"Found cached prompt and response")
                prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
                logging.debug(f"  prompt_hash: {prompt_hash}")

                # pop because retry should be sent to the model
                response_dict = prompt_response_cache.pop(prompt_hash, None) 
                logging.debug(f"  response_dict: {response_dict}")

                if response_dict:
                    logging.info(f"Returning from earlier cache: {prompt_hash}")
                    return response_dict

            model_name = self.get_current_model_name()

            if "gpt-4o-mini" in self.model_name.lower():
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=prompt) 
                ]
                response = self.chat(messages)
                return {"response": response.content, "model_name": model_name}
            elif "huggingchat" in self.model_name.lower():
                response = self.chat.chat(prompt)
                response.wait_until_done()

                # Implement Hugging Face API request here
                return {"response": response, "model_name": model_name}
            else:
                raise ValueError("Unsupported language model")

        def check_json_array_length(json_structure, array_element_name, num_of_elements):
            """
            Checks whether the specified array in the JSON structure contains the expected number of elements.

            Args:
                json_structure (dict): The JSON structure to check.
                array_element_name (str): The key name of the array in the JSON structure.
                num_of_elements (int): The expected number of elements in the array.

            Raises:
                ValueError: If the number of elements in the array does not match the expected number.
            """
            if array_element_name not in json_structure:
                raise KeyError(f"Key '{array_element_name}' not found in the JSON structure.")

            array = json_structure[array_element_name]
            if not isinstance(array, list):
                raise TypeError(f"'{array_element_name}' is not an array in the JSON structure.")

            if len(array) != num_of_elements:
                raise ValueError(f"Incorrect number of elements found in the array: expected {num_of_elements}, found {len(array)}")

        def send_request_with_retries(prompt, prompt_response_type, max_retries=3, prompt_response_array_constraint=None):
            # returns dict containing "response" and "model_name"
            prepended_message_added = False

            for attempt in range(1, max_retries + 1):
                try:
                    logging.info(f"prompt: {prompt}")
                    response_dict = execute_send_request(prompt)
                    model_name = response_dict.get("model_name", "Internal error: Unknown model name")
                    logging.info(f"model_name: {model_name}")

                    # Validate JSON response if required
                    if prompt_response_type == "json":
                        response = str(response_dict.get("response", "Internal error: No response found"))
                        logging.info(f"response: {response}")
                        response_json = json.loads(response)
                        if prompt_response_array_constraint: 
                            str_part, num_part = prompt_response_array_constraint.split(',')
                            check_json_array_length(response_json, str_part, int(num_part))

                    return response_dict

                except json.JSONDecodeError as e:
                    logging.error(f"JSONDecodeError on attempt {attempt}/{max_retries}: {str(e)}")
                    if 'Failed to parse response' in str(e):
                        time.sleep(60)
                    else:
                        if not prepended_message_added:
                            prompt = ("Previous response was in an invalid JSON format. Make sure the response is in a valid JSON format. "
                                    + prompt)
                            prepended_message_added = True
                except Exception as e:
                    logging.error(f"Error on attempt {attempt}/{max_retries}: {str(e)}")
                    if any(msg in str(e) for msg in ["too many messages", "Model is overloaded"]):
                        logging.error("Sleeping 5 mins due to detected issue")
                        time.sleep(300)

                logging.info("Retrying...")

            raise RuntimeError(f"Failed to send request after {max_retries} retries")

        # Increment request count
        self.request_count += 1

        # For HuggingChat, start a new conversation every 13 requests
        if "huggingchat" in self.model_name.lower() and self.request_count > 13:
            logging.info("Resetting conversation after 13 requests")
            # Create a new conversation with the system prompt
            conversation_id = self.chat.new_conversation(
                system_prompt=self.system_prompt,
                switch_to=True  # Automatically switch to this conversation
            )
            logging.info(f"New conversation_id created: {conversation_id}")
            
            # Reset request count
            self.request_count = 1

        return send_request_with_retries(prompt, prompt_response_type, 3, prompt_response_array_constraint)


