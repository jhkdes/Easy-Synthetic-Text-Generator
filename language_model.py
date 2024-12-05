from datetime import datetime
from hugchat import hugchat
from hugchat.login import Login
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

def log_step(step_name):
    """Log execution step with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Executing: {step_name}")

def print_current_model(chatbot):
    """Print current model information"""

    """Print current model information using available methods"""
    try:
        # Try get_available_llm_models first
        models = chatbot.get_available_llm_models()
        log_step("Available models:")
        for i, model in enumerate(models):
            log_step(f"  [{i}] {model}")
        
        # Try to get current model id
        try:
            current_id = chatbot.active_model
            log_step(f"Current model ID: {current_id}")
        except:
            log_step("Could not get current_model ID")
                
    except Exception as e:
        log_step(f"Error getting model information: {str(e)}")

class LanguageModel:
    def __init__(self, model_name, api_key, system_prompt=None, temperature=0.7, email="", passwd="", model_index=1):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.request_count = 0  # Track number of send_request() calls

        # Initialize the appropriate API client based on the model name
        if "gpt-4o-mini" in model_name.lower():
            self.chat = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=temperature)

        elif "huggingchat" in model_name.lower():
            self.chat = self.setup_huggingchat(email, passwd, model_index, system_prompt)

        else:
            raise ValueError("Unsupported language model")

        # Set up the initial conversation with the system prompt, if provided
        if system_prompt:
            self.initial_conversation = self.send_request(system_prompt)

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
        log_step("Logging in to HuggingChat")
        sign = Login(email, password)
        log_step("Signing in with sign.login()...")
        cookies = sign.login()

        log_step("Inspecting cookies object...")
        if cookies:
            log_step(f"Cookies object type: {type(cookies)}")
            log_step(f"Cookies content (if dict-like): {cookies.get_dict() if hasattr(cookies, 'get_dict') else 'Not a dict-like object'}")
        else:
            log_step("Cookies object is None. Login might have failed.")

        # Create ChatBot
        log_step("Creating ChatBot instance")
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

        try:
            # Select model by index
            log_step("Switching model by index")
            chatbot.switch_llm(model_index)  # Assuming the model might be at index 1
            log_step(f"Successfully switched to model index = {model_index}")
            print_current_model(chatbot)  # Print updated model info
        except:
            log_step(f"Warning: Could not switch to model index = {model_index}. Using default model")
            print_current_model(chatbot)  # Print current model info

        # Create a new conversation with the system prompt
        log_step("Creating new conversation with system prompt")
        conversation_id = chatbot.new_conversation(
            system_prompt=system_prompt,
            switch_to=True  # Automatically switch to this conversation
        )

        log_step(f"conversation_id created: {conversation_id}")

        return chatbot

    def send_request(self, prompt):
        # Increment request count
        self.request_count += 1

        # For HuggingChat, start a new conversation every 13 requests
        if "huggingchat" in self.model_name.lower() and self.request_count > 13:
            log_step("Resetting conversation after 13 requests")
            # Create a new conversation with the system prompt
            conversation_id = self.chat.new_conversation(
                system_prompt=self.system_prompt,
                switch_to=True  # Automatically switch to this conversation
            )
            log_step(f"New conversation_id created: {conversation_id}")
            
            # Reset request count
            self.request_count = 1

        if "gpt-4o-mini" in self.model_name.lower():
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt) 
            ]
            response = self.chat(messages)
            return response.content
        elif "huggingchat" in self.model_name.lower():
            response = self.chat.chat(prompt)

            # Implement Hugging Face API request here
            return response
        else:
            raise ValueError("Unsupported language model")
