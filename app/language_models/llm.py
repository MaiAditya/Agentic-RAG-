from langchain_community.chat_models import ChatOpenAI
from langchain_core.callbacks import AsyncCallbackHandler
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

class AsyncIteratorCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""
    
    def __init__(self):
        self.tokens = []
        
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Collect tokens as they are generated."""
        self.tokens.append(token)
        
    def get_tokens(self):
        """Get the collected tokens."""
        return self.tokens

async def create_llm(streaming: bool = False):
    """Create a language model instance."""
    try:
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Base configuration
        config = {
            "temperature": 0.7,
            "model_name": "gpt-3.5-turbo",
            "openai_api_key": api_key
        }
        
        # Add streaming handler if requested
        if streaming:
            callback_handler = AsyncIteratorCallbackHandler()
            config["streaming"] = True
            config["callbacks"] = [callback_handler]
        
        llm = ChatOpenAI(**config)
        logger.info("Successfully initialized language model")
        return llm
        
    except Exception as e:
        logger.error(f"Error initializing language model: {str(e)}")
        raise