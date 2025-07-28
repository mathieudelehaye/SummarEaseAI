"""
Raw LLM API utilities - NO business logic
Used by services for actual API interactions
"""

import logging
import os

import openai
from dotenv import load_dotenv
from transformers import pipeline

# LangChain imports
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        ChatOpenAI = None
        logging.warning("LangChain not available. Some LLM functionality will be limited.")

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient:
    """Raw LLM API utilities with no business logic"""

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.7, max_tokens: int = 1500):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key

    def call_openai_chat(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        Raw OpenAI API call - no business logic

        Args:
            prompt: The prompt to send to OpenAI
            model: Model name to use (uses instance default if None)
            temperature: Sampling temperature (uses instance default if None)
            max_tokens: Maximum tokens in response (uses instance default if None)

        Returns:
            Generated text response
        """
        if not self.api_key:
            return "Error: OpenAI API key not configured"
        
        if not prompt.strip():
            return "Error: Empty prompt provided"
            
        try:
            if LANGCHAIN_AVAILABLE and ChatOpenAI:
                # Use LangChain ChatOpenAI
                llm = ChatOpenAI(
                    openai_api_key=self.api_key,
                    model_name=model or self.model,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )
                response = llm.invoke(prompt)
                return response.content.strip()
            else:
                # Fallback to raw OpenAI API
                response = openai.ChatCompletion.create(
                    model=model or self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("OpenAI API call failed: %s", str(e))
            return f"Error calling OpenAI: {str(e)}"

    def call_openai_with_system_message(
        self,
        user_prompt: str,
        system_message: str,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        Raw OpenAI API call with system message

        Args:
            user_prompt: User message content
            system_message: System message for context
            model: Model name to use (uses instance default if None)
            temperature: Sampling temperature (uses instance default if None)
            max_tokens: Maximum tokens in response (uses instance default if None)

        Returns:
            Generated text response
        """
        if not self.api_key:
            return "Error: OpenAI API key not configured"
            
        try:
            if LANGCHAIN_AVAILABLE and ChatOpenAI:
                # Use LangChain ChatOpenAI with system message
                llm = ChatOpenAI(
                    openai_api_key=self.api_key,
                    model_name=model or self.model,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )
                # For LangChain, we need to format the message properly
                formatted_prompt = f"System: {system_message}\n\nUser: {user_prompt}"
                response = llm.invoke(formatted_prompt)
                return response.content.strip()
            else:
                # Fallback to raw OpenAI API
                response = openai.ChatCompletion.create(
                    model=model or self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("OpenAI API call with system message failed: %s", str(e))
            return f"Error: {str(e)}"

    def call_huggingface_summarizer(
        self,
        text: str,
        model_name: str = "facebook/bart-large-cnn",
        max_length: int = 250,
        min_length: int = 50,
    ) -> str:
        """
        Raw HuggingFace API call - no business logic

        Args:
            text: Text to summarize
            model_name: HuggingFace model name
            max_length: Maximum summary length
            min_length: Minimum summary length

        Returns:
            Generated summary text
        """
        try:
            summarizer = pipeline("summarization", model=model_name)
            result = summarizer(
                text, max_length=max_length, min_length=min_length, do_sample=False
            )
            return result[0]["summary_text"]
        except Exception as e:
            logger.error("HuggingFace summarization failed: %s", str(e))
            return f"Error: {str(e)}"

    def check_openai_availability(self) -> bool:
        """Check if OpenAI API is available and configured"""
        return self.api_key is not None and len(self.api_key.strip()) > 0

    def get_model_info(self) -> dict:
        """Get model information and configuration"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "openai_available": self.check_openai_availability(),
            "langchain_available": LANGCHAIN_AVAILABLE,
        }


class _LLMClientSingleton:
    """Singleton wrapper for LLMClient"""

    _instance = None

    @classmethod
    def get_instance(cls) -> LLMClient:
        """Get or create the singleton client instance"""
        if cls._instance is None:
            # Check if API key is available
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and len(api_key.strip()) > 0:
                cls._instance = LLMClient()
            else:
                # Return a dummy client instead of None to avoid NoneType errors
                logger.warning("OpenAI API key not found, creating dummy LLM client")
                cls._instance = LLMClient()  # Will handle missing key internally
        return cls._instance


def get_llm_client() -> LLMClient:
    """Get or create global LLM client instance"""
    return _LLMClientSingleton.get_instance()
