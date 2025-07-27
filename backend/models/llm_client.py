"""
Raw LLM API utilities - NO business logic
Used by services for actual API interactions
"""

import os
import logging
import openai
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient:
    """Raw LLM API utilities with no business logic"""

    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    @staticmethod
    def call_openai_chat(
        prompt: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ) -> str:
        """
        Raw OpenAI API call - no business logic

        Args:
            prompt: The prompt to send to OpenAI
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Generated text response
        """
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("OpenAI API call failed: %s", str(e))
            raise

    @staticmethod
    def call_openai_with_system_message(
        user_prompt: str,
        system_message: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ) -> str:
        """
        Raw OpenAI API call with system message

        Args:
            user_prompt: User message content
            system_message: System message for context
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Generated text response
        """
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("OpenAI API call with system message failed: %s", str(e))
            raise

    @staticmethod
    def call_huggingface_summarizer(
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
            raise

    @staticmethod
    def check_openai_availability() -> bool:
        """Check if OpenAI API is available and configured"""
        api_key = os.getenv("OPENAI_API_KEY")
        return api_key is not None and len(api_key.strip()) > 0


class _LLMClientSingleton:
    """Singleton wrapper for LLMClient"""

    _instance = None

    @classmethod
    def get_instance(cls) -> LLMClient:
        """Get or create the singleton client instance"""
        if cls._instance is None:
            cls._instance = LLMClient()
        return cls._instance


def get_llm_client() -> LLMClient:
    """Get or create global LLM client instance"""
    return _LLMClientSingleton.get_instance()
