"""
Custom exceptions for summarization services
Provides specific error types for better error handling and user feedback
"""


class SummarizationError(Exception):
    """Base exception for all summarization-related errors"""


class ServiceUnavailableError(SummarizationError):
    """Raised when required services (OpenAI, Wikipedia, etc.) are unavailable"""
