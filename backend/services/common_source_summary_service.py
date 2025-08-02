"""
Common Source Summary Service
Base class containing shared functionality for summarization services
Reduces code duplication between single-source and multi-source services
"""

import logging
from abc import ABC
from typing import Any, Dict, Optional

from backend.services.query_processing_service import get_query_processing_service

logger = logging.getLogger(__name__)


class CommonSourceSummaryService(ABC):
    """
    Abstract base class for summarization services

    Contains shared functionality and patterns used by both single-source
    and multi-source summarization services.

    Common features:
    - Error response standardization
    - Usage tracking and statistics
    - Query processing integration
    - Health status reporting
    """

    def __init__(self):
        """Initialize common service components"""
        # Common service instances that all children should have
        self.common_services = {
            "query_processor": get_query_processing_service(),
        }

        # Initialize usage tracking structure
        self._init_usage_tracking()

        logger.info("✅ Common Source Summary Service base initialized")

    def _init_usage_tracking(self):
        """Initialize usage tracking structure"""
        self.usage = {
            "openai_calls_made": 0,
            "wikipedia_calls_made": 0,
            "articles_processed": 0,
        }

    def _create_error_response(
        self,
        error_message: str,
        user_query: str,
        intent: str,
        method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create standardized error response

        Args:
            error_message: Description of the error
            user_query: The original user query
            intent: Detected or provided intent
            method: Service method that generated the error (auto-detected if None)

        Returns:
            Dict with standardized error response format
        """
        # Auto-detect method from class name if not provided
        if method is None:
            class_name = self.__class__.__name__
            if "Multi" in class_name:
                method = "multi_source_agent_error"
            elif "Single" in class_name:
                method = "single_source_agent_error"
            else:
                method = "common_summary_error"

        return {
            "error": error_message,
            "user_query": user_query,
            "detected_intent": intent,
            "cost_mode": getattr(self, "config", {}).get("cost_mode", "unknown"),
            "usage_stats": self._get_usage_stats(),
            "method": method,
            "success": False,
        }

    def _get_usage_stats(self) -> Dict[str, int]:
        """
        Get current usage statistics

        Returns:
            Dict containing usage counters and remaining capacity
        """
        # Get limits from config if available, otherwise use defaults
        config_limits = getattr(self, "config", {}).get("limits", {})
        max_secondary_queries = config_limits.get("max_secondary_queries", 4)
        max_wikipedia_searches = config_limits.get("max_wikipedia_searches", 8)

        return {
            "openai_calls_made": self.usage["openai_calls_made"],
            "wikipedia_calls_made": self.usage["wikipedia_calls_made"],
            "articles_processed": self.usage["articles_processed"],
            "openai_calls_remaining": max(
                0,
                max_secondary_queries - self.usage["openai_calls_made"],
            ),
            "wikipedia_calls_remaining": max(
                0,
                max_wikipedia_searches - self.usage["wikipedia_calls_made"],
            ),
        }

    def _reset_usage_tracking(self):
        """Reset usage tracking for new request"""
        self.usage["openai_calls_made"] = 0
        self.usage["wikipedia_calls_made"] = 0
        self.usage["articles_processed"] = 0

    def _validate_query(self, query: str) -> bool:
        """
        Validate user query using common query processing service

        Args:
            query: User's search query

        Returns:
            bool: True if query is valid and likely to succeed
        """
        return self.common_services["query_processor"].validate_query(query)
