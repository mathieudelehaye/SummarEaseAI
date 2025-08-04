"""
Single-Source Summary Service
Business logic layer for single Wikipedia source summarization
Follows MVC pattern with proper separation of concerns
"""

import logging
from typing import Any, Dict

from backend.exceptions.common_service_exceptions import COMMON_SERVICE_EXCEPTIONS
from backend.models.wikipedia.wikipedia_model import get_wikipedia_service
from backend.services.common_source_summary_service import CommonSourceSummaryService

logger = logging.getLogger(__name__)


class SingleSourceSummaryService(CommonSourceSummaryService):
    """
    Single-source summarization service with BERT intent classification.

    Features:
    - BERT-based intent classification
    - Wikipedia content retrieval through model layer
    - Proper MVC separation of concerns
    - Fallback handling for missing dependencies
    """

    def __init__(self):
        # Initialize base class first
        super().__init__()

        # Configuration
        self.config = {
            **self.common_config,
            "cost_mode": "MINIMAL",  # Single source uses minimal cost mode
        }

        # Service instances (merge with common services)
        self.services = {
            **self.common_services,
            "wikipedia_model": get_wikipedia_service(),
        }

        logger.info("✅ Single-Source Summary Service initialized successfully")

    def get_single_source_summary(
        self, user_query: str, user_intent: str = None
    ) -> Dict[str, Any]:
        """
        Main method for single-source summarization with proper MVC separation.

        Args:
            query: The user's question
            model_type: Type of model to use (currently only Wikipedia)
            use_intent: Whether to use intent classification

        Returns:
            Dict with summary, intent info, and metadata
        """
        logger.info("🚀 Starting single-source summarization for: '%s'", user_query)

        # Step 1: First validate if the query is likely to find Wikipedia articles
        if not self._validate_query(user_query):
            return self._create_empty_response(
                "Query is not likely to find relevant Wikipedia articles",
                user_query,
                "N/A",
            )

        # Step 2: Intent classification
        detected_intent = self._classify_intent(user_query, user_intent)
        logger.info("🧠 Intent classified as: %s", detected_intent)

        # Step 3: Summarize articles
        try:
            # Get Wikipedia content through model layer
            result = self.services["wikipedia_model"].search_wikipedia_basic(user_query)

            # Check if result is None or contains error
            if result is None:
                logger.error("❌ Wikipedia service returned None")
                return self._create_empty_response(
                    "Wikipedia search returned no data", user_query, detected_intent
                )

            if "error" in result:
                logger.error("❌ Wikipedia service returned error: %s", result["error"])
                return self._create_empty_response(
                    result["error"], user_query, detected_intent
                )

            # Add service metadata and analytics, including intent
            summary_text = result.get("summary", "")
            result.update(
                {
                    "query": user_query,
                    "method": "single_source_agents",
                    "service_layer": "single_source_summary_service",
                    "bert_model_available": self.bert_model_loaded,
                    "intent": detected_intent,
                    "summary": summary_text,
                    "summary_length": len(summary_text),
                    "summary_lines": (
                        len(summary_text.split("\n")) if summary_text else 0
                    ),
                    "total_sources": 1,  # Single source always uses 1 article
                    "wikipedia_pages": [result.get("title", "")],
                    "search_queries_used": 1,
                    "confidence": 0.7,  # Default confidence for single-source results
                    "cost_mode": self.config["cost_mode"],
                    "usage_stats": self._get_usage_stats(),
                    "cost_tracking": result.get("usage_stats", {}),
                }
            )

            logger.info("✅ Single-source summarization completed successfully")
            return result

        except COMMON_SERVICE_EXCEPTIONS as e:
            logger.error("❌ Summary creation failed: %s", str(e))
            return self._create_empty_response(
                f"Summary creation failed: {str(e)}", user_query, detected_intent
            )


class _SingleSourceSummaryServiceSingleton:
    """Singleton wrapper for SingleSourceSummaryService"""

    _instance = None

    @classmethod
    def get_instance(cls) -> SingleSourceSummaryService:
        """Get or create the singleton service instance"""
        if cls._instance is None:
            cls._instance = SingleSourceSummaryService()
        return cls._instance


def get_single_source_summary_service() -> SingleSourceSummaryService:
    """Get or create global single-source summary service instance"""
    return _SingleSourceSummaryServiceSingleton.get_instance()
