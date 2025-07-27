"""
Summarization Controller - Business Logic Layer
Handles summarization requests and coordinates with services
"""

import logging
from typing import Dict, Any

from backend.services.summarization_service import get_summarization_service

logger = logging.getLogger(__name__)


def format_summarization_response(
    result: Dict[str, Any], cost_mode: str, articles=None
):
    """Format a standardized summarization response"""
    response = {
        "query": result["query"],
        "summary": result["summary"],
        "metadata": {
            "intent": result.get("intent"),
            "confidence": result.get("confidence"),
            "method": result.get("method"),
            "total_sources": result.get("total_sources", 0),
            "summary_length": result.get("summary_length", 0),
            "summary_lines": result.get("summary_lines", 0),
            "agent_powered": result.get("agent_powered", False),
            "cost_mode": cost_mode,
        },
    }

    if articles:
        response["articles"] = [
            {
                "title": article["title"],
                "url": article["url"],
                "selection_method": article.get("selection_method", "unknown"),
            }
            for article in articles
        ]

    return response


class SummarizationController:
    """
    Handles HTTP requests for summarization endpoints
    Pure HTTP concerns - no business logic
    """

    def __init__(self):
        self.summarization_service = get_summarization_service()

    def _validate_multi_source_request(
        self, request_data: Dict[str, Any]
    ) -> tuple[Dict[str, Any], int] | None:
        """Validate multi-source request parameters.
        Returns error tuple if invalid, None if valid."""
        query = request_data.get("query", "").strip()
        max_lines = request_data.get("max_lines", 30)
        max_articles = request_data.get("max_articles", 3)
        cost_mode = request_data.get("cost_mode", "BALANCED")

        if not query:
            return {"error": "Missing query parameter"}, 400
        if max_lines < 5 or max_lines > 100:
            return {"error": "max_lines must be between 5 and 100"}, 400
        if max_articles < 1 or max_articles > 10:
            return {"error": "max_articles must be between 1 and 10"}, 400
        if cost_mode not in ["MINIMAL", "BALANCED", "COMPREHENSIVE"]:
            return {
                "error": "cost_mode must be MINIMAL, BALANCED, or COMPREHENSIVE"
            }, 400
        return None

    def handle_multi_source_request(
        self, request_data: Dict[str, Any]
    ) -> tuple[Dict[str, Any], int]:
        """
        Handle /summarize_multi_source endpoint
        Pure HTTP concerns - validates input and calls service
        """
        # Input validation
        validation_error = self._validate_multi_source_request(request_data)
        if validation_error:
            return validation_error

        # Extract validated input
        query = request_data.get("query", "").strip()
        max_lines = request_data.get("max_lines", 30)
        max_articles = request_data.get("max_articles", 3)
        cost_mode = request_data.get("cost_mode", "BALANCED")

        logger.info(
            "📝 Multi-source summarization request: '%s' (max_lines: %s, max_articles: %s)",
            query,
            max_lines,
            max_articles,
        )

        try:
            # Call service (business logic happens here)
            result = self.summarization_service.summarize_multi_source_with_agents(
                query=query,
                max_articles=max_articles,
                max_lines=max_lines,
                cost_mode=cost_mode,
            )

            if "error" in result:
                return result, 500

            # Format successful response using shared utility
            response = format_summarization_response(
                result, cost_mode, result.get("articles", [])
            )

            logger.info("✅ Multi-source summarization completed successfully")
            return response, 200

        except (ValueError, KeyError, ConnectionError, TimeoutError) as e:
            logger.error("❌ Multi-source summarization failed: %s", str(e))
            return {"error": "Internal server error", "details": str(e)}, 500

    def handle_single_source_request(
        self, request_data: Dict[str, Any]
    ) -> tuple[Dict[str, Any], int]:
        """
        Handle /summarize endpoint (single source)
        Pure HTTP concerns - validates input and calls service
        """
        # Extract and validate input
        query = request_data.get("query", "").strip()
        max_lines = request_data.get("max_lines", 30)

        # Input validation
        if not query:
            return {"error": "Missing query parameter"}, 400

        if max_lines < 5 or max_lines > 100:
            return {"error": "max_lines must be between 5 and 100"}, 400

        logger.info(
            "📝 Single-source summarization request: '%s' (max_lines: %s)",
            query,
            max_lines,
        )

        try:
            # Call service (business logic happens here)
            result = self.summarization_service.summarize_single_source(
                query=query, max_lines=max_lines
            )

            if "error" in result:
                if "No Wikipedia content found" in result["error"]:
                    return result, 404
                return result, 500

            # Format successful response
            response = {
                "query": result["query"],
                "summary": result["summary"],
                "metadata": {
                    "processed_query": result.get("processed_query"),
                    "was_converted": result.get("was_converted", False),
                    "method": result.get("method"),
                    "model": result.get("model"),
                    "summary_length": result.get("summary_length", 0),
                },
                "article": result.get("article", {}),
            }

            logger.info("✅ Single-source summarization completed successfully")
            return response, 200

        except (ValueError, KeyError, ConnectionError, TimeoutError) as e:
            logger.error("❌ Single-source summarization failed: %s", str(e))
            return {"error": "Internal server error", "details": str(e)}, 500


class _SummarizationControllerSingleton:
    """Singleton wrapper for SummarizationController"""

    _instance = None

    @classmethod
    def get_instance(cls) -> SummarizationController:
        """Get or create the singleton controller instance"""
        if cls._instance is None:
            cls._instance = SummarizationController()
        return cls._instance


def get_summarization_controller() -> SummarizationController:
    """Get or create global summarization controller instance"""
    return _SummarizationControllerSingleton.get_instance()
