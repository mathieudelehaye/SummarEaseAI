"""
Summarization controller - HTTP request/response handling
No business logic - just web layer coordination
"""

import logging
from typing import Dict, Any
from services.summarization_service import get_summarization_service

logger = logging.getLogger(__name__)


class SummarizationController:
    """
    Handles HTTP requests for summarization endpoints
    Pure HTTP concerns - no business logic
    """

    def __init__(self):
        self.summarization_service = get_summarization_service()

    def handle_multi_source_request(
        self, request_data: Dict[str, Any]
    ) -> tuple[Dict[str, Any], int]:
        """
        Handle /summarize_multi_source endpoint
        Pure HTTP concerns - validates input and calls service
        """
        # Extract and validate input
        query = request_data.get("query", "").strip()
        max_lines = request_data.get("max_lines", 30)
        max_articles = request_data.get("max_articles", 3)
        cost_mode = request_data.get("cost_mode", "BALANCED")

        # Input validation
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

            # Format successful response
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
                "articles": [
                    {
                        "title": article["title"],
                        "url": article["url"],
                        "selection_method": article.get("selection_method", "unknown"),
                    }
                    for article in result.get("articles", [])
                ],
            }

            logger.info("✅ Multi-source summarization completed successfully")
            return response, 200

        except Exception as e:
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

        except Exception as e:
            logger.error("❌ Single-source summarization failed: %s", str(e))
            return {"error": "Internal server error", "details": str(e)}, 500


# Global controller instance for reuse
_SUMMARIZATION_CONTROLLER = None


def get_summarization_controller() -> SummarizationController:
    """Get or create global summarization controller instance"""
    global _SUMMARIZATION_CONTROLLER
    if _SUMMARIZATION_CONTROLLER is None:
        _SUMMARIZATION_CONTROLLER = SummarizationController()
    return _SUMMARIZATION_CONTROLLER
