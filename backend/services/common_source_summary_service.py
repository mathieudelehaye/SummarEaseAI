"""
Common Source Summary Service
Base class containing shared functionality for summarization services
Reduces code duplication between single-source and multi-source services
"""

import logging
import sys
from abc import ABC
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from ml_models.bert_intents import BERT_CATEGORIES

from backend.services.query_processing_service import get_query_processing_service

# Import BERT classifier
try:
    from ml_models.bert_classifier import get_classifier as get_bert_classifier

    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    get_bert_classifier = None

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Load environment variables from backend/.env
backend_dir = Path(__file__).parent.parent
env_path = backend_dir / ".env"
load_dotenv(env_path)

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

        # Common configuration
        self.common_config = {
            "bert_model_path": repo_root / "ml_models" / "bert_gpu_model",
        }

        # Initialize common services
        self.common_services = {
            "bert_classifier": get_bert_classifier(
                str(self.common_config["bert_model_path"])
            ),
            "query_processor": get_query_processing_service(),
        }

        # Initialize usage tracking structure
        self._init_usage_tracking()

        # Initialize BERT classifier
        self._init_bert_classifier()

        # Define categories
        self.bert_categories = BERT_CATEGORIES
        self.special_categories = ["NO DETECTED"]

        logger.info("✅ Common Source Summary Service base initialized")

    def _init_bert_classifier(self):
        """Initialize BERT classifier if available"""
        try:
            if not BERT_AVAILABLE:
                logger.warning("BERT classifier not available")
                self.bert_classifier = None
                self.bert_model_loaded = False
                return

            logger.info(
                "Loading BERT model from: %s", self.common_config["bert_model_path"]
            )

            self.bert_classifier = get_bert_classifier(
                str(self.common_config["bert_model_path"])
            )

            if self.bert_classifier is None:
                logger.error("❌ Failed to initialize BERT classifier!")
                self.bert_model_loaded = False
            else:
                if not self.bert_classifier.is_loaded():
                    logger.info("🔄 Loading BERT model...")
                    self.bert_model_loaded = self.bert_classifier.load_model()
                    if self.bert_model_loaded:
                        logger.info("✅ BERT model loaded successfully")
                    else:
                        logger.error("❌ Failed to load BERT model!")
                else:
                    self.bert_model_loaded = True
                    logger.info("✅ BERT model already loaded")
        except Exception as e:
            logger.warning("BERT classifier not available: %s", e)
            self.bert_classifier = None
            self.bert_model_loaded = False

    def _init_usage_tracking(self):
        """Initialize usage tracking structure"""
        self.usage = {
            "openai_calls_made": 0,
            "wikipedia_calls_made": 0,
            "articles_processed": 0,
        }

    def _reset_usage_tracking(self):
        """Reset usage tracking for new request"""
        self.usage["openai_calls_made"] = 0
        self.usage["wikipedia_calls_made"] = 0
        self.usage["articles_processed"] = 0

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

    def _validate_query(self, query: str) -> bool:
        """
        Validate user query using common query processing service

        Args:
            query: User's search query

        Returns:
            bool:
                True if query is valid and likely to succeed.
                False otherwise.
        """
        if not self.common_services["query_processor"].validate_query(query):
            logger.warning(
                "⚠️ Query is not likely to find relevant Wikipedia articles: '%s'", query
            )
            return False
        return True

    def _classify_intent(self, user_query: str, user_intent: str = None) -> str:
        """Classify user intent using BERT or fallback methods"""
        if user_intent:
            return user_intent

        if self.bert_model_loaded:
            try:
                # Use correct BERT classifier method: predict() returns (intent, confidence)
                intent, _ = self.services["bert_classifier"].predict(user_query)
                return intent.lower() if intent else "general_knowledge"
            except (ValueError, AttributeError, KeyError, RuntimeError) as e:
                logger.warning("⚠️ BERT classification failed: %s", str(e))

        # Fallback intent classification
        return self._fallback_intent_classification(user_query)

    def _fallback_intent_classification(self, query: str) -> str:
        """Simple rule-based intent classification"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["who is", "who was", "who were"]):
            return "biography"
        if any(word in query_lower for word in ["what is", "what are", "define"]):
            return "definition"
        if any(word in query_lower for word in ["when did", "when was", "date"]):
            return "historical_event"
        return "general_knowledge"

    # TODO: move to controller
    def _create_empty_response(
        self,
        message: str,
        user_query: str,
        intent: str,
        method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create standardized no-results response (instead of error)

        Args:
            error_message: Description of why no results were found
            user_query: The original user query
            intent: Detected or provided intent
            method: Service method that generated the response (auto-detected if None)

        Returns:
            Dict with standardized no-results response format
        """
        # Auto-detect method from class name if not provided
        if method is None:
            class_name = self.__class__.__name__
            if "Multi" in class_name:
                method = "multi_source_agent"
            elif "Single" in class_name:
                method = "single_source_agent"
            else:
                method = "common_summary"

        return {
            "summary": message,
            "synthesis": message,
            "user_query": user_query,
            "detected_intent": intent,
            "cost_mode": getattr(self, "config", {}).get("cost_mode", "unknown"),
            "usage_stats": self._get_usage_stats(),
            "method": method,
            "success": True,
            "articles_found": 0,
            "summary_length": len(message),
            "summary_lines": len(message.split("\n")) if message else 0,
        }
