"""
Single-Source Agent Service
Business logic layer for single Wikipedia source summarization
Follows MVC pattern with proper separation of concerns
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Import BERT classifier
try:
    from ml_models.bert_classifier import get_classifier as get_bert_classifier

    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    get_bert_classifier = None

from backend.models.wikipedia.wikipedia_model import get_wikipedia_service
from backend.services.common_source_summary_service import CommonSourceSummaryService

# Load environment variables from backend/.env
backend_dir = Path(__file__).parent.parent
env_path = backend_dir / ".env"
load_dotenv(env_path)

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

logger = logging.getLogger(__name__)


class SingleSourceAgentService(CommonSourceSummaryService):
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
            "bert_model_path": repo_root / "ml_models" / "bert_gpu_model",
            "cost_mode": "MINIMAL",  # Single source uses minimal cost mode
        }

        # Service instances (merge with common services)
        self.services = {
            **self.common_services,
            "wikipedia_model": get_wikipedia_service(),
        }

        # Initialize BERT classifier
        self._init_bert_classifier()

        # Define categories
        self.bert_categories = [
            "History",
            "Music",
            "Science",
            "Sports",
            "Technology",
            "Finance",
        ]
        self.special_categories = ["NO DETECTED"]
        self.all_categories = self.bert_categories + self.special_categories

        logger.info("✅ Single-Source Agent Service initialized successfully")

    def _init_bert_classifier(self):
        """Initialize BERT classifier if available"""
        try:
            if not BERT_AVAILABLE:
                logger.warning("BERT classifier not available")
                self.bert_classifier = None
                self.bert_model_loaded = False
                return

            logger.info("Loading BERT model from: %s", self.config["bert_model_path"])

            self.bert_classifier = get_bert_classifier(
                str(self.config["bert_model_path"])
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

    def classify_intent(self, text: str) -> Dict[str, Any]:
        """Classify intent using BERT classifier"""
        if not hasattr(self, "bert_model_loaded") or not self.bert_model_loaded:
            return {
                "error": "BERT model not loaded",
                "predicted_category": "NO DETECTED",
                "confidence": 0.0,
                "all_scores": {},
            }

        try:
            prediction = self.bert_classifier.predict(text)
            if isinstance(prediction, tuple) and len(prediction) == 2:
                predicted_intent, confidence = prediction
                return {
                    "text": text,
                    "predicted_category": predicted_intent,
                    "confidence": confidence,
                    "all_scores": {},
                }
            return {"error": "Invalid prediction format"}
        except Exception as e:
            logger.error("Error in BERT prediction: %s", e)
            return {
                "error": str(e),
                "predicted_category": "NO DETECTED",
                "confidence": 0.0,
                "all_scores": {},
            }

    def get_single_source_summary(
        self,
        query: str,
        model_type: str = "wikipedia",
        use_intent: bool = True,
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
        logger.info("🚀 Starting single-source summarization for: '%s'", query)

        try:
            # Step 1: First validate if the query is likely to find Wikipedia articles
            if not self._validate_query(query):
                logger.warning(
                    "⚠️ Query is not likely to find relevant Wikipedia articles"
                )
                return self._create_empty_response(
                    "Query is not likely to find relevant Wikipedia articles",
                    query,
                    "N/A",
                )

            # Step 2: Intent classification if requested
            intent_info = None
            if use_intent and self.bert_model_loaded:
                intent_result = self.classify_intent(query)
                intent_info = {
                    "category": intent_result.get("predicted_category", "NO DETECTED"),
                    "confidence": intent_result.get("confidence", 0.0),
                }
                logger.info(
                    "🧠 Intent classified as: %s (%.3f)",
                    intent_info["category"],
                    intent_info["confidence"],
                )

            # Step 3: Get Wikipedia content through model layer
            result = self.services["wikipedia_model"].search_wikipedia_basic(query)

            # Step 4: Add intent information
            if intent_info:
                result["intent"] = intent_info

            # Step 5: Add service metadata and analytics
            summary_text = result.get("summary", "")
            result.update(
                {
                    "method": "single_source_agent",
                    "model_type": model_type,
                    "bert_model_available": self.bert_model_loaded,
                    "service_layer": "single_source_summary_service",
                    "summary_length": len(summary_text),
                    "summary_lines": (
                        len(summary_text.split("\n")) if summary_text else 0
                    ),
                }
            )

            logger.info("✅ Single-source summarization completed successfully")
            return result

        except Exception as e:
            logger.error("❌ Error in single source summarization: %s", e)
            return {
                "error": str(e),
                "query": query,
                "summary": None,
                "method": "single_source_agent_error",
                "model_type": model_type,
            }


class _SingleSourceAgentServiceSingleton:
    """Singleton wrapper for SingleSourceAgentService"""

    _instance = None

    @classmethod
    def get_instance(cls) -> SingleSourceAgentService:
        """Get or create the singleton service instance"""
        if cls._instance is None:
            cls._instance = SingleSourceAgentService()
        return cls._instance


def get_single_source_summary_service() -> SingleSourceAgentService:
    """Get or create global single-source agent service instance"""
    return _SingleSourceAgentServiceSingleton.get_instance()
