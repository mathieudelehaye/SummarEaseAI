"""
Summarization Controller - Business Logic Layer
Handles summarization requests and coordinates with services
"""

import logging
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Import BERT classifier at module level to avoid import-outside-toplevel
try:
    from ml_models.bert_classifier import get_classifier as get_bert_classifier

    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    get_bert_classifier = None

# Import MultiSourceAgentService for multi-source summarization
try:
    from backend.services.multi_source_summary_service import (
        get_multi_source_summary_service,
    )

    MULTI_SOURCE_AVAILABLE = True
except ImportError:
    MULTI_SOURCE_AVAILABLE = False
    get_multi_source_summary_service = None
    logging.warning("MultiSourceAgentService not available")

# Import SingleSourceAgentService for single-source summarization
try:
    from backend.services.single_source_summary_service import (
        get_single_source_summary_service,
    )

    SINGLE_SOURCE_AVAILABLE = True
except ImportError:
    SINGLE_SOURCE_AVAILABLE = False
    get_single_source_summary_service = None
    logging.warning("SingleSourceAgentService not available")

logger = logging.getLogger(__name__)


class SummarizationController:
    """Controller class for handling all summarization operations"""

    def __init__(self):
        """Initialize the summarization controller"""
        # Load environment variables from backend/.env
        backend_dir = Path(__file__).parent.parent
        env_path = backend_dir / ".env"
        load_dotenv(env_path)

        # Get repository root
        self.repo_root = Path(__file__).resolve().parent.parent.parent

        # Initialize BERT classifier if available
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

        logger.info("✅ SummarizationController initialized successfully")

    def _init_bert_classifier(self):
        """Initialize BERT classifier if available"""
        try:
            if not BERT_AVAILABLE:
                logger.warning("BERT classifier not available")
                self.bert_classifier = None
                self.bert_model_loaded = False
                return

            model_path = self.repo_root / "ml_models" / "bert_gpu_model"
            logger.info("Loading BERT model from: %s", model_path)

            self.bert_classifier = get_bert_classifier(str(model_path))

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
        except ImportError as e:
            logger.warning("BERT classifier not available: %s", e)
            self.bert_classifier = None
            self.bert_model_loaded = False

    def predict_intent_bert(self, text: str) -> Dict[str, Any]:
        """Predict intent using BERT classifier"""
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
                    # BERT classifier doesn't return all scores in current implementation
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

    def summarize_single_source(
        self,
        query: str,
        max_lines: int = 30,
        model_type: str = "wikipedia",
        use_intent: bool = True,
    ) -> Dict[str, Any]:
        """Summarize using single source with proper service delegation"""
        try:
            # Always delegate to service layer (proper MVC separation)
            service = get_single_source_summary_service()
            result = service.get_single_source_summary(
                query=query,
                model_type=model_type,
                use_intent=use_intent,
            )

            logger.info("✅ Single-source summarization delegated to service layer")
            return result

        except Exception as e:
            logger.error("❌ Error in single source service delegation: %s", str(e))
            return {
                "error": str(e),
                "query": query,
                "summary": None,
                "method": "service_delegation_error",
            }

    def get_health_status(self) -> Dict[str, Any]:
        """Get application health status"""
        return {
            "status": "healthy",
            "bert_model_loaded": getattr(self, "bert_model_loaded", False),
            "categories": self.bert_categories,
            "repo_root": str(self.repo_root),
            "wikipedia_available": True,
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status for frontend"""
        return {
            "models": {
                "bert": {
                    "loaded": getattr(self, "bert_model_loaded", False),
                    "categories": self.bert_categories,
                }
            },
            "services": {
                "wikipedia": True,
                "openai": True,
            },
            "status": "healthy",
        }

    def classify_intent(self, text: str) -> Dict[str, Any]:
        """Classify intent using BERT model"""
        return self.predict_intent_bert(text)

    def summarize_multi_source_with_agents(
        self,
        query: str,
        max_lines: int = 10,
    ) -> Dict[str, Any]:
        """Summarize using multi-source approach with agent integration"""
        if not MULTI_SOURCE_AVAILABLE or not get_multi_source_summary_service:
            logger.warning(
                "MultiSourceAgentService not available, falling back to single source"
            )
            result = self.summarize_single_source(query, max_lines)
            result["method"] = "single_source_fallback"
            result["agent_powered"] = False
            result["total_sources"] = 1
            return result

        try:
            # Get the multi-source agent service
            agent_service = get_multi_source_summary_service()

            # Call the agent service with proper parameters (note: uses user_query not query)
            result = agent_service.get_multi_source_summary(
                user_query=query,
                user_intent=None,  # Let the service detect intent automatically
            )

            # Ensure proper response format
            if "error" not in result:
                # Format the response to match expected API format
                formatted_result = {
                    "query": result.get("user_query", query),
                    "summary": result.get("synthesis", ""),
                    "method": "multi_source_agents",
                    "agent_powered": True,
                    "total_sources": result.get("articles_used", 0),
                    "articles": result.get("individual_summaries", []),
                    "intent": result.get("detected_intent", "Unknown"),
                    "confidence": 0.9,  # Default confidence for agent-powered results
                    "usage_stats": result.get("usage_stats", {}),
                    "cost_tracking": result.get("usage_stats", {}),
                    "summary_length": len(result.get("synthesis", "")),
                    "summary_lines": len(result.get("synthesis", "").split("\n")),
                    "wikipedia_pages": [
                        article.get("title", "")
                        for article in result.get("individual_summaries", [])
                    ],
                }

                logger.info("✅ Multi-source summarization completed with agents")
                return formatted_result

            return result

        except Exception as e:
            logger.error("❌ Error in multi-source agent summarization: %s", str(e))
            # Fallback to single source on error
            result = self.summarize_single_source(query, max_lines)
            result["method"] = "single_source_error_fallback"
            result["agent_powered"] = False
            result["total_sources"] = 1
            result["error_details"] = str(e)
            return result


class _SummarizationControllerSingleton:
    """Singleton wrapper for SummarizationController"""

    _instance = None

    @classmethod
    def get_instance(cls) -> SummarizationController:
        """Get or create the singleton service instance"""
        if cls._instance is None:
            cls._instance = SummarizationController()
        return cls._instance


def get_summarization_controller() -> SummarizationController:
    """Get or create the global summarization controller instance"""
    return _SummarizationControllerSingleton.get_instance()
