"""
Summarization Controller - Business Logic Layer
Handles summarization requests and coordinates with services
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from ml_models.bert_intents import BERT_CATEGORIES

from backend.exceptions.summarization_exceptions import ServiceUnavailableError

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


class SummarizationMethod(Enum):
    """Internal enum for summarization method selection"""

    MULTI_SOURCE = "Multi-source"
    SINGLE_SOURCE = "Single-source"


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
        self.bert_categories = BERT_CATEGORIES
        self.special_categories = ["NO DETECTED"]

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

    def classify_intent(self, text: str) -> Dict[str, Any]:
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

    def summarize(
        self,
        query: str,
        source_type: SummarizationMethod = SummarizationMethod.SINGLE_SOURCE,
    ) -> Dict[str, Any]:
        """Summarize using the best available method with specific error handling"""
        try:
            if (
                not MULTI_SOURCE_AVAILABLE
                and source_type == SummarizationMethod.MULTI_SOURCE
            ):
                raise ServiceUnavailableError(
                    "Multi-source summarization service is not available"
                )

            if (
                not SINGLE_SOURCE_AVAILABLE
                and source_type == SummarizationMethod.SINGLE_SOURCE
            ):
                raise ServiceUnavailableError(
                    "Single-source summarization service is not available"
                )

            if source_type == SummarizationMethod.MULTI_SOURCE:
                return self._summarize_multi_source(query)

            return self._summarize_single_source(query)

        except ServiceUnavailableError as e:
            logger.error("Service unavailable: %s", str(e))
            return {
                "error": f"Service unavailable: {str(e)}",
                "error_type": "service_unavailable",
                "query": query,
                "summary": None,
                "method": "service_unavailable",
            }

        except Exception as e:
            logger.error("Unexpected error in summarization: %s", str(e))
            return {
                "error": f"An unexpected error occurred: {str(e)}",
                "error_type": "unexpected",
                "query": query,
                "summary": None,
                "method": "unexpected_error",
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

    def _summarize_single_source(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """Summarize using single source method"""
        service = get_single_source_summary_service()
        return service.get_single_source_summary(
            user_query=query,
            user_intent=None,  # Let the service detect intent automatically
        )

    def _summarize_multi_source(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """Summarize using multi source method with agents"""
        service = get_multi_source_summary_service()
        return service.get_multi_source_summary(
            user_query=query,
            user_intent=None,  # Let the service detect intent automatically
        )


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
