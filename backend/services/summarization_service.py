#!/usr/bin/env python3
"""
SummarEaseAI Summarization Service
Centralized service for all summarization operations - Simplified MVC version
"""

import logging
from pathlib import Path
from typing import Dict, Any

import wikipedia
from dotenv import load_dotenv

# Import BERT classifier at module level to avoid import-outside-toplevel
try:
    from ml_models.bert_classifier import get_classifier as get_bert_classifier

    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    get_bert_classifier = None

logger = logging.getLogger(__name__)


class SummarizationService:
    """Service class for handling all summarization operations"""

    def __init__(self):
        """Initialize the summarization service"""
        # Load environment variables
        load_dotenv()

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

        logger.info("✅ SummarizationService initialized successfully")

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
        except (KeyError, ValueError, AttributeError) as e:
            logger.error("Error in BERT prediction: %s", e)
            return {
                "error": str(e),
                "predicted_category": "NO DETECTED",
                "confidence": 0.0,
                "all_scores": {},
            }

    def _search_wikipedia(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Basic Wikipedia search functionality"""
        try:
            # Search for articles
            search_results = wikipedia.search(query, results=max_results)
            if not search_results:
                return {
                    "error": "No Wikipedia articles found",
                    "query": query,
                    "summary": None,
                }

            # Get the first article
            article_title = search_results[0]
            page = wikipedia.page(article_title)

            # Get summary (first few sentences)
            summary = wikipedia.summary(article_title, sentences=5)

            return {
                "query": query,
                "title": article_title,
                "summary": summary,
                "url": page.url,
                "status": "success",
            }

        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation by taking the first option
            try:
                page = wikipedia.page(e.options[0])
                summary = wikipedia.summary(e.options[0], sentences=5)
                return {
                    "query": query,
                    "title": e.options[0],
                    "summary": summary,
                    "url": page.url,
                    "status": "success",
                }
            except (wikipedia.PageError, KeyError, ValueError) as inner_e:
                logger.error("Error handling disambiguation: %s", inner_e)
                return {
                    "error": f"Disambiguation error: {str(inner_e)}",
                    "query": query,
                    "summary": None,
                }
        except (ConnectionError, TimeoutError) as e:
            logger.error("Error in Wikipedia search: %s", e)
            return {"error": str(e), "query": query, "summary": None}

    def summarize_single_source(
        self,
        query: str,
        max_lines: int = 30,
        model_type: str = "wikipedia",
        use_intent: bool = True,
    ) -> Dict[str, Any]:
        """Summarize using single source (Wikipedia)"""
        try:
            # Get intent if requested
            intent_info = None
            if (
                use_intent
                and hasattr(self, "bert_model_loaded")
                and self.bert_model_loaded
            ):
                intent_result = self.predict_intent_bert(query)
                intent_info = {
                    "category": intent_result.get("predicted_category", "NO DETECTED"),
                    "confidence": intent_result.get("confidence", 0.0),
                }

            # Get Wikipedia summary
            result = self._search_wikipedia(query)

            if intent_info:
                result["intent"] = intent_info

            return result

        except (ValueError, KeyError, wikipedia.PageError, ConnectionError) as e:
            logger.error("Error in single source summarization: %s", e)
            return {
                "error": str(e),
                "query": query,
                "summary": None,
                "wikipedia_url": None,
            }

    def summarize_multi_source(
        self, query: str, cost_mode: str = "BALANCED", max_articles: int = 3
    ) -> Dict[str, Any]:
        """Summarize using multiple sources (placeholder for now)"""
        try:
            # For now, just return enhanced single source
            result = self.summarize_single_source(query)
            result["cost_mode"] = cost_mode
            result["articles_used"] = 1
            result["method"] = "single_source_placeholder"
            return result

        except (ValueError, KeyError, wikipedia.PageError, ConnectionError) as e:
            logger.error("Error in multi-source summarization: %s", e)
            return {"error": str(e), "query": query, "summary": None}

    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
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
        max_articles: int = 3,
        max_lines: int = 10,
        cost_mode: str = "balanced",
    ) -> Dict[str, Any]:
        """Summarize using multi-source approach with agent integration"""
        return self.summarize_multi_source(
            query=query, max_articles=max_articles, cost_mode=cost_mode
        )


class _SummarizationServiceSingleton:
    """Singleton wrapper for SummarizationService"""

    _instance = None

    @classmethod
    def get_instance(cls) -> SummarizationService:
        """Get or create the singleton service instance"""
        if cls._instance is None:
            cls._instance = SummarizationService()
        return cls._instance


def get_summarization_service() -> SummarizationService:
    """Get or create the global summarization service instance"""
    return _SummarizationServiceSingleton.get_instance()
