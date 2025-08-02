"""
OpenAI Query Generation Service
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

from backend.models.llm.llm_client import get_llm_client

# Common exceptions for service error handling
COMMON_SERVICE_EXCEPTIONS = (
    ValueError,
    KeyError,
    AttributeError,
    ConnectionError,
    TimeoutError,
)

# Initialize service environment
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

logger = logging.getLogger(__name__)

# Check ChatOpenAI availability for query generation
try:
    from langchain.chat_models import ChatOpenAI

    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain ChatOpenAI available for query generation")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None
    logger.warning(
        "LangChain ChatOpenAI not available - using fallback query generation"
    )


class QueryGenerationService:
    """
    OpenAI-powered query generation service for multi-source search
    Business logic for intelligent secondary query generation
    """

    def __init__(self):
        self.llm_client = get_llm_client()
        self.llm = None

        logger.info("🔧 Initializing OpenAI Query Generation Service...")
        logger.info("🔧 LANGCHAIN_AVAILABLE: %s", LANGCHAIN_AVAILABLE)
        logger.info(
            "🔧 OpenAI availability check: %s",
            self.llm_client.check_openai_availability(),
        )

        if LANGCHAIN_AVAILABLE and self.llm_client.check_openai_availability():
            try:
                logger.info("🔧 Creating ChatOpenAI instance...")
                # Pass the API key explicitly to ensure it's available

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.error("❌ OPENAI_API_KEY not found in environment")
                    self.llm = None
                    return

                self.llm = ChatOpenAI(
                    openai_api_key=api_key,
                    model="gpt-3.5-turbo",
                    temperature=0.3,  # Low temperature for consistent results
                    max_tokens=1000,
                )
                logger.info(
                    "✅ OpenAI Query Generation Service initialized successfully"
                )
            except (ImportError, ValueError, AttributeError, OSError) as e:
                logger.error(
                    "❌ Failed to initialize OpenAI query generation: %s",
                    str(e),
                    exc_info=True,
                )
                self.llm = None
        else:
            logger.warning(
                "⚠️ OpenAI not available - will use fallback query generation"
            )

    def generate_secondary_queries(
        self, primary_query: str, intent: str, max_queries: int = 6
    ) -> Dict[str, List[str]]:
        """
        Generate intelligent secondary queries using OpenAI
        Core business logic for query expansion

        Args:
            primary_query: The original user query
            intent: Detected intent category (Biography, History, Science, etc.)
            max_queries: Maximum number of queries to generate

        Returns:
            Dictionary with generated queries and metadata
        """
        logger.info(
            "🧠 Generating secondary queries for: '%s' (intent: %s)",
            primary_query,
            intent,
        )

        logger.info(
            "🔧 LLM status: %s", "Available" if self.llm else "None - using fallback"
        )
        if self.llm:
            return self._openai_query_generation(primary_query, intent, max_queries)
        return self._fallback_query_generation(primary_query, intent, max_queries)

    def _openai_query_generation(
        self, primary_query: str, intent: str, max_queries: int
    ) -> Dict[str, List[str]]:
        """Generate queries using OpenAI/LangChain"""
        try:
            logger.info(
                "🔧 Starting OpenAI query generation for: %s (intent: %s)",
                primary_query,
                intent,
            )
            # Intent-specific prompts for better query generation
            intent_strategies = {
                "history": "historical events, key figures, timeline, causes and effects",
                "science": "scientific principles, research, discoveries, applications",
                "technology": "innovations, development, applications, impact",
                "music": "musical style, albums, influence, band members, career",
                "sports": "competitions, achievements, rules, history",
                "finance": "economic impact, market influence, financial aspects",
            }

            strategy = intent_strategies.get(intent, "comprehensive information")

            prompt_template = (
                f"You are an expert research assistant helping to find "
                f"comprehensive information about a topic.\n\n"
                f'Original query: "{primary_query}"\n'
                f"Topic category: {intent}\n"
                f"Research focus: {strategy}\n\n"
                f"Generate {max_queries} different Wikipedia search queries that would "
                f"find the most relevant and comprehensive information about this topic.\n\n"
                f"Requirements:\n"
                f"1. Each query should target a different aspect of the topic\n"
                f"2. Use terms that are likely to be Wikipedia article titles"
            )
            prompt_template += (
                "\n3. Avoid generic question words (who, what, when, where)\n"
                "4. Make queries specific and searchable\n"
                f"5. Focus on {strategy}\n\n"
                'Return ONLY a JSON list of search queries, like: ["query1", "query2", "query3"]'
            )

            response = self.llm_client.call_openai_chat(
                prompt=prompt_template, temperature=0.3, max_tokens=500
            )

            # Parse JSON response
            try:
                logger.info("🔍 OpenAI raw response: %s", response.strip()[:200])
                queries = json.loads(response.strip())
                if isinstance(queries, list):
                    logger.info(
                        "✅ Generated %d secondary queries using OpenAI: %s",
                        len(queries),
                        queries,
                    )
                    return {
                        "queries": queries[:max_queries],
                        "method": "openai_generation",
                        "intent_strategy": strategy,
                        "confidence": 0.9,
                    }

                logger.warning("OpenAI response is not a list: %s", type(queries))
            except json.JSONDecodeError as e:
                logger.warning(
                    "Failed to parse OpenAI JSON response: %s. Raw response: %s",
                    str(e),
                    response.strip()[:200],
                )

        except COMMON_SERVICE_EXCEPTIONS as e:
            logger.error("OpenAI query generation failed: %s", str(e), exc_info=True)

        return self._fallback_query_generation(primary_query, intent, max_queries)

    def _fallback_query_generation(
        self, primary_query: str, intent: str, max_queries: int
    ) -> Dict[str, List[str]]:
        """Fallback query generation using rule-based approach"""

        # Clean the primary query
        clean_query = primary_query.lower()
        for prefix in [
            "who were ",
            "who was ",
            "what is ",
            "what are ",
            "tell me about ",
        ]:
            if clean_query.startswith(prefix):
                clean_query = clean_query[len(prefix) :].strip()
                break

        # Generate variations based on intent
        queries = [clean_query]

        if intent == "History":
            queries.extend(
                [
                    f"{clean_query} history",
                    f"{clean_query} timeline",
                    f"{clean_query} events",
                ]
            )
        elif intent == "Biography":
            queries.extend(
                [
                    f"{clean_query} biography",
                    f"{clean_query} life",
                    f"{clean_query} career",
                ]
            )
        elif intent == "Music":
            queries.extend(
                [
                    f"{clean_query} band",
                    f"{clean_query} discography",
                    f"{clean_query} albums",
                ]
            )
        elif intent == "Science":
            queries.extend(
                [
                    f"{clean_query} research",
                    f"{clean_query} theory",
                    f"{clean_query} discovery",
                ]
            )
        elif intent == "Technology":
            queries.extend(
                [
                    f"{clean_query} technology",
                    f"{clean_query} innovation",
                    f"{clean_query} development",
                ]
            )
        else:
            queries.extend([f"{clean_query} overview", f"{clean_query} information"])

        # Remove duplicates and limit
        unique_queries = list(dict.fromkeys(queries))[:max_queries]

        logger.info(
            "✅ Generated %d secondary queries using fallback method",
            len(unique_queries),
        )

        return {
            "queries": unique_queries,
            "method": "rule_based_fallback",
            "intent_strategy": f"{intent} focused",
            "confidence": 0.6,
        }


class _QueryGenerationServiceSingleton:
    """Singleton wrapper for OpenAIQueryGenerationService"""

    _instance = None

    @classmethod
    def get_instance(cls) -> QueryGenerationService:
        """Get or create the singleton service instance"""
        if cls._instance is None:
            cls._instance = QueryGenerationService()
        return cls._instance


def get_query_generation_service() -> QueryGenerationService:
    """Get or create global query generation service instance"""
    return _QueryGenerationServiceSingleton.get_instance()
