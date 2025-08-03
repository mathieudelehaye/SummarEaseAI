"""
Query Processing Service
Centralized service for validating and processing user queries
Reduces code duplication between single-source and multi-source services
"""

import logging
from enum import Enum

from backend.models.llm.llm_client import get_llm_client

try:
    # TODO: don't import langchain here, move this to the MVC model layer
    from langchain.chat_models import ChatOpenAI

    from backend.models.agents.query_validation_agent import QueryValidationAgent

    AGENT_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None
    logging.warning("LangChain not available. Some LLM functionality will be limited.")

logger = logging.getLogger(__name__)


class _WikipediaQueryViability(Enum):
    """Internal enum for Wikipedia query viability assessment"""

    VERY_LIKELY = "Very likely"
    VERY_UNLIKELY = "Very unlikely"


class QueryProcessingService:
    """
    Service for processing and validating user queries before summarization

    Features:
    - Wikipedia query validation using ML models
    - Centralized query processing logic
    - Error handling and fallback mechanisms
    """

    def __init__(self):
        """Initialize the query processing service"""
        # Service instances
        self.llm_client = get_llm_client()

        # Initialize validation agent if available
        self.validation_agent = None
        if AGENT_AVAILABLE:
            try:
                # Create dedicated LLM for validation agent with temperature=0.3
                # for deterministic responses
                validation_llm = ChatOpenAI(
                    api_key=self.llm_client.api_key,
                    model="gpt-3.5-turbo",
                    temperature=0.3,
                )
                self.validation_agent = QueryValidationAgent(validation_llm)
                logger.info("Query validation agent initialized successfully")
            except Exception as e:
                logger.warning(
                    "Failed to initialize query validation agent: %s", str(e)
                )
                self.validation_agent = None

        logger.info("Query Processing Service initialized")

    def validate_query(self, query: str) -> bool:
        """
        Validate if the query is likely to find relevant Wikipedia articles

        This is the main public method that should be called by other services
        to validate user queries before processing.

        Args:
            query: The user's search query

        Returns:
            bool: True if query is likely to find Wikipedia articles, False otherwise
        """
        try:
            if not query or not query.strip():
                logger.warning("Empty or whitespace-only query provided")
                return False

            # Use internal validation logic (moved from query_expansion_model)
            validation_result = self._validate_wikipedia_query_internal(query)

            if validation_result == _WikipediaQueryViability.VERY_UNLIKELY:
                logger.warning("Query unlikely to find Wikipedia articles: '%s'", query)
                logger.info(
                    "Recommendation: Skipping query enhancement and secondary query generation"
                )
                return False

            logger.info("Query validation passed, proceeding with search")
            return True

        except Exception as e:
            logger.warning("Query validation failed: %s, proceeding anyway", str(e))
            # Default to True on validation failure to avoid blocking legitimate queries
            return True

    def _validate_wikipedia_query_internal(
        self, user_query: str
    ) -> _WikipediaQueryViability:
        """
        Internal method containing the validation logic.

        Args:
            user_query: The user's question to validate

        Returns:
            _WikipediaQueryViability enum indicating if query is likely to succeed
        """

        # Try to use the agent-based validation first
        if self.validation_agent:
            try:
                logger.info("Using query validation agent for: '%s'", user_query)
                agent_result = self.validation_agent.validate_query(user_query)

                viability_str = agent_result.get("viability", "Very likely")
                if viability_str == "Very likely":
                    logger.info("Agent validation result: VERY_LIKELY")
                    return _WikipediaQueryViability.VERY_LIKELY
                if viability_str == "Very unlikely":
                    logger.info("Agent validation result: VERY_UNLIKELY")
                    return _WikipediaQueryViability.VERY_UNLIKELY

                logger.warning(
                    "Unexpected agent response: '%s', falling back to LLM",
                    viability_str,
                )

            except Exception as e:
                logger.warning(
                    "Query validation agent failed: %s, falling back to LLM", str(e)
                )

        # Fallback to original LLM-based validation
        if not self.llm_client.check_openai_availability():
            logger.warning(
                "OpenAI not available for query validation, assuming VERY_LIKELY"
            )
            return _WikipediaQueryViability.VERY_LIKELY

        try:
            logger.info("Using fallback LLM validation for: '%s'", user_query)

            validation_prompt = (
                "If I look for an answer to the following question in Wikipedia will I "
                f'find any related article: "{user_query}". '
                "Return ONLY a string which can be matched to an enum variable with "
                "two possible values: Very likely, Very unlikely"
            )

            logger.info("=' Calling OpenAI for query validation...")
            response = self.llm_client.call_openai_chat(
                validation_prompt, max_tokens=10
            )

            if not response:
                logger.warning(
                    "Empty response from OpenAI validation, assuming VERY_LIKELY"
                )
                return _WikipediaQueryViability.VERY_LIKELY

            # Clean and parse response
            response_text = response.strip().lower()
            logger.info("OpenAI validation response: '%s'", response)

            # Match response to enum values
            if "very likely" in response_text:
                result = _WikipediaQueryViability.VERY_LIKELY
                logger.info("Query validation result: VERY_LIKELY")
                return result

            if "very unlikely" in response_text:
                result = _WikipediaQueryViability.VERY_UNLIKELY
                logger.info("Query validation result: VERY_UNLIKELY")
                return result

            logger.warning(
                "Unexpected validation response: '%s', assuming VERY_LIKELY",
                response,
            )
            return _WikipediaQueryViability.VERY_LIKELY

        except Exception as e:
            logger.error("Error in Wikipedia query validation: %s", str(e))
            logger.info("=Assuming VERY_LIKELY due to validation error")
            return _WikipediaQueryViability.VERY_LIKELY


class _QueryProcessingServiceSingleton:
    """Singleton wrapper for QueryProcessingService"""

    _instance = None

    @classmethod
    def get_instance(cls) -> QueryProcessingService:
        """Get or create the singleton service instance"""
        if cls._instance is None:
            cls._instance = QueryProcessingService()
        return cls._instance


def get_query_processing_service() -> QueryProcessingService:
    """Get or create global query processing service instance"""
    return _QueryProcessingServiceSingleton.get_instance()
