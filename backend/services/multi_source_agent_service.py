"""
Multi-Source Agent Service
Moved from utils/multi_source_agent.py to proper services layer
Main orchestration service with rate limiting and cost control
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import wikipedia
from dotenv import load_dotenv
from ml_models.bert_classifier import get_classifier as get_bert_classifier

# Load environment variables from backend/.env
from pathlib import Path
backend_dir = Path(__file__).parent.parent
env_path = backend_dir / ".env"
load_dotenv(env_path)

from backend.models.langchain_model import get_langchain_agents_service
from backend.models.llm_client import get_llm_client
from backend.models.query_expansion_model import get_query_generation_service
from backend.models.wikipedia_model import WikipediaService
from backend.services.summarization_workflow_service import (
    summarize_article_with_intent,
)

# Common exceptions for service error handling
COMMON_SERVICE_EXCEPTIONS = (
    ValueError,
    KeyError,
    AttributeError,
    ConnectionError,
    TimeoutError,
)

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

logger = logging.getLogger(__name__)


# RATE LIMITING CONFIGURATION
class RateLimitConfig:
    """Configuration for controlling OpenAI API usage and costs."""

    # Core limits
    MAX_ARTICLES_PER_SUMMARY = 3  # Default: conservative limit
    MAX_SECONDARY_QUERIES = 4  # Limit OpenAI query generation calls
    MAX_WIKIPEDIA_SEARCHES = 8  # Total Wikipedia API calls

    # Feature toggles
    ENABLE_OPENAI_QUERY_GENERATION = True  # Set to False to disable OpenAI query gen
    ENABLE_LANGCHAIN_AGENTS = True  # Set to False to disable LangChain agents
    ENABLE_MULTI_SOURCE = True  # Set to False for single-article fallback

    # Cost control modes
    COST_MODE = "BALANCED"  # Options: "MINIMAL", "BALANCED", "COMPREHENSIVE"

    @classmethod
    def get_limits_for_mode(cls, mode: str) -> Dict[str, int]:
        """Get rate limits based on cost control mode."""
        limits = {
            "MINIMAL": {
                "max_articles": 1,
                "max_secondary_queries": 0,  # No secondary queries
                "max_wikipedia_searches": 1,
                "enable_openai": False,
                "enable_agents": False,
            },
            "BALANCED": {
                "max_articles": 3,
                "max_secondary_queries": 3,
                "max_wikipedia_searches": 6,
                "enable_openai": True,
                "enable_agents": True,
            },
            "COMPREHENSIVE": {
                "max_articles": 6,
                "max_secondary_queries": 6,
                "max_wikipedia_searches": 12,
                "enable_openai": True,
                "enable_agents": True,
            },
        }
        return limits.get(mode, limits["BALANCED"])


class MultiSourceAgentService:
    """
    Enhanced Multi-Source Agent with comprehensive logging and rate limiting.

    Features:
    - OpenAI query generation logging
    - Wikipedia search tracking
    - Article selection reasoning
    - Smart rate limiting for cost control
    """

    def __init__(self, cost_mode: str = "BALANCED"):
        # Configuration objects to reduce instance attributes
        self.config = {
            "bert_model_path": repo_root / "ml_models" / "bert_gpu_model",
            "cost_mode": cost_mode.upper(),
            "limits": RateLimitConfig.get_limits_for_mode(cost_mode.upper()),
        }

        # Service instances
        self.services = {
            "bert_classifier": get_bert_classifier(str(self.config["bert_model_path"])),
            "llm_client": get_llm_client(),
            "query_generator": get_query_generation_service(),
            "langchain_agents": get_langchain_agents_service(),
        }

        # Usage tracking
        self.usage = {
            "openai_calls_made": 0,
            "wikipedia_calls_made": 0,
            "articles_processed": 0,
        }

        # Initialize BERT model
        self.bert_model_loaded = (
            self.services["bert_classifier"].load_model()
            if self.services["bert_classifier"]
            else False
        )

        logger.info(
            "🎛️ Multi-Source Agent Service initialized in %s mode",
            self.config["cost_mode"],
        )
        logger.info(
            "📊 Limits: %s articles, %s secondary queries",
            self.config["limits"]["max_articles"],
            self.config["limits"]["max_secondary_queries"],
        )
        logger.info("🚀 GPU BERT model loaded: %s", self.bert_model_loaded)

    def get_multi_source_summary(
        self, user_query: str, user_intent: str = None
    ) -> Dict[str, Any]:
        """
        Main method for multi-source summarization with comprehensive error handling and logging.

        Args:
            user_query: The user's question
            user_intent: Optional pre-classified intent

        Returns:
            Dict with summary, articles used, and metadata
        """
        logger.info("🚀 Starting multi-source summarization for: '%s'", user_query)
        self._reset_usage_tracking()

        # Step 1: Intent classification
        detected_intent = self._classify_intent(user_query, user_intent)
        logger.info("🧠 Intent classified as: %s", detected_intent)

        # Step 2: Generate search queries
        search_queries = self._generate_search_queries(user_query, detected_intent)
        if not search_queries:
            logger.warning("⚠️ No valid search queries generated")
            return self._create_error_response(
                "No valid search queries generated", user_query, detected_intent
            )

        logger.info("🔍 Generated %s search queries", len(search_queries))

        # Step 3: Search and gather articles
        articles_data = self._search_and_gather_articles(search_queries, user_query)

        if not articles_data:
            raise ValueError("No articles to summarize")

        logger.info("📚 Successfully gathered %d articles", len(articles_data))

        # Step 4: Summarize articles
        try:
            summary_result = self._create_multi_source_summary(
                articles_data, user_query, detected_intent
            )

            # Add metadata
            summary_result.update(
                {
                    "user_query": user_query,
                    "detected_intent": detected_intent,
                    "cost_mode": self.config["cost_mode"],
                    "usage_stats": self._get_usage_stats(),
                    "articles_used": len(articles_data),
                    "search_queries_used": len(search_queries),
                }
            )

            logger.info("✅ Multi-source summary completed successfully")
            return summary_result

        except COMMON_SERVICE_EXCEPTIONS as e:
            logger.error("❌ Summary creation failed: %s", str(e))
            return self._create_error_response(
                f"Summary creation failed: {str(e)}", user_query, detected_intent
            )

    def _classify_intent(self, user_query: str, user_intent: str = None) -> str:
        """Classify user intent using BERT or fallback methods"""
        if user_intent:
            return user_intent

        if self.bert_model_loaded:
            try:
                # Use correct BERT classifier method: predict() returns (intent, confidence)
                intent, confidence = self.services["bert_classifier"].predict(user_query)
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

    def _generate_search_queries(self, user_query: str, intent: str) -> Optional[List[str]]:
        """Generate search queries using various methods"""
        queries = [user_query]  # Always include original query

        # First validate if the query is likely to find Wikipedia articles
        try:
            from backend.models.query_expansion_model import WikipediaQueryViability
            validation_result = self.services["query_generator"].validate_wikipedia_query(user_query)

            if validation_result == WikipediaQueryViability.VERY_UNLIKELY:
                logger.warning("⚠️ Query unlikely to find Wikipedia articles: '%s'", user_query)
                logger.info("🔄 Skipping query enhancement and secondary query generation")
                return None

            logger.info("✅ Query validation passed, proceeding with enhancements")

        except Exception as e:
            logger.warning("⚠️ Query validation failed: %s, proceeding anyway", str(e))

        # Use enhanced query based on intent
        # Initialize Wikipedia service
        wikipedia_service = WikipediaService()
        enhanced_query = wikipedia_service.enhance_query_with_intent(user_query, intent)
        if enhanced_query != user_query:
            queries.append(enhanced_query)

        # Generate secondary queries if enabled and within limits
        if (
            self.config["limits"]["enable_openai"]
            and self.usage["openai_calls_made"]
            < self.config["limits"]["max_secondary_queries"]
        ):

            try:
                logger.info("🔧 Calling OpenAI query generator...")
                secondary_queries_result = self.services[
                    "query_generator"
                ].generate_secondary_queries(
                    user_query,
                    intent,
                    max_queries=min(
                        2,
                        self.config["limits"]["max_secondary_queries"]
                        - self.usage["openai_calls_made"],
                    ),
                )
                logger.info("🔧 Query generator returned: %s", secondary_queries_result)
                
                # Extract queries from the returned dictionary
                if isinstance(secondary_queries_result, dict) and "queries" in secondary_queries_result:
                    secondary_queries = secondary_queries_result["queries"]
                    queries.extend(secondary_queries)
                    self.usage["openai_calls_made"] += len(secondary_queries)
                    logger.info("🤖 Generated %d secondary queries using OpenAI: %s", 
                               len(secondary_queries), secondary_queries)
                else:
                    # Handle case where it returns a list directly
                    if isinstance(secondary_queries_result, list):
                        queries.extend(secondary_queries_result)
                        logger.info("🤖 Generated %d secondary queries (direct list): %s", 
                                   len(secondary_queries_result), secondary_queries_result)
                    else:
                        logger.warning("⚠️ Unexpected query generator response type: %s", 
                                      type(secondary_queries_result))
            except COMMON_SERVICE_EXCEPTIONS as e:
                logger.warning("⚠️ Secondary query generation failed: %s", str(e))

        # Remove duplicates while preserving order
        unique_queries = []
        for query in queries:
            if query not in unique_queries:
                unique_queries.append(query)

        return unique_queries[: self.config["limits"]["max_secondary_queries"] + 1]

    def _search_and_gather_articles(
        self, search_queries: List[str], original_query: str
    ) -> List[Dict[str, Any]]:
        """Search for articles using multiple queries and methods"""
        articles_data = []
        seen_titles = set()

        for query in search_queries:
            if (
                self.usage["wikipedia_calls_made"]
                >= self.config["limits"]["max_wikipedia_searches"]
            ):
                logger.info(
                    "⚠️ Wikipedia search limit reached (%d)",
                    self.config["limits"]["max_wikipedia_searches"],
                )
                break

            if len(articles_data) >= self.config["limits"]["max_articles"]:
                logger.info(
                    "📊 Article limit reached (%d)",
                    self.config["limits"]["max_articles"],
                )
                break

            try:
                # Try LangChain agents first if enabled
                if self.config["limits"]["enable_agents"]:
                    article_result = self.services[
                        "langchain_agents"
                    ].intelligent_wikipedia_search(query)
                    if (
                        "article_info" in article_result
                        and "error" not in article_result["article_info"]
                    ):
                        article_info = article_result["article_info"]
                        if article_info["title"] not in seen_titles:
                            articles_data.append(article_info)
                            seen_titles.add(article_info["title"])
                            self.usage["wikipedia_calls_made"] += 1
                            logger.info(
                                "🎯 LangChain agents found: '%s'", article_info["title"]
                            )
                        continue

                # Fallback to regular Wikipedia search
                # Initialize Wikipedia service
                wikipedia_service = WikipediaService()
                article_info = wikipedia_service.search_and_fetch_article_info(query)
                self.usage["wikipedia_calls_made"] += 1

                if article_info and "error" not in article_info:
                    if article_info["title"] not in seen_titles:
                        articles_data.append(article_info)
                        seen_titles.add(article_info["title"])
                        logger.info(
                            "📚 Standard search found: '%s'", article_info["title"]
                        )

            except (
                ConnectionError,
                TimeoutError,
                ValueError,
                wikipedia.PageError,
            ) as e:
                logger.warning("⚠️ Search failed for query '%s': %s", query, str(e))
                continue

        return articles_data

    def _create_multi_source_summary(
        self, articles_data: List[Dict[str, Any]], user_query: str, intent: str
    ) -> Dict[str, Any]:
        """Create a comprehensive summary from multiple articles"""
        if not articles_data:
            raise ValueError("No articles to summarize")

        # Prepare article summaries
        article_summaries = []
        article_metadata = []

        for article in articles_data:
            try:
                # Get individual article summary
                summary_result = summarize_article_with_intent(
                    article["content"], user_query, intent
                )

                article_summaries.append(
                    {
                        "title": article["title"],
                        "url": article["url"],
                        "summary": summary_result.get("summary", ""),
                        "relevance_score": summary_result.get("relevance_score", 0.5),
                    }
                )

                article_metadata.append(
                    {
                        "title": article["title"],
                        "url": article["url"],
                        "content_length": len(article["content"]),
                        "summary_length": len(summary_result.get("summary", "")),
                    }
                )

                self.usage["articles_processed"] += 1

            except COMMON_SERVICE_EXCEPTIONS as e:
                logger.warning(
                    "⚠️ Failed to summarize article '%s': %s",
                    article.get("title", "Unknown"),
                    str(e),
                )
                continue

        if not article_summaries:
            raise RuntimeError("Failed to summarize any articles")

        # Create synthesis
        synthesis = self._synthesize_multi_source_content(
            article_summaries, user_query, intent
        )

        return {
            "synthesis": synthesis,
            "individual_summaries": article_summaries,
            "article_metadata": article_metadata,
            "method": "multi_source_agent",
        }

    def _synthesize_multi_source_content(
        self, article_summaries: List[Dict], user_query: str, intent: str
    ) -> str:
        """Synthesize content from multiple article summaries using OpenAI like the original app"""
        if len(article_summaries) == 1:
            return article_summaries[0]["summary"]

        # Use OpenAI to create unified synthesis like the original app
        try:
            # Import here to avoid circular imports
            from backend.models.openai_summarizer_model import get_openai_api_key
            
            # Check imports at runtime
            try:
                from langchain_openai import ChatOpenAI
                from langchain.prompts import PromptTemplate
                from langchain.chains import LLMChain
                langchain_available = True
            except ImportError:
                try:
                    from langchain.chat_models import ChatOpenAI
                    from langchain.prompts import PromptTemplate
                    from langchain.chains import LLMChain
                    langchain_available = True
                except ImportError:
                    langchain_available = False

            if not langchain_available:
                logger.warning("⚠️ LangChain not available for final synthesis, using fallback")
                return self._create_fallback_synthesis(article_summaries, user_query, intent)
                
            api_key = get_openai_api_key()
            if not api_key:
                logger.warning("⚠️ OpenAI API key not available for final synthesis, using fallback")
                return self._create_fallback_synthesis(article_summaries, user_query, intent)
            
            logger.info(f"🎯 Creating final synthesis from {len(article_summaries)} articles")
            
            # Prepare the content for synthesis (like original app)
            articles_content = []
            for i, summary in enumerate(article_summaries, 1):
                title = summary.get('title', 'Unknown')
                content = summary.get('summary', 'No summary available')
                articles_content.append(f"Article {i}: {title}\n{content}\n")
            
            combined_content = "\n".join(articles_content)
            
            # Create intent-specific synthesis prompt (like original app)
            intent_specific_instructions = {
                'music': "Focus on the band's formation, key members, musical style, major albums, cultural impact, and legacy in popular music.",
                'biography': "Focus on the person's life story, major achievements, contributions, and historical significance.",
                'history': "Focus on causes, key events, major figures, consequences, and historical significance.",
                'science': "Focus on scientific principles, discoveries, applications, and impact on the field.",
                'technology': "Focus on how the technology works, development, applications, and impact on society.",
                'general': "Focus on the main topics, key information, and overall significance."
            }
            
            instruction = intent_specific_instructions.get(intent.lower(), intent_specific_instructions['general'])
            
            prompt_template = f"""You are tasked with creating a comprehensive final summary by synthesizing information from multiple Wikipedia articles.

User's Question: "{user_query}"

Your task: Create ONE unified summary that answers the user's question comprehensively by combining insights from all the provided articles.

Requirements:
- Maximum 30 lines
- Well-structured and easy to read
- {instruction}
- Synthesize information rather than just listing facts
- Provide a coherent narrative that flows logically
- Include the most important and relevant information from all sources

Articles to synthesize:
{{combined_content}}

Final Comprehensive Summary:
"""
            
            # Create and run the synthesis chain (like original app)
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,
                max_completion_tokens=1500,
                api_key=api_key
            )
            
            prompt = PromptTemplate(
                input_variables=["combined_content"],
                template=prompt_template
            )
            
            chain = LLMChain(llm=llm, prompt=prompt)
            
            logger.info(f"🤖 Requesting final synthesis from OpenAI...")
            logger.info(f"   📊 Synthesizing {len(article_summaries)} articles")
            logger.info(f"   🎯 Intent: {intent}")
            
            final_summary = chain.run(combined_content=combined_content)
            
            logger.info(f"✅ Final synthesis completed successfully")
            logger.info(f"   📝 Generated {len(final_summary.split())} words")
            
            # Track OpenAI usage
            self.usage["openai_calls_made"] += 1
            
            return final_summary.strip()
            
        except Exception as e:
            logger.error(f"❌ Failed to create OpenAI synthesis: {str(e)}")
            logger.info("🔄 Falling back to basic synthesis")
            return self._create_fallback_synthesis(article_summaries, user_query, intent)

    def _create_fallback_synthesis(self, article_summaries: List[Dict], user_query: str, intent: str) -> str:
        """Fallback synthesis when OpenAI is not available"""
        # Create a combined synthesis (original behavior)
        synthesis_parts = [
            f"Based on {len(article_summaries)} Wikipedia articles, here's a "
            f'comprehensive answer to your question: "{user_query}"\n'
        ]

        # Sort by relevance score if available
        sorted_summaries = sorted(
            article_summaries, key=lambda x: x.get("relevance_score", 0.5), reverse=True
        )

        for i, summary_data in enumerate(sorted_summaries, 1):
            synthesis_parts.append(
                f"{i}. **{summary_data['title']}**: {summary_data['summary']}\n"
                f"   Source: {summary_data['url']}\n"
            )

        # Add conclusion
        synthesis_parts.append(
            f"\nThis synthesis combines information from {len(article_summaries)} sources "
            f"to provide a comprehensive answer about {intent.replace('_', ' ')} topics."
        )

        return "\n".join(synthesis_parts)

    def _create_error_response(
        self, error_message: str, user_query: str, intent: str
    ) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "error": error_message,
            "user_query": user_query,
            "detected_intent": intent,
            "cost_mode": self.config["cost_mode"],
            "usage_stats": self._get_usage_stats(),
            "method": "multi_source_agent_error",
        }

    def _reset_usage_tracking(self):
        """Reset usage tracking for new request"""
        self.usage["openai_calls_made"] = 0
        self.usage["wikipedia_calls_made"] = 0
        self.usage["articles_processed"] = 0

    def _get_usage_stats(self) -> Dict[str, int]:
        """Get current usage statistics"""
        return {
            "openai_calls_made": self.usage["openai_calls_made"],
            "wikipedia_calls_made": self.usage["wikipedia_calls_made"],
            "articles_processed": self.usage["articles_processed"],
            "openai_calls_remaining": max(
                0,
                self.config["limits"]["max_secondary_queries"]
                - self.usage["openai_calls_made"],
            ),
            "wikipedia_calls_remaining": max(
                0,
                self.config["limits"]["max_wikipedia_searches"]
                - self.usage["wikipedia_calls_made"],
            ),
        }

    def set_cost_mode(self, mode: str):
        """Change cost control mode"""
        self.config["cost_mode"] = mode.upper()
        self.config["limits"] = RateLimitConfig.get_limits_for_mode(
            self.config["cost_mode"]
        )
        logger.info("🎛️ Cost mode changed to %s", self.config["cost_mode"])
        logger.info("📊 New limits: %s", self.config["limits"])


class _MultiSourceAgentServiceSingleton:
    """Singleton wrapper for MultiSourceAgentService"""

    _instance = None

    @classmethod
    def get_instance(cls) -> MultiSourceAgentService:
        """Get or create the singleton service instance"""
        if cls._instance is None:
            cls._instance = MultiSourceAgentService()
        return cls._instance


def get_multi_source_agent_service() -> MultiSourceAgentService:
    """Get or create global multi-source agent service instance"""
    return _MultiSourceAgentServiceSingleton.get_instance()
