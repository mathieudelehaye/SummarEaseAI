"""
LangChain Agents Service
Moved from utils/agents_service.py to proper services layer
Specialized LangChain agent orchestration
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import wikipedia

try:
    from langchain.chat_models import ChatOpenAI

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None
    logging.warning(
        "LangChain not available. Some LLM functionality will be limited."
    )

from backend.models.llm.openai_summarizer_model import get_openai_api_key

from ..llm.llm_client import get_llm_client
from .article_selection_agent import ArticleSelectionAgent
from .query_enhancement_agent import QueryEnhancementAgent

# Setup agent service environment
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

logger = logging.getLogger(__name__)


class LangChainAgentsModel:
    """
    LangChain-based agent orchestration service
    Handles query enhancement and article selection using LangChain framework
    """

    def __init__(self):
        self.llm_client = get_llm_client()
        self.query_enhancement_agent = None
        self.article_selection_agent = None

        if LANGCHAIN_AVAILABLE and self.llm_client.check_openai_availability():
            try:
                # Create shared LLM once
                api_key = get_openai_api_key()
                shared_llm = ChatOpenAI(
                    api_key=api_key, model="gpt-3.5-turbo", temperature=0.3
                )

                # Pass shared LLM to both agents
                self.query_enhancement_agent = QueryEnhancementAgent(shared_llm)
                self.article_selection_agent = ArticleSelectionAgent(shared_llm)
                logger.info("✅ LangChain Agents Service initialized successfully")
            except (ImportError, ValueError, AttributeError, OSError) as e:
                logger.error("❌ Failed to initialize LangChain agents: %s", str(e))
        else:
            logger.warning(
                "⚠️ LangChain agents not available - falling back to basic search"
            )

    def intelligent_wikipedia_search(
        self, user_query: str, max_options: int = 5
    ) -> Dict[str, Any]:
        """
        Complete intelligent Wikipedia search using LangChain agents

        Workflow:
        1. Query Enhancement Agent improves the search query
        2. Search Wikipedia with enhanced query
        3. Article Selection Agent picks the best result
        4. Return comprehensive result with reasoning
        """
        logger.info("🚀 Starting LangChain agent-powered search for: '%s'", user_query)

        if not (self.query_enhancement_agent and self.article_selection_agent):
            return self._fallback_search(user_query, max_options)

        try:
            # Step 1: Enhance query using LangChain agent
            enhancement_result = self.query_enhancement_agent.enhance_query(user_query)
            enhanced_query = enhancement_result["enhanced_query"]

            logger.info("🧠 Query enhanced: '%s' → '%s'", user_query, enhanced_query)

            # Step 2: Search Wikipedia with enhanced query
            search_results = wikipedia.search(enhanced_query, results=max_options)

            if not search_results:
                # Try original query as fallback
                search_results = wikipedia.search(user_query, results=max_options)
                logger.info("📚 Fallback search found %d articles", len(search_results))
            else:
                logger.info("📚 Enhanced search found %d articles", len(search_results))

            if not search_results:
                return {
                    "error": "No Wikipedia articles found",
                    "user_query": user_query,
                    "enhanced_query": enhanced_query,
                    "enhancement_result": enhancement_result,
                }

            # Step 3: Select best article using LangChain agent
            selection_result = self.article_selection_agent.select_best_article(
                user_query, search_results
            )
            selected_title = selection_result["selected_article"]

            logger.info("🎯 Agent selected: '%s'", selected_title)

            # Step 4: Get the selected article content
            article_info = self._fetch_article_content(selected_title, search_results)

            return {
                "user_query": user_query,
                "enhancement_result": enhancement_result,
                "search_results": search_results,
                "selection_result": selection_result,
                "article_info": article_info,
                "agent_system": "agents_service",
                "total_articles_considered": len(search_results),
                "agents_used": ["QueryEnhancementAgent", "ArticleSelectionAgent"],
            }

        except (
            ValueError,
            KeyError,
            AttributeError,
            ImportError,
            ConnectionError,
        ) as e:
            logger.error("❌ LangChain agent search failed: %s", str(e))
            return self._fallback_search(user_query, max_options)

    def _fetch_article_content(
        self, selected_title: str, search_results: List[str]
    ) -> Dict[str, Any]:
        """Fetch content for the selected article"""
        try:
            # Try to get the page with auto-suggest disabled first
            try:
                selected_page = wikipedia.page(selected_title, auto_suggest=False)
            except wikipedia.exceptions.DisambiguationError as e:
                logger.info(
                    "📝 Disambiguation page, selecting first option: %s", e.options[0]
                )
                selected_page = wikipedia.page(e.options[0], auto_suggest=False)
            except (wikipedia.exceptions.PageError, ConnectionError, ValueError):
                logger.info(
                    "⚠️ Exact match failed, trying with auto-suggest for: '%s'",
                    selected_title,
                )
                selected_page = wikipedia.page(selected_title, auto_suggest=True)

            return {
                "title": selected_page.title,
                "url": selected_page.url,
                "content": selected_page.content,
                "summary": selected_page.summary,
            }

        except (
            wikipedia.PageError,
            wikipedia.DisambiguationError,
            ConnectionError,
            ValueError,
        ) as e:
            logger.error("❌ Failed to get selected article: %s", str(e))
            # Try fallback to first search result
            if search_results:
                try:
                    logger.info(
                        "🔄 Trying fallback to first search result: '%s'",
                        search_results[0],
                    )
                    fallback_page = wikipedia.page(
                        search_results[0], auto_suggest=False
                    )
                    return {
                        "title": fallback_page.title,
                        "url": fallback_page.url,
                        "content": fallback_page.content,
                        "summary": fallback_page.summary,
                    }
                except (
                    wikipedia.PageError,
                    wikipedia.DisambiguationError,
                    ConnectionError,
                ) as fallback_error:
                    logger.error("❌ Fallback also failed: %s", str(fallback_error))

            return {"error": f"Failed to get article: {str(e)}"}

    def _fallback_search(self, user_query: str, max_options: int) -> Dict[str, Any]:
        """Fallback search without LangChain agents"""
        try:
            search_results = wikipedia.search(user_query, results=max_options)

            if not search_results:
                return {"error": "No Wikipedia articles found"}

            # Use first result as fallback
            selected_title = search_results[0]
            article_info = self._fetch_article_content(selected_title, search_results)

            return {
                "user_query": user_query,
                "search_results": search_results,
                "article_info": article_info,
                "agent_system": "fallback_basic_search",
                "total_articles_considered": len(search_results),
            }

        except (
            wikipedia.PageError,
            wikipedia.DisambiguationError,
            ConnectionError,
            ValueError,
        ) as e:
            logger.error("❌ Fallback search failed: %s", str(e))
            return {"error": f"Search failed: {str(e)}"}


class _LangChainAgentsServiceSingleton:
    """Singleton wrapper for LangChainAgentsService"""

    _instance = None

    @classmethod
    def get_instance(cls) -> LangChainAgentsModel:
        """Get or create the singleton service instance"""
        if cls._instance is None:
            cls._instance = LangChainAgentsModel()
        return cls._instance


def get_langchain_agents_service() -> LangChainAgentsModel:
    """Get or create global LangChain agents service instance"""
    return _LangChainAgentsServiceSingleton.get_instance()
