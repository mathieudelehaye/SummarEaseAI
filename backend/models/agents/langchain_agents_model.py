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
    logging.warning("LangChain not available. Some LLM functionality will be limited.")

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
        self, user_queries: List[str], max_options: int = 5, top_n_articles: int = 1
    ) -> Dict[str, Any]:
        """
        Complete intelligent Wikipedia search using LangChain agents

        Args:
            user_queries: List of search queries to process
            max_options: Max search results per query
            top_n_articles: Number of top articles to select from all results

        Workflow:
        1. Query Enhancement Agent improves each search query
        2. Search Wikipedia with enhanced queries
        3. Collect all articles into single list
        4. Article Selection Agent picks the best N results from all
        5. Return comprehensive result with reasoning
        """
        logger.info(
            "🚀 Starting LangChain agent-powered search for %d queries",
            len(user_queries),
        )

        if not (self.query_enhancement_agent and self.article_selection_agent):
            return self._fallback_search(user_queries, max_options, top_n_articles)

        try:
            all_search_results = []
            enhancement_results = []

            # Step 1 & 2: Enhance each query and collect search results
            for query in user_queries:
                logger.info("🧠 Processing query: '%s'", query)

                # Enhance the query
                enhancement_result = self.query_enhancement_agent.enhance_query(query)
                enhanced_query = enhancement_result["enhanced_query"]
                enhancement_results.append(
                    {
                        "original_query": query,
                        "enhanced_query": enhanced_query,
                        "enhancement_result": enhancement_result,
                    }
                )

                logger.info("🧠 Query enhanced: '%s' → '%s'", query, enhanced_query)

                # Search with enhanced query
                search_results = wikipedia.search(enhanced_query, results=max_options)
                if not search_results:
                    # Fallback to original query
                    search_results = wikipedia.search(query, results=max_options)
                    logger.info(
                        "📚 Fallback search for '%s' found %d articles",
                        query,
                        len(search_results),
                    )
                else:
                    logger.info(
                        "📚 Enhanced search for '%s' found %d articles",
                        query,
                        len(search_results),
                    )

                # Add to combined results with metadata
                for result in search_results:
                    if result not in [item["title"] for item in all_search_results]:
                        all_search_results.append(
                            {
                                "title": result,
                                "source_query": query,
                                "enhanced_query": enhanced_query,
                            }
                        )

            logger.info(
                "📚 Combined search found %d unique articles from all queries",
                len(all_search_results),
            )

            if not all_search_results:
                return {
                    "error": "No Wikipedia articles found for any query",
                    "user_queries": user_queries,
                    "enhancement_results": enhancement_results,
                }

            # Step 3: Select top N articles from ALL results using ONE agent call
            article_titles = [item["title"] for item in all_search_results]
            combined_query = " OR ".join(
                user_queries[:3]
            )  # Combine queries for selection context

            selection_result = self.article_selection_agent.select_top_n_articles(
                combined_query, article_titles, top_n_articles
            )
            selected_titles = selection_result.get("selected_articles", [])
            logger.info(
                "🎯 Agent selected %d articles from %d candidates: %s",
                len(selected_titles),
                len(article_titles),
                selected_titles,
            )

            # Step 4: Fetch content for selected articles
            selected_articles = []
            for article_title in selected_titles:
                article_info = self._fetch_article_content(
                    article_title, article_titles
                )
                if "error" not in article_info:
                    # Add source query info
                    source_info = next(
                        (
                            item
                            for item in all_search_results
                            if item["title"] == article_title
                        ),
                        {},
                    )
                    article_info["source_query"] = source_info.get("source_query", "")
                    article_info["enhanced_query"] = source_info.get(
                        "enhanced_query", ""
                    )
                    selected_articles.append(article_info)

            return {
                "user_queries": user_queries,
                "enhancement_results": enhancement_results,
                "all_search_results": all_search_results,
                "selected_articles": selected_articles,
                "selection_result": selection_result,
                "agent_system": "agents_service_multi_query",
                "total_articles_considered": len(all_search_results),
                "articles_selected": len(selected_articles),
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
            return self._fallback_search(user_queries, max_options, top_n_articles)

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

    def _fallback_search(
        self, user_queries: List[str], max_options: int, top_n_articles: int = 1
    ) -> Dict[str, Any]:
        """Fallback search without LangChain agents"""
        try:
            all_search_results = []

            for query in user_queries:
                search_results = wikipedia.search(query, results=max_options)
                for result in search_results:
                    if result not in [item["title"] for item in all_search_results]:
                        all_search_results.append(
                            {
                                "title": result,
                                "source_query": query,
                                "enhanced_query": query,
                            }
                        )

            if not all_search_results:
                return {"error": "No Wikipedia articles found"}

            # Select top N articles (simple: take first N unique results)
            selected_titles = [
                item["title"] for item in all_search_results[:top_n_articles]
            ]
            selected_articles = []

            for title in selected_titles:
                article_info = self._fetch_article_content(title, selected_titles)
                if "error" not in article_info:
                    source_info = next(
                        (item for item in all_search_results if item["title"] == title),
                        {},
                    )
                    article_info["source_query"] = source_info.get("source_query", "")
                    selected_articles.append(article_info)

            return {
                "user_queries": user_queries,
                "all_search_results": all_search_results,
                "selected_articles": selected_articles,
                "agent_system": "fallback_multi_search",
                "total_articles_considered": len(all_search_results),
                "articles_selected": len(selected_articles),
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
