"""
LangChain Agents Service
Moved from utils/langchain_agents.py to proper services layer
Specialized LangChain agent orchestration
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import wikipedia

from .llm_client import get_llm_client

# Setup agent service environment
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

logger = logging.getLogger(__name__)

# Initialize LangChain agent components
try:
    from langchain.agents import AgentType, Tool, initialize_agent
    from langchain.memory import ConversationBufferMemory

    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain agents available for orchestration")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("⚠️ LangChain not available - agents will be disabled")
    # Define dummy classes to avoid name errors
    AgentType = None
    initialize_agent = None
    Tool = None
    ConversationBufferMemory = None


class WikipediaSearchTool:
    """Wikipedia search tools for LangChain agents"""

    @staticmethod
    def search_wikipedia(query: str) -> str:
        """Search Wikipedia and return results summary."""
        try:
            search_results = wikipedia.search(query, results=5)
            if not search_results:
                return f"No Wikipedia articles found for: {query}"

            result_summary = (
                f"Found {len(search_results)} Wikipedia articles for '{query}':\n"
            )
            for i, title in enumerate(search_results, 1):
                result_summary += f"{i}. {title}\n"

            return result_summary
        except (
            wikipedia.PageError,
            wikipedia.DisambiguationError,
            ConnectionError,
            ValueError,
        ) as e:
            return f"Wikipedia search error: {str(e)}"

    @staticmethod
    def get_article_preview(title: str) -> str:
        """Get a preview of a Wikipedia article."""
        try:
            page = wikipedia.page(title)
            summary = (
                page.summary[:300] + "..." if len(page.summary) > 300 else page.summary
            )
            return f"Article: {page.title}\nURL: {page.url}\nSummary: {summary}"
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Disambiguation page. Options: {e.options[:3]}"
        except (wikipedia.PageError, ConnectionError, ValueError, KeyError) as e:
            return f"Error getting article preview: {str(e)}"


class LangChainAgentsService:
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
                self.query_enhancement_agent = QueryEnhancementAgent()
                self.article_selection_agent = ArticleSelectionAgent()
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
                "agent_system": "langchain_agents",
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


class QueryEnhancementAgent:
    """LangChain agent for query enhancement"""

    def __init__(self):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available")

        self.llm = get_llm_client()
        self._initialize_agent()

    def _initialize_agent(self):
        """Initialize the LangChain agent with tools"""
        tools = [
            Tool(
                name="wikipedia_search",
                func=WikipediaSearchTool.search_wikipedia,
                description="Search Wikipedia for articles. Returns list of found article titles.",
            )
        ]

        memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
        )

    def enhance_query(self, original_query: str) -> Dict[str, Any]:
        """Enhance query using LangChain agent"""
        try:
            agent_input = (
                f"Transform this user query into optimized Wikipedia search terms: "
                f'"{original_query}"\n\n'
                "Create a more targeted Wikipedia search query that will find the best "
                "articles to answer this question.\n"
                "Respond with ONLY the enhanced query."
            )

            result = self.agent.run(agent_input)
            enhanced_query = self._extract_query_from_response(result, original_query)

            return {
                "original_query": original_query,
                "enhanced_query": enhanced_query,
                "enhancement_method": "langchain_agent",
                "confidence": 0.9,
            }

        except (
            ValueError,
            KeyError,
            AttributeError,
            ImportError,
            ConnectionError,
        ) as e:
            logger.error("Query enhancement failed: %s", str(e))
            return self._fallback_enhancement(original_query)

    def _extract_query_from_response(self, response: str, original_query: str) -> str:
        """Extract enhanced query from agent response"""
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith(("Thought:", "Action:", "Observation:")):
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]
                if len(line) > 0 and len(line) < 100:
                    return line
        return original_query

    def _fallback_enhancement(self, query: str) -> Dict[str, Any]:
        """Fallback enhancement when agent fails"""
        enhanced = query
        query_lower = query.lower()

        if query_lower.startswith("who were "):
            enhanced = query.replace("Who were ", "").replace("who were ", "").strip()
        elif query_lower.startswith("who was "):
            enhanced = query.replace("Who was ", "").replace("who was ", "").strip()
        elif query_lower.startswith("what is "):
            enhanced = query.replace("What is ", "").replace("what is ", "").strip()

        return {
            "original_query": query,
            "enhanced_query": enhanced,
            "enhancement_method": "rule_based_fallback",
            "confidence": 0.6,
        }


class ArticleSelectionAgent:
    """LangChain agent for article selection"""

    def __init__(self):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available")

        self.llm = get_llm_client()
        self._initialize_agent()

    def _initialize_agent(self):
        """Initialize article selection agent"""
        tools = [
            Tool(
                name="get_article_preview",
                func=WikipediaSearchTool.get_article_preview,
                description="Get a preview of a Wikipedia article by title.",
            )
        ]

        memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
        )

    def select_best_article(
        self, user_query: str, article_options: List[str]
    ) -> Dict[str, Any]:
        """Select best article using LangChain agent"""
        if not article_options:
            return {"selected_article": None, "method": "no_options"}

        try:
            options_str = "\n".join(
                [f"{i+1}. {title}" for i, title in enumerate(article_options)]
            )

            agent_input = (
                f'Select the BEST Wikipedia article that answers this query: "{user_query}"\n\n'
                f"Available articles:\n{options_str}\n\n"
                "Respond with ONLY the exact title of the best article."
            )

            result = self.agent.run(agent_input)
            selected_article = self._extract_article_from_response(
                result, article_options
            )

            return {
                "selected_article": selected_article,
                "method": "langchain_agent",
                "confidence": 0.9,
            }

        except (
            ValueError,
            KeyError,
            AttributeError,
            ImportError,
            ConnectionError,
        ) as e:
            logger.error("Article selection failed: %s", str(e))
            return {
                "selected_article": article_options[0],
                "method": "fallback_first",
                "confidence": 0.5,
            }

    def _extract_article_from_response(self, response: str, options: List[str]) -> str:
        """Extract selected article from agent response"""
        response_clean = response.strip().lower()

        for option in options:
            if option.lower() in response_clean:
                return option

        return options[0] if options else ""


class _LangChainAgentsServiceSingleton:
    """Singleton wrapper for LangChainAgentsService"""

    _instance = None

    @classmethod
    def get_instance(cls) -> LangChainAgentsService:
        """Get or create the singleton service instance"""
        if cls._instance is None:
            cls._instance = LangChainAgentsService()
        return cls._instance


def get_langchain_agents_service() -> LangChainAgentsService:
    """Get or create global LangChain agents service instance"""
    return _LangChainAgentsServiceSingleton.get_instance()
