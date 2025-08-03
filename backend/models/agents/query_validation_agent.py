"""
Query Validation Agent Module

This module provides an intelligent query validation agent using LangChain
to validate Wikipedia query viability using search and suggestion functions.

The QueryValidationAgent uses a conversational agent with memory to analyze
user queries and determine if they are likely to find relevant Wikipedia articles.
It includes Wikipedia search and suggest tools for RAG capabilities and confidence scoring.

Classes:
    QueryValidationAgent: LangChain-based agent for query validation

Dependencies:
    - langchain: For agent functionality and tools
    - langchain.tools: For Wikipedia search tools
    - langchain.memory: For conversation memory
    - langchain.agents: For agent initialization and types
    - wikipedia: For search and suggest functions
"""

import logging
from typing import Any, Dict

try:
    from langchain.agents import AgentType, initialize_agent
    from langchain.memory import ConversationBufferMemory
    from langchain.tools import Tool
    import wikipedia

    LANGCHAIN_AVAILABLE = True
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    WIKIPEDIA_AVAILABLE = False

logger = logging.getLogger(__name__)


class QueryValidationAgent:
    """LangChain agent for Wikipedia query validation"""

    def __init__(self, llm):
        self.llm = llm
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available")
        if not WIKIPEDIA_AVAILABLE:
            raise ImportError("Wikipedia not available")

        self._initialize_agent()

    def _initialize_agent(self):
        """Initialize the LangChain agent with Wikipedia search and suggest tools"""
        tools = [
            Tool(
                name="wikipedia_search",
                func=self._wikipedia_search_wrapper,
                description="Search Wikipedia for articles matching a query. Returns list of article titles found.",
            ),
            Tool(
                name="wikipedia_suggest",
                func=self._wikipedia_suggest_wrapper,
                description="Get Wikipedia search suggestions for a query. Returns alternative search terms.",
            ),
            Tool(
                name="get_article_preview",
                func=self._article_preview_wrapper,
                description="Get a preview of a Wikipedia article by title to check its actual content and relevance.",
            ),
        ]

        memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
            agent_kwargs={
                "prefix": """You are an expert Query Validation Agent for Wikipedia search viability assessment.

Your mission: Determine if a user query is likely to find RELEVANT Wikipedia articles by thoroughly checking content relevance.

STRICT Validation criteria:
1. Use wikipedia_search to find potential articles
2. Use get_article_preview to check if found articles are actually RELEVANT to the query
3. Check if article content genuinely relates to what the user is asking about
4. Use wikipedia_suggest only as supplementary evidence
5. Return 'Very likely' ONLY if articles contain relevant information about the query topic
6. Return 'Very unlikely' if search results exist but are clearly unrelated to the query

CRITICAL: Finding articles with unrelated content should result in 'Very unlikely'. Quality over quantity.""",
                "suffix": """Begin!

Question: {input}
{agent_scratchpad}""",
                "input_variables": ["input", "agent_scratchpad"]
            }
        )

    # TODO: move wrapper to wikipedia_search_tool.py and merge with search_wikipedia
    def _wikipedia_search_wrapper(self, query: str) -> str:
        """Wrapper for wikipedia.search function"""
        try:
            results = wikipedia.search(query, results=5)
            if not results:
                return f"No Wikipedia articles found for: {query}"
            
            return f"Found {len(results)} articles: {', '.join(results)}"
        except (
            wikipedia.PageError,
            wikipedia.DisambiguationError,
            ConnectionError,
            ValueError,
        ) as e:
            return f"Wikipedia search error: {str(e)}"

    # TODO: move wrapper to wikipedia_search_tool.py
    def _wikipedia_suggest_wrapper(self, query: str) -> str:
        """Wrapper for wikipedia.suggest function"""
        try:
            suggestion = wikipedia.suggest(query)
            if suggestion:
                return f"Wikipedia suggests: {suggestion}"
            else:
                return f"No suggestions found for: {query}"
        except (ConnectionError, ValueError) as e:
            return f"Wikipedia suggest error: {str(e)}"

    # TODO: move wrapper to wikipedia_search_tool.py and merge with get_article_preview
    def _article_preview_wrapper(self, title: str) -> str:
        """Wrapper for wikipedia article preview to check content relevance"""
        try:
            page = wikipedia.page(title)
            summary = (
                page.summary[:200] + "..." if len(page.summary) > 200 else page.summary
            )
            return f"Article: {page.title}\nSummary: {summary}"
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Disambiguation page. Options: {e.options[:3]}"
        except (wikipedia.PageError, ConnectionError, ValueError, KeyError) as e:
            return f"Error getting article preview: {str(e)}"

    def validate_query(self, user_query: str) -> Dict[str, Any]:
        """Validate if query is likely to find relevant Wikipedia articles"""
        try:
            agent_input = (
                f'Determine if this query is likely to find RELEVANT Wikipedia articles: "{user_query}"\n\n'
                "IMPORTANT: You must check actual content relevance, not just existence of search results.\n\n"
                "Process:\n"
                "1. Use wikipedia_search to find potential articles\n"
                "2. Use get_article_preview to examine the actual content of found articles\n"
                "3. Evaluate if the article content is genuinely related to the query topic\n"
                "4. Consider wikipedia_suggest for alternative terms if needed\n\n"
                "Based on your content analysis, respond with ONLY one of these exact phrases:\n"
                "- 'Very likely' if you find articles with content genuinely relevant to the query\n"
                "- 'Very unlikely' if no articles contain relevant content (even if unrelated articles exist)"
            )

            result = self.agent.run(agent_input)
            viability = self._extract_viability_from_response(result)

            return {
                "query": user_query,
                "viability": viability,
                "validation_method": "langchain_agent_with_rag",
                "confidence": 0.95,
                "agent_response": result,
            }

        except (
            ValueError,
            KeyError,
            AttributeError,
            ImportError,
            ConnectionError,
        ) as e:
            logger.error("Query validation failed: %s", str(e))
            return self._fallback_validation(user_query)

    def _extract_viability_from_response(self, response: str) -> str:
        """Extract viability assessment from agent response"""
        response_lower = response.strip().lower()
        
        if "very likely" in response_lower:
            return "Very likely"
        elif "very unlikely" in response_lower:
            return "Very unlikely"
        else:
            # Default to likely if unclear
            logger.warning(f"Unclear validation response: {response}, defaulting to 'Very likely'")
            return "Very likely"

    def _fallback_validation(self, query: str) -> Dict[str, Any]:
        """Fallback validation when agent fails"""
        # Simple heuristic fallback
        if len(query.strip()) < 3:
            viability = "Very unlikely"
        else:
            viability = "Very likely"

        return {
            "query": query,
            "viability": viability,
            "validation_method": "fallback_heuristic",
            "confidence": 0.5,
            "agent_response": "Agent failed, using fallback",
        }