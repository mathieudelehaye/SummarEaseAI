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

    from ..wikipedia.wikipedia_search_tool import WikipediaSearchTool

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
                func=lambda query: WikipediaSearchTool.search_wikipedia(
                    query, format_style="simple"
                ),
                description="Search Wikipedia for articles matching a query. "
                "Returns list of article titles found.",
            ),
            Tool(
                name="wikipedia_suggest",
                func=WikipediaSearchTool.suggest_wikipedia,
                description="Get Wikipedia search suggestions for a query. "
                "Returns alternative search terms.",
            ),
            Tool(
                name="get_article_preview",
                func=lambda title: WikipediaSearchTool.get_article_preview(
                    title, summary_length=200, include_url=False
                ),
                description="Get a preview of a Wikipedia article by title to check "
                "its actual content and relevance.",
            ),
        ]

        memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

        # pylint: disable=C0301
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
                "input_variables": ["input", "agent_scratchpad"],
            },
        )
        # pylint: enable=C0301

    def validate_query(self, user_query: str) -> Dict[str, Any]:
        """Validate if query is likely to find relevant Wikipedia articles"""
        try:
            # pylint: disable=C0301
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
            # pylint: enable=C0301

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

        if "very unlikely" in response_lower:
            return "Very unlikely"

        # Default to likely if unclear
        logger.warning(
            "Unclear validation response: %s, defaulting to 'Very likely'", response
        )
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
