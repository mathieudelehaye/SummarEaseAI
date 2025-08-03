"""
Query Enhancement Agent Module

This module provides an intelligent query enhancement agent using LangChain
to transform user queries into optimized Wikipedia search terms.

The QueryEnhancementAgent uses a conversational agent with memory to analyze
user queries and generate more targeted search terms that will yield better
Wikipedia article results. It includes fallback mechanisms for error handling
and confidence scoring.

Classes:
    QueryEnhancementAgent: LangChain-based agent for query optimization

Dependencies:
    - langchain: For agent functionality and tools
    - langchain.tools: For Wikipedia search tools
    - langchain.memory: For conversation memory
    - langchain.agents: For agent initialization and types
"""

import logging
from typing import Any, Dict

try:
    from langchain.agents import AgentType, initialize_agent
    from langchain.memory import ConversationBufferMemory
    from langchain.tools import Tool

    from ..wikipedia.wikipedia_search_tool import WikipediaSearchTool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)


class QueryEnhancementAgent:
    """LangChain agent for query enhancement"""

    def __init__(self, llm):
        self.llm = llm
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available")

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

        # pylint: disable=C0301
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
            agent_kwargs={
                "prefix": """You are an expert Query Enhancement Agent for Wikipedia search optimization.

Your mission: Transform user queries into optimized Wikipedia search terms that find the most relevant articles.

Key principles:
1. Understand the user's true intent behind their question
2. Remove unnecessary question words ("who was", "what is", "tell me about")  
3. Target actual Wikipedia article titles
4. Make queries more specific and Wikipedia-friendly
5. Test your enhanced queries to ensure they work

Always test your enhanced query before giving the final answer.""",
                "suffix": """Begin!

Question: {input}
{agent_scratchpad}""",
                "input_variables": ["input", "agent_scratchpad"],
            },
        )
        # pylint: enable=C0301

    def enhance_query(self, original_query: str) -> Dict[str, Any]:
        """Enhance query using LangChain agent"""
        try:
            # pylint: disable=C0301
            agent_input = (
                f"Transform this user query into optimized Wikipedia search terms: "
                f'"{original_query}"\n\n'
                "Create a more targeted Wikipedia search query that will find the best "
                "articles to answer this question.\n"
                "Respond with ONLY the enhanced query."
            )
            # pylint: enable=C0301

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

        # Remove trailing question mark if present
        if enhanced.endswith("?"):
            enhanced = enhanced[:-1].strip()

        return {
            "original_query": query,
            "enhanced_query": enhanced,
            "enhancement_method": "rule_based_fallback",
            "confidence": 0.6,
        }
