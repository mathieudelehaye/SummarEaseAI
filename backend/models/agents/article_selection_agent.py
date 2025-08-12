"""
Article Selection Agent Module

This module provides an intelligent article selection agent using LangChain
to choose the most relevant Wikipedia articles based on user queries.

The ArticleSelectionAgent uses a conversational agent with memory to analyze
article options and select the best match for a given query. It includes
fallback mechanisms for error handling and confidence scoring.

Classes:
    ArticleSelectionAgent: LangChain-based agent for intelligent article selection

Dependencies:
    - langchain: For agent functionality and tools
    - langchain.tools: For Wikipedia search tools
    - langchain.memory: For conversation memory
    - langchain.agents: For agent initialization and types
"""

import logging
from typing import Any, Dict, List

try:
    from langchain.agents import AgentType, initialize_agent
    from langchain.memory import ConversationBufferMemory
    from langchain.tools import Tool

    from ..wikipedia.wikipedia_search_tool import WikipediaSearchTool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)


class ArticleSelectionAgent:
    """LangChain agent for article selection"""

    def __init__(self, llm):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available")
        self.llm = llm

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

        # pylint: disable=C0301
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
            agent_kwargs={
                "prefix": """You are an expert Article Selection Agent for Wikipedia.

Your mission: Given a user query and multiple Wikipedia article options, select the BEST article that answers their question.

Selection criteria:
1. Primary relevance: Does the article directly answer the user's question?
2. Comprehensiveness: Does it provide thorough information?
3. Avoid disambiguation pages unless the user specifically wants them
4. Avoid "List of" articles unless the user wants a list
5. Prefer main topic articles over sub-topics

Use your tools to preview articles and evaluate their relevance before deciding.""",
                "suffix": """Begin!

Question: {input}
{agent_scratchpad}""",
                "input_variables": ["input", "agent_scratchpad"],
            },
        )
        # pylint: enable=C0301

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

            # pylint: disable=C0301
            agent_input = (
                f'Select the BEST Wikipedia article that answers this query: "{user_query}"\n\n'
                f"Available articles:\n{options_str}\n\n"
                "Respond with ONLY the exact title of the best article."
            )
            # pylint: enable=C0301

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

    def select_top_n_articles(
        self, user_query: str, article_options: List[str], top_n: int = 3
    ) -> Dict[str, Any]:
        """Select top N best articles using LangChain agent"""
        if not article_options:
            return {"selected_articles": [], "method": "no_options"}

        if len(article_options) <= top_n:
            return {
                "selected_articles": article_options,
                "method": "all_available",
                "confidence": 0.8,
            }

        try:
            options_str = "\n".join(
                [f"{i+1}. {title}" for i, title in enumerate(article_options)]
            )

            # pylint: disable=C0301
            agent_input = (
                f'Select the TOP {top_n} Wikipedia articles that best answer this query: "{user_query}"\n\n'
                f"Available articles:\n{options_str}\n\n"
                f"Instructions:\n"
                f"1. Use the get_article_preview tool to examine promising articles if needed\n"
                f"2. Consider relevance, comprehensiveness, and quality\n"
                f"3. Avoid disambiguation pages unless specifically relevant\n"
                f"4. Prefer main topic articles over sub-topics\n\n"
                f"Respond with EXACTLY {top_n} article titles, each on a separate line, "
                "in order of relevance (best first). Use the exact titles from the list above."
            )
            # pylint: enable=C0301

            result = self.agent.run(agent_input)
            selected_articles = self._extract_multiple_articles_from_response(
                result, article_options, top_n
            )

            return {
                "selected_articles": selected_articles,
                "method": "langchain_agent_top_n",
                "confidence": 0.9,
                "requested_count": top_n,
                "actual_count": len(selected_articles),
            }

        except (
            ValueError,
            KeyError,
            AttributeError,
            ImportError,
            ConnectionError,
        ) as e:
            logger.error("Top-N article selection failed: %s", str(e))
            return {
                "selected_articles": article_options[:top_n],
                "method": "fallback_first_n",
                "confidence": 0.5,
                "requested_count": top_n,
                "actual_count": min(len(article_options), top_n),
            }

    def _extract_article_from_response(self, response: str, options: List[str]) -> str:
        """Extract selected article from agent response"""
        response_clean = response.strip().lower()

        for option in options:
            if option.lower() in response_clean:
                return option

        return options[0] if options else ""

    def _extract_multiple_articles_from_response(
        self, response: str, options: List[str], max_count: int
    ) -> List[str]:
        """Extract multiple selected articles from agent response"""
        response_lines = [line.strip() for line in response.split("\n") if line.strip()]
        selected = []

        # Try to match each line to an article title
        for line in response_lines:
            line_clean = line.lower()
            # Remove numbering if present (e.g., "1. Article Title" -> "Article Title")
            line_clean = line_clean.split(".", 1)[-1].strip()

            best_match = None
            best_score = 0

            for option in options:
                if option not in selected:  # Avoid duplicates
                    option_clean = option.lower()
                    # Check for exact match or substring match
                    if (
                        option_clean == line_clean
                        or option_clean in line_clean
                        or line_clean in option_clean
                    ):
                        # Prefer exact matches
                        score = 2 if option_clean == line_clean else 1
                        if score > best_score:
                            best_match = option
                            best_score = score

            if best_match:
                selected.append(best_match)

            if len(selected) >= max_count:
                break

        # Fill remaining slots with unselected options if needed
        if len(selected) < max_count:
            for option in options:
                if option not in selected:
                    selected.append(option)
                    if len(selected) >= max_count:
                        break

        return selected[:max_count]
