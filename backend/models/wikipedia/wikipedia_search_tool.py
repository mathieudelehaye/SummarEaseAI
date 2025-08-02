"""
Wikipedia Search Tool Module

This module provides Wikipedia search functionality as tools for LangChain agents.
It includes methods for searching Wikipedia articles and retrieving article previews.

The WikipediaSearchTool class provides static methods that can be used as tools
by LangChain agents to interact with Wikipedia's API. It includes error handling
for various Wikipedia API exceptions and connection issues.

Classes:
    WikipediaSearchTool: Collection of Wikipedia search and preview tools

Dependencies:
    - wikipedia: For Wikipedia API access and search functionality
    - typing: For type hints

Methods:
    - search_wikipedia: Search Wikipedia and return results summary
    - get_article_preview: Get preview of a specific Wikipedia article
"""

import logging

try:
    import wikipedia

    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False
    # Fallback for type checking
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        import wikipedia

logger = logging.getLogger(__name__)


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
