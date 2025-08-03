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
    def search_wikipedia(
        query: str, results: int = 5, format_style: str = "detailed"
    ) -> str:
        """Search Wikipedia and return results summary.

        Args:
            query: Search query string
            results: Number of results to return (default 5)
            format_style: "detailed" for numbered list, "simple" for comma-separated
        """
        try:
            search_results = wikipedia.search(query, results=results)
            if not search_results:
                return f"No Wikipedia articles found for: {query}"

            if format_style == "simple":
                return (
                    f"Found {len(search_results)} articles: {', '.join(search_results)}"
                )

            # Default detailed format
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
    def get_article_preview(
        title: str, summary_length: int = 300, include_url: bool = True
    ) -> str:
        """Get a preview of a Wikipedia article.

        Args:
            title: Article title to preview
            summary_length: Maximum length of summary (default 300)
            include_url: Whether to include article URL (default True)
        """
        try:
            page = wikipedia.page(title)
            summary = (
                page.summary[:summary_length] + "..."
                if len(page.summary) > summary_length
                else page.summary
            )

            if include_url:
                return f"Article: {page.title}\nURL: {page.url}\nSummary: {summary}"

            return f"Article: {page.title}\nSummary: {summary}"
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Disambiguation page. Options: {e.options[:3]}"
        except (wikipedia.PageError, ConnectionError, ValueError, KeyError) as e:
            return f"Error getting article preview: {str(e)}"

    @staticmethod
    def suggest_wikipedia(query: str) -> str:
        """Get Wikipedia search suggestions for a query.

        Args:
            query: Query string to get suggestions for
        """
        try:
            suggestion = wikipedia.suggest(query)
            if suggestion:
                return f"Wikipedia suggests: {suggestion}"

            return f"No suggestions found for: {query}"
        except (ConnectionError, ValueError) as e:
            return f"Wikipedia suggest error: {str(e)}"
