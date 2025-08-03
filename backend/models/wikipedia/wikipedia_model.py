"""
Wikipedia Service
Moved from utils/wikipedia_fetcher.py to proper services layer
Handles all Wikipedia content fetching and processing
"""

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import wikipedia
import wikipediaapi

from ..intent.intent_data import INTENT_ENHANCEMENTS

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

logger = logging.getLogger(__name__)


class WikipediaModel:
    """
    Handles all Wikipedia content fetching and processing
    Pure business logic for Wikipedia operations
    """

    def __init__(self):
        # Set user agent for API compliance
        try:
            self.wiki_api = wikipediaapi.Wikipedia(
                language="en",
                user_agent="SummarEaseAI/1.0 (https://github.com/SummarEaseAI) Python/WikipediaAPI",
            )
        except Exception as e:
            logger.error("Failed to initialize Wikipedia API: %s", str(e))
            self.wiki_api = None

        # Set user agent for wikipedia library
        try:
            wikipedia.set_user_agent(
                "SummarEaseAI/1.0 (https://github.com/SummarEaseAI)"
            )
        except Exception as e:
            logger.error("Failed to set Wikipedia user agent: %s", str(e))

        logger.info("✅ Wikipedia Service initialized")

    def preprocess_historical_query(self, query: str) -> Tuple[str, bool]:
        """
        Preprocess queries to better handle historical date searches

        Returns:
            tuple: (processed_query, was_converted)
        """
        # Common historical date patterns and their likely topics
        historical_events = {
            r"july\s+20.*1969|20.*july.*1969": "Apollo 11",
            r"july\s+16.*1969|16.*july.*1969": "Apollo 11 launch",
            r"neil\s+armstrong.*moon|moon.*neil\s+armstrong": "Apollo 11",
            r"buzz\s+aldrin.*moon|moon.*buzz\s+aldrin": "Apollo 11",
            r"december\s+7.*1941|7.*december.*1941": "Pearl Harbor attack December 7 1941",
            r"november\s+22.*1963|22.*november.*1963": (
                "John F Kennedy assassination November 22 1963"
            ),
            r"september\s+11.*2001|11.*september.*2001": "September 11 attacks 2001",
            r"april\s+14.*1865|14.*april.*1865": (
                "Abraham Lincoln assassination April 14 1865"
            ),
        }

        query_lower = query.lower()
        for pattern, replacement in historical_events.items():
            if re.search(pattern, query_lower):
                logger.info(
                    "Historical query pattern detected: '%s' -> '%s'",
                    query,
                    replacement,
                )
                return replacement, True

        return query, False

    def sanitize_wikipedia_content(self, content: str) -> str:
        """
        Sanitize Wikipedia content to prevent format string errors
        Removes or replaces characters that could cause issues with LangChain templates
        """
        if not content:
            return ""

        # Replace curly braces that could cause format code errors
        sanitized = str(content).replace("{", "(").replace("}", ")")

        # Log if we made changes
        if "{" in content or "}" in content:
            logger.info(
                "🔧 Sanitized Wikipedia content: removed %d opening and %d closing curly braces",
                content.count("{"),
                content.count("}"),
            )

        return sanitized

    def search_and_fetch_article_info(
        self, query: str, max_results: int = 1
    ) -> Optional[Dict[str, str]]:
        """
        Search Wikipedia and fetch article with complete information

        Returns:
            Dictionary with content, title, url, and summary, or None if not found
        """
        try:
            # Preprocess historical queries
            processed_query, was_converted = self.preprocess_historical_query(query)

            # Log the exact search parameters being sent to Wikipedia
            logger.info("=" * 80)
            logger.info("🔍 WIKIPEDIA API SEARCH REQUEST")
            logger.info("=" * 80)
            logger.info("📝 Original query: '%s'", query)
            logger.info("📝 Processed query: '%s'", processed_query)
            logger.info("🔄 Query was converted: %s", was_converted)
            logger.info("📊 Max results requested: %d", max_results + 2)
            logger.info(
                "🌐 Wikipedia search API call: wikipedia.search('%s', results=%d)",
                processed_query,
                max_results + 2,
            )

            # Use wikipedia library for better search functionality
            try:
                search_results = wikipedia.search(
                    processed_query, results=max_results + 2
                )
            except Exception as search_error:
                logger.error("Error in Wikipedia search: %s", str(search_error))
                return None

            logger.info(
                "✅ Wikipedia API returned %d search results:", len(search_results)
            )
            for i, result in enumerate(search_results):
                logger.info("   %d. '%s'", i + 1, result)

            for result in search_results:
                try:
                    logger.info("🔄 Attempting to fetch page: '%s'", result)
                    logger.info(
                        "🌐 Wikipedia page API call: wikipedia.page('%s')", result
                    )

                    page = wikipedia.page(result)

                    logger.info("✅ Successfully fetched page!")
                    logger.info("📄 Found and fetched article: %s", result)
                    logger.info("🔗 Search result URL: %s", page.url)
                    logger.info("📝 Search result page title: %s", page.title)
                    content_preview = (
                        page.content[:500] + "..."
                        if len(page.content) > 500
                        else page.content
                    )
                    logger.info(
                        "📄 Search result content starts with: %s", content_preview
                    )
                    logger.info("=" * 80)

                    # Sanitize content before returning
                    sanitized_content = self.sanitize_wikipedia_content(page.content)

                    return {
                        "content": sanitized_content,
                        "title": page.title,
                        "url": page.url,
                        "summary": page.summary,
                    }
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation pages by taking the first option
                    logger.info("⚠️  Disambiguation page encountered for '%s'", result)
                    logger.info(
                        "📋 Available options: %s...", e.options[:5]
                    )  # Show first 5 options
                    try:
                        logger.info(
                            "🔄 Trying first disambiguation option: '%s'", e.options[0]
                        )
                        logger.info(
                            "🌐 Wikipedia page API call: wikipedia.page('%s')",
                            e.options[0],
                        )

                        page = wikipedia.page(e.options[0])
                        logger.info("✅ Resolved disambiguation to: %s", e.options[0])
                        logger.info("=" * 80)

                        # Sanitize content before returning
                        sanitized_content = self.sanitize_wikipedia_content(
                            page.content
                        )

                        return {
                            "content": sanitized_content,
                            "title": page.title,
                            "url": page.url,
                            "summary": page.summary,
                        }
                    except (
                        wikipedia.PageError,
                        wikipedia.DisambiguationError,
                        ConnectionError,
                        ValueError,
                    ) as disambiguation_error:
                        logger.error(
                            "❌ Failed to resolve disambiguation: %s",
                            str(disambiguation_error),
                        )
                        continue
                except (wikipedia.PageError, ConnectionError, ValueError) as page_error:
                    logger.error(
                        "❌ Error fetching page '%s': %s", result, str(page_error)
                    )
                    continue

            logger.warning("❌ No suitable article found for query: '%s'", query)
            logger.info("=" * 80)
            return None

        except Exception as e:
            logger.error("❌ Error searching for article '%s': %s", query, str(e))
            logger.info("=" * 80)
            return None

    def enhance_query_with_intent(
        self, query: str, intent: str, confidence: float = 0.5
    ) -> str:
        """
        Enhance search query based on detected intent to get better Wikipedia results

        Args:
            query: Original user query
            intent: Detected intent category
            confidence: Confidence score of intent prediction

        Returns:
            Enhanced query string for better Wikipedia search
        """
        # Only enhance if we have high confidence in the intent
        if confidence < 0.4:
            logger.info(
                "🚫 Not enhancing query - confidence %.3f below threshold 0.4",
                confidence,
            )
            return query

        # Intent-based query enhancement patterns
        intent_enhancements = INTENT_ENHANCEMENTS

        if intent in intent_enhancements:
            enhancement = intent_enhancements[intent]

            # Check if query already contains intent-related keywords
            query_lower = query.lower()
            has_intent_keywords = any(
                keyword in query_lower for keyword in enhancement["keywords"]
            )

            # For Science: be extra careful with specific terms
            if intent == "Science":
                # Don't enhance if already contains specific science terms
                specific_science_terms = [
                    "quantum",
                    "physics",
                    "chemistry",
                    "biology",
                    "mechanics",
                    "thermodynamics",
                ]
                if any(term in query_lower for term in specific_science_terms):
                    logger.info(
                        "🧪 Science query already contains specific terms - not enhancing: '%s'",
                        query,
                    )
                    return query

            # Only enhance if doesn't already have intent keywords
            if not has_intent_keywords:
                # Use the most relevant suffix
                enhanced_query = f"{query} {enhancement['suffixes'][0]}"
                logger.info(
                    "✨ Enhanced query with intent '%s': '%s' -> '%s'",
                    intent,
                    query,
                    enhanced_query,
                )
                return enhanced_query
            logger.info(
                "✅ Query already contains %s keywords - not enhancing: '%s'",
                intent.lower(),
                query,
            )

        return query

    def search_wikipedia_basic(
        self, query: str, max_results: int = 3
    ) -> Dict[str, any]:
        """
        Basic Wikipedia search functionality moved from controller
        Returns search results with title, summary, and URL
        """

        try:
            # Search for articles
            search_results = wikipedia.search(query, results=max_results)
            if not search_results:
                return {
                    "error": "No Wikipedia articles found",
                    "query": query,
                    "summary": None,
                }

            # Get the first article
            article_title = search_results[0]
            page = wikipedia.page(article_title)

            # Get summary (first few sentences)
            summary = wikipedia.summary(article_title, sentences=5)

            return {
                "query": query,
                "title": article_title,
                "summary": summary,
                "url": page.url,
                "status": "success",
            }

        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation by taking the first option
            try:
                page = wikipedia.page(e.options[0])
                summary = wikipedia.summary(e.options[0], sentences=5)
                return {
                    "query": query,
                    "title": e.options[0],
                    "summary": summary,
                    "url": page.url,
                    "status": "success",
                }
            except (wikipedia.PageError, KeyError, ValueError) as inner_e:
                logger.error("Error handling disambiguation: %s", inner_e)
                return {
                    "error": f"Disambiguation error: {str(inner_e)}",
                    "query": query,
                    "summary": None,
                }
        except (ConnectionError, TimeoutError) as e:
            logger.error("Error in Wikipedia search: %s", e)
            return {"error": str(e), "query": query, "summary": None}


class _WikipediaModelSingleton:
    """Singleton wrapper for WikipediaModel"""

    _instance = None

    @classmethod
    def get_instance(cls) -> WikipediaModel:
        """Get or create the singleton service instance"""
        if cls._instance is None:
            cls._instance = WikipediaModel()
        return cls._instance


def get_wikipedia_service() -> WikipediaModel:
    """Get or create global Wikipedia service instance"""
    return _WikipediaModelSingleton.get_instance()
