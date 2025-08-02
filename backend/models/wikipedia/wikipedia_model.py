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

    def fetch_article(self, topic: str) -> Optional[str]:
        """
        Fetch Wikipedia article content by topic/title
        """
        try:
            # Preprocess historical queries
            processed_topic, _ = self.preprocess_historical_query(topic)

            # Check if wiki_api is available
            if self.wiki_api is None:
                logger.error("Wikipedia API not initialized")
                return None

            page = self.wiki_api.page(processed_topic)

            if page.exists():
                logger.info("Successfully fetched article: %s", processed_topic)
                logger.info(
                    "🔗 Article URL: https://en.wikipedia.org/wiki/%s",
                    processed_topic.replace(" ", "_"),
                )
                logger.info("📝 Article title from Wikipedia: %s", page.title)

                # Sanitize content before returning
                raw_content = page.text
                sanitized_content = self.sanitize_wikipedia_content(raw_content)

                article_preview = (
                    sanitized_content[:500] + "..."
                    if len(sanitized_content) > 500
                    else sanitized_content
                )
                logger.info("📄 Article content starts with: %s", article_preview)
                return sanitized_content
            # Try searching for the topic if direct page doesn't exist
            logger.info(
                "Direct page not found for '%s', trying search...", processed_topic
            )
            return self.search_and_fetch_article(processed_topic)

        except Exception as e:
            logger.error("Error fetching article '%s': %s", topic, str(e))
            return None

    def search_and_fetch_article(
        self, query: str, max_results: int = 1
    ) -> Optional[str]:
        """
        Search Wikipedia and fetch the first relevant article
        """
        try:
            # Preprocess historical queries
            processed_query, _ = self.preprocess_historical_query(query)

            # Use wikipedia library for better search functionality
            search_results = wikipedia.search(processed_query, results=max_results + 2)

            for result in search_results:
                try:
                    page = wikipedia.page(result)
                    logger.info("Found and fetched article: %s", result)
                    logger.info(
                        "🔗 Search result URL: https://en.wikipedia.org/wiki/%s",
                        result.replace(" ", "_"),
                    )
                    logger.info("📝 Search result page title: %s", page.title)

                    # Sanitize content before returning
                    raw_content = page.content
                    sanitized_content = self.sanitize_wikipedia_content(raw_content)

                    content_preview = (
                        sanitized_content[:500] + "..."
                        if len(sanitized_content) > 500
                        else sanitized_content
                    )
                    logger.info(
                        "📄 Search result content starts with: %s", content_preview
                    )
                    return sanitized_content
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation pages by taking the first option
                    try:
                        page = wikipedia.page(e.options[0])
                        logger.info("Resolved disambiguation to: %s", e.options[0])
                        # Sanitize content before returning
                        return self.sanitize_wikipedia_content(page.content)
                    except (wikipedia.PageError, ConnectionError, ValueError):
                        continue
                except (wikipedia.PageError, ConnectionError, ValueError):
                    continue

            logger.warning("No suitable article found for query: %s", query)
            return None

        except (
            wikipedia.PageError,
            wikipedia.DisambiguationError,
            ConnectionError,
            ValueError,
        ) as e:
            logger.error("Error searching for article '%s': %s", query, str(e))
            logger.info("=" * 80)
            return None

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

    def _fetch_wikipedia_page(
        self,
        page_title: str,
        query: str,
        search_results: List[str],
        optimized_query: str,
    ) -> Optional[Dict[str, str]]:
        """Fetch a Wikipedia page and return article info."""
        try:
            # Use auto_suggest=False to prevent disambiguation issues
            page = wikipedia.page(page_title, auto_suggest=False)

            logger.info("✅ Successfully fetched page!")
            logger.info("📄 Page title: %s", page.title)
            logger.info("🔗 Page URL: %s", page.url)
            logger.info("=" * 80)

            # Sanitize content before returning
            sanitized_content = self.sanitize_wikipedia_content(page.content)

            return {
                "content": sanitized_content,
                "title": page.title,
                "url": page.url,
                "summary": page.summary,
                "search_method": "simple_agentic",
                "original_query": query,
                "optimized_query": optimized_query,
                "selected_from": search_results,
            }

        except wikipedia.exceptions.DisambiguationError as e:
            logger.info("⚠️  Disambiguation page encountered for '%s'", page_title)
            logger.info("📋 Available options: %s...", e.options[:5])

            # Use simple logic to select disambiguation option
            best_option = self._simple_disambiguation_selection(query, e.options[:5])

            logger.info("🔄 Trying disambiguation option: '%s'", best_option)

            try:
                page = wikipedia.page(best_option, auto_suggest=False)
                logger.info("✅ Resolved disambiguation to: %s", best_option)
                logger.info("=" * 80)

                # Sanitize content before returning
                sanitized_content = self.sanitize_wikipedia_content(page.content)

                return {
                    "content": sanitized_content,
                    "title": page.title,
                    "url": page.url,
                    "summary": page.summary,
                    "search_method": "simple_agentic",
                    "original_query": query,
                    "optimized_query": optimized_query,
                    "selected_from": search_results,
                    "disambiguation_resolved": best_option,
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
                return None

        except (
            wikipedia.PageError,
            ConnectionError,
            ValueError,
        ) as page_error:
            logger.error("❌ Error fetching page '%s': %s", page_title, str(page_error))
            return None

    def _simple_disambiguation_selection(self, query: str, options: List[str]) -> str:
        """Simple disambiguation selection"""
        query_lower = query.lower()

        # Prefer options that match the query intent
        for option in options:
            option_lower = option.lower()

            # Avoid obvious mismatches
            if any(
                unwanted in option_lower for unwanted in ["album", "song", "film", "tv"]
            ):
                continue

            # Look for matches
            if any(
                word in option_lower for word in query_lower.split() if len(word) > 3
            ):
                logger.info("🎯 Selected disambiguation option: '%s'", option)
                return option

        # Default to first option
        logger.info("🎯 Using first disambiguation option: '%s'", options[0])
        return options[0]

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
