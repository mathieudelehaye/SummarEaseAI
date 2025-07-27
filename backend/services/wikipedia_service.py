"""
Wikipedia Service
Moved from utils/wikipedia_fetcher.py to proper services layer
Handles all Wikipedia content fetching and processing
"""

import logging
import re
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import wikipedia
import wikipediaapi

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

logger = logging.getLogger(__name__)


class WikipediaService:
    """
    Handles all Wikipedia content fetching and processing
    Pure business logic for Wikipedia operations
    """

    def __init__(self):
        # Set user agent for API compliance
        self.wiki_api = wikipediaapi.Wikipedia(
            language="en",
            user_agent="SummarEaseAI/1.0 (https://github.com/your-repo) Python/WikipediaAPI",
        )

        # Set user agent for wikipedia library
        wikipedia.set_user_agent("SummarEaseAI/1.0 (https://github.com/your-repo)")

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
        except (
            wikipedia.PageError,
            wikipedia.DisambiguationError,
            ConnectionError,
            ValueError,
            KeyError,
        ) as e:
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
            search_results = wikipedia.search(processed_query, results=max_results + 2)

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

        except (
            wikipedia.PageError,
            wikipedia.DisambiguationError,
            ConnectionError,
            ValueError,
            KeyError,
        ) as e:
            logger.error("❌ Error searching for article '%s': %s", query, str(e))
            logger.info("=" * 80)
            return None

    def fetch_article_with_conversion_info(
        self, topic: str
    ) -> Tuple[Optional[str], str, bool]:
        """
        Fetch Wikipedia article and return conversion information

        Returns:
            tuple: (article_content, processed_topic, was_converted)
        """
        try:
            # Preprocess historical queries
            processed_topic, was_converted = self.preprocess_historical_query(topic)

            # Fetch article using the processed topic
            article_content = (
                self.fetch_article(processed_topic)
                if was_converted
                else self.fetch_article(topic)
            )

            return article_content, processed_topic, was_converted

        except (
            wikipedia.PageError,
            wikipedia.DisambiguationError,
            ConnectionError,
            ValueError,
            KeyError,
        ) as e:
            logger.error(
                "Error fetching article with conversion info '%s': %s", topic, str(e)
            )
            return None, topic, False

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
        intent_enhancements = {
            "Science": {
                "keywords": [
                    "quantum",
                    "physics",
                    "chemistry",
                    "biology",
                    "scientific",
                    "theory",
                    "principle",
                ],
                "suffixes": [
                    "theory",
                    "principles",
                ],  # More specific than just 'science'
            },
            "History": {
                "keywords": [
                    "war",
                    "battle",
                    "historical",
                    "timeline",
                    "events",
                    "period",
                    "ancient",
                ],
                "suffixes": ["history", "timeline"],
            },
            "Biography": {
                "keywords": [
                    "biography",
                    "life",
                    "who was",
                    "born",
                    "died",
                    "achievements",
                ],
                "suffixes": ["biography", "life"],
            },
            "Technology": {
                "keywords": [
                    "technology",
                    "innovation",
                    "development",
                    "advancement",
                    "computer",
                    "software",
                ],
                "suffixes": ["technology", "innovation"],
            },
            "Sports": {
                "keywords": [
                    "sports",
                    "game",
                    "competition",
                    "tournament",
                    "olympics",
                    "team",
                ],
                "suffixes": ["sports", "game"],
            },
            "Arts": {
                "keywords": [
                    "art",
                    "artistic",
                    "cultural",
                    "creative",
                    "painting",
                    "music",
                ],
                "suffixes": ["art", "culture"],
            },
            "Politics": {
                "keywords": [
                    "political",
                    "government",
                    "policy",
                    "democracy",
                    "election",
                ],
                "suffixes": ["politics", "government"],
            },
            "Geography": {
                "keywords": [
                    "geographic",
                    "location",
                    "region",
                    "country",
                    "city",
                    "mountain",
                ],
                "suffixes": ["geography", "location"],
            },
        }

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

    def search_and_fetch_article_agentic_simple(
        self, query: str, max_results: int = 3
    ) -> Optional[Dict[str, str]]:
        """
        Simple agentic Wikipedia search with basic optimization

        Args:
            query: Original user query
            max_results: Maximum number of search results to consider

        Returns:
            Dictionary with article info or None if not found
        """
        try:
            logger.info("🤖 STARTING SIMPLE AGENTIC WIKIPEDIA SEARCH")
            logger.info("=" * 80)
            logger.info("📝 Original user query: '%s'", query)

            # Check if we can do basic optimization
            optimized_query = self._simple_query_optimization(query)

            # Use optimized query for search
            search_query = optimized_query if optimized_query != query else query

            # Preprocess historical queries (keep existing logic)
            processed_query, was_converted = self.preprocess_historical_query(
                search_query
            )

            logger.info("📝 Final processed query: '%s'", processed_query)
            logger.info("🔄 Query was converted: %s", was_converted)
            logger.info("📊 Max results requested: %d", max_results)

            # Search Wikipedia
            search_results = wikipedia.search(processed_query, results=max_results)

            logger.info(
                "✅ Wikipedia API returned %d search results:", len(search_results)
            )
            for i, result in enumerate(search_results):
                logger.info("   %d. '%s'", i + 1, result)

            if not search_results:
                logger.warning(
                    "❌ No search results found for query: '%s'", processed_query
                )
                return None

            # Simple page selection logic
            selected_page = self._simple_page_selection(query, search_results)

            # Fetch the selected page
            logger.info("=" * 80)
            logger.info("📄 FETCHING SELECTED WIKIPEDIA PAGE")
            logger.info("=" * 80)
            logger.info("🎯 Selected page: '%s'", selected_page)

            return self._fetch_wikipedia_page(
                selected_page, query, search_results, optimized_query
            )

        except (
            wikipedia.PageError,
            wikipedia.DisambiguationError,
            ConnectionError,
            ValueError,
            KeyError,
        ) as e:
            logger.error(
                "❌ Error in simple agentic search for '%s': %s", query, str(e)
            )
            logger.info("=" * 80)
            # Fallback to basic search
            logger.info("🔄 Falling back to basic Wikipedia search")
            return self.search_and_fetch_article_info(query, max_results)

    def _simple_query_optimization(self, query: str) -> str:
        """Simple rule-based query optimization"""
        query_lower = query.lower()

        # Simple optimization rules
        optimizations = {
            # Questions about people/bands
            "who were the beatles": "The Beatles",
            "who was": query.replace("Who was", "")
            .replace("who was", "")
            .strip()
            .title(),
            "who were": query.replace("Who were", "")
            .replace("who were", "")
            .strip()
            .title(),
            # Science questions
            "explain quantum": "Quantum mechanics",
            "quantum physics": "Quantum mechanics",
            # Historical dates
            "what happened on july 20, 1969": "Apollo 11",
            "july 20 1969": "Apollo 11",
            "apollo 11 moon landing": "Apollo 11",
        }

        # Check for direct matches
        if query_lower in optimizations:
            optimized = optimizations[query_lower]
            logger.info("🧠 Simple optimization: '%s' → '%s'", query, optimized)
            return optimized

        # Handle "who was/were" patterns
        if query_lower.startswith("who was ") or query_lower.startswith("who were "):
            # Extract the subject
            subject = query.split(" ", 2)[-1].replace("?", "").strip()
            logger.info("🧠 Person/entity optimization: '%s' → '%s'", query, subject)
            return subject

        logger.info("🧠 No optimization applied to: '%s'", query)
        return query

    def _simple_page_selection(self, query: str, page_options: List[str]) -> str:
        """Simple rule-based page selection"""
        if len(page_options) <= 1:
            return page_options[0] if page_options else ""

        query_lower = query.lower()

        # Prefer main articles over sub-articles
        for page in page_options:
            page_lower = page.lower()

            # For Beatles questions, prefer main "The Beatles" page
            if "beatles" in query_lower and page_lower == "the beatles":
                logger.info("🎯 Selected main Beatles page: '%s'", page)
                return page

            # Prefer pages without parentheses or "list of"
            if "(" not in page and "list of" not in page_lower:
                # If it's a simple match to the query intent
                if any(
                    word in page_lower for word in query_lower.split() if len(word) > 3
                ):
                    logger.info("🎯 Selected main page: '%s'", page)
                    return page

        # Default to first option
        logger.info("🎯 Using first option: '%s'", page_options[0])
        return page_options[0]

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


class _WikipediaServiceSingleton:
    """Singleton wrapper for WikipediaService"""

    _instance = None

    @classmethod
    def get_instance(cls) -> WikipediaService:
        """Get or create the singleton service instance"""
        if cls._instance is None:
            cls._instance = WikipediaService()
        return cls._instance


def get_wikipedia_service() -> WikipediaService:
    """Get or create global Wikipedia service instance"""
    return _WikipediaServiceSingleton.get_instance()


# Compatibility functions for existing imports
def search_and_fetch_article_info(
    query: str, max_results: int = 1
) -> Optional[Dict[str, str]]:
    """Compatibility function for existing imports"""
    return get_wikipedia_service().search_and_fetch_article_info(query, max_results)


def enhance_query_with_intent(query: str, intent: str, confidence: float = 0.5) -> str:
    """Compatibility function for existing imports"""
    return get_wikipedia_service().enhance_query_with_intent(query, intent, confidence)


def fetch_article_with_conversion_info(topic: str) -> Tuple[Optional[str], str, bool]:
    """Compatibility function for existing imports"""
    return get_wikipedia_service().fetch_article_with_conversion_info(topic)


def search_and_fetch_article_agentic_simple(
    query: str, max_results: int = 3
) -> Optional[Dict[str, str]]:
    """Compatibility function for existing imports"""
    return get_wikipedia_service().search_and_fetch_article_agentic_simple(
        query, max_results
    )
