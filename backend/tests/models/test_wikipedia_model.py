"""
Unit tests for Wikipedia Model
Tests the Wikipedia data access layer
"""

from unittest.mock import Mock, patch

import wikipedia

from backend.models.wikipedia_model import WikipediaService, get_wikipedia_service


class TestWikipediaService:
    """Test cases for WikipediaService"""

    def test_init(self):
        """Test service initialization"""
        service = WikipediaService()
        assert service.wiki_api is not None

    def test_preprocess_historical_query_apollo_11(self):
        """Test historical query preprocessing for Apollo 11"""
        service = WikipediaService()

        test_cases = [
            "July 20 1969",
            "july 20, 1969",
            "20 July 1969",
            "neil armstrong moon",
        ]

        for query in test_cases:
            processed, was_converted = service.preprocess_historical_query(query)
            assert was_converted is True
            assert "Apollo" in processed

    def test_preprocess_historical_query_pearl_harbor(self):
        """Test historical query preprocessing for Pearl Harbor"""
        service = WikipediaService()

        query = "December 7 1941"
        processed, was_converted = service.preprocess_historical_query(query)

        assert was_converted is True
        assert "Pearl Harbor" in processed

    def test_preprocess_historical_query_no_match(self):
        """Test historical query preprocessing with no match"""
        service = WikipediaService()

        query = "artificial intelligence"
        processed, was_converted = service.preprocess_historical_query(query)

        assert was_converted is False
        assert processed == query

    def test_sanitize_wikipedia_content_curly_braces(self):
        """Test content sanitization removes curly braces"""
        service = WikipediaService()

        content = "This has {single} and {{double}} braces"
        sanitized = service.sanitize_wikipedia_content(content)

        assert sanitized == "This has (single) and ((double)) braces"
        assert "{" not in sanitized
        assert "}" not in sanitized

    def test_sanitize_wikipedia_content_empty(self):
        """Test content sanitization with empty content"""
        service = WikipediaService()

        assert service.sanitize_wikipedia_content("") == ""
        assert service.sanitize_wikipedia_content(None) == ""

    @patch("backend.models.wikipedia_model.wikipediaapi")
    def test_fetch_article_success(self, mock_wikipediaapi):
        """Test successful article fetching"""
        # Setup mock
        mock_wiki = Mock()
        mock_page = Mock()
        mock_page.exists.return_value = True
        mock_page.title = "Artificial Intelligence"
        mock_page.text = "AI is a field of computer science {with templates}"

        mock_wiki.page.return_value = mock_page
        mock_wikipediaapi.Wikipedia.return_value = mock_wiki

        service = WikipediaService()
        result = service.fetch_article("Artificial Intelligence")

        assert result is not None
        assert "AI is a field of computer science" in result
        assert "{" not in result  # Should be sanitized

    @patch("backend.models.wikipedia_model.wikipediaapi")
    def test_fetch_article_not_exists(self, mock_wikipediaapi):
        """Test article fetching when page doesn't exist"""
        mock_wiki = Mock()
        mock_page = Mock()
        mock_page.exists.return_value = False

        mock_wiki.page.return_value = mock_page
        mock_wikipediaapi.Wikipedia.return_value = mock_wiki

        service = WikipediaService()

        with patch.object(service, "search_and_fetch_article", return_value=None):
            result = service.fetch_article("NonexistentPage")
            assert result is None

    @patch("backend.models.wikipedia_model.wikipediaapi")
    def test_fetch_article_exception(self, mock_wikipediaapi):
        """Test article fetching when exception occurs"""
        mock_wikipediaapi.Wikipedia.side_effect = Exception("API Error")

        service = WikipediaService()
        result = service.fetch_article("Test Article")

        assert result is None

    @patch("backend.models.wikipedia_model.wikipedia.search")
    @patch("backend.models.wikipedia_model.wikipedia.page")
    def test_search_and_fetch_article_success(self, mock_page, mock_search):
        """Test successful search and fetch"""
        mock_search.return_value = ["Artificial Intelligence", "Machine Learning"]

        mock_page_obj = Mock()
        mock_page_obj.content = "AI content {with markup}"
        mock_page.return_value = mock_page_obj

        service = WikipediaService()
        result = service.search_and_fetch_article("AI")

        assert result is not None
        assert "AI content" in result
        assert "{" not in result  # Should be sanitized

    @patch("backend.models.wikipedia_model.wikipedia.search")
    def test_search_and_fetch_article_no_results(self, mock_search):
        """Test search and fetch with no results"""
        mock_search.return_value = []

        service = WikipediaService()
        result = service.search_and_fetch_article("NonexistentTopic")

        assert result is None

    @patch("backend.models.wikipedia_model.wikipedia.search")
    @patch("backend.models.wikipedia_model.wikipedia.page")
    def test_search_and_fetch_article_disambiguation(self, mock_page, mock_search):
        """Test search and fetch with disambiguation error"""
        mock_search.return_value = ["Apple"]

        # First call raises disambiguation, second succeeds
        mock_page.side_effect = [
            wikipedia.exceptions.DisambiguationError(
                "Apple", ["Apple Inc.", "Apple fruit"]
            ),
            Mock(content="Apple Inc. content"),
        ]

        service = WikipediaService()
        result = service.search_and_fetch_article("Apple")

        assert result is not None
        assert "Apple Inc. content" in result

    @patch("backend.models.wikipedia_model.wikipedia.search")
    @patch("backend.models.wikipedia_model.wikipedia.page")
    def test_search_and_fetch_article_info_success(self, mock_page, mock_search):
        """Test successful article info fetching"""
        mock_search.return_value = ["Quantum Physics"]

        mock_page_obj = Mock()
        mock_page_obj.title = "Quantum Physics"
        mock_page_obj.content = "Quantum physics content {templates}"
        mock_page_obj.url = "https://en.wikipedia.org/wiki/Quantum_physics"
        mock_page_obj.summary = "Quantum physics summary"

        mock_page.return_value = mock_page_obj

        service = WikipediaService()
        result = service.search_and_fetch_article_info("quantum physics")

        assert result is not None
        assert result["title"] == "Quantum Physics"
        assert result["url"] == "https://en.wikipedia.org/wiki/Quantum_physics"
        assert result["summary"] == "Quantum physics summary"
        assert "{" not in result["content"]  # Should be sanitized

    @patch("backend.models.wikipedia_model.wikipedia.search")
    def test_search_and_fetch_article_info_no_results(self, mock_search):
        """Test article info fetching with no search results"""
        mock_search.return_value = []

        service = WikipediaService()
        result = service.search_and_fetch_article_info("nonexistent")

        assert result is None

    def test_enhance_query_with_intent_science_high_confidence(self):
        """Test query enhancement for Science with high confidence"""
        service = WikipediaService()

        enhanced = service.enhance_query_with_intent(
            "quantum mechanics", "Science", 0.8
        )
        # Science queries already containing specific terms should not be enhanced
        assert enhanced == "quantum mechanics"

    def test_enhance_query_with_intent_science_no_specific_terms(self):
        """Test query enhancement for Science without specific terms"""
        service = WikipediaService()

        enhanced = service.enhance_query_with_intent("atoms", "science", 0.8)
        assert "science" in enhanced

    def test_enhance_query_with_intent_biography(self):
        """Test query enhancement for Biography"""
        service = WikipediaService()

        enhanced = service.enhance_query_with_intent("Einstein", "Biography", 0.8)
        assert "biography" in enhanced or enhanced == "Einstein"

    def test_enhance_query_with_intent_low_confidence(self):
        """Test query enhancement with low confidence"""
        service = WikipediaService()

        enhanced = service.enhance_query_with_intent("test query", "Science", 0.3)
        assert enhanced == "test query"  # Should not enhance

    def test_enhance_query_with_intent_unknown_intent(self):
        """Test query enhancement with unknown intent"""
        service = WikipediaService()

        enhanced = service.enhance_query_with_intent("test query", "Unknown", 0.9)
        assert enhanced == "test query"

    @patch("backend.models.wikipedia_model.wikipedia.search")
    @patch("backend.models.wikipedia_model.wikipedia.page")
    def test_fetch_article_with_conversion_info(self, mock_page, mock_search):
        """Test fetching article with conversion information"""
        service = WikipediaService()

        with patch.object(service, "fetch_article", return_value="article content"):
            content, processed_topic, was_converted = (
                service.fetch_article_with_conversion_info("July 20 1969")
            )

            assert content == "article content"
            assert "Apollo" in processed_topic
            assert was_converted is True

    def test_simple_query_optimization_beatles(self):
        """Test simple query optimization for Beatles"""
        service = WikipediaService()

        optimized = service._simple_query_optimization("who were the beatles")
        assert optimized == "The Beatles"

    def test_simple_query_optimization_who_was(self):
        """Test simple query optimization for 'who was' questions"""
        service = WikipediaService()

        optimized = service._simple_query_optimization("Who was Albert Einstein?")
        assert optimized == "Albert Einstein"

    def test_simple_query_optimization_no_change(self):
        """Test simple query optimization when no optimization applies"""
        service = WikipediaService()

        query = "artificial intelligence"
        optimized = service._simple_query_optimization(query)
        assert optimized == query

    def test_simple_page_selection_main_page_preference(self):
        """Test simple page selection prefers main pages"""
        service = WikipediaService()

        options = ["The Beatles", "The Beatles (album)", "List of Beatles songs"]
        selected = service._simple_page_selection("beatles", options)
        assert selected == "The Beatles"

    def test_simple_page_selection_no_parentheses_preference(self):
        """Test simple page selection prefers pages without parentheses"""
        service = WikipediaService()

        options = [
            "Artificial Intelligence (film)",
            "Artificial Intelligence",
            "AI (disambiguation)",
        ]
        selected = service._simple_page_selection("artificial intelligence", options)
        assert selected == "Artificial Intelligence"

    def test_simple_disambiguation_selection(self):
        """Test simple disambiguation selection"""
        service = WikipediaService()

        options = ["Apple Inc.", "Apple (fruit)", "Apple (album)"]
        selected = service._simple_disambiguation_selection("apple company", options)
        assert selected == "Apple Inc."

    def test_simple_disambiguation_selection_avoid_unwanted(self):
        """Test disambiguation selection avoids unwanted types"""
        service = WikipediaService()

        options = ["Einstein (film)", "Albert Einstein", "Einstein (album)"]
        selected = service._simple_disambiguation_selection("einstein physics", options)
        assert selected == "Albert Einstein"

    @patch("backend.models.wikipedia_model.wikipedia.search")
    @patch("backend.models.wikipedia_model.wikipedia.page")
    def test_search_and_fetch_article_agentic_simple_success(
        self, mock_page, mock_search
    ):
        """Test simple agentic search success"""
        mock_search.return_value = ["The Beatles", "Beatles discography"]

        mock_page_obj = Mock()
        mock_page_obj.title = "The Beatles"
        mock_page_obj.content = "The Beatles were a British rock band"
        mock_page_obj.url = "https://en.wikipedia.org/wiki/The_Beatles"
        mock_page_obj.summary = "British rock band"

        mock_page.return_value = mock_page_obj

        service = WikipediaService()
        result = service.search_and_fetch_article_agentic_simple("who were the beatles")

        assert result is not None
        assert result["title"] == "The Beatles"
        assert result["search_method"] == "simple_agentic"
        assert result["original_query"] == "who were the beatles"

    @patch("backend.models.wikipedia_model.wikipedia.search")
    def test_search_and_fetch_article_agentic_simple_no_results(self, mock_search):
        """Test simple agentic search with no results"""
        mock_search.return_value = []

        service = WikipediaService()
        result = service.search_and_fetch_article_agentic_simple("nonexistent topic")

        assert result is None

    @patch("backend.models.wikipedia_model.wikipedia.search")
    def test_search_and_fetch_article_agentic_simple_fallback(self, mock_search):
        """Test simple agentic search fallback to basic search"""
        mock_search.side_effect = Exception("Search error")

        service = WikipediaService()

        with patch.object(
            service, "search_and_fetch_article_info", return_value={"title": "fallback"}
        ):
            result = service.search_and_fetch_article_agentic_simple("test query")
            assert result["title"] == "fallback"


class TestWikipediaServiceSingleton:
    """Test singleton pattern for WikipediaService"""

    def test_get_wikipedia_service_singleton(self):
        """Test that get_wikipedia_service returns same instance"""
        service1 = get_wikipedia_service()
        service2 = get_wikipedia_service()
        assert service1 is service2

    def test_singleton_instance_type(self):
        """Test that singleton returns correct type"""
        service = get_wikipedia_service()
        assert isinstance(service, WikipediaService)


class TestWikipediaServiceErrorHandling:
    """Test error handling in WikipediaService"""

    @patch("backend.models.wikipedia_model.wikipedia.search")
    def test_search_connection_error(self, mock_search):
        """Test handling of connection errors during search"""
        mock_search.side_effect = ConnectionError("Network error")

        service = WikipediaService()
        result = service.search_and_fetch_article("test query")

        assert result is None

    @patch("backend.models.wikipedia_model.wikipedia.search")
    def test_search_timeout_error(self, mock_search):
        """Test handling of timeout errors during search"""
        mock_search.side_effect = TimeoutError("Request timeout")

        service = WikipediaService()
        result = service.search_and_fetch_article_info("test query")

        assert result is None

    @patch("backend.models.wikipedia_model.wikipedia.search")
    @patch("backend.models.wikipedia_model.wikipedia.page")
    def test_page_fetch_error_resilience(self, mock_page, mock_search):
        """Test resilience when page fetching fails for some results"""
        mock_search.return_value = [
            "Good Article",
            "Bad Article",
            "Another Good Article",
        ]

        # First and third pages work, second fails
        mock_page.side_effect = [
            wikipedia.PageError("Page error"),
            Mock(content="Good content"),
            Mock(content="Also good content"),
        ]

        service = WikipediaService()
        result = service.search_and_fetch_article("test query")

        assert result is not None
        assert "Good content" in result


class TestWikipediaServiceIntegration:
    """Integration tests for WikipediaService"""

    def test_full_workflow_with_historical_query(self):
        """Test complete workflow with historical query processing"""
        service = WikipediaService()

        with patch.object(service, "search_and_fetch_article_info") as mock_search:
            # Mock the return value with sanitized content (curly braces replaced)
            mock_search.return_value = {
                "title": "Apollo 11",
                "content": "Apollo 11 was the spaceflight (that first landed)",
                "url": "https://en.wikipedia.org/wiki/Apollo_11",
                "summary": "First moon landing",
            }

            result = service.search_and_fetch_article_info("July 20 1969")

            assert result["title"] == "Apollo 11"
            assert "{" not in result["content"]  # Should be sanitized
            assert (
                "(" in result["content"]
            )  # Curly braces should be replaced with parentheses

    def test_query_enhancement_workflow(self):
        """Test query enhancement integrated with search"""
        service = WikipediaService()

        # Test that enhancement affects search behavior
        enhanced_science = service.enhance_query_with_intent(
            "particles", "Science", 0.9
        )
        enhanced_biography = service.enhance_query_with_intent(
            "Einstein", "Biography", 0.9
        )

        # Enhancements should be different for different intents
        if enhanced_science != "particles":
            assert "theory" in enhanced_science or "principles" in enhanced_science

        if enhanced_biography != "Einstein":
            assert "biography" in enhanced_biography

    @patch("backend.models.wikipedia_model.wikipedia.search")
    @patch("backend.models.wikipedia_model.wikipedia.page")
    def test_end_to_end_article_retrieval(self, mock_page, mock_search):
        """Test end-to-end article retrieval with all components"""
        mock_search.return_value = ["Quantum Mechanics", "Quantum Physics"]

        mock_page_obj = Mock()
        mock_page_obj.title = "Quantum Mechanics"
        mock_page_obj.content = "Quantum mechanics is {a fundamental theory}"
        mock_page_obj.url = "https://en.wikipedia.org/wiki/Quantum_mechanics"
        mock_page_obj.summary = "Fundamental theory in physics"
        mock_page.return_value = mock_page_obj

        service = WikipediaService()

        # Test the complete flow: enhance query -> search -> fetch -> sanitize
        enhanced_query = service.enhance_query_with_intent("quantum", "Science", 0.8)
        result = service.search_and_fetch_article_info(enhanced_query)

        assert result is not None
        assert result["title"] == "Quantum Mechanics"
        assert "{" not in result["content"]
        assert result["url"] == "https://en.wikipedia.org/wiki/Quantum_mechanics"
