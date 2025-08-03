"""
Unit tests for WikipediaSearchTool
Tests the Wikipedia search tool functionality with proper mocking
"""

from unittest.mock import Mock, patch

from backend.models.wikipedia.wikipedia_search_tool import WikipediaSearchTool


class TestWikipediaSearchTool:
    """Test cases for WikipediaSearchTool"""

    @patch("backend.models.wikipedia.wikipedia_search_tool.wikipedia")
    def test_search_wikipedia_detailed_format(self, mock_wikipedia):
        """Test Wikipedia search with detailed format"""
        # Mock wikipedia search results
        mock_wikipedia.search.return_value = ["Article 1", "Article 2", "Article 3"]

        # Test search with detailed format (default)
        result = WikipediaSearchTool.search_wikipedia("test query")

        # Verify result format
        expected = "Found 3 Wikipedia articles for 'test query':\n1. Article 1\n2. Article 2\n3. Article 3\n"
        assert result == expected
        mock_wikipedia.search.assert_called_once_with("test query", results=5)

    @patch("backend.models.wikipedia.wikipedia_search_tool.wikipedia")
    def test_search_wikipedia_simple_format(self, mock_wikipedia):
        """Test Wikipedia search with simple format"""
        # Mock wikipedia search results
        mock_wikipedia.search.return_value = ["Article A", "Article B"]

        # Test search with simple format
        result = WikipediaSearchTool.search_wikipedia(
            "test query", format_style="simple"
        )

        # Verify result format
        expected = "Found 2 articles: Article A, Article B"
        assert result == expected
        mock_wikipedia.search.assert_called_once_with("test query", results=5)

    @patch("backend.models.wikipedia.wikipedia_search_tool.wikipedia")
    def test_search_wikipedia_no_results(self, mock_wikipedia):
        """Test Wikipedia search with no results"""
        # Mock empty search results
        mock_wikipedia.search.return_value = []

        # Test search with no results
        result = WikipediaSearchTool.search_wikipedia("nonexistent query")

        # Verify no results message
        expected = "No Wikipedia articles found for: nonexistent query"
        assert result == expected

    @patch("backend.models.wikipedia.wikipedia_search_tool.wikipedia")
    def test_search_wikipedia_custom_results_count(self, mock_wikipedia):
        """Test Wikipedia search with custom results count"""
        # Mock wikipedia search results
        mock_wikipedia.search.return_value = ["Article 1", "Article 2"]

        # Test search with custom results count
        result = WikipediaSearchTool.search_wikipedia("test query", results=2)

        # Verify custom results count was used
        mock_wikipedia.search.assert_called_once_with("test query", results=2)
        assert "Found 2" in result

    @patch("backend.models.wikipedia.wikipedia_search_tool.wikipedia")
    def test_search_wikipedia_exception_handling(self, mock_wikipedia):
        """Test Wikipedia search exception handling"""
        # Mock wikipedia search to raise exception
        mock_wikipedia.search.side_effect = ValueError("Search error")
        mock_wikipedia.PageError = ValueError
        mock_wikipedia.DisambiguationError = ValueError

        # Test search with exception
        result = WikipediaSearchTool.search_wikipedia("error query")

        # Verify error message
        assert "Wikipedia search error: Search error" in result

    @patch("backend.models.wikipedia.wikipedia_search_tool.wikipedia")
    def test_get_article_preview_with_url(self, mock_wikipedia):
        """Test getting article preview with URL included"""
        # Mock wikipedia page
        mock_page = Mock()
        mock_page.title = "Test Article"
        mock_page.url = "https://en.wikipedia.org/wiki/Test_Article"
        mock_page.summary = (
            "This is a test article summary with enough content to test truncation."
        )
        mock_wikipedia.page.return_value = mock_page

        # Test article preview with URL (default)
        result = WikipediaSearchTool.get_article_preview("Test Article")

        # Verify result includes URL
        assert "Article: Test Article" in result
        assert "URL: https://en.wikipedia.org/wiki/Test_Article" in result
        assert (
            "Summary: This is a test article summary with enough content to test truncation."
            in result
        )

    @patch("backend.models.wikipedia.wikipedia_search_tool.wikipedia")
    def test_get_article_preview_without_url(self, mock_wikipedia):
        """Test getting article preview without URL"""
        # Mock wikipedia page
        mock_page = Mock()
        mock_page.title = "Test Article"
        mock_page.summary = "Short summary"
        mock_wikipedia.page.return_value = mock_page

        # Test article preview without URL
        result = WikipediaSearchTool.get_article_preview(
            "Test Article", include_url=False
        )

        # Verify result doesn't include URL
        assert "Article: Test Article" in result
        assert "URL:" not in result
        assert "Summary: Short summary" in result

    @patch("backend.models.wikipedia.wikipedia_search_tool.wikipedia")
    def test_get_article_preview_summary_truncation(self, mock_wikipedia):
        """Test article preview summary truncation"""
        # Mock wikipedia page with long summary
        mock_page = Mock()
        mock_page.title = "Long Article"
        mock_page.summary = "x" * 500  # 500 character summary
        mock_wikipedia.page.return_value = mock_page

        # Test article preview with custom summary length
        result = WikipediaSearchTool.get_article_preview(
            "Long Article", summary_length=100
        )

        # Verify summary is truncated
        assert "Summary: " + "x" * 100 + "..." in result

    @patch("backend.models.wikipedia.wikipedia_search_tool.wikipedia")
    def test_get_article_preview_disambiguation_error(self, mock_wikipedia):
        """Test article preview with disambiguation error"""

        # Create a proper DisambiguationError mock
        class MockDisambiguationError(Exception):
            def __init__(self, title, options):
                super().__init__()
                self.options = options

        # Set up the mock
        mock_wikipedia.exceptions.DisambiguationError = MockDisambiguationError
        options = ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"]
        mock_wikipedia.page.side_effect = MockDisambiguationError("title", options)

        # Test article preview with disambiguation
        result = WikipediaSearchTool.get_article_preview("Ambiguous Title")

        # Verify disambiguation message shows first 3 options
        assert (
            "Disambiguation page. Options: ['Option 1', 'Option 2', 'Option 3']"
            in result
        )

    @patch("backend.models.wikipedia.wikipedia_search_tool.wikipedia")
    def test_get_article_preview_page_error(self, mock_wikipedia):
        """Test article preview with page error"""

        # Create proper exception classes
        class MockPageError(Exception):
            pass

        class MockDisambiguationError(Exception):
            def __init__(self, title, options):
                super().__init__()
                self.options = options

        # Set up mock exceptions
        mock_wikipedia.PageError = MockPageError
        mock_wikipedia.exceptions.DisambiguationError = MockDisambiguationError
        mock_wikipedia.page.side_effect = MockPageError("Page not found")

        # Test article preview with error
        result = WikipediaSearchTool.get_article_preview("Nonexistent Article")

        # Verify error message
        assert "Error getting article preview: Page not found" in result

    @patch("backend.models.wikipedia.wikipedia_search_tool.wikipedia")
    def test_suggest_wikipedia_success(self, mock_wikipedia):
        """Test Wikipedia suggestions with successful result"""
        # Mock wikipedia suggest
        mock_wikipedia.suggest.return_value = "corrected spelling"

        # Test suggestion
        result = WikipediaSearchTool.suggest_wikipedia("wrong spelling")

        # Verify suggestion
        assert result == "Wikipedia suggests: corrected spelling"
        mock_wikipedia.suggest.assert_called_once_with("wrong spelling")

    @patch("backend.models.wikipedia.wikipedia_search_tool.wikipedia")
    def test_suggest_wikipedia_no_suggestions(self, mock_wikipedia):
        """Test Wikipedia suggestions with no results"""
        # Mock wikipedia suggest with no results
        mock_wikipedia.suggest.return_value = None

        # Test suggestion
        result = WikipediaSearchTool.suggest_wikipedia("perfect spelling")

        # Verify no suggestions message
        assert result == "No suggestions found for: perfect spelling"

    @patch("backend.models.wikipedia.wikipedia_search_tool.wikipedia")
    def test_suggest_wikipedia_exception_handling(self, mock_wikipedia):
        """Test Wikipedia suggestions exception handling"""
        # Mock wikipedia suggest to raise exception
        mock_wikipedia.suggest.side_effect = ConnectionError("Network error")

        # Test suggestion with exception
        result = WikipediaSearchTool.suggest_wikipedia("error query")

        # Verify error message
        assert "Wikipedia suggest error: Network error" in result
