"""
Unit tests for Summarization Service
Tests the main summarization service business logic
"""

from unittest.mock import Mock, patch

import wikipedia

from backend.services.summarization_service import (
    SummarizationService,
    get_summarization_service,
)


class TestSummarizationService:
    """Test cases for SummarizationService"""

    @patch("backend.services.summarization_service.get_bert_classifier")
    def test_init_with_bert_available(self, mock_get_classifier):
        """Test service initialization when BERT is available"""
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_classifier.is_loaded.return_value = True
        mock_get_classifier.return_value = mock_classifier

        with patch("backend.services.summarization_service.BERT_AVAILABLE", True):
            service = SummarizationService()
            assert service.bert_model_loaded is True
            assert service.bert_classifier is not None

    @patch("backend.services.summarization_service.get_bert_classifier")
    def test_init_with_bert_unavailable(self, mock_get_classifier):
        """Test service initialization when BERT is unavailable"""
        with patch("backend.services.summarization_service.BERT_AVAILABLE", False):
            service = SummarizationService()
            assert service.bert_model_loaded is False
            assert service.bert_classifier is None

    @patch("backend.services.summarization_service.get_bert_classifier")
    def test_init_with_bert_load_failure(self, mock_get_classifier):
        """Test service initialization when BERT model fails to load"""
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = False
        mock_classifier.is_loaded.return_value = False
        mock_get_classifier.return_value = mock_classifier

        with patch("backend.services.summarization_service.BERT_AVAILABLE", True):
            service = SummarizationService()
            assert service.bert_model_loaded is False

    def test_bert_categories_defined(self):
        """Test that BERT categories are properly defined"""
        service = SummarizationService()
        expected_categories = [
            "History",
            "Music",
            "Science",
            "Sports",
            "Technology",
            "Finance",
        ]
        assert service.bert_categories == expected_categories
        assert service.special_categories == ["NO DETECTED"]
        assert len(service.all_categories) == len(expected_categories) + 1

    @patch("backend.services.summarization_service.get_bert_classifier")
    def test_predict_intent_bert_success(self, mock_get_classifier):
        """Test successful BERT intent prediction"""
        mock_classifier = Mock()
        mock_classifier.predict.return_value = ("Technology", 0.85)
        mock_get_classifier.return_value = mock_classifier

        with patch("backend.services.summarization_service.BERT_AVAILABLE", True):
            service = SummarizationService()
            service.bert_model_loaded = True
            service.bert_classifier = mock_classifier

            result = service.predict_intent_bert("Tell me about AI")

            assert result["predicted_category"] == "Technology"
            assert result["confidence"] == 0.85
            assert result["text"] == "Tell me about AI"

    @patch("backend.services.summarization_service.get_bert_classifier")
    def test_predict_intent_bert_model_not_loaded(self, mock_get_classifier):
        """Test BERT intent prediction when model not loaded"""
        service = SummarizationService()
        service.bert_model_loaded = False

        result = service.predict_intent_bert("Test text")

        assert result["error"] == "BERT model not loaded"
        assert result["predicted_category"] == "NO DETECTED"
        assert result["confidence"] == 0.0

    @patch("backend.services.summarization_service.get_bert_classifier")
    def test_predict_intent_bert_exception(self, mock_get_classifier):
        """Test BERT intent prediction when classifier raises exception"""
        mock_classifier = Mock()
        mock_classifier.predict.side_effect = Exception("Classification error")
        mock_get_classifier.return_value = mock_classifier

        with patch("backend.services.summarization_service.BERT_AVAILABLE", True):
            service = SummarizationService()
            service.bert_model_loaded = True
            service.bert_classifier = mock_classifier

            result = service.predict_intent_bert("Test text")

            assert "error" in result
            assert result["predicted_category"] == "NO DETECTED"
            assert result["confidence"] == 0.0

    @patch("backend.services.summarization_service.wikipedia.search")
    @patch("backend.services.summarization_service.wikipedia.page")
    @patch("backend.services.summarization_service.wikipedia.summary")
    def test_search_wikipedia_success(self, mock_summary, mock_page, mock_search):
        """Test successful Wikipedia search"""
        mock_search.return_value = ["Artificial Intelligence"]
        mock_page_obj = Mock()
        mock_page_obj.url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
        mock_page.return_value = mock_page_obj
        mock_summary.return_value = "AI is a field of computer science."

        service = SummarizationService()
        result = service._search_wikipedia("AI")

        assert result["status"] == "success"
        assert result["title"] == "Artificial Intelligence"
        assert result["summary"] == "AI is a field of computer science."
        assert result["url"] == "https://en.wikipedia.org/wiki/Artificial_intelligence"

    @patch("backend.services.summarization_service.wikipedia.search")
    def test_search_wikipedia_no_results(self, mock_search):
        """Test Wikipedia search with no results"""
        mock_search.return_value = []

        service = SummarizationService()
        result = service._search_wikipedia("NonExistentTopic")

        assert "error" in result
        assert result["error"] == "No Wikipedia articles found"
        assert result["summary"] is None

    @patch("backend.services.summarization_service.wikipedia.search")
    @patch("backend.services.summarization_service.wikipedia.page")
    def test_search_wikipedia_disambiguation_error(self, mock_page, mock_search):
        """Test Wikipedia search with disambiguation error"""
        mock_search.return_value = ["Apple"]
        mock_page.side_effect = [
            wikipedia.exceptions.DisambiguationError(
                "Apple", ["Apple Inc.", "Apple fruit"]
            ),
            Mock(url="https://en.wikipedia.org/wiki/Apple_Inc."),
        ]

        with patch(
            "backend.services.summarization_service.wikipedia.summary"
        ) as mock_summary:
            mock_summary.return_value = "Apple Inc. is a technology company."

            service = SummarizationService()
            result = service._search_wikipedia("Apple")

            assert result["status"] == "success"
            assert result["title"] == "Apple Inc."

    @patch("backend.services.summarization_service.get_bert_classifier")
    def test_summarize_single_source_success(self, mock_get_classifier):
        """Test successful single source summarization"""
        mock_classifier = Mock()
        mock_classifier.predict.return_value = ("Technology", 0.9)
        mock_get_classifier.return_value = mock_classifier

        service = SummarizationService()
        service.bert_model_loaded = True
        service.bert_classifier = mock_classifier

        with patch.object(service, "_search_wikipedia") as mock_search:
            mock_search.return_value = {
                "status": "success",
                "title": "AI",
                "summary": "AI summary",
                "url": "https://en.wikipedia.org/wiki/AI",
                "query": "artificial intelligence",
            }

            result = service.summarize_single_source("artificial intelligence")

            assert result["query"] == "artificial intelligence"
            assert result["summary"] == "AI summary"
            assert result["intent"]["category"] == "Technology"
            assert result["intent"]["confidence"] == 0.9

    @patch("backend.services.summarization_service.get_bert_classifier")
    def test_summarize_single_source_without_intent(self, mock_get_classifier):
        """Test single source summarization without intent classification"""
        service = SummarizationService()
        service.bert_model_loaded = False

        with patch.object(service, "_search_wikipedia") as mock_search:
            mock_search.return_value = {
                "status": "success",
                "title": "Test Article",
                "summary": "Test summary",
                "url": "https://test.com",
                "query": "test query",
            }

            result = service.summarize_single_source("test query", use_intent=False)

            assert result["query"] == "test query"
            assert result["summary"] == "Test summary"
            assert "intent" not in result

    def test_summarize_single_source_search_error(self):
        """Test single source summarization when search fails"""
        service = SummarizationService()

        with patch.object(service, "_search_wikipedia") as mock_search:
            mock_search.return_value = {
                "error": "Search failed",
                "query": "test",
                "summary": None,
            }

            result = service.summarize_single_source("test query")

            assert "error" in result

    def test_summarize_multi_source_placeholder(self):
        """Test multi-source summarization placeholder implementation"""
        service = SummarizationService()

        with patch.object(service, "summarize_single_source") as mock_single:
            mock_single.return_value = {
                "query": "test",
                "summary": "test summary",
                "title": "Test Article",
            }

            result = service.summarize_multi_source("test query", "BALANCED", 3)

            assert result["cost_mode"] == "BALANCED"
            assert result["articles_used"] == 1
            assert result["method"] == "single_source_placeholder"

    def test_get_health_status(self):
        """Test health status retrieval"""
        service = SummarizationService()
        status = service.get_health_status()

        assert status["status"] == "healthy"
        assert "bert_model_loaded" in status
        assert status["categories"] == service.bert_categories
        assert status["wikipedia_available"] is True

    def test_get_system_status(self):
        """Test system status retrieval"""
        service = SummarizationService()
        status = service.get_system_status()

        assert status["status"] == "healthy"
        assert "models" in status
        assert "services" in status
        assert status["models"]["bert"]["categories"] == service.bert_categories
        assert status["services"]["wikipedia"] is True

    @patch("backend.services.summarization_service.get_bert_classifier")
    def test_classify_intent(self, mock_get_classifier):
        """Test intent classification method"""
        mock_classifier = Mock()
        mock_classifier.predict.return_value = ("Science", 0.8)
        mock_get_classifier.return_value = mock_classifier

        with patch("backend.services.summarization_service.BERT_AVAILABLE", True):
            service = SummarizationService()
            service.bert_model_loaded = True
            service.bert_classifier = mock_classifier

            result = service.classify_intent("quantum physics")

            assert result["predicted_category"] == "Science"
            assert result["confidence"] == 0.8

    def test_summarize_multi_source_with_agents_delegates(self):
        """Test that multi-source with agents delegates to basic multi-source"""
        service = SummarizationService()

        with patch.object(service, "summarize_multi_source") as mock_multi:
            mock_multi.return_value = {"result": "test"}

            result = service.summarize_multi_source_with_agents(
                "test query", max_articles=3, max_lines=10, cost_mode="balanced"
            )

            assert result == {"result": "test"}
            mock_multi.assert_called_once_with(
                query="test query", max_articles=3, cost_mode="balanced"
            )


class TestSummarizationServiceSingleton:
    """Test singleton pattern for SummarizationService"""

    def test_get_summarization_service_singleton(self):
        """Test that get_summarization_service returns same instance"""
        service1 = get_summarization_service()
        service2 = get_summarization_service()
        assert service1 is service2

    def test_singleton_instance_type(self):
        """Test that singleton returns correct type"""
        service = get_summarization_service()
        assert isinstance(service, SummarizationService)


class TestSummarizationServiceErrorHandling:
    """Test error handling in SummarizationService"""

    @patch("backend.services.summarization_service.wikipedia.search")
    def test_search_wikipedia_connection_error(self, mock_search):
        """Test Wikipedia search with connection error"""
        mock_search.side_effect = ConnectionError("Network error")

        service = SummarizationService()
        result = service._search_wikipedia("test query")

        assert "error" in result
        assert result["summary"] is None

    @patch("backend.services.summarization_service.wikipedia.search")
    def test_search_wikipedia_timeout_error(self, mock_search):
        """Test Wikipedia search with timeout error"""
        mock_search.side_effect = TimeoutError("Request timeout")

        service = SummarizationService()
        result = service._search_wikipedia("test query")

        assert "error" in result
        assert result["summary"] is None

    def test_summarize_single_source_exception_handling(self):
        """Test exception handling in single source summarization"""
        service = SummarizationService()

        with patch.object(service, "_search_wikipedia") as mock_search:
            mock_search.side_effect = Exception("Unexpected error")

            result = service.summarize_single_source("test query")

            assert "error" in result
            assert "summary" in result
            assert result["summary"] is None

    def test_summarize_multi_source_exception_handling(self):
        """Test exception handling in multi-source summarization"""
        service = SummarizationService()

        with patch.object(service, "summarize_single_source") as mock_single:
            mock_single.side_effect = Exception("Service error")

            result = service.summarize_multi_source("test query")

            assert "error" in result
            assert "summary" in result
            assert result["summary"] is None


class TestSummarizationServiceIntegration:
    """Integration tests for SummarizationService"""

    @patch("backend.services.summarization_service.get_bert_classifier")
    @patch("backend.services.summarization_service.wikipedia.search")
    @patch("backend.services.summarization_service.wikipedia.page")
    @patch("backend.services.summarization_service.wikipedia.summary")
    def test_full_workflow_with_intent(
        self, mock_summary, mock_page, mock_search, mock_get_classifier
    ):
        """Test complete workflow with intent classification and Wikipedia search"""
        # Setup BERT classifier
        mock_classifier = Mock()
        mock_classifier.predict.return_value = ("Technology", 0.9)
        mock_get_classifier.return_value = mock_classifier

        # Setup Wikipedia mocks
        mock_search.return_value = ["Artificial Intelligence"]
        mock_page_obj = Mock()
        mock_page_obj.url = "https://en.wikipedia.org/wiki/AI"
        mock_page.return_value = mock_page_obj
        mock_summary.return_value = "AI is a comprehensive field of study."

        with patch("backend.services.summarization_service.BERT_AVAILABLE", True):
            service = SummarizationService()
            service.bert_model_loaded = True
            service.bert_classifier = mock_classifier

            result = service.summarize_single_source(
                "artificial intelligence", use_intent=True
            )

            assert result["query"] == "artificial intelligence"
            assert result["summary"] == "AI is a comprehensive field of study."
            assert result["intent"]["category"] == "Technology"
            assert result["intent"]["confidence"] == 0.9

    def test_service_resilience_with_partial_failures(self):
        """Test service resilience when some components fail"""
        service = SummarizationService()

        # Even if BERT fails, basic functionality should work
        with patch.object(service, "_search_wikipedia") as mock_search:
            mock_search.return_value = {
                "status": "success",
                "title": "Test",
                "summary": "Test summary",
                "url": "https://test.com",
                "query": "test",
            }

            result = service.summarize_single_source("test", use_intent=False)

            assert result["summary"] == "Test summary"
            assert "error" not in result
