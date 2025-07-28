"""
Unit tests for Summarization Controller
Tests the controller layer business logic
"""

from unittest.mock import Mock, patch

from backend.controllers.summarization_controller import (
    SummarizationController,
    format_summarization_response,
    get_summarization_controller,
)


class TestSummarizationController:
    """Test cases for SummarizationController"""

    @patch("backend.controllers.summarization_controller.get_summarization_service")
    def test_init(self, mock_get_service):
        """Test controller initialization"""
        mock_service = Mock()
        mock_get_service.return_value = mock_service

        controller = SummarizationController()
        assert controller.summarization_service == mock_service

    def test_validate_multi_source_request_valid(self):
        """Test validation of valid multi-source request"""
        controller = SummarizationController()
        request_data = {
            "query": "test query",
            "max_lines": 25,
            "max_articles": 3,
            "cost_mode": "BALANCED",
        }

        result = controller._validate_multi_source_request(request_data)
        assert result is None  # No validation error

    def test_validate_multi_source_request_missing_query(self):
        """Test validation with missing query"""
        controller = SummarizationController()
        request_data = {}

        error_response, status_code = controller._validate_multi_source_request(
            request_data
        )
        assert status_code == 400
        assert error_response["error"] == "Missing query parameter"

    def test_validate_multi_source_request_empty_query(self):
        """Test validation with empty query"""
        controller = SummarizationController()
        request_data = {"query": "   "}

        error_response, status_code = controller._validate_multi_source_request(
            request_data
        )
        assert status_code == 400
        assert error_response["error"] == "Missing query parameter"

    def test_validate_multi_source_request_invalid_max_lines(self):
        """Test validation with invalid max_lines"""
        controller = SummarizationController()
        request_data = {"query": "test", "max_lines": 150}  # Too high

        error_response, status_code = controller._validate_multi_source_request(
            request_data
        )
        assert status_code == 400
        assert "max_lines must be between 5 and 100" in error_response["error"]

    def test_validate_multi_source_request_invalid_max_articles(self):
        """Test validation with invalid max_articles"""
        controller = SummarizationController()
        request_data = {"query": "test", "max_articles": 15}  # Too high

        error_response, status_code = controller._validate_multi_source_request(
            request_data
        )
        assert status_code == 400
        assert "max_articles must be between 1 and 10" in error_response["error"]

    def test_validate_multi_source_request_invalid_cost_mode(self):
        """Test validation with invalid cost_mode"""
        controller = SummarizationController()
        request_data = {"query": "test", "cost_mode": "INVALID"}

        error_response, status_code = controller._validate_multi_source_request(
            request_data
        )
        assert status_code == 400
        assert (
            "cost_mode must be MINIMAL, BALANCED, or COMPREHENSIVE"
            in error_response["error"]
        )

    @patch("backend.controllers.summarization_controller.get_summarization_service")
    def test_handle_multi_source_request_success(self, mock_get_service):
        """Test successful multi-source request handling"""
        mock_service = Mock()
        mock_service.summarize_multi_source_with_agents.return_value = {
            "query": "test query",
            "summary": "Test summary",
            "method": "multi_source",
            "articles": [{"title": "Test", "url": "test.com"}],
        }
        mock_get_service.return_value = mock_service

        controller = SummarizationController()
        request_data = {
            "query": "test query",
            "max_lines": 30,
            "max_articles": 3,
            "cost_mode": "BALANCED",
        }

        response, status_code = controller.handle_multi_source_request(request_data)

        assert status_code == 200
        assert response["query"] == "test query"
        assert response["summary"] == "Test summary"
        assert "metadata" in response
        assert response["metadata"]["cost_mode"] == "BALANCED"

    @patch("backend.controllers.summarization_controller.get_summarization_service")
    def test_handle_multi_source_request_service_error(self, mock_get_service):
        """Test multi-source request when service returns error"""
        mock_service = Mock()
        mock_service.summarize_multi_source_with_agents.return_value = {
            "error": "Service error occurred"
        }
        mock_get_service.return_value = mock_service

        controller = SummarizationController()
        request_data = {"query": "test query"}

        response, status_code = controller.handle_multi_source_request(request_data)

        assert status_code == 500
        assert response["error"] == "Service error occurred"

    @patch("backend.controllers.summarization_controller.get_summarization_service")
    def test_handle_multi_source_request_exception(self, mock_get_service):
        """Test multi-source request when service raises exception"""
        mock_service = Mock()
        mock_service.summarize_multi_source_with_agents.side_effect = ValueError(
            "Service exception"
        )
        mock_get_service.return_value = mock_service

        controller = SummarizationController()
        request_data = {"query": "test query"}

        response, status_code = controller.handle_multi_source_request(request_data)

        assert status_code == 500
        assert response["error"] == "Internal server error"
        assert "Service exception" in response["details"]

    def test_handle_multi_source_request_validation_error(self):
        """Test multi-source request with validation error"""
        controller = SummarizationController()
        request_data = {}  # Missing query

        response, status_code = controller.handle_multi_source_request(request_data)

        assert status_code == 400
        assert response["error"] == "Missing query parameter"

    @patch("backend.controllers.summarization_controller.get_summarization_service")
    def test_handle_single_source_request_success(self, mock_get_service):
        """Test successful single-source request handling"""
        mock_service = Mock()
        mock_service.summarize_single_source.return_value = {
            "query": "test query",
            "summary": "Test summary",
            "method": "single_source",
            "article": {"title": "Test Article", "url": "test.com"},
        }
        mock_get_service.return_value = mock_service

        controller = SummarizationController()
        request_data = {"query": "test query", "max_lines": 25}

        response, status_code = controller.handle_single_source_request(request_data)

        assert status_code == 200
        assert response["query"] == "test query"
        assert response["summary"] == "Test summary"
        assert "metadata" in response
        assert response["article"]["title"] == "Test Article"

    @patch("backend.controllers.summarization_controller.get_summarization_service")
    def test_handle_single_source_request_not_found(self, mock_get_service):
        """Test single-source request when content not found"""
        mock_service = Mock()
        mock_service.summarize_single_source.return_value = {
            "error": "No Wikipedia content found"
        }
        mock_get_service.return_value = mock_service

        controller = SummarizationController()
        request_data = {"query": "nonexistent topic"}

        response, status_code = controller.handle_single_source_request(request_data)

        assert status_code == 404
        assert response["error"] == "No Wikipedia content found"

    @patch("backend.controllers.summarization_controller.get_summarization_service")
    def test_handle_single_source_request_server_error(self, mock_get_service):
        """Test single-source request with server error"""
        mock_service = Mock()
        mock_service.summarize_single_source.return_value = {
            "error": "Internal processing error"
        }
        mock_get_service.return_value = mock_service

        controller = SummarizationController()
        request_data = {"query": "test query"}

        response, status_code = controller.handle_single_source_request(request_data)

        assert status_code == 500
        assert response["error"] == "Internal processing error"

    def test_handle_single_source_request_missing_query(self):
        """Test single-source request with missing query"""
        controller = SummarizationController()
        request_data = {}

        response, status_code = controller.handle_single_source_request(request_data)

        assert status_code == 400
        assert response["error"] == "Missing query parameter"

    def test_handle_single_source_request_empty_query(self):
        """Test single-source request with empty query"""
        controller = SummarizationController()
        request_data = {"query": ""}

        response, status_code = controller.handle_single_source_request(request_data)

        assert status_code == 400
        assert response["error"] == "Missing query parameter"

    def test_handle_single_source_request_invalid_max_lines(self):
        """Test single-source request with invalid max_lines"""
        controller = SummarizationController()
        request_data = {"query": "test", "max_lines": 200}  # Too high

        response, status_code = controller.handle_single_source_request(request_data)

        assert status_code == 400
        assert "max_lines must be between 5 and 100" in response["error"]

    @patch("backend.controllers.summarization_controller.get_summarization_service")
    def test_handle_single_source_request_exception(self, mock_get_service):
        """Test single-source request when service raises exception"""
        mock_service = Mock()
        mock_service.summarize_single_source.side_effect = ConnectionError(
            "Network error"
        )
        mock_get_service.return_value = mock_service

        controller = SummarizationController()
        request_data = {"query": "test query"}

        response, status_code = controller.handle_single_source_request(request_data)

        assert status_code == 500
        assert response["error"] == "Internal server error"
        assert "Network error" in response["details"]


class TestFormatSummarizationResponse:
    """Test the format_summarization_response utility function"""

    def test_format_summarization_response_basic(self):
        """Test basic response formatting"""
        result = {
            "query": "test query",
            "summary": "test summary",
            "method": "test_method",
            "intent": {"category": "Technology", "confidence": 0.8},
        }
        cost_mode = "BALANCED"

        response = format_summarization_response(result, cost_mode)

        assert response["query"] == "test query"
        assert response["summary"] == "test summary"
        assert response["metadata"]["method"] == "test_method"
        assert response["metadata"]["cost_mode"] == "BALANCED"
        assert response["metadata"]["intent"]["category"] == "Technology"

    def test_format_summarization_response_with_articles(self):
        """Test response formatting with articles"""
        result = {
            "query": "test query",
            "summary": "test summary",
        }
        articles = [
            {"title": "Article 1", "url": "url1.com", "selection_method": "agent"},
            {"title": "Article 2", "url": "url2.com", "selection_method": "search"},
        ]

        response = format_summarization_response(result, "COMPREHENSIVE", articles)

        assert len(response["articles"]) == 2
        assert response["articles"][0]["title"] == "Article 1"
        assert response["articles"][0]["selection_method"] == "agent"

    def test_format_summarization_response_missing_fields(self):
        """Test response formatting with missing optional fields"""
        result = {"query": "test query", "summary": "test summary"}

        response = format_summarization_response(result, "MINIMAL")

        assert response["metadata"]["intent"] is None
        assert response["metadata"]["confidence"] is None
        assert response["metadata"]["total_sources"] == 0
        assert response["metadata"]["cost_mode"] == "MINIMAL"

    def test_format_summarization_response_with_all_metadata(self):
        """Test response formatting with all metadata fields"""
        result = {
            "query": "test query",
            "summary": "test summary",
            "intent": {"category": "Science", "confidence": 0.9},
            "method": "multi_source",
            "total_sources": 3,
            "summary_length": 150,
            "summary_lines": 8,
            "agent_powered": True,
        }

        response = format_summarization_response(result, "COMPREHENSIVE")

        metadata = response["metadata"]
        assert metadata["intent"]["category"] == "Science"
        assert metadata["intent"]["confidence"] == 0.9
        assert metadata["method"] == "multi_source"
        assert metadata["total_sources"] == 3
        assert metadata["summary_length"] == 150
        assert metadata["summary_lines"] == 8
        assert metadata["agent_powered"] is True


class TestSummarizationControllerSingleton:
    """Test singleton pattern for SummarizationController"""

    def test_get_summarization_controller_singleton(self):
        """Test that get_summarization_controller returns same instance"""
        controller1 = get_summarization_controller()
        controller2 = get_summarization_controller()
        assert controller1 is controller2

    def test_singleton_instance_type(self):
        """Test that singleton returns correct type"""
        controller = get_summarization_controller()
        assert isinstance(controller, SummarizationController)


class TestSummarizationControllerIntegration:
    """Integration tests for SummarizationController"""

    @patch("backend.controllers.summarization_controller.get_summarization_service")
    def test_multi_source_workflow_complete(self, mock_get_service):
        """Test complete multi-source workflow"""
        mock_service = Mock()
        mock_service.summarize_multi_source_with_agents.return_value = {
            "query": "artificial intelligence",
            "summary": "AI is a comprehensive field...",
            "method": "multi_source_agent",
            "intent": {"category": "Technology", "confidence": 0.9},
            "total_sources": 3,
            "summary_length": 200,
            "agent_powered": True,
            "articles": [
                {"title": "AI Overview", "url": "ai.com", "selection_method": "agent"},
                {
                    "title": "Machine Learning",
                    "url": "ml.com",
                    "selection_method": "search",
                },
            ],
        }
        mock_get_service.return_value = mock_service

        controller = SummarizationController()
        request_data = {
            "query": "artificial intelligence",
            "max_lines": 30,
            "max_articles": 3,
            "cost_mode": "BALANCED",
        }

        response, status_code = controller.handle_multi_source_request(request_data)

        assert status_code == 200
        assert response["query"] == "artificial intelligence"
        assert response["summary"] == "AI is a comprehensive field..."
        assert len(response["articles"]) == 2
        assert response["metadata"]["agent_powered"] is True
        assert response["metadata"]["cost_mode"] == "BALANCED"

    @patch("backend.controllers.summarization_controller.get_summarization_service")
    def test_single_source_workflow_complete(self, mock_get_service):
        """Test complete single-source workflow"""
        mock_service = Mock()
        mock_service.summarize_single_source.return_value = {
            "query": "quantum physics",
            "summary": "Quantum physics is the study of...",
            "method": "single_source",
            "processed_query": "quantum physics",
            "was_converted": False,
            "model": "wikipedia",
            "summary_length": 150,
            "article": {
                "title": "Quantum Physics",
                "url": "https://en.wikipedia.org/wiki/Quantum_physics",
            },
        }
        mock_get_service.return_value = mock_service

        controller = SummarizationController()
        request_data = {"query": "quantum physics", "max_lines": 25}

        response, status_code = controller.handle_single_source_request(request_data)

        assert status_code == 200
        assert response["query"] == "quantum physics"
        assert response["summary"] == "Quantum physics is the study of..."
        assert response["article"]["title"] == "Quantum Physics"
        assert response["metadata"]["processed_query"] == "quantum physics"
        assert response["metadata"]["was_converted"] is False

    def test_error_handling_consistency(self):
        """Test that error handling is consistent across methods"""
        controller = SummarizationController()

        # Test validation errors format
        multi_response, multi_status = controller.handle_multi_source_request({})
        single_response, single_status = controller.handle_single_source_request({})

        assert multi_status == 400
        assert single_status == 400
        assert "error" in multi_response
        assert "error" in single_response
        assert isinstance(multi_response["error"], str)
        assert isinstance(single_response["error"], str)
