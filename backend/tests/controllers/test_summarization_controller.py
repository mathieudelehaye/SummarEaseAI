"""
Unit tests for Summarization Controller
Tests the controller layer business logic
"""

from backend.controllers.request_validation import validate_multi_source_request


class TestSummarizationController:
    """Test cases for SummarizationController"""

    def test_validate_multi_source_request_valid(self):
        """Test validation of valid multi-source request"""
        request_data = {
            "query": "test query",
            "cost_mode": "BALANCED",
        }

        result = validate_multi_source_request(request_data)
        assert result is None  # No validation error

    def test_validate_multi_source_request_missing_query(self):
        """Test validation with missing query"""
        request_data = {}

        error_response, status_code = validate_multi_source_request(request_data)
        assert status_code == 400
        assert error_response["error"] == "Missing query parameter"

    def test_validate_multi_source_request_empty_query(self):
        """Test validation with empty query"""
        request_data = {"query": "   "}

        error_response, status_code = validate_multi_source_request(request_data)
        assert status_code == 400
        assert error_response["error"] == "Empty query provided"

    def test_validate_multi_source_request_invalid_max_lines(self):
        """Test validation with invalid max_lines"""
        request_data = {"query": "test", "max_lines": 150}  # Too high

        error_response, status_code = validate_multi_source_request(request_data)
        assert status_code == 400
        assert "max_lines must be between 5 and 100" in error_response["error"]

    def test_validate_multi_source_request_invalid_max_articles(self):
        """Test validation with invalid max_articles"""
        request_data = {"query": "test", "max_articles": 15}  # Too high

        error_response, status_code = validate_multi_source_request(request_data)
        assert status_code == 400
        assert "max_articles must be between 1 and 10" in error_response["error"]

    def test_validate_multi_source_request_invalid_cost_mode(self):
        """Test validation with invalid cost_mode"""
        request_data = {"query": "test", "cost_mode": "INVALID"}

        error_response, status_code = validate_multi_source_request(request_data)
        assert status_code == 400
        assert (
            "cost_mode must be MINIMAL, BALANCED, or COMPREHENSIVE"
            in error_response["error"]
        )
