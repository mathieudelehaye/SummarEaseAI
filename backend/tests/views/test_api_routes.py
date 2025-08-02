"""
Unit tests for Flask API endpoints
Tests the critical backend API functionality
"""

import json
from unittest.mock import patch

import pytest

# Import Flask app
from backend.views.api_routes import app


class TestAPIEndpoints:
    """Test cases for API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_home_endpoint(self, client):
        """Test the home endpoint"""
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.content_type
        assert b"SummarEaseAI Backend API" in response.data

    def test_health_endpoint(self, client):
        """Test the health check endpoint"""
        response = client.get("/health")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert data["backend"] == "running"
        assert "bert_model" in data

    def test_status_endpoint(self, client):
        """Test the status endpoint"""
        response = client.get("/status")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "running"
        assert "features" in data
        assert "endpoints" in data
        assert isinstance(data["endpoints"], list)

    @patch("backend.views.api_routes.SUMMARIZATION_CONTROLLER")
    def test_intent_bert_endpoint_success(self, mock_controller, client):
        """Test successful BERT intent prediction"""
        # Mock service response
        mock_controller.classify_intent.return_value = {
            "intent": "Science",
            "confidence": 0.92,
            "model_type": "BERT",
            "timestamp": "2024-01-01T00:00:00Z",
        }

        response = client.post(
            "/intent_bert",
            json={"text": "Tell me about quantum physics"},
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["intent"] == "Science"
        assert data["confidence"] == 0.92
        assert data["model_type"] == "BERT"

    @patch("backend.views.api_routes.SUMMARIZATION_CONTROLLER")
    def test_intent_bert_endpoint_model_not_loaded(self, mock_controller, client):
        """Test BERT intent endpoint when model is not loaded"""
        # Mock service response with error
        mock_controller.classify_intent.return_value = {
            "error": "BERT model not loaded"
        }

        response = client.post(
            "/intent_bert",
            json={"text": "Tell me about quantum physics"},
            content_type="application/json",
        )

        assert response.status_code == 503
        data = json.loads(response.data)
        assert "error" in data

    @patch("backend.views.api_routes.SUMMARIZATION_CONTROLLER")
    def test_summarize_endpoint_success(self, mock_controller, client):
        """Test successful single-source summarization"""
        # Mock service response
        mock_controller.summarize_single_source.return_value = {
            "query": "Apollo 11",
            "summary": "Apollo 11 was the first manned mission to land on the Moon.",
            "intent": {"category": "History", "confidence": 0.85},
            "method": "single_source",
            "total_sources": 1,
            "summary_length": 500,
            "summary_lines": 5,
        }

        response = client.post(
            "/summarize",
            json={"query": "Apollo 11", "max_lines": 30},
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["query"] == "Apollo 11"
        assert "summary" in data
        assert data["intent"] == "History"

    def test_summarize_endpoint_missing_query(self, client):
        """Test summarize endpoint with missing query"""
        response = client.post("/summarize", json={}, content_type="application/json")

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
        assert "Missing query parameter" in data["error"]

    def test_summarize_endpoint_empty_query(self, client):
        """Test summarize endpoint with empty query"""
        response = client.post(
            "/summarize", json={"query": ""}, content_type="application/json"
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
        assert "Empty query provided" in data["error"]

    @patch("backend.views.api_routes.SUMMARIZATION_CONTROLLER")
    def test_summarize_multi_source_success(self, mock_controller, client):
        """Test successful multi-source summarization"""
        # Mock service response
        mock_controller.summarize_multi_source_with_agents.return_value = {
            "query": "Space exploration",
            "summary": "Space exploration has been a major human endeavor.",
            "intent": "Science",
            "confidence": 0.88,
            "method": "multi_source",
            "total_sources": 3,
            "summary_length": 800,
            "summary_lines": 8,
            "agent_powered": True,
            "articles": [
                {
                    "title": "Space Exploration",
                    "url": "https://example.com/1",
                    "selection_method": "relevance",
                },
                {
                    "title": "NASA Missions",
                    "url": "https://example.com/2",
                    "selection_method": "relevance",
                },
                {
                    "title": "Space Technology",
                    "url": "https://example.com/3",
                    "selection_method": "relevance",
                },
            ],
            "usage_stats": {"tokens_used": 1500},
            "cost_tracking": {"total_cost": 0.05},
        }

        response = client.post(
            "/summarize_multi_source",
            json={"query": "Space exploration", "max_lines": 30, "max_articles": 3},
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["query"] == "Space exploration"
        assert "summary" in data
        assert data["intent"] == "Science"
        assert len(data["articles"]) == 3

    @patch("backend.views.api_routes.SUMMARIZATION_CONTROLLER")
    def test_summarize_multi_source_error(self, mock_controller, client):
        """Test multi-source summarization with error"""
        # Mock service response with error
        mock_controller.summarize_multi_source_with_agents.return_value = {
            "error": "No articles found for query"
        }

        response = client.post(
            "/summarize_multi_source",
            json={"query": "Invalid query", "max_lines": 30},
            content_type="application/json",
        )

        assert response.status_code == 404
        data = json.loads(response.data)
        assert "error" in data


class TestAPIErrorHandling:
    """Test error handling in API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    @patch("backend.views.api_routes.SUMMARIZATION_CONTROLLER")
    def test_intent_bert_endpoint_exception(self, mock_controller, client):
        """Test BERT intent endpoint when classifier raises exception"""
        mock_controller.classify_intent.side_effect = Exception("Service error")

        response = client.post(
            "/intent_bert", json={"text": "test text"}, content_type="application/json"
        )

        assert response.status_code == 500
        data = json.loads(response.data)
        assert "error" in data
        assert "Internal server error" in data["error"]

    @patch("backend.views.api_routes.SUMMARIZATION_CONTROLLER")
    def test_summarize_endpoint_exception(self, mock_controller, client):
        """Test summarize endpoint when summarizer raises exception"""
        mock_controller.summarize_single_source.side_effect = Exception("Service error")

        response = client.post(
            "/summarize", json={"query": "test query"}, content_type="application/json"
        )

        assert response.status_code == 500
        data = json.loads(response.data)
        assert "error" in data


class TestAPIResponseFormat:
    """Test API response formatting"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    @patch("backend.views.api_routes.SUMMARIZATION_CONTROLLER")
    def test_intent_bert_response_format(self, mock_controller, client):
        """Test that BERT intent response has correct format"""
        mock_controller.classify_intent.return_value = {
            "intent": "Science",
            "confidence": 0.92,
            "model_type": "BERT",
            "model_loaded": True,
            "categories_available": ["Science", "Technology", "History"],
            "gpu_accelerated": True,
            "timestamp": "2024-01-01T00:00:00Z",
        }

        response = client.post(
            "/intent_bert",
            json={"text": "quantum physics"},
            content_type="application/json",
        )

        data = json.loads(response.data)

        # Check required fields
        required_fields = [
            "text",
            "intent",
            "confidence",
            "model_type",
            "model_loaded",
            "categories_available",
            "gpu_accelerated",
            "timestamp",
        ]
        for field in required_fields:
            assert field in data

        # Check BERT-specific fields
        assert data["model_type"] == "BERT"
        assert data["gpu_accelerated"] is True

    def test_cors_headers(self, client):
        """Test that CORS headers are present"""
        response = client.get("/health")

        # CORS headers should be present due to CORS(app)
        assert response.status_code == 200


class TestAPIIntegration:
    """Integration tests for API functionality"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_api_consistency(self, client):
        """Test that API endpoints are consistent"""
        # Test that all endpoints return JSON (except home)
        json_endpoints = ["/health", "/status"]

        for endpoint in json_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
            assert "application/json" in response.content_type

    def test_error_response_format(self, client):
        """Test that error responses have consistent format"""
        # Test various error conditions
        error_cases = [
            ("/intent_bert", {}),  # Missing text
            ("/summarize", {}),  # Missing query
            ("/summarize_multi_source", {}),  # Missing query
        ]

        for endpoint, payload in error_cases:
            response = client.post(
                endpoint, json=payload, content_type="application/json"
            )

            # All endpoints should return 400 for missing required fields
            assert response.status_code == 400
            data = json.loads(response.data)
            assert "error" in data
            assert isinstance(data["error"], str)
