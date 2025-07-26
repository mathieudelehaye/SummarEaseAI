"""
Pytest configuration and shared fixtures for SummarEaseAI tests
"""

import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test OpenAI response"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_bert_classifier():
    """Mock BERT classifier for testing"""
    mock_classifier = Mock()
    mock_classifier.load_model.return_value = True
    mock_classifier.predict.return_value = ("Technology", 0.85)
    mock_classifier.is_loaded.return_value = True
    mock_classifier.get_performance_stats.return_value = {
        "total_predictions": 100,
        "avg_inference_time": 0.15,
        "confidence_distribution": {"high": 70, "medium": 25, "low": 5},
    }
    return mock_classifier


@pytest.fixture
def mock_intent_classifier():
    """Mock TensorFlow intent classifier for testing"""
    mock_classifier = Mock()
    mock_classifier.load_model.return_value = True
    mock_classifier.predict_intent.return_value = ("Science", 0.92)
    mock_classifier.predict_intent_fallback.return_value = ("General", 0.6)
    mock_classifier.intent_categories = [
        "History",
        "Science",
        "Biography",
        "Technology",
        "Arts",
        "Sports",
        "Politics",
        "Geography",
        "General",
    ]
    return mock_classifier


@pytest.fixture
def sample_wikipedia_article():
    """Sample Wikipedia article data for testing"""
    return {
        "title": "Artificial Intelligence",
        "content": 'Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents".',
        "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "summary": "A brief summary about AI.",
        "length": 150,
    }


@pytest.fixture
def sample_intent_data():
    """Sample intent classification data"""
    return [
        ("Tell me about quantum physics", "Science"),
        ("Who was Albert Einstein?", "Biography"),
        ("What happened in World War II?", "History"),
        ("How do computers work?", "Technology"),
        ("Olympic Games history", "Sports"),
        ("Renaissance art movement", "Arts"),
        ("Democracy principles", "Politics"),
        ("Where is Mount Everest?", "Geography"),
        ("General information", "General"),
    ]


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables"""
    os.environ["TESTING"] = "true"
    # Mock API keys for testing
    os.environ["OPENAI_API_KEY"] = "test-key-12345"
    yield
    # Cleanup
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
