# SummarEaseAI Test Suite

This directory contains comprehensive unit tests for the SummarEaseAI project, covering the most critical components of both backend and frontend functionality.

## 🧪 Test Structure

### Core Test Files

- **`conftest.py`** - Pytest configuration and shared fixtures
- **`test_bert_gpu_classifier.py`** - Tests for GPU BERT intent classification
- **`test_intent_classifier.py`** - Tests for TensorFlow LSTM intent classification  
- **`test_multi_source_agent.py`** - Tests for multi-source intelligence agent
- **`test_api_endpoints.py`** - Tests for Flask API endpoints
- **`test_wikipedia_utils.py`** - Tests for Wikipedia utility functions

### Test Categories

Tests are organized with pytest markers:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.api` - API endpoint tests
- `@pytest.mark.gpu` - GPU-dependent tests
- `@pytest.mark.slow` - Slower tests that can be skipped

## 🚀 How to Run Tests

### Prerequisites

1. **Install test dependencies:**
   ```bash
   pip install -r requirements_test.txt
   ```

2. **Or use the automated test runner:**
   ```bash
   python run_tests.py
   ```

### Running All Tests

```bash
# Method 1: Using the test runner (recommended)
python run_tests.py

# Method 2: Direct pytest
pytest tests/ -v --cov=backend --cov=tensorflow_models --cov=utils
```

### Running Specific Test Categories

```bash
# Unit tests only (fast)
python run_tests.py unit

# API tests only
python run_tests.py api

# Integration tests
python run_tests.py integration

# GPU tests (requires GPU)
python run_tests.py gpu

# Fast tests (exclude slow ones)
python run_tests.py fast

# Specific test file
python run_tests.py tests/test_bert_gpu_classifier.py
```

### Advanced Test Options

```bash
# Run with coverage report
pytest tests/ --cov=backend --cov-report=html

# Run specific test method
pytest tests/test_bert_gpu_classifier.py::TestGPUBERTClassifier::test_predict_success -v

# Run tests in parallel (faster)
pytest tests/ -n auto

# Run tests with detailed output
pytest tests/ -v -s
```

## 📊 Test Coverage

The test suite aims for **70%+ code coverage** on critical components:

### Primary Coverage Areas

1. **GPU BERT Classifier** (`tensorflow_models/bert_gpu_classifier.py`)
   - Model loading and initialization
   - Prediction accuracy and performance
   - Error handling and fallback behavior
   - GPU memory management

2. **TensorFlow Intent Classifier** (`tensorflow_models/intent_classifier.py`)
   - LSTM model building and training
   - Intent prediction with fallback
   - Model persistence (save/load)
   - Keyword-based classification

3. **Multi-Source Agent** (`utils/multi_source_agent.py`)
   - Search strategy planning
   - Rate limiting and cost control
   - Article gathering with agents
   - Intent analysis integration

4. **API Endpoints** (`backend/api.py`)
   - Request/response handling
   - Error handling and validation
   - JSON formatting
   - Authentication and CORS

5. **Wikipedia Utilities** (`utils/wikipedia_fetcher.py`)
   - Article fetching and search
   - Error handling (disambiguation, not found)
   - Content processing and sanitization
   - Query enhancement

## 🎯 What is Being Tested

### Functional Testing

- **Intent Classification Accuracy**: Both BERT and LSTM models
- **API Response Formats**: Consistent JSON responses
- **Error Handling**: Graceful failure modes
- **Wikipedia Integration**: Article fetching and processing
- **Multi-Source Intelligence**: Agent-powered search and synthesis

### Performance Testing

- **Model Loading Times**: GPU vs CPU performance
- **Prediction Speed**: Inference time benchmarks
- **Memory Usage**: GPU memory management
- **Rate Limiting**: Cost control mechanisms

### Integration Testing

- **End-to-End Workflows**: Complete user request flows
- **Component Interaction**: Model + API + utilities
- **External Service Mocking**: Wikipedia API, OpenAI API
- **Environment Configuration**: Test vs production settings

## 🔧 Test Configuration

### Environment Variables

Tests automatically set these environment variables:

```bash
TESTING=true                    # Enables test mode
OPENAI_API_KEY=test-key-12345  # Mock API key
CUDA_VISIBLE_DEVICES=-1        # Disable GPU by default
```

### Fixtures Available

- `temp_dir` - Temporary directory for test files
- `mock_openai_client` - Mocked OpenAI API client
- `mock_bert_classifier` - Mocked GPU BERT classifier
- `mock_intent_classifier` - Mocked TensorFlow classifier
- `sample_wikipedia_article` - Sample article data
- `sample_intent_data` - Sample intent classification data

## 📈 Coverage Reports

After running tests, coverage reports are generated:

- **Terminal**: Shows missing lines immediately
- **HTML Report**: `htmlcov/index.html` - Detailed interactive report

```bash
# View HTML coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html # Windows
xdg-open htmlcov/index.html # Linux
```

## 🐛 Debugging Tests

### Running Tests in Debug Mode

```bash
# Run with Python debugger
pytest tests/ --pdb

# Run with detailed output
pytest tests/ -v -s --tb=long

# Run single test with maximum detail
pytest tests/test_bert_gpu_classifier.py::TestGPUBERTClassifier::test_predict_success -v -s --tb=long
```

### Common Issues

1. **Import Errors**: Ensure project root is in Python path
2. **Missing Dependencies**: Run `pip install -r requirements_test.txt`
3. **GPU Tests Failing**: Set `CUDA_VISIBLE_DEVICES=-1` to disable GPU
4. **API Key Errors**: Tests use mock keys, check environment setup

## 🔄 Continuous Integration

The test suite is designed to work in CI/CD environments:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    pip install -r requirements_test.txt
    python run_tests.py all
```

## 📝 Adding New Tests

### Test File Structure

```python
"""
Unit tests for [Component Name]
Tests the [brief description of functionality]
"""
import pytest
from unittest.mock import Mock, patch

class Test[ComponentName]:
    """Test cases for [ComponentName]"""
    
    def test_[functionality]_success(self):
        """Test successful [functionality]"""
        # Arrange
        # Act  
        # Assert
        
    def test_[functionality]_error(self):
        """Test [functionality] error handling"""
        # Test error cases
```

### Best Practices

1. **Use descriptive test names** that explain what is being tested
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **Mock external dependencies** (APIs, file systems, etc.)
4. **Test both success and failure cases**
5. **Use appropriate pytest markers** for categorization
6. **Keep tests independent** - no shared state between tests

## 📞 Support

If you encounter issues with the test suite:

1. Check the test logs for specific error messages
2. Verify all dependencies are installed
3. Ensure environment variables are set correctly
4. Run tests individually to isolate issues
5. Check the coverage report to identify untested code paths

The test suite is designed to be comprehensive, fast, and reliable - helping ensure the quality and stability of the SummarEaseAI project. 