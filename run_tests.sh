#!/bin/bash

# SummarEaseAI Test Runner
# Simple script to run unit tests with proper environment setup

echo "🧪 SummarEaseAI Test Suite"
echo "=========================="

# Set environment variables for testing
export TESTING=true
export OPENAI_API_KEY=test-key-12345
export CUDA_VISIBLE_DEVICES=-1  # Disable GPU for tests by default

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "📦 Installing test dependencies..."
    pip install pytest pytest-cov pytest-mock
fi

# Create tests directory if it doesn't exist
mkdir -p tests

echo "🏃 Running tests..."
echo "📁 Test directory: $(pwd)/tests"

# Run tests based on argument
case "${1:-all}" in
    "unit")
        echo "🎯 Running unit tests only..."
        pytest tests/ -v -m "unit" --tb=short
        ;;
    "api")
        echo "🎯 Running API tests only..."
        pytest tests/test_api_endpoints.py -v --tb=short
        ;;
    "integration")
        echo "🎯 Running integration tests..."
        pytest tests/ -v -m "integration" --tb=short
        ;;
    "fast")
        echo "🎯 Running fast tests (excluding slow ones)..."
        pytest tests/ -v -m "not slow" --tb=short
        ;;
    "coverage")
        echo "🎯 Running tests with coverage..."
        pytest tests/ -v --cov=backend --cov=tensorflow_models --cov=utils --cov-report=term-missing --cov-report=html:htmlcov
        ;;
    "all"|*)
        echo "🎯 Running all tests..."
        pytest tests/ -v --tb=short
        ;;
esac

test_result=$?

echo ""
echo "=========================="
if [ $test_result -eq 0 ]; then
    echo "✅ Tests completed successfully!"
else
    echo "❌ Some tests failed!"
fi

exit $test_result 