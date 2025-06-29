# SummarEaseAI Test Runner (PowerShell)
# Script to run unit tests with proper environment setup on Windows

Write-Host "🧪 SummarEaseAI Test Suite" -ForegroundColor Cyan
Write-Host "==========================" -ForegroundColor Cyan

# Set environment variables for testing
$env:TESTING = "true"
$env:OPENAI_API_KEY = "test-key-12345"
$env:CUDA_VISIBLE_DEVICES = "-1"  # Disable GPU for tests by default

# Check if pytest is installed
$pytestInstalled = $false
try {
    $pytestVersion = & pytest --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ pytest is available" -ForegroundColor Green
        $pytestInstalled = $true
    }
}
catch {
    # pytest not found
}

if (-not $pytestInstalled) {
    Write-Host "📦 Installing test dependencies..." -ForegroundColor Yellow
    & pip install pytest pytest-cov pytest-mock
}

# Create tests directory if it doesn't exist
if (-not (Test-Path "tests")) {
    New-Item -ItemType Directory -Path "tests" | Out-Null
}

Write-Host "🏃 Running tests..." -ForegroundColor Yellow
Write-Host "📁 Test directory: $((Get-Location).Path)\tests" -ForegroundColor Gray

# Get test type from argument (default to "all")
$TestType = if ($args.Count -gt 0) { $args[0] } else { "all" }

switch ($TestType.ToLower()) {
    "unit" {
        Write-Host "🎯 Running unit tests only..." -ForegroundColor Magenta
        & pytest tests/ -v -m "unit" --tb=short
    }
    "api" {
        Write-Host "🎯 Running API tests only..." -ForegroundColor Magenta
        & pytest tests/test_api_endpoints.py -v --tb=short
    }
    "integration" {
        Write-Host "🎯 Running integration tests..." -ForegroundColor Magenta
        & pytest tests/ -v -m "integration" --tb=short
    }
    "fast" {
        Write-Host "🎯 Running fast tests (excluding slow ones)..." -ForegroundColor Magenta
        & pytest tests/ -v -m "not slow" --tb=short
    }
    "coverage" {
        Write-Host "🎯 Running tests with coverage..." -ForegroundColor Magenta
        & pytest tests/ -v --cov=backend --cov=tensorflow_models --cov=utils --cov-report=term-missing --cov-report=html:htmlcov
    }
    default {
        Write-Host "🎯 Running all tests..." -ForegroundColor Magenta
        & pytest tests/ -v --tb=short
    }
}

$TestResult = $LASTEXITCODE

Write-Host ""
Write-Host "==========================" -ForegroundColor Cyan
if ($TestResult -eq 0) {
    Write-Host "✅ Tests completed successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Some tests failed!" -ForegroundColor Red
}

# Show usage information
Write-Host ""
Write-Host "💡 Usage examples:" -ForegroundColor Yellow
Write-Host "  .\run_tests.ps1          # Run all tests" -ForegroundColor Gray
Write-Host "  .\run_tests.ps1 unit     # Run unit tests only" -ForegroundColor Gray
Write-Host "  .\run_tests.ps1 api      # Run API tests only" -ForegroundColor Gray
Write-Host "  .\run_tests.ps1 coverage # Run with coverage report" -ForegroundColor Gray

exit $TestResult 