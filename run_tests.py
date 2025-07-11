#!/usr/bin/env python3
"""
Test runner script for SummarEaseAI
Handles environment setup and runs pytest with appropriate configurations
"""
import os
import sys
import subprocess
from pathlib import Path

def setup_test_environment():
    """Setup environment variables for testing"""
    # Set testing flag
    os.environ['TESTING'] = 'true'
    
    # Mock API keys for testing (don't use real keys in tests)
    os.environ['OPENAI_API_KEY'] = 'test-key-12345'
    
    # Disable GPU for tests by default (can be overridden)
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Set Python path to include project root
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

def install_test_dependencies():
    """Install test dependencies if not present"""
    try:
        import pytest
        print("✅ pytest is already installed")
    except ImportError:
        print("📦 Installing test dependencies...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements_test.txt'
        ])

def run_tests(test_args=None):
    """Run pytest with appropriate configuration"""
    setup_test_environment()
    install_test_dependencies()
    
    # Default test arguments
    default_args = [
        '-v',                    # Verbose output
        '--tb=short',           # Short traceback format
        '--cov=backend',        # Coverage for backend
        '--cov=ml_models',  # Coverage for models
        '--cov=utils',          # Coverage for utilities
        '--cov-report=term-missing',  # Show missing lines
        '--cov-report=html:htmlcov',  # HTML coverage report
        'tests/'                # Test directory
    ]
    
    # Use provided args or defaults
    args = test_args or default_args
    
    print("🧪 Running SummarEaseAI unit tests...")
    print(f"📁 Test directory: {Path('tests').absolute()}")
    print(f"🔧 Test command: pytest {' '.join(args)}")
    print("-" * 60)
    
    try:
        # Run pytest
        result = subprocess.run([sys.executable, '-m', 'pytest'] + args)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("✅ All tests passed!")
            print("📊 Coverage report generated in htmlcov/index.html")
        else:
            print("\n" + "=" * 60)
            print("❌ Some tests failed!")
            
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

def run_specific_tests():
    """Run specific test categories"""
    print("🎯 Test Categories Available:")
    print("1. Unit tests only: python run_tests.py unit")
    print("2. API tests only: python run_tests.py api") 
    print("3. Integration tests: python run_tests.py integration")
    print("4. GPU tests: python run_tests.py gpu")
    print("5. Fast tests (exclude slow): python run_tests.py fast")
    print("6. All tests: python run_tests.py all")
    print("7. Specific file: python run_tests.py tests/test_specific_file.py")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "unit":
            # Run only unit tests
            args = ['-v', '-m', 'unit', 'tests/']
            run_tests(args)
        elif test_type == "api":
            # Run only API tests
            args = ['-v', '-m', 'api', 'tests/test_api_endpoints.py']
            run_tests(args)
        elif test_type == "integration":
            # Run integration tests
            args = ['-v', '-m', 'integration', 'tests/']
            run_tests(args)
        elif test_type == "gpu":
            # Run GPU tests (enable GPU)
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            args = ['-v', '-m', 'gpu', 'tests/']
            run_tests(args)
        elif test_type == "fast":
            # Run fast tests only
            args = ['-v', '-m', 'not slow', 'tests/']
            run_tests(args)
        elif test_type == "all":
            # Run all tests
            run_tests()
        elif test_type == "help":
            run_specific_tests()
        elif test_type.startswith("tests/"):
            # Run specific test file
            args = ['-v', test_type]
            run_tests(args)
        else:
            print(f"❌ Unknown test type: {test_type}")
            run_specific_tests()
    else:
        # Run all tests by default
        run_tests() 