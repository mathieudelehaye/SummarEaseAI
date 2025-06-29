#!/usr/bin/env python3
"""
Simple cross-platform test runner for SummarEaseAI
Handles dependency installation and test execution
"""
import os
import sys
import subprocess
import argparse

def setup_environment():
    """Set up environment variables for testing"""
    os.environ['TESTING'] = 'true'
    os.environ['OPENAI_API_KEY'] = 'test-key-12345'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU by default
    print("✅ Test environment configured")

def install_dependencies():
    """Install pytest and required testing packages"""
    try:
        import pytest
        print("✅ pytest is already installed")
        return True
    except ImportError:
        print("📦 Installing pytest and test dependencies...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                'pytest>=7.0.0', 'pytest-cov>=4.0.0', 'pytest-mock>=3.0.0'
            ])
            print("✅ Test dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False

def run_pytest(test_args):
    """Run pytest with given arguments"""
    cmd = [sys.executable, '-m', 'pytest'] + test_args
    print(f"🏃 Running: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, cwd=os.getcwd())
        return result.returncode
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(description='SummarEaseAI Test Runner')
    parser.add_argument('test_type', nargs='?', default='all', 
                       choices=['all', 'unit', 'api', 'integration', 'fast', 'coverage'],
                       help='Type of tests to run')
    parser.add_argument('--file', help='Specific test file to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("🧪 SummarEaseAI Test Suite")
    print("=" * 30)
    
    # Setup
    setup_environment()
    if not install_dependencies():
        return 1
    
    # Prepare test arguments
    test_args = []
    
    if args.verbose:
        test_args.extend(['-v'])
    
    test_args.extend(['--tb=short'])  # Short traceback format
    
    # Determine what to test
    if args.file:
        test_args.append(args.file)
        print(f"🎯 Running specific file: {args.file}")
    elif args.test_type == 'unit':
        test_args.extend(['-m', 'unit', 'tests/'])
        print("🎯 Running unit tests only")
    elif args.test_type == 'api':
        test_args.append('tests/test_api_endpoints.py')
        print("🎯 Running API tests only")
    elif args.test_type == 'integration':
        test_args.extend(['-m', 'integration', 'tests/'])
        print("🎯 Running integration tests")
    elif args.test_type == 'fast':
        test_args.extend(['-m', 'not slow', 'tests/'])
        print("🎯 Running fast tests (excluding slow ones)")
    elif args.test_type == 'coverage':
        test_args.extend([
            '--cov=backend', '--cov=tensorflow_models', '--cov=utils',
            '--cov-report=term-missing', '--cov-report=html:htmlcov',
            'tests/'
        ])
        print("🎯 Running tests with coverage analysis")
    else:  # all
        test_args.append('tests/')
        print("🎯 Running all tests")
    
    # Run tests
    result = run_pytest(test_args)
    
    print("\n" + "=" * 30)
    if result == 0:
        print("✅ All tests passed!")
        if args.test_type == 'coverage':
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("❌ Some tests failed!")
    
    print("\n💡 Usage examples:")
    print("  python test_runner.py                    # Run all tests")
    print("  python test_runner.py unit               # Run unit tests only")
    print("  python test_runner.py api                # Run API tests only")
    print("  python test_runner.py coverage           # Run with coverage")
    print("  python test_runner.py --file tests/test_bert_gpu_classifier.py")
    
    return result

if __name__ == '__main__':
    sys.exit(main()) 