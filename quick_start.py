#!/usr/bin/env python3
"""
Quick Start Script for SummarEaseAI

This script sets up the environment and starts the application.
Run 'pip install -r requirements.txt' first to install dependencies.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """Print the application banner"""
    print("=" * 60)
    print("🚀 SummarEaseAI v2.0 - Quick Start")
    print("AI-Powered Wikipedia Summarization")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if core dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = {
        'streamlit': 'Streamlit frontend',
        'flask': 'Flask backend API',
        'requests': 'HTTP requests',
        'wikipedia': 'Wikipedia API'
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package} - {description}")
        except ImportError:
            print(f"  ❌ {package} - {description}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    print("✅ All core dependencies are installed!")
    return True

def setup_environment():
    """Set up environment variables"""
    print("\n⚙️ Setting up environment...")
    
    env_file = Path(".env")
    if not env_file.exists():
        # Create basic .env file
        env_content = """# SummarEaseAI Configuration

# OpenAI API (optional - for cloud summarization)
# OPENAI_API_KEY=your_openai_api_key_here

# Flask Configuration
FLASK_DEBUG=false
PORT=5000

# Model Configuration (optional)
# HF_SUMMARIZATION_MODEL=facebook/bart-large-cnn
# HF_INTENT_MODEL=bert-base-uncased
# HF_EMBEDDING_MODEL=all-MiniLM-L6-v2
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env configuration file")
    else:
        print("✅ Environment file already exists")
    
    return True

def check_optional_features():
    """Check which optional AI features are available"""
    print("\n🔍 Checking optional AI features...")
    
    features = {
        "Core Flask API": True,
        "Streamlit Frontend": True,
        "Wikipedia Integration": True,
        "Basic Intent Classification": True,  # Fallback always available
    }
    
    # Check for DirectML GPU support
    directml_available = False
    try:
        import tensorflow as tf
        devices = tf.config.list_physical_devices('GPU')
        if devices and any('DML' in str(device) for device in devices):
            directml_available = True
            features["🚀 DirectML GPU (RTX 4070)"] = True
        else:
            features["GPU Acceleration"] = False
    except:
        features["GPU Acceleration"] = False
    
    # Check optional AI/ML features with better error handling
    try:
        import tensorflow as tf
        features["TensorFlow Models"] = True
    except ImportError:
        features["TensorFlow Models"] = False
    except Exception as e:
        print(f"  ⚠️ TensorFlow issue: {str(e)[:80]}...")
        features["TensorFlow Models"] = False
    
    try:
        import transformers
        # If DirectML is available, Transformers might conflict
        if directml_available:
            print(f"  ⚠️ Hugging Face disabled for DirectML compatibility")
            features["🤗 Hugging Face (DirectML disabled)"] = False
        else:
            features["🤗 Hugging Face Transformers"] = True
    except ImportError:
        features["🤗 Hugging Face Transformers"] = False
    except Exception as e:
        print(f"  ⚠️ Transformers compatibility issue detected")
        if "tf-keras" in str(e):
            print(f"  💡 Fix: pip install tf-keras")
        features["🤗 Hugging Face Transformers"] = False
    
    try:
        import langchain
        features["LangChain (OpenAI)"] = True
    except ImportError:
        features["LangChain (OpenAI)"] = False
    except Exception as e:
        print(f"  ⚠️ LangChain issue: {str(e)[:50]}...")
        features["LangChain (OpenAI)"] = False
    
    try:
        from sentence_transformers import SentenceTransformer
        features["Semantic Search"] = True
    except ImportError:
        features["Semantic Search"] = False
    except Exception as e:
        print(f"  ⚠️ Sentence Transformers issue: {str(e)[:50]}...")
        features["Semantic Search"] = False
    
    # Print feature status
    for feature, available in features.items():
        status = "✅" if available else "⚠️"
        print(f"  {status} {feature}")
    
    # Show installation hints for missing features
    missing_features = [f for f, available in features.items() if not available and f != "Core Flask API"]
    if missing_features:
        print("\n💡 To enable missing features, they're included in requirements.txt:")
        print("   pip install -r requirements.txt")
    
    # DirectML compatibility info
    if directml_available:
        print("\n🚀 DirectML GPU detected - using compatibility mode:")
        print("   • GPU training: ✅ Available for TensorFlow models")
        print("   • Backend: ✅ Uses DirectML-compatible version")
        print("   • Hugging Face: ⚠️ Disabled to avoid conflicts")
    
    return features

def start_application():
    """Start the SummarEaseAI application"""
    print("\n🚀 Starting SummarEaseAI...")
    
    # Check if we're in the right directory
    if not Path("backend/api_simple.py").exists():
        print("❌ Please run this script from the SummarEaseAI root directory")
        return False
    
    print("\n📋 Application URLs:")
    print("  🔗 Backend API: http://localhost:5000")
    print("  🔗 Frontend UI: http://localhost:8501")
    
    print("\n💡 Manual startup commands:")
    print("  Terminal 1: python backend/api_simple.py  (DirectML compatible)")
    print("  Terminal 2: streamlit run frontend/app.py")
    
    # Wait for user input
    input("\nPress Enter to continue...")
    
    print("\n🔄 Starting DirectML-compatible backend API...")
    
    # Start the backend API
    import subprocess
    backend_process = subprocess.Popen([
        sys.executable, "backend/api.py"
    ])
    
    print("💡 After backend starts, run in another terminal: streamlit run frontend/app.py")
    
    # Ask user how they want to start
    print("\nStartup options:")
    print("  1. Start backend only (you start frontend manually)")
    print("  2. Show commands only (manual startup)")
    print("  3. Exit")
    
    try:
        choice = input("\nChoose option (1-3): ").strip()
        
        if choice == "1":
            print("\n🔄 Starting DirectML-compatible backend API...")
            print("💡 After backend starts, run in another terminal: streamlit run frontend/app.py")
            print("🛑 Press Ctrl+C to stop the backend")
            
            # Start the DirectML-compatible API
            subprocess.call([sys.executable, "backend/api_simple.py"])
            
        elif choice == "2":
            print("\n✅ Use the manual commands above to start both services")
            
        elif choice == "3":
            print("\n👋 Goodbye!")
            
        else:
            print("\n✅ Use the manual commands above to start both services")
            
    except KeyboardInterrupt:
        print("\n\n✅ Setup complete!")
    
    return True

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check if dependencies are installed
    if not check_dependencies():
        return False
    
    # Setup environment
    if not setup_environment():
        return False
    
    # Check available features
    features = check_optional_features()
    
    # Show usage information
    print("\n📚 How SummarEaseAI works:")
    print("  • App works immediately with fallback mechanisms")
    print("  • Intent classification uses rules if TensorFlow model not trained")
    print("  • OpenAI summarization requires OPENAI_API_KEY in .env")
    print("  • Hugging Face models download automatically on first use")
    print("  • All features gracefully degrade if dependencies missing")
    
    # Start application
    start_application()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 SummarEaseAI setup complete!")
        else:
            print("\n❌ Setup failed. Please check the errors above.")
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your Python installation and try again.") 