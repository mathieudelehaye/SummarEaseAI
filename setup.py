#!/usr/bin/env python3
"""
SummarEaseAI Setup Script

This script helps you set up the SummarEaseAI project quickly by:
1. Checking system requirements
2. Installing dependencies
3. Setting up environment variables
4. Training the intent classification model
5. Providing next steps

Usage: python setup.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import time

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🚀 {text}")
    print("="*60)

def print_step(step, description):
    """Print a formatted step"""
    print(f"\n📋 Step {step}: {description}")
    print("-" * 40)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_command_exists(command):
    """Check if a command exists in the system"""
    return shutil.which(command) is not None

def install_dependencies():
    """Install project dependencies"""
    try:
        print("Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False

def setup_environment():
    """Set up environment variables"""
    env_file = Path(".env")
    template_file = Path("env.template")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if template_file.exists():
        try:
            shutil.copy(template_file, env_file)
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file and add your OpenAI API key")
            return True
        except Exception as e:
            print(f"❌ Error creating .env file: {e}")
            return False
    else:
        print("❌ env.template file not found")
        return False

def check_openai_key():
    """Check if OpenAI API key is configured"""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            if "your_openai_api_key_here" in content:
                print("⚠️  OpenAI API key not configured in .env file")
                return False
            elif "OPENAI_API_KEY=" in content:
                print("✅ OpenAI API key appears to be configured")
                return True
    print("❌ .env file not found")
    return False

def train_model():
    """Train the intent classification model"""
    training_script = Path("tensorflow_models/train_model.py")
    if not training_script.exists():
        print("❌ Training script not found")
        return False
    
    print("🧠 Training intent classification model...")
    print("This may take a few minutes...")
    
    try:
        # Change to the correct directory and run training
        os.chdir("tensorflow_models")
        result = subprocess.run([sys.executable, "train_model.py"], 
                              capture_output=True, text=True, timeout=300)
        os.chdir("..")
        
        if result.returncode == 0:
            print("✅ Model training completed successfully")
            return True
        else:
            print(f"❌ Model training failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  Model training timed out (>5 minutes)")
        os.chdir("..")
        return False
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        os.chdir("..")
        return False

def create_startup_scripts():
    """Create convenient startup scripts"""
    
    # Backend startup script
    backend_script = """#!/bin/bash
# Start SummarEaseAI Backend API
cd backend
echo "🚀 Starting SummarEaseAI Backend API..."
python api.py
"""
    
    # Frontend startup script
    frontend_script = """#!/bin/bash
# Start SummarEaseAI Frontend
echo "🎨 Starting SummarEaseAI Frontend..."
streamlit run app.py --server.port 8501
"""
    
    # Windows batch files
    backend_bat = """@echo off
cd backend
echo Starting SummarEaseAI Backend API...
python api.py
pause
"""
    
    frontend_bat = """@echo off
echo Starting SummarEaseAI Frontend...
streamlit run app.py --server.port 8501
pause
"""
    
    try:
        # Create shell scripts
        with open("start_backend.sh", "w") as f:
            f.write(backend_script)
        with open("start_frontend.sh", "w") as f:
            f.write(frontend_script)
        
        # Create batch files for Windows
        with open("start_backend.bat", "w") as f:
            f.write(backend_bat)
        with open("start_frontend.bat", "w") as f:
            f.write(frontend_bat)
        
        # Make shell scripts executable (Unix-like systems)
        if os.name != 'nt':
            os.chmod("start_backend.sh", 0o755)
            os.chmod("start_frontend.sh", 0o755)
        
        print("✅ Startup scripts created")
        return True
    except Exception as e:
        print(f"⚠️  Could not create startup scripts: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print_header("SETUP COMPLETE! 🎉")
    
    print("""
🎯 Next Steps:

1. 🔑 Configure your OpenAI API key:
   • Edit the .env file
   • Replace 'your_openai_api_key_here' with your actual API key
   • Get your key from: https://platform.openai.com/api-keys

2. 🚀 Start the Backend API:
   • Run: cd backend && python api.py
   • Or use: ./start_backend.sh (Unix) or start_backend.bat (Windows)

3. 🎨 Start the Frontend (in a new terminal):
   • Run: streamlit run app.py
   • Or use: ./start_frontend.sh (Unix) or start_frontend.bat (Windows)

4. 🌐 Access the Application:
   • Open your browser to: http://localhost:8501
   • Try some sample queries like "What happened on July 20, 1969?"

📚 Additional Resources:
   • README.md - Comprehensive documentation
   • tensorflow_models/train_model.py - Model training details
   • backend/api.py - API endpoints documentation

🆘 Need Help?
   • Check the logs in the terminal
   • Ensure your OpenAI API key is valid
   • Make sure all dependencies are installed

Happy summarizing! 🤖✨
""")

def main():
    """Main setup function"""
    print_header("SummarEaseAI Setup")
    print("This script will help you set up SummarEaseAI on your system.")
    
    # Step 1: Check Python version
    print_step(1, "Checking Python version")
    if not check_python_version():
        return False
    
    # Step 2: Install dependencies
    print_step(2, "Installing dependencies")
    if not install_dependencies():
        print("Please fix the dependency installation issues and try again.")
        return False
    
    # Step 3: Set up environment
    print_step(3, "Setting up environment")
    setup_environment()
    
    # Step 4: Check OpenAI configuration
    print_step(4, "Checking OpenAI configuration")
    openai_configured = check_openai_key()
    
    # Step 5: Train model (if OpenAI is configured)
    if openai_configured:
        print_step(5, "Training intent classification model")
        model_trained = train_model()
    else:
        print("⚠️  Skipping model training - configure OpenAI API key first")
        model_trained = False
    
    # Step 6: Create startup scripts
    print_step(6, "Creating startup scripts")
    create_startup_scripts()
    
    # Print next steps
    print_next_steps()
    
    # Final status
    if openai_configured and model_trained:
        print("🎉 Setup completed successfully! Your SummarEaseAI is ready to use.")
    else:
        print("⚠️  Setup partially completed. Please configure OpenAI API key and train the model.")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error during setup: {e}")
        sys.exit(1) 