#!/usr/bin/env python3
"""
SummarEaseAI Backend Starter (CPU-Only Mode)

This script starts the Flask API backend without GPU/DirectML to avoid conflicts.
Perfect for running the backend while keeping GPU available for training.
"""

import os
import sys

# Disable GPU/DirectML for TensorFlow in this process
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable CUDA
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'  # Disable GPU memory growth

print("🚀 Starting SummarEaseAI Backend (CPU-Only Mode)")
print("=" * 50)
print("✅ GPU/DirectML disabled for backend compatibility")
print("✅ Your RTX 4070 remains available for training scripts")
print("✅ Backend will use CPU for model inference (fast enough)")
print()

# Now import and run the backend
if __name__ == "__main__":
    # Add backend directory to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
    
    # Import and run the API
    from backend.api import app, logger
    
    print("🌐 Starting Flask API server...")
    print("📡 Backend will be available at: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the backend")
    print()
    
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False  # Disable reloader to avoid import issues
        )
    except KeyboardInterrupt:
        print("\n🛑 Backend stopped by user")
    except Exception as e:
        print(f"\n❌ Backend error: {e}")
        print("💡 Try: pip install -r requirements.txt") 