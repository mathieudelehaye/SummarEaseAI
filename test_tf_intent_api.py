#!/usr/bin/env python3
"""
Test script for TensorFlow LSTM Intent Classification REST API
"""

import requests
import json

# API endpoint
BASE_URL = "http://localhost:5000"
INTENT_ENDPOINT = f"{BASE_URL}/predict_intent"

def test_tf_intent_api():
    """Test the TensorFlow LSTM intent classification API"""
    print("🧪 Testing TensorFlow LSTM Intent Classification API")
    print(f"📡 Endpoint: {INTENT_ENDPOINT}")
    print("-" * 60)
    
    # Test cases
    test_cases = [
        "Tell me about World War II battles",
        "How does machine learning work?", 
        "Who was Albert Einstein?",
        "What are the latest AI technologies?",
        "Explain Renaissance art",
        "Olympic Games history",
        "Democracy and voting systems",
        "Mountain formation processes",
        "General knowledge questions"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: '{text}'")
        
        try:
            # Make API request
            payload = {"text": text}
            response = requests.post(INTENT_ENDPOINT, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Intent: {result['predicted_intent']}")
                print(f"📊 Confidence: {result['confidence']:.3f}")
                print(f"🤖 Model: {result['model_type']}")
                print(f"🔧 Model Loaded: {result['model_loaded']}")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    print("\n" + "="*60)
    print("🏁 Test completed!")

def test_api_status():
    """Check API status"""
    print("🔍 Checking API status...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
            return True
        else:
            print(f"❌ API returned status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API not accessible: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TensorFlow LSTM Intent Classification API Test")
    print("="*60)
    
    # Check if API is running
    if test_api_status():
        print()
        test_tf_intent_api()
    else:
        print("\n💡 Make sure to start the backend API first:")
        print("   python backend/api.py")
        print("   or")
        print("   python -m flask --app backend.api run --port 5000") 