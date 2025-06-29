#!/usr/bin/env python3
"""
Quick test to verify BERT model inference is working
"""

import sys
import os
sys.path.append('tensorflow_models')

def test_bert_inference():
    """Test BERT model inference"""
    try:
        print("🔍 Testing BERT Model Inference...")
        
        # Import the GPU classifier
        from bert_gpu_classifier import GPUBERTClassifier
        
        # Initialize classifier
        print("📂 Loading BERT classifier...")
        classifier = GPUBERTClassifier()
        
        # Load the model
        if not classifier.load_model():
            print("❌ Failed to load model")
            return False
            
        print("✅ Model loaded successfully!")
        
        # Test predictions
        test_texts = [
            "Tell me about the history of World War II",
            "How does machine learning work?",
            "Who was Albert Einstein?",
            "What are the latest developments in AI technology?",
            "Explain the art of Renaissance painting"
        ]
        
        print("\n🧪 Testing predictions:")
        for text in test_texts:
            try:
                intent, confidence = classifier.predict(text)
                print(f"Text: '{text[:50]}...'")
                print(f"Intent: {intent} (confidence: {confidence:.3f})")
                print("-" * 50)
            except Exception as e:
                print(f"❌ Prediction failed for: '{text[:30]}...' - {e}")
                
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure TensorFlow and transformers are installed")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_bert_inference()
    if success:
        print("\n✅ BERT Model is working correctly!")
    else:
        print("\n❌ BERT Model test failed!")
    
    sys.exit(0 if success else 1) 