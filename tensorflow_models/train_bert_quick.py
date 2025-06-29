#!/usr/bin/env python3
"""
Quick GPU BERT Training Script
Train the GPU BERT model for intent classification
"""

import sys
import os
sys.path.append('.')

import logging
from tensorflow_models.train_bert_gpu import GPUBERTIntentClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Quick training of GPU BERT model"""
    logger.info("🚀 Quick GPU BERT Training")
    
    # Initialize classifier
    classifier = GPUBERTIntentClassifier(model_name="distilbert-base-uncased")
    
    # Get training data using the correct method
    texts, labels = classifier.generate_enhanced_training_data()
    logger.info(f"📚 Training with {len(texts)} samples")
    
    # Train the model
    results = classifier.train(texts, labels)
    
    if results and 'final_accuracy' in results:
        logger.info("🎉 GPU BERT model trained successfully!")
        logger.info(f"🎯 Final accuracy: {results['final_accuracy']:.4f}")
        logger.info("✅ Model saved and ready for use")
        
        # Test a few predictions
        test_queries = [
            "Who were the Beatles?",
            "Explain quantum physics",
            "Tell me about World War II",
            "How does artificial intelligence work?"
        ]
        
        logger.info("🧪 Testing trained model:")
        for query in test_queries:
            try:
                intent, confidence = classifier.predict(query)
                logger.info(f"   '{query}' → {intent} ({confidence:.3f})")
            except Exception as e:
                logger.error(f"   Error testing '{query}': {e}")
    else:
        logger.error("❌ GPU BERT training failed!")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ GPU BERT model is ready!")
        print("You can now use the SummarEaseAI backend with GPU BERT acceleration.")
    else:
        print("\n❌ Training failed. Please check the logs.")
        sys.exit(1) 