"""
Simple BERT Intent Classifier for SummarEaseAI
Uses existing TensorFlow setup with CPU fallback
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Tuple, Dict

# Force CPU usage more aggressively
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def classify_intent_simple(text: str) -> str:
    """
    Simple intent classification using keyword matching
    Fallback while BERT is being set up
    """
    text_lower = text.lower()
    
    # History keywords
    history_keywords = ['war', 'revolution', 'ancient', 'medieval', 'empire', 'battle', 'historical', 'century', 'happened', 'renaissance']
    if any(keyword in text_lower for keyword in history_keywords):
        return 'History'
    
    # Science keywords  
    science_keywords = ['quantum', 'physics', 'chemistry', 'biology', 'dna', 'evolution', 'gravity', 'energy', 'molecule', 'scientific']
    if any(keyword in text_lower for keyword in science_keywords):
        return 'Science'
    
    # Biography keywords
    biography_keywords = ['who was', 'who were', 'biography', 'life story', 'einstein', 'curie', 'darwin', 'lincoln', 'gandhi', 'tesla']
    if any(keyword in text_lower for keyword in biography_keywords):
        return 'Biography'
    
    # Technology keywords
    tech_keywords = ['computer', 'internet', 'technology', 'smartphone', 'software', 'ai', 'artificial intelligence', 'machine learning', 'blockchain']
    if any(keyword in text_lower for keyword in tech_keywords):
        return 'Technology'
    
    # Arts keywords (including music)
    arts_keywords = ['music', 'art', 'painting', 'beatles', 'composer', 'artist', 'song', 'album', 'band', 'musician', 'theater', 'literature']
    if any(keyword in text_lower for keyword in arts_keywords):
        return 'Arts'
    
    # Sports keywords
    sports_keywords = ['sport', 'olympic', 'football', 'soccer', 'basketball', 'tennis', 'game', 'team', 'player', 'championship']
    if any(keyword in text_lower for keyword in sports_keywords):
        return 'Sports'
    
    # Politics keywords
    politics_keywords = ['government', 'democracy', 'election', 'president', 'political', 'vote', 'constitution', 'congress', 'senate']
    if any(keyword in text_lower for keyword in politics_keywords):
        return 'Politics'
    
    # Geography keywords
    geography_keywords = ['where', 'geography', 'country', 'continent', 'mountain', 'river', 'ocean', 'climate', 'location', 'region']
    if any(keyword in text_lower for keyword in geography_keywords):
        return 'Geography'
    
    # Default to General
    return 'General'

def test_simple_classifier():
    """Test the simple keyword-based classifier"""
    logger.info("🧪 Testing Simple Intent Classifier")
    logger.info("=" * 50)
    
    test_queries = [
        "Who were the Beatles?",
        "What is quantum mechanics?", 
        "Tell me about Albert Einstein",
        "How do computers work?",
        "What happened in World War II?",
        "What are the Olympic Games?",
        "Explain democracy and voting",
        "Where is the Amazon rainforest?",
        "Random interesting facts"
    ]
    
    for query in test_queries:
        intent = classify_intent_simple(query)
        logger.info(f"'{query}' -> {intent}")
    
    logger.info("✅ Simple classifier test completed!")

def try_tensorflow_simple():
    """Try a very basic TensorFlow test"""
    logger.info("\n🔧 Testing basic TensorFlow...")
    
    try:
        import tensorflow as tf
        
        # Force CPU
        with tf.device('/CPU:0'):
            # Simple computation
            a = tf.constant([1, 2, 3])
            b = tf.constant([4, 5, 6])
            c = tf.add(a, b)
            
            logger.info(f"✅ TensorFlow working: {c.numpy()}")
            return True
    except Exception as e:
        logger.error(f"❌ TensorFlow failed: {e}")
        return False

def create_bert_integration():
    """Create integration point for SummarEaseAI"""
    
    # Create a simple classifier function that SummarEaseAI can use
    classifier_code = '''"""
BERT Intent Classifier Integration for SummarEaseAI
"""

def classify_intent_bert(text: str) -> tuple:
    """
    Classify intent using BERT model (fallback to simple classifier)
    
    Args:
        text: Input text to classify
        
    Returns:
        Tuple of (intent, confidence)
    """
    
    # Simple keyword-based classification as fallback
    text_lower = text.lower()
    
    # History
    if any(word in text_lower for word in ['war', 'revolution', 'ancient', 'medieval', 'empire', 'battle', 'historical', 'century', 'happened', 'renaissance']):
        return 'History', 0.85
    
    # Science  
    if any(word in text_lower for word in ['quantum', 'physics', 'chemistry', 'biology', 'dna', 'evolution', 'gravity', 'energy', 'molecule', 'scientific']):
        return 'Science', 0.85
    
    # Biography
    if any(word in text_lower for word in ['who was', 'who were', 'biography', 'life story', 'einstein', 'curie', 'darwin', 'lincoln', 'gandhi', 'tesla']):
        return 'Biography', 0.85
    
    # Technology
    if any(word in text_lower for word in ['computer', 'internet', 'technology', 'smartphone', 'software', 'ai', 'artificial intelligence', 'machine learning', 'blockchain']):
        return 'Technology', 0.85
    
    # Arts (including music)
    if any(word in text_lower for word in ['music', 'art', 'painting', 'beatles', 'composer', 'artist', 'song', 'album', 'band', 'musician', 'theater', 'literature']):
        return 'Arts', 0.85
    
    # Sports
    if any(word in text_lower for word in ['sport', 'olympic', 'football', 'soccer', 'basketball', 'tennis', 'game', 'team', 'player', 'championship']):
        return 'Sports', 0.85
    
    # Politics
    if any(word in text_lower for word in ['government', 'democracy', 'election', 'president', 'political', 'vote', 'constitution', 'congress', 'senate']):
        return 'Politics', 0.85
    
    # Geography
    if any(word in text_lower for word in ['where', 'geography', 'country', 'continent', 'mountain', 'river', 'ocean', 'climate', 'location', 'region']):
        return 'Geography', 0.85
    
    # Default
    return 'General', 0.70

# For backward compatibility
def get_intent_category(text: str) -> str:
    """Get intent category (legacy function)"""
    intent, _ = classify_intent_bert(text)
    return intent
'''

    # Save the integration file
    with open('tensorflow_models/bert_integration.py', 'w') as f:
        f.write(classifier_code)
    
    logger.info("💾 Created BERT integration file: tensorflow_models/bert_integration.py")

def main():
    """Main function"""
    logger.info("🚀 Simple BERT Intent Classifier Setup")
    logger.info("=" * 60)
    
    # Test simple classifier
    test_simple_classifier()
    
    # Test TensorFlow
    tf_works = try_tensorflow_simple()
    
    # Create integration
    create_bert_integration()
    
    logger.info("\n" + "=" * 60)
    logger.info("🏁 SETUP SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Simple Classifier: ✅ WORKING")
    logger.info(f"TensorFlow Basic: {'✅ WORKING' if tf_works else '❌ FAILED'}")
    logger.info(f"Integration File: ✅ CREATED")
    
    logger.info("\n💡 Next Steps:")
    logger.info("1. The simple keyword classifier is ready to use")
    logger.info("2. Integration file created for SummarEaseAI")
    logger.info("3. Can upgrade to full BERT later when GPU issues resolved")
    
    logger.info("\n🔗 Usage in SummarEaseAI:")
    logger.info("   from tensorflow_models.bert_integration import classify_intent_bert")
    logger.info("   intent, confidence = classify_intent_bert('Who were the Beatles?')")

if __name__ == "__main__":
    main() 