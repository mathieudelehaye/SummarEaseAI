#!/usr/bin/env python3
"""
Music-Only BERT Training
Simple binary classifier using Wikipedia music portal data
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.wikipedia_training_fetcher import WikipediaPortalFetcher
from tensorflow_models.train_bert_gpu import GPUBERTIntentClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MusicOnlyTrainer:
    """Simple trainer focused only on Music classification using Wikipedia"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.data_dir = Path("training_data")
        self.data_dir.mkdir(exist_ok=True)
        self.wikipedia_fetcher = WikipediaPortalFetcher()
    
    def collect_wikipedia_music_data(self) -> List[Dict]:
        """Collect music data from Wikipedia"""
        logger.info("🎵 Collecting Wikipedia music data...")
        
        music_search_terms = [
            # Beatles and rock bands
            'The Beatles', 'Rolling Stones', 'Led Zeppelin', 'Pink Floyd', 'Queen',
            # Classical composers  
            'Mozart', 'Beethoven', 'Bach', 'Chopin', 'Vivaldi',
            # Music genres
            'rock music', 'jazz music', 'classical music', 'blues music', 'pop music',
            # Music concepts
            'music theory', 'musical composition', 'sound recording'
        ]
        
        music_data = []
        
        for search_term in music_search_terms:
            try:
                logger.info(f"   🔍 Searching: {search_term}")
                import wikipedia
                
                search_results = wikipedia.search(search_term, results=8)
                
                for page_title in search_results[:5]:  # Take top 5 results
                    data = self.wikipedia_fetcher.fetch_page_training_data(
                        page_title, 'Music'
                    )
                    if data and len(data['text']) > 100:
                        music_data.append(data)
                
                time.sleep(0.3)  # Be respectful to Wikipedia
                
            except Exception as e:
                logger.warning(f"Error with search term {search_term}: {e}")
                continue
        
        # Remove duplicates
        seen_titles = set()
        unique_data = []
        for item in music_data:
            if item['title'] not in seen_titles:
                seen_titles.add(item['title'])
                unique_data.append(item)
        
        logger.info(f"✅ Collected {len(unique_data)} unique music samples from Wikipedia")
        return unique_data
    
    def create_music_training_data(self) -> Tuple[List[str], List[str]]:
        """Create music vs non-music training data"""
        logger.info("🎵 Creating Music vs Non-Music training data...")
        
        # Get Wikipedia music data
        wikipedia_music_data = self.collect_wikipedia_music_data()
        
        # Add Beatles-specific samples
        beatles_samples = [
            {'text': 'Tell me about the Beatles and their music', 'intent': 'Music'},
            {'text': 'Who were the Beatles members', 'intent': 'Music'},
            {'text': 'Beatles songs and albums', 'intent': 'Music'},
            {'text': 'What genre of music did the Beatles play', 'intent': 'Music'},
            {'text': 'Beatles discography and history', 'intent': 'Music'},
            {'text': 'John Lennon and Paul McCartney songwriting', 'intent': 'Music'},
            {'text': 'Beatles rock band influence on music', 'intent': 'Music'},
            {'text': 'Abbey Road and Beatles albums', 'intent': 'Music'}
        ]
        
        # Non-music samples (minimal but diverse)
        non_music_samples = [
            {'text': 'Tell me about quantum physics', 'intent': 'Non-Music'},
            {'text': 'What is machine learning', 'intent': 'Non-Music'},
            {'text': 'History of World War II', 'intent': 'Non-Music'},
            {'text': 'How do computers work', 'intent': 'Non-Music'},
            {'text': 'What is democracy', 'intent': 'Non-Music'},
            {'text': 'Explain climate change', 'intent': 'Non-Music'},
            {'text': 'Tell me about space exploration', 'intent': 'Non-Music'},
            {'text': 'What is artificial intelligence', 'intent': 'Non-Music'},
            {'text': 'Biology and genetics', 'intent': 'Non-Music'},
            {'text': 'Mathematics and calculus', 'intent': 'Non-Music'},
            {'text': 'Geography and continents', 'intent': 'Non-Music'},
            {'text': 'Sports and athletics', 'intent': 'Non-Music'},
            {'text': 'Cooking and recipes', 'intent': 'Non-Music'},
            {'text': 'Travel and tourism', 'intent': 'Non-Music'},
            {'text': 'Business and economics', 'intent': 'Non-Music'},
            {'text': 'Health and medicine', 'intent': 'Non-Music'}
        ]
        
        # Combine all data
        texts = []
        labels = []
        
        # Add Wikipedia music data
        for item in wikipedia_music_data:
            texts.append(item['text'])
            labels.append('Music')
        
        # Add Beatles samples
        for item in beatles_samples:
            texts.append(item['text'])
            labels.append('Music')
        
        # Add non-music samples
        for item in non_music_samples:
            texts.append(item['text'])
            labels.append('Non-Music')
        
        music_count = len(wikipedia_music_data) + len(beatles_samples)
        non_music_count = len(non_music_samples)
        
        logger.info(f"✅ Created {music_count} Music samples (Wikipedia + Beatles)")
        logger.info(f"✅ Created {non_music_count} Non-Music samples")
        logger.info(f"🎯 Total: {len(texts)} samples")
        
        return texts, labels
    
    def train_music_classifier(self):
        """Train the music-only BERT classifier"""
        logger.info("🎵 Starting Music-Only BERT Training")
        logger.info("=" * 50)
        
        # Create training data
        texts, labels = self.create_music_training_data()
        
        # Initialize BERT trainer
        bert_classifier = GPUBERTIntentClassifier(model_name=self.model_name)
        bert_classifier.batch_size = 8  # Small dataset, can use larger batch
        bert_classifier.epochs = 5  # More epochs for better learning
        bert_classifier.max_length = 64
        
        # Train model
        logger.info("🚀 Starting GPU-accelerated training...")
        training_results = bert_classifier.train(texts=texts, labels=labels)
        
        # Save model (fix: don't pass save_dir argument)
        logger.info("💾 Saving model...")
        bert_classifier.save_model()  # Uses default save directory
        success = training_results is not None
        
        if success:
            logger.info("✅ Music-only BERT model training completed!")
            logger.info("📁 Model saved to: tensorflow_models/bert_gpu_models")
            
            # Test with Beatles queries
            logger.info("\n🧪 Testing Beatles classification...")
            test_queries = [
                "Tell me about the Beatles",
                "Who were the Beatles members", 
                "What is quantum physics",
                "History of World War II"
            ]
            
            for query in test_queries:
                result = bert_classifier.predict(query)
                logger.info(f"'{query}' → {result}")
                
        else:
            logger.error("❌ Training failed!")

def main():
    """Main training function"""
    trainer = MusicOnlyTrainer()
    trainer.train_music_classifier()

if __name__ == "__main__":
    main() 