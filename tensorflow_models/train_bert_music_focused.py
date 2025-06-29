#!/usr/bin/env python3
"""
Music-Focused BERT Training
Improved version with separate Music category and Beatles-specific training
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

class MusicFocusedTrainer:
    """Enhanced trainer with separate Music category and Beatles focus"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.data_dir = Path("training_data")
        self.data_dir.mkdir(exist_ok=True)
        self.wikipedia_fetcher = WikipediaPortalFetcher()
        
        # IMPROVED categories with separate Music
        self.enhanced_portals = {
            'History': {
                'search_terms': [
                    'World War II', 'Ancient Rome', 'Medieval Europe', 'American Civil War',
                    'French Revolution', 'Renaissance', 'Cold War', 'Industrial Revolution'
                ]
            },
            'Science': {
                'search_terms': [
                    'quantum physics', 'molecular biology', 'organic chemistry', 'astrophysics',
                    'genetics', 'thermodynamics', 'evolution', 'particle physics'
                ]
            },
            'Biography': {
                'search_terms': [
                    'Albert Einstein', 'Marie Curie', 'Leonardo da Vinci', 'Winston Churchill',
                    'Martin Luther King', 'Gandhi', 'Napoleon', 'Shakespeare'
                ]
            },
            'Technology': {
                'search_terms': [
                    'artificial intelligence', 'machine learning', 'computer science',
                    'programming', 'internet', 'software engineering', 'robotics'
                ]
            },
            'Music': {  # ✨ NEW: Separate Music category
                'search_terms': [
                    'The Beatles', 'Rolling Stones', 'Led Zeppelin', 'Pink Floyd',  # Rock bands
                    'Mozart', 'Beethoven', 'Bach', 'Chopin',  # Classical
                    'jazz music', 'blues music', 'rock music', 'pop music',  # Genres
                    'guitar', 'piano', 'drums', 'violin',  # Instruments
                    'music theory', 'musical composition', 'sound recording'  # Music concepts
                ]
            },
            'Arts': {  # 🎨 REFINED: Visual arts only
                'search_terms': [
                    'painting', 'sculpture', 'drawing', 'photography',
                    'architecture', 'design', 'visual arts', 'fine arts'
                ]
            },
            'Sports': {
                'search_terms': [
                    'Olympic Games', 'football', 'basketball', 'tennis', 'soccer',
                    'baseball', 'swimming', 'athletics', 'gymnastics'
                ]
            },
            'Politics': {
                'search_terms': [
                    'democracy', 'government', 'elections', 'political parties',
                    'constitution', 'international relations', 'diplomacy'
                ]
            },
            'Geography': {
                'search_terms': [
                    'continents', 'countries', 'mountains', 'rivers', 'climate',
                    'ecosystems', 'capitals', 'geography'
                ]
            }
        }
    
    def collect_music_focused_data(self, use_cache: bool = True) -> pd.DataFrame:
        """Collect Wikipedia data with strong focus on music"""
        cache_file = self.data_dir / "music_focused_wikipedia_data.json"
        
        if use_cache and cache_file.exists():
            logger.info(f"📂 Loading cached music-focused data from {cache_file}")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                df = pd.DataFrame(cached_data)
                logger.info(f"✅ Loaded {len(df)} cached samples")
                return df
            except Exception as e:
                logger.warning(f"⚠️  Cache loading failed: {e}, collecting fresh data")
        
        logger.info("🎵 Collecting music-focused Wikipedia data...")
        all_training_data = []
        
        for intent_category, config in self.enhanced_portals.items():
            logger.info(f"\n📚 Collecting data for {intent_category}...")
            category_data = []
            
            for search_term in config['search_terms']:
                try:
                    logger.info(f"   🔍 Searching: {search_term}")
                    import wikipedia
                    
                    # For music category, get more results
                    results_count = 15 if intent_category == 'Music' else 10
                    search_results = wikipedia.search(search_term, results=results_count)
                    
                    # For music, take more pages per search
                    pages_per_search = 12 if intent_category == 'Music' else 8
                    
                    for page_title in search_results[:pages_per_search]:
                        data = self.wikipedia_fetcher.fetch_page_training_data(
                            page_title, intent_category
                        )
                        if data and len(data['text']) > 100:
                            category_data.append(data)
                    
                    time.sleep(0.5)  # Be respectful to Wikipedia
                    
                except Exception as e:
                    logger.warning(f"Error with search term {search_term}: {e}")
                    continue
            
            # Remove duplicates
            seen_titles = set()
            unique_data = []
            for item in category_data:
                if item['title'] not in seen_titles:
                    seen_titles.add(item['title'])
                    unique_data.append(item)
            
            logger.info(f"✅ Collected {len(unique_data)} unique {intent_category} samples")
            all_training_data.extend(unique_data)
        
        # Add Beatles-specific training samples
        beatles_samples = self._create_beatles_specific_samples()
        all_training_data.extend(beatles_samples)
        logger.info(f"🎸 Added {len(beatles_samples)} Beatles-specific samples")
        
        df = pd.DataFrame(all_training_data)
        
        # Save cache
        logger.info(f"💾 Caching data to {cache_file}")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(all_training_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Total collected: {len(df)} samples across {len(self.enhanced_portals)} categories")
        return df
    
    def _create_beatles_specific_samples(self) -> List[Dict]:
        """Create Beatles and music-specific training samples"""
        beatles_samples = [
            {'title': 'Beatles Music Query 1', 'text': 'Tell me about the Beatles and their music', 'intent': 'Music'},
            {'title': 'Beatles Music Query 2', 'text': 'Who were the Beatles members', 'intent': 'Music'},
            {'title': 'Beatles Music Query 3', 'text': 'Beatles songs and albums', 'intent': 'Music'},
            {'title': 'Beatles Music Query 4', 'text': 'What genre of music did the Beatles play', 'intent': 'Music'},
            {'title': 'Rock Music Query 1', 'text': 'Tell me about rock music bands', 'intent': 'Music'},
            {'title': 'Classical Music Query 1', 'text': 'Who was Mozart and what music did he compose', 'intent': 'Music'},
            {'title': 'Music Instruments Query 1', 'text': 'How does a guitar work and what music can you play', 'intent': 'Music'},
            {'title': 'Music Theory Query 1', 'text': 'Explain music theory and composition', 'intent': 'Music'}
        ]
        return beatles_samples
    
    def create_balanced_training_set(self, wikipedia_df: pd.DataFrame, samples_per_category: int = 60) -> Tuple[List[str], List[str]]:
        """Create balanced training set with emphasis on Music"""
        logger.info(f"⚖️  Creating balanced training set ({samples_per_category} per category)")
        
        balanced_texts = []
        balanced_labels = []
        
        for category in self.enhanced_portals.keys():
            category_data = wikipedia_df[wikipedia_df['intent'] == category]
            
            if len(category_data) == 0:
                logger.warning(f"⚠️  No data found for category: {category}")
                continue
            
            # For Music category, ensure we have enough samples
            target_samples = samples_per_category
            if category == 'Music':
                target_samples = min(80, len(category_data))  # More music samples
            
            if len(category_data) < target_samples:
                logger.warning(f"⚠️  Only {len(category_data)} samples for {category}, using all")
                sampled_data = []
                for _, row in category_data.iterrows():
                    sampled_data.append(row.to_dict())
            else:
                sampled_df = category_data.sample(target_samples)
                sampled_data = []
                for _, row in sampled_df.iterrows():
                    sampled_data.append(row.to_dict())
            
            for item in sampled_data:
                balanced_texts.append(item['text'])
                balanced_labels.append(item['intent'])
            
            logger.info(f"✅ {category}: {len(sampled_data)} samples")
        
        # Add General samples
        general_samples = [
            {'text': 'What is the weather like today', 'intent': 'General'},
            {'text': 'How are you doing', 'intent': 'General'},
            {'text': 'Tell me something interesting', 'intent': 'General'},
            {'text': 'What can you help me with', 'intent': 'General'},
            {'text': 'I need some information', 'intent': 'General'}
        ] * 6  # 30 samples
        
        for sample in general_samples:
            balanced_texts.append(sample['text'])
            balanced_labels.append('General')
        
        logger.info(f"🎯 Final balanced set: {len(balanced_texts)} samples")
        return balanced_texts, balanced_labels
    
    def train_music_focused_model(self, samples_per_category: int = 60, use_cache: bool = True):
        """Train the music-focused BERT model"""
        logger.info("🎵 Starting Music-Focused BERT Training")
        logger.info("=" * 70)
        
        # Collect data
        wikipedia_df = self.collect_music_focused_data(use_cache=use_cache)
        
        # Create balanced training set
        texts, labels = self.create_balanced_training_set(wikipedia_df, samples_per_category)
        
        # Initialize BERT trainer
        bert_classifier = GPUBERTIntentClassifier(model_name=self.model_name)
        bert_classifier.batch_size = 4  # Conservative for stability
        bert_classifier.epochs = 3  # More epochs for better learning
        bert_classifier.max_length = 64
        
        # Train model
        logger.info("🚀 Starting GPU-accelerated training...")
        training_results = bert_classifier.train(texts=texts, labels=labels)
        
        # Save model
        save_dir = "tensorflow_models/bert_music_models"
        os.makedirs(save_dir, exist_ok=True)
        bert_classifier.save_model(save_dir)
        success = training_results is not None
        
        if success:
            logger.info("✅ Music-focused BERT model training completed successfully!")
            logger.info("📁 Model saved to: tensorflow_models/bert_music_models")
            
            # Test with Beatles query
            logger.info("\n🧪 Testing Beatles classification...")
            result = bert_classifier.predict("Tell me about the Beatles")
            logger.info(f"Beatles query result: {result}")
            
        else:
            logger.error("❌ Training failed!")

def main():
    """Main training function"""
    trainer = MusicFocusedTrainer()
    trainer.train_music_focused_model(samples_per_category=60, use_cache=False)

if __name__ == "__main__":
    main() 