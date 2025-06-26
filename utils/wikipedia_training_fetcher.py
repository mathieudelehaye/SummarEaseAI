#!/usr/bin/env python3
"""
Wikipedia Portal Training Data Fetcher
Fetch pages from Wikipedia portals to build better training datasets
Completely FREE - no API costs!
"""

import wikipedia
import wikipediaapi
import requests
import re
import json
import time
import random
from typing import List, Dict, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikipediaPortalFetcher:
    """Fetch training data from Wikipedia portals"""
    
    def __init__(self):
        # Set user agent for Wikipedia API compliance
        wikipedia.set_user_agent("SummarEaseAI-Training/1.0 (https://github.com/your-repo)")
        self.wiki_wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='SummarEaseAI-Training/1.0 (https://github.com/your-repo) Python/WikipediaAPI'
        )
        
        # Portal mappings to intent categories
        self.portals = {
            'Music': {
                'portal_url': 'https://en.wikipedia.org/wiki/Portal:Music',
                'intent_label': 'Music',
                'subcategories': ['Popular music', 'Classical music', 'Jazz', 'Rock music', 'Hip hop']
            },
            'Science': {
                'portal_url': 'https://en.wikipedia.org/wiki/Portal:Science',
                'intent_label': 'Science',
                'subcategories': ['Physics', 'Chemistry', 'Biology', 'Mathematics', 'Computer science']
            },
            'History': {
                'portal_url': 'https://en.wikipedia.org/wiki/Portal:History',
                'intent_label': 'History',
                'subcategories': ['World War II', 'Ancient history', 'American history', 'Medieval history']
            },
            'Technology': {
                'portal_url': 'https://en.wikipedia.org/wiki/Portal:Technology',
                'intent_label': 'Technology',
                'subcategories': ['Computing', 'Electronics', 'Engineering', 'Internet', 'Software']
            },
            'Sports': {
                'portal_url': 'https://en.wikipedia.org/wiki/Portal:Sports',
                'intent_label': 'Sports',
                'subcategories': ['Football', 'Basketball', 'Tennis', 'Olympics', 'Baseball']
            },
            'Biography': {
                'portal_url': 'https://en.wikipedia.org/wiki/Portal:Biography',
                'intent_label': 'Biography',
                'subcategories': ['Politicians', 'Scientists', 'Artists', 'Musicians', 'Writers']
            }
        }
    
    def search_portal_pages(self, portal_name: str, max_pages: int = 50) -> List[str]:
        """Search for pages related to a portal"""
        if portal_name not in self.portals:
            logger.error(f"Unknown portal: {portal_name}")
            return []
        
        portal_info = self.portals[portal_name]
        all_pages = []
        
        logger.info(f"🔍 Searching for {portal_name} pages...")
        
        # Search using portal subcategories
        for subcategory in portal_info['subcategories']:
            try:
                logger.info(f"   📝 Searching subcategory: {subcategory}")
                search_results = wikipedia.search(subcategory, results=10)
                
                for result in search_results:
                    if result not in all_pages:
                        all_pages.append(result)
                        if len(all_pages) >= max_pages:
                            break
                
                # Add some delay to be respectful to Wikipedia
                time.sleep(0.5)
                
                if len(all_pages) >= max_pages:
                    break
                    
            except Exception as e:
                logger.warning(f"Error searching {subcategory}: {e}")
                continue
        
        # Also search directly for the portal name
        try:
            direct_search = wikipedia.search(f"{portal_name} wikipedia", results=20)
            for result in direct_search:
                if result not in all_pages and len(all_pages) < max_pages:
                    all_pages.append(result)
        except Exception as e:
            logger.warning(f"Error in direct search for {portal_name}: {e}")
        
        logger.info(f"✅ Found {len(all_pages)} potential {portal_name} pages")
        return all_pages[:max_pages]
    
    def fetch_page_training_data(self, page_title: str, intent_label: str) -> Optional[Dict]:
        """Fetch a single page and extract training data"""
        try:
            # Use auto_suggest=False to prevent Apollo 11 → Apollo 1 issues
            page = wikipedia.page(page_title, auto_suggest=False)
            
            # Extract intro paragraph (first paragraph)
            intro = page.summary.split('\n')[0] if page.summary else ""
            
            # Clean the intro text
            intro = re.sub(r'\([^)]*\)', '', intro)  # Remove parentheses
            intro = re.sub(r'\s+', ' ', intro).strip()  # Clean whitespace
            
            # Skip very short articles
            if len(intro) < 50:
                logger.debug(f"Skipping short article: {page_title}")
                return None
            
            # Create training sample
            training_data = {
                'title': page.title,
                'text': intro,
                'intent': intent_label,
                'url': page.url,
                'length': len(intro),
                'word_count': len(intro.split())
            }
            
            logger.debug(f"✅ Extracted: {page.title} → {intent_label}")
            return training_data
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Try first disambiguation option
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                intro = page.summary.split('\n')[0] if page.summary else ""
                intro = re.sub(r'\([^)]*\)', '', intro)
                intro = re.sub(r'\s+', ' ', intro).strip()
                
                if len(intro) >= 50:
                    return {
                        'title': page.title,
                        'text': intro,
                        'intent': intent_label,
                        'url': page.url,
                        'length': len(intro),
                        'word_count': len(intro.split()),
                        'disambiguation_resolved': True
                    }
            except Exception:
                pass
            
            logger.debug(f"Disambiguation issue: {page_title}")
            return None
            
        except wikipedia.exceptions.PageError:
            logger.debug(f"Page not found: {page_title}")
            return None
            
        except Exception as e:
            logger.debug(f"Error fetching {page_title}: {e}")
            return None
    
    def collect_portal_training_data(self, portal_name: str, max_pages: int = 50) -> List[Dict]:
        """Collect training data from a specific portal"""
        logger.info(f"🚀 Collecting training data for portal: {portal_name}")
        
        # Search for pages
        page_titles = self.search_portal_pages(portal_name, max_pages)
        
        if not page_titles:
            logger.warning(f"No pages found for portal: {portal_name}")
            return []
        
        intent_label = self.portals[portal_name]['intent_label']
        training_data = []
        
        logger.info(f"📄 Fetching {len(page_titles)} pages...")
        
        for i, page_title in enumerate(page_titles, 1):
            logger.info(f"   {i}/{len(page_titles)}: {page_title}")
            
            data = self.fetch_page_training_data(page_title, intent_label)
            if data:
                training_data.append(data)
            
            # Be respectful to Wikipedia - add delays
            if i % 10 == 0:
                time.sleep(2)
            else:
                time.sleep(0.3)
        
        logger.info(f"✅ Successfully collected {len(training_data)} training samples for {portal_name}")
        return training_data
    
    def collect_all_portals_data(self, pages_per_portal: int = 30) -> Dict[str, List[Dict]]:
        """Collect training data from all portals"""
        logger.info(f"🌍 Collecting training data from ALL portals ({pages_per_portal} pages each)")
        
        all_data = {}
        total_samples = 0
        
        for portal_name in self.portals.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"Portal: {portal_name}")
            logger.info(f"{'='*60}")
            
            portal_data = self.collect_portal_training_data(portal_name, pages_per_portal)
            all_data[portal_name] = portal_data
            total_samples += len(portal_data)
            
            logger.info(f"Portal {portal_name}: {len(portal_data)} samples")
        
        logger.info(f"\n🎉 COLLECTION COMPLETE!")
        logger.info(f"Total samples collected: {total_samples}")
        
        # Print summary
        for portal, data in all_data.items():
            logger.info(f"   {portal}: {len(data)} samples")
        
        return all_data
    
    def save_training_data(self, training_data: Dict[str, List[Dict]], filename: str = "wikipedia_training_data.json"):
        """Save training data to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Training data saved to: {filename}")
            
            # Also create a flat format for easier TensorFlow integration
            flat_filename = filename.replace('.json', '_flat.json')
            flat_data = []
            
            for portal, samples in training_data.items():
                for sample in samples:
                    flat_data.append({
                        'text': sample['text'],
                        'intent': sample['intent'],
                        'title': sample['title']
                    })
            
            with open(flat_filename, 'w', encoding='utf-8') as f:
                json.dump(flat_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Flat training data saved to: {flat_filename}")
            logger.info(f"📊 Total flat samples: {len(flat_data)}")
            
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
    
    def create_beatles_specific_samples(self) -> List[Dict]:
        """Create specific samples to fix Beatles → Science misclassification"""
        beatles_queries = [
            "Who were the Beatles?",
            "Tell me about The Beatles",
            "Beatles band members",
            "Beatles music",
            "John Lennon Paul McCartney",
            "Beatles songs",
            "British rock band Beatles",
            "Beatles albums",
            "Fab Four",
            "Beatles Liverpool"
        ]
        
        samples = []
        for query in beatles_queries:
            samples.append({
                'text': query,
                'intent': 'Music',
                'title': 'Beatles Query Training Sample'
            })
        
        logger.info(f"📝 Created {len(samples)} Beatles-specific training samples")
        return samples

def main():
    """Main function to demonstrate usage"""
    fetcher = WikipediaPortalFetcher()
    
    # Test with just Music portal first
    print("🎵 Testing Music portal data collection...")
    music_data = fetcher.collect_portal_training_data('Music', max_pages=20)
    
    print(f"\n📋 Sample Music data:")
    for i, sample in enumerate(music_data[:3], 1):
        print(f"{i}. {sample['title']}")
        print(f"   Text: {sample['text'][:100]}...")
        print(f"   Intent: {sample['intent']}")
        print()
    
    # Create Beatles-specific samples
    beatles_samples = fetcher.create_beatles_specific_samples()
    
    # Combine data
    all_data = {'Music': music_data + beatles_samples}
    
    # Save
    fetcher.save_training_data(all_data, 'music_training_sample.json')
    
    print("✅ Sample collection complete!")

if __name__ == "__main__":
    main() 