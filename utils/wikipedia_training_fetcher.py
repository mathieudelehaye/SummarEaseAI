#!/usr/bin/env python3
"""
Wikipedia Portal Training Data Fetcher
Fetch pages from Wikipedia portals to build better training datasets
Completely FREE - no API costs!
"""

import logging
import wikipedia
import wikipediaapi
import requests
import re
import json
import time
import random
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys

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
        
        # Use same 6 categories as intent classifier
        self.portals = {
            'History': {
                'portal_url': 'https://en.wikipedia.org/wiki/Portal:History',
                'intent_label': 'History',
                'subcategories': ['World War II', 'Ancient history', 'American history', 'Medieval history', 'Modern history']
            },
            'Music': {
                'portal_url': 'https://en.wikipedia.org/wiki/Portal:Music',
                'intent_label': 'Music',
                'subcategories': ['Popular music', 'Classical music', 'Jazz', 'Rock music', 'Hip hop', 'Music history']
            },
            'Science': {
                'portal_url': 'https://en.wikipedia.org/wiki/Portal:Science',
                'intent_label': 'Science',
                'subcategories': ['Physics', 'Chemistry', 'Biology', 'Mathematics', 'Computer science', 'Scientific discoveries']
            },
            'Sports': {
                'portal_url': 'https://en.wikipedia.org/wiki/Portal:Sports',
                'intent_label': 'Sports',
                'subcategories': ['Football', 'Basketball', 'Tennis', 'Olympics', 'Baseball', 'Sports history']
            },
            'Technology': {
                'portal_url': 'https://en.wikipedia.org/wiki/Portal:Technology',
                'intent_label': 'Technology',
                'subcategories': ['Computing', 'Electronics', 'Engineering', 'Internet', 'Software', 'Technological innovations']
            },
            'Finance': {
                'portal_url': 'https://en.wikipedia.org/wiki/Portal:Finance',
                'intent_label': 'Finance',
                'subcategories': ['Economics', 'Banking', 'Investment', 'Stock market', 'Business', 'Financial history']
            }
        }
        
        # Set up save directory
        self.save_dir = Path("tensorflow_models/training_data")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing data if available
        self.data_file = self.save_dir / "enhanced_wikipedia_training_data.csv"
        self.existing_data = pd.DataFrame()
        self.existing_urls = set()
        
        if self.data_file.exists():
            try:
                self.existing_data = pd.read_csv(self.data_file)
                
                # Handle missing columns
                required_columns = ['title', 'text', 'text_clean', 'intent', 'url', 'length', 'word_count']
                missing_columns = [col for col in required_columns if col not in self.existing_data.columns]
                
                if missing_columns:
                    logger.warning(f"Adding missing columns: {missing_columns}")
                    for col in missing_columns:
                        self.existing_data[col] = ''
                
                # Filter out any rows with invalid categories
                valid_categories = list(self.portals.keys())
                invalid_mask = ~self.existing_data['intent'].isin(valid_categories)
                if invalid_mask.any():
                    invalid_rows = self.existing_data[invalid_mask]
                    logger.warning(f"Removing {len(invalid_rows)} rows with invalid categories:")
                    for cat, count in invalid_rows['intent'].value_counts().items():
                        logger.warning(f"   {cat}: {count} samples")
                    self.existing_data = self.existing_data[~invalid_mask]
                
                # Get existing URLs, handling the case where url column might be empty
                if 'url' in self.existing_data.columns:
                    self.existing_urls = set(self.existing_data['url'].dropna())
                
                # Clean up any rows with missing required data
                self.existing_data = self.existing_data.dropna(subset=['text', 'intent'])
                
                logger.info(f"Loaded {len(self.existing_data)} existing samples")
                logger.info("\nExisting samples per category:")
                for intent, count in self.existing_data['intent'].value_counts().items():
                    logger.info(f"   {intent}: {count} samples")
                
            except Exception as e:
                logger.warning(f"Error loading existing data: {e}")
                logger.warning("Starting with empty dataset")
                self.existing_data = pd.DataFrame(columns=required_columns)
        
        logger.info("WikipediaPortalFetcher initialized with 6 standard categories")
        logger.info(f"Save directory: {self.save_dir}")

    def clean_text(self, text: str) -> str:
        """Clean text for model training"""
        if pd.isna(text):
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace numbers with <NUM> token to preserve numeric patterns
        text = re.sub(r'\d+(\.\d+)?', ' <NUM> ', text)
        
        # Keep certain special characters that might be meaningful
        text = re.sub(r'[^\w\s\$\%\-\.]', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text

    def search_portal_pages(self, portal_name: str, max_pages: int = 50) -> List[str]:
        """Search for pages in a portal"""
        try:
            # Get subcategories for this portal
            subcategories = self.portals[portal_name]['subcategories']
            
            # Search for pages in each subcategory
            all_pages = []
            for subcategory in subcategories:
                search_query = f"{portal_name} {subcategory}"
                logger.debug(f"Searching: {search_query}")
                
                try:
                    # Search Wikipedia
                    search_results = wikipedia.search(search_query, results=max_pages//len(subcategories))
                    all_pages.extend(search_results)
                    
                    # Small delay between searches
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Error searching {search_query}: {e}")
                    continue
            
            # Remove duplicates while preserving order
            seen = set()
            unique_pages = [x for x in all_pages if not (x in seen or seen.add(x))]
            
            # Limit to max_pages
            return unique_pages[:max_pages]
            
        except Exception as e:
            logger.error(f"Error searching portal {portal_name}: {e}")
            return []

    def fetch_portal_data(self, portal_url: str, intent_label: str, subcategories: List[str], max_articles: int) -> Tuple[int, int]:
        """
        Fetch data from a specific portal
        Returns tuple of (new_samples_count, skipped_count)
        """
        new_samples = 0
        skipped = 0
        
        # Process each subcategory
        for subcategory in subcategories:
            try:
                # Get pages from subcategory
                pages = self.get_subcategory_pages(portal_url, subcategory)
                
                # Process each page
                for page_title in pages[:max_articles // len(subcategories)]:
                    if page_title in self.existing_urls:
                        logger.info(f"Skipping {page_title} - already exists")
                        skipped += 1
                        continue
                        
                    page_data = self.fetch_page_training_data(page_title, intent_label)
                    if page_data:
                        self.existing_data = pd.concat([self.existing_data, pd.DataFrame([page_data])], ignore_index=True)
                        self.existing_urls.add(page_data['url'])
                        new_samples += 1
                        logger.info(f"Added {page_title}")
                    else:
                        skipped += 1
                        
            except Exception as e:
                logger.error(f"Error processing subcategory {subcategory}: {e}")
                continue
                
        return new_samples, skipped

    def save_data(self) -> bool:
        """Save the current dataset to CSV"""
        try:
            # Ensure required columns exist and are in correct order
            required_columns = ['title', 'text', 'text_clean', 'intent', 'url', 'length', 'word_count']
            self.existing_data = self.existing_data.reindex(columns=required_columns)
            
            # Save to CSV
            self.existing_data.to_csv(self.data_file, index=False)
            logger.info(f"Saved {len(self.existing_data)} samples to {self.data_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False

    def get_subcategory_pages(self, portal_url: str, subcategory: str) -> List[str]:
        """Get list of page titles from a subcategory"""
        try:
            # Search for pages in subcategory
            search_query = f"{subcategory} {portal_url.split('Portal:')[1]}"
            pages = wikipedia.search(search_query, results=20)
            return pages
            
        except Exception as e:
            logger.error(f"Error getting pages for {subcategory}: {e}")
            return []

    def get_categories_to_fetch(self) -> Dict[str, bool]:
        """
        Determine which categories need more samples based on current distribution.
        Returns a dict of category -> should_fetch boolean.
        """
        if self.existing_data.empty:
            # If no data exists, fetch all categories
            return {cat: True for cat in self.portals.keys()}
            
        # Get current counts per category
        category_counts = self.existing_data['intent'].value_counts()
        
        # Find categories with lowest sample count
        min_samples = category_counts.min() if not category_counts.empty else 0
        
        # Determine which categories to fetch
        should_fetch = {}
        for category in self.portals.keys():
            current_samples = category_counts.get(category, 0)
            # Skip if this category already has significantly more samples
            if current_samples >= min_samples * 1.5:  # 50% more than minimum is our threshold
                should_fetch[category] = False
                logger.info(f"Skipping {category} - already has {current_samples} samples (min: {min_samples})")
            else:
                should_fetch[category] = True
                if current_samples > 0:
                    logger.info(f"Will fetch {category} - has {current_samples} samples (min: {min_samples})")
                else:
                    logger.info(f"Will fetch {category} - no existing samples")
                    
        return should_fetch

    def fetch_all_portal_data(self, max_articles_per_category: int = 50) -> None:
        """Fetch data from all Wikipedia portals"""
        try:
            # Get categories that need more samples
            categories_to_fetch = self.get_categories_to_fetch()
            
            # Track overall stats
            total_new = 0
            total_skipped = 0
            
            # Show initial distribution
            logger.info("\n=== Initial Category Distribution ===")
            initial_counts = self.existing_data['intent'].value_counts()
            min_samples = initial_counts.min() if not initial_counts.empty else 0
            for category in self.portals.keys():
                current_count = initial_counts.get(category, 0)
                logger.info(f"   {category}: {current_count} samples")
                if not categories_to_fetch[category]:
                    logger.warning(f"   ⚠️  SKIPPING {category} - Already has {current_count} samples (50% more than minimum of {min_samples})")
            
            for category, portal_info in self.portals.items():
                if not categories_to_fetch[category]:
                    continue
                    
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing {category} portal...")
                logger.info(f"{'='*60}")
                
                # Show existing samples for this category
                existing_count = len(self.existing_data[self.existing_data['intent'] == category])
                logger.info(f"Existing {category} samples: {existing_count}")
                
                try:
                    new_samples, skipped = self.fetch_portal_data(
                        portal_info['portal_url'],
                        portal_info['intent_label'],
                        portal_info['subcategories'],
                        max_articles_per_category
                    )
                    total_new += new_samples
                    total_skipped += skipped
                    
                    # Show updated count for this category
                    new_count = len(self.existing_data[self.existing_data['intent'] == category])
                    logger.info(f"\nCategory {category} update:")
                    logger.info(f"   Before: {existing_count} samples")
                    logger.info(f"   Added: {new_samples} samples")
                    logger.info(f"   After: {new_count} samples")
                    logger.info(f"   Skipped: {skipped} duplicates")
                    
                    # Save after each category in case of errors
                    if new_samples > 0:
                        self.save_data()
                        
                except Exception as e:
                    logger.error(f"Error processing {category} portal: {e}")
                    continue
            
            # Final save and stats
            if total_new > 0:
                self.save_data()
                
            logger.info("\n=== Final Category Distribution ===")
            final_counts = self.existing_data['intent'].value_counts()
            for category in self.portals.keys():
                initial = initial_counts.get(category, 0)
                final = final_counts.get(category, 0)
                diff = final - initial
                if diff > 0:
                    logger.info(f"   {category}: {initial} → {final} (+{diff} samples)")
                else:
                    if not categories_to_fetch[category]:
                        logger.info(f"   {category}: {initial} → {final} (SKIPPED - already balanced)")
                    else:
                        logger.info(f"   {category}: {initial} → {final} (no new samples)")
            
            logger.info(f"\nCollection complete! Added {total_new} new samples, skipped {total_skipped} duplicates")
            
        except Exception as e:
            logger.error(f"Error in fetch_all_portal_data: {e}")
            raise

    def fetch_page_training_data(self, page_title: str, intent_label: str) -> Optional[Dict]:
        """Fetch and process a single Wikipedia page"""
        try:
            # Get page
            page = wikipedia.page(page_title, auto_suggest=False)
            
            # Check for duplicates by URL and title
            if page.url in self.existing_urls:
                # Find the existing article details
                existing_article = self.existing_data[self.existing_data['url'] == page.url].iloc[0]
                logger.info(f"PAGE {page.title} SKIPPED BECAUSE ALREADY DOWNLOADED AND UNCHANGED (matches {existing_article['title']})")
                return None
            
            # Also check title similarity to catch redirects/variants
            if not self.existing_data.empty:
                # Convert titles to lowercase for comparison
                page_title_lower = page.title.lower()
                existing_titles = self.existing_data['title'].str.lower()
                
                # Check for exact matches
                exact_matches = self.existing_data[existing_titles == page_title_lower]
                if not exact_matches.empty:
                    logger.info(f"PAGE {page.title} SKIPPED BECAUSE ALREADY DOWNLOADED AND UNCHANGED (exact title match)")
                    return None
                
                # Check for titles that contain this one or vice versa
                similar_titles = existing_titles[
                    existing_titles.str.contains(page_title_lower, regex=False) |
                    existing_titles.apply(lambda x: page_title_lower in x)
                ]
                if not similar_titles.empty:
                    similar_articles = self.existing_data[existing_titles.isin(similar_titles)]
                    logger.info(f"PAGE {page.title} SKIPPED BECAUSE ALREADY DOWNLOADED AND UNCHANGED (similar to: {', '.join(similar_articles['title'].values[:3])})")
                    return None
            
            # Extract intro paragraph (first paragraph)
            intro = page.summary.split('\n')[0] if page.summary else ""
            
            # Clean the intro text
            intro = re.sub(r'\([^)]*\)', '', intro)  # Remove parentheses
            intro = re.sub(r'\s+', ' ', intro).strip()  # Clean whitespace
            
            # Skip very short articles
            if len(intro) < 50:
                logger.info(f"PAGE {page.title} SKIPPED BECAUSE TOO SHORT ({len(intro)} chars)")
                return None
            
            # Create training sample
            training_data = {
                'title': page.title,
                'text': intro,
                'text_clean': self.clean_text(intro),
                'intent': intent_label,
                'url': page.url,
                'length': len(intro),
                'word_count': len(intro.split())
            }
            
            logger.info(f"✅ DOWNLOADED: {page.title} → {intent_label}")
            return training_data
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Try first disambiguation option
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                
                # Check for duplicates
                if page.url in self.existing_urls:
                    existing_article = self.existing_data[self.existing_data['url'] == page.url].iloc[0]
                    logger.info(f"PAGE {e.options[0]} SKIPPED BECAUSE ALREADY DOWNLOADED AND UNCHANGED (disambiguation resolves to {existing_article['title']})")
                    return None
                
                # Check title similarity for disambiguation
                if not self.existing_data.empty:
                    page_title_lower = page.title.lower()
                    exact_matches = self.existing_data[self.existing_data['title'].str.lower() == page_title_lower]
                    if not exact_matches.empty:
                        logger.info(f"PAGE {e.options[0]} SKIPPED BECAUSE ALREADY DOWNLOADED AND UNCHANGED (disambiguation exact match)")
                        return None
                
                intro = page.summary.split('\n')[0] if page.summary else ""
                intro = re.sub(r'\([^)]*\)', '', intro)
                intro = re.sub(r'\s+', ' ', intro).strip()
                
                if len(intro) >= 50:
                    return {
                        'title': page.title,
                        'text': intro,
                        'text_clean': self.clean_text(intro),
                        'intent': intent_label,
                        'url': page.url,
                        'length': len(intro),
                        'word_count': len(intro.split()),
                        'disambiguation_resolved': True
                    }
                else:
                    logger.info(f"PAGE {e.options[0]} SKIPPED BECAUSE TOO SHORT ({len(intro)} chars)")
            except Exception as e:
                logger.info(f"PAGE {page_title} SKIPPED BECAUSE DISAMBIGUATION FAILED: {str(e)}")
            
            return None
            
        except wikipedia.exceptions.PageError:
            logger.info(f"PAGE {page_title} SKIPPED BECAUSE NOT FOUND")
            return None
            
        except Exception as e:
            logger.info(f"PAGE {page_title} SKIPPED BECAUSE ERROR: {str(e)}")
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
        
        logger.info(f"✅ Successfully collected {len(training_data)} new samples for {portal_name}")
        return training_data
    
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

    def collect_beatles_data(self, max_articles: int = 30) -> bool:
        """Collect additional Beatles-specific training data"""
        try:
            logger.info("\nCollecting Beatles-specific training data...")
            
            # Only collect Beatles data if we successfully collected main data
            if not self.fetch_all_portal_data(max_articles_per_category=50):
                logger.error("Main data collection failed, skipping Beatles data")
                return False
                
            # Search for Beatles-related articles
            search_terms = [
                "The Beatles songs",
                "The Beatles albums",
                "The Beatles members",
                "The Beatles history",
                "Beatles concerts",
                "Beatles recordings"
            ]
            
            new_samples = 0
            skipped = 0
            
            for term in search_terms:
                try:
                    pages = wikipedia.search(term, results=5)
                    for page_title in pages:
                        if page_title in self.existing_urls:
                            logger.info(f"Skipping {page_title} - already exists")
                            skipped += 1
                            continue
                            
                        page_data = self.fetch_page_training_data(page_title, "Music")
                        if page_data:
                            self.existing_data = pd.concat([self.existing_data, pd.DataFrame([page_data])], ignore_index=True)
                            self.existing_urls.add(page_data['url'])
                            new_samples += 1
                            logger.info(f"Added {page_title}")
                        else:
                            skipped += 1
                            
                except Exception as e:
                    logger.error(f"Error processing search term '{term}': {e}")
                    continue
            
            # Save Beatles data
            if new_samples > 0:
                if self.save_data():
                    logger.info(f"\nBeatles data collection complete! Added {new_samples} samples, skipped {skipped}")
                    return True
                else:
                    logger.error("Failed to save Beatles data")
                    return False
            else:
                logger.info("\nNo new Beatles samples to add")
                return True
                
        except Exception as e:
            logger.error(f"Error collecting Beatles data: {e}")
            return False

def main():
    """Main function to collect training data"""
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Initialize fetcher
        fetcher = WikipediaPortalFetcher()
        
        # Collect main portal data
        logger.info("Starting Wikipedia training data collection...")
        if fetcher.fetch_all_portal_data(max_articles_per_category=50):
            # If main collection succeeds, try to add Beatles data
            fetcher.collect_beatles_data(max_articles=30)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 