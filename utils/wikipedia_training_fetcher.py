#!/usr/bin/env python3
"""
Wikipedia Portal Training Data Fetcher
Fetch pages from Wikipedia portals to build better training datasets
Completely FREE - no API costs!
"""

import logging
import wikipedia
import requests
import re
import json
import time
import random
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import sys
import warnings
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress BeautifulSoup warning about parser
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

class WikipediaPortalFetcher:
    """Fetch training data from Wikipedia portals"""
    
    def __init__(self):
        # Set user agent for Wikipedia API compliance
        wikipedia.set_user_agent("SummarEaseAI-Training/1.0 (https://github.com/your-repo)")
        
        # Use 6 categories as intent classifier
        self.portals = {
            'History': {
                'portal_url': 'https://en.wikipedia.org/wiki/Portal:History',
                'intent_label': 'History',
                'subcategories': ['World War II', 'Ancient history', 'American history', 'Medieval history', 'Modern history']
            },
            'Music': {
                'portal_url': 'https://en.wikipedia.org/wiki/Portal:Music',
                'intent_label': 'Music',
                'subcategories': [
                    'The Beatles',  # Primary focus
                    'Beatles songs',
                    'Beatles albums',
                    'John Lennon',
                    'Paul McCartney',
                    'George Harrison',
                    'Ringo Starr',
                    'Rock music',
                    'Popular music',
                    'Music history'
                ]
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
                'subcategories': [
                    'Nvidia',  # Added specific focus
                    'Graphics processing unit',
                    'Computing',
                    'Electronics',
                    'Engineering',
                    'Internet',
                    'Software',
                    'Technological innovations'
                ]
            },
            'Finance': {
                'portal_url': 'https://en.wikipedia.org/wiki/Portal:Finance',
                'intent_label': 'Finance',
                'subcategories': [
                    'Fischer Black',  # Added specific focus
                    'Black-Scholes model',
                    'Nasdaq',  # Added specific focus
                    'Stock market',
                    'Investment',
                    'Banking',
                    'Economics',
                    'Financial history'
                ]
            }
        }
        
        # Set up save directory
        self.save_dir = Path("ml_models/training_data")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize with new data file path
        self.data_file = self.save_dir / "wikipedia_training_data.csv"
        self.existing_data = pd.DataFrame()
        self.existing_urls = set()
        
        # Initialize empty DataFrame with required columns
        self.existing_data = pd.DataFrame(columns=[
            'title', 'text', 'text_clean', 'intent', 'url', 'length', 'word_count'
        ])
        
        logger.info("WikipediaPortalFetcher initialized with 6 balanced categories")
        logger.info(f"Save directory: {self.save_dir}")
        logger.info(f"Data file: {self.data_file}")

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
                    # Search Wikipedia using wikipedia package
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
            # Search for pages in subcategory using wikipedia package
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

    def check_category_balance(self) -> Dict[str, int]:
        """
        Check the balance of categories in existing data
        Returns dict with category counts and identifies imbalances
        """
        if self.existing_data.empty:
            return {}
            
        # Get counts per category
        category_counts = self.existing_data['intent'].value_counts().to_dict()
        
        # Calculate stats
        min_count = min(category_counts.values())
        max_count = max(category_counts.values())
        mean_count = sum(category_counts.values()) / len(category_counts)
        
        # Log the balance status
        logger.info("\nCategory Balance Status:")
        logger.info(f"Minimum samples: {min_count}")
        logger.info(f"Maximum samples: {max_count}")
        logger.info(f"Average samples: {mean_count:.1f}")
        logger.info("\nSamples per category:")
        
        # Check each category
        imbalanced_categories = []
        for category, count in category_counts.items():
            status = ""
            if count > mean_count * 1.5:  # 50% more than average
                status = "⚠️ OVER-REPRESENTED"
                imbalanced_categories.append((category, count, "over"))
            elif count < mean_count * 0.75:  # 25% less than average
                status = "⚠️ UNDER-REPRESENTED"
                imbalanced_categories.append((category, count, "under"))
                
            logger.info(f"{category}: {count} samples {status}")
            
        # Provide recommendations
        if imbalanced_categories:
            logger.info("\nRecommendations:")
            for category, count, status in imbalanced_categories:
                if status == "over":
                    logger.info(f"- Consider skipping {category} (has {count} samples, significantly more than average)")
                else:
                    logger.info(f"- Prioritize collecting {category} (only {count} samples, significantly less than average)")
                    
        return category_counts

    def fetch_all_portal_data(self, max_articles_per_category: int = 50) -> bool:
        """
        Fetch data from all portals
        Returns True if successful, False otherwise
        """
        try:
            # Check current balance
            logger.info("Checking current category balance...")
            category_counts = self.check_category_balance()
            
            if category_counts:
                # Ask about over-represented categories
                min_count = min(category_counts.values())
                for category, count in category_counts.items():
                    if count > min_count * 1.5:  # 50% more samples than minimum
                        logger.warning(f"\n⚠️ {category} has {count} samples (minimum is {min_count})")
                        logger.warning(f"Consider skipping {category} to maintain balance")
                        response = input(f"Do you want to collect more {category} samples? (y/n): ")
                        if response.lower() != 'y':
                            logger.info(f"Skipping {category}")
                            continue
            
            # Proceed with data collection
            total_new = 0
            total_skipped = 0
            
            for portal_name, portal_info in self.portals.items():
                logger.info(f"\nProcessing {portal_name} portal...")
                
                # Adjust max articles based on current balance
                adjusted_max = max_articles_per_category
                if category_counts:
                    current_count = category_counts.get(portal_name, 0)
                    min_count = min(category_counts.values())
                    if current_count > min_count:
                        adjusted_max = min(max_articles_per_category, min_count - current_count)
                        if adjusted_max <= 0:
                            logger.info(f"Skipping {portal_name} - already has sufficient samples")
                            continue
                        logger.info(f"Adjusted target: {adjusted_max} articles to maintain balance")
                
                new_samples, skipped = self.fetch_portal_data(
                    portal_info['portal_url'],
                    portal_info['intent_label'],
                    portal_info['subcategories'],
                    adjusted_max
                )
                
                total_new += new_samples
                total_skipped += skipped
                
                if new_samples > 0:
                    # Save after each portal in case of errors
                    if not self.save_data():
                        logger.error(f"Failed to save data after {portal_name}")
                        return False
                        
            logger.info(f"\nData collection complete! Added {total_new} samples, skipped {total_skipped}")
            
            # Final balance check
            logger.info("\nFinal category balance:")
            self.check_category_balance()
            
            return True
            
        except Exception as e:
            logger.error(f"Error collecting portal data: {e}")
            return False

    def fetch_page_training_data(self, page_title: str, intent_label: str) -> Optional[Dict]:
        """Fetch training data for a single page"""
        try:
            # Get page content using wikipedia package, preserving exact title
            page = wikipedia.page(title=page_title, auto_suggest=False)
            
            # Skip if no content
            if not page or not page.content:
                logger.info(f"PAGE {page_title} SKIPPED - No content")
                return None
                
            # Clean and prepare text
            text = page.content
            text_clean = self.clean_text(text)
            
            # Skip if too short after cleaning
            if len(text_clean.split()) < 50:
                logger.info(f"PAGE {page_title} SKIPPED - Too short")
                return None
                
            # Create training sample
            sample = {
                'title': page_title,  # Use original title
                'text': text,
                'text_clean': text_clean,
                'intent': intent_label,
                'url': page.url,
                'length': len(text),
                'word_count': len(text_clean.split())
            }
            
            logger.info(f"Added {page_title}")
            return sample
            
        except Exception as e:
            logger.info(f"PAGE {page_title} SKIPPED BECAUSE ERROR: {str(e)}")
            return None
    
    def collect_portal_training_data(self, portal_name: str, max_pages: int = 50) -> List[Dict]:
        """Collect training data from a specific portal"""
        try:
            # Get portal info
            portal_info = self.portals[portal_name]
            
            # Get pages from portal
            pages = self.search_portal_pages(portal_name, max_pages)
            
            # Process each page
            training_data = []
            for page_title in pages:
                try:
                    # Get page data
                    page_data = self.fetch_page_training_data(page_title, portal_info['intent_label'])
                    if page_data:
                        training_data.append(page_data)
                        
                except Exception as e:
                    logger.error(f"Error processing page {page_title}: {e}")
                    continue
                    
            return training_data
            
        except Exception as e:
            logger.error(f"Error collecting portal data for {portal_name}: {e}")
            return []
    
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
        """
        Collect Beatles-specific training data
        Returns True if successful, False otherwise
        """
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