#!/usr/bin/env python3
"""
Financial Times Content Fetcher
Fetch training data from FT articles with proper authentication and rate limiting
"""

import requests
import time
import json
import re
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FTArticle:
    """Data class for FT article information"""
    title: str
    content: str
    category: str
    url: str
    published_date: Optional[str] = None
    author: Optional[str] = None
    summary: str = ""
    word_count: int = 0

class FTContentFetcher:
    """Fetch content from Financial Times with proper authentication"""
    
    def __init__(self, session_cookies: Optional[Dict[str, str]] = None):
        """
        Initialize FT content fetcher
        
        Args:
            session_cookies: Dictionary of authentication cookies from your browser
                            (extract from browser dev tools after logging in)
        """
        self.base_url = "https://www.ft.com"
        self.session = requests.Session()
        
        # Set user agent to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # Add authentication cookies if provided
        if session_cookies:
            self.session.cookies.update(session_cookies)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_delay = 2.0  # Minimum 2 seconds between requests
        
        # Category mappings for training data
        self.ft_categories = {
            'markets': 'Finance',
            'companies': 'Business', 
            'tech': 'Technology',
            'world': 'Politics',
            'us': 'Politics',
            'opinion': 'Analysis',
            'climate': 'Environment',
            'work-careers': 'Business'
        }
    
    def _rate_limit(self):
        """Implement rate limiting to be respectful"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            logger.info(f"⏳ Rate limiting: sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def extract_article_links(self, section: str = "tech", max_links: int = 20) -> List[str]:
        """
        Extract article links from FT section pages
        
        Args:
            section: FT section (tech, markets, companies, etc.)
            max_links: Maximum number of links to extract
            
        Returns:
            List of article URLs
        """
        self._rate_limit()
        
        section_url = f"{self.base_url}/{section}"
        logger.info(f"🔍 Extracting links from: {section_url}")
        
        try:
            response = self.session.get(section_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links (FT uses specific CSS classes)
            article_links = []
            
            # Common FT article link selectors
            selectors = [
                'a[href*="/content/"]',  # FT article URLs contain /content/
                'a[data-trackable="heading-link"]',  # FT tracking attribute
                '.o-teaser__heading a',  # FT teaser component
                '.js-teaser-heading-link'  # FT JavaScript class
            ]
            
            for selector in selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href and isinstance(href, str) and href.startswith('/'):
                        full_url = urljoin(self.base_url, href)
                        if full_url not in article_links:
                            article_links.append(full_url)
                            if len(article_links) >= max_links:
                                break
                
                if len(article_links) >= max_links:
                    break
            
            logger.info(f"✅ Found {len(article_links)} article links in {section}")
            return article_links[:max_links]
            
        except requests.RequestException as e:
            logger.error(f"❌ Error fetching {section_url}: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Error parsing {section_url}: {e}")
            return []
    
    def fetch_article_content(self, url: str) -> Optional[FTArticle]:
        """
        Fetch and parse a single FT article
        
        Args:
            url: Article URL
            
        Returns:
            FTArticle object or None if failed
        """
        self._rate_limit()
        
        logger.info(f"📄 Fetching article: {url}")
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article title
            title_selectors = [
                'h1[data-trackable="heading"]',
                'h1.o-typography-headline',
                '.article-title h1',
                'h1'
            ]
            
            title = None
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text().strip()
                    break
            
            if not title:
                logger.warning(f"⚠️  No title found for {url}")
                return None
            
            # Extract article content
            content_selectors = [
                '.article-body',
                '[data-trackable="article-body"]',
                '.n-content-body',
                '.article-content'
            ]
            
            content_paragraphs = []
            for selector in content_selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    paragraphs = content_div.find_all('p')
                    content_paragraphs = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
                    break
            
            if not content_paragraphs:
                logger.warning(f"⚠️  No content found for {url}")
                return None
            
            # Join paragraphs and clean content
            content = ' '.join(content_paragraphs)
            content = re.sub(r'\s+', ' ', content).strip()
            
            # Extract summary (first paragraph or first 200 chars)
            summary = content_paragraphs[0] if content_paragraphs else content[:200] + "..."
            
            # Determine category from URL
            category = self._determine_category(url)
            
            # Extract author if available
            author_selectors = [
                '[data-trackable="author"]',
                '.article-author',
                '.o-author-name'
            ]
            
            author = None
            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem:
                    author = author_elem.get_text().strip()
                    break
            
            article = FTArticle(
                title=title,
                content=content,
                category=category,
                url=url,
                author=author,
                summary=summary,
                word_count=len(content.split())
            )
            
            logger.info(f"✅ Successfully fetched: {title[:50]}... ({article.word_count} words)")
            return article
            
        except requests.RequestException as e:
            logger.error(f"❌ Network error fetching {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error parsing {url}: {e}")
            return None
    
    def _determine_category(self, url: str) -> str:
        """Determine article category from URL"""
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.split('/')
        
        for part in path_parts:
            if part in self.ft_categories:
                return self.ft_categories[part]
        
        # Default category
        return 'General'
    
    def collect_training_data(self, sections: List[str], articles_per_section: int = 10) -> List[Dict]:
        """
        Collect training data from multiple FT sections
        
        Args:
            sections: List of FT sections to scrape
            articles_per_section: Number of articles per section
            
        Returns:
            List of training data dictionaries
        """
        logger.info(f"🚀 Collecting FT training data from {len(sections)} sections")
        
        all_training_data = []
        
        for section in sections:
            logger.info(f"\n📰 Processing section: {section}")
            
            # Get article links
            article_links = self.extract_article_links(section, articles_per_section)
            
            if not article_links:
                logger.warning(f"⚠️  No links found for section: {section}")
                continue
            
            # Fetch articles
            section_data = []
            for i, url in enumerate(article_links, 1):
                logger.info(f"   {i}/{len(article_links)}: Processing article")
                
                article = self.fetch_article_content(url)
                if article and len(article.content) > 200:  # Ensure substantial content
                    training_sample = {
                        'title': article.title,
                        'text': article.summary,  # Use summary for training
                        'intent': article.category,
                        'url': article.url,
                        'source': 'Financial Times',
                        'word_count': article.word_count,
                        'author': article.author,
                        'full_content': article.content  # Keep full content for reference
                    }
                    section_data.append(training_sample)
                    all_training_data.append(training_sample)
                
                # Extra delay between articles
                time.sleep(1)
            
            logger.info(f"✅ Collected {len(section_data)} articles from {section}")
        
        logger.info(f"🎉 Total FT training data collected: {len(all_training_data)} articles")
        return all_training_data
    
    def save_training_data(self, training_data: List[Dict], filename: str = "ft_training_data.json"):
        """Save training data to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 FT training data saved to: {filename}")
            
            # Create summary
            categories = {}
            for item in training_data:
                cat = item['intent']
                categories[cat] = categories.get(cat, 0) + 1
            
            logger.info("📊 Training data summary:")
            for category, count in categories.items():
                logger.info(f"   {category}: {count} articles")
                
        except Exception as e:
            logger.error(f"❌ Error saving training data: {e}")

def main():
    """Example usage - requires authentication cookies"""
    
    print("🚨 IMPORTANT: This requires FT subscription access!")
    print("📋 To use this:")
    print("1. Log into FT in your browser")
    print("2. Open Developer Tools (F12)")
    print("3. Go to Application/Storage > Cookies")
    print("4. Copy session cookies and add them to the fetcher")
    print()
    
    # Example with no cookies (will only work for free articles)
    fetcher = FTContentFetcher()
    
    # Test sections
    sections = ['tech', 'markets', 'companies']
    
    # Collect training data
    training_data = fetcher.collect_training_data(sections, articles_per_section=5)
    
    if training_data:
        fetcher.save_training_data(training_data)
    else:
        print("❌ No training data collected - authentication may be required")

if __name__ == "__main__":
    main() 