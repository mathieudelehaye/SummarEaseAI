#!/usr/bin/env python3
"""
Test Financial Times Data Fetcher
Simple script to test fetching FT articles without any training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.ft_content_fetcher import FTContentFetcher
import json

def test_ft_fetcher():
    """Test the FT content fetcher"""
    print("🚀 Testing Financial Times Content Fetcher")
    print("=" * 50)
    
    # Initialize fetcher (no authentication cookies for now)
    fetcher = FTContentFetcher()
    
    # Test sections to try
    test_sections = ['tech', 'markets', 'companies']
    
    print(f"📰 Testing {len(test_sections)} sections...")
    
    for section in test_sections:
        print(f"\n🔍 Testing section: {section}")
        print("-" * 30)
        
        # Try to extract article links
        try:
            links = fetcher.extract_article_links(section, max_links=3)
            
            if links:
                print(f"✅ Found {len(links)} article links:")
                for i, link in enumerate(links, 1):
                    print(f"   {i}. {link}")
                
                # Try to fetch first article
                if links:
                    print(f"\n📄 Attempting to fetch first article...")
                    article = fetcher.fetch_article_content(links[0])
                    
                    if article:
                        print(f"✅ Successfully fetched article:")
                        print(f"   Title: {article.title}")
                        print(f"   Category: {article.category}")
                        print(f"   Word count: {article.word_count}")
                        print(f"   Author: {article.author}")
                        print(f"   Summary: {article.summary[:100]}...")
                    else:
                        print("❌ Failed to fetch article content (likely paywall)")
            else:
                print("❌ No article links found")
                
        except Exception as e:
            print(f"❌ Error testing section {section}: {e}")
    
    print(f"\n🎯 Test completed!")
    print("\n📝 Notes:")
    print("- Free articles may work without authentication")
    print("- Premium articles require FT subscription + cookies")
    print("- Rate limiting is applied (2 seconds between requests)")

def test_with_sample_data():
    """Test with sample data structure"""
    print("\n🧪 Testing with sample data structure...")
    
    # Create sample training data format
    sample_data = [
        {
            'title': 'Sample Tech Article',
            'text': 'This is a sample technology article about AI and machine learning developments.',
            'intent': 'Technology',
            'url': 'https://www.ft.com/content/sample-tech',
            'source': 'Financial Times',
            'word_count': 12,
            'author': 'Sample Author'
        },
        {
            'title': 'Sample Market Article', 
            'text': 'This is a sample markets article about stock performance and trading.',
            'intent': 'Finance',
            'url': 'https://www.ft.com/content/sample-markets',
            'source': 'Financial Times',
            'word_count': 10,
            'author': 'Market Reporter'
        }
    ]
    
    # Save sample data
    with open('sample_ft_data.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Sample FT data structure saved to 'sample_ft_data.json'")
    
    # Show categories
    categories = {}
    for item in sample_data:
        cat = item['intent']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("📊 Sample data categories:")
    for category, count in categories.items():
        print(f"   {category}: {count} articles")

if __name__ == "__main__":
    test_ft_fetcher()
    test_with_sample_data() 