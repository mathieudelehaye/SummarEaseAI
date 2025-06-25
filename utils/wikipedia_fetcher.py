# Wikipedia fetching utility
import wikipediaapi
import wikipedia
import logging
import re
from typing import Optional, List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_historical_query(query: str) -> tuple[str, bool]:
    """
    Preprocess queries to better handle historical date searches
    
    Returns:
        tuple: (processed_query, was_converted)
    """
    # Common historical date patterns and their likely topics
    historical_events = {
        r'july\s+20.*1969|20.*july.*1969': 'Apollo 11',  # Simpler, more direct
        r'july\s+16.*1969|16.*july.*1969': 'Apollo 11 launch',
        r'neil\s+armstrong.*moon|moon.*neil\s+armstrong': 'Apollo 11',  # Handle Neil Armstrong queries
        r'buzz\s+aldrin.*moon|moon.*buzz\s+aldrin': 'Apollo 11',  # Handle Buzz Aldrin queries
        r'december\s+7.*1941|7.*december.*1941': 'Pearl Harbor attack December 7 1941',
        r'november\s+22.*1963|22.*november.*1963': 'John F Kennedy assassination November 22 1963',
        r'september\s+11.*2001|11.*september.*2001': 'September 11 attacks 2001',
        r'april\s+14.*1865|14.*april.*1865': 'Abraham Lincoln assassination April 14 1865'
    }
    
    query_lower = query.lower()
    for pattern, replacement in historical_events.items():
        if re.search(pattern, query_lower):
            logger.info(f"Historical query pattern detected: '{query}' -> '{replacement}'")
            return replacement, True
    
    return query, False

def fetch_article(topic: str) -> Optional[str]:
    """
    Fetch Wikipedia article content by topic/title
    """
    try:
        # Preprocess historical queries
        processed_topic, was_converted = preprocess_historical_query(topic)
        
        wiki_wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='SummarEaseAI/1.0 (https://github.com/your-repo) Python/WikipediaAPI'
        )
        page = wiki_wiki.page(processed_topic)
        
        if page.exists():
            logger.info(f"Successfully fetched article: {processed_topic}")
            logger.info(f"🔗 Article URL: https://en.wikipedia.org/wiki/{processed_topic.replace(' ', '_')}")
            logger.info(f"📝 Article title from Wikipedia: {page.title}")
            article_preview = page.text[:500] + "..." if len(page.text) > 500 else page.text
            logger.info(f"📄 Article content starts with: {article_preview}")
            return page.text
        else:
            # Try searching for the topic if direct page doesn't exist
            logger.info(f"Direct page not found for '{processed_topic}', trying search...")
            return search_and_fetch_article(processed_topic)
    except Exception as e:
        logger.error(f"Error fetching article '{topic}': {str(e)}")
        return None

def search_and_fetch_article(query: str, max_results: int = 1) -> Optional[str]:
    """
    Search Wikipedia and fetch the first relevant article
    """
    try:
        # Preprocess historical queries
        processed_query, was_converted = preprocess_historical_query(query)
        
        # Use wikipedia library for better search functionality
        search_results = wikipedia.search(processed_query, results=max_results + 2)
        
        for result in search_results:
            try:
                page = wikipedia.page(result)
                logger.info(f"Found and fetched article: {result}")
                logger.info(f"🔗 Search result URL: https://en.wikipedia.org/wiki/{result.replace(' ', '_')}")
                logger.info(f"📝 Search result page title: {page.title}")
                content_preview = page.content[:500] + "..." if len(page.content) > 500 else page.content
                logger.info(f"📄 Search result content starts with: {content_preview}")
                return page.content
            except wikipedia.exceptions.DisambiguationError as e:
                # Handle disambiguation pages by taking the first option
                try:
                    page = wikipedia.page(e.options[0])
                    logger.info(f"Resolved disambiguation to: {e.options[0]}")
                    return page.content
                except Exception:
                    continue
            except Exception:
                continue
        
        logger.warning(f"No suitable article found for query: {query}")
        return None
        
    except Exception as e:
        logger.error(f"Error searching for article '{query}': {str(e)}")
        return None

def get_article_intro(topic: str) -> Optional[Dict[str, str]]:
    """
    Fetch article title and intro paragraph for training data
    """
    try:
        wiki_wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='SummarEaseAI/1.0 (https://github.com/your-repo) Python/WikipediaAPI'
        )
        page = wiki_wiki.page(topic)
        
        if page.exists():
            intro_paragraph = page.summary.split('\n')[0] if page.summary else ""
            return {
                'title': page.title,
                'intro': intro_paragraph,
                'category': extract_category(page)
            }
    except Exception as e:
        logger.error(f"Error fetching intro for '{topic}': {str(e)}")
    
    return None

def extract_category(page) -> str:
    """
    Extract primary category from Wikipedia page (simplified)
    """
    try:
        categories = list(page.categories.keys())
        if categories:
            # Simple heuristic to get the most relevant category
            for cat in categories:
                if any(keyword in cat.lower() for keyword in ['history', 'science', 'biography', 'technology']):
                    return cat.replace('Category:', '')
        return 'General'
    except:
        return 'General'

def fetch_article_with_conversion_info(topic: str) -> tuple[Optional[str], str, bool]:
    """
    Fetch Wikipedia article and return conversion information
    
    Returns:
        tuple: (article_content, processed_topic, was_converted)
    """
    try:
        # Preprocess historical queries
        processed_topic, was_converted = preprocess_historical_query(topic)
        
        # Fetch article using the processed topic
        article_content = fetch_article(processed_topic) if was_converted else fetch_article(topic)
        
        return article_content, processed_topic, was_converted
        
    except Exception as e:
        logger.error(f"Error fetching article with conversion info '{topic}': {str(e)}")
        return None, topic, False

def enhance_query_with_intent(query: str, intent: str, confidence: float) -> str:
    """
    Enhance search query based on detected intent to get better Wikipedia results
    
    Args:
        query: Original user query
        intent: Detected intent category
        confidence: Confidence score of intent prediction
        
    Returns:
        Enhanced query string for better Wikipedia search
    """
    # Only enhance if we have high confidence in the intent
    if confidence < 0.4:
        logger.info(f"🚫 Not enhancing query - confidence {confidence:.3f} below threshold 0.4")
        return query
    
    # Intent-based query enhancement patterns
    intent_enhancements = {
        'Science': {
            'keywords': ['quantum', 'physics', 'chemistry', 'biology', 'scientific', 'theory', 'principle'],
            'suffixes': ['theory', 'principles']  # More specific than just 'science'
        },
        'History': {
            'keywords': ['war', 'battle', 'historical', 'timeline', 'events', 'period', 'ancient'],
            'suffixes': ['history', 'timeline']
        },
        'Biography': {
            'keywords': ['biography', 'life', 'who was', 'born', 'died', 'achievements'],
            'suffixes': ['biography', 'life']
        },
        'Technology': {
            'keywords': ['technology', 'innovation', 'development', 'advancement', 'computer', 'software'],
            'suffixes': ['technology', 'innovation']
        },
        'Sports': {
            'keywords': ['sports', 'game', 'competition', 'tournament', 'olympics', 'team'],
            'suffixes': ['sports', 'game']
        },
        'Arts': {
            'keywords': ['art', 'artistic', 'cultural', 'creative', 'painting', 'music'],
            'suffixes': ['art', 'culture']
        },
        'Politics': {
            'keywords': ['political', 'government', 'policy', 'democracy', 'election'],
            'suffixes': ['politics', 'government']
        },
        'Geography': {
            'keywords': ['geographic', 'location', 'region', 'country', 'city', 'mountain'],
            'suffixes': ['geography', 'location']
        }
    }
    
    if intent in intent_enhancements:
        enhancement = intent_enhancements[intent]
        
        # Check if query already contains intent-related keywords
        query_lower = query.lower()
        has_intent_keywords = any(keyword in query_lower for keyword in enhancement['keywords'])
        
        # For Science: be extra careful with specific terms
        if intent == 'Science':
            # Don't enhance if already contains specific science terms
            specific_science_terms = ['quantum', 'physics', 'chemistry', 'biology', 'mechanics', 'thermodynamics']
            if any(term in query_lower for term in specific_science_terms):
                logger.info(f"🧪 Science query already contains specific terms - not enhancing: '{query}'")
                return query
        
        # Only enhance if doesn't already have intent keywords
        if not has_intent_keywords:
            # Use the most relevant suffix
            enhanced_query = f"{query} {enhancement['suffixes'][0]}"
            logger.info(f"✨ Enhanced query with intent '{intent}': '{query}' -> '{enhanced_query}'")
            return enhanced_query
        else:
            logger.info(f"✅ Query already contains {intent.lower()} keywords - not enhancing: '{query}'")
    
    return query

def search_and_fetch_article_info(query: str, max_results: int = 1) -> Optional[Dict[str, str]]:
    """
    Search Wikipedia and fetch article with complete information
    
    Returns:
        Dictionary with content, title, url, and summary, or None if not found
    """
    try:
        # Preprocess historical queries
        processed_query, was_converted = preprocess_historical_query(query)
        
        # Log the exact search parameters being sent to Wikipedia
        logger.info("=" * 80)
        logger.info("🔍 WIKIPEDIA API SEARCH REQUEST")
        logger.info("=" * 80)
        logger.info(f"📝 Original query: '{query}'")
        logger.info(f"📝 Processed query: '{processed_query}'")
        logger.info(f"🔄 Query was converted: {was_converted}")
        logger.info(f"📊 Max results requested: {max_results + 2}")
        logger.info(f"🌐 Wikipedia search API call: wikipedia.search('{processed_query}', results={max_results + 2})")
        
        # Use wikipedia library for better search functionality
        search_results = wikipedia.search(processed_query, results=max_results + 2)
        
        logger.info(f"✅ Wikipedia API returned {len(search_results)} search results:")
        for i, result in enumerate(search_results):
            logger.info(f"   {i+1}. '{result}'")
        
        for result in search_results:
            try:
                logger.info(f"🔄 Attempting to fetch page: '{result}'")
                logger.info(f"🌐 Wikipedia page API call: wikipedia.page('{result}')")
                
                page = wikipedia.page(result)
                
                logger.info(f"✅ Successfully fetched page!")
                logger.info(f"📄 Found and fetched article: {result}")
                logger.info(f"🔗 Search result URL: {page.url}")
                logger.info(f"📝 Search result page title: {page.title}")
                content_preview = page.content[:500] + "..." if len(page.content) > 500 else page.content
                logger.info(f"📄 Search result content starts with: {content_preview}")
                logger.info("=" * 80)
                
                return {
                    'content': page.content,
                    'title': page.title,
                    'url': page.url,
                    'summary': page.summary
                }
            except wikipedia.exceptions.DisambiguationError as e:
                # Handle disambiguation pages by taking the first option
                logger.info(f"⚠️  Disambiguation page encountered for '{result}'")
                logger.info(f"📋 Available options: {e.options[:5]}...")  # Show first 5 options
                try:
                    logger.info(f"🔄 Trying first disambiguation option: '{e.options[0]}'")
                    logger.info(f"🌐 Wikipedia page API call: wikipedia.page('{e.options[0]}')")
                    
                    page = wikipedia.page(e.options[0])
                    logger.info(f"✅ Resolved disambiguation to: {e.options[0]}")
                    logger.info("=" * 80)
                    
                    return {
                        'content': page.content,
                        'title': page.title,
                        'url': page.url,
                        'summary': page.summary
                    }
                except Exception as disambiguation_error:
                    logger.error(f"❌ Failed to resolve disambiguation: {str(disambiguation_error)}")
                    continue
            except Exception as page_error:
                logger.error(f"❌ Error fetching page '{result}': {str(page_error)}")
                continue
        
        logger.warning(f"❌ No suitable article found for query: '{query}'")
        logger.info("=" * 80)
        return None
        
    except Exception as e:
        logger.error(f"❌ Error searching for article '{query}': {str(e)}")
        logger.info("=" * 80)
        return None
