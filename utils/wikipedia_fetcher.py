# Wikipedia fetching utility
import wikipediaapi
import wikipedia
import logging
from typing import Optional, List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_article(topic: str) -> Optional[str]:
    """
    Fetch Wikipedia article content by topic/title
    """
    try:
        wiki_wiki = wikipediaapi.Wikipedia('en')
        page = wiki_wiki.page(topic)
        
        if page.exists():
            logger.info(f"Successfully fetched article: {topic}")
            return page.text
        else:
            # Try searching for the topic if direct page doesn't exist
            logger.info(f"Direct page not found for '{topic}', trying search...")
            return search_and_fetch_article(topic)
    except Exception as e:
        logger.error(f"Error fetching article '{topic}': {str(e)}")
        return None

def search_and_fetch_article(query: str, max_results: int = 1) -> Optional[str]:
    """
    Search Wikipedia and fetch the first relevant article
    """
    try:
        # Use wikipedia library for better search functionality
        search_results = wikipedia.search(query, results=max_results + 2)
        
        for result in search_results:
            try:
                page = wikipedia.page(result)
                logger.info(f"Found and fetched article: {result}")
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
        wiki_wiki = wikipediaapi.Wikipedia('en')
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
