"""
Advanced Search Strategies for Multi-Source Wikipedia Agent
Shows different approaches to find multiple relevant articles about any topic.
"""

import logging
from typing import List, Dict, Tuple, Set
from utils.wikipedia_fetcher import search_and_fetch_article_info
import wikipedia

logger = logging.getLogger(__name__)

class AdvancedSearchStrategies:
    """
    Advanced search strategies for finding multiple relevant Wikipedia articles.
    """
    
    def find_multiple_articles(self, query: str, max_articles: int = 5) -> List[Dict]:
        """
        Main method that combines multiple search strategies to find diverse, relevant articles.
        """
        logger.info(f"🔍 Advanced search for: '{query}' (targeting {max_articles} articles)")
        
        all_articles = []
        seen_titles = set()
        
        # Strategy 1: Semantic expansion search
        semantic_articles = self.semantic_expansion_search(query, max_results=3)
        self._add_unique_articles(all_articles, semantic_articles, seen_titles)
        
        # Strategy 2: Hierarchical search (broader -> narrower)
        hierarchical_articles = self.hierarchical_search(query, max_results=3)
        self._add_unique_articles(all_articles, hierarchical_articles, seen_titles)
        
        # Strategy 3: Related topics search
        related_articles = self.related_topics_search(query, max_results=3)
        self._add_unique_articles(all_articles, related_articles, seen_titles)
        
        # Strategy 4: Category-based search
        category_articles = self.category_based_search(query, max_results=3)
        self._add_unique_articles(all_articles, category_articles, seen_titles)
        
        # Strategy 5: Temporal search (if applicable)
        temporal_articles = self.temporal_search(query, max_results=2)
        self._add_unique_articles(all_articles, temporal_articles, seen_titles)
        
        # Rank by relevance and return top articles
        ranked_articles = self._rank_articles_by_relevance(all_articles, query)
        
        logger.info(f"📚 Found {len(ranked_articles)} total unique articles")
        return ranked_articles[:max_articles]
    
    def semantic_expansion_search(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Strategy 1: Expand query semantically to find related concepts.
        Example: "The Beatles" -> "British rock band", "Liverpool musicians", "1960s music"
        """
        logger.info(f"🧠 Semantic expansion search for: '{query}'")
        
        # Define semantic expansions for common topics
        expansions = self._get_semantic_expansions(query)
        
        articles = []
        for expansion in expansions[:max_results]:
            logger.info(f"  🔍 Searching expansion: '{expansion}'")
            article = search_and_fetch_article_info(expansion)
            if article:
                article['search_method'] = 'semantic_expansion'
                article['expansion_query'] = expansion
                articles.append(article)
                logger.info(f"    ✅ Found: '{article['title']}'")
        
        return articles
    
    def hierarchical_search(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Strategy 2: Search at different levels of specificity.
        Example: "John Lennon" -> "John Lennon" (specific), "The Beatles" (broader), "Rock music" (broadest)
        """
        logger.info(f"🔺 Hierarchical search for: '{query}'")
        
        # Generate hierarchical queries (specific -> broad)
        hierarchical_queries = self._get_hierarchical_queries(query)
        
        articles = []
        for level, h_query in enumerate(hierarchical_queries[:max_results]):
            logger.info(f"  🔍 Level {level+1}: '{h_query}'")
            article = search_and_fetch_article_info(h_query)
            if article:
                article['search_method'] = 'hierarchical'
                article['hierarchy_level'] = level + 1
                article['hierarchy_query'] = h_query
                articles.append(article)
                logger.info(f"    ✅ Found: '{article['title']}'")
        
        return articles
    
    def related_topics_search(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Strategy 3: Search for related topics and associated concepts.
        Example: "The Beatles" -> "Beatles members", "Beatles albums", "Beatles influence"
        """
        logger.info(f"🔗 Related topics search for: '{query}'")
        
        related_queries = self._get_related_topics(query)
        
        articles = []
        for related in related_queries[:max_results]:
            logger.info(f"  🔍 Related topic: '{related}'")
            article = search_and_fetch_article_info(related)
            if article:
                article['search_method'] = 'related_topics'
                article['related_query'] = related
                articles.append(article)
                logger.info(f"    ✅ Found: '{article['title']}'")
        
        return articles
    
    def category_based_search(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Strategy 4: Search within specific Wikipedia categories.
        Example: If "Beatles" -> search in "British rock bands", "1960s music groups"
        """
        logger.info(f"📂 Category-based search for: '{query}'")
        
        # This would use Wikipedia's category system
        # For now, we'll simulate with category-aware queries
        category_queries = self._get_category_queries(query)
        
        articles = []
        for category_query in category_queries[:max_results]:
            logger.info(f"  🔍 Category search: '{category_query}'")
            article = search_and_fetch_article_info(category_query)
            if article:
                article['search_method'] = 'category_based'
                article['category_query'] = category_query
                articles.append(article)
                logger.info(f"    ✅ Found: '{article['title']}'")
        
        return articles
    
    def temporal_search(self, query: str, max_results: int = 2) -> List[Dict]:
        """
        Strategy 5: Search for time-related aspects (before, during, after).
        Example: "Apollo 11" -> "Space Race" (before), "Moon landing" (during), "Space exploration" (after)
        """
        logger.info(f"⏰ Temporal search for: '{query}'")
        
        temporal_queries = self._get_temporal_queries(query)
        
        articles = []
        for temporal in temporal_queries[:max_results]:
            logger.info(f"  🔍 Temporal aspect: '{temporal}'")
            article = search_and_fetch_article_info(temporal)
            if article:
                article['search_method'] = 'temporal'
                article['temporal_query'] = temporal
                articles.append(article)
                logger.info(f"    ✅ Found: '{article['title']}'")
        
        return articles
    
    def _get_semantic_expansions(self, query: str) -> List[str]:
        """Generate semantically related search terms."""
        query_lower = query.lower()
        
        # Topic-specific semantic expansions
        if 'beatles' in query_lower:
            return [
                "British rock band Liverpool",
                "1960s popular music",
                "Beatlemania cultural phenomenon",
                "Rock and roll history"
            ]
        elif 'apollo 11' in query_lower or 'moon landing' in query_lower:
            return [
                "NASA space program",
                "Space Race Cold War",
                "Lunar exploration missions",
                "Astronaut Neil Armstrong"
            ]
        elif 'quantum' in query_lower:
            return [
                "Quantum physics principles",
                "Quantum mechanics theory",
                "Particle physics quantum",
                "Modern physics quantum theory"
            ]
        else:
            # Generic expansions
            return [
                f"{query} overview",
                f"{query} history",
                f"{query} background"
            ]
    
    def _get_hierarchical_queries(self, query: str) -> List[str]:
        """Generate queries at different levels of specificity."""
        query_lower = query.lower()
        
        if 'beatles' in query_lower:
            return [
                "The Beatles",  # Specific
                "British rock bands 1960s",  # Medium
                "Rock music history"  # Broad
            ]
        elif 'john lennon' in query_lower:
            return [
                "John Lennon",  # Specific
                "The Beatles members",  # Medium
                "Rock musicians"  # Broad
            ]
        elif 'apollo 11' in query_lower:
            return [
                "Apollo 11",  # Specific
                "Apollo program",  # Medium
                "Space exploration"  # Broad
            ]
        else:
            return [
                query,  # Original
                f"{query} category",  # Medium
                f"{query.split()[0]} general"  # Broad
            ]
    
    def _get_related_topics(self, query: str) -> List[str]:
        """Generate related topic searches."""
        query_lower = query.lower()
        
        if 'beatles' in query_lower:
            return [
                "John Lennon biography",
                "Paul McCartney career",
                "Beatles discography complete",
                "British Invasion music",
                "Abbey Road Studios"
            ]
        elif 'apollo 11' in query_lower:
            return [
                "Neil Armstrong biography",
                "Buzz Aldrin astronaut",
                "Moon landing conspiracy theories",
                "Space Race timeline",
                "NASA Apollo missions"
            ]
        elif 'quantum' in query_lower:
            return [
                "Albert Einstein quantum theory",
                "Schrödinger equation",
                "Quantum entanglement",
                "Heisenberg uncertainty principle",
                "Quantum computing"
            ]
        else:
            # Generic related topics
            words = query.split()
            if len(words) > 1:
                return [f"{word} {words[-1]}" for word in words[:-1]]
            else:
                return [f"{query} related", f"{query} associated"]
    
    def _get_category_queries(self, query: str) -> List[str]:
        """Generate category-based searches."""
        query_lower = query.lower()
        
        if 'beatles' in query_lower:
            return [
                "English rock bands",
                "Grammy Award winners music",
                "Rock and Roll Hall of Fame inductees"
            ]
        elif 'apollo' in query_lower or 'space' in query_lower:
            return [
                "NASA missions",
                "Human spaceflight programs",
                "Moon exploration"
            ]
        elif 'quantum' in query_lower:
            return [
                "Physics theories",
                "Quantum mechanics concepts",
                "Modern physics"
            ]
        else:
            return [f"{query} category", f"{query} type"]
    
    def _get_temporal_queries(self, query: str) -> List[str]:
        """Generate time-based related searches."""
        query_lower = query.lower()
        
        if 'beatles' in query_lower:
            return [
                "Music before Beatles 1950s",
                "British music after Beatles 1970s"
            ]
        elif 'apollo 11' in query_lower:
            return [
                "Space exploration before 1969",
                "NASA missions after Apollo"
            ]
        elif '1969' in query_lower:
            return [
                "1960s historical events",
                "1970s aftermath"
            ]
        else:
            return []
    
    def _add_unique_articles(self, all_articles: List[Dict], new_articles: List[Dict], seen_titles: Set[str]):
        """Add articles to collection, avoiding duplicates."""
        for article in new_articles:
            title = article.get('title', '').lower()
            if title and title not in seen_titles:
                all_articles.append(article)
                seen_titles.add(title)
                logger.info(f"  ➕ Added unique article: '{article['title']}'")
            else:
                logger.info(f"  ➖ Skipped duplicate: '{article.get('title', 'Unknown')}'")
    
    def _rank_articles_by_relevance(self, articles: List[Dict], original_query: str) -> List[Dict]:
        """Rank articles by relevance to original query."""
        
        def calculate_relevance_score(article: Dict) -> float:
            score = 0.0
            title = article.get('title', '').lower()
            query_words = set(original_query.lower().split())
            title_words = set(title.split())
            
            # Word overlap scoring
            overlap = len(query_words.intersection(title_words))
            score += overlap * 2.0
            
            # Bonus for search method
            method_bonuses = {
                'semantic_expansion': 1.0,
                'hierarchical': 0.8,
                'related_topics': 0.6,
                'category_based': 0.4,
                'temporal': 0.2
            }
            score += method_bonuses.get(article.get('search_method', ''), 0.0)
            
            # Penalty for disambiguation pages and lists
            if 'disambiguation' in title:
                score -= 1.0
            if 'list of' in title:
                score -= 0.5
            
            return score
        
        # Sort by relevance score
        ranked = sorted(articles, key=calculate_relevance_score, reverse=True)
        
        logger.info("📊 Article relevance ranking:")
        for i, article in enumerate(ranked[:10]):  # Show top 10
            score = calculate_relevance_score(article)
            logger.info(f"  {i+1}. '{article['title']}' (score: {score:.2f}, method: {article.get('search_method', 'unknown')})")
        
        return ranked

# Example usage and demo
def demo_beatles_search():
    """Demonstrate how advanced search finds multiple Beatles articles."""
    searcher = AdvancedSearchStrategies()
    
    articles = searcher.find_multiple_articles("Who were the Beatles?", max_articles=8)
    
    print("\n🎵 BEATLES MULTI-SOURCE SEARCH RESULTS:")
    print("=" * 60)
    
    for i, article in enumerate(articles):
        print(f"{i+1}. {article['title']}")
        print(f"   Method: {article.get('search_method', 'unknown')}")
        print(f"   URL: {article.get('url', 'N/A')[:60]}...")
        print()
    
    # This would find articles like:
    # 1. The Beatles (main article)
    # 2. John Lennon (member biography)  
    # 3. Beatles discography (music catalog)
    # 4. British Invasion (cultural context)
    # 5. Abbey Road Studios (recording location)
    # 6. Beatlemania (cultural phenomenon)
    # 7. Rock and roll history (genre context)
    # 8. 1960s popular music (time period)

if __name__ == "__main__":
    demo_beatles_search() 