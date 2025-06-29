"""
Multi-Source Agent for SummarEaseAI
Gathers information from multiple Wikipedia articles and synthesizes them.
"""

import logging
from typing import List, Dict, Optional, Any
from utils.wikipedia_fetcher import search_and_fetch_article_info, enhance_query_with_intent
from backend.summarizer import summarize_article_with_intent
from tensorflow_models.bert_gpu_classifier import get_gpu_classifier
from utils.openai_query_generator import OpenAIQueryGenerator
from utils.langchain_agents import WikipediaAgentSystem
import wikipedia

logger = logging.getLogger(__name__)

# RATE LIMITING CONFIGURATION
class RateLimitConfig:
    """Configuration for controlling OpenAI API usage and costs."""
    
    # Core limits
    MAX_ARTICLES_PER_SUMMARY = 3  # Default: conservative limit
    MAX_SECONDARY_QUERIES = 4     # Limit OpenAI query generation calls
    MAX_WIKIPEDIA_SEARCHES = 8    # Total Wikipedia API calls
    
    # Feature toggles
    ENABLE_OPENAI_QUERY_GENERATION = True  # Set to False to disable OpenAI query gen
    ENABLE_LANGCHAIN_AGENTS = True         # Set to False to disable LangChain agents
    ENABLE_MULTI_SOURCE = True             # Set to False for single-article fallback
    
    # Cost control modes
    COST_MODE = "BALANCED"  # Options: "MINIMAL", "BALANCED", "COMPREHENSIVE"
    
    @classmethod
    def get_limits_for_mode(cls, mode: str) -> Dict[str, int]:
        """Get rate limits based on cost control mode."""
        limits = {
            "MINIMAL": {
                "max_articles": 1,
                "max_secondary_queries": 0,  # No secondary queries
                "max_wikipedia_searches": 1,
                "enable_openai": False,
                "enable_agents": False
            },
            "BALANCED": {
                "max_articles": 3,
                "max_secondary_queries": 3,
                "max_wikipedia_searches": 6,
                "enable_openai": True,
                "enable_agents": True
            },
            "COMPREHENSIVE": {
                "max_articles": 6,
                "max_secondary_queries": 6,
                "max_wikipedia_searches": 12,
                "enable_openai": True,
                "enable_agents": True
            }
        }
        return limits.get(mode, limits["BALANCED"])

class MultiSourceAgent:
    """
    Enhanced Multi-Source Agent with comprehensive logging and rate limiting.
    
    Features:
    - OpenAI query generation logging
    - Wikipedia search tracking
    - Article selection reasoning
    - Smart rate limiting for cost control
    """
    
    def __init__(self, cost_mode: str = "BALANCED"):
        self.bert_classifier = get_gpu_classifier()
        self.bert_model_loaded = self.bert_classifier.load_model()
        self.query_generator = OpenAIQueryGenerator()
        self.agent_system = WikipediaAgentSystem()  # Real LangChain agents
        
        # Apply rate limiting configuration
        self.limits = RateLimitConfig.get_limits_for_mode(cost_mode)
        self.cost_mode = cost_mode
        
        # Tracking for logging
        self.openai_calls_made = 0
        self.wikipedia_calls_made = 0
        self.articles_processed = 0
        
        logger.info(f"🎛️ Multi-Source Agent initialized in {cost_mode} mode")
        logger.info(f"📊 Limits: {self.limits['max_articles']} articles, {self.limits['max_secondary_queries']} secondary queries")
        logger.info(f"🚀 GPU BERT model loaded: {self.bert_model_loaded}")
        
    def plan_search_strategy(self, query: str, intent: str, confidence: float) -> Dict[str, Any]:
        """
        Plan what articles to search for using OpenAI-generated secondary queries.
        Includes comprehensive logging and rate limiting.
        """
        logger.info(f"🧠 Planning OpenAI-powered search strategy for '{query}' ({intent})")
        logger.info(f"💰 Rate limiting: {self.cost_mode} mode, {self.openai_calls_made} OpenAI calls made")
        
        # Check if we should use OpenAI for secondary query generation
        openai_used = False
        search_plan = None
        
        if (self.limits.get('enable_openai', True) and 
            self.limits.get('max_secondary_queries', 0) > 0 and
            self.openai_calls_made < self.limits.get('max_secondary_queries', 4)):
            
            try:
                logger.info(f"📡 REQUESTING OpenAI secondary query generation...")
                logger.info(f"   📊 Call limit: {self.openai_calls_made + 1}/{self.limits.get('max_secondary_queries', 4)}")
                
                # Use OpenAI to generate intelligent secondary queries
                search_plan = self.query_generator.generate_comprehensive_search_plan(query, intent)
                self.openai_calls_made += 1
                openai_used = True
                
                logger.info(f"✅ OpenAI RESPONSE RECEIVED:")
                logger.info(f"   🎯 Synthesis focus: {search_plan['synthesis_focus']}")
                logger.info(f"   📋 Categories: {list(search_plan['secondary_queries'].keys())}")
                
                # Log each category and its queries
                for category, queries in search_plan['secondary_queries'].items():
                    logger.info(f"   📂 {category.upper()}:")
                    for i, sq in enumerate(queries, 1):
                        logger.info(f"      {i}. '{sq}'")
                
            except Exception as e:
                logger.error(f"❌ OpenAI query generation FAILED: {str(e)}")
                search_plan = None
        else:
            # Rate limiting engaged or OpenAI disabled
            if not self.limits.get('enable_openai', True):
                logger.info(f"🚫 OpenAI query generation DISABLED in {self.cost_mode} mode")
            else:
                logger.info(f"⏸️ OpenAI rate limit REACHED ({self.openai_calls_made}/{self.limits.get('max_secondary_queries', 4)})")
        
        # Fallback if OpenAI failed or disabled
        if not search_plan:
            logger.info(f"🔄 Using FALLBACK query generation strategy")
            search_plan = self._generate_fallback_search_plan(query, intent)
        
        # Convert OpenAI search plan to our strategy format
        all_secondary_queries = []
        for category, queries in search_plan['secondary_queries'].items():
            all_secondary_queries.extend(queries)
        
        # Apply rate limiting
        max_articles = min(self.limits.get('max_articles', 3), len(all_secondary_queries) + 1)
        max_secondary = self.limits.get('max_secondary_queries', 4)
        
        if len(all_secondary_queries) > max_secondary:
            logger.info(f"✂️ TRUNCATING secondary queries from {len(all_secondary_queries)} to {max_secondary}")
            all_secondary_queries = all_secondary_queries[:max_secondary]
        
        strategy = {
            'primary_queries': [query],
            'secondary_queries': all_secondary_queries,
            'max_articles': max_articles,
            'synthesis_focus': search_plan['synthesis_focus'],
            'search_strategy': search_plan['search_strategy'],
            'openai_categories': search_plan['secondary_queries'],  # Keep categorized queries
            'openai_used': openai_used,
            'cost_mode': self.cost_mode,
            'rate_limits': self.limits
        }
        
        total_queries = len(strategy['primary_queries'] + strategy['secondary_queries'])
        logger.info(f"📊 FINAL STRATEGY:")
        logger.info(f"   🔍 Total queries: {total_queries} (1 primary + {len(all_secondary_queries)} secondary)")
        logger.info(f"   📚 Max articles: {max_articles}")
        logger.info(f"   🎯 Synthesis focus: {strategy['synthesis_focus']}")
        logger.info(f"   💰 OpenAI used: {openai_used}")
        
        return strategy
    
    def _generate_fallback_search_plan(self, query: str, intent: str) -> Dict[str, Any]:
        """Generate fallback search plan when OpenAI is unavailable."""
        logger.info(f"🔄 Generating fallback search plan for '{query}' ({intent})")
        
        # Simple rule-based secondary queries
        secondary_queries = {}
        
        if intent.lower() in ['biography', 'person']:
            secondary_queries['contextual'] = [f"{query} biography", f"{query} early life"]
            secondary_queries['related'] = [f"{query} achievements", f"{query} legacy"]
        elif intent.lower() in ['history', 'historical']:
            secondary_queries['contextual'] = [f"{query} timeline", f"{query} causes"]
            secondary_queries['related'] = [f"{query} aftermath", f"{query} significance"]
        elif intent.lower() in ['science', 'technology']:
            secondary_queries['contextual'] = [f"{query} explanation", f"{query} principles"]
            secondary_queries['related'] = [f"{query} applications", f"{query} research"]
        else:
            # General fallback
            secondary_queries['contextual'] = [f"{query} overview", f"{query} history"]
            secondary_queries['related'] = [f"{query} significance", f"{query} impact"]
        
        return {
            'secondary_queries': secondary_queries,
            'synthesis_focus': 'comprehensive',
            'search_strategy': 'fallback_rule_based'
        }
    
    def gather_articles_with_agents(self, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Gather articles using real LangChain agents for enhanced search and selection.
        Includes comprehensive Wikipedia search logging and OpenAI article selection.
        """
        articles = []
        seen_titles = set()
        all_queries = strategy['primary_queries'] + strategy['secondary_queries']
        
        # Initialize tracking
        wikipedia_searches_made = 0
        max_wikipedia_searches = self.limits.get('max_wikipedia_searches', 8)
        
        logger.info(f"🤖 STARTING agent-powered article gathering")
        logger.info(f"   📋 Queries to process: {len(all_queries)}")
        logger.info(f"   📚 Max articles: {strategy.get('max_articles', 3)}")
        logger.info(f"   🔍 Wikipedia search limit: {max_wikipedia_searches}")
        
        for i, search_query in enumerate(all_queries):
            # Check limits
            if len(articles) >= strategy.get('max_articles', 3):
                logger.info(f"⏹️ Article limit reached ({len(articles)}/{strategy.get('max_articles', 3)})")
                break
                
            if wikipedia_searches_made >= max_wikipedia_searches:
                logger.info(f"⏹️ Wikipedia search limit reached ({wikipedia_searches_made}/{max_wikipedia_searches})")
                break
                
            is_primary = i < len(strategy['primary_queries'])
            query_type = "PRIMARY" if is_primary else "SECONDARY"
            
            logger.info(f"🔍 [{query_type}] Wikipedia search {i+1}/{len(all_queries)}: '{search_query}'")
            
            # Track Wikipedia API usage
            wikipedia_searches_made += 1
            self.wikipedia_calls_made += 1
            
            # Use LangChain agents for intelligent search
            if self.limits.get('enable_agents', True):
                logger.info(f"   🤖 Using LangChain agents for intelligent search...")
                agent_result = self.agent_system.intelligent_wikipedia_search(search_query, max_options=3)
                
                if 'error' not in agent_result and agent_result.get('article_info'):
                    article_info = agent_result['article_info']
                    title = article_info.get('title', '')
                    
                    logger.info(f"   📡 Agent enhancement: '{search_query}' → '{agent_result['enhancement_result']['enhanced_query']}'")
                    logger.info(f"   🎯 Agent selected from {len(agent_result.get('search_results', []))} options: '{title}'")
                    
                    # Check for duplicates
                    if title and title.lower() not in seen_titles:
                        # Add comprehensive metadata
                        article_info['search_query'] = search_query
                        article_info['is_primary'] = str(is_primary)
                        article_info['query_type'] = query_type
                        article_info['agent_enhanced_query'] = agent_result['enhancement_result']['enhanced_query']
                        article_info['agent_reasoning'] = str({
                            'query_enhancement': agent_result['enhancement_result'].get('agent_reasoning', ''),
                            'article_selection': agent_result['selection_result'].get('agent_reasoning', ''),
                            'search_options': agent_result.get('search_results', [])
                        })
                        article_info['relevance_score'] = str(self._calculate_relevance(
                            search_query, title, strategy['primary_queries'][0]
                        ))
                        article_info['selection_method'] = 'langchain_agent'
                        
                        articles.append(article_info)
                        seen_titles.add(title.lower())
                        self.articles_processed += 1
                        
                        logger.info(f"   ✅ SELECTED: '{title}' (relevance: {article_info['relevance_score']})")
                        logger.info(f"   📊 Available options were: {agent_result.get('search_results', [])}")
                    else:
                        logger.info(f"   ⚠️ DUPLICATE skipped: '{title}'")
                else:
                    logger.warning(f"   ❌ Agent search FAILED: {agent_result.get('error', 'Unknown error')}")
                    # Fallback handled below
            else:
                logger.info(f"   🚫 LangChain agents disabled in {self.cost_mode} mode")
                agent_result = None
            
            # Fallback to basic search if agents failed or disabled
            if not agent_result or 'error' in agent_result:
                logger.info(f"   🔄 Using FALLBACK Wikipedia search...")
                fallback_article = search_and_fetch_article_info(search_query)
                
                if fallback_article and fallback_article.get('title'):
                    title = fallback_article['title']
                    logger.info(f"   📚 Wikipedia returned: '{title}'")
                    
                    if title.lower() not in seen_titles:
                        fallback_article['search_query'] = search_query
                        fallback_article['is_primary'] = str(is_primary)
                        fallback_article['query_type'] = query_type
                        fallback_article['agent_enhanced_query'] = search_query  # No enhancement
                        fallback_article['agent_reasoning'] = str({'fallback': 'Used basic search due to agent failure'})
                        fallback_article['relevance_score'] = str(self._calculate_relevance(
                            search_query, title, strategy['primary_queries'][0]
                        ))
                        fallback_article['selection_method'] = 'fallback_basic'
                        
                        articles.append(fallback_article)
                        seen_titles.add(title.lower())
                        self.articles_processed += 1
                        
                        logger.info(f"   ✅ FALLBACK selected: '{title}'")
                    else:
                        logger.info(f"   ⚠️ DUPLICATE (fallback): '{title}'")
                else:
                    logger.warning(f"   ❌ NO article found for: '{search_query}'")
        
        # Sort by relevance score
        articles.sort(key=lambda x: float(x['relevance_score']), reverse=True)
        
        # Final logging summary
        logger.info(f"📊 ARTICLE GATHERING COMPLETED:")
        logger.info(f"   📚 Articles collected: {len(articles)}")
        logger.info(f"   🔍 Wikipedia searches made: {wikipedia_searches_made}")
        logger.info(f"   🤖 Articles processed: {self.articles_processed}")
        logger.info(f"   📋 Selected articles:")
        
        for i, article in enumerate(articles, 1):
            logger.info(f"      {i}. '{article['title']}' (query: '{article['search_query']}', relevance: {article['relevance_score']})")
        
        return articles

    def gather_articles(self, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Gather articles based on the search strategy.
        
        Returns list of article information with metadata.
        """
        articles = []
        all_queries = strategy['primary_queries'] + strategy['secondary_queries']
        
        for i, search_query in enumerate(all_queries):
            if len(articles) >= strategy['max_articles']:
                break
                
            logger.info(f"🔍 Gathering article {i+1}/{len(all_queries)}: '{search_query}'")
            
            article_info = search_and_fetch_article_info(search_query)
            if article_info:
                # Add metadata about this article's role
                article_info['search_query'] = search_query
                article_info['is_primary'] = str(i < len(strategy['primary_queries']))
                article_info['relevance_score'] = str(self._calculate_relevance(
                    search_query, 
                    article_info['title'],
                    strategy['primary_queries'][0]  # Original query
                ))
                articles.append(article_info)
                logger.info(f"✅ Found: '{article_info['title']}'")
            else:
                logger.warning(f"❌ No article found for: '{search_query}'")
        
        # Sort by relevance score (convert back to float for sorting)
        articles.sort(key=lambda x: float(x['relevance_score']), reverse=True)
        
        logger.info(f"📚 Gathered {len(articles)} articles for synthesis")
        return articles
    
    def _calculate_relevance(self, search_query: str, article_title: str, original_query: str) -> float:
        """Calculate relevance score for article selection."""
        score = 0.0
        
        # Exact title match
        if search_query.lower() in article_title.lower():
            score += 1.0
        
        # Original query relevance
        original_words = set(original_query.lower().split())
        title_words = set(article_title.lower().split())
        word_overlap = len(original_words.intersection(title_words))
        score += word_overlap * 0.3
        
        # Prefer main articles over lists/disambiguation
        if 'list of' in article_title.lower():
            score -= 0.2
        if '(' in article_title and ')' in article_title:
            score -= 0.1
            
        return score
    
    def synthesize_articles(self, articles: List[Dict[str, Any]], strategy: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        """
        Synthesize multiple articles into a comprehensive summary.
        
        This is where the agent intelligence really shines.
        """
        if not articles:
            return {'error': 'No articles to synthesize'}
        
        # If only one article, use standard summarization
        if len(articles) == 1:
            article = articles[0]
            summary = summarize_article_with_intent(
                article['content'], 
                article['title'],
                strategy.get('synthesis_focus', 'general'),
                1.0  # High confidence for single article
            )
            summary['synthesis_method'] = 'single_article'
            summary['articles_used'] = [article['title']]
            return summary
        
        # Multi-article synthesis
        logger.info(f"🔄 Synthesizing {len(articles)} articles with focus: {strategy['synthesis_focus']}")
        
        # Create synthesis prompt based on focus
        synthesis_prompts = {
            'chronological': self._create_chronological_synthesis,
            'causal': self._create_causal_synthesis,
            'conceptual': self._create_conceptual_synthesis,
            'evolutionary': self._create_evolutionary_synthesis,
            'comprehensive': self._create_comprehensive_synthesis
        }
        
        synthesis_method = synthesis_prompts.get(
            strategy['synthesis_focus'], 
            self._create_general_synthesis
        )
        
        # Generate synthesis
        result = synthesis_method(articles, original_query)
        synthesis_focus = str(strategy.get('synthesis_focus', 'general')).replace('{', '').replace('}', '')
        result['synthesis_method'] = f"multi_source_{synthesis_focus}"
        result['articles_used'] = [article['title'] for article in articles]
        result['article_count'] = len(articles)
        
        return result
    
    def _create_comprehensive_synthesis(self, articles: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Create comprehensive synthesis (good for complex topics like The Beatles)."""
        
        # Combine article contents with structure
        safe_query = str(query).replace('{', '').replace('}', '')
        combined_content = f"# Comprehensive Information about: {safe_query}\n\n"
        
        for i, article in enumerate(articles):
            safe_title = str(article.get('title', 'Unknown')).replace('{', '').replace('}', '')
            combined_content += f"## Source {i+1}: {safe_title}\n"
            combined_content += f"Relevance: {article['relevance_score']:.2f}\n"
            combined_content += f"Content: {article['content'][:2000]}...\n\n"  # Limit to avoid token limits
        
        # Use enhanced summarization with multi-source awareness
        safe_query_title = str(query).replace('{', '').replace('}', '')
        summary = summarize_article_with_intent(
            combined_content,
            f"Comprehensive overview of {safe_query_title}",
            "comprehensive",
            0.9
        )
        
        return summary
    
    def _create_chronological_synthesis(self, articles: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Synthesis focused on timeline (good for biographies/history)."""
        # Implementation for chronological synthesis
        return self._create_comprehensive_synthesis(articles, query)  # Simplified for now
    
    def _create_causal_synthesis(self, articles: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Synthesis focused on cause and effect (good for historical events)."""
        # Implementation for causal synthesis
        return self._create_comprehensive_synthesis(articles, query)  # Simplified for now
    
    def _create_conceptual_synthesis(self, articles: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Synthesis focused on concepts and relationships (good for science)."""
        # Implementation for conceptual synthesis
        return self._create_comprehensive_synthesis(articles, query)  # Simplified for now
    
    def _create_evolutionary_synthesis(self, articles: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Synthesis focused on development over time (good for technology)."""
        # Implementation for evolutionary synthesis
        return self._create_comprehensive_synthesis(articles, query)  # Simplified for now
    
    def _create_general_synthesis(self, articles: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """General synthesis method."""
        return self._create_comprehensive_synthesis(articles, query)
    
    def _create_final_synthesis(self, summaries: List[Dict[str, Any]], query: str, intent: str) -> Optional[str]:
        """
        Create a final synthesized summary from all individual article summaries.
        Uses OpenAI to combine multiple summaries into one coherent response under 30 lines.
        """
        try:
            from backend.summarizer import get_openai_api_key, LANGCHAIN_AVAILABLE
            from langchain_openai import ChatOpenAI
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            from pydantic import SecretStr
            
            if not LANGCHAIN_AVAILABLE:
                logger.warning("⚠️ LangChain not available for final synthesis")
                return None
                
            api_key = get_openai_api_key()
            if not api_key:
                logger.warning("⚠️ OpenAI API key not available for final synthesis")
                return None
            
            logger.info(f"🎯 Creating final synthesis from {len(summaries)} articles")
            
            # Prepare the content for synthesis
            articles_content = []
            for i, summary in enumerate(summaries, 1):
                title = summary.get('title', 'Unknown')
                content = summary.get('summary', 'No summary available')
                articles_content.append(f"Article {i}: {title}\n{content}\n")
            
            combined_content = "\n".join(articles_content)
            
            # Create intent-specific synthesis prompt
            intent_specific_instructions = {
                'Music': "Focus on the band's formation, key members, musical style, major albums, cultural impact, and legacy in popular music.",
                'Biography': "Focus on the person's life story, major achievements, contributions, and historical significance.",
                'History': "Focus on causes, key events, major figures, consequences, and historical significance.",
                'Science': "Focus on scientific principles, discoveries, applications, and impact on the field.",
                'Technology': "Focus on how the technology works, development, applications, and impact on society.",
                'General': "Focus on the main topics, key information, and overall significance."
            }
            
            instruction = intent_specific_instructions.get(intent, intent_specific_instructions['General'])
            
            prompt_template = f"""You are tasked with creating a comprehensive final summary by synthesizing information from multiple Wikipedia articles.

User's Question: "{query}"

Your task: Create ONE unified summary that answers the user's question comprehensively by combining insights from all the provided articles.

Requirements:
- Maximum 30 lines
- Well-structured and easy to read
- {instruction}
- Synthesize information rather than just listing facts
- Provide a coherent narrative that flows logically
- Include the most important and relevant information from all sources

Articles to synthesize:
{{combined_content}}

Final Comprehensive Summary:
"""
            
            # Create and run the synthesis chain
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,
                max_completion_tokens=1500,
                api_key=SecretStr(api_key)
            )
            
            prompt = PromptTemplate(
                input_variables=["combined_content"],
                template=prompt_template
            )
            
            chain = LLMChain(llm=llm, prompt=prompt)
            
            logger.info(f"🤖 Requesting final synthesis from OpenAI...")
            logger.info(f"   📊 Synthesizing {len(summaries)} articles")
            logger.info(f"   🎯 Intent: {intent}")
            
            final_summary = chain.run(combined_content=combined_content)
            
            logger.info(f"✅ Final synthesis completed successfully")
            logger.info(f"   📝 Generated {len(final_summary.split())} words")
            
            return final_summary.strip()
            
        except Exception as e:
            logger.error(f"❌ Failed to create final synthesis: {str(e)}")
            return None
    
    def run_multi_source_search_with_agents(self, query: str, max_articles: Optional[int] = None) -> Dict[str, Any]:
        """
        Main entry point using LangChain agents for enhanced multi-source search.
        
        Features smart cost control and can fall back to single-article mode.
        
        Args:
            query: User's search query
            max_articles: Override for max articles (None = use mode default)
        """
        # Apply cost control - determine if we should use multi-source
        effective_max_articles = max_articles or self.limits.get('max_articles', 3)
        
        # COST CONTROL: Check if we should fall back to single article
        if effective_max_articles == 1 or not self.limits.get('enable_openai', True):
            logger.info(f"💰 SINGLE-ARTICLE MODE activated (cost control: {self.cost_mode})")
            return self._run_single_article_mode(query)
        
        logger.info(f"🤖 Starting AGENT-POWERED multi-source search for: '{query}'")
        logger.info(f"💰 Cost mode: {self.cost_mode}, Max articles: {effective_max_articles}")
        
        # Reset tracking for this search
        search_start_calls = {
            'openai': self.openai_calls_made,
            'wikipedia': self.wikipedia_calls_made,
            'articles': self.articles_processed
        }
        
        try:
            # Step 1: Intent Detection
            intent_result = self._analyze_intent(query)
            intent = intent_result['intent']
            confidence = intent_result['confidence']
            
            logger.info(f"🧠 Intent analysis: {intent} (confidence: {confidence:.2f})")
            
            # Step 2: Plan search strategy with OpenAI query generation
            strategy = self.plan_search_strategy(query, intent, confidence)
            strategy['max_articles'] = effective_max_articles
            
            # Log OpenAI query generation results
            if strategy.get('openai_used', False):
                logger.info(f"📡 OpenAI secondary queries GENERATED:")
                for category, queries in strategy.get('openai_categories', {}).items():
                    logger.info(f"   📂 {category.upper()}: {queries}")
            
            logger.info(f"📋 Strategy planned: {len(strategy['primary_queries'])} primary + {len(strategy['secondary_queries'])} secondary queries")
            
            # Step 3: Gather articles using LangChain agents
            articles = self.gather_articles_with_agents(strategy)
            
            if not articles:
                logger.warning("⚠️ No articles found with agents - trying fallback")
                articles = self.gather_articles(strategy)
            
            if not articles:
                return {'error': 'No articles found for multi-source synthesis'}
            
            # Step 4: Rank articles by relevance
            ranked_articles = self.rank_articles(articles, query, intent)[:effective_max_articles]
            
            # Step 5: Summarize articles with comprehensive metadata
            summaries = []
            for article in ranked_articles:
                try:
                    summary_result = summarize_article_with_intent(
                        article['content'][:8000],  # Limit content for processing
                        article['title'],
                        intent,
                        confidence
                    )
                    
                    # Debug logging to see what we're getting
                    logger.info(f"📄 Summarizing '{article['title']}':")
                    logger.info(f"   🔍 Summary result type: {type(summary_result)}")
                    logger.info(f"   📝 Summary result keys: {summary_result.keys() if isinstance(summary_result, dict) else 'Not a dict'}")
                    if isinstance(summary_result, dict) and 'summary' in summary_result:
                        summary_text = summary_result['summary']
                        logger.info(f"   ✅ Summary extracted: {len(summary_text)} chars")
                    else:
                        summary_text = str(summary_result)[:100] + "..." if len(str(summary_result)) > 100 else str(summary_result)
                        logger.info(f"   ⚠️ Using fallback summary: {summary_text}")
                    
                    summaries.append({
                        'title': article['title'],
                        'url': article['url'],
                        'summary': summary_result.get('summary', 'Summary not available') if isinstance(summary_result, dict) else str(summary_result),
                        'relevance_score': article['relevance_score'],
                        'search_query': article['search_query'],
                        'is_primary': article['is_primary'],
                        'query_type': article.get('query_type', 'UNKNOWN'),
                        'agent_enhanced_query': article.get('agent_enhanced_query', ''),
                        'agent_reasoning': article.get('agent_reasoning', ''),
                        'selection_method': article.get('selection_method', 'unknown')
                    })
                except Exception as e:
                    logger.error(f"❌ Failed to summarize {article['title']}: {str(e)}")
                    continue
            
            # Step 6: Generate final synthesized summary from all articles
            final_summary = None
            if len(summaries) > 1 and self.limits.get('enable_openai', True):
                final_summary = self._create_final_synthesis(summaries, query, intent)
                if final_summary:
                    # Count the additional OpenAI call
                    self.openai_calls_made += 1
            
            # Calculate costs for this search
            search_costs = {
                'openai_calls': self.openai_calls_made - search_start_calls['openai'],
                'wikipedia_calls': self.wikipedia_calls_made - search_start_calls['wikipedia'],
                'articles_processed': self.articles_processed - search_start_calls['articles']
            }
            
            # Compile comprehensive result with cost tracking
            result = {
                'query': query,
                'intent': intent,
                'confidence': confidence,
                'strategy': strategy,
                'total_articles_found': len(articles),
                'articles_summarized': len(summaries),
                'summaries': summaries,
                'final_synthesis': final_summary,
                
                # Agent and cost metadata
                'agent_powered': True,
                'search_method': 'multi_source_with_langchain_agents',
                'agents_used': ['QueryEnhancementAgent', 'ArticleSelectionAgent'],
                'cost_mode': self.cost_mode,
                'cost_tracking': search_costs,
                'rate_limits_applied': self.limits,
                
                # Detailed logging for debugging
                'wikipedia_pages_used': [s['title'] for s in summaries],
                'openai_secondary_queries': strategy.get('secondary_queries', []),
                'openai_query_categories': strategy.get('openai_categories', {}),
                'wikipedia_searches_made': [s['search_query'] for s in summaries]
            }
            
            # Final cost summary
            logger.info(f"💰 COST SUMMARY for this search:")
            logger.info(f"   📡 OpenAI calls: {search_costs['openai_calls']}")
            logger.info(f"   📚 Wikipedia calls: {search_costs['wikipedia_calls']}")
            logger.info(f"   📄 Articles processed: {search_costs['articles_processed']}")
            logger.info(f"   📋 Final Wikipedia pages used: {result['wikipedia_pages_used']}")
            
            logger.info(f"✅ Agent-powered multi-source search completed: {len(summaries)} articles")
            return result
            
        except Exception as e:
            logger.error(f"❌ Agent-powered multi-source search error: {str(e)}")
            return {'error': f'Agent-powered search failed: {str(e)}'}
    
    def _run_single_article_mode(self, query: str) -> Dict[str, Any]:
        """
        Fallback to single-article mode for cost control.
        
        This is the default behavior when max_articles == 1.
        """
        logger.info(f"📄 Running SINGLE-ARTICLE mode for: '{query}'")
        
        try:
            # Simple single article search
            from utils.wikipedia_fetcher import search_and_fetch_article_info
            article_info = search_and_fetch_article_info(query)
            
            if not article_info:
                return {'error': 'No article found'}
            
            # Simple intent detection
            intent_result = self._analyze_intent(query)
            intent = intent_result['intent']
            confidence = intent_result['confidence']
            
            # Single article summary
            summary_result = summarize_article_with_intent(
                article_info['content'][:8000],
                article_info['title'],
                intent,
                confidence
            )
            
            result = {
                'query': query,
                'intent': intent,
                'confidence': confidence,
                'total_articles_found': 1,
                'articles_summarized': 1,
                'summaries': [{
                    'title': article_info['title'],
                    'url': article_info['url'],
                    'summary': summary_result.get('summary', 'Summary not available'),
                    'relevance_score': '1.0',
                    'search_query': query,
                    'is_primary': 'True',
                    'query_type': 'PRIMARY',
                    'selection_method': 'single_article_default'
                }],
                
                # Metadata
                'agent_powered': False,
                'search_method': 'single_article_cost_control',
                'cost_mode': self.cost_mode,
                'cost_tracking': {'openai_calls': 0, 'wikipedia_calls': 1, 'articles_processed': 1},
                'wikipedia_pages_used': [article_info['title']],
                'wikipedia_searches_made': [query]
            }
            
            logger.info(f"✅ Single-article mode completed: '{article_info['title']}'")
            return result
            
        except Exception as e:
            logger.error(f"❌ Single-article mode error: {str(e)}")
            return {'error': f'Single-article search failed: {str(e)}'}

    def rank_articles(self, articles: List[Dict[str, Any]], query: str, intent: str) -> List[Dict[str, Any]]:
        """Rank articles by relevance score and other factors."""
        # Sort by relevance score (convert back to float for sorting)
        return sorted(articles, key=lambda x: float(x.get('relevance_score', 0)), reverse=True)

    def _analyze_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent using GPU BERT classifier"""
        # Try GPU BERT model first
        if self.bert_classifier and self.bert_model_loaded:
            try:
                intent, confidence = self.bert_classifier.predict(query)
                return {
                    'intent': intent,
                    'confidence': confidence,
                    'method': 'gpu_bert'
                }
            except Exception as e:
                logger.error(f"GPU BERT prediction failed: {str(e)}")
        
        # Fallback to keyword-based
        return {
            'intent': 'General',
            'confidence': 0.5,
            'method': 'fallback'
        }

    def run_multi_source_search(self, query: str) -> Dict[str, Any]:
        """
        Main entry point for multi-source agent workflow.
        
        This is what would be called from the API endpoint.
        """
        logger.info(f"🤖 Starting multi-source agent for: '{query}'")
        
        try:
            # Step 1: Intent Detection
            intent_result = self._analyze_intent(query)
            intent = intent_result['intent']
            confidence = intent_result['confidence']
            
            logger.info(f"🎯 Detected intent: {intent} (confidence: {confidence:.3f})")
            
            # Step 2: Plan Search Strategy
            strategy = self.plan_search_strategy(query, intent, confidence)
            
            # Step 3: Gather Articles
            articles = self.gather_articles(strategy)
            
            if not articles:
                return {'error': 'No articles found for multi-source synthesis'}
            
            # Step 4: Synthesize Results
            result = self.synthesize_articles(articles, strategy, query)
            
            # Add agent metadata
            result['agent_type'] = 'multi_source'
            result['original_query'] = query
            result['detected_intent'] = intent
            result['intent_confidence'] = confidence
            result['search_strategy'] = strategy['synthesis_focus']
            
            logger.info(f"✅ Multi-source agent completed for: '{query}'")
            return result
            
        except Exception as e:
            logger.error(f"❌ Multi-source agent error: {str(e)}")
            return {'error': f'Multi-source agent failed: {str(e)}'}

# Usage Example:
def example_usage():
    """Example of how the multi-source agent would work."""
    agent = MultiSourceAgent()
    
    # Complex query that benefits from multiple sources
    result = agent.run_multi_source_search("Who were the Beatles?")
    
    # This would gather:
    # 1. "The Beatles" (main band article)
    # 2. "The Beatles discography" (their music)
    # 3. "The Beatles members" (individual members)
    # 4. "The Beatles cultural impact" (influence)
    
    # Then synthesize into a comprehensive summary covering:
    # - Band formation and history
    # - Key albums and songs
    # - Individual member contributions
    # - Cultural and musical impact
    
    return result

if __name__ == "__main__":
    # Test the agent
    example = example_usage()
    print(example) 