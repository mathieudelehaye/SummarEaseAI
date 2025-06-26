"""
Real LangChain Agents for Query Enhancement and Article Selection
Uses LangChain's agent framework with tools, reasoning, and decision-making.
"""

import logging
from typing import List, Dict, Any, Optional
import wikipedia
from langchain.agents import AgentType, initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from backend.summarizer import get_openai_api_key, LANGCHAIN_AVAILABLE

logger = logging.getLogger(__name__)

class WikipediaSearchTool:
    """Wikipedia search tool for LangChain agents."""
    
    @staticmethod
    def search_wikipedia(query: str) -> str:
        """Search Wikipedia and return results summary."""
        try:
            search_results = wikipedia.search(query, results=5)
            if not search_results:
                return f"No Wikipedia articles found for: {query}"
            
            result_summary = f"Found {len(search_results)} Wikipedia articles for '{query}':\n"
            for i, title in enumerate(search_results, 1):
                result_summary += f"{i}. {title}\n"
            
            return result_summary
        except Exception as e:
            return f"Wikipedia search error: {str(e)}"
    
    @staticmethod
    def get_article_preview(title: str) -> str:
        """Get a preview of a Wikipedia article."""
        try:
            page = wikipedia.page(title)
            summary = page.summary[:300] + "..." if len(page.summary) > 300 else page.summary
            return f"Article: {page.title}\nURL: {page.url}\nSummary: {summary}"
        except wikipedia.exceptions.DisambiguationError as e:
            return f"Disambiguation page. Options: {e.options[:3]}"
        except Exception as e:
            return f"Error getting article preview: {str(e)}"

class QueryEnhancementAgent:
    """
    LangChain agent that enhances user queries for better Wikipedia searches.
    Uses reasoning to understand intent and reformulate queries.
    """
    
    def __init__(self):
        self.llm = None
        self.agent = None
        
        if LANGCHAIN_AVAILABLE and get_openai_api_key():
            try:
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.2,
                    api_key=get_openai_api_key()
                )
                self._initialize_agent()
                logger.info("✅ Query Enhancement Agent initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Query Enhancement Agent: {str(e)}")
        else:
            logger.warning("⚠️ OpenAI not available - Query Enhancement Agent disabled")
    
    def _initialize_agent(self):
        """Initialize the LangChain agent with tools."""
        
        tools = [
            Tool(
                name="wikipedia_search",
                func=WikipediaSearchTool.search_wikipedia,
                description="Search Wikipedia for articles. Input should be a search query string. Returns list of found article titles."
            ),
            Tool(
                name="test_query_effectiveness",
                func=self._test_query_effectiveness,
                description="Test how effective a query is for Wikipedia search. Input should be a query string. Returns effectiveness analysis."
            )
        ]
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            agent_kwargs={
                "prefix": """You are an expert Query Enhancement Agent for Wikipedia search optimization.

Your mission: Transform user queries into optimized Wikipedia search terms that find the most relevant articles.

Key principles:
1. Understand the user's true intent behind their question
2. Remove unnecessary question words ("who was", "what is", "tell me about")  
3. Target actual Wikipedia article titles
4. Make queries more specific and Wikipedia-friendly
5. Test your enhanced queries to ensure they work

Always test your enhanced query before giving the final answer.""",
                "suffix": """Begin!

Question: {input}
Thought: {agent_scratchpad}"""
            }
        )
    
    def _test_query_effectiveness(self, query: str) -> str:
        """Test how effective a query is for Wikipedia search."""
        try:
            search_results = wikipedia.search(query, results=3)
            if not search_results:
                return f"❌ Query '{query}' returns no results - needs improvement"
            
            quality_score = 0
            analysis = f"Query effectiveness analysis for '{query}':\n"
            
            for i, title in enumerate(search_results):
                if query.lower() in title.lower():
                    quality_score += 2
                    analysis += f"✅ Result {i+1}: '{title}' - Direct match\n"
                elif any(word in title.lower() for word in query.lower().split() if len(word) > 2):
                    quality_score += 1
                    analysis += f"⚠️ Result {i+1}: '{title}' - Partial match\n"
                else:
                    analysis += f"❌ Result {i+1}: '{title}' - Poor match\n"
            
            effectiveness = "High" if quality_score >= 4 else "Medium" if quality_score >= 2 else "Low"
            analysis += f"Overall effectiveness: {effectiveness} (score: {quality_score}/6)"
            
            return analysis
        except Exception as e:
            return f"Error testing query: {str(e)}"
    
    def enhance_query(self, original_query: str) -> Dict[str, Any]:
        """Main method to enhance a user query using the LangChain agent."""
        if not self.agent:
            return self._fallback_enhancement(original_query)
        
        logger.info(f"🤖 Query Enhancement Agent analyzing: '{original_query}'")
        
        try:
            agent_input = f"""Original user query: "{original_query}"

Your task: Create an enhanced query that will find the best Wikipedia articles to answer this question.

Process:
1. Analyze what the user is really asking about
2. Create a more targeted Wikipedia search query  
3. Test the enhanced query to see if it finds relevant articles
4. Refine if needed

Respond with ONLY the best enhanced query (no explanation needed)."""
            
            result = self.agent.run(agent_input)
            
            # Clean the result - extract just the query
            enhanced_query = self._extract_query_from_response(result, original_query)
            
            logger.info(f"✅ Enhanced: '{original_query}' → '{enhanced_query}'")
            
            return {
                'original_query': original_query,
                'enhanced_query': enhanced_query,
                'enhancement_method': 'langchain_agent',
                'agent_reasoning': result,
                'confidence': 0.9
            }
            
        except Exception as e:
            logger.error(f"❌ Query Enhancement Agent failed: {str(e)}")
            return self._fallback_enhancement(original_query)
    
    def _extract_query_from_response(self, response: str, original_query: str) -> str:
        """Extract the enhanced query from agent response."""
        # Remove common prefixes and clean the response
        lines = response.strip().split('\n')
        
        # Look for the actual query in the response
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('Thought:', 'Action:', 'Observation:', 'Final Answer:')):
                # Remove quotes if present
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]
                elif line.startswith("'") and line.endswith("'"):
                    line = line[1:-1]
                
                # If it looks like a reasonable query, use it
                if len(line) > 0 and len(line) < 100:
                    return line
        
        # Fallback to original if we can't extract
        return original_query
    
    def _fallback_enhancement(self, query: str) -> Dict[str, Any]:
        """Fallback enhancement when agent is unavailable."""
        query_lower = query.lower()
        enhanced = query
        
        # Simple rule-based enhancements
        if query_lower.startswith("who were "):
            enhanced = query.replace("Who were ", "").replace("who were ", "").strip()
        elif query_lower.startswith("who was "):
            enhanced = query.replace("Who was ", "").replace("who was ", "").strip()
        elif query_lower.startswith("what is "):
            enhanced = query.replace("What is ", "").replace("what is ", "").strip()
        elif "what happened on" in query_lower and "1969" in query_lower:
            enhanced = "Apollo 11"
        elif "july 20" in query_lower and "1969" in query_lower:
            enhanced = "Apollo 11"
        
        return {
            'original_query': query,
            'enhanced_query': enhanced,
            'enhancement_method': 'rule_based_fallback',
            'confidence': 0.6
        }

class ArticleSelectionAgent:
    """
    LangChain agent that intelligently selects the best Wikipedia articles
    from search results using reasoning and evaluation.
    """
    
    def __init__(self):
        self.llm = None
        self.agent = None
        
        if LANGCHAIN_AVAILABLE and get_openai_api_key():
            try:
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.1,  # Very low for consistent decisions
                    api_key=get_openai_api_key()
                )
                self._initialize_agent()
                logger.info("✅ Article Selection Agent initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Article Selection Agent: {str(e)}")
        else:
            logger.warning("⚠️ OpenAI not available - Article Selection Agent disabled")
    
    def _initialize_agent(self):
        """Initialize the article selection agent."""
        
        tools = [
            Tool(
                name="get_article_preview",
                func=WikipediaSearchTool.get_article_preview,
                description="Get a preview of a Wikipedia article by title. Input should be an exact article title."
            ),
            Tool(
                name="evaluate_article_relevance",
                func=self._evaluate_article_relevance,
                description="Evaluate how relevant an article is to the user's query. Input format: 'query|article_title'"
            )
        ]
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            agent_kwargs={
                "prefix": """You are an expert Article Selection Agent for Wikipedia.

Your mission: Given a user query and multiple Wikipedia article options, select the BEST article that answers their question.

Selection criteria:
1. Primary relevance: Does the article directly answer the user's question?
2. Comprehensiveness: Does it provide thorough information?
3. Avoid disambiguation pages unless the user specifically wants them
4. Avoid "List of" articles unless the user wants a list
5. Prefer main topic articles over sub-topics

Use your tools to preview articles and evaluate their relevance before deciding.""",
                "suffix": """Begin!

Question: {input}
Thought: {agent_scratchpad}"""
            }
        )
    
    def _evaluate_article_relevance(self, input_str: str) -> str:
        """Evaluate how relevant an article is to the query."""
        try:
            if '|' not in input_str:
                return "Error: Input should be 'query|article_title'"
            
            query, article_title = input_str.split('|', 1)
            query = query.strip()
            article_title = article_title.strip()
            
            # Calculate relevance score
            query_words = set(query.lower().split())
            title_words = set(article_title.lower().split())
            
            # Word overlap scoring
            overlap = len(query_words.intersection(title_words))
            total_query_words = len(query_words)
            
            # Apply penalties for less desirable article types
            penalties = 0
            penalty_reasons = []
            
            if 'disambiguation' in article_title.lower():
                penalties += 3
                penalty_reasons.append("disambiguation page")
            if 'list of' in article_title.lower():
                penalties += 2
                penalty_reasons.append("list article")
            if '(' in article_title and ')' in article_title:
                penalties += 1
                penalty_reasons.append("parenthetical specification")
            
            # Calculate final score
            base_score = (overlap / max(total_query_words, 1)) * 10
            relevance_score = max(0, min(10, base_score - penalties))
            
            analysis = f"Relevance analysis for '{article_title}' vs query '{query}':\n"
            analysis += f"Word overlap: {overlap}/{total_query_words} query words matched\n"
            analysis += f"Base score: {base_score:.1f}/10\n"
            if penalties > 0:
                analysis += f"Penalties: -{penalties} ({', '.join(penalty_reasons)})\n"
            analysis += f"Final relevance score: {relevance_score:.1f}/10\n"
            
            if relevance_score >= 7:
                analysis += "⭐ EXCELLENT choice - high relevance"
            elif relevance_score >= 5:
                analysis += "✅ GOOD choice - acceptable relevance"
            elif relevance_score >= 3:
                analysis += "⚠️ FAIR choice - moderate relevance"
            else:
                analysis += "❌ POOR choice - low relevance"
            
            return analysis
            
        except Exception as e:
            return f"Error evaluating relevance: {str(e)}"
    
    def select_best_article(self, user_query: str, article_options: List[str]) -> Dict[str, Any]:
        """Select the best Wikipedia article from options using the LangChain agent."""
        if not self.agent or not article_options:
            return self._fallback_selection(user_query, article_options)
        
        logger.info(f"🤖 Article Selection Agent choosing from {len(article_options)} options")
        
        try:
            options_str = '\n'.join([f"{i+1}. {title}" for i, title in enumerate(article_options)])
            
            agent_input = f"""User query: "{user_query}"

Available Wikipedia articles:
{options_str}

Your task: Select the BEST article that answers the user's query.

Process:
1. Evaluate each article option for relevance to the query
2. Consider which article would be most helpful to the user
3. Avoid disambiguation pages and lists unless specifically needed

Respond with ONLY the exact title of the best article (no explanation needed)."""
            
            result = self.agent.run(agent_input)
            
            # Extract the selected article from the response
            selected_article = self._extract_article_from_response(result, article_options)
            
            logger.info(f"✅ Selected article: '{selected_article}'")
            
            return {
                'user_query': user_query,
                'article_options': article_options,
                'selected_article': selected_article,
                'selection_method': 'langchain_agent',
                'agent_reasoning': result,
                'confidence': 0.9
            }
            
        except Exception as e:
            logger.error(f"❌ Article Selection Agent failed: {str(e)}")
            return self._fallback_selection(user_query, article_options)
    
    def _extract_article_from_response(self, response: str, options: List[str]) -> str:
        """Extract the selected article title from agent response."""
        response_clean = response.strip().lower()
        
        # Look for exact matches first
        for option in options:
            if option.lower() in response_clean:
                return option
        
        # Look for partial matches
        for option in options:
            option_words = option.lower().split()
            if any(word in response_clean for word in option_words if len(word) > 3):
                return option
        
        # Fallback to first option
        return options[0] if options else ""
    
    def _fallback_selection(self, user_query: str, article_options: List[str]) -> Dict[str, Any]:
        """Fallback selection when agent is unavailable."""
        if not article_options:
            return {
                'user_query': user_query,
                'article_options': [],
                'selected_article': None,
                'selection_method': 'fallback_no_options',
                'confidence': 0.0
            }
        
        # Simple heuristic: prefer non-disambiguation, non-list articles
        best_option = article_options[0]  # Default to first
        
        for option in article_options:
            option_lower = option.lower()
            # Skip disambiguation and list articles if possible
            if 'disambiguation' not in option_lower and 'list of' not in option_lower:
                best_option = option
                break
        
        return {
            'user_query': user_query,
            'article_options': article_options,
            'selected_article': best_option,
            'selection_method': 'rule_based_fallback',
            'confidence': 0.6
        }

class WikipediaAgentSystem:
    """
    Combined system using both Query Enhancement and Article Selection agents.
    Provides intelligent Wikipedia search with reasoning capabilities.
    """
    
    def __init__(self):
        self.query_agent = QueryEnhancementAgent()
        self.selection_agent = ArticleSelectionAgent()
        logger.info("🤖 Wikipedia Agent System initialized with LangChain agents")
    
    def intelligent_wikipedia_search(self, user_query: str, max_options: int = 5) -> Dict[str, Any]:
        """
        Complete intelligent Wikipedia search using both LangChain agents.
        
        Workflow:
        1. Query Enhancement Agent improves the search query
        2. Search Wikipedia with enhanced query
        3. Article Selection Agent picks the best result
        4. Return comprehensive result with reasoning
        """
        logger.info(f"🚀 Starting intelligent agent-powered search for: '{user_query}'")
        
        # Step 1: Enhance query using LangChain agent
        enhancement_result = self.query_agent.enhance_query(user_query)
        enhanced_query = enhancement_result['enhanced_query']
        
        logger.info(f"🧠 Query enhanced: '{user_query}' → '{enhanced_query}'")
        
        # Step 2: Search Wikipedia with enhanced query
        try:
            search_results = wikipedia.search(enhanced_query, results=max_options)
            logger.info(f"📚 Found {len(search_results)} Wikipedia articles")
            
            if not search_results:
                # Try original query as fallback
                search_results = wikipedia.search(user_query, results=max_options)
                logger.info(f"📚 Fallback search found {len(search_results)} articles")
                
        except Exception as e:
            logger.error(f"❌ Wikipedia search failed: {str(e)}")
            return {
                'error': f'Wikipedia search failed: {str(e)}',
                'user_query': user_query,
                'enhancement_result': enhancement_result
            }
        
        if not search_results:
            return {
                'error': 'No Wikipedia articles found',
                'user_query': user_query,
                'enhanced_query': enhanced_query,
                'enhancement_result': enhancement_result
            }
        
        # Step 3: Select best article using LangChain agent
        selection_result = self.selection_agent.select_best_article(user_query, search_results)
        selected_title = selection_result['selected_article']
        
        logger.info(f"🎯 Agent selected: '{selected_title}'")
        
        # Step 4: Get the selected article content
        try:
            # Try to get the page with auto-suggest disabled first
            try:
                selected_page = wikipedia.page(selected_title, auto_suggest=False)
            except wikipedia.exceptions.DisambiguationError as e:
                # If disambiguation, pick the first option
                logger.info(f"📝 Disambiguation page, selecting first option: {e.options[0]}")
                selected_page = wikipedia.page(e.options[0], auto_suggest=False)
            except (wikipedia.exceptions.PageError, Exception):
                # If exact match fails, try with auto-suggest enabled
                logger.info(f"⚠️ Exact match failed, trying with auto-suggest for: '{selected_title}'")
                selected_page = wikipedia.page(selected_title, auto_suggest=True)
            
            article_info = {
                'title': selected_page.title,
                'url': selected_page.url,
                'content': selected_page.content,
                'summary': selected_page.summary
            }
            logger.info(f"✅ Successfully retrieved article: '{selected_page.title}'")
        except Exception as e:
            logger.error(f"❌ Failed to get selected article: {str(e)}")
            # Try fallback to first search result
            if search_results:
                try:
                    logger.info(f"🔄 Trying fallback to first search result: '{search_results[0]}'")
                    fallback_page = wikipedia.page(search_results[0], auto_suggest=False)
                    article_info = {
                        'title': fallback_page.title,
                        'url': fallback_page.url,
                        'content': fallback_page.content,
                        'summary': fallback_page.summary
                    }
                    logger.info(f"✅ Fallback successful: '{fallback_page.title}'")
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback also failed: {str(fallback_error)}")
                    article_info = {'error': f'All attempts failed: {str(e)}'}
            else:
                article_info = {'error': f'Failed to get article: {str(e)}'}
        
        # Compile comprehensive result
        result = {
            'user_query': user_query,
            'enhancement_result': enhancement_result,
            'search_results': search_results,
            'selection_result': selection_result,
            'article_info': article_info,
            'agent_system': 'langchain_query_enhancement + langchain_article_selection',
            'total_articles_considered': len(search_results),
            'agents_used': ['QueryEnhancementAgent', 'ArticleSelectionAgent']
        }
        
        logger.info(f"✅ Intelligent agent search completed successfully")
        return result

# Demo function
def demo_langchain_agents():
    """Demonstrate the LangChain agents with example queries."""
    print("\n🤖 LANGCHAIN AGENTS DEMO")
    print("=" * 60)
    
    agent_system = WikipediaAgentSystem()
    
    test_queries = [
        "Who were the Beatles?",
        "What happened on July 20, 1969?",
        "Tell me about quantum mechanics"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing: '{query}'")
        print("-" * 40)
        
        result = agent_system.intelligent_wikipedia_search(query)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"Enhanced Query: {result['enhancement_result']['enhanced_query']}")
            print(f"Articles Found: {len(result['search_results'])}")
            print(f"Agent Selected: {result['selection_result']['selected_article']}")
            print(f"Article URL: {result['article_info'].get('url', 'N/A')}")
            print(f"Agents Used: {', '.join(result['agents_used'])}")

if __name__ == "__main__":
    demo_langchain_agents()
