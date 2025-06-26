"""
OpenAI-Powered Query Generator for Multi-Source Wikipedia Search
Uses GPT to intelligently generate secondary search queries.
"""

import logging
from typing import List, Dict, Tuple
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import SecretStr
from backend.summarizer import get_openai_api_key, LANGCHAIN_AVAILABLE

logger = logging.getLogger(__name__)

class OpenAIQueryGenerator:
    """
    Uses OpenAI/ChatGPT to generate intelligent secondary queries for multi-source search.
    """
    
    def __init__(self):
        self.llm = None
        if LANGCHAIN_AVAILABLE and get_openai_api_key():
            try:
                api_key = get_openai_api_key()
                if api_key:
                    self.llm = ChatOpenAI(
                        model="gpt-3.5-turbo",
                        temperature=0.3,  # Low temperature for consistent, focused results
                        max_completion_tokens=1000,
                        api_key=SecretStr(api_key)
                    )
                else:
                    self.llm = None
                logger.info("✅ OpenAI Query Generator initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI: {str(e)}")
                self.llm = None
        else:
            logger.warning("⚠️ OpenAI not available - will use fallback query generation")
    
    def generate_secondary_queries(self, primary_query: str, intent: str, max_queries: int = 6) -> Dict[str, List[str]]:
        """
        Generate intelligent secondary queries using OpenAI.
        
        Args:
            primary_query: The original user query
            intent: Detected intent category (Biography, History, Science, etc.)
            max_queries: Maximum number of queries to generate
            
        Returns:
            Dictionary with categorized secondary queries
        """
        if not self.llm:
            logger.info("🔄 OpenAI unavailable - using fallback query generation")
            return self._fallback_query_generation(primary_query, intent, max_queries)
        
        logger.info(f"🤖 Generating secondary queries for: '{primary_query}' (intent: {intent})")
        
        try:
            # Create specialized prompt based on intent
            prompt_template = self._get_intent_specific_prompt(intent)
            
            prompt = PromptTemplate(
                input_variables=["query", "intent", "max_queries"],
                template=prompt_template
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            # Generate queries using OpenAI
            result = chain.run(
                query=primary_query,
                intent=intent,
                max_queries=max_queries
            )
            
            # Parse the OpenAI response
            secondary_queries = self._parse_openai_response(result)
            
            logger.info(f"✅ Generated {sum(len(queries) for queries in secondary_queries.values())} secondary queries")
            return secondary_queries
            
        except Exception as e:
            logger.error(f"❌ OpenAI query generation failed: {str(e)}")
            return self._fallback_query_generation(primary_query, intent, max_queries)
    
    def _get_intent_specific_prompt(self, intent: str) -> str:
        """Get specialized prompt template based on detected intent."""
        
        base_instruction = """You are an expert Wikipedia research assistant. Given a user query, generate strategic secondary queries to find comprehensive, diverse information from multiple Wikipedia articles.

Primary Query: "{query}"
Detected Intent: "{intent}"

Generate {max_queries} secondary Wikipedia search queries that will find different aspects and perspectives. Return ONLY a JSON object with this structure:

{{
    "contextual": ["query1", "query2"],
    "related_entities": ["query3", "query4"], 
    "broader_context": ["query5", "query6"]
}}

"""
        
        intent_specific_guidance = {
            "Biography": """
For biographical queries, find:
- contextual: Early life, career highlights, major achievements
- related_entities: Associated people, organizations, movements they were part of
- broader_context: Historical period, cultural/social context, legacy and influence

Example for "Who was Albert Einstein?":
- contextual: "Albert Einstein early life", "Einstein scientific achievements"
- related_entities: "Manhattan Project scientists", "Princeton University faculty"
- broader_context: "20th century physics", "German-Jewish immigration America"
""",
            
            "History": """
For historical queries, find:
- contextual: Causes, key events, immediate consequences
- related_entities: Major figures, countries/organizations involved
- broader_context: Historical period, long-term significance, related events

Example for "What was World War II?":
- contextual: "World War II causes", "World War II major battles"
- related_entities: "Winston Churchill", "Nazi Germany"
- broader_context: "20th century warfare", "Post-war reconstruction"
""",
            
            "Science": """
For scientific queries, find:
- contextual: Core principles, applications, research history
- related_entities: Key scientists, related theories, experimental evidence
- broader_context: Scientific field overview, technological applications

Example for "What is quantum mechanics?":
- contextual: "Quantum mechanics principles", "Quantum physics experiments"
- related_entities: "Max Planck", "Schrödinger equation"
- broader_context: "Modern physics", "Quantum computing applications"
""",
            
            "Music": """
For music queries, find:
- contextual: Musical style, discography, career timeline
- related_entities: Band members, collaborators, influenced artists
- broader_context: Musical genre, cultural movement, historical context

Example for "Who were The Beatles?":
- contextual: "Beatles discography", "Beatles career timeline"
- related_entities: "John Lennon", "British Invasion"
- broader_context: "1960s popular music", "Rock music history"
""",
            
            "Technology": """
For technology queries, find:
- contextual: How it works, development history, current applications
- related_entities: Inventors, companies, related technologies
- broader_context: Technological field, societal impact, future developments

Example for "What is artificial intelligence?":
- contextual: "AI algorithms", "Machine learning history"
- related_entities: "Alan Turing", "Deep learning"
- broader_context: "Computer science", "AI ethics"
"""
        }
        
        # Get intent-specific guidance or use generic
        specific_guidance = intent_specific_guidance.get(intent, intent_specific_guidance["Biography"])
        
        return base_instruction + specific_guidance
    
    def _parse_openai_response(self, response: str) -> Dict[str, List[str]]:
        """Parse OpenAI response to extract secondary queries."""
        try:
            # Clean the response to extract JSON
            response = response.strip()
            
            # Find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)
            
            # Validate structure
            required_keys = ["contextual", "related_entities", "broader_context"]
            for key in required_keys:
                if key not in parsed:
                    parsed[key] = []
                elif not isinstance(parsed[key], list):
                    parsed[key] = [str(parsed[key])]
            
            # Log the parsed queries
            logger.info("🎯 OpenAI generated queries:")
            for category, queries in parsed.items():
                logger.info(f"  {category}: {queries}")
            
            return parsed
            
        except Exception as e:
            logger.error(f"❌ Failed to parse OpenAI response: {str(e)}")
            logger.error(f"Raw response: {response}")
            
            # Return fallback structure
            return {
                "contextual": [],
                "related_entities": [],
                "broader_context": []
            }
    
    def _fallback_query_generation(self, primary_query: str, intent: str, max_queries: int = 6) -> Dict[str, List[str]]:
        """Fallback query generation when OpenAI is unavailable."""
        logger.info(f"🔄 Using fallback query generation for: '{primary_query}'")
        
        query_lower = primary_query.lower()
        
        # Basic pattern-based generation
        fallback_patterns = {
            "contextual": [
                f"{primary_query} history",
                f"{primary_query} overview"
            ],
            "related_entities": [
                f"{primary_query} related",
                f"{primary_query} associated"
            ],
            "broader_context": [
                f"{primary_query} context",
                f"{primary_query} significance"
            ]
        }
        
        # Intent-specific improvements
        if 'beatles' in query_lower:
            fallback_patterns = {
                "contextual": ["Beatles discography", "Beatles career timeline"],
                "related_entities": ["John Lennon", "Paul McCartney"],
                "broader_context": ["British Invasion", "1960s popular music"]
            }
        elif 'apollo' in query_lower or 'moon landing' in query_lower:
            fallback_patterns = {
                "contextual": ["Apollo program", "Moon landing 1969"],
                "related_entities": ["Neil Armstrong", "NASA astronauts"],
                "broader_context": ["Space Race", "Space exploration"]
            }
        elif 'quantum' in query_lower:
            fallback_patterns = {
                "contextual": ["Quantum mechanics principles", "Quantum physics"],
                "related_entities": ["Albert Einstein", "Max Planck"],
                "broader_context": ["Modern physics", "Physics theories"]
            }
        
        logger.info("🔄 Fallback queries generated:")
        for category, queries in fallback_patterns.items():
            logger.info(f"  {category}: {queries}")
        
        return fallback_patterns
    
    def generate_comprehensive_search_plan(self, primary_query: str, intent: str) -> Dict[str, any]:
        """
        Generate a comprehensive search plan using OpenAI insights.
        """
        logger.info(f"📋 Creating comprehensive search plan for: '{primary_query}'")
        
        # Generate secondary queries
        secondary_queries = self.generate_secondary_queries(primary_query, intent, max_queries=6)
        
        # Create search plan
        search_plan = {
            "primary_query": primary_query,
            "intent": intent,
            "secondary_queries": secondary_queries,
            "total_searches": 1 + sum(len(queries) for queries in secondary_queries.values()),
            "search_strategy": self._determine_search_strategy(intent),
            "synthesis_focus": self._determine_synthesis_focus(intent)
        }
        
        logger.info(f"📊 Search plan created: {search_plan['total_searches']} total searches")
        return search_plan
    
    def _determine_search_strategy(self, intent: str) -> str:
        """Determine the best search strategy based on intent."""
        strategy_map = {
            "Biography": "chronological_comprehensive",
            "History": "causal_temporal",
            "Science": "conceptual_hierarchical", 
            "Technology": "evolutionary_applications",
            "Music": "cultural_comprehensive",
            "Sports": "competitive_achievements",
            "Politics": "systemic_impact",
            "Geography": "spatial_contextual"
        }
        return strategy_map.get(intent, "general_comprehensive")
    
    def _determine_synthesis_focus(self, intent: str) -> str:
        """Determine how to synthesize multiple articles."""
        focus_map = {
            "Biography": "chronological",
            "History": "causal",
            "Science": "conceptual", 
            "Technology": "evolutionary",
            "Music": "comprehensive",
            "Sports": "competitive",
            "Politics": "analytical",
            "Geography": "spatial"
        }
        return focus_map.get(intent, "comprehensive")

# Example usage and testing
def test_openai_query_generation():
    """Test the OpenAI query generator with different examples."""
    generator = OpenAIQueryGenerator()
    
    test_cases = [
        ("Who were the Beatles?", "Music"),
        ("What happened on July 20, 1969?", "History"),
        ("What is quantum mechanics?", "Science"),
        ("Who was Albert Einstein?", "Biography")
    ]
    
    print("\n🤖 OPENAI QUERY GENERATION TEST")
    print("=" * 60)
    
    for query, intent in test_cases:
        print(f"\n🔍 Query: '{query}' (Intent: {intent})")
        print("-" * 40)
        
        search_plan = generator.generate_comprehensive_search_plan(query, intent)
        
        print(f"Strategy: {search_plan['search_strategy']}")
        print(f"Synthesis: {search_plan['synthesis_focus']}")
        print(f"Total searches: {search_plan['total_searches']}")
        print("\nSecondary queries:")
        
        for category, queries in search_plan['secondary_queries'].items():
            print(f"  {category.replace('_', ' ').title()}:")
            for q in queries:
                print(f"    • {q}")

if __name__ == "__main__":
    test_openai_query_generation() 