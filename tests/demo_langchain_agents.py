"""
Demo script for testing LangChain agents for Query Enhancement and Article Selection
"""

import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.langchain_agents import WikipediaAgentSystem
from utils.multi_source_agent import MultiSourceAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def demo_individual_agents():
    """Test individual LangChain agents."""
    print("\n🤖 DEMO: Individual LangChain Agents")
    print("=" * 60)
    
    # Test query enhancement agent
    from utils.langchain_agents import QueryEnhancementAgent
    query_agent = QueryEnhancementAgent()
    
    test_queries = [
        "Who were the Beatles?",
        "What happened on July 20, 1969?",
        "Tell me about quantum mechanics"
    ]
    
    print("\n🧠 QUERY ENHANCEMENT AGENT")
    print("-" * 30)
    
    for query in test_queries:
        print(f"\n🔍 Original: '{query}'")
        result = query_agent.enhance_query(query)
        print(f"✨ Enhanced: '{result['enhanced_query']}'")
        print(f"📝 Method: {result['enhancement_method']}")
        if 'agent_reasoning' in result:
            print(f"🤔 Reasoning: {result['agent_reasoning'][:100]}...")
    
    # Test article selection agent
    from utils.langchain_agents import ArticleSelectionAgent
    selection_agent = ArticleSelectionAgent()
    
    print("\n\n🎯 ARTICLE SELECTION AGENT")
    print("-" * 30)
    
    # Simulate some article options
    test_cases = [
        {
            'query': 'Who were the Beatles?',
            'options': ['The Beatles', 'List of songs recorded by the Beatles', 'The Beatles (album)', 'Beatles (disambiguation)']
        },
        {
            'query': 'Apollo 11',
            'options': ['Apollo 11', 'Apollo program', 'Neil Armstrong', 'Moon landing conspiracy theories']
        }
    ]
    
    for case in test_cases:
        print(f"\n🔍 Query: '{case['query']}'")
        print(f"📋 Options: {case['options']}")
        result = selection_agent.select_best_article(case['query'], case['options'])
        print(f"⭐ Selected: '{result['selected_article']}'")
        print(f"📝 Method: {result['selection_method']}")

def demo_combined_agent_system():
    """Test the combined Wikipedia agent system."""
    print("\n\n🌟 DEMO: Combined Wikipedia Agent System")
    print("=" * 60)
    
    agent_system = WikipediaAgentSystem()
    
    test_queries = [
        "Who were the Beatles?",
        "What happened on July 20, 1969?"
    ]
    
    for query in test_queries:
        print(f"\n🚀 Testing combined agents with: '{query}'")
        print("-" * 50)
        
        result = agent_system.intelligent_wikipedia_search(query)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"🧠 Enhanced Query: {result['enhancement_result']['enhanced_query']}")
            print(f"📚 Search Results: {len(result['search_results'])} articles found")
            print(f"🎯 Agent Selected: {result['selection_result']['selected_article']}")
            print(f"🔗 Article URL: {result['article_info'].get('url', 'N/A')}")
            print(f"🤖 Agents Used: {', '.join(result['agents_used'])}")

def demo_multi_source_with_agents():
    """Test the multi-source agent with LangChain agents."""
    print("\n\n🌟 DEMO: Multi-Source Agent with LangChain Agents")
    print("=" * 60)
    
    agent = MultiSourceAgent()
    
    test_queries = [
        "Who were the Beatles?",
        "What happened on July 20, 1969?"
    ]
    
    for query in test_queries:
        print(f"\n🚀 Testing multi-source + LangChain agents: '{query}'")
        print("-" * 50)
        
        try:
            result = agent.run_multi_source_search_with_agents(query, max_articles=4)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"🧠 Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
                print(f"📊 Strategy: {len(result['strategy']['primary_queries'])} primary + {len(result['strategy']['secondary_queries'])} secondary queries")
                print(f"📚 Articles Found: {result['total_articles_found']}")
                print(f"📝 Articles Summarized: {result['articles_summarized']}")
                print(f"🤖 Agent Powered: {result['agent_powered']}")
                
                print("\n📋 Article Summary:")
                for i, summary in enumerate(result['summaries'][:3], 1):  # Show first 3
                    print(f"  {i}. {summary['title']}")
                    print(f"     🔍 Search Query: {summary['search_query']}")
                    print(f"     ✨ Enhanced: {summary.get('agent_enhanced_query', 'N/A')}")
                    print(f"     ⭐ Relevance: {summary['relevance_score']}")
                    
        except Exception as e:
            print(f"❌ Exception: {str(e)}")

def main():
    """Run all demos."""
    print("🤖 LANGCHAIN AGENTS DEMONSTRATION")
    print("=" * 60)
    print("Testing real LangChain agents for query enhancement and article selection")
    print("Combined with OpenAI-powered multi-source strategy")
    
    try:
        # Test individual agents
        demo_individual_agents()
        
        # Test combined system
        demo_combined_agent_system()
        
        # Test multi-source with agents
        demo_multi_source_with_agents()
        
        print("\n\n✅ DEMONSTRATION COMPLETED")
        print("=" * 60)
        print("🎯 Key Features Demonstrated:")
        print("   • LangChain Query Enhancement Agent with reasoning")
        print("   • LangChain Article Selection Agent with evaluation tools")
        print("   • Combined Wikipedia Agent System")
        print("   • Multi-Source Agent enhanced with LangChain agents")
        print("   • OpenAI-powered query generation")
        print("   • Intelligent fallback mechanisms")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 