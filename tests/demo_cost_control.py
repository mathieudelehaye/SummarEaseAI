"""
Demo script showcasing comprehensive logging and cost control for Multi-Source Agent
"""

import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.multi_source_agent import MultiSourceAgent, RateLimitConfig

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('multi_source_demo.log')
    ]
)

def demo_cost_control_modes():
    """Demonstrate different cost control modes."""
    print("\n💰 COST CONTROL MODES DEMONSTRATION")
    print("=" * 60)
    
    # Test query
    query = "Who were the Beatles?"
    
    modes = ["MINIMAL", "BALANCED", "COMPREHENSIVE"]
    
    for mode in modes:
        print(f"\n🎛️ Testing {mode} mode:")
        print("-" * 40)
        
        try:
            agent = MultiSourceAgent(cost_mode=mode)
            result = agent.run_multi_source_search_with_agents(query)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"✅ {mode} mode results:")
                print(f"   📄 Articles: {result.get('articles_summarized', 0)}")
                print(f"   💰 OpenAI calls: {result.get('cost_tracking', {}).get('openai_calls', 0)}")
                print(f"   📚 Wikipedia calls: {result.get('cost_tracking', {}).get('wikipedia_calls', 0)}")
                print(f"   🎯 Search method: {result.get('search_method', 'unknown')}")
                
                if result.get('openai_secondary_queries'):
                    print(f"   📡 OpenAI queries: {result['openai_secondary_queries']}")
                    
                print(f"   📋 Wikipedia pages: {result.get('wikipedia_pages_used', [])}")
        
        except Exception as e:
            print(f"❌ {mode} mode failed: {str(e)}")

def demo_detailed_logging():
    """Demonstrate detailed logging of the multi-source process."""
    print("\n📝 DETAILED LOGGING DEMONSTRATION")
    print("=" * 60)
    
    query = "What happened on July 20, 1969?"
    
    print(f"🔍 Query: '{query}'")
    print(f"🎛️ Mode: BALANCED (full logging)")
    print("\nWatch the logs above for comprehensive tracking of:")
    print("• OpenAI secondary query generation")
    print("• Wikipedia search API calls")
    print("• LangChain agent reasoning")
    print("• Article selection process")
    print("• Cost tracking")
    print("\n" + "="*60)
    
    try:
        agent = MultiSourceAgent(cost_mode="BALANCED")
        result = agent.run_multi_source_search_with_agents(query, max_articles=3)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print(f"\n📊 DETAILED RESULTS SUMMARY:")
        print(f"   🎯 Intent: {result.get('intent', 'unknown')} (confidence: {result.get('confidence', 0):.2f})")
        print(f"   📋 Strategy: {result.get('strategy', {}).get('synthesis_focus', 'unknown')}")
        
        # Show OpenAI query generation details
        if result.get('openai_query_categories'):
            print(f"\n📡 OpenAI GENERATED QUERIES BY CATEGORY:")
            for category, queries in result['openai_query_categories'].items():
                print(f"   📂 {category.upper()}:")
                for i, q in enumerate(queries, 1):
                    print(f"      {i}. '{q}'")
        
        # Show Wikipedia searches made
        print(f"\n🔍 WIKIPEDIA SEARCHES PERFORMED:")
        for i, search in enumerate(result.get('wikipedia_searches_made', []), 1):
            print(f"   {i}. '{search}'")
        
        # Show articles selected and why
        print(f"\n📚 ARTICLES SELECTED FOR SUMMARY:")
        for i, summary in enumerate(result.get('summaries', []), 1):
            print(f"   {i}. '{summary['title']}'")
            print(f"      🔍 Original query: '{summary['search_query']}'")
            print(f"      ✨ Agent enhanced: '{summary.get('agent_enhanced_query', 'N/A')}'")
            print(f"      🎯 Type: {summary.get('query_type', 'UNKNOWN')}")
            print(f"      ⭐ Relevance: {summary.get('relevance_score', '0')}")
            print(f"      🤖 Selection: {summary.get('selection_method', 'unknown')}")
        
        # Show cost breakdown
        costs = result.get('cost_tracking', {})
        print(f"\n💰 COST BREAKDOWN:")
        print(f"   📡 OpenAI API calls: {costs.get('openai_calls', 0)}")
        print(f"   📚 Wikipedia API calls: {costs.get('wikipedia_calls', 0)}")
        print(f"   📄 Articles processed: {costs.get('articles_processed', 0)}")
        print(f"   🎛️ Rate limits: {result.get('rate_limits_applied', {})}")
        
    except Exception as e:
        print(f"❌ Detailed logging demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

def demo_single_article_fallback():
    """Demonstrate single-article fallback for cost control."""
    print("\n📄 SINGLE-ARTICLE FALLBACK DEMONSTRATION")
    print("=" * 60)
    
    query = "quantum mechanics"
    
    print(f"🔍 Query: '{query}'")
    print(f"🎛️ Testing with max_articles=1 (should trigger single-article mode)")
    
    try:
        agent = MultiSourceAgent(cost_mode="BALANCED")
        result = agent.run_multi_source_search_with_agents(query, max_articles=1)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Single-article mode results:")
            print(f"   🎯 Search method: {result.get('search_method', 'unknown')}")
            print(f"   🤖 Agent powered: {result.get('agent_powered', False)}")
            print(f"   📄 Articles: {result.get('articles_summarized', 0)}")
            print(f"   💰 Cost: {result.get('cost_tracking', {})}")
            print(f"   📋 Page used: {result.get('wikipedia_pages_used', [])}")
    
    except Exception as e:
        print(f"❌ Single-article demo failed: {str(e)}")

def demo_rate_limits():
    """Demonstrate rate limiting behavior."""
    print("\n⏱️ RATE LIMITING DEMONSTRATION")
    print("=" * 60)
    
    print("Testing how the system handles rate limits...")
    
    # Show different mode limits
    for mode in ["MINIMAL", "BALANCED", "COMPREHENSIVE"]:
        limits = RateLimitConfig.get_limits_for_mode(mode)
        print(f"\n🎛️ {mode} mode limits:")
        print(f"   📄 Max articles: {limits['max_articles']}")
        print(f"   📡 Max OpenAI calls: {limits['max_secondary_queries']}")
        print(f"   📚 Max Wikipedia calls: {limits['max_wikipedia_searches']}")
        print(f"   🤖 OpenAI enabled: {limits['enable_openai']}")
        print(f"   🧠 Agents enabled: {limits['enable_agents']}")

def main():
    """Run all demonstrations."""
    print("💰 MULTI-SOURCE AGENT: COMPREHENSIVE LOGGING & COST CONTROL")
    print("=" * 80)
    print("This demo showcases:")
    print("• Detailed logging of OpenAI query generation")
    print("• Wikipedia search tracking")
    print("• LangChain agent reasoning")
    print("• Comprehensive cost control")
    print("• Smart fallback mechanisms")
    
    try:
        # Demo 1: Cost control modes
        demo_cost_control_modes()
        
        # Demo 2: Detailed logging
        demo_detailed_logging()
        
        # Demo 3: Single-article fallback
        demo_single_article_fallback()
        
        # Demo 4: Rate limits
        demo_rate_limits()
        
        print("\n\n✅ ALL DEMONSTRATIONS COMPLETED")
        print("=" * 80)
        print("📋 KEY FEATURES DEMONSTRATED:")
        print("• 💰 Smart cost control with 3 modes (MINIMAL/BALANCED/COMPREHENSIVE)")
        print("• 📡 Detailed OpenAI API call logging")
        print("• 📚 Wikipedia search tracking")
        print("• 🤖 LangChain agent reasoning")
        print("• 🎯 Article selection transparency")
        print("• ⏱️ Rate limiting protection")
        print("• 📄 Single-article fallback")
        print("\n💡 FOR PRODUCTION USE:")
        print("• Set cost_mode='MINIMAL' for max savings")
        print("• Use max_articles=1 for single-article mode")
        print("• Monitor logs for cost tracking")
        print("• Adjust RateLimitConfig for your budget")
        print("\nCheck 'multi_source_demo.log' for detailed logs!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 