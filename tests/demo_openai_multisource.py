"""
Demo: OpenAI-Powered Multi-Source Agent for "The Beatles"
Shows how ChatGPT generates intelligent secondary queries for comprehensive search.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.openai_query_generator import OpenAIQueryGenerator
from utils.multi_source_agent import MultiSourceAgent

def demo_beatles_openai_search():
    """
    Demonstrate how OpenAI generates intelligent secondary queries for Beatles search.
    """
    print("\n🎵 DEMO: OpenAI-Powered Multi-Source Search for 'The Beatles'")
    print("=" * 70)
    
    # Initialize the OpenAI query generator
    generator = OpenAIQueryGenerator()
    
    # Test query and intent
    query = "Who were the Beatles?"
    intent = "Music"
    
    print(f"🔍 Primary Query: '{query}'")
    print(f"🎯 Detected Intent: {intent}")
    print()
    
    # Generate comprehensive search plan
    print("🤖 Asking ChatGPT to generate secondary queries...")
    search_plan = generator.generate_comprehensive_search_plan(query, intent)
    
    print("\n📋 OpenAI SEARCH PLAN:")
    print("-" * 40)
    print(f"Strategy: {search_plan['search_strategy']}")
    print(f"Synthesis Focus: {search_plan['synthesis_focus']}")
    print(f"Total Searches: {search_plan['total_searches']}")
    print()
    
    print("🎯 GENERATED SECONDARY QUERIES:")
    for category, queries in search_plan['secondary_queries'].items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
    
    print("\n" + "=" * 70)
    print("💡 EXPECTED BENEFITS:")
    print()
    print("Instead of just 3 Wikipedia results:")
    print("❌ 1. 'List of songs recorded by the Beatles' (less relevant)")
    print("❌ 2. 'The Beatles (album)' (less relevant)")
    print("✅ 3. 'The Beatles' (most relevant)")
    print()
    print("OpenAI Multi-Source finds 7+ comprehensive articles:")
    print("✅ 1. 'The Beatles' (main band article)")
    print("✅ 2. 'John Lennon' (contextual: member biography)")
    print("✅ 3. 'Beatles discography' (contextual: complete music)")
    print("✅ 4. 'British Invasion' (related: cultural movement)")
    print("✅ 5. 'Paul McCartney' (related: member biography)")
    print("✅ 6. '1960s popular music' (broader: historical context)")
    print("✅ 7. 'Rock music history' (broader: genre evolution)")
    print()
    
    return search_plan

def demo_comparison_with_fallback():
    """
    Compare OpenAI-generated queries vs fallback queries.
    """
    print("\n⚖️ COMPARISON: OpenAI vs Fallback Query Generation")
    print("=" * 70)
    
    generator = OpenAIQueryGenerator()
    query = "Who were the Beatles?"
    intent = "Music"
    
    print("🤖 OPENAI-GENERATED QUERIES:")
    openai_queries = generator.generate_secondary_queries(query, intent, max_queries=6)
    for category, queries in openai_queries.items():
        print(f"{category}: {queries}")
    
    print("\n🔄 FALLBACK QUERIES (if OpenAI unavailable):")
    fallback_queries = generator._fallback_query_generation(query, intent, max_queries=6)
    for category, queries in fallback_queries.items():
        print(f"{category}: {queries}")
    
    print("\n📊 ANALYSIS:")
    print("OpenAI queries are more:")
    print("✅ Specific and targeted")
    print("✅ Contextually relevant") 
    print("✅ Diverse in perspectives")
    print("✅ Natural and Wikipedia-optimized")
    print()
    print("Fallback queries are:")
    print("⚠️ Generic pattern-based")
    print("⚠️ Limited coverage")
    print("✅ Reliable when OpenAI is down")

def demo_full_multisource_agent():
    """
    Demo the complete multi-source agent with OpenAI integration.
    """
    print("\n🚀 DEMO: Complete Multi-Source Agent")
    print("=" * 70)
    
    # Note: This would actually call Wikipedia APIs and OpenAI
    # For demo purposes, we'll show the workflow
    
    agent = MultiSourceAgent()
    query = "Who were the Beatles?"
    
    print(f"🔍 Running multi-source search for: '{query}'")
    print()
    print("Workflow:")
    print("1. 🎯 Intent Detection: Music (85% confidence)")
    print("2. 🤖 OpenAI Query Generation: 6 secondary queries")
    print("3. 🔍 Wikipedia Search: 7 total searches")
    print("4. 📚 Article Collection: Find unique, relevant articles")
    print("5. 📊 Relevance Ranking: Score and sort by importance")
    print("6. 🔄 Multi-Article Synthesis: Combine into comprehensive summary")
    print()
    
    # This would be the actual result structure
    expected_result = {
        'agent_type': 'multi_source',
        'original_query': query,
        'detected_intent': 'Music',
        'intent_confidence': 0.85,
        'search_strategy': 'cultural_comprehensive',
        'synthesis_method': 'multi_source_comprehensive',
        'articles_used': [
            'The Beatles',
            'John Lennon', 
            'Beatles discography',
            'British Invasion',
            'Paul McCartney',
            '1960s popular music',
            'Rock music history'
        ],
        'article_count': 7,
        'summary': 'A comprehensive summary combining all 7 articles...',
        'openai_categories': {
            'contextual': ['Beatles discography', 'Beatles career timeline'],
            'related_entities': ['John Lennon', 'British Invasion'],
            'broader_context': ['1960s popular music', 'Rock music history']
        }
    }
    
    print("📄 EXPECTED RESULT STRUCTURE:")
    for key, value in expected_result.items():
        if key == 'summary':
            print(f"{key}: {value}")
        elif isinstance(value, list):
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    print("🎵 OpenAI Multi-Source Agent Demo")
    print("This demo shows how ChatGPT enhances Wikipedia search")
    print()
    
    # Run demos
    demo_beatles_openai_search()
    demo_comparison_with_fallback()
    demo_full_multisource_agent()
    
    print("\n✨ CONCLUSION:")
    print("OpenAI-powered query generation creates more intelligent,")
    print("comprehensive Wikipedia searches that find diverse, relevant")
    print("articles for synthesis into rich, multi-perspective summaries!") 