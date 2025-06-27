#!/usr/bin/env python3
"""
Test script for the enhanced MultiSourceAgent with final synthesis.
"""

import os
import sys
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_final_synthesis():
    """Test the final synthesis feature."""
    print("🧪 Testing Final Synthesis Feature")
    print("=" * 50)
    
    try:
        from utils.multi_source_agent import MultiSourceAgent
        
        # Test with multiple articles
        print("📝 Testing with Beatles query (should trigger final synthesis)...")
        agent = MultiSourceAgent(cost_mode="BALANCED")
        
        result = agent.run_multi_source_search_with_agents("Who were the Beatles?", max_articles=2)
        
        print(f"\n📊 Results:")
        print(f"   Articles found: {result.get('total_articles_found', 0)}")
        print(f"   Articles summarized: {result.get('articles_summarized', 0)}")
        print(f"   Cost tracking: {result.get('cost_tracking', {})}")
        
        # Check individual summaries
        summaries = result.get('summaries', [])
        print(f"\n📄 Individual Summaries ({len(summaries)}):")
        for i, summary in enumerate(summaries, 1):
            title = summary.get('title', 'Unknown')
            content = summary.get('summary', 'No summary')
            print(f"   {i}. {title}")
            print(f"      Summary length: {len(content)} chars")
            print(f"      Content preview: {content[:100]}..." if len(content) > 100 else f"      Content: {content}")
        
        # Check final synthesis
        final_synthesis = result.get('final_synthesis')
        if final_synthesis:
            print(f"\n🎯 Final Synthesis:")
            print(f"   Length: {len(final_synthesis)} chars")
            print(f"   Content: {final_synthesis}")
        else:
            print(f"\n❌ No final synthesis found in result")
            print(f"   Available keys: {list(result.keys())}")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_single_article():
    """Test single article mode (should not trigger final synthesis)."""
    print("\n🧪 Testing Single Article Mode")
    print("=" * 50)
    
    try:
        from utils.multi_source_agent import MultiSourceAgent
        
        agent = MultiSourceAgent(cost_mode="MINIMAL")
        result = agent.run_multi_source_search_with_agents("Who were the Beatles?", max_articles=1)
        
        print(f"📊 Results:")
        print(f"   Articles found: {result.get('total_articles_found', 0)}")
        print(f"   Articles summarized: {result.get('articles_summarized', 0)}")
        print(f"   Final synthesis present: {'final_synthesis' in result}")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return None

if __name__ == "__main__":
    # Set environment variable for protobuf compatibility
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    
    print("🚀 Multi-Source Agent Final Synthesis Test")
    print("=" * 60)
    
    # Test 1: Multiple articles (should trigger synthesis)
    result1 = test_final_synthesis()
    
    # Test 2: Single article (should not trigger synthesis)
    result2 = test_single_article()
    
    print("\n✅ Testing complete!")
    
    if result1 and result2:
        print("🎉 Both tests completed successfully!")
    else:
        print("⚠️ Some tests had issues.") 