#!/usr/bin/env python3
"""Test script to debug the multi-source agent formatting error."""

import sys
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_multi_source_agent():
    """Test the multi-source agent to find the formatting error."""
    try:
        logger.info("🔍 Testing multi-source agent...")
        
        # Import the agent
        from utils.multi_source_agent import MultiSourceAgent
        
        # Create agent
        agent = MultiSourceAgent()
        logger.info("✅ Agent created successfully")
        
        # Test the search
        query = "Who were the Beatles?"
        logger.info(f"🎵 Testing query: '{query}'")
        
        result = agent.run_multi_source_search_with_agents(query)
        
        if 'error' in result:
            logger.error(f"❌ Agent returned error: {result['error']}")
        else:
            logger.info("✅ Agent completed successfully!")
            logger.info(f"📄 Articles found: {len(result.get('summaries', []))}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in test: {str(e)}")
        logger.error(f"📍 Traceback:\n{traceback.format_exc()}")
        return {'error': str(e)}

if __name__ == "__main__":
    print("🧪 Multi-Source Agent Debug Test")
    print("=" * 50)
    
    result = test_multi_source_agent()
    
    print("\n📊 RESULT:")
    print(f"Success: {'error' not in result}")
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Articles: {len(result.get('summaries', []))}")
        print(f"Final synthesis: {bool(result.get('final_synthesis'))}") 