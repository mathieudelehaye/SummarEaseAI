"""
Test script to demonstrate the agentic Wikipedia search workflow
with full OpenAI API request/response logging
"""

import os
import logging
from typing import Optional, List
import wikipedia

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_openai_query_optimization(query: str) -> str:
    """Test OpenAI query optimization with full logging"""
    try:
        # Import here to avoid dependency issues
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            logger.warning("🚫 OpenAI API key not found - returning original query")
            return query
        
        # Initialize ChatOpenAI
        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=50
        )
        
        optimization_prompt = f"""
Original user query: "{query}"

Your task: Convert this query into the best Wikipedia page title or search term that would answer the user's question.

Rules:
- Return ONLY the Wikipedia page title/search term
- No explanations, no quotes, no extra text
- Be specific and concise
- Focus on the main topic the user wants to know about

Examples:
- "Who were the Beatles?" → "The Beatles"
- "What happened on July 20, 1969?" → "Apollo 11"
- "Explain quantum physics concepts" → "Quantum mechanics"

Optimized search term:"""

        logger.info("=" * 80)
        logger.info("🤖 OPENAI API CALL #1: QUERY OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(f"📝 Original query: '{query}'")
        logger.info(f"🌐 OpenAI API endpoint: https://api.openai.com/v1/chat/completions")
        logger.info(f"🔧 Model: gpt-3.5-turbo")
        logger.info(f"🌡️ Temperature: 0.1")
        logger.info(f"📊 Max tokens: 50")
        logger.info(f"💬 Full prompt sent to OpenAI:")
        logger.info(f"{optimization_prompt}")
        logger.info("=" * 40)
        
        # Make the API call
        response = llm.predict(optimization_prompt)
        optimized_query = response.strip()
        
        logger.info(f"✅ OpenAI API response received:")
        logger.info(f"📥 Raw response: '{response}'")
        logger.info(f"🎯 Optimized query: '{optimized_query}'")
        logger.info("=" * 80)
        
        return optimized_query
        
    except Exception as e:
        logger.error(f"❌ Error optimizing query with OpenAI: {str(e)}")
        return query

def test_openai_page_selection(query: str, page_options: List[str]) -> str:
    """Test OpenAI page selection with full logging"""
    try:
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key or len(page_options) <= 1:
            logger.info(f"🚫 Skipping page selection - API key: {'✅' if api_key else '❌'}, Options: {len(page_options)}")
            return page_options[0] if page_options else None
        
        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=50
        )
        
        # Format page options
        formatted_options = "\n".join([f"{i+1}. '{page}'" for i, page in enumerate(page_options)])
        
        selection_prompt = f"""
Original user question: "{query}"

Available Wikipedia pages:
{formatted_options}

Your task: Select the Wikipedia page that BEST answers the original user question.

Rules:
- Return ONLY the exact page title (without quotes or numbers)
- Choose the most relevant page for answering the user's question
- Focus on what the user actually wants to know

Best page title:"""

        logger.info("=" * 80)
        logger.info("🤖 OPENAI API CALL #2: PAGE SELECTION")
        logger.info("=" * 80)
        logger.info(f"📝 Original query: '{query}'")
        logger.info(f"📋 Available pages: {page_options}")
        logger.info(f"🌐 OpenAI API endpoint: https://api.openai.com/v1/chat/completions")
        logger.info(f"🔧 Model: gpt-3.5-turbo")
        logger.info(f"🌡️ Temperature: 0.1")
        logger.info(f"📊 Max tokens: 50")
        logger.info(f"💬 Full prompt sent to OpenAI:")
        logger.info(f"{selection_prompt}")
        logger.info("=" * 40)
        
        # Make the API call
        response = llm.predict(selection_prompt)
        selected_page = response.strip()
        
        logger.info(f"✅ OpenAI API response received:")
        logger.info(f"📥 Raw response: '{response}'")
        logger.info(f"🎯 Selected page: '{selected_page}'")
        
        # Validate selection
        if selected_page in page_options:
            logger.info(f"✅ Valid selection confirmed")
            logger.info("=" * 80)
            return selected_page
        else:
            logger.warning(f"⚠️ Invalid selection '{selected_page}', using first option: '{page_options[0]}'")
            logger.info("=" * 80)
            return page_options[0]
        
    except Exception as e:
        logger.error(f"❌ Error selecting page with OpenAI: {str(e)}")
        return page_options[0] if page_options else None

def test_agentic_workflow(query: str):
    """Test the complete agentic workflow"""
    logger.info("🚀 STARTING AGENTIC WORKFLOW TEST")
    logger.info("=" * 80)
    logger.info(f"📝 Original user query: '{query}'")
    
    # Step 1: OpenAI Query Optimization
    optimized_query = test_openai_query_optimization(query)
    
    # Step 2: Wikipedia Search
    logger.info("=" * 80)
    logger.info("🔍 WIKIPEDIA API SEARCH")
    logger.info("=" * 80)
    logger.info(f"🌐 Wikipedia search API call: wikipedia.search('{optimized_query}', results=3)")
    
    search_results = wikipedia.search(optimized_query, results=3)
    
    logger.info(f"✅ Wikipedia API returned {len(search_results)} results:")
    for i, result in enumerate(search_results):
        logger.info(f"   {i+1}. '{result}'")
    
    # Step 3: OpenAI Page Selection
    if len(search_results) > 1:
        selected_page = test_openai_page_selection(query, search_results)
    else:
        selected_page = search_results[0] if search_results else None
        logger.info(f"🎯 Only one result, selecting: '{selected_page}'")
    
    # Step 4: Fetch Wikipedia Page
    if selected_page:
        logger.info("=" * 80)
        logger.info("📄 WIKIPEDIA PAGE FETCH")
        logger.info("=" * 80)
        logger.info(f"🌐 Wikipedia page API call: wikipedia.page('{selected_page}')")
        
        try:
            page = wikipedia.page(selected_page)
            logger.info(f"✅ Successfully fetched: {page.title}")
            logger.info(f"🔗 URL: {page.url}")
            logger.info(f"📄 Content length: {len(page.content)} characters")
            
            # Summary of the workflow
            logger.info("=" * 80)
            logger.info("🎯 AGENTIC WORKFLOW SUMMARY")
            logger.info("=" * 80)
            logger.info(f"📝 Original query: '{query}'")
            logger.info(f"🤖 Optimized query: '{optimized_query}'")
            logger.info(f"📋 Search results: {search_results}")
            logger.info(f"🎯 Selected page: '{selected_page}'")
            logger.info(f"📄 Final article: {page.title}")
            logger.info(f"🔗 URL: {page.url}")
            logger.info("=" * 80)
            
            return {
                'original_query': query,
                'optimized_query': optimized_query,
                'search_results': search_results,
                'selected_page': selected_page,
                'final_title': page.title,
                'url': page.url,
                'content_length': len(page.content)
            }
            
        except Exception as e:
            logger.error(f"❌ Error fetching page: {str(e)}")
            return None
    else:
        logger.error("❌ No page selected")
        return None

if __name__ == "__main__":
    # Test cases
    test_queries = [
        "Who were the Beatles?",
        "Explain quantum physics concepts",
        "What happened on July 20, 1969?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*100}")
        print(f"TESTING: {query}")
        print(f"{'='*100}")
        
        result = test_agentic_workflow(query)
        
        if result:
            print(f"\n✅ SUCCESS: {result['final_title']}")
        else:
            print(f"\n❌ FAILED")
        
        print(f"\n{'='*100}")
        input("Press Enter to continue to next test...") 