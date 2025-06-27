import requests
import json

def test_single_source_endpoint():
    """Test the regular single-source summarization endpoint"""
    print("🔍 Testing single-source endpoint...")
    
    url = "http://localhost:5000/summarize"
    query = "Who were the Beatles?"
    
    payload = {
        "query": query,
        "max_lines": 30,
        "use_intent": True
    }
    
    print(f"📡 URL: {url}")
    print(f"📝 Query: {query}")
    print("=" * 60)
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Single-source endpoint is working")
            print("=" * 60)
            
            # Display key information
            print("📋 Response Summary:")
            print(f"   🎯 Query: {result.get('query', 'Unknown')}")
            print(f"   🧠 Intent: {result.get('intent', 'Unknown')}")
            print(f"   📄 Method: {result.get('method', 'Unknown')}")
            print(f"   📝 Title: {result.get('title', 'Unknown')}")
            
            # Analytics information
            print(f"\n📊 Analytics:")
            print(f"   📏 Max Lines: {result.get('max_lines', 'N/A')}")
            print(f"   📰 Article Length: {result.get('article_length', 0):,} chars")
            print(f"   📝 Summary Length: {result.get('summary_length', 0):,} chars")
            
            # Calculate compression if available
            article_len = result.get('article_length', 0)
            summary_len = result.get('summary_length', 0)
            if article_len > 0 and summary_len > 0:
                compression = (1 - summary_len / article_len) * 100
                print(f"   📦 Compression: {compression:.1f}%")
            
            # Summary preview
            summary = result.get('summary', '')
            if summary:
                print(f"\n📝 Summary (first 200 chars):")
                print(f"   {summary[:200]}...")
            
            print("\n" + "=" * 60)
            print("🎉 RESULT: Single-source endpoint analytics working!")
            return True
            
        else:
            print(f"❌ FAILED! Status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Network request failed - {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON response - {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Unexpected error - {e}")
        return False

def compare_endpoints():
    """Compare single-source vs multi-source endpoints"""
    print("\n🔄 Comparing Single-Source vs Multi-Source...")
    print("=" * 60)
    
    # Test both endpoints with the same query
    query = "Who were the Beatles?"
    
    # Single source test
    print("1️⃣ Testing Single-Source (OpenAI)...")
    single_response = requests.post("http://localhost:5000/summarize", 
                                  json={"query": query, "max_lines": 25})
    
    # Multi source test  
    print("2️⃣ Testing Multi-Source (Agent)...")
    multi_response = requests.post("http://localhost:5000/summarize_multi_source",
                                 json={"query": query, "max_lines": 25})
    
    if single_response.status_code == 200 and multi_response.status_code == 200:
        single_result = single_response.json()
        multi_result = multi_response.json()
        
        print("\n📊 Comparison Results:")
        print("=" * 60)
        
        # Method comparison
        print(f"Single-Source Method: {single_result.get('method', 'Unknown')}")
        print(f"Multi-Source Method:  {multi_result.get('method', 'Unknown')}")
        
        # Analytics comparison
        print(f"\n📏 Analytics Comparison:")
        print(f"{'Metric':<20} {'Single-Source':<15} {'Multi-Source':<15}")
        print("-" * 50)
        print(f"{'Article Length':<20} {single_result.get('article_length', 0):<15,} {multi_result.get('article_length', 0):<15,}")
        print(f"{'Summary Length':<20} {single_result.get('summary_length', 0):<15,} {multi_result.get('summary_length', 0):<15,}")
        print(f"{'Max Lines':<20} {single_result.get('max_lines', 'N/A'):<15} {multi_result.get('max_lines', 'N/A'):<15}")
        
        # Articles used
        single_pages = single_result.get('wikipedia_pages_used', [])
        multi_pages = multi_result.get('wikipedia_pages_used', [])
        
        print(f"\n📚 Articles Used:")
        print(f"Single-Source: {len(single_pages)} article(s)")
        if single_pages:
            for i, page in enumerate(single_pages, 1):
                print(f"   {i}. {page}")
        
        print(f"Multi-Source: {len(multi_pages)} article(s)")
        if multi_pages:
            for i, page in enumerate(multi_pages, 1):
                print(f"   {i}. {page}")
        
        print("\n" + "=" * 60)
        print("✅ Endpoint comparison completed!")
        
    else:
        print(f"❌ Comparison failed - Single: {single_response.status_code}, Multi: {multi_response.status_code}")

if __name__ == "__main__":
    # Test single source
    success = test_single_source_endpoint()
    
    if success:
        # Compare both endpoints
        compare_endpoints()
    else:
        print("\n❌ Single-source test failed, skipping comparison") 