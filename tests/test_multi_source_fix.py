import requests
import json

def test_multi_source_endpoint():
    """Test the multi-source endpoint to verify the formatting fix works"""
    
    # Test data
    url = "http://localhost:5000/summarize_multi_source"
    data = {"query": "Who were the Beatles?"}
    headers = {"Content-Type": "application/json"}
    
    try:
        print("🔍 Testing multi-source endpoint...")
        print(f"📡 URL: {url}")
        print(f"📝 Query: {data['query']}")
        print("=" * 60)
        
        # Make the request
        response = requests.post(url, json=data, headers=headers, timeout=120)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Multi-source endpoint is working")
            print("=" * 60)
            print("📋 Response Summary:")
            print(f"   🎯 Query: {result.get('query', 'N/A')}")
            print(f"   🧠 Intent: {result.get('intent', 'N/A')}")
            print(f"   📚 Articles found: {result.get('total_articles_found', 0)}")
            print(f"   📄 Articles summarized: {result.get('articles_summarized', 0)}")
            print(f"   🤖 Agent powered: {result.get('agent_powered', False)}")
            print(f"   💰 Cost mode: {result.get('cost_mode', 'N/A')}")
            
            if 'final_synthesis' in result and result['final_synthesis']:
                synthesis = result['final_synthesis']
                print(f"\n📝 Final Synthesis (first 200 chars):")
                print(f"   {synthesis[:200]}...")
            
            if 'wikipedia_pages_used' in result:
                print(f"\n📚 Wikipedia pages used:")
                for i, page in enumerate(result['wikipedia_pages_used'], 1):
                    print(f"   {i}. {page}")
                    
            return True
            
        else:
            print("❌ FAILED! Error response:")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Raw response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Backend server is not running")
        print("   Start the server with: python -m backend.api_simple")
        return False
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_multi_source_endpoint()
    print("\n" + "=" * 60)
    if success:
        print("🎉 RESULT: Multi-source endpoint formatting fix SUCCESSFUL!")
    else:
        print("🚨 RESULT: Multi-source endpoint still has issues") 