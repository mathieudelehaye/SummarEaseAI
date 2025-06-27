"""Test script to verify the format code error fix"""

def test_curly_brace_issue():
    """Test if curly braces in text cause format errors"""
    
    # Test different patterns that could cause 'f' format code error
    test_cases = [
        # Simple curly braces (shouldn't cause f error)
        "The Beatles were a {famous} rock band",
        
        # Pattern that could trigger 'f' format code error
        "The albums include {folk rock}, {funk}, and {fusion}",
        
        # Wikipedia-style infobox patterns
        "{{Infobox musical artist|name=Beatles|genre={rock}}}",
        
        # More complex patterns that might appear in Wikipedia
        "Released in {format=LP|year=1967} with {frequency=33rpm}",
        
        # Pattern specifically designed to trigger 'f' format error
        "Band members: {firstname=John, lastname=Lennon}",
    ]
    
    print("🔍 Testing different format string patterns...")
    
    for i, content in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {content[:50]}...")
        
        # Test direct format call
        try:
            template = "Summarize this: {article_text}"
            result = template.format(article_text=content)
            print(f"   ✅ Direct format succeeded")
        except Exception as e:
            print(f"   ❌ Direct format failed: {e}")
            
            # Test sanitized version
            try:
                safe_content = content.replace('{', '(').replace('}', ')')
                result = template.format(article_text=safe_content)
                print(f"   🔧 Sanitized format succeeded")
            except Exception as e2:
                print(f"   💥 Even sanitized format failed: {e2}")

def test_wikipedia_content_simulation():
    """Test with content similar to what Wikipedia might return"""
    
    # Simulate actual Wikipedia content that might contain problematic patterns
    wikipedia_style_content = """
    The Beatles were an English rock band formed in Liverpool in 1960. 
    The group consisted of John Lennon {rhythm guitar, vocals}, Paul McCartney 
    {bass guitar, vocals}, George Harrison {lead guitar, vocals}, and Ringo Starr {drums, vocals}.
    
    Their discography includes:
    - Please Please Me {format=LP, year=1963}
    - With the Beatles {format=LP, year=1963}  
    - A Hard Day's Night {format=LP, year=1964}
    
    {{Infobox band
    |name = The Beatles
    |image = 
    |caption = 
    |background = group_or_band
    |genre = {{flatlist|
    * [[Rock music|Rock]]
    * [[Pop music|pop]]
    * [[Psychedelic rock|psychedelia]]
    }}
    |years_active = 1960–1970
    }}
    
    The band's musical style evolved from {folk rock} in early albums to 
    experimental sounds including {funk}, {fusion}, and {free jazz} elements.
    """
    
    print("\n🎵 Testing Wikipedia-style content...")
    print(f"📄 Content preview: {wikipedia_style_content[:100]}...")
    
    try:
        template = "Summarize this: {article_text}"
        result = template.format(article_text=wikipedia_style_content)
        print("❌ Wikipedia-style format succeeded (unexpected)")
    except Exception as e:
        print(f"✅ Wikipedia-style format failed as expected: {e}")
        
        # Test with sanitization
        try:
            safe_content = wikipedia_style_content.replace('{', '(').replace('}', ')')
            result = template.format(article_text=safe_content)
            print("🔧 Sanitized Wikipedia-style format succeeded")
            print(f"📄 Result preview: {result[:200]}...")
        except Exception as e2:
            print(f"💥 Even sanitized failed: {e2}")

if __name__ == "__main__":
    test_curly_brace_issue()
    test_wikipedia_content_simulation() 