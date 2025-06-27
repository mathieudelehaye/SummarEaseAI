#!/usr/bin/env python3
"""
SummarEaseAI Simple Backend API (DirectML Compatible)

This is a simplified version of the Flask backend that works with DirectML TensorFlow
by avoiding problematic Hugging Face/transformers imports.
"""

import os
import sys
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core imports (these work fine with DirectML)
from utils.wikipedia_fetcher import fetch_article, search_and_fetch_article, search_and_fetch_article_info, fetch_article_with_conversion_info, enhance_query_with_intent
from backend.summarizer import summarize_article_with_limit, summarize_article, get_summarization_status, summarize_article_with_intent
from tensorflow_models.intent_classifier import get_intent_classifier

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize intent classifier (TensorFlow basic - works with DirectML)
intent_classifier = get_intent_classifier()

# Try to load pre-trained model
tf_model_loaded = intent_classifier.load_model("tensorflow_models/saved_model")

# Get summarization status
summarization_status = get_summarization_status()

# Status flags (simplified - no Hugging Face features)
HF_AVAILABLE = False
HF_SUMMARIZER_AVAILABLE = False
HF_BERT_AVAILABLE = False
HF_SEMANTIC_AVAILABLE = False

logger.info("🚀 SummarEaseAI Simple Backend initialized")
logger.info(f"✅ TensorFlow model loaded: {tf_model_loaded}")
logger.info(f"✅ OpenAI summarization: {summarization_status.get('openai_available', False)}")
logger.info("ℹ️  Hugging Face features disabled (DirectML compatibility mode)")

# HTML template for the main page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SummarEaseAI Backend</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        .status { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 15px 0; }
        .available { color: #27ae60; font-weight: bold; }
        .unavailable { color: #e74c3c; font-weight: bold; }
        .endpoint { background: #f8f9fa; padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; }
        code { background: #2c3e50; color: white; padding: 2px 5px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 SummarEaseAI Backend API</h1>
        
        <div class="status">
            <h3>📊 System Status</h3>
            <p><strong>TensorFlow Model:</strong> <span class="{{ 'available' if tf_model_loaded else 'unavailable' }}">{{ '✅ Loaded' if tf_model_loaded else '❌ Not Available' }}</span></p>
            <p><strong>OpenAI Integration:</strong> <span class="{{ 'available' if openai_available else 'unavailable' }}">{{ '✅ Available' if openai_available else '❌ Not Configured' }}</span></p>
            <p><strong>Wikipedia Fetching:</strong> <span class="available">✅ Available</span></p>
            <p><strong>Mode:</strong> <span class="available">✅ DirectML Compatible</span></p>
        </div>

        <h3>🔗 Available Endpoints</h3>
        
        <div class="endpoint">
            <strong>GET /status</strong> - System status and health check
        </div>
        
        <div class="endpoint">
            <strong>POST /summarize</strong> - Summarize Wikipedia article<br>
            <code>{"query": "your search query"}</code>
        </div>
        
        <div class="endpoint">
            <strong>POST /predict_intent</strong> - Predict user intent (TensorFlow)<br>
            <code>{"text": "your text"}</code>
        </div>
        
        <div class="endpoint">
            <strong>POST /search_wikipedia</strong> - Search Wikipedia articles<br>
            <code>{"query": "search term"}</code>
        </div>



        <div class="endpoint">
            <strong>POST /summarize_agentic</strong> - Agentic summarization with OpenAI query optimization and page selection<br>
            <code>{"query": "your search query"}</code>
        </div>

        <p><strong>Frontend URL:</strong> <a href="http://localhost:8501" target="_blank">http://localhost:8501</a></p>
        <p><strong>Backend URL:</strong> <a href="http://localhost:5000" target="_blank">http://localhost:5000</a></p>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    """Main page showing backend status and available endpoints"""
    return render_template_string(HTML_TEMPLATE, 
                                tf_model_loaded=tf_model_loaded,
                                openai_available=summarization_status.get('openai_available', False))

@app.route('/status')
def status():
    """System status endpoint"""
    return jsonify({
        'status': 'running',
        'mode': 'DirectML Compatible',
        'features': {
            'tensorflow_model': tf_model_loaded,
            'openai_summarization': summarization_status.get('openai_available', False),
            'wikipedia_fetching': True,
            'huggingface_features': False
        },
        'endpoints': ['/status', '/summarize', '/predict_intent', '/search_wikipedia', '/summarize_agentic', '/summarize_multi_source']
    })

@app.route('/health')
def health():
    """Health check endpoint for frontend compatibility"""
    return jsonify({
        'status': 'healthy',
        'backend': 'running',
        'mode': 'DirectML Compatible',
        'tensorflow_model': tf_model_loaded
    })



@app.route('/predict_intent', methods=['POST'])
def predict_intent():
    """Predict intent using hybrid approach (Keyword-based + TensorFlow)"""
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # First try keyword-based classification (more reliable)
        keyword_intent, keyword_confidence = classify_intent_keywords(text)
        
        # Use TensorFlow model if available for comparison
        if intent_classifier and tf_model_loaded:
            tf_intent, tf_confidence = intent_classifier.predict_intent(text)
            logger.info(f"TF prediction - Text: '{text}' -> Intent: {tf_intent} (confidence: {tf_confidence:.3f})")
            
            # If keyword classifier has high confidence, use it; otherwise use TF
            if keyword_confidence >= 0.6:
                intent, confidence = keyword_intent, keyword_confidence
                model_used = "Keyword-based"
            else:
                intent, confidence = tf_intent, tf_confidence
                model_used = "TensorFlow LSTM"
        else:
            # Fallback to keyword classification only
            intent, confidence = keyword_intent, keyword_confidence
            model_used = "Keyword-based"
        
        response = {
            'text': text,
            'predicted_intent': intent,
            'confidence': confidence,
            'model_type': model_used,
            'model_loaded': tf_model_loaded
        }
        
        logger.info(f"Final prediction - Text: '{text}' -> Intent: {intent} (confidence: {confidence:.3f}, model: {model_used})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in intent prediction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/search_wikipedia', methods=['POST'])
def search_wikipedia():
    """Search Wikipedia articles"""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Empty query provided'}), 400
        
        # Search Wikipedia
        article_info = search_and_fetch_article_info(query)
        
        if not article_info:
            return jsonify({'error': 'No Wikipedia articles found'}), 404
        
        response = {
            'query': query,
            'title': article_info.get('title', 'Unknown'),
            'url': article_info.get('url', ''),
            'summary': article_info.get('summary', ''),
            'content_length': len(article_info.get('content', ''))
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in Wikipedia search: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    """Summarize Wikipedia articles with intent detection and targeted search"""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Empty query provided'}), 400
        
        logger.info(f"Summarization request: '{query}'")
        
        # Step 1: Predict intent
        keyword_intent, keyword_confidence = classify_intent_keywords(query)
        
        if intent_classifier and tf_model_loaded:
            tf_intent, tf_confidence = intent_classifier.predict_intent(query)
            
            # Choose best intent prediction
            if keyword_confidence >= 0.6:
                intent, confidence = keyword_intent, keyword_confidence
                intent_model = "Keyword-based"
            else:
                intent, confidence = tf_intent, tf_confidence
                intent_model = "TensorFlow LSTM"
        else:
            intent, confidence = keyword_intent, keyword_confidence
            intent_model = "Keyword-based"
        
        logger.info(f"Intent detected: {intent} (confidence: {confidence:.3f}, model: {intent_model})")
        
        # Step 2: Enhance search query based on intent
        enhanced_query = enhance_query_with_intent(query, intent, confidence)
        
        # Step 3: Search Wikipedia with simple agentic approach
        try:
            from utils.wikipedia_fetcher import search_and_fetch_article_agentic_simple
            article_info = search_and_fetch_article_agentic_simple(query)  # Use original query for agentic optimization
        except ImportError:
            logger.warning("🚫 Simple agentic search not available - falling back to enhanced search")
            # Fallback to enhanced query approach
            article_info = search_and_fetch_article_info(enhanced_query)
            if not article_info or 'content' not in article_info:
                # Fallback to original query if enhanced search fails
                logger.info("Enhanced search failed, trying original query")
                article_info = search_and_fetch_article_info(query)
        
        if not article_info or 'content' not in article_info:
            return jsonify({'error': 'Could not fetch Wikipedia article'}), 404
        
        # Step 4: Intent-aware summarization
        summary_result = summarize_article_with_intent(
            article_info['content'], 
            article_info.get('title', 'Unknown'),
            intent,
            confidence
        )
        
        # Calculate analytics for frontend
        max_lines_requested = data.get('max_lines', 30)
        article_length = len(article_info['content'])
        summary_length = len(summary_result['summary'])
        
        response = {
            'query': query,
            'enhanced_query': enhanced_query if enhanced_query != query else None,
            'title': article_info.get('title', 'Unknown'),
            'url': article_info.get('url', ''),
            'summary': summary_result['summary'],
            'method': summary_result['method'],
            'intent_detection': {
                'predicted_intent': intent,
                'confidence': confidence,
                'model_used': intent_model
            },
            'word_count': {
                'original': len(article_info['content'].split()),
                'summary': len(summary_result['summary'].split())
            },
            
            # Analytics for frontend
            'max_lines': max_lines_requested,
            'article_length': article_length,
            'summary_length': summary_length,
            'summarization_method': summary_result['method']
        }
        
        logger.info(f"Summarization completed: {summary_result['method']}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in enhanced summarization: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Add new endpoint for agentic enhanced summarization
@app.route('/summarize_agentic', methods=['POST'])
def summarize_agentic():
    """Agentic summarization with OpenAI query optimization and page selection"""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Empty query provided'}), 400
        
        logger.info(f"🤖 Agentic summarization request: '{query}'")
        
        # Check if we have OpenAI available
        from backend.summarizer import get_openai_api_key, LANGCHAIN_AVAILABLE
        
        if not LANGCHAIN_AVAILABLE or not get_openai_api_key():
            logger.info("🚫 OpenAI not available - falling back to enhanced method")
            return summarize_enhanced()
        
        # Step 1: Predict intent (existing logic)
        keyword_intent, keyword_confidence = classify_intent_keywords(query)
        
        tf_intent = None
        tf_confidence = 0.0
        if tf_model_loaded:
            try:
                tf_intent, tf_confidence = intent_classifier.predict_intent(query)
                logger.info(f"TF prediction - Text: '{query}' -> Intent: {tf_intent} (confidence: {tf_confidence:.3f})")
            except Exception as e:
                logger.error(f"Error with TensorFlow prediction: {e}")
        
        # Choose the best intent prediction
        if keyword_confidence > 0.5:
            final_intent = keyword_intent
            final_confidence = keyword_confidence
            model_used = "Keyword-based"
        elif tf_confidence > keyword_confidence:
            final_intent = tf_intent
            final_confidence = tf_confidence
            model_used = "TensorFlow LSTM"
        else:
            final_intent = keyword_intent
            final_confidence = keyword_confidence
            model_used = "Keyword-based"
        
        logger.info(f"Intent detected: {final_intent} (confidence: {final_confidence:.3f}, model: {model_used})")
        
        # Step 2: Use agentic Wikipedia search
        try:
            from utils.wikipedia_fetcher import search_and_fetch_article_agentic
            article_info = search_and_fetch_article_agentic(query)
        except ImportError:
            logger.warning("🚫 Agentic search not available - falling back to basic search")
            article_info = search_and_fetch_article_info(query)
        
        if not article_info or 'content' not in article_info:
            return jsonify({'error': 'Could not fetch Wikipedia article'}), 404
        
        # Step 3: Intent-aware summarization (existing logic)
        summary_result = summarize_article_with_intent(
            article_info['content'], 
            article_info.get('title', 'Unknown'),
            final_intent, 
            final_confidence
        )
        
        if not summary_result:
            return jsonify({'error': 'Could not generate summary'}), 500
        
        # Build enhanced response
        response = {
            'query': query,
            'title': article_info.get('title', 'Unknown'),
            'url': article_info.get('url', ''),
            'summary': summary_result.get('summary', 'Summary not available'),
            'method': summary_result.get('method', 'unknown'),
            'word_count': summary_result.get('word_count', {}),
            'intent_detection': {
                'predicted_intent': final_intent,
                'confidence': final_confidence,
                'model_used': model_used
            },
            'search_method': article_info.get('search_method', 'agentic'),
            'openai_optimization': {
                'original_query': article_info.get('original_query', query),
                'optimized_query': article_info.get('optimized_query', query),
                'selected_from': article_info.get('selected_from', [])
            }
        }
        
        logger.info(f"✅ Agentic summarization completed successfully")
        return jsonify(response)
       
    except Exception as e:
        logger.error(f"Error in agentic summarization: {str(e)}")
        return jsonify({'error': f'Summarization failed: {str(e)}'}), 500

# Add a simple but effective keyword-based classifier
def classify_intent_keywords(text):
    """Simple keyword-based intent classification that actually works"""
    text_lower = text.lower()
    
    # Science keywords
    science_keywords = ['quantum', 'physics', 'chemistry', 'biology', 'scientific', 'experiment', 
                       'molecule', 'atom', 'dna', 'photosynthesis', 'gravity', 'relativity',
                       'thermodynamics', 'biochemistry', 'nuclear', 'particle']
    
    # History keywords  
    history_keywords = ['war', 'battle', 'ancient', 'historical', 'century', 'revolution',
                       'empire', 'medieval', 'happened', '1969', 'timeline', 'past']
    
    # Biography keywords
    biography_keywords = ['biography', 'who was', 'life story', 'einstein', 'curie', 'gandhi',
                         'person', 'scientist', 'leader', 'inventor']
    
    # Technology keywords
    technology_keywords = ['ai', 'artificial intelligence', 'computer', 'internet', 'robot',
                          'machine learning', 'blockchain', 'smartphone', 'technology']
    
    # Sports keywords
    sports_keywords = ['olympic', 'football', 'soccer', 'basketball', 'tennis', 'sports',
                      'game', 'tournament', 'athlete', 'competition']
    
    # Arts keywords
    arts_keywords = ['art', 'painting', 'music', 'literature', 'sculpture', 'theater',
                    'renaissance', 'artistic', 'creative', 'culture']
    
    # Politics keywords
    politics_keywords = ['democracy', 'government', 'politics', 'election', 'constitution',
                        'political', 'diplomacy', 'voting', 'federal']
    
    # Geography keywords
    geography_keywords = ['mountain', 'ocean', 'continent', 'climate', 'geography', 'desert',
                         'river', 'volcanic', 'population', 'location', 'where is']
    
    # Calculate scores
    scores = {
        'Science': sum(1 for keyword in science_keywords if keyword in text_lower),
        'History': sum(1 for keyword in history_keywords if keyword in text_lower),
        'Biography': sum(1 for keyword in biography_keywords if keyword in text_lower),
        'Technology': sum(1 for keyword in technology_keywords if keyword in text_lower),
        'Sports': sum(1 for keyword in sports_keywords if keyword in text_lower),
        'Arts': sum(1 for keyword in arts_keywords if keyword in text_lower),
        'Politics': sum(1 for keyword in politics_keywords if keyword in text_lower),
        'Geography': sum(1 for keyword in geography_keywords if keyword in text_lower),
        'General': 0
    }
    
    # Find best match
    if max(scores.values()) > 0:
        best_intent = max(scores, key=scores.get)
        confidence = min(0.85, max(scores.values()) * 0.3)  # More realistic confidence
    else:
        best_intent = "General"
        confidence = 0.5
    
    return best_intent, confidence

@app.route('/summarize_multi_source', methods=['POST'])
def summarize_multi_source():
    """Multi-source summarization with article synthesis"""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Empty query provided'}), 400
        
        logger.info(f"Multi-source summarization request: '{query}'")
        
        # Import here to avoid circular imports
        from utils.multi_source_agent import MultiSourceAgent
        
        # Run multi-source agent
        agent = MultiSourceAgent()
        result = agent.run_multi_source_search_with_agents(query)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        # Calculate analytics for frontend
        summary_text = result.get('final_synthesis', 'No summary available')
        max_lines_requested = data.get('max_lines', 30)
        
        # Calculate total article length from summaries
        total_article_length = 0
        summaries = result.get('summaries', [])
        for summary_item in summaries:
            # Estimate article length from summary (summaries are typically 10-20% of original)
            summary_len = len(summary_item.get('summary', ''))
            estimated_article_len = summary_len * 5  # Rough estimate
            total_article_length += estimated_article_len
        
        summary_length = len(summary_text)
        
        # Map multi-source agent response to frontend-expected format
        frontend_result = {
            'query': result.get('query', query),
            'summary': summary_text,
            'title': f"Multi-Source Analysis: {query}",
            'method': 'multi_source_agent',
            'intent': result.get('intent', 'Unknown'),
            'confidence': result.get('confidence', 0.0),
            'articles_found': result.get('total_articles_found', 0),
            'articles_summarized': result.get('articles_summarized', 0),
            'agent_powered': result.get('agent_powered', True),
            'cost_mode': result.get('cost_mode', 'BALANCED'),
            'wikipedia_pages_used': result.get('wikipedia_pages_used', []),
            'agents_used': result.get('agents_used', []),
            'cost_tracking': result.get('cost_tracking', {}),
            
            # Analytics for frontend
            'max_lines': max_lines_requested,
            'article_length': total_article_length,
            'summary_length': summary_length,
            'summarization_method': f"Multi-Source Agent ({result.get('articles_summarized', 0)} articles)",
            
            # Include raw multi-source data for debugging
            'multi_source_data': result
        }
        
        logger.info(f"Multi-source summarization completed: {result.get('synthesis_method', 'unknown')}")
        return jsonify(frontend_result)
        
    except Exception as e:
        logger.error(f"Error in multi-source summarization: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/summarize_with_agents', methods=['POST'])
def summarize_with_agents():
    """Enhanced multi-source summarization using real LangChain agents with cost control"""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Empty query provided'}), 400
        
        # Cost control parameters
        max_articles = data.get('max_articles', None)  # None = use mode default
        cost_mode = data.get('cost_mode', 'BALANCED')  # MINIMAL, BALANCED, COMPREHENSIVE
        
        logger.info(f"🤖 Agent-powered summarization request: '{query}'")
        logger.info(f"💰 Cost control: mode={cost_mode}, max_articles={max_articles}")
        
        # Import here to avoid circular imports
        from utils.multi_source_agent import MultiSourceAgent
        
        # Initialize agent with cost control
        agent = MultiSourceAgent(cost_mode=cost_mode)
        
        # Run agent-powered multi-source search
        result = agent.run_multi_source_search_with_agents(query, max_articles)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        # Log comprehensive results
        logger.info(f"✅ Agent-powered summarization completed:")
        logger.info(f"   📄 Articles: {len(result.get('summaries', []))}")
        logger.info(f"   💰 OpenAI calls: {result.get('cost_tracking', {}).get('openai_calls', 0)}")
        logger.info(f"   📚 Wikipedia calls: {result.get('cost_tracking', {}).get('wikipedia_calls', 0)}")
        logger.info(f"   📋 Pages used: {result.get('wikipedia_pages_used', [])}")
        
        if result.get('openai_secondary_queries'):
            logger.info(f"   📡 OpenAI secondary queries: {result['openai_secondary_queries']}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Error in agent-powered summarization: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting SummarEaseAI Simple Backend")
    print("=" * 50)
    print("✅ DirectML compatible mode")
    print("✅ Core features: Wikipedia + OpenAI + TensorFlow")
    print("ℹ️  Hugging Face features disabled")
    print()
    print("📡 Backend API: http://localhost:5000")
    print("🌐 Frontend UI: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    print()
    
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n🛑 Backend stopped by user") 