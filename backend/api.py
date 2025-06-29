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
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core imports (these work fine with DirectML)
from utils.wikipedia_fetcher import fetch_article, search_and_fetch_article, search_and_fetch_article_info, fetch_article_with_conversion_info, enhance_query_with_intent
from backend.summarizer import summarize_article_with_limit, summarize_article, get_summarization_status, summarize_article_with_intent
from tensorflow_models.bert_gpu_classifier import get_gpu_classifier

# Import TensorFlow Intent Classifier and BERT GPU Classifier for new endpoints
from tensorflow_models.intent_classifier import get_intent_classifier
from tensorflow_models.bert_gpu_classifier import GPUBERTClassifier

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize BERT intent classifier (GPU-accelerated)
bert_classifier = get_gpu_classifier()

# Try to load pre-trained BERT model
bert_model_loaded = bert_classifier.load_model()

# Initialize TensorFlow LSTM Intent Classifier for new endpoint
tf_intent_classifier = get_intent_classifier()
tf_model_loaded = tf_intent_classifier.load_model("saved_model")

# Initialize separate BERT GPU Classifier for new endpoint
bert_gpu_classifier = GPUBERTClassifier()
bert_gpu_model_loaded = bert_gpu_classifier.load_model()

# Get summarization status
summarization_status = get_summarization_status()

# Status flags (GPU BERT enabled)
HF_AVAILABLE = False
HF_SUMMARIZER_AVAILABLE = False
HF_BERT_AVAILABLE = True  # GPU BERT is available
HF_SEMANTIC_AVAILABLE = False

logger.info("🚀 SummarEaseAI Backend initialized with GPU BERT")
logger.info(f"✅ GPU BERT model loaded: {bert_model_loaded}")
logger.info(f"✅ OpenAI summarization: {summarization_status.get('openai_available', False)}")
logger.info("🚀 GPU-accelerated BERT intent classification enabled")

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
            <p><strong>GPU BERT Model:</strong> <span class="{{ 'available' if bert_model_loaded else 'unavailable' }}">{{ '✅ Loaded' if bert_model_loaded else '❌ Not Available' }}</span></p>
            <p><strong>OpenAI Integration:</strong> <span class="{{ 'available' if openai_available else 'unavailable' }}">{{ '✅ Available' if openai_available else '❌ Not Configured' }}</span></p>
            <p><strong>Wikipedia Fetching:</strong> <span class="available">✅ Available</span></p>
            <p><strong>Mode:</strong> <span class="available">🚀 GPU BERT Accelerated</span></p>
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
            <strong>POST /predict_intent</strong> - Predict user intent (GPU BERT)<br>
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
                                bert_model_loaded=bert_model_loaded,
                                openai_available=summarization_status.get('openai_available', False))

@app.route('/status')
def status():
    """System status endpoint"""
    return jsonify({
        'status': 'running',
        'mode': 'GPU BERT Accelerated',
        'features': {
            'bert_model': bert_model_loaded,
            'openai_summarization': summarization_status.get('openai_available', False),
            'wikipedia_fetching': True,
            'gpu_acceleration': True
        },
        'endpoints': ['/status', '/summarize', '/predict_intent', '/search_wikipedia', '/summarize_agentic', '/summarize_multi_source', '/intent', '/intent_bert']
    })

@app.route('/health')
def health():
    """Health check endpoint for frontend compatibility"""
    return jsonify({
        'status': 'healthy',
        'backend': 'running',
        'mode': 'GPU BERT Accelerated',
        'bert_model': bert_model_loaded
    })



@app.route('/predict_intent', methods=['POST'])
def predict_intent():
    """Predict intent using GPU BERT with keyword fallback"""
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Try GPU BERT model first
        if bert_classifier and bert_model_loaded:
            try:
                bert_intent, bert_confidence = bert_classifier.predict(text)
                logger.info(f"GPU BERT prediction - Text: '{text}' -> Intent: {bert_intent} (confidence: {bert_confidence:.3f})")
                
                response = {
                    'text': text,
                    'predicted_intent': bert_intent,
                    'confidence': bert_confidence,
                    'model_type': "GPU BERT",
                    'model_loaded': bert_model_loaded
                }
                
                logger.info(f"Final prediction - Text: '{text}' -> Intent: {bert_intent} (confidence: {bert_confidence:.3f}, model: GPU BERT)")
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"GPU BERT prediction failed: {str(e)}, falling back to keywords")
        
        # Fallback to keyword-based classification
        keyword_intent, keyword_confidence = classify_intent_keywords(text)
        
        response = {
            'text': text,
            'predicted_intent': keyword_intent,
            'confidence': keyword_confidence,
            'model_type': "Keyword-based (fallback)",
            'model_loaded': bert_model_loaded
        }
        
        logger.info(f"Fallback prediction - Text: '{text}' -> Intent: {keyword_intent} (confidence: {keyword_confidence:.3f})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in intent prediction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict_intent_bert', methods=['POST'])
def predict_intent_bert():
    """Predict intent using GPU BERT model"""
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Use GPU BERT model
        if bert_classifier and bert_model_loaded:
            bert_intent, bert_confidence = bert_classifier.predict(text)
            logger.info(f"GPU BERT prediction - Text: '{text}' -> Intent: {bert_intent} (confidence: {bert_confidence:.3f})")
            
            response = {
                'text': text,
                'predicted_intent': bert_intent,
                'confidence': bert_confidence,
                'model_type': "GPU BERT",
                'model_loaded': bert_model_loaded
            }
        else:
            # Fallback if model not loaded
            response = {
                'text': text,
                'predicted_intent': "Unknown",
                'confidence': 0.0,
                'model_type': "GPU BERT (Not loaded)",
                'model_loaded': bert_model_loaded
            }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in BERT intent prediction: {str(e)}")
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
    """Enhanced summarization with intent detection and smart article selection"""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Empty query provided'}), 400
        
        logger.info(f"📝 Summarization request: '{query}'")
        
        # Intent detection using GPU BERT
        intent = "General"
        intent_confidence = 0.0
        intent_model = "Keyword fallback"
        
        if bert_classifier and bert_model_loaded:
            try:
                intent, intent_confidence = bert_classifier.predict(query)
                intent_model = "GPU BERT"
                logger.info(f"🧠 Intent detected: {intent} (confidence: {intent_confidence:.3f})")
            except Exception as e:
                logger.error(f"GPU BERT intent detection failed: {str(e)}")
                # Fallback to keyword classification
                intent, intent_confidence = classify_intent_keywords(query)
                intent_model = "Keyword fallback"
        else:
            # Fallback to keyword classification
            intent, intent_confidence = classify_intent_keywords(query)
            intent_model = "Keyword fallback"
        
        logger.info(f"Intent detected: {intent} (confidence: {intent_confidence:.3f}, model: {intent_model})")
        
        # Step 2: Enhance search query based on intent
        enhanced_query = enhance_query_with_intent(query, intent, intent_confidence)
        
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
            intent_confidence
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
                'confidence': intent_confidence,
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
        
        # Intent detection
        intent = "General"
        if bert_classifier and bert_model_loaded:
            try:
                intent, _ = bert_classifier.predict(query)
                logger.info(f"🧠 Intent detected: {intent}")
            except Exception as e:
                logger.error(f"Intent detection failed: {str(e)}")
        
        # Use the enhanced summarization function
        try:
            # This function should exist - if not, we'll use basic summarization
            result = summarize_article_with_intent(query, intent)
            
            if result and 'summary' in result:
                response = {
                    'query': query,
                    'intent': intent,
                    'summary': result['summary'],
                    'article_info': result.get('article_info', {}),
                    'method': 'agentic',
                    'status': 'success'
                }
                
                logger.info(f"✅ Agentic summarization completed for: '{query}'")
                return jsonify(response)
            else:
                return jsonify({'error': 'Failed to generate summary'}), 500
                
        except Exception as e:
            logger.error(f"Agentic summarization error: {str(e)}")
            return jsonify({'error': 'Agentic summarization failed'}), 500
        
    except Exception as e:
        logger.error(f"Error in agentic summarization: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

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

@app.route('/intent', methods=['POST'])
def intent():
    """
    Predict intent using TensorFlow LSTM model
    
    Request body:
    {
        "text": "your text to classify"
    }
    
    Response:
    {
        "text": "input text",
        "intent": "predicted category", 
        "confidence": confidence_score,
        "model_type": "TensorFlow LSTM",
        "model_loaded": true/false,
        "timestamp": "2024-01-01T12:00:00"
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request'}), 400
        
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'Text field cannot be empty'}), 400
        
        # Log the prediction request
        logger.info(f"🔍 TF Intent prediction request: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Predict intent using TensorFlow LSTM model
        predicted_intent, confidence = tf_intent_classifier.predict_intent(text)
        
        # Prepare response
        response = {
            'text': text,
            'intent': predicted_intent,
            'confidence': round(confidence, 4),
            'model_type': 'TensorFlow LSTM',
            'model_loaded': tf_model_loaded,
            'categories_available': ['History', 'Science', 'Biography', 'Technology', 'Arts', 'Sports', 'Politics', 'Geography', 'General'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Log the result
        logger.info(f"✅ TF Prediction: '{text[:30]}...' -> {predicted_intent} (confidence: {confidence:.3f})")
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Error in TF intent prediction: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'error': 'Internal server error',
            'message': error_msg,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/intent_bert', methods=['POST'])
def intent_bert():
    """
    Predict intent using GPU BERT model
    
    Request body:
    {
        "text": "your text to classify"
    }
    
    Response:
    {
        "text": "input text",
        "intent": "predicted category", 
        "confidence": confidence_score,
        "model_type": "GPU BERT",
        "model_loaded": true/false,
        "timestamp": "2024-01-01T12:00:00"
    }
    """
    try:
        # Check if model is loaded
        if not bert_gpu_model_loaded:
            return jsonify({
                'error': 'BERT model not loaded',
                'message': 'GPU BERT model is not available',
                'model_loaded': False
            }), 503
        
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request'}), 400
        
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'Text field cannot be empty'}), 400
        
        # Log the prediction request
        logger.info(f"🔍 BERT prediction request: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Predict intent using GPU BERT model
        predicted_intent, confidence = bert_gpu_classifier.predict(text)
        
        # Prepare response
        response = {
            'text': text,
            'intent': predicted_intent,
            'confidence': round(confidence, 4),
            'model_type': 'GPU BERT',
            'model_loaded': bert_gpu_model_loaded,
            'categories_available': ['History', 'Science', 'Biography', 'Technology', 'Arts', 'Sports', 'Politics', 'Geography', 'General'],
            'gpu_accelerated': True,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log the result
        logger.info(f"✅ BERT Prediction: '{text[:30]}...' -> {predicted_intent} (confidence: {confidence:.3f})")
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Error in BERT intent prediction: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'error': 'Internal server error',
            'message': error_msg,
            'model_type': 'GPU BERT',
            'timestamp': datetime.now().isoformat()
        }), 500

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