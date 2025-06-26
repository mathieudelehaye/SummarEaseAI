# Flask backend API
import os
import sys
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.wikipedia_fetcher import fetch_article, search_and_fetch_article, fetch_article_with_conversion_info
from backend.summarizer import summarize_article_with_limit, summarize_article, get_summarization_status
from tensorflow_models.intent_classifier import get_intent_classifier

# Check if DirectML is available and disable conflicting imports
DIRECTML_MODE = False
try:
    import tensorflow as tf
    devices = tf.config.list_physical_devices('GPU')
    if devices and any('DML' in str(device) for device in devices):
        DIRECTML_MODE = True
        logging.info("DirectML GPU detected - running in DirectML compatibility mode")
        logging.info("Disabling Hugging Face imports to avoid TensorFlow conflicts")
except:
    pass

# Import Hugging Face modules only if not in DirectML mode
if not DIRECTML_MODE:
try:
    from backend.hf_summarizer import get_hf_summarizer, summarize_with_huggingface
    HF_SUMMARIZER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hugging Face summarizer not available: {e}")
    HF_SUMMARIZER_AVAILABLE = False

try:
    from tensorflow_models.bert_intent_classifier import get_bert_intent_classifier, predict_intent_with_bert
    HF_BERT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"BERT classifier not available: {e}")
    HF_BERT_AVAILABLE = False

try:
    from utils.semantic_search import semantic_search_wikipedia, get_semantic_search
    HF_SEMANTIC_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Semantic search not available: {e}")
    HF_SEMANTIC_AVAILABLE = False
else:
    # DirectML mode - disable all Hugging Face features
    HF_SUMMARIZER_AVAILABLE = False
    HF_BERT_AVAILABLE = False
    HF_SEMANTIC_AVAILABLE = False
    bert_classifier = None
    predict_intent_with_bert = None
    logging.info("Hugging Face features disabled for DirectML compatibility")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Streamlit integration

# Initialize intent classifiers
intent_classifier = get_intent_classifier()

# Initialize BERT classifier only if available
if HF_BERT_AVAILABLE and bert_classifier is None:
    try:
        bert_classifier = get_bert_intent_classifier()
    except Exception as e:
        logger.warning(f"Could not initialize BERT classifier: {e}")
        HF_BERT_AVAILABLE = False
        bert_classifier = None

# Try to load pre-trained models
tf_model_loaded = intent_classifier.load_model("tensorflow_models/saved_model")
bert_model_loaded = False
if bert_classifier:
    try:
        bert_model_loaded = bert_classifier.create_classifier_pipeline()
    except Exception as e:
        logger.warning(f"BERT model not available: {e}")

# Get summarization status
summarization_status = get_summarization_status()

# Set consolidated availability flags
HF_AVAILABLE = HF_SUMMARIZER_AVAILABLE or HF_BERT_AVAILABLE or HF_SEMANTIC_AVAILABLE

logger.info(f"TensorFlow model loaded: {tf_model_loaded}")
logger.info(f"Hugging Face summarizer available: {HF_SUMMARIZER_AVAILABLE}")
logger.info(f"BERT classifier available: {HF_BERT_AVAILABLE}")
logger.info(f"Semantic search available: {HF_SEMANTIC_AVAILABLE}")
logger.info(f"OpenAI summarization ready: {summarization_status['summarization_ready']}")

@app.route('/', methods=['GET'])
def home():
    """API status endpoint"""
    status = {
        'service': 'SummarEaseAI Backend API',
        'status': 'running',
        'version': '2.0.0',
        'features': {
            'tensorflow_intent_model': tf_model_loaded,
            'bert_intent_model': bert_model_loaded if HF_AVAILABLE else False,
            'huggingface_summarization': HF_AVAILABLE,
            'openai_summarization': True,
            'semantic_search': HF_AVAILABLE,
            'sentence_embeddings': HF_AVAILABLE
        },
        'endpoints': {
            '/summarize': 'POST - Summarize Wikipedia articles',
            '/summarize_local': 'POST - Local Hugging Face summarization',
            '/predict_intent': 'POST - Predict user intent (TensorFlow)',
            '/predict_intent_bert': 'POST - Predict user intent (BERT)',
            '/semantic_search': 'POST - Semantic Wikipedia search',
            '/compare_models': 'POST - Compare different AI models',
            '/health': 'GET - Health check'
        }
    }
    return jsonify(status)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'SummarEaseAI API is running'})

@app.route('/predict_intent', methods=['POST'])
def predict_intent():
    """Predict intent using TensorFlow model"""
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Predict intent using TensorFlow model
        intent, confidence = intent_classifier.predict_intent(text)
        
        response = {
            'text': text,
            'predicted_intent': intent,
            'confidence': confidence,
            'model_type': 'TensorFlow LSTM',
            'model_loaded': tf_model_loaded
        }
        
        logger.info(f"TF Intent prediction - Text: '{text}' -> Intent: {intent} (confidence: {confidence:.3f})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in TensorFlow intent prediction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict_intent_bert', methods=['POST'])
def predict_intent_bert():
    """Predict intent using BERT model"""
    if not HF_BERT_AVAILABLE or not bert_classifier:
        return jsonify({'error': 'BERT model not available (disabled for DirectML compatibility)'}), 503
    
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Predict intent using BERT model
        intent, confidence = bert_classifier.predict_intent(text)
        
        response = {
            'text': text,
            'predicted_intent': intent,
            'confidence': confidence,
            'model_type': 'BERT Transformer',
            'model_loaded': bert_model_loaded
        }
        
        logger.info(f"BERT Intent prediction - Text: '{text}' -> Intent: {intent} (confidence: {confidence:.3f})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in BERT intent prediction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    """Summarize Wikipedia articles with OpenAI/LangChain"""
    try:
        data = request.json
        if not data or 'topic' not in data:
            return jsonify({'error': 'Missing topic parameter'}), 400
        
        topic = data['topic'].strip()
        max_lines = data.get('max_lines', 30)
        use_intent = data.get('use_intent', True)
        
        if not topic:
            return jsonify({'error': 'Empty topic provided'}), 400
        
        logger.info(f"OpenAI Summarization request - Topic: '{topic}', Max lines: {max_lines}")
        
        # Predict intent if enabled and model is loaded
        predicted_intent = None
        intent_confidence = 0.0
        
        if use_intent and tf_model_loaded:
            predicted_intent, intent_confidence = intent_classifier.predict_intent(topic)
            logger.info(f"Predicted intent: {predicted_intent} (confidence: {intent_confidence:.3f})")
        
        # Fetch article content with conversion info
        article_text, processed_topic, was_converted = fetch_article_with_conversion_info(topic)
        
        # Log query conversion before sending to ChatOpenAI
        if was_converted:
            logger.info(f"📝 Query conversion detected: '{topic}' -> '{processed_topic}' before sending to ChatOpenAI gpt-3.5-turbo-16k")
        
        if not article_text:
            # Try search-based retrieval
            logger.info(f"Direct article not found, trying search for: {processed_topic if was_converted else topic}")
            article_text = search_and_fetch_article(processed_topic if was_converted else topic)
        
        if not article_text:
            return jsonify({
                'error': f'No Wikipedia article found for topic: {topic}',
                'predicted_intent': predicted_intent,
                'intent_confidence': intent_confidence
            }), 404
        
        # Generate summary using OpenAI
        if max_lines and max_lines > 0:
            summary = summarize_article_with_limit(article_text, max_lines=max_lines)
        else:
            summary = summarize_article(article_text)
        
        if summary.startswith("Error"):
            return jsonify({
                'error': summary,
                'predicted_intent': predicted_intent,
                'intent_confidence': intent_confidence
            }), 500
        
        response = {
            'topic': topic,
            'processed_topic': processed_topic if was_converted else topic,
            'query_converted': was_converted,
            'summary': summary,
            'predicted_intent': predicted_intent,
            'intent_confidence': intent_confidence,
            'max_lines': max_lines,
            'article_length': len(article_text),
            'summary_length': len(summary),
            'summarization_method': 'OpenAI + LangChain'
        }
        
        logger.info(f"OpenAI summary generated successfully for topic: {topic}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in OpenAI summarization: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/summarize_local', methods=['POST'])
def summarize_local():
    """Summarize Wikipedia articles with local Hugging Face models"""
    if not HF_AVAILABLE:
        return jsonify({'error': 'Hugging Face models not available'}), 503
    
    try:
        data = request.json
        if not data or 'topic' not in data:
            return jsonify({'error': 'Missing topic parameter'}), 400
        
        topic = data['topic'].strip()
        max_lines = data.get('max_lines', 30)
        model_name = data.get('model', 'facebook/bart-large-cnn')
        
        if not topic:
            return jsonify({'error': 'Empty topic provided'}), 400
        
        logger.info(f"Local HF Summarization request - Topic: '{topic}', Model: {model_name}")
        
        # Fetch article content with conversion info
        article_text, processed_topic, was_converted = fetch_article_with_conversion_info(topic)
        
        # Log query conversion for HF models too
        if was_converted:
            logger.info(f"📝 Query conversion detected: '{topic}' -> '{processed_topic}' before sending to Hugging Face {model_name}")
        
        if not article_text:
            article_text = search_and_fetch_article(processed_topic if was_converted else topic)
        
        if not article_text:
            return jsonify({'error': f'No Wikipedia article found for topic: {topic}'}), 404
        
        # Generate summary using Hugging Face
        summary = summarize_with_huggingface(article_text, max_lines, model_name)
        
        if summary.startswith("Error"):
            return jsonify({'error': summary}), 500
        
        response = {
            'topic': topic,
            'summary': summary,
            'max_lines': max_lines,
            'article_length': len(article_text),
            'summary_length': len(summary),
            'summarization_method': f'Hugging Face {model_name}',
            'model_name': model_name
        }
        
        logger.info(f"Local HF summary generated successfully for topic: {topic}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in local summarization: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/semantic_search', methods=['POST'])
def semantic_search():
    """Perform semantic search using sentence embeddings"""
    if not HF_AVAILABLE:
        return jsonify({'error': 'Semantic search not available'}), 503
    
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query = data['query'].strip()
        max_results = data.get('max_results', 5)
        
        if not query:
            return jsonify({'error': 'Empty query provided'}), 400
        
        logger.info(f"Semantic search request - Query: '{query}'")
        
        # Perform semantic search
        similar_titles = semantic_search_wikipedia(query, max_results)
        
        response = {
            'query': query,
            'similar_articles': similar_titles,
            'search_method': 'Sentence Embeddings',
            'model_used': 'all-MiniLM-L6-v2'
        }
        
        logger.info(f"Semantic search completed for query: {query}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/compare_models', methods=['POST'])
def compare_models():
    """Compare different AI models for intent classification"""
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        results = {}
        
        # TensorFlow LSTM model
        if tf_model_loaded:
            tf_intent, tf_confidence = intent_classifier.predict_intent(text)
            results['tensorflow_lstm'] = {
                'intent': tf_intent,
                'confidence': tf_confidence,
                'model_type': 'TensorFlow LSTM'
            }
        
        # BERT model
        if HF_AVAILABLE and bert_model_loaded:
            bert_intent, bert_confidence = bert_classifier.predict_intent(text)
            results['bert_transformer'] = {
                'intent': bert_intent,
                'confidence': bert_confidence,
                'model_type': 'BERT Transformer'
            }
        
        response = {
            'text': text,
            'model_predictions': results,
            'comparison_available': len(results) > 1
        }
        
        logger.info(f"Model comparison completed for text: '{text}'")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in model comparison: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/models_info', methods=['GET'])
def models_info():
    """Get information about available models"""
    info = {
        'tensorflow_model': intent_classifier.get_model_info() if hasattr(intent_classifier, 'get_model_info') else {},
        'huggingface_available': HF_AVAILABLE,
        'available_summarization_models': [],
        'available_intent_models': []
    }
    
    if HF_AVAILABLE:
        try:
            from backend.hf_summarizer import HuggingFaceSummarizer
            info['available_summarization_models'] = list(HuggingFaceSummarizer.get_available_models().keys())
            
            if bert_classifier:
                info['bert_model'] = bert_classifier.get_model_info()
        except:
            pass
    
    return jsonify(info)

@app.route('/categories', methods=['GET'])
def get_categories():
    """Get available intent categories"""
    return jsonify({
        'categories': intent_classifier.intent_categories,
        'total_categories': len(intent_classifier.intent_categories)
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting SummarEaseAI Backend API v2.0 on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"TensorFlow model loaded: {tf_model_loaded}")
    logger.info(f"Hugging Face available: {HF_AVAILABLE}")
    logger.info(f"BERT model loaded: {bert_model_loaded}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
