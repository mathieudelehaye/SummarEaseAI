#!/usr/bin/env python3
"""
Fresh TensorFlow LSTM Intent Classification API
Clean, focused backend for intent prediction using TensorFlow LSTM model
"""

import os
import sys
import logging

# Suppress TensorFlow logging before any imports that might load TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR messages
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TensorFlow Intent Classifier
from tensorflow_models.intent_classifier import get_intent_classifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize TensorFlow LSTM Intent Classifier
logger.info("🚀 Initializing TensorFlow LSTM Intent Classifier...")
intent_classifier = get_intent_classifier()

# Try to load the trained model
model_path = "saved_model"
model_loaded = intent_classifier.load_model(model_path)

if model_loaded:
    logger.info("✅ TensorFlow LSTM model loaded successfully")
else:
    logger.warning("⚠️  TensorFlow LSTM model not found - using fallback classification")

# Intent categories
INTENT_CATEGORIES = [
    'History', 'Science', 'Biography', 'Technology', 
    'Arts', 'Sports', 'Politics', 'Geography', 'General'
]

@app.route('/', methods=['GET'])
def home():
    """API home page with status and documentation"""
    return jsonify({
        'service': 'TensorFlow LSTM Intent Classification API',
        'version': '1.0.0',
        'status': 'running',
        'model': {
            'type': 'TensorFlow LSTM',
            'loaded': model_loaded,
            'categories': INTENT_CATEGORIES
        },
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'GET /status': 'Detailed status',
            'POST /predict': 'Predict intent from text',
            'GET /categories': 'List intent categories'
        },
        'usage': {
            'predict_endpoint': 'POST /predict',
            'request_format': {'text': 'your text to classify'},
            'response_format': {
                'text': 'input text',
                'intent': 'predicted category',
                'confidence': 'confidence score',
                'model_type': 'TensorFlow LSTM',
                'timestamp': 'prediction time'
            }
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'tf-intent-api',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/status', methods=['GET'])
def status():
    """Detailed system status"""
    model_info = intent_classifier.get_model_info()
    
    return jsonify({
        'service': 'TensorFlow LSTM Intent Classification API',
        'status': 'running',
        'model': {
            'loaded': model_loaded,
            'type': 'TensorFlow LSTM',
            'architecture': 'Bidirectional LSTM with embeddings',
            'categories': len(INTENT_CATEGORIES),
            'details': model_info
        },
        'features': {
            'tensorflow_model': model_loaded,
            'fallback_classification': True,
            'cors_enabled': True
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict_intent():
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
        logger.info(f"🔍 Intent prediction request: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Predict intent using TensorFlow LSTM model
        predicted_intent, confidence = intent_classifier.predict_intent(text)
        
        # Prepare response
        response = {
            'text': text,
            'intent': predicted_intent,
            'confidence': round(confidence, 4),
            'model_type': 'TensorFlow LSTM',
            'model_loaded': model_loaded,
            'categories_available': INTENT_CATEGORIES,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log the result
        logger.info(f"✅ Prediction: '{text[:30]}...' -> {predicted_intent} (confidence: {confidence:.3f})")
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Error in intent prediction: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'error': 'Internal server error',
            'message': error_msg,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    """Get list of available intent categories"""
    return jsonify({
        'categories': INTENT_CATEGORIES,
        'total': len(INTENT_CATEGORIES),
        'model_type': 'TensorFlow LSTM',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict intents for multiple texts in batch
    
    Request body:
    {
        "texts": ["text1", "text2", "text3"]
    }
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing "texts" field in request'}), 400
        
        texts = data.get('texts', [])
        if not isinstance(texts, list):
            return jsonify({'error': '"texts" must be a list'}), 400
        
        if len(texts) == 0:
            return jsonify({'error': 'Texts list cannot be empty'}), 400
        
        if len(texts) > 100:  # Limit batch size
            return jsonify({'error': 'Maximum 100 texts per batch'}), 400
        
        logger.info(f"🔍 Batch prediction request: {len(texts)} texts")
        
        # Process each text
        results = []
        for i, text in enumerate(texts):
            text = str(text).strip()
            if not text:
                results.append({
                    'index': i,
                    'text': text,
                    'error': 'Empty text'
                })
                continue
            
            try:
                intent, confidence = intent_classifier.predict_intent(text)
                results.append({
                    'index': i,
                    'text': text,
                    'intent': intent,
                    'confidence': round(confidence, 4)
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'text': text,
                    'error': str(e)
                })
        
        response = {
            'results': results,
            'total_processed': len(texts),
            'model_type': 'TensorFlow LSTM',
            'model_loaded': model_loaded,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Batch prediction completed: {len(texts)} texts processed")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Error in batch prediction: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'error': 'Internal server error',
            'message': error_msg,
            'timestamp': datetime.now().isoformat()
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/', '/health', '/status', '/predict', '/categories', '/batch_predict'],
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    logger.info("🚀 Starting TensorFlow LSTM Intent Classification API")
    logger.info(f"📊 Model loaded: {model_loaded}")
    logger.info(f"🎯 Intent categories: {len(INTENT_CATEGORIES)}")
    logger.info("🌐 Server starting on http://localhost:5000")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Set to True for development
        threaded=True
    ) 