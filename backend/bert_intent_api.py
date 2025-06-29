#!/usr/bin/env python3
"""
Fresh BERT Intent Classification API
Clean, focused backend for intent prediction using GPU-accelerated BERT model
"""

import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import BERT GPU Classifier
from tensorflow_models.bert_gpu_classifier import GPUBERTClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize GPU BERT Intent Classifier
logger.info("🚀 Initializing GPU BERT Intent Classifier...")
bert_classifier = GPUBERTClassifier()

# Try to load the trained BERT model
model_loaded = bert_classifier.load_model()

if model_loaded:
    logger.info("✅ GPU BERT model loaded successfully")
else:
    logger.warning("⚠️  GPU BERT model not found - service unavailable")

# Intent categories (same as TensorFlow model for consistency)
INTENT_CATEGORIES = [
    'History', 'Science', 'Biography', 'Technology', 
    'Arts', 'Sports', 'Politics', 'Geography', 'General'
]

@app.route('/', methods=['GET'])
def home():
    """API home page with status and documentation"""
    return jsonify({
        'service': 'GPU BERT Intent Classification API',
        'version': '1.0.0',
        'status': 'running',
        'model': {
            'type': 'GPU BERT (DistilBERT)',
            'loaded': model_loaded,
            'categories': INTENT_CATEGORIES,
            'gpu_accelerated': True
        },
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'GET /status': 'Detailed status',
            'POST /predict': 'Predict intent from text (BERT)',
            'GET /categories': 'List intent categories'
        },
        'usage': {
            'predict_endpoint': 'POST /predict',
            'request_format': {'text': 'your text to classify'},
            'response_format': {
                'text': 'input text',
                'intent': 'predicted category',
                'confidence': 'confidence score',
                'model_type': 'GPU BERT',
                'timestamp': 'prediction time'
            }
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'bert-intent-api',
        'model_loaded': model_loaded,
        'gpu_accelerated': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/status', methods=['GET'])
def status():
    """Detailed system status"""
    return jsonify({
        'service': 'GPU BERT Intent Classification API',
        'status': 'running',
        'model': {
            'loaded': model_loaded,
            'type': 'GPU BERT (DistilBERT)',
            'architecture': 'Transformer with GPU acceleration',
            'categories': len(INTENT_CATEGORIES),
            'gpu_accelerated': True,
            'model_path': 'tensorflow_models/bert_gpu_models/'
        },
        'features': {
            'bert_model': model_loaded,
            'gpu_acceleration': True,
            'directml_compatible': True,
            'cors_enabled': True
        },
        'performance': {
            'inference_times': getattr(bert_classifier, 'inference_times', []),
            'avg_inference_time': sum(getattr(bert_classifier, 'inference_times', [0])) / max(len(getattr(bert_classifier, 'inference_times', [1])), 1)
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict_intent():
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
        if not model_loaded:
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
        predicted_intent, confidence = bert_classifier.predict(text)
        
        # Prepare response
        response = {
            'text': text,
            'intent': predicted_intent,
            'confidence': round(confidence, 4),
            'model_type': 'GPU BERT',
            'model_loaded': model_loaded,
            'categories_available': INTENT_CATEGORIES,
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

@app.route('/categories', methods=['GET'])
def get_categories():
    """Get list of available intent categories"""
    return jsonify({
        'categories': INTENT_CATEGORIES,
        'total': len(INTENT_CATEGORIES),
        'model_type': 'GPU BERT',
        'gpu_accelerated': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict intents for multiple texts in batch using GPU BERT
    
    Request body:
    {
        "texts": ["text1", "text2", "text3"]
    }
    """
    try:
        # Check if model is loaded
        if not model_loaded:
            return jsonify({
                'error': 'BERT model not loaded',
                'message': 'GPU BERT model is not available'
            }), 503
        
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
        
        if len(texts) > 50:  # Smaller batch size for BERT (more computationally intensive)
            return jsonify({'error': 'Maximum 50 texts per batch for BERT'}), 400
        
        logger.info(f"🔍 BERT batch prediction request: {len(texts)} texts")
        
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
                intent, confidence = bert_classifier.predict(text)
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
            'model_type': 'GPU BERT',
            'model_loaded': model_loaded,
            'gpu_accelerated': True,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ BERT batch prediction completed: {len(texts)} texts processed")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Error in BERT batch prediction: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'error': 'Internal server error',
            'message': error_msg,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/benchmark', methods=['POST'])
def benchmark():
    """
    Benchmark BERT model performance
    
    Request body:
    {
        "text": "test text",
        "iterations": 10
    }
    """
    try:
        if not model_loaded:
            return jsonify({'error': 'BERT model not loaded'}), 503
        
        data = request.get_json()
        text = data.get('text', 'This is a test for benchmarking the BERT model performance')
        iterations = min(data.get('iterations', 10), 100)  # Max 100 iterations
        
        logger.info(f"🏃 Starting BERT benchmark: {iterations} iterations")
        
        # Run benchmark
        times = []
        predictions = []
        
        for i in range(iterations):
            start_time = datetime.now()
            intent, confidence = bert_classifier.predict(text)
            end_time = datetime.now()
            
            inference_time = (end_time - start_time).total_seconds()
            times.append(inference_time)
            predictions.append({'intent': intent, 'confidence': confidence})
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        response = {
            'benchmark_results': {
                'text': text,
                'iterations': iterations,
                'avg_inference_time_seconds': round(avg_time, 4),
                'min_inference_time_seconds': round(min_time, 4),
                'max_inference_time_seconds': round(max_time, 4),
                'predictions_per_second': round(1/avg_time, 2),
                'model_type': 'GPU BERT',
                'gpu_accelerated': True
            },
            'sample_predictions': predictions[:5],  # First 5 predictions
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ BERT benchmark completed: {avg_time:.4f}s avg, {1/avg_time:.2f} pred/sec")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in benchmark: {str(e)}")
        return jsonify({'error': 'Benchmark failed', 'message': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/', '/health', '/status', '/predict', '/categories', '/batch_predict', '/benchmark'],
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
    logger.info("🚀 Starting GPU BERT Intent Classification API")
    logger.info(f"📊 Model loaded: {model_loaded}")
    logger.info(f"🎯 Intent categories: {len(INTENT_CATEGORIES)}")
    logger.info("🌐 Server starting on http://localhost:5001")
    
    # Run the Flask app on port 5001 (different from TensorFlow API)
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=False,  # Set to True for development
        threaded=True
    ) 