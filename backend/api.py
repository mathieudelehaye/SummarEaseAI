#!/usr/bin/env python3
"""
SummarEaseAI Backend API
BERT intent classification API
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Configure logging before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backend.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Suppress transformers logging
logging.getLogger('transformers').setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# Add parent directory to path for local development
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
    logger.info(f"Added {repo_root} to Python path")

# Core imports
from ml_models.bert_classifier import get_classifier as get_bert_classifier

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize BERT classifier with absolute path
model_path = repo_root / "ml_models" / "bert_gpu_model"
logger.info(f"Loading BERT model from: {model_path}")
bert_classifier = get_bert_classifier(str(model_path))

if bert_classifier is None:
    logger.error("❌ Failed to initialize BERT classifier!")
    bert_model_loaded = False
else:
    if not bert_classifier.is_loaded():
        logger.info("🔄 Loading BERT model...")
        bert_model_loaded = bert_classifier.load_model()
        if bert_model_loaded:
            logger.info("✅ BERT model loaded successfully")
        else:
            logger.error("❌ Failed to load BERT model!")
    else:
        bert_model_loaded = True
        logger.info("✅ BERT model already loaded")

# BERT model categories
BERT_CATEGORIES = ['History', 'Music', 'Science', 'Sports', 'Technology', 'Finance']
SPECIAL_CATEGORIES = ['NO DETECTED']  # Categories for error handling
ALL_CATEGORIES = BERT_CATEGORIES + SPECIAL_CATEGORIES

# Log initialization status
logger.info("=" * 60)
logger.info("SummarEaseAI Backend Starting")
logger.info(f"Repository root: {repo_root}")
logger.info(f"BERT model path: {model_path}")
logger.info(f"BERT model loaded: {bert_model_loaded}")
logger.info(f"BERT categories: {BERT_CATEGORIES}")
logger.info(f"Special categories: {SPECIAL_CATEGORIES}")
logger.info("=" * 60)

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
        .categories { background: #e8f6f3; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .category { display: inline-block; margin: 2px 5px; padding: 3px 8px; background: #2980b9; color: white; border-radius: 3px; }
        .special-category { background: #e74c3c; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 SummarEaseAI Backend API</h1>
        
        <div class="status">
            <h3>📊 System Status</h3>
            <p><strong>BERT Model:</strong> <span class="{{ 'available' if bert_model_loaded else 'unavailable' }}">{{ '✅ Loaded' if bert_model_loaded else '❌ Not Available' }}</span></p>
        </div>

        <div class="categories">
            <h3>🏷️ BERT Categories</h3>
            {% for category in bert_categories %}
            <span class="category">{{ category }}</span>
            {% endfor %}
            
            <h4>Special Categories</h4>
            {% for category in special_categories %}
            <span class="category special-category">{{ category }}</span>
            {% endfor %}
        </div>

        <h3>🔗 Available Endpoints</h3>
        
        <div class="endpoint">
            <strong>GET /status</strong> - System status and health check
        </div>
        
        <div class="endpoint">
            <strong>POST /intent_bert</strong> - Predict intent using BERT model<br>
            <code>{"text": "your text"}</code>
        </div>

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
                                bert_categories=BERT_CATEGORIES,
                                special_categories=SPECIAL_CATEGORIES)

@app.route('/status')
def status():
    """System status endpoint"""
    return jsonify({
        'status': 'running',
        'models': {
            'bert': {
                'loaded': bert_model_loaded,
                'type': 'PyTorch BERT (CPU)',
                'gpu_enabled': False,
                'categories': BERT_CATEGORIES,
                'special_categories': SPECIAL_CATEGORIES
            }
        },
        'endpoints': [
            '/status',
            '/intent_bert'
        ]
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'backend': 'running',
        'bert_model': bert_model_loaded,
        'bert_categories': BERT_CATEGORIES
    })

@app.route('/intent_bert', methods=['POST'])
def intent_bert():
    """Predict intent using BERT model"""
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        if not bert_model_loaded:
            return jsonify({
                'error': 'BERT model not loaded',
                'message': 'BERT model is not available'
            }), 503
        
        # Use BERT model
        bert_intent, bert_confidence = bert_classifier.predict(text)
        logger.info(f"BERT prediction - Text: '{text}' -> Intent: {bert_intent} (confidence: {bert_confidence:.3f})")
        
        # Get model performance stats
        stats = bert_classifier.get_performance_stats()
        
        return jsonify({
            'text': text,
            'intent': bert_intent,
            'confidence': bert_confidence,
            'model_type': 'PyTorch BERT',
            'performance': {
                'avg_inference_time': f"{stats['avg_inference_time']*1000:.2f}ms",
                'total_predictions': stats['total_predictions']
            }
        })
        
    except Exception as e:
        logger.error(f"Error in BERT intent prediction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 