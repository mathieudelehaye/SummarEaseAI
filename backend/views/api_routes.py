#!/usr/bin/env python3
"""
SummarEaseAI API Routes - View Layer
Pure HTTP concerns - routing, request/response handling
No business logic - delegates to services
"""

import os
import sys
import logging
import warnings
from pathlib import Path

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# Configure logging before any other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Suppress transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add parent directory to path for local development
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
    logger.info("Added %s to Python path", repo_root)

# Service imports - all business logic comes from here
from backend.services.summarization_service import get_summarization_service

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize service
summarization_service = get_summarization_service()

# BERT categories for display
BERT_CATEGORIES = ["History", "Music", "Science", "Sports", "Technology", "Finance"]
SPECIAL_CATEGORIES = ["NO DETECTED"]

# Log initialization status
logger.info("=" * 60)
logger.info("SummarEaseAI API Routes Starting")
logger.info("Repository root: %s", repo_root)
logger.info("Initializing Flask routes...")
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
        
        <div class="endpoint">
            <strong>POST /summarize</strong> - Single-source summarization<br>
            <code>{"query": "your query", "max_lines": 30}</code>
        </div>
        
        <div class="endpoint">
            <strong>POST /summarize_multi_source</strong> - Multi-source summarization with agents<br>
            <code>{"query": "your query", "max_lines": 30, "max_articles": 3, "cost_mode": "BALANCED"}</code>
        </div>

        <p><strong>Backend URL:</strong> <a href="http://localhost:5000" target="_blank">http://localhost:5000</a></p>
    </div>
</body>
</html>
"""


@app.route("/")
def home():
    """Main page showing backend status and available endpoints"""
    # Get status from service
    system_status = summarization_service.get_system_status()
    bert_model_loaded = system_status["models"]["bert"]["loaded"]

    return render_template_string(
        HTML_TEMPLATE,
        bert_model_loaded=bert_model_loaded,
        bert_categories=BERT_CATEGORIES,
        special_categories=SPECIAL_CATEGORIES,
    )


@app.route("/status")
def status():
    """System status endpoint - delegates to service"""
    return jsonify(summarization_service.get_system_status())


@app.route("/health")
def health():
    """Health check endpoint - delegates to service"""
    return jsonify(summarization_service.get_health_status())


@app.route("/intent_bert", methods=["POST"])
def intent_bert():
    """Predict intent using BERT model - HTTP handling only"""
    try:
        # Extract and validate input
        data = request.json
        if not data or "text" not in data:
            return jsonify({"error": "Missing text parameter"}), 400

        text = data["text"].strip()
        if not text:
            return jsonify({"error": "Empty text provided"}), 400

        # Delegate to service
        result = summarization_service.classify_intent(text)

        # Handle service errors
        if "error" in result:
            if "not loaded" in result["error"]:
                return jsonify(result), 503
            return jsonify(result), 500

        # Return successful result
        return jsonify(result)

    except Exception as e:
        logger.error("Error in intent_bert endpoint: %s", str(e))
        return jsonify({"error": "Internal server error"}), 500


@app.route("/summarize", methods=["POST"])
def summarize():
    """Single-source summarization endpoint - HTTP handling only"""
    try:
        # Extract and validate input
        data = request.json
        if not data or "query" not in data:
            return jsonify({"error": "Missing query parameter"}), 400

        query = data["query"].strip()
        max_lines = data.get("max_lines", 30)

        if not query:
            return jsonify({"error": "Empty query provided"}), 400

        if max_lines < 5 or max_lines > 100:
            return jsonify({"error": "max_lines must be between 5 and 100"}), 400

        # Delegate to service
        result = summarization_service.summarize_single_source(query, max_lines)

        # Handle service errors
        if "error" in result:
            if "No Wikipedia content found" in result["error"]:
                return jsonify(result), 404
            return jsonify(result), 500

        # Return successful result
        return jsonify(result)

    except Exception as e:
        logger.error("Error in summarize endpoint: %s", str(e))
        return jsonify({"error": "Internal server error"}), 500


@app.route("/summarize_multi_source", methods=["POST"])
def summarize_multi_source():
    """Multi-source summarization endpoint - HTTP handling only"""
    try:
        # Extract and validate input
        data = request.json
        if not data or "query" not in data:
            return jsonify({"error": "Missing query parameter"}), 400

        query = data["query"].strip()
        max_lines = data.get("max_lines", 30)
        max_articles = data.get("max_articles", 3)
        cost_mode = data.get("cost_mode", "BALANCED")

        if not query:
            return jsonify({"error": "Empty query provided"}), 400

        if max_lines < 5 or max_lines > 100:
            return jsonify({"error": "max_lines must be between 5 and 100"}), 400

        if max_articles < 1 or max_articles > 10:
            return jsonify({"error": "max_articles must be between 1 and 10"}), 400

        if cost_mode not in ["MINIMAL", "BALANCED", "COMPREHENSIVE"]:
            return (
                jsonify(
                    {"error": "cost_mode must be MINIMAL, BALANCED, or COMPREHENSIVE"}
                ),
                400,
            )

        # Delegate to service
        result = summarization_service.summarize_multi_source_with_agents(
            query=query,
            max_articles=max_articles,
            max_lines=max_lines,
            cost_mode=cost_mode,
        )

        # Handle service errors
        if "error" in result:
            if "No articles found" in result["error"]:
                return jsonify(result), 404
            return jsonify(result), 500

        # Format successful response for HTTP
        response = {
            "query": result["query"],
            "summary": result["summary"],
            "metadata": {
                "intent": result.get("intent"),
                "confidence": result.get("confidence"),
                "method": result.get("method"),
                "total_sources": result.get("total_sources", 0),
                "summary_length": result.get("summary_length", 0),
                "summary_lines": result.get("summary_lines", 0),
                "agent_powered": result.get("agent_powered", False),
                "cost_mode": cost_mode,
            },
            "articles": [
                {
                    "title": article["title"],
                    "url": article["url"],
                    "selection_method": article.get("selection_method", "unknown"),
                    "relevance_score": article.get("relevance_score"),
                }
                for article in result.get("articles", [])
            ],
            "usage_stats": result.get("usage_stats", {}),
        }

        logger.info(
            "✅ Multi-source summarization completed successfully for query: '%s'",
            query,
        )
        return jsonify(response), 200

    except Exception as e:
        logger.error("❌ Error in summarize_multi_source endpoint: %s", str(e))
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Use PORT environment variable or default to 7860 for Hugging Face Spaces
    port = int(os.getenv("PORT", "7860"))
    logger.info("🚀 Starting Flask API server on port %d", port)
    app.run(debug=True, host="0.0.0.0", port=port)
