#!/usr/bin/env python3
"""
SummarEaseAI API Routes - View Layer
Pure HTTP concerns - routing, request/response handling
No business logic - delegates to services
"""

import logging
import os
import sys
import warnings
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from ml_models.bert_intents import BERT_CATEGORIES

from backend.controllers.request_validation import (
    validate_multi_source_request,
    validate_summarize_request,
)
from backend.controllers.summarization_controller import (
    SummarizationMethod,
    get_summarization_controller,
)

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

# Initialize Flask app
app = Flask(__name__, template_folder=str(Path(__file__).parent.parent / "templates"))
CORS(app)

# Initialize service with proper constant naming
SUMMARIZATION_CONTROLLER = get_summarization_controller()

# BERT special categories for display
SPECIAL_CATEGORIES = ["NO DETECTED"]

# Log initialization status
logger.info("=" * 60)
logger.info("SummarEaseAI API Routes Starting")
logger.info("Repository root: %s", repo_root)
logger.info("Initializing Flask routes...")
logger.info("=" * 60)


@app.route("/")
def home():
    """Main page showing backend status and available endpoints"""
    # Get status from service
    system_status = SUMMARIZATION_CONTROLLER.get_system_status()
    bert_model_loaded = system_status["models"]["bert"]["loaded"]

    return render_template(
        "main.html",
        bert_model_loaded=bert_model_loaded,
        bert_categories=BERT_CATEGORIES,
        special_categories=SPECIAL_CATEGORIES,
    )


@app.route("/status")
def status():
    """System status endpoint - delegates to service"""
    system_status = SUMMARIZATION_CONTROLLER.get_system_status()

    # Format response to match test expectations
    response = {
        "status": "running",
        "features": system_status.get("features", {}),
        "endpoints": [
            "/status",
            "/health",
            "/intent",
            "/intent_bert",
            "/summarize",
            "/summarize_multi_source",
        ],
        "models": system_status.get("models", {}),
        "services": system_status.get("services", {}),
    }

    return jsonify(response)


@app.route("/health")
def health():
    """Health check endpoint - delegates to service"""
    system_status = SUMMARIZATION_CONTROLLER.get_system_status()

    # Format response to match test expectations
    response = {
        "status": "healthy",
        "backend": "running",
        "bert_model": system_status.get("models", {}).get("bert", {}),
        "services": system_status.get("services", {}),
        "timestamp": system_status.get("timestamp", ""),
    }

    return jsonify(response)


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
        result = SUMMARIZATION_CONTROLLER.classify_intent(text)

        # Handle service errors
        if "error" in result:
            if "not loaded" in result["error"]:
                return jsonify(result), 503
            return jsonify(result), 500

        # Format response to match test expectations
        response = {
            "intent": result.get("intent", result.get("predicted_category", "Unknown")),
            "confidence": result.get("confidence", 0.0),
            "model_type": "BERT",
            "timestamp": result.get("timestamp", ""),
            "text": text,
            "model_loaded": result.get("model_loaded", True),
            "categories_available": result.get("categories_available", []),
            "gpu_accelerated": result.get("gpu_accelerated", True),
        }

        # Return successful result
        return jsonify(response)

    except Exception as e:
        logger.error("Error in intent_bert endpoint: %s", str(e))
        return jsonify({"error": "Internal server error"}), 500


@app.route("/summarize", methods=["POST"])
def summarize():
    """Single-source summarization endpoint - HTTP handling only"""
    try:
        # Extract and validate input
        data = request.json
        validation_error = validate_summarize_request(data)
        if validation_error:
            return jsonify(validation_error[0]), validation_error[1]

        query = data["query"].strip()

        # Delegate to service
        result = SUMMARIZATION_CONTROLLER.summarize(
            query, SummarizationMethod.SINGLE_SOURCE
        )

        # Handle service errors
        if "error" in result:
            if "No Wikipedia content found" in result["error"]:
                return jsonify(result), 404
            return jsonify(result), 500

        # Format response to match test expectations
        response = {
            "query": result.get("query", query),
            "summary": result.get("summary", ""),
            "intent": result.get("intent", ""),
            "confidence": result.get("confidence", 0.0),
            "method": result.get("method", "single_source"),
            "total_sources": result.get("total_sources", 1),
            "summary_length": result.get("summary_length", 0),
            "summary_lines": result.get("summary_lines", 0),
        }

        # Return successful result
        return jsonify(response)

    except Exception as e:
        logger.error("Error in summarize endpoint: %s", str(e))
        return jsonify({"error": "Internal server error"}), 500


@app.route("/summarize_multi_source", methods=["POST"])
def summarize_multi_source():
    """Multi-source summarization endpoint - HTTP handling only"""
    try:
        # Extract and validate input
        data = request.json
        validation_error = validate_multi_source_request(data)
        if validation_error:
            return jsonify(validation_error[0]), validation_error[1]

        query = data["query"].strip()

        # Delegate to service
        result = SUMMARIZATION_CONTROLLER.summarize(
            query=query,
            source_type=SummarizationMethod.MULTI_SOURCE,
        )

        # Handle service errors
        if "error" in result:
            if "No articles found" in result["error"]:
                return jsonify(result), 404
            return jsonify(result), 500

        # Extract intent from the result (it might be nested)
        intent_data = result.get("intent", {})
        if isinstance(intent_data, dict):
            intent = intent_data.get("category", "Unknown")
            confidence = intent_data.get("confidence", 0.0)
        else:
            intent = intent_data
            confidence = result.get("confidence", 0.0)

        # Format response to match test expectations
        response = {
            "query": result.get("query", query),
            "summary": result.get("summary", ""),
            "intent": intent,
            "confidence": confidence,
            "method": result.get("method", "multi_source"),
            "total_sources": result.get("total_sources", 0),
            "summary_length": result.get("summary_length", 0),
            "summary_lines": result.get("summary_lines", 0),
            "articles": result.get("articles", []),
            "usage_stats": result.get("usage_stats", {}),
            "cost_tracking": result.get("cost_tracking", {}),
            "wikipedia_pages": result.get("wikipedia_pages", []),
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
