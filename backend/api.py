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
import wikipedia
import openai
from dotenv import load_dotenv
from utils.wikipedia_fetcher import search_and_fetch_article_agentic_simple, fetch_article_with_conversion_info

# Multi-source agent imports
from utils.multi_source_agent import MultiSourceAgent
from utils.langchain_agents import WikipediaAgentSystem
from utils.openai_query_generator import OpenAIQueryGenerator

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

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

# Initialize Multi-Source Agent with shared BERT model
logger.info("🔄 Initializing Multi-Source Agent...")
multi_source_agent = MultiSourceAgent(cost_mode="BALANCED")
multi_source_agent.bert_classifier = bert_classifier  # Share the loaded model
multi_source_agent.bert_model_loaded = bert_model_loaded  # Share the loaded state

if multi_source_agent.bert_model_loaded:
    logger.info("✅ Multi-Source Agent initialized successfully")
else:
    logger.error("❌ Failed to initialize Multi-Source Agent!")

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
        'features': {
            'bert_model': bert_model_loaded,
            'openai_summarization': bool(openai.api_key),
            'wikipedia_fetching': True
        },
        'endpoints': [
            '/status',
            '/health',
            '/intent_bert',
            '/summarize',
            '/summarize_multi_source'
        ]
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'backend': 'running',
        'bert_model': bert_model_loaded,
        'bert_categories': BERT_CATEGORIES,
        'openai_available': bool(openai.api_key)
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

@app.route('/summarize', methods=['POST'])
def summarize():
    """Summarize a topic using OpenAI"""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query = data['query'].strip()
        max_lines = data.get('max_lines', 30)
        
        if not query:
            return jsonify({'error': 'Empty query provided'}), 400
        
        # Log the summarization request
        logger.info("=" * 80)
        logger.info("📝 SINGLE SOURCE SUMMARIZATION REQUEST")
        logger.info("=" * 80)
        logger.info(f"Query: '{query}'")
        logger.info(f"Max lines: {max_lines}")
        
        # Get Wikipedia content
        try:
            # Use enhanced article fetching with logging
            article_content, processed_query, was_converted = fetch_article_with_conversion_info(query)
            
            if not article_content:
                logger.error(f"❌ No Wikipedia content found for query: '{query}'")
                return jsonify({
                    'error': 'No Wikipedia content found',
                    'message': f"Could not find Wikipedia article for: {query}"
                }), 404

            # Truncate content to avoid token limit (roughly 4 chars per token)
            max_tokens = 14000  # Leave room for system message and completion
            max_chars = max_tokens * 4
            if len(article_content) > max_chars:
                logger.info(f"📄 Truncating content from {len(article_content)} to {max_chars} characters to fit token limit")
                article_content = article_content[:max_chars]
            
            # Log article details
            logger.info("=" * 80)
            logger.info("📄 ARTICLE DETAILS")
            logger.info("=" * 80)
            logger.info(f"Original query: '{query}'")
            logger.info(f"Processed query: '{processed_query}'")
            logger.info(f"Query was converted: {was_converted}")
            logger.info(f"Article length: {len(article_content)} characters")
            
            # Get page details for response
            page = wikipedia.page(processed_query)
            article_info = {
                'title': page.title,
                'url': page.url,
                'length': len(article_content)
            }
            
            # Use OpenAI for summarization
            logger.info("=" * 80)
            logger.info("🤖 OPENAI SUMMARIZATION")
            logger.info("=" * 80)
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates clear, accurate summaries of Wikipedia articles."},
                    {"role": "user", "content": f"Please summarize this Wikipedia article in {max_lines} lines or less:\n\n{article_content}"}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info(f"✅ Summary generated ({len(summary.splitlines())} lines)")
            
            return jsonify({
                'query': query,
                'processed_query': processed_query,
                'was_converted': was_converted,
                'summary': summary,
                'method': 'openai',
                'model': 'gpt-3.5-turbo',
                'article': article_info,
                'summary_length': len(summary)
            })
            
        except Exception as e:
            logger.error(f"Error in Wikipedia fetching: {str(e)}")
            return jsonify({
                'error': 'Wikipedia error',
                'message': str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/summarize_multi_source', methods=['POST'])
def summarize_multi_source():
    """Summarize a topic using multiple Wikipedia articles and OpenAI with LangChain agents"""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query = data['query'].strip()
        max_lines = data.get('max_lines', 30)
        use_intent = data.get('use_intent', True)
        
        if not query:
            return jsonify({'error': 'Empty query provided'}), 400
            
        # Log the multi-source request
        logger.info("=" * 80)
        logger.info("🚀 MULTI-SOURCE SUMMARIZATION REQUEST")
        logger.info("=" * 80)
        logger.info(f"Query: '{query}'")
        logger.info(f"Max lines: {max_lines}")
        logger.info(f"Use intent: {use_intent}")
        
        # Get intent if requested
        intent = None
        confidence = 0.0
        if use_intent and bert_model_loaded:
            intent, confidence = bert_classifier.predict(query)
            logger.info(f"🎯 Intent detection: {intent} (confidence: {confidence:.3f})")
        
        # Use Multi-Source Agent for intelligent article gathering
        logger.info("=" * 80)
        logger.info("🤖 MULTI-SOURCE AGENT PROCESSING")
        logger.info("=" * 80)
        
        result = multi_source_agent.run_multi_source_search_with_agents(
            query=query,
            max_articles=3  # Get up to 3 relevant articles
        )
        
        if not result or not result.get('summaries'):
            logger.error(f"❌ No articles found for query: '{query}' with intent: {intent}")
            return jsonify({
                'error': 'No articles found',
                'message': f"Could not find relevant Wikipedia articles for: {query}"
            }), 404

        # Log search strategy and results
        logger.info(f"📊 Search Strategy: {result.get('strategy', {}).get('search_strategy', 'N/A')}")
        logger.info(f"🎯 Synthesis Focus: {result.get('strategy', {}).get('synthesis_focus', 'N/A')}")
        logger.info(f"🔍 OpenAI Used: {result.get('agent_powered', False)}")
        
        # Log article details
        logger.info("\n📚 Articles Found:")
        articles_list = []
        combined_content = ""
        total_chars = 0
        
        for i, article in enumerate(result['summaries'], 1):
            logger.info(f"\n📄 Article {i}:")
            logger.info(f"Title: {article['title']}")
            logger.info(f"URL: {article['url']}")
            logger.info(f"Summary length: {len(article['summary'])} characters")
            logger.info(f"Relevance score: {article.get('relevance_score', 'N/A')}")
            logger.info(f"Search method: {article.get('selection_method', 'direct')}")
            
            articles_list.append({
                'title': article['title'],
                'url': article['url'],
                'length': len(article['summary']),
                'relevance_score': article.get('relevance_score', None),
                'search_method': article.get('selection_method', 'direct'),
                'is_primary': article.get('is_primary', i == 1)
            })
            
            # Add article content with clear separation
            if i > 1:
                combined_content += f"\n\nAdditional information from '{article['title']}':\n"
            combined_content += article['summary']
            total_chars += len(article['summary'])

        # Truncate if needed (roughly 4 chars per token)
        max_tokens = 14000  # Leave room for system message and completion
        max_chars = max_tokens * 4
        
        if total_chars > max_chars:
            logger.info(f"\n✂️ Content exceeds limit. Truncating from {total_chars} to {max_chars} characters")
            combined_content = combined_content[:max_chars]
            
            # Update article lengths after truncation
            truncation_ratio = max_chars / total_chars
            for article in articles_list:
                article['length'] = int(article['length'] * truncation_ratio)
                logger.info(f"Truncated '{article['title']}' to {article['length']} characters")
        
        # Use OpenAI for final synthesis
        logger.info("\n" + "=" * 80)
        logger.info("🤖 OPENAI SYNTHESIS")
        logger.info("=" * 80)
        
        synthesis_focus = result.get('strategy', {}).get('synthesis_focus', 'comprehensive')
        synthesis_prompt = f"""Create a coherent {max_lines}-line summary from these Wikipedia articles.

Focus on {synthesis_focus} aspects as they relate to the query: "{query}"

Articles provided:
{', '.join(article['title'] for article in articles_list)}

Content:
{combined_content}"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates clear, accurate summaries from multiple Wikipedia articles, focusing on synthesis and connections between sources."},
                {"role": "user", "content": synthesis_prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        summary = response.choices[0].message.content.strip()
        logger.info(f"✅ Multi-source summary generated ({len(summary.splitlines())} lines)")
        
        return jsonify({
            'query': query,
            'summary': summary,
            'intent': intent,
            'confidence': confidence,
            'method': 'langchain_multi_source',
            'model': 'gpt-3.5-turbo',
            'articles': articles_list,
            'total_sources': len(articles_list),
            'total_content_length': len(combined_content),
            'summary_length': len(summary),
            'summary_lines': len(summary.splitlines()),
            'search_strategy': result.get('strategy', {}).get('search_strategy', 'default'),
            'synthesis_focus': synthesis_focus,
            'agent_powered': result.get('agent_powered', True),
            'stats': {
                'articles_found': len(articles_list),
                'total_chars': total_chars,
                'summary_chars': len(summary),
                'summary_lines': len(summary.splitlines()),
                'openai_calls': result.get('cost_tracking', {}).get('openai_calls', 0),
                'wikipedia_calls': result.get('cost_tracking', {}).get('wikipedia_calls', 0),
                'articles_processed': result.get('cost_tracking', {}).get('articles_processed', 0)
            },
            'cost_tracking': result.get('cost_tracking', {}),
            'rate_limits': result.get('rate_limits_applied', {}),
            'wikipedia_pages': result.get('wikipedia_pages_used', [article['title'] for article in articles_list])
        })
            
    except Exception as e:
        logger.error(f"❌ Error in multi-source summarization: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 