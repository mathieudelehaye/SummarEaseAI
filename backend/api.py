# Flask backend API
from flask import Flask, request, jsonify
from utils.wikipedia_fetcher import fetch_article
from backend.summarizer import summarize_article_with_limit

app = Flask(__name__)

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    topic = data.get('topic', '')
    article_text = fetch_article(topic)
    if article_text:
        summary = summarize_article_with_limit(article_text, max_lines=30)
        return jsonify({'summary': summary})
    return jsonify({'error': 'Article not found'}), 404

if __name__ == '__main__':
    app.run(port=5000)
