# SummarEaseAI Backend - Flask API with PyTorch & Hugging Face
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files and models
COPY backend/api.py backend/
COPY backend/hf_summarizer.py backend/
COPY backend/summarizer.py backend/
COPY utils/langchain_agents.py utils/
COPY utils/multi_source_agent.py utils/
COPY utils/openai_query_generator.py utils/
COPY utils/wikipedia_fetcher.py utils/
COPY ml_models/bert_classifier.py ml_models/
COPY ml_models/intent_classifier.py ml_models/
COPY ml_models/bert_gpu_model/ ml_models/bert_gpu_model/

# Set Python path
ENV PYTHONPATH=/app

# Create directory for models
RUN mkdir -p /app/ml_models/saved_models

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Set environment variables
ENV FLASK_APP=backend/api.py
ENV FLASK_ENV=production

# Run the Flask application
CMD ["python", "backend/api.py"] 