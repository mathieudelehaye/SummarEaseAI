# Build stage
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY backend/requirements.txt .

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    find /opt/venv -type d -name "__pycache__" -exec rm -r {} + && \
    find /opt/venv -type d -name "tests" -exec rm -r {} + && \
    find /opt/venv -type d -name "examples" -exec rm -r {} + && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type f -name "*.pyo" -delete && \
    find /opt/venv -type f -name "*.pyd" -delete

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy MVC structured backend files (only production-needed files)
COPY backend/__init__.py backend/

# Controllers
COPY backend/controllers/__init__.py backend/controllers/
COPY backend/controllers/request_validation.py backend/controllers/
COPY backend/controllers/summarization_controller.py backend/controllers/

# Models
COPY backend/models/__init__.py backend/models/
# Agents models
COPY backend/models/agents/__init__.py backend/models/agents/
COPY backend/models/agents/article_selection_agent.py backend/models/agents/
COPY backend/models/agents/langchain_agents_model.py backend/models/agents/
COPY backend/models/agents/query_enhancement_agent.py backend/models/agents/
# Intent models
COPY backend/models/intent/__init__.py backend/models/intent/
COPY backend/models/intent/intent_data.py backend/models/intent/
# LLM models
COPY backend/models/llm/__init__.py backend/models/llm/
COPY backend/models/llm/llm_client.py backend/models/llm/
COPY backend/models/llm/openai_summarizer_model.py backend/models/llm/
# Wikipedia models
COPY backend/models/wikipedia/__init__.py backend/models/wikipedia/
COPY backend/models/wikipedia/wikipedia_model.py backend/models/wikipedia/
COPY backend/models/wikipedia/wikipedia_search_tool.py backend/models/wikipedia/

# Services
COPY backend/services/__init__.py backend/services/
COPY backend/services/common_source_summary_service.py backend/services/
COPY backend/services/multi_source_summary_service.py backend/services/
COPY backend/services/single_source_summary_service.py backend/services/
COPY backend/services/query_generation_service.py backend/services/
COPY backend/services/query_processing_service.py backend/services/
COPY backend/services/summarization_workflow_service.py backend/services/

# Templates
COPY backend/templates/main.html backend/templates/

# Views
COPY backend/views/__init__.py backend/views/
COPY backend/views/api_routes.py backend/views/

# ML Models
COPY ml_models/bert_classifier.py ml_models/
COPY ml_models/intent_classifier.py ml_models/

# Create model directory and copy only needed model files
RUN mkdir -p /app/ml_models/bert_gpu_model
COPY ml_models/bert_gpu_model/config.json \
     ml_models/bert_gpu_model/model.safetensors \
     ml_models/bert_gpu_model/tokenizer.json \
     ml_models/bert_gpu_model/vocab.txt \
     ml_models/bert_gpu_model/special_tokens_map.json \
     ml_models/bert_gpu_model/tokenizer_config.json \
     ml_models/bert_gpu_model/label_encoder.pkl \
     /app/ml_models/bert_gpu_model/

# Create writable directories for cache and logs
RUN mkdir -p /tmp/cache /tmp/logs && \
    chmod 777 /tmp/cache /tmp/logs

# Set Python path and environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=backend/views/api_routes.py
ENV FLASK_ENV=production
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV CONTAINER_ENV=true
ENV TRANSFORMERS_CACHE=/tmp/cache
ENV HF_HOME=/tmp/cache

# Clean up unnecessary files
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    find /usr/local -type d -name "__pycache__" -exec rm -r {} + && \
    find /usr/local -type f -name "*.pyc" -delete

# Expose Hugging Face's default port
EXPOSE 7860

# Health check using Hugging Face's port
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the application with Hugging Face's port
CMD ["python", "-m", "backend.views.api_routes"] 