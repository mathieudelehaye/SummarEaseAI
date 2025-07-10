# SummarEaseAI Backend - Flask API with TensorFlow CPU
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
COPY requirements.txt .
COPY backend/requirements.txt backend_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r backend_requirements.txt

# Copy the entire project (needed for imports)
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Create directory for models
RUN mkdir -p /app/tensorflow_models/bert_gpu_models
RUN mkdir -p /app/saved_model

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Set environment variables
ENV FLASK_APP=backend/api.py
ENV FLASK_ENV=production
ENV TF_CPP_MIN_LOG_LEVEL=3

# Run the Flask application
CMD ["python", "backend/api.py"] 