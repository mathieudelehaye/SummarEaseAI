# SummarEaseAI - Microservices Deployment Guide

## 🏗️ Architecture Overview

SummarEaseAI is deployed as **two microservices** that both **scale to 0** on Azure:

1. **Backend Service** (Flask API) - Port 5000
   - **Local**: TensorFlow with DirectML GPU acceleration (Windows)
   - **Azure**: CPU-optimized TensorFlow (Linux containers)
   - AI/ML processing with automatic CPU/GPU detection
   - Wikipedia content fetching
   - OpenAI integration
   - Scales: 0-5 instances

2. **Frontend Service** (Streamlit) - Port 8501
   - User interface
   - Real-time API communication
   - Scales: 0-3 instances

## 🖥️ CPU vs GPU Architecture

### Local Development (Windows)
- **BERT Model**: DirectML GPU acceleration (RTX 4070)
- **TensorFlow**: Standard `tensorflow` with DirectML support
- **Performance**: ~200ms inference with GPU acceleration

### Azure Deployment (Linux)
- **BERT Model**: Same model, CPU-only execution
- **TensorFlow**: Standard `tensorflow` (CPU automatically used)
- **Performance**: ~500-1000ms inference with CPU optimization
- **No fallback needed**: Full BERT model runs on CPU

### Automatic Environment Detection
The backend automatically detects Azure deployment and forces CPU execution:
```python
# In bert_gpu_classifier.py
def _setup_gpu(self):
    if os.getenv('AZURE_DEPLOYMENT', 'false').lower() == 'true':
        # Force CPU-only execution for Azure
        tf.config.set_visible_devices([], 'GPU')
        logger.info("🖥️ Azure deployment detected - using CPU-only execution")
        return
    
    # Use GPU for local development
    physical_devices = tf.config.list_physical_devices('GPU')
    # ... GPU setup code
```

## 🚀 Key Benefits

### ✅ Same Model, Different Execution
- **Local**: GPU-accelerated inference
- **Azure**: CPU-only inference  
- **No separate models**: Same BERT model works on both
- **No keyword fallback**: Full neural network inference everywhere

### ✅ Standard TensorFlow
- **Package**: `tensorflow==2.13.0` (not tensorflow-gpu or tensorflow-cpu)
- **Automatic**: TensorFlow uses available hardware (GPU locally, CPU on Azure)
- **Simple**: No complex environment detection or separate classifiers

## 🚀 Quick Start

### Local Development
```bash
# 1. Clone and setup
git clone <your-repo>
cd SummarEaseAI

# 2. Create environment file
cp env.template .env
# Edit .env with your API keys

# 3. Start both services (uses GPU acceleration on Windows)
./deploy.sh local
```

### Azure Deployment
```bash
# 1. Setup Azure credentials
az login

# 2. Configure deployment
cp terraform/terraform.tfvars.example terraform/terraform.tfvars
# Edit terraform.tfvars with your values

# 3. Deploy to Azure (automatically uses CPU optimization)
./deploy.sh deploy
```

## 📋 Prerequisites

### Required Tools
- **Docker** (20.10+)
- **Terraform** (1.0+)
- **Azure CLI** (2.50+)
- **Bash** (for deployment script)

### Azure Setup
1. **Azure Subscription** with Container Apps enabled
2. **Service Principal** with Contributor role:
   ```bash
   az ad sp create-for-rbac --name "summarease-sp" --role contributor
   ```
3. **Resource Group** permissions

## 🐳 Docker Configuration

### Backend Dockerfile
```dockerfile
# Location: backend/Dockerfile
FROM python:3.11-slim
# CPU-optimized TensorFlow for Azure
# Flask API with health checks
# Automatic environment detection
# Scales to 0 capability
```

### Frontend Dockerfile
```dockerfile
# Location: frontend/Dockerfile
FROM python:3.11-slim
# Streamlit web interface
# Health checks for Container Apps
# Scales to 0 capability
```

## 🏗️ Project Structure

```
SummarEaseAI/
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── api.py
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py
│   └── static/
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── docker-compose.yml
└── deploy.sh
```

## 🔧 Local Development

### Using Docker Compose
```bash
# Start both services
docker-compose up --build

# Access services
Frontend: http://localhost:8501
Backend:  http://localhost:5000

# Check if GPU acceleration is working
curl http://localhost:5000/status
```

### Environment Variables
```bash
# Required in .env file
OPENAI_API_KEY=your_openai_key_here
FLASK_ENV=development
TF_CPP_MIN_LOG_LEVEL=3
# AZURE_DEPLOYMENT=false (automatically detected)
```

## ☁️ Azure Deployment

### Terraform Configuration

The Terraform setup creates:
- **Resource Group**
- **Container Registry** (ACR)
- **Log Analytics Workspace**
- **Container Apps Environment**
- **2 Container Apps** (Backend + Frontend)

### Scale-to-Zero Configuration
```hcl
# Both services scale to 0 when not in use
template {
  min_replicas = 0  # Scale to zero
  max_replicas = 5  # Scale up under load
  
  # CPU-optimized resources
  resources {
    cpu    = 1.0
    memory = "2Gi"
  }
}
```

## 🔄 Model Performance Comparison

### Local Development (GPU)
```
Environment: Windows + DirectML
Model: GPU-accelerated BERT
Inference: ~200ms
Accuracy: High (full BERT model)
```

### Azure Deployment (CPU)
```
Environment: Linux + CPU-only
Model: CPU-optimized BERT
Inference: ~500-1000ms
Accuracy: High (same BERT model, slower)
Fallback: Keyword-based classification
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Failures on Azure**
   ```bash
   # Expected: CPU models may take longer to load
   # Solution: Health checks allow 30s startup time
   # Fallback: Keyword-based classification works immediately
   ```

2. **Performance Differences**
   ```bash
   # Local: GPU acceleration = fast inference
   # Azure: CPU inference = slower but still functional
   # Solution: This is expected and acceptable for most use cases
   ```

3. **Memory Issues**
   ```bash
   # Solution: Backend allocated 2Gi memory for TensorFlow models
   # Increase if needed in terraform/main.tf
   ```

## 📊 Performance Expectations

### Local Development (GPU)
- **Backend Cold Start**: 5-10 seconds
- **BERT Inference**: 200ms
- **Model Loading**: 3-5 seconds

### Azure Deployment (CPU)
- **Backend Cold Start**: 15-30 seconds
- **BERT Inference**: 500-1000ms
- **Model Loading**: 10-20 seconds
- **Fallback Response**: 50ms (keyword-based)

### Scaling Behavior
```
Users: 0     → Replicas: 0 (scaled to zero)
Users: 1-10  → Replicas: 1 (CPU inference)
Users: 11-50 → Replicas: 2-3 (load distributed)
Users: 50+   → Replicas: 4-5 (max scaling)
```

## 💰 Cost Optimization

### Scale-to-Zero Benefits
- **Backend**: Scales to 0 when no API calls
- **Frontend**: Scales to 0 when no users
- **Cold Start**: 15-30 seconds (acceptable for AI workloads)
- **Cost**: Pay only for actual CPU usage (no GPU costs)

### Resource Efficiency
- **CPU-only**: No expensive GPU instances required
- **Memory**: 2Gi sufficient for TensorFlow CPU inference
- **Fallback**: Instant keyword classification if models fail

---

**Result**: Automatic CPU/GPU detection with optimized deployment for both local development and Azure production! 🚀 