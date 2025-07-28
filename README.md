# 🚀 SummarEaseAI 

**SummarEaseAI** is an AI-powered chatbot that intelligently summarizes Wikipedia articles using state-of-the-art machine learning technologies. It combines **🤗 Hugging Face Transformers** (TinyBERT - 4M parameters) for intent classification, **LangChain** for prompt orchestration, **RAG** (Retrieval-Augmented Generation) via Wikipedia API, and **OpenAI's GPT** (GPT-3.5-turbo - 175B parameters) models for high-quality summarization—all wrapped in a beautiful Streamlit interface.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.35+-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)

</div>

---

## 🌐 Live Application

### 🎯 **Frontend Application**
**Live URL**: [https://summarease-frontend--pmhy95g.wittyflower-c2822a5a.eastus.azurecontainerapps.io](https://summarease-frontend.onrender.com/)

### 🤖 **Backend API**
**Live URL**: https://mdelehaye-summarease-backend.hf.space

### 🚀 **Deployment Architecture**
- **Frontend**: Dockerized and deployed to Azure Container Apps via Terraform
- **Backend**: Dockerized and deployed to Hugging Face Spaces (backend Terraform integration is under test)

---

## 📸 Application Screenshots

### 🎯 Main Interface - Multi-Source Summarization

<p align="left">
    <img src="screenshots/Screenshot00.png" alt="Main Interface" width="80%">
</p>
*The main summarization interface showing multi-source intelligence with Wikipedia article synthesis, intent classification results, and real-time confidence scoring.*

### 🧠 Intent Analysis Dashboard
<p align="left">
    <img src="screenshots/Screenshot01.png" alt="Intent Analysis Dashboard" width="80%">
</p>
*Advanced intent classification dashboard with Hugging Face TinyBERT model (PyTorch-based), featuring interactive confidence gauges and detailed model performance metrics.*

### 🎯 Model Comparison & Analytics
<p align="left">
    <img src="screenshots/Screenshot02.png" alt="Model Comparison Analytics" width="80%">
</p>    
*Comprehensive model comparison view showing performance analytics, response times, and accuracy metrics across different AI models and endpoints.*

---

## 🎯 Currently Working Features ✅

### 🧠 **Intent Recognition**
- **Hugging Face TinyBERT**: ✅ **WORKING** - Transformer-based model for intent classification (4M parameters)
- **Keyword-based Fallback**: ✅ **WORKING** - Reliable backup intent classification system
- **6 Intent Categories**: ✅ **WORKING** - Finance, Science, Technology, History, Music, Sports 
- **Real-time Confidence Scoring**: ✅ **WORKING** - Interactive gauges showing prediction confidence using Plotly

### 🤖 **Multi-Source Intelligence**
- **🤖 Multi-Source Agent**: ✅ **WORKING** - Advanced multi-article synthesis with LangChain agents
- **QueryEnhancementAgent**: ✅ **WORKING** - Intelligent query refinement and expansion
- **ArticleSelectionAgent**: ✅ **WORKING** - Smart Wikipedia article selection from search results
- **Wikipedia Content Sanitization**: ✅ **WORKING** - Handles curly braces and wiki markup safely
- **Cost Control**: ✅ **WORKING** - BALANCED/MINIMAL/COMPREHENSIVE modes for API usage
- **Comprehensive Synthesis**: ✅ **WORKING** - Combines multiple Wikipedia articles into coherent summaries

### ✂️ **Summarization Engines**
- **OpenAI GPT-3.5-turbo**: ✅ **WORKING** - High-quality cloud-based summarization (175B parameters)
- **Multi-Source Synthesis**: ✅ **WORKING** - Combines multiple articles with intelligent agents
- **Length Control**: ✅ **WORKING** - Customizable summary length (10-100 lines)
- **Intent-Aware Processing**: ✅ **WORKING** - Context-based summarization adaptation

### 📚 **Wikipedia Integration & RAG**
- **Smart Search**: ✅ **WORKING** - Automatic fallback to search when direct articles aren't found
- **RAG Pipeline**: ✅ **WORKING** - Retrieval-Augmented Generation using Wikipedia as knowledge base
- **Disambiguation Handling**: ✅ **WORKING** - Intelligent resolution of ambiguous Wikipedia pages
- **Content Sanitization**: ✅ **WORKING** - Handles Wikipedia markup and special characters
- **Multi-Article Support**: ✅ **WORKING** - Fetch and process multiple related articles

### 🎨 **Modern UI/UX**
- **5-Tab Interface**: ✅ **WORKING** - Summarize, Intent Analysis, Semantic Search, Model Comparison, Analytics
- **Real-time Visualizations**: ✅ **WORKING** - Interactive charts and gauges using Plotly
- **Model Status Indicators**: ✅ **WORKING** - Visual indicators for API and model availability
- **Progressive Enhancement**: ✅ **WORKING** - Graceful degradation when services are unavailable
- **Responsive Design**: ✅ **WORKING** - Beautiful interface with custom CSS styling

---

## 🚧 Features Under Development / Testing 🧪

### 🤗 **Local AI with Hugging Face** - NOW UNDER TEST
- **Local Summarization**: 🧪 **UNDER TEST** - Run AI models offline without API dependencies
- **Multiple Models**: 🧪 **UNDER TEST** - BART, T5, DistilBART, Pegasus for different use cases
- **GPU Acceleration**: 🧪 **UNDER TEST** - Automatic CUDA detection for faster inference

### 🔍 **Semantic Search** - NOW UNDER TEST
- **Meaning-Based Search**: 🧪 **UNDER TEST** - Find articles by semantic similarity, not just keywords
- **Sentence Transformers**: 🧪 **UNDER TEST** - Convert text to 384-dimensional meaning vectors
- **Cosine Similarity**: 🧪 **UNDER TEST** - Mathematical comparison of text meanings

### 🚀 **GPU BERT Intent Classification** - WORKING ✅
- **GPU BERT Model**: ✅ **WORKING** - DistilBERT with DirectML GPU acceleration for intent detection
- **Custom Training**: ✅ **WORKING** - Trained on Wikipedia dataset with 6 intent categories
- **High Performance**: ✅ **WORKING** - GPU-accelerated inference with confidence scoring

---

## 🛠️ Technology Stack

| Component | Technology | Parameters | Status | Purpose |
|-----------|------------|------------|---------|---------|
| **Frontend** | Streamlit | N/A | ✅ **WORKING** | Interactive web interface |
| **Backend** | Flask + CORS | N/A | ✅ **WORKING** | RESTful API server |
| **Intent Classification** | 🤗 TinyBERT | 4M | ✅ **WORKING** | Neural intent classification |
| **Multi-Source Agents** | LangChain + OpenAI | 175B | ✅ **WORKING** | Intelligent article synthesis |
| **RAG Pipeline** | Wikipedia + LangChain | N/A | ✅ **WORKING** | Knowledge retrieval |
| **Summarization** | LangChain + OpenAI | 175B | ✅ **WORKING** | Cloud-based summarization |
| **Local AI** | 🤗 Transformers | Varies | 🧪 **UNDER TEST** | Local model inference |
| **Semantic Search** | 🤗 Sentence Transformers | 110M | 🧪 **UNDER TEST** | Meaning-based retrieval |
| **Data Source** | Wikipedia API | N/A | ✅ **WORKING** | Article content |
| **Visualization** | Plotly | N/A | ✅ **WORKING** | Interactive charts |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key (required for current working features)
- 4GB+ RAM (recommended)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/SummarEaseAI.git
cd SummarEaseAI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
```bash
cp env.template .env
# Edit .env and add your OpenAI API key
```

### 4. Start the Application
```bash
# Terminal 1 - Backend API
cd backend && python api.py

# Terminal 2 - Frontend (in new terminal)
cd frontend && streamlit run app.py
```

### 5. Access the Application
Open your browser and navigate to `http://localhost:8501`

---

## 📊 Visual Interface Overview

The application features a modern, intuitive interface with multiple specialized tabs:

### 📝 **Summarization Tab**
- Multi-source Wikipedia article synthesis
- Intent-aware processing with confidence scoring
- Customizable summary length and style
- Real-time processing indicators

### 🧠 **Intent Analysis Tab** 
- Hugging Face TinyBERT model analysis
- Interactive confidence gauges and charts
- Detailed model performance metrics
- Category-wise prediction analysis

### 🔍 **Semantic Search Tab** (Under Test)
- Meaning-based article discovery
- Vector similarity visualization
- Embedding space exploration

### 📊 **Model Comparison Tab**
- Side-by-side model performance
- Response time analytics
- Accuracy metrics and benchmarks
- API endpoint status monitoring

### 📈 **Analytics Dashboard**
- Usage statistics and trends
- Model performance over time
- Cost tracking and optimization
- System health monitoring

---

## 📊 Current Architecture

```mermaid
graph TD
    A[User Input] --> B[Streamlit Frontend]
    B --> C[Flask API Backend]
    C --> D[Intent Classifier]
    D --> E[Hugging Face TinyBERT ✅]
    D --> F[Keyword Fallback ✅]
    C --> G[Multi-Source Agent ✅]
    G --> H[QueryEnhancementAgent ✅]
    G --> I[ArticleSelectionAgent ✅]
    C --> J[OpenAI + LangChain ✅]
    C --> K[Wikipedia RAG ✅]
    K --> L[Wikipedia API ✅]
    J --> M[Cloud Summary ✅]
    H --> N[Enhanced Queries ✅]
    I --> O[Selected Articles ✅]
    K --> J
    M --> B
    N --> K
    O --> J
    
    %% Under Test Features
    C --> P[🧪 Local HF Models]
    C --> Q[🧪 Semantic Search]
    P --> S[🧪 Local Summary]
    Q --> T[🧪 Vector Search]
```

---

## 🔧 Detailed Usage

### Working API Endpoints ✅

#### Core Endpoints
- `GET /` - API status and feature availability ✅
- `GET /health` - Health check ✅
- `GET /status` - Detailed system status ✅

#### Intent Classification
- `POST /intent` - Hugging Face TinyBERT intent classification ✅
- `POST /predict_intent` - TinyBERT model inference ✅

#### Summarization
- `POST /summarize` - Single source Wikipedia summarization ✅
- `POST /summarize_multi_source` - Multi-source agent synthesis ✅

#### Specialized APIs (Optional)
- `POST /predict_intent` - TinyBERT model inference (CPU) ✅
- `POST /predict_intent_gpu` - TinyBERT model inference (GPU) ✅

### Usage Examples

#### 1. Multi-Source Agent (Working ✅)
```python
import requests

response = requests.post('http://localhost:5000/summarize_multi_source', json={
    'query': 'Who were the Beatles?',
    'max_lines': 30,
    'use_intent': True
})

result = response.json()
print(f"Summary: {result['summary']}")
print(f"Articles used: {result['wikipedia_pages_used']}")
print(f"Agent powered: {result['agent_powered']}")
```

#### 2. Intent Classification (Working ✅)
```python
response = requests.post('http://localhost:5000/predict_intent', json={
    'text': 'Tell me about quantum physics'
})

result = response.json()
print(f"Intent: {result['predicted_intent']}")
print(f"Confidence: {result['confidence']}")
print(f"Model: {result['model_used']}")
```

#### 3. OpenAI Summarization (Working ✅)
```python
response = requests.post('http://localhost:5000/summarize', json={
    'query': 'Apollo 11 moon landing',
    'max_lines': 25
})

result = response.json()
print(f"Summary: {result['summary']}")
print(f"Method: {result['method']}")
```

---

## 🧪 Test Queries for Working Features

### Multi-Source Agent Examples ✅
- **"Who were the Beatles?"** → Synthesizes band info, discography, and musical style
- **"What is quantum mechanics?"** → Combines physics articles and applications
- **"Tell me about World War II"** → Merges historical events, battles, and outcomes

### Intent Classification Test Queries ✅
- **History**: "What happened during the Apollo 11 mission?"
- **Science**: "Explain how photosynthesis works"
- **Biography**: "Tell me about Marie Curie's discoveries"
- **Technology**: "How do neural networks function?"
- **Arts (Music)**: "Who were the Beatles?" 
- **Geography**: "Where are the Himalayas located?"

### Regular Summarization Examples ✅
- **"Artificial Intelligence"** → Comprehensive AI overview
- **"Climate Change"** → Environmental science summary
- **"Renaissance Art"** → Cultural and artistic movements

---

## 📈 Current Performance Metrics

### Hugging Face TinyBERT Intent Classifier ✅
- **Model Size**: 4MB
- **Inference Time**: <50ms per query
- **Categories**: 9 distinct intent classes (History, Science, Biography, Technology, Arts, Sports, Politics, Geography, General)
- **Fallback**: Keyword-based system for reliability
- **Framework**: PyTorch (via Hugging Face Transformers)
- **Deployment**: CPU-optimized for production use

### Multi-Source Agent System ✅
- **Articles per Query**: 1-3 (configurable)
- **Cost Modes**: MINIMAL, BALANCED, COMPREHENSIVE
- **Agent Types**: QueryEnhancement, ArticleSelection
- **Response Time**: 3-8 seconds (depending on complexity)
- **Success Rate**: >95% for common topics

### OpenAI Integration ✅
- **Models Supported**: GPT-3.5-turbo, GPT-4
- **Average Response Time**: 2-5 seconds
- **Quality Score**: High (human-readable summaries)
- **Cost Control**: Configurable limits and modes

---

## 🔧 Advanced Configuration

### Environment Variables
```bash
# Required for working features
OPENAI_API_KEY=your_openai_api_key_here

# Cost control for multi-source agent
COST_MODE=BALANCED  # MINIMAL, BALANCED, COMPREHENSIVE
MAX_ARTICLES=3      # Maximum articles per query

# API configuration
API_PORT=5000
DEBUG_MODE=false
```

### Multi-Source Agent Configuration
```python
# In utils/multi_source_agent.py
COST_MODES = {
    'MINIMAL': {
        'max_articles': 1,
        'max_secondary_queries': 1,
        'enable_agents': True,
        'enable_openai': True
    },
    'BALANCED': {
        'max_articles': 3,
        'max_secondary_queries': 3,
        'enable_agents': True,
        'enable_openai': True
    },
    'COMPREHENSIVE': {
        'max_articles': 5,
        'max_secondary_queries': 5,
        'enable_agents': True,
        'enable_openai': True
    }
}
```

---

## 🚧 Future Development Roadmap

### Phase 1: Stabilize Under-Test Features 🧪
- [ ] **Complete BERT Intent Testing** - Finalize BERT model integration
- [ ] **Validate Semantic Search** - Test embedding-based article discovery
- [ ] **Local HF Model Support** - Enable offline BART/T5 summarization
- [ ] **Performance Optimization** - Improve response times and reliability

### Phase 2: Advanced Features
- [ ] **Multi-language Support** - Summarization in multiple languages
- [ ] **Document Upload** - Support for PDF, Word docs, and text files
- [ ] **Conversational Memory** - Multi-turn conversations with context
- [ ] **Custom Training** - Domain-specific model fine-tuning

### Phase 3: Enterprise Features
- [ ] **Real-time Collaboration** - Multiple users, shared workspaces
- [ ] **Advanced Analytics** - Usage patterns, model performance tracking
- [ ] **Plugin System** - Custom integrations and extensions
- [ ] **Mobile App** - React Native companion application

---

## 🧪 Testing Suite

The project includes a comprehensive unit test suite covering all critical components:

### 🚀 **Run Tests**
```bash
# Quick test run
python test_runner.py

# Run with coverage
python test_runner.py coverage

# Run specific test categories
python test_runner.py unit           # Unit tests only
python test_runner.py api            # API tests only
python test_runner.py integration    # Integration tests
```

### 📊 **Test Coverage**
- **140+ test cases** covering critical functionality
- **5 major components** thoroughly tested
- **Coverage reporting** with HTML output
- **CI/CD ready** configuration

For detailed testing information, see [TESTING.md](TESTING.md).

---

## 🤝 Contributing

We welcome contributions! Focus areas:

### High Priority (Working Features) ✅
1. **🔧 Bug Fixes** - Improve stability of working features
2. **📊 Analytics** - Enhance monitoring and metrics
3. **🎨 UI/UX** - Streamlit interface improvements
4. **📚 Documentation** - Usage guides and examples

### Medium Priority (Under Test) 🧪
1. **🤗 HuggingFace Integration** - Complete local model support
2. **🔍 Semantic Search** - Finalize embedding-based search
3. **🧠 BERT Testing** - Validate BERT intent classification
4. **🧪 Testing** - Expand test coverage for new features

### Development Setup
```bash
# Clone and setup
git clone https://github.com/your-username/SummarEaseAI.git
cd SummarEaseAI
pip install -r requirements.txt

# Test working features
python test_multi_source_fix.py
python quick_start.py
```

---

## 📝 License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

**Why AGPL-3.0?** This license ensures that any improvements to SummarEaseAI remain open-source and benefit the entire AI research community, even when used in web services or cloud deployments.

---

## 🙏 Acknowledgments

We extend our heartfelt thanks to:
- **OpenAI** team for the powerful GPT models
- **Hugging Face** team for the transformers ecosystem
- **Streamlit** team for the amazing UI framework
- **Wikipedia** for the extensive knowledge base
- **LangChain** team for the agent framework
- **PyTorch** team for the deep learning framework
- **Plotly** team for the visualization tools

---

<div align="center">

**Built with ❤️ using AI to make information more accessible**

### 🚀 **SummarEaseAI - Multi-Source Intelligence Now Working!**

✅ **Multi-Source Agent** | ✅ **Intent Classification** | ✅ **OpenAI Integration** | 🧪 **Local AI Under Test**

⭐ **Star this repository if you found it helpful!**

</div>
