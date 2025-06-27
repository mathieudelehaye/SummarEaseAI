# 🧪 SummarEaseAI Tests & Demos

This directory contains all test scripts and demonstration files for SummarEaseAI. The tests are organized by functionality and complexity.

## 📋 Test Categories

### ✅ **Core Functionality Tests**

#### **`test_single_source.py`**
- **Purpose**: Tests the single-source OpenAI summarization endpoint (`/summarize`)
- **Features**: Analytics validation, endpoint comparison
- **Usage**: `python tests/test_single_source.py`

#### **`test_multi_source_fix.py`**
- **Purpose**: Tests the multi-source agent endpoint (`/summarize_multi_source`)
- **Features**: Agent validation, response format verification
- **Usage**: `python tests/test_multi_source_fix.py`

#### **`test_format_fix.py`**
- **Purpose**: Tests Wikipedia content sanitization (curly braces handling)
- **Features**: Format error prevention, content cleaning validation
- **Usage**: `python tests/test_format_fix.py`

### 🔧 **Debug & Development Tests**

#### **`test_multi_source_debug.py`**
- **Purpose**: Debugging multi-source agent issues
- **Features**: Detailed response inspection, error tracking
- **Usage**: `python tests/test_multi_source_debug.py`

#### **`test_synthesis.py`**
- **Purpose**: Tests article synthesis and combination logic
- **Features**: Multi-article processing validation
- **Usage**: `python tests/test_synthesis.py`

#### **`test_gpu.py`**
- **Purpose**: Tests GPU/CUDA availability for TensorFlow and PyTorch
- **Features**: Hardware detection, performance testing
- **Usage**: `python tests/test_gpu.py`

### 🌐 **Wikipedia Integration Tests**

#### **`test_wikipedia_only.py`**
- **Purpose**: Tests Wikipedia fetching without AI processing
- **Features**: Article retrieval, search functionality, disambiguation
- **Usage**: `python tests/test_wikipedia_only.py`

#### **`test_agentic_workflow.py`**
- **Purpose**: Tests the full agentic workflow with LangChain agents
- **Features**: End-to-end agent testing, workflow validation
- **Usage**: `python tests/test_agentic_workflow.py`

---

## 🎯 **Demo Scripts**

### **`demo_cost_control.py`**
- **Purpose**: Demonstrates cost control features for OpenAI API usage
- **Features**: Rate limiting, usage tracking, cost estimation
- **Usage**: `python tests/demo_cost_control.py`

### **`demo_langchain_agents.py`**
- **Purpose**: Demonstrates LangChain agent capabilities
- **Features**: Agent interactions, query enhancement, article selection
- **Usage**: `python tests/demo_langchain_agents.py`

### **`demo_openai_multisource.py`**
- **Purpose**: Demonstrates multi-source summarization with OpenAI
- **Features**: Multiple article processing, synthesis demonstration
- **Usage**: `python tests/demo_openai_multisource.py`

---

## 🚀 **Quick Test Commands**

```bash
# Test core functionality
python tests/test_single_source.py
python tests/test_multi_source_fix.py

# Test format handling
python tests/test_format_fix.py

# Test system capabilities
python tests/test_gpu.py

# Demo advanced features
python tests/demo_cost_control.py
python tests/demo_langchain_agents.py
```

---

## 📊 **Test Results Interpretation**

### **✅ Success Indicators**
- **Status Code 200**: Endpoint is working
- **Non-empty Analytics**: Response contains valid metrics
- **Agent Powered: True**: Multi-source agents are functioning
- **Wikipedia Pages Listed**: Articles are being fetched correctly

### **❌ Failure Indicators**
- **Status Code 500/404**: Backend or API issues
- **Empty Responses**: Missing data or processing errors
- **Format Errors**: Wikipedia content sanitization problems
- **Missing Analytics**: Response mapping issues

---

## 🔧 **Running Tests from Root Directory**

All tests can be run from the project root directory:

```bash
# From project root
python tests/test_single_source.py
python tests/test_multi_source_fix.py
# etc.
```

---

## 📝 **Adding New Tests**

When adding new test files:

1. **Name convention**: Use `test_*.py` for unit tests, `demo_*.py` for demonstrations
2. **Add documentation**: Update this README with the new test description
3. **Include error handling**: All tests should handle network/API failures gracefully
4. **Add success criteria**: Clear indicators of what constitutes a passing test

---

## 🎯 **Test Coverage**

Current test coverage includes:
- ✅ Single-source summarization
- ✅ Multi-source agent functionality  
- ✅ Wikipedia content sanitization
- ✅ Analytics and response formatting
- ✅ GPU/hardware detection
- ✅ LangChain agent workflows
- ✅ Cost control mechanisms

---

**Note**: Make sure the backend is running (`python backend/api_simple.py`) before executing any tests that make API calls. 