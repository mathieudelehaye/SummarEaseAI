# 🚀 SummarEaseAI
**SummarEaseAI** is an interactive chatbot that summarizes Wikipedia articles using the power of modern AI. It combines TensorFlow for intent recognition, LangChain for prompt orchestration, and OpenAI for high-quality language generation—all wrapped in a user-friendly Streamlit interface.

---

## 🎯 Key Features

- 🧠 **Intent Recognition**  
  Uses a TensorFlow-based neural network to classify the user's query (e.g., history, science, finance).

- 📚 **Wikipedia Integration**  
  Fetches and summarizes content from Wikipedia based on identified intent or user query.

- ✂️ **Concise Summarization**  
  Generates clean summaries using OpenAI, limited to a max number of lines (e.g., 30) for clarity.

- ⚙️ **Extendable Architecture**  
  Built for future integration with authenticated sources like Financial Times.

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit  
- **Backend**: Flask  
- **ML & NLP**: TensorFlow, LangChain, OpenAI API  
- **Data Source**: Wikipedia API

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/SummarEaseAI.git
cd SummarEaseAI
