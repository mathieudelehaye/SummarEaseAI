# SummarEaseAI - Enhanced Streamlit Frontend with Hugging Face Integration
import streamlit as st
import requests
import json
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="SummarEaseAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
    }
    .hf-badge {
        background-color: #ff6b35;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .tf-badge {
        background-color: #ff6f00;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:5000"

def check_api_health():
    """Check if the Flask API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_api_status():
    """Get API status information"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def predict_intent(text, model_type='tensorflow'):
    """Predict intent using specified model"""
    try:
        endpoint = '/predict_intent' if model_type == 'tensorflow' else '/predict_intent_bert'
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json={'text': text},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error predicting intent: {str(e)}")
    return None

def compare_models(text):
    """Compare different AI models"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/compare_models",
            json={'text': text},
            timeout=15
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error comparing models: {str(e)}")
    return None

def summarize_article(topic, max_lines=30, use_intent=True, model_type='openai'):
    """Summarize Wikipedia article using specified method"""
    try:
        if model_type == 'openai':
            endpoint = '/summarize'
        elif model_type == 'multi_source':
            endpoint = '/summarize_multi_source'
        else:
            endpoint = '/summarize_local'
        payload = {
            'query': topic,  # Changed from 'topic' to 'query' to match backend
            'max_lines': max_lines,
            'use_intent': use_intent
        }
        
        if model_type == 'huggingface':
            payload['model'] = 'facebook/bart-large-cnn'  # Default HF model
        
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload,
            timeout=60  # Increased timeout for local models
        )
        
        return response.json(), response.status_code
    except Exception as e:
        return {'error': f"Request failed: {str(e)}"}, 500

def semantic_search(query, max_results=5):
    """Perform semantic search"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/semantic_search",
            json={'query': query, 'max_results': max_results},
            timeout=15
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error in semantic search: {str(e)}")
    return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 SummarEaseAI v2.0</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Wikipedia Summarization with <span class="hf-badge">🤗 Hugging Face</span> & <span class="tf-badge">TensorFlow</span></p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Configuration")
        
        # API Status
        st.subheader("API Status")
        api_healthy = check_api_health()
        if api_healthy:
            st.success("✅ API is running")
            api_status = get_api_status()
            if api_status and 'features' in api_status:
                features = api_status['features']
                st.info(f"🧠 TensorFlow: {'✅' if features.get('tensorflow_intent_model') else '❌'}")
                st.info(f"🤗 BERT: {'✅' if features.get('bert_intent_model') else '❌'}")
                st.info(f"🤗 HuggingFace: {'✅' if features.get('huggingface_summarization') else '❌'}")
                st.info(f"🔍 Semantic Search: {'✅' if features.get('semantic_search') else '❌'}")
        else:
            st.error("❌ API is not responding")
            st.markdown("**To start the API:**")
            st.code("cd backend && python api.py", language="bash")
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        max_lines = st.slider("Summary Length (lines)", 10, 100, 30, 5)
        use_intent = st.checkbox("Use Intent Classification", value=True)
        show_analytics = st.checkbox("Show Analytics", value=True)
        
        st.divider()
        
        # Model Selection
        st.subheader("🧠 AI Models")
        st.markdown("**Summarization:**")
        summary_model = st.radio(
            "Choose summarization method:",
            ["OpenAI + LangChain", "🤗 Local Hugging Face"],
            help="OpenAI provides high quality but requires API key. Hugging Face runs locally."
        )
        
        st.markdown("**Intent Classification:**")
        intent_model = st.radio(
            "Choose intent model:",
            ["TensorFlow LSTM", "🤗 BERT Transformer"],
            help="Compare custom TensorFlow model vs pre-trained BERT."
        )
        
        st.divider()
        
        # About
        st.subheader("📖 About v2.0")
        st.markdown("""
        **New Features:**
        - 🤗 **Hugging Face Integration**
        - 🔍 **Semantic Search** with embeddings
        - 🧠 **BERT Intent Classification**
        - 📊 **Model Comparison**
        - 🚀 **Local AI Models**
        
        **Tech Stack:**
        - **Frontend**: Streamlit
        - **Backend**: Flask + CORS
        - **AI/ML**: TensorFlow + 🤗 Transformers
        - **NLP**: LangChain + OpenAI + Sentence Transformers
        """)
    
    # Main content
    if not api_healthy:
        st.error("Please start the Flask API backend to use SummarEaseAI.")
        st.markdown("Run: `cd backend && python api.py`")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Summarize", 
        "🧠 Intent Analysis", 
        "🔍 Semantic Search", 
        "⚖️ Model Comparison",
        "📊 Analytics"
    ])
    
    with tab1:
        st.header("Article Summarization")
        
        # Model selection for this tab
        col1, col2 = st.columns([3, 1])
        with col2:
            local_summary_model = st.selectbox(
                "AI Model:",
                ["OpenAI + LangChain", "🤖 Multi-Source Agent", "🤗 BART (Local)", "🤗 T5 (Local)"],
                help="Choose between cloud, multi-source agent, or local AI models"
            )
        
        # Input form
        with st.form("summarize_form"):
            with col1:
                topic = st.text_input(
                    "Enter topic or question:",
                    placeholder="e.g., 'What happened on July 20, 1969?' or 'Quantum mechanics'",
                    help="Enter any topic or question about Wikipedia articles"
                )
            
            submit_button = st.form_submit_button("📝 Summarize", use_container_width=True)
        
        if submit_button and topic:
            # Determine model type
            if 'OpenAI' in local_summary_model:
                model_type = 'openai'
            elif 'Multi-Source' in local_summary_model:
                model_type = 'multi_source'
            else:
                model_type = 'huggingface'
            
            spinner_text = {
                'openai': '🤖 OpenAI',
                'multi_source': '🤖 Multi-Source Agent',
                'huggingface': '🤗 Hugging Face'
            }.get(model_type, '🤖 AI')
            
            with st.spinner(f"{spinner_text} is working on your request..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                result, status_code = summarize_article(topic, max_lines, use_intent, model_type)
                progress_bar.empty()
            
            if status_code == 200:
                # Success case
                st.success("✅ Summary generated successfully!")
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📄 Summary")
                    st.markdown(f"**Topic:** {result.get('topic', result.get('query', 'Unknown'))}")
                    
                    # Show model used
                    method = result.get('summarization_method', 'Unknown')
                    if 'Hugging Face' in method:
                        st.markdown(f"**Model:** <span class='hf-badge'>{method}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Model:** {method}")
                    
                    if result.get('predicted_intent'):
                        confidence = result.get('intent_confidence', 0)
                        st.markdown(f"**Predicted Category:** {result['predicted_intent']} ({confidence:.1%} confidence)")
                    
                    # Display summary
                    st.markdown("---")
                    st.markdown(result['summary'])
                
                with col2:
                    if show_analytics:
                        st.subheader("📊 Analytics")
                        
                        # Metrics
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Summary Lines", result.get('max_lines', 'N/A'))
                            st.metric("Article Length", f"{result.get('article_length', 0):,} chars")
                        with col_b:
                            st.metric("Summary Length", f"{result.get('summary_length', 0):,} chars")
                            article_len = int(result.get('article_length', 0))
                            summary_len = int(result.get('summary_length', 0))
                            if article_len > 0:
                                compression = (1 - summary_len / article_len) * 100
                                st.metric("Compression", f"{compression:.1f}%")
            else:
                st.error(f"❌ {result.get('error', 'Unknown error occurred')}")
    
    with tab2:
        st.header("Intent Classification Analysis")
        st.markdown("Compare TensorFlow LSTM vs 🤗 BERT models for intent classification.")
        
        # Model selection
        col1, col2 = st.columns([3, 1])
        with col2:
            selected_intent_model = st.selectbox(
                "Model:",
                ["TensorFlow LSTM", "🤗 BERT", "Compare Both"],
                help="Choose intent classification model"
            )
        
        # Intent prediction form
        with st.form("intent_form"):
            with col1:
                intent_text = st.text_area(
                    "Enter text to classify:",
                    placeholder="e.g., 'Tell me about the Apollo moon landing'",
                    height=100
                )
            predict_button = st.form_submit_button("🧠 Predict Intent")
        
        if predict_button and intent_text:
            if selected_intent_model == "Compare Both":
                # Compare models
                with st.spinner("Comparing AI models..."):
                    comparison_result = compare_models(intent_text)
                
                if comparison_result and comparison_result.get('model_predictions'):
                    st.subheader("⚖️ Model Comparison")
                    predictions = comparison_result['model_predictions']
                    
                    col1, col2 = st.columns(2)
                    
                    # TensorFlow results
                    if 'tensorflow_lstm' in predictions:
                        with col1:
                            tf_result = predictions['tensorflow_lstm']
                            st.markdown("### 🧠 TensorFlow LSTM")
                            st.markdown(f"**Intent:** {tf_result['intent']}")
                            st.markdown(f"**Confidence:** {tf_result['confidence']:.1%}")
                            
                            # Confidence gauge
                            fig_tf = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=tf_result['confidence'] * 100,
                                title={'text': "TF Confidence"},
                                gauge={'axis': {'range': [None, 100]},
                                       'bar': {'color': "orange"}},
                                domain={'x': [0, 1], 'y': [0, 1]}
                            ))
                            fig_tf.update_layout(height=200)
                            st.plotly_chart(fig_tf, use_container_width=True)
                    
                    # BERT results
                    if 'bert_transformer' in predictions:
                        with col2:
                            bert_result = predictions['bert_transformer']
                            st.markdown("### 🤗 BERT Transformer")
                            st.markdown(f"**Intent:** {bert_result['intent']}")
                            st.markdown(f"**Confidence:** {bert_result['confidence']:.1%}")
                            
                            # Confidence gauge
                            fig_bert = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=bert_result['confidence'] * 100,
                                title={'text': "BERT Confidence"},
                                gauge={'axis': {'range': [None, 100]},
                                       'bar': {'color': "green"}},
                                domain={'x': [0, 1], 'y': [0, 1]}
                            ))
                            fig_bert.update_layout(height=200)
                            st.plotly_chart(fig_bert, use_container_width=True)
                
            else:
                # Single model prediction
                model_type = 'tensorflow' if 'TensorFlow' in selected_intent_model else 'bert'
                
                with st.spinner("Analyzing intent..."):
                    intent_result = predict_intent(intent_text, model_type)
                
                if intent_result:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("🎯 Prediction Result")
                        st.markdown(f"**Text:** {intent_result['text']}")
                        st.markdown(f"**Model:** {intent_result['model_type']}")
                        st.markdown(f"**Predicted Intent:** {intent_result['predicted_intent']}")
                        st.markdown(f"**Confidence:** {intent_result['confidence']:.1%}")
                    
                    with col2:
                        # Confidence gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=intent_result['confidence'] * 100,
                            title={'text': "Confidence Score"},
                            delta={'reference': 80},
                            gauge={'axis': {'range': [None, 100]},
                                   'bar': {'color': "darkblue"}},
                            domain={'x': [0, 1], 'y': [0, 1]}
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🔍 Semantic Search")
        st.markdown("Find Wikipedia articles using **sentence embeddings** - search by meaning, not just keywords!")
        
        # Explanation of sentence embeddings
        with st.expander("🧠 What are Sentence Embeddings?"):
            st.markdown("""
            **Sentence embeddings** convert text into numerical vectors that capture semantic meaning:
            
            1. **Traditional keyword search**: "Apollo 11" only finds articles with those exact words
            2. **Semantic search**: "moon landing mission" finds related articles about Apollo 11, lunar exploration, NASA missions, etc.
            
            **Example:**
            - Query: "How to land on the moon?"
            - Traditional: Limited results with exact keywords
            - Semantic: Finds "Apollo 11", "Lunar Module", "NASA missions", "Space exploration"
            
            **Technology**: Uses 🤗 SentenceTransformers to create 384-dimensional vectors representing meaning.
            """)
        
        # Search form
        with st.form("semantic_search_form"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                search_query = st.text_input(
                    "Enter your search query:",
                    placeholder="e.g., 'space exploration missions' or 'renewable energy technology'",
                    help="Search by meaning - try natural language questions!"
                )
            
            with col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                search_button = st.form_submit_button("🔍 Semantic Search", use_container_width=True)
        
        if search_button and search_query:
            with st.spinner("🔍 Searching with sentence embeddings..."):
                search_result = semantic_search(search_query, max_results=5)
            
            if search_result:
                st.subheader("📋 Search Results")
                st.markdown(f"**Query:** {search_result['query']}")
                st.markdown(f"**Method:** {search_result['search_method']}")
                st.markdown(f"**Model:** {search_result['model_used']}")
                
                # Display results
                articles = search_result.get('similar_articles', [])
                if articles:
                    for i, article in enumerate(articles, 1):
                        st.markdown(f"**{i}. {article}**")
                        
                        # Add quick summarize button
                        if st.button(f"📝 Summarize '{article}'", key=f"sum_{i}"):
                            with st.spinner(f"Summarizing {article}..."):
                                sum_result, sum_status = summarize_article(article, max_lines=20)
                            
                            if sum_status == 200:
                                st.success(f"Summary of **{article}**:")
                                st.write(sum_result['summary'])
                            else:
                                st.error(f"Could not summarize {article}")
                else:
                    st.info("No similar articles found. Try a different query.")
    
    with tab4:
        st.header("⚖️ AI Model Comparison")
        st.markdown("Compare different AI models side-by-side to understand their strengths and differences.")
        
        # Quick comparison section
        st.subheader("🧪 Quick Model Test")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            test_text = st.text_input(
                "Test text for model comparison:",
                value="Tell me about artificial intelligence and machine learning",
                help="Enter text to compare how different models classify intent"
            )
        
        with col2:
            if st.button("⚖️ Compare Models", use_container_width=True):
                if test_text:
                    with st.spinner("Comparing AI models..."):
                        comparison = compare_models(test_text)
                    
                    if comparison and comparison.get('model_predictions'):
                        st.subheader("📊 Comparison Results")
                        
                        # Create comparison table
                        predictions = comparison['model_predictions']
                        comparison_data = []
                        
                        for model_name, result in predictions.items():
                            comparison_data.append({
                                'Model': result['model_type'],
                                'Predicted Intent': result['intent'],
                                'Confidence': f"{result['confidence']:.1%}",
                                'Confidence Score': result['confidence']
                            })
                        
                        if comparison_data:
                            df = pd.DataFrame(comparison_data)
                            st.dataframe(df, use_container_width=True)
                            
                            # Visualization
                            if len(comparison_data) > 1:
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=[d['Model'] for d in comparison_data],
                                    y=[d['Confidence Score'] for d in comparison_data],
                                    text=[d['Confidence'] for d in comparison_data],
                                    textposition='auto',
                                    marker_color=['orange', 'green']
                                ))
                                fig.update_layout(
                                    title="Model Confidence Comparison",
                                    xaxis_title="AI Model",
                                    yaxis_title="Confidence Score",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
        
        # Model information
        st.subheader("🔧 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧠 TensorFlow LSTM")
            st.markdown("""
            - **Architecture**: Bidirectional LSTM with embeddings
            - **Training**: Custom dataset, 20 epochs
            - **Vocabulary**: 10,000 words
            - **Sequence Length**: 100 tokens
            - **Categories**: 9 intent classes
            - **Model Size**: ~2.5MB
            """)
        
        with col2:
            st.markdown("### 🤗 BERT Transformer")
            st.markdown("""
            - **Architecture**: Pre-trained BERT base model
            - **Fine-tuning**: Intent classification task
            - **Vocabulary**: 30,522 subword tokens
            - **Sequence Length**: 512 tokens
            - **Categories**: 9 intent classes
            - **Model Size**: ~110MB
            """)
    
    with tab5:
        st.header("📊 System Analytics")
        
        # System information
        api_status = get_api_status()
        if api_status:
            # Feature availability
            st.subheader("🚀 Available Features")
            features = api_status.get('features', {})
            
            feature_data = [
                {'Feature': 'TensorFlow Intent Model', 'Status': '✅ Available' if features.get('tensorflow_intent_model') else '❌ Unavailable'},
                {'Feature': 'BERT Intent Model', 'Status': '✅ Available' if features.get('bert_intent_model') else '❌ Unavailable'},
                {'Feature': 'OpenAI Summarization', 'Status': '✅ Available' if features.get('openai_summarization') else '❌ Unavailable'},
                {'Feature': 'HuggingFace Summarization', 'Status': '✅ Available' if features.get('huggingface_summarization') else '❌ Unavailable'},
                {'Feature': 'Semantic Search', 'Status': '✅ Available' if features.get('semantic_search') else '❌ Unavailable'},
                {'Feature': 'Sentence Embeddings', 'Status': '✅ Available' if features.get('sentence_embeddings') else '❌ Unavailable'}
            ]
            
            feature_df = pd.DataFrame(feature_data)
            st.dataframe(feature_df, use_container_width=True)
            
            # API info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("API Version", api_status.get('version', 'Unknown'))
            with col2:
                st.metric("Total Endpoints", len(api_status.get('endpoints', {})))
            with col3:
                available_features = sum(1 for f in features.values() if f)
                st.metric("Active Features", f"{available_features}/{len(features)}")
            
            # Technology comparison
            st.subheader("🛠️ Technology Stack Comparison")
            
            tech_comparison = pd.DataFrame({
                'Component': ['Intent Classification', 'Text Summarization', 'Semantic Search', 'Model Serving'],
                'Traditional': ['Rule-based/Keywords', 'Extractive methods', 'Keyword matching', 'Single model'],
                'SummarEaseAI v2.0': ['TensorFlow LSTM + BERT', 'OpenAI + Hugging Face', 'Sentence embeddings', 'Multiple AI models']
            })
            
            st.dataframe(tech_comparison, use_container_width=True)
        
        # Sample queries for testing
        st.subheader("🧪 Sample Test Queries")
        sample_queries = [
            "What happened on July 20, 1969?",
            "Explain quantum mechanics principles",
            "Tell me about Albert Einstein's discoveries",
            "How does renewable energy work?",
            "Olympic Games history and significance",
            "Democracy and political systems",
            "Mountain formation geological processes",
            "Renaissance art and cultural movement"
        ]
        
        st.markdown("Try these sample queries to test different AI capabilities:")
        
        cols = st.columns(2)
        for i, query in enumerate(sample_queries):
            with cols[i % 2]:
                if st.button(f"📝 {query}", key=f"sample_{i}"):
                    st.session_state.sample_query = query
                    st.rerun()
        
        # Handle sample query selection
        if hasattr(st.session_state, 'sample_query'):
            st.success(f"Selected query: **{st.session_state.sample_query}**")
            st.info("Go to the 'Summarize' or 'Intent Analysis' tab to run this query!")

if __name__ == "__main__":
    main()
