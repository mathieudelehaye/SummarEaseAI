# SummarEaseAI - Enhanced Streamlit Frontend with Hugging Face Integration
import json
import os
import time
from datetime import datetime

import requests
import streamlit as st

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
API_BASE_URL = os.getenv('BACKEND_URL', 'http://backend:5000')  # Use environment variable with fallback for local Docker

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

def predict_intent(text):
    """Predict intent using BERT model"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/intent_bert",
            json={'text': text},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error predicting intent: {str(e)}")
    return None



def summarize_article(topic, max_lines=30, use_intent=True, model_type='openai'):
    """Summarize Wikipedia article using specified method"""
    try:
        # Map model types to api_simple endpoints
        if model_type == 'multi_source':
            endpoint = '/summarize_multi_source'
            payload = {
                'query': topic,
                'max_lines': max_lines
            }
        else:
            endpoint = '/summarize'
            payload = {
                'query': topic,
                'max_lines': max_lines
            }
        
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload,
            timeout=60
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
    st.markdown('<h1 class="main-header">🤖 SummarEaseAI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Wikipedia Summarization with <span class="tf-badge">TensorFlow</span> & DirectML</p>',
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
                st.info(f"🚀 BERT Model: {'✅' if features.get('bert_model') else '❌'}")
                st.info(f"📡 OpenAI: {'✅' if features.get('openai_summarization') else '❌'}")
                st.info(f"🌍 Wikipedia: {'✅' if features.get('wikipedia_fetching') else '❌'}")
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
            ["Single Source", "Multi-Source Agent"],
            help="Single source uses one Wikipedia article. Multi-source synthesizes multiple sources."
        )
        
        st.divider()
        
        # About
        st.subheader("📖 About")
        st.markdown("""
        **Features:**
        - 🚀 **BERT Intent Classification**
        - 🌍 **Wikipedia Portal Integration**
        - 📊 **Intent Classification**
        - 🤖 **OpenAI Summarization**
        
        **Tech Stack:**
        - **Frontend**: Streamlit
        - **Backend**: Flask + CORS
        - **AI/ML**: BERT
        - **NLP**: LangChain + OpenAI
        - **Data**: Wikipedia API
        """)
    
    # Main content
    if not api_healthy:
        st.error("Please start the Flask API backend to use SummarEaseAI.")
        st.markdown("Run: `cd backend && python api.py`")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 Summarize", 
        "🧠 Intent Analysis", 
        "📊 Analytics"
    ])
    
    with tab1:
        st.header("Article Summarization")
        
        # Model selection for this tab
        col1, col2 = st.columns([3, 1])
        with col2:
            local_summary_model = st.selectbox(
                "AI Model:",
                ["Single Source (OpenAI)", "Multi-Source Agent"],
                help="Choose between single Wikipedia article or multi-source synthesis"
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
            if 'Multi-Source' in local_summary_model:
                model_type = 'multi_source'
            else:
                model_type = 'openai'
            
            spinner_text = {
                'openai': '🤖 Single Source',
                'multi_source': '🤖 Multi-Source Agent'
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
                    method = result.get('method', 'Unknown')
                    model = result.get('model', '')
                    model_display = f"{method} ({model})" if model else method
                    if 'Hugging Face' in model_display:
                        st.markdown(f"**Model:** <span class='hf-badge'>{model_display}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Model:** {model_display}")
                    
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
                            st.metric("Summary Lines", len(result.get('summary', '').splitlines()))
                            if 'article' in result:
                                st.metric("Article Length", f"{result['article'].get('length', 0):,} chars")
                        with col_b:
                            st.metric("Summary Length", f"{result.get('summary_length', 0):,} chars")
                            if 'article' in result:
                                article_len = int(result['article'].get('length', 0))
                            summary_len = int(result.get('summary_length', 0))
                            if article_len > 0:
                                compression = (1 - summary_len / article_len) * 100
                                st.metric("Compression", f"{compression:.1f}%")
                            
                        # Additional stats for multi-source
                        if 'summaries' in result:
                            st.markdown("---")
                            st.markdown("**Multi-Source Stats:**")
                            col_c, col_d = st.columns(2)
                            with col_c:
                                st.metric("Articles Found", len(result.get('summaries', [])))
                                st.metric("OpenAI Calls", result.get('openai_calls', 1))
                            with col_d:
                                st.metric("Wikipedia Calls", result.get('wikipedia_calls', 1))
                                st.metric("Articles Used", len(result.get('summaries', [])))
                            
                            if result.get('summaries'):
                                st.markdown("**Articles Used:**")
                                for article in result['summaries']:
                                    st.markdown(f"- [{article['title']}]({article['url']})")
            else:
                st.error(f"❌ {result.get('error', 'Unknown error occurred')}")
    
    with tab2:
        st.header("Intent Classification Analysis")
        st.markdown("Use BERT model for intent classification.")
        
        # Intent prediction form
        with st.form("intent_form"):
                intent_text = st.text_area(
                    "Enter text to classify:",
                    placeholder="e.g., 'Tell me about the Apollo moon landing'",
                    height=100
                )
            predict_button = st.form_submit_button("🧠 Predict Intent")
        
        if predict_button and intent_text:
            with st.spinner("Analyzing intent..."):
                intent_result = predict_intent(intent_text)
            
            if intent_result:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("🎯 Prediction Result")
                    st.markdown(f"**Text:** {intent_result['text']}")
                    st.markdown(f"**Model:** {intent_result['model_type']}")
                    st.markdown(f"**Predicted Intent:** {intent_result['intent']}")
                    
                    # Add confidence score speedometer
                    confidence = float(intent_result['confidence']) * 100
                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        st.markdown(f"**Confidence:** {confidence:.1f}%")
                    with col_b:
                        # Create plotly gauge
                        import plotly.graph_objects as go
                        
                    fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = confidence,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Confidence Score"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#1f77b4"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 30], 'color': '#ffcdd2'},
                                    {'range': [30, 70], 'color': '#fff9c4'},
                                    {'range': [70, 100], 'color': '#c8e6c9'}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': confidence
                                }
                            }
                    ))
                        
                        fig.update_layout(
                            height=200,
                            margin=dict(l=10, r=10, t=40, b=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={'color': "#666666"}
                        )
                        
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("📊 System Analytics")
        
        # System information
        api_status = get_api_status()
        if api_status:
            # Feature availability
            st.subheader("🚀 Available Features")
            features = api_status.get('features', {})
            
            feature_data = [
                {'Feature': 'BERT Intent Model', 'Status': '✅ Available' if features.get('bert_model') else '❌ Unavailable'},
                {'Feature': 'OpenAI Summarization', 'Status': '✅ Available' if features.get('openai_summarization') else '❌ Unavailable'},
                {'Feature': 'Wikipedia Integration', 'Status': '✅ Available' if features.get('wikipedia_fetching') else '❌ Unavailable'}
            ]
            
            for feature in feature_data:
                st.info(f"{feature['Feature']}: {feature['Status']}")
            
            # API info
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("API Version", api_status.get('version', 'Unknown'))
            with col2:
                available_features = sum(1 for f in features.values() if f)
                st.metric("Active Features", f"{available_features}/{len(features)}")
        
        # Sample queries for testing
        st.subheader("🧪 Sample Test Queries")
        sample_queries = [
            "What happened on July 20, 1969?",
            "Explain quantum mechanics principles",
            "Tell me about Albert Einstein's discoveries",
            "How does renewable energy work?",
            "Olympic Games history and significance",
            "Democracy and political systems"
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
