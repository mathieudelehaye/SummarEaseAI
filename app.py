# Streamlit frontend app
import streamlit as st
import requests

st.title('SummarEaseAI Chatbot')

topic = st.text_input('Enter your topic or question:', '')

if st.button('Summarize'):
    if topic:
        response = requests.post('http://localhost:5000/summarize', json={'topic': topic})
        if response.ok:
            summary = response.json().get('summary')
            st.write(summary)
        else:
            st.error('Error fetching summary. Please try another topic.')
