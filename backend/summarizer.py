# LangChain/OpenAI summarization logic
import os
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

def summarize_article_with_limit(article_text, max_lines=30):
    """
    Summarize article text with a specified line limit using OpenAI and LangChain
    """
    try:
        # Initialize OpenAI LLM
        llm = OpenAI(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            temperature=0.2,
            max_tokens=1000
        )
        
        # Custom prompt template for length-limited summarization
        prompt_template = PromptTemplate(
            input_variables=["text"],
            template=f"""Summarize the following text concisely in at most {max_lines} lines. 
            Make sure each line contains meaningful information and avoid redundancy:

            {{text}}

            Summary:"""
        )
        
        # Create summarization chain
        chain = load_summarize_chain(
            llm=llm, 
            chain_type="stuff",
            prompt=prompt_template
        )
        
        # Create document
        docs = [Document(page_content=article_text)]
        
        # Generate summary
        summary = chain.run(docs)
        return summary.strip()
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def summarize_article(article_text):
    """
    Basic article summarization without line limit
    """
    return summarize_article_with_limit(article_text, max_lines=50)
