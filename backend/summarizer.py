"""
Summarization module using LangChain and OpenAI for intelligent Wikipedia article summarization.
"""

import os
import logging
from typing import Optional

# Updated LangChain imports for newer versions
try:
    from langchain_openai import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.schema import BaseOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        # Fallback for older LangChain versions
        from langchain.llms import OpenAI
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        from langchain.schema import BaseOutputParser
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        logging.warning("LangChain not available. OpenAI summarization will be disabled.")

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummaryOutputParser(BaseOutputParser):
    """Custom output parser for summary responses"""
    
    def parse(self, text: str) -> str:
        """Parse the output from the LLM"""
        # Clean up the response
        summary = text.strip()
        
        # Remove any unwanted prefixes
        prefixes_to_remove = [
            "Summary:",
            "Here's a summary:",
            "Here is a summary:",
            "The summary is:",
        ]
        
        for prefix in prefixes_to_remove:
            if summary.lower().startswith(prefix.lower()):
                summary = summary[len(prefix):].strip()
        
        return summary

def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment variables"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.warning("OpenAI API key not found in environment variables")
        return None
    return api_key

def create_summarization_chain():
    """Create a LangChain summarization chain"""
    if not LANGCHAIN_AVAILABLE:
        return None
    
    api_key = get_openai_api_key()
    if not api_key:
        return None
    
    try:
        # Initialize OpenAI LLM
        llm = OpenAI(
            openai_api_key=api_key,
            temperature=0.3,
            max_tokens=1000
        )
        
        # Create prompt template
        prompt_template = """
        Please provide a comprehensive summary of the following Wikipedia article content.
        Make the summary informative, well-structured, and easy to understand.
        
        Article Content:
        {article_text}
        
        Summary:
        """
        
        prompt = PromptTemplate(
            input_variables=["article_text"],
            template=prompt_template
        )
        
        # Create the chain
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            output_parser=SummaryOutputParser()
        )
        
        return chain
        
    except Exception as e:
        logger.error(f"Error creating summarization chain: {str(e)}")
        return None

def create_line_limited_chain(max_lines: int = 30):
    """Create a summarization chain with line limit"""
    if not LANGCHAIN_AVAILABLE:
        return None
    
    api_key = get_openai_api_key()
    if not api_key:
        return None
    
    try:
        llm = OpenAI(
            openai_api_key=api_key,
            temperature=0.3,
            max_tokens=min(max_lines * 20, 1000)  # Estimate tokens per line
        )
        
        prompt_template = f"""
        Please provide a summary of the following Wikipedia article content.
        The summary should be exactly {max_lines} lines or fewer.
        Make it informative and well-structured.
        
        Article Content:
        {{article_text}}
        
        Summary (max {max_lines} lines):
        """
        
        prompt = PromptTemplate(
            input_variables=["article_text"],
            template=prompt_template
        )
        
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            output_parser=SummaryOutputParser()
        )
        
        return chain
        
    except Exception as e:
        logger.error(f"Error creating line-limited chain: {str(e)}")
        return None

def summarize_article(article_text: str) -> str:
    """
    Summarize article text using LangChain and OpenAI
    
    Args:
        article_text: The article content to summarize
        
    Returns:
        Generated summary or error message
    """
    if not LANGCHAIN_AVAILABLE:
        return "Error: LangChain not available. Please install langchain and langchain-openai packages."
    
    if not article_text or len(article_text.strip()) == 0:
        return "Error: No article content provided"
    
    # Check if article is too short
    if len(article_text.strip()) < 100:
        return "Error: Article content too short to summarize effectively"
    
    try:
        chain = create_summarization_chain()
        if not chain:
            return "Error: Could not initialize OpenAI summarization. Check your API key."
        
        # Generate summary
        logger.info("Generating summary using OpenAI...")
        summary = chain.run(article_text=article_text)
        
        if not summary or len(summary.strip()) == 0:
            return "Error: Generated summary is empty"
        
        logger.info("Summary generated successfully")
        return summary.strip()
        
    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}")
        return f"Error generating summary: {str(e)}"

def summarize_article_with_limit(article_text: str, max_lines: int = 30) -> str:
    """
    Summarize article with a specific line limit
    
    Args:
        article_text: The article content to summarize
        max_lines: Maximum number of lines in the summary
        
    Returns:
        Generated summary or error message
    """
    if not LANGCHAIN_AVAILABLE:
        return "Error: LangChain not available. Please install langchain and langchain-openai packages."
    
    if not article_text or len(article_text.strip()) == 0:
        return "Error: No article content provided"
    
    if max_lines < 5:
        max_lines = 5
    elif max_lines > 100:
        max_lines = 100
    
    try:
        chain = create_line_limited_chain(max_lines)
        if not chain:
            return "Error: Could not initialize OpenAI summarization. Check your API key."
        
        logger.info(f"Generating summary with max {max_lines} lines...")
        summary = chain.run(article_text=article_text)
        
        if not summary or len(summary.strip()) == 0:
            return "Error: Generated summary is empty"
        
        # Post-process to ensure line limit
        lines = summary.strip().split('\n')
        if len(lines) > max_lines:
            summary = '\n'.join(lines[:max_lines])
        
        logger.info(f"Summary generated successfully ({len(lines)} lines)")
        return summary.strip()
        
    except Exception as e:
        logger.error(f"Error during line-limited summarization: {str(e)}")
        return f"Error generating summary: {str(e)}"

def get_summarization_status() -> dict:
    """Get the status of summarization capabilities"""
    return {
        "langchain_available": LANGCHAIN_AVAILABLE,
        "openai_api_key_configured": get_openai_api_key() is not None,
        "summarization_ready": LANGCHAIN_AVAILABLE and get_openai_api_key() is not None
    }
