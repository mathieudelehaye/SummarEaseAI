"""
Summarization module using LangChain and OpenAI for intelligent Wikipedia article summarization.
"""

import os
import logging
from typing import Optional

# Updated LangChain imports for newer versions
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.schema import BaseOutputParser
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        # Fallback for older LangChain versions
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        from langchain.schema import BaseOutputParser
        from langchain.text_splitter import RecursiveCharacterTextSplitter
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

def chunk_text_for_openai(text: str, max_chunk_tokens: int = 12000) -> list[str]:
    """
    Split text into chunks suitable for OpenAI processing
    
    Args:
        text: Input text to chunk
        max_chunk_tokens: Maximum tokens per chunk (leaving room for prompt and completion)
        
    Returns:
        List of text chunks
    """
    if not LANGCHAIN_AVAILABLE:
        # Simple fallback chunking
        words = text.split()
        chunk_size = max_chunk_tokens * 3  # Rough estimation: 1 token ≈ 0.75 words
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    try:
        # Use LangChain's text splitter for better chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_tokens * 3,  # Rough estimation for character count
            chunk_overlap=400,  # More overlap to maintain context
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        
        # If we have multiple chunks, prioritize the first chunk as it usually contains
        # the main topic and introduction
        if len(chunks) > 1:
            logger.info(f"Text split into {len(chunks)} chunks. First chunk (likely most relevant) has {len(chunks[0])} characters.")
        
        return chunks
    except Exception as e:
        logger.warning(f"Error using LangChain text splitter, falling back to simple chunking: {e}")
        # Fallback to simple chunking
        words = text.split()
        chunk_size = max_chunk_tokens * 3
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

def estimate_tokens(text: str) -> int:
    """Rough estimation of token count for text"""
    # Rough estimation: 1 token ≈ 4 characters for English text
    return len(text) // 4

def log_chatgpt_request(prompt_template: str, article_text: str, chunk_number: int = None):
    """Log the exact request being sent to ChatGPT for debugging"""
    full_prompt = prompt_template.format(article_text=article_text)
    
    chunk_info = f" (chunk {chunk_number})" if chunk_number else ""
    logger.info(f"🚀 FULL PROMPT being sent to ChatGPT{chunk_info}:")
    logger.info(f"📋 Prompt length: {len(full_prompt)} characters, ~{estimate_tokens(full_prompt)} tokens")
    
    # Log the actual formatted prompt (truncated for readability)
    if len(full_prompt) > 1000:
        logger.info(f"📄 PROMPT PREVIEW: {full_prompt[:500]}...{full_prompt[-500:]}")
    else:
        logger.info(f"📄 FULL PROMPT: {full_prompt}")
    
    logger.info("=" * 80)

def create_summarization_chain():
    """Create a LangChain summarization chain"""
    if not LANGCHAIN_AVAILABLE:
        return None
    
    api_key = get_openai_api_key()
    if not api_key:
        return None
    
    try:
        # Initialize ChatOpenAI with a model that has larger context window
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo-16k",  # 16k context window
            temperature=0.3,
            max_tokens=1000
        )
        
        # Create prompt template
        prompt_template = """
        Please provide a comprehensive summary of the following Wikipedia article content.
        Make the summary informative, well-structured, and easy to understand.
        Focus on the main events, key people, and historical significance described in the article.
        
        IMPORTANT: Focus on the primary topic and main events described in the article. If this is about a specific historical event or date, prioritize information about that main event rather than background information or related incidents.
        
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
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo-16k",  # 16k context window
            temperature=0.3,
            max_tokens=min(max_lines * 20, 1000)  # Estimate tokens per line
        )
        
        prompt_template = f"""
        Please provide a summary of the following Wikipedia article content.
        The summary should be exactly {max_lines} lines or fewer.
        Make it informative and well-structured.
        
        IMPORTANT: Focus on the main topic and events described in the article. If this is about a specific historical event or date, prioritize information about that main event rather than background or related incidents.
        
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
        
        # Check if text is too long and needs chunking
        estimated_tokens = estimate_tokens(article_text)
        logger.info(f"Estimated tokens in article: {estimated_tokens}")
        
        # Log article preview for debugging
        article_preview = article_text[:500] + "..." if len(article_text) > 500 else article_text
        logger.info(f"📄 Article content preview (first 500 chars): {article_preview}")
        
        if estimated_tokens > 12000:  # Leave room for prompt and completion
            logger.info("Article too long, chunking for processing...")
            chunks = chunk_text_for_openai(article_text, max_chunk_tokens=12000)
            summaries = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                chunk_summary = chain.run(article_text=chunk)
                if chunk_summary and chunk_summary.strip():
                    summaries.append(chunk_summary.strip())
            
            if not summaries:
                return "Error: No summaries generated from chunks"
            
            # If we have multiple chunk summaries, combine them
            if len(summaries) > 1:
                logger.info("Combining chunk summaries...")
                combined_text = "\n\n".join(summaries)
                # Summarize the combined summaries if they're still too long
                if estimate_tokens(combined_text) > 12000:
                    final_summary = chain.run(article_text=combined_text[:12000*4])  # Rough char limit
                else:
                    final_summary = chain.run(article_text=combined_text)
                return final_summary.strip() if final_summary else "Error: Final summary generation failed"
            else:
                return summaries[0]
        else:
            # Text is short enough, process normally
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
        
        # Get the prompt template for logging
        prompt_template = f"""
        Please provide a summary of the following Wikipedia article content.
        The summary should be exactly {max_lines} lines or fewer.
        Make it informative and well-structured.
        
        IMPORTANT: Focus on the main topic and events described in the article. If this is about a specific historical event or date, prioritize information about that main event rather than background or related incidents.
        
        Article Content:
        {{article_text}}
        
        Summary (max {max_lines} lines):
        """
        
        # Check if text is too long and needs chunking
        estimated_tokens = estimate_tokens(article_text)
        logger.info(f"Estimated tokens in article: {estimated_tokens}")
        
        # Log article preview for debugging
        article_preview = article_text[:500] + "..." if len(article_text) > 500 else article_text
        logger.info(f"📄 Article content preview (first 500 chars): {article_preview}")
        
        if estimated_tokens > 12000:  # Leave room for prompt and completion
            logger.info("Article too long, chunking for processing...")
            chunks = chunk_text_for_openai(article_text, max_chunk_tokens=12000)
            summaries = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)} (line-limited)")
                
                # Log what's being sent to ChatGPT for this chunk
                chunk_preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
                logger.info(f"🤖 Sending to ChatGPT chunk {i+1} preview: {chunk_preview}")
                
                # Log the exact prompt being sent to ChatGPT
                log_chatgpt_request(prompt_template, chunk, chunk_number=i+1)
                
                chunk_summary = chain.run(article_text=chunk)
                
                # Log what ChatGPT returned for this chunk
                if chunk_summary and chunk_summary.strip():
                    logger.info(f"✅ ChatGPT returned for chunk {i+1}: {chunk_summary.strip()[:200]}...")
                    summaries.append(chunk_summary.strip())
                else:
                    logger.warning(f"❌ ChatGPT returned empty/invalid response for chunk {i+1}")
            
            if not summaries:
                return "Error: No summaries generated from chunks"
            
            # If we have multiple chunk summaries, combine them
            if len(summaries) > 1:
                logger.info("Combining chunk summaries...")
                combined_text = "\n\n".join(summaries)
                
                # Create a special chain for combining summaries with better focus
                try:
                    llm = ChatOpenAI(
                        openai_api_key=get_openai_api_key(),
                        model_name="gpt-3.5-turbo-16k",
                        temperature=0.3,
                        max_tokens=min(max_lines * 20, 1000)
                    )
                    
                    combine_prompt = f"""
                    Please combine and synthesize the following summaries into a single, coherent summary.
                    Focus on the main topic and primary events. The summary should be exactly {max_lines} lines or fewer.
                    Prioritize the most important information about the main subject.
                    
                    Summaries to combine:
                    {{article_text}}
                    
                    Final unified summary (max {max_lines} lines):
                    """
                    
                    combine_prompt_template = PromptTemplate(
                        input_variables=["article_text"], 
                        template=combine_prompt
                    )
                    
                    combine_chain = LLMChain(
                        llm=llm,
                        prompt=combine_prompt_template,
                        output_parser=SummaryOutputParser()
                    )
                    
                    # Log what's being sent to ChatGPT for final combination
                    combine_preview = combined_text[:500] + "..." if len(combined_text) > 500 else combined_text
                    logger.info(f"🔄 Sending combined summaries to ChatGPT for final synthesis: {combine_preview}")
                    
                    if estimate_tokens(combined_text) > 12000:
                        final_summary = combine_chain.run(article_text=combined_text[:12000*4])
                    else:
                        final_summary = combine_chain.run(article_text=combined_text)
                    
                    # Log final ChatGPT response
                    if final_summary:
                        logger.info(f"🎯 Final ChatGPT response: {final_summary.strip()[:300]}...")
                    else:
                        logger.warning("❌ Final ChatGPT response was empty")
                        
                except Exception as e:
                    logger.warning(f"Error creating specialized combine chain: {e}, falling back to regular chain")
                    # Fallback to original method
                    if estimate_tokens(combined_text) > 12000:
                        final_summary = chain.run(article_text=combined_text[:12000*4])
                    else:
                        final_summary = chain.run(article_text=combined_text)
                
                if not final_summary:
                    return "Error: Final summary generation failed"
                
                # Post-process to ensure line limit
                lines = final_summary.strip().split('\n')
                if len(lines) > max_lines:
                    final_summary = '\n'.join(lines[:max_lines])
                
                return final_summary.strip()
            else:
                # Post-process single summary to ensure line limit
                lines = summaries[0].split('\n')
                if len(lines) > max_lines:
                    summary = '\n'.join(lines[:max_lines])
                else:
                    summary = summaries[0]
                return summary
        else:
            # Text is short enough, process normally
            logger.info(f"Generating summary with max {max_lines} lines...")
            
            # Log the exact prompt being sent to ChatGPT for single processing
            log_chatgpt_request(prompt_template, article_text)
            
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
