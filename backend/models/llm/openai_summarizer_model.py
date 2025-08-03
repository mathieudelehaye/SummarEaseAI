"""
OpenAI Summarizer Model
Pure data access layer for OpenAI API interactions
Handles all LangChain and OpenAI API communication without business logic
"""

import logging
import os
from typing import List, Optional

# Third-party imports
from dotenv import load_dotenv

# LangChain imports
try:
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.schema import BaseOutputParser
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Set these to None for mocking in tests
    LLMChain = None
    ChatOpenAI = None
    PromptTemplate = None
    RecursiveCharacterTextSplitter = None
    logging.warning("LangChain not available. OpenAI summarization will be disabled.")

# Load environment variables
load_dotenv()

# Set up logging
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
                summary = summary[len(prefix) :].strip()

        return summary


def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment variables"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OpenAI API key not found in environment variables")
        return None
    return api_key


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count for text"""
    if not text:
        return 0
    # Rough estimation: 1 token ≈ 4 characters for English text
    return len(text) // 4


def sanitize_article_text(text: str) -> str:
    """Sanitize article text to prevent format string errors"""
    if not text:
        return ""
    # Remove or escape any characters that could cause format string issues
    # Replace curly braces to prevent format code errors
    sanitized = str(text).replace("{", "(").replace("}", ")")
    return sanitized


def chunk_text_for_openai(text: str, max_chunk_tokens: int = 12000) -> List[str]:
    """
    Split text into chunks suitable for OpenAI processing

    Args:
        text: Input text to chunk
        max_chunk_tokens: Maximum tokens per chunk (leaving room for prompt and completion)

    Returns:
        List of text chunks
    """
    if not LANGCHAIN_AVAILABLE or not RecursiveCharacterTextSplitter:
        # Simple fallback chunking
        words = text.split()
        chunk_size = max_chunk_tokens * 3  # Rough estimation: 1 token ≈ 0.75 words
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
        return chunks

    try:
        # Use LangChain's text splitter for better chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_tokens * 3,  # Rough estimation for character count
            chunk_overlap=400,  # More overlap to maintain context
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = text_splitter.split_text(text)

        # If we have multiple chunks, prioritize the first chunk as it usually contains
        # the main topic and introduction
        if len(chunks) > 1:
            logger.info(
                "Text split into %d chunks. First chunk (likely most relevant) has %d characters.",
                len(chunks),
                len(chunks[0]),
            )

        return chunks
    except (
        ImportError,
        AttributeError,
        ValueError,
        TypeError,
        OSError,
        Exception,
    ) as e:
        logger.warning(
            "Error using LangChain text splitter, falling back to simple chunking: %s",
            e,
        )
        # Fallback to simple chunking
        words = text.split()
        chunk_size = max_chunk_tokens * 3
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
        return chunks


def create_summarization_chain():
    """Create a LangChain summarization chain"""
    if not LANGCHAIN_AVAILABLE or not ChatOpenAI or not LLMChain or not PromptTemplate:
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
            max_tokens=1000,
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
            input_variables=["article_text"], template=prompt_template
        )

        # Create the chain
        chain = LLMChain(llm=llm, prompt=prompt, output_parser=SummaryOutputParser())

        return chain

    except (ImportError, AttributeError, ValueError, TypeError, OSError) as e:
        logger.error("Error creating summarization chain: %s", str(e))
        return None


def create_line_limited_chain(max_lines: int = 30):
    """Create a summarization chain with line limit"""
    if not LANGCHAIN_AVAILABLE or not ChatOpenAI or not LLMChain or not PromptTemplate:
        return None

    api_key = get_openai_api_key()
    if not api_key:
        return None

    try:
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo-16k",  # 16k context window
            temperature=0.3,
            max_tokens=min(max_lines * 20, 1000),  # Estimate tokens per line
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
            input_variables=["article_text"], template=prompt_template
        )

        chain = LLMChain(llm=llm, prompt=prompt, output_parser=SummaryOutputParser())

        return chain

    except (ImportError, AttributeError, ValueError, TypeError, OSError) as e:
        logger.error("Error creating line-limited chain: %s", str(e))
        return None


def create_intent_aware_chain(intent: str, confidence: float):
    """Create a summarization chain tailored to the detected intent"""
    if not LANGCHAIN_AVAILABLE or not ChatOpenAI or not LLMChain or not PromptTemplate:
        return None

    api_key = get_openai_api_key()
    if not api_key:
        return None

    try:
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo-16k",
            temperature=0.3,
            max_tokens=1000,
        )

        # Intent-specific prompt templates
        intent_prompts = {
            "science": """
            Please provide a comprehensive scientific summary of the following article.
            Focus on: scientific principles, theories, mechanisms, research findings,
        and discoveries.
            Explain complex concepts in an accessible way while maintaining scientific
        accuracy.
            Include key discoveries, methodologies, and the scientific significance of
        the topic.

            Article Content:
            {article_text}

            Scientific Summary:
            """,
            "history": """
            Please provide a comprehensive historical summary of the following article.
            Focus on: key historical events, timeline, important figures, causes and
        effects.
            Emphasize the historical significance, context, and impact on subsequent
        events.
            Include dates, locations, and the broader historical context.

            Article Content:
            {article_text}

            Historical Summary:
            """,
            "music": """
            Please provide a comprehensive music summary of the following article.
            Focus on: musical style, band/artist history, album releases, cultural
        impact, and musical influence.
            Emphasize musical innovations, genre contributions, career highlights, and
        legacy.
            Include formation dates, key members, notable songs/albums, and impact on
        music history.

            Article Content:
            {article_text}

            Music Summary:
            """,
            "technology": """
            Please provide a comprehensive technology summary of the following article.
            Focus on: technological innovations, development process, applications, and
        impact.
            Explain how the technology works, its advantages, limitations, and future
        potential.
            Include technical specifications, adoption timeline, and industry
        significance.

            Article Content:
            {article_text}

            Technology Summary:
            """,
            "sports": """
            Please provide a comprehensive sports summary of the following article.
            Focus on: game rules, competition history, notable athletes, records and
        achievements.
            Emphasize sporting significance, competition formats, and cultural impact.
            Include key statistics, memorable moments, and the sport's evolution.

            Article Content:
            {article_text}

            Sports Summary:
            """,
            "finance": """
            Please provide a comprehensive finance summary of the following article.
            Focus on: financial concepts, market mechanisms, economic impact, and
        monetary systems.
            Emphasize financial significance, market effects, investment implications,
        and economic context.
            Include key figures, financial metrics, market trends, and economic
        consequences.

            Article Content:
            {article_text}

            Finance Summary:
            """,
        }

        # Use intent-specific prompt if available and confidence is high enough
        if intent in intent_prompts and confidence >= 0.5:
            prompt_template = intent_prompts[intent]
            logger.info(
                "Using intent-aware summarization for '%s' category (confidence: %.3f)",
                intent,
                confidence,
            )
        else:
            # Fall back to general prompt
            prompt_template = """
            Please provide a comprehensive summary of the following article content.
            Make the summary informative, well-structured, and easy to understand.
            Focus on the main topics, key information, and significance described in the article.
            
            Article Content:
            {article_text}
            
            Summary:
            """
            logger.info(
                "Using general summarization (intent: %s, confidence: %.3f)",
                intent,
                confidence,
            )

        prompt = PromptTemplate(
            input_variables=["article_text"], template=prompt_template
        )

        chain = LLMChain(llm=llm, prompt=prompt, output_parser=SummaryOutputParser())

        return chain

    except (ImportError, AttributeError, ValueError, TypeError, OSError) as e:
        logger.error("Error creating intent-aware summarization chain: %s", str(e))
        return None
