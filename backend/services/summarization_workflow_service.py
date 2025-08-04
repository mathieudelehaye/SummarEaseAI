# pylint: disable=ungrouped-imports
"""
Summarization module using LangChain and OpenAI for intelligent Wikipedia article summarization.
"""

import logging

from dotenv import load_dotenv

# First-party imports
from ..models.llm.openai_summarizer_model import (
    chunk_text_for_openai,
    create_intent_aware_chain,
    create_line_limited_chain,
    estimate_tokens,
    get_openai_api_key,
    sanitize_article_text,
)

# LangChain imports
try:
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. OpenAI summarization will be disabled.")

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_chatgpt_request(
    prompt_template: str, article_text: str, chunk_number: int = None
):
    """Log the exact request being sent to ChatGPT for debugging"""
    # Safely replace template placeholders to avoid format code errors
    safe_article_text = str(article_text).replace("{", "{{").replace("}", "}}")
    full_prompt = prompt_template.format(article_text=safe_article_text)

    chunk_info = f" (chunk {chunk_number})" if chunk_number else ""
    logger.info("🚀 FULL PROMPT being sent to ChatGPT%s:", chunk_info)
    logger.info(
        "📋 Prompt length: %d characters, ~%d tokens",
        len(full_prompt),
        estimate_tokens(full_prompt),
    )

    # Log the actual formatted prompt (truncated for readability)
    if len(full_prompt) > 1000:
        logger.info("📄 PROMPT PREVIEW: %s...%s", full_prompt[:500], full_prompt[-500:])
    else:
        logger.info("📄 FULL PROMPT: %s", full_prompt)

    logger.info("=" * 80)


def _validate_line_limited_input(
    article_text: str, max_lines: int
) -> tuple[str | None, int]:
    """Validate input for line-limited summarization and return error message if invalid."""

    if not article_text or len(article_text.strip()) == 0:
        return "Error: No article content provided", max_lines

    if max_lines < 5:
        max_lines = 5
    elif max_lines > 100:
        max_lines = 100

    return None, max_lines


def _process_single_chunk_summary(summaries: list[str], max_lines: int) -> str:
    """Process a single chunk summary to ensure line limit."""
    lines = summaries[0].split("\n")
    if len(lines) > max_lines:
        summary = "\n".join(lines[:max_lines])
    else:
        summary = summaries[0]
    return summary


def _process_line_limited_chunks(
    chain, chunks: list[str], prompt_template: str
) -> list[str]:
    """Process chunks for line-limited summarization."""
    summaries = []

    for i, chunk in enumerate(chunks):
        logger.info("Processing chunk %d/%d (line-limited)", i + 1, len(chunks))

        # Log what's being sent to ChatGPT for this chunk
        chunk_preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
        logger.info("🤖 Sending to ChatGPT chunk %d preview: %s", i + 1, chunk_preview)

        # Log the exact prompt being sent to ChatGPT
        log_chatgpt_request(prompt_template, chunk, chunk_number=i + 1)

        chunk_summary = chain.run(article_text=chunk)

        # Log what ChatGPT returned for this chunk
        if chunk_summary and chunk_summary.strip():
            logger.info(
                "✅ ChatGPT returned for chunk %d: %s...",
                i + 1,
                chunk_summary.strip()[:200],
            )
            summaries.append(chunk_summary.strip())
        else:
            logger.warning(
                "❌ ChatGPT returned empty/invalid response for chunk %d", i + 1
            )

    return summaries


def _combine_chunk_summaries(summaries: list[str], max_lines: int, chain) -> str:
    """Combine multiple chunk summaries into a single summary."""
    logger.info("Combining chunk summaries...")
    combined_text = "\n\n".join(summaries)

    # Create a special chain for combining summaries with better focus
    try:
        llm = ChatOpenAI(
            openai_api_key=get_openai_api_key(),
            model_name="gpt-3.5-turbo-16k",
            temperature=0.3,
            max_tokens=min(max_lines * 20, 1000),
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
            input_variables=["article_text"], template=combine_prompt
        )

        combine_chain = LLMChain(
            llm=llm,
            prompt=combine_prompt_template,
        )

        # Log what's being sent to ChatGPT for final combination
        combine_preview = (
            combined_text[:500] + "..." if len(combined_text) > 500 else combined_text
        )
        logger.info(
            "🔄 Sending combined summaries to ChatGPT for final synthesis: %s",
            combine_preview,
        )

        if estimate_tokens(combined_text) > 12000:
            final_summary = combine_chain.run(article_text=combined_text[: 12000 * 4])
        else:
            final_summary = combine_chain.run(article_text=combined_text)

        # Log final ChatGPT response
        if final_summary:
            logger.info(
                "🎯 Final ChatGPT response: %s...",
                final_summary.strip()[:300],
            )
        else:
            logger.warning("❌ Final ChatGPT response was empty")

    except (
        ValueError,
        KeyError,
        AttributeError,
        ImportError,
    ) as e:
        logger.warning(
            "Error creating specialized combine chain: %s, "
            "falling back to regular chain",
            str(e),
        )
        # Fallback to original method
        if estimate_tokens(combined_text) > 12000:
            final_summary = chain.run(article_text=combined_text[: 12000 * 4])
        else:
            final_summary = chain.run(article_text=combined_text)

    if not final_summary:
        return "Error: Final summary generation failed"

    # Post-process to ensure line limit
    lines = final_summary.strip().split("\n")
    if len(lines) > max_lines:
        final_summary = "\n".join(lines[:max_lines])

    return final_summary.strip()


def _process_short_article(
    chain, article_text: str, prompt_template: str, max_lines: int
) -> str:
    """Process short articles that don't need chunking."""
    logger.info("Generating summary with max %d lines...", max_lines)

    # Log the exact prompt being sent to ChatGPT for single processing
    log_chatgpt_request(prompt_template, article_text)

    summary = chain.run(article_text=article_text)

    if not summary or len(summary.strip()) == 0:
        return "Error: Generated summary is empty"

    # Post-process to ensure line limit
    lines = summary.strip().split("\n")
    if len(lines) > max_lines:
        summary = "\n".join(lines[:max_lines])

    logger.info("Summary generated successfully (%d lines)", len(lines))
    return summary.strip()


def _initialize_summarization_chain(max_lines: int) -> str | tuple:
    """Initialize the summarization chain and return error message or (chain, prompt_template)."""
    chain = create_line_limited_chain(max_lines)
    if not chain:
        return "Error: Could not initialize OpenAI summarization. Check your API key."

    # Get the prompt template for logging
    prompt_template = f"""
    Please provide a summary of the following Wikipedia article content.
    The summary should be exactly {max_lines} lines or fewer.
    Make it informative and well-structured.
    
    IMPORTANT: Focus on the main topic and events described in the article. 
    If this is about a specific historical event or date, prioritize information 
    about that main event rather than background or related incidents.
    
    Article Content:
    {{article_text}}
    
    Summary (max {max_lines} lines):
    """

    return chain, prompt_template


def _process_long_article_with_limit(
    chain, article_text: str, prompt_template: str, max_lines: int
) -> str:
    """Process long articles that need chunking for line-limited summarization."""
    logger.info("Article too long, chunking for processing...")
    chunks = chunk_text_for_openai(article_text, max_chunk_tokens=12000)
    summaries = _process_line_limited_chunks(chain, chunks, prompt_template)

    if not summaries:
        return "Error: No summaries generated from chunks"

    # If we have multiple chunk summaries, combine them
    if len(summaries) > 1:
        return _combine_chunk_summaries(summaries, max_lines, chain)

    # Post-process single summary to ensure line limit
    return _process_single_chunk_summary(summaries, max_lines)


def summarize_article_with_limit(article_text: str, max_lines: int = 30) -> str:
    """
    Summarize article with a specific line limit

    Args:
        article_text: The article content to summarize
        max_lines: Maximum number of lines in the summary

    Returns:
        Generated summary or error message
    """
    # Validate input
    error, adjusted_max_lines = _validate_line_limited_input(article_text, max_lines)
    if error:
        return error

    try:
        # Initialize chain
        chain_result = _initialize_summarization_chain(adjusted_max_lines)
        if isinstance(chain_result, str):
            return chain_result

        chain, prompt_template = chain_result

        # Check if text is too long and needs chunking
        estimated_tokens = estimate_tokens(article_text)
        logger.info("Estimated tokens in article: %d", estimated_tokens)

        # Log article preview for debugging
        article_preview = (
            article_text[:500] + "..." if len(article_text) > 500 else article_text
        )
        logger.info("📄 Article content preview (first 500 chars): %s", article_preview)

        if estimated_tokens > 12000:  # Leave room for prompt and completion
            return _process_long_article_with_limit(
                chain, article_text, prompt_template, adjusted_max_lines
            )

        # Text is short enough, process normally
        return _process_short_article(
            chain, article_text, prompt_template, adjusted_max_lines
        )

    except (
        ValueError,
        KeyError,
        AttributeError,
        ConnectionError,
        TimeoutError,
        TypeError,
    ) as e:
        logger.error("Error during line-limited summarization: %s", str(e))
        return f"Error generating summary: {str(e)}"


def _create_error_response(error_message: str, intent: str, confidence: float) -> dict:
    """Create a standardized error response."""
    return {
        "summary": error_message,
        "method": "error",
        "intent": intent,
        "confidence": confidence,
    }


def _process_intent_aware_chunks(
    article_text: str, intent: str, confidence: float
) -> dict | None:
    """Process long articles with intent-aware chunking."""
    estimated_tokens = estimate_tokens(article_text)
    logger.info("Article estimated tokens: %d", estimated_tokens)

    if estimated_tokens <= 12000:
        return None

    # Handle long articles with chunking
    chunks = chunk_text_for_openai(article_text, max_chunk_tokens=10000)
    logger.info(
        "Article too long (%d tokens), splitting into %d chunks",
        estimated_tokens,
        len(chunks),
    )

    summaries = []
    chain = create_intent_aware_chain(intent, confidence)
    if not chain:
        return None

    for i, chunk in enumerate(chunks[:2]):  # Limit to first 2 chunks
        try:
            log_chatgpt_request(chain.prompt.template, chunk, i + 1)
            safe_chunk = sanitize_article_text(chunk)
            chunk_summary = chain.run(article_text=safe_chunk)
            summaries.append(chunk_summary)
            logger.info("✅ Chunk %d summarized successfully", i + 1)
        except (
            ValueError,
            KeyError,
            AttributeError,
            ConnectionError,
            TimeoutError,
        ) as e:
            logger.error("Error summarizing chunk %d: %s", i + 1, str(e))
            continue

    if not summaries:
        return None

    combined_summary = " ".join(summaries)
    return {
        "summary": combined_summary,
        "method": f"intent_aware_chunked_openai_{intent.lower()}",
        "intent": intent,
        "confidence": confidence,
        "chunks_processed": len(summaries),
    }


def _process_intent_aware_single(
    article_text: str, intent: str, confidence: float
) -> dict | None:
    """Process normal-sized articles with intent-aware summarization."""
    chain = create_intent_aware_chain(intent, confidence)
    if not chain:
        return None

    log_chatgpt_request(chain.prompt.template, article_text)
    safe_article_text = sanitize_article_text(article_text)
    summary = chain.run(article_text=safe_article_text)
    logger.info("✅ Intent-aware OpenAI summarization completed successfully")

    return {
        "summary": summary,
        "method": f"intent_aware_openai_{intent.lower()}",
        "intent": intent,
        "confidence": confidence,
    }


def summarize_article_with_intent(
    article_text: str,
    title: str = "Unknown",
    intent: str = "General",
    confidence: float = 0.5,
) -> dict:
    """
    Summarize an article using intent-aware prompting

    Args:
        article_text: The article content to summarize
        title: Article title for context
        intent: Detected intent category
        confidence: Confidence score of intent prediction

    Returns:
        Dictionary with summary, method used, and metadata
    """
    if not article_text or not article_text.strip():
        return _create_error_response(
            "No content available to summarize.", intent, confidence
        )

    # Check if content is too short to need summarization
    if len(article_text.split()) < 50:
        return {
            "summary": article_text,
            "method": "content_too_short",
            "intent": intent,
            "confidence": confidence,
        }

    # Try intent-aware OpenAI summarization first:
    # First try chunked processing for long articles
    chunks_result = _process_intent_aware_chunks(article_text, intent, confidence)
    if chunks_result:
        return chunks_result

    # Otherwise try single processing for normal articles
    single_result = _process_intent_aware_single(article_text, intent, confidence)
    if single_result:
        return single_result

    # Fallback to regular summarization
    logger.info("Falling back to regular summarization")
    regular_result = summarize_article_with_limit(article_text, title)
    if isinstance(regular_result, dict):
        regular_result["intent"] = intent
        regular_result["confidence"] = confidence
        return regular_result

    return {
        "summary": regular_result,
        "method": "fallback_basic",
        "intent": intent,
        "confidence": confidence,
    }
