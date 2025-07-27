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
        logging.warning(
            "LangChain not available. OpenAI summarization will be disabled."
        )

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model functions
from models.openai_summarizer_model import (
    get_openai_api_key, 
    create_summarization_chain,
    create_line_limited_chain,
    create_intent_aware_chain,
    estimate_tokens,
    chunk_text_for_openai,
    sanitize_article_text,
    get_summarization_status,
    parse_summary_output,
    get_openai_summarizer_model
)


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











def log_chatgpt_request(
    prompt_template: str, article_text: str, chunk_number: int = None
):
    """Log the exact request being sent to ChatGPT for debugging"""
    # Safely replace template placeholders to avoid format code errors
    safe_article_text = str(article_text).replace("{", "{{").replace("}", "}}")
    full_prompt = prompt_template.format(article_text=safe_article_text)

    chunk_info = f" (chunk {chunk_number})" if chunk_number else ""
    logger.info(f"🚀 FULL PROMPT being sent to ChatGPT{chunk_info}:")
    logger.info(
        f"📋 Prompt length: {len(full_prompt)} characters, ~{estimate_tokens(full_prompt)} tokens"
    )

    # Log the actual formatted prompt (truncated for readability)
    if len(full_prompt) > 1000:
        logger.info(f"📄 PROMPT PREVIEW: {full_prompt[:500]}...{full_prompt[-500:]}")
    else:
        logger.info(f"📄 FULL PROMPT: {full_prompt}")

    logger.info("=" * 80)








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
            return (
                "Error: Could not initialize OpenAI summarization. Check your API key."
            )

        # Check if text is too long and needs chunking
        estimated_tokens = estimate_tokens(article_text)
        logger.info(f"Estimated tokens in article: {estimated_tokens}")

        # Log article preview for debugging
        article_preview = (
            article_text[:500] + "..." if len(article_text) > 500 else article_text
        )
        logger.info(f"📄 Article content preview (first 500 chars): {article_preview}")

        if estimated_tokens > 12000:  # Leave room for prompt and completion
            logger.info("Article too long, chunking for processing...")
            chunks = chunk_text_for_openai(article_text, max_chunk_tokens=12000)
            summaries = []

            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                safe_chunk = sanitize_article_text(chunk)
                chunk_summary = chain.run(article_text=safe_chunk)
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
                    final_summary = chain.run(
                        article_text=combined_text[: 12000 * 4]
                    )  # Rough char limit
                else:
                    final_summary = chain.run(article_text=combined_text)
                return (
                    final_summary.strip()
                    if final_summary
                    else "Error: Final summary generation failed"
                )
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
            return (
                "Error: Could not initialize OpenAI summarization. Check your API key."
            )

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
        article_preview = (
            article_text[:500] + "..." if len(article_text) > 500 else article_text
        )
        logger.info(f"📄 Article content preview (first 500 chars): {article_preview}")

        if estimated_tokens > 12000:  # Leave room for prompt and completion
            logger.info("Article too long, chunking for processing...")
            chunks = chunk_text_for_openai(article_text, max_chunk_tokens=12000)
            summaries = []

            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)} (line-limited)")

                # Log what's being sent to ChatGPT for this chunk
                chunk_preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
                logger.info(
                    f"🤖 Sending to ChatGPT chunk {i+1} preview: {chunk_preview}"
                )

                # Log the exact prompt being sent to ChatGPT
                log_chatgpt_request(prompt_template, chunk, chunk_number=i + 1)

                chunk_summary = chain.run(article_text=chunk)

                # Log what ChatGPT returned for this chunk
                if chunk_summary and chunk_summary.strip():
                    logger.info(
                        f"✅ ChatGPT returned for chunk {i+1}: {chunk_summary.strip()[:200]}..."
                    )
                    summaries.append(chunk_summary.strip())
                else:
                    logger.warning(
                        f"❌ ChatGPT returned empty/invalid response for chunk {i+1}"
                    )

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
                        output_parser=SummaryOutputParser(),
                    )

                    # Log what's being sent to ChatGPT for final combination
                    combine_preview = (
                        combined_text[:500] + "..."
                        if len(combined_text) > 500
                        else combined_text
                    )
                    logger.info(
                        f"🔄 Sending combined summaries to ChatGPT for final synthesis: {combine_preview}"
                    )

                    if estimate_tokens(combined_text) > 12000:
                        final_summary = combine_chain.run(
                            article_text=combined_text[: 12000 * 4]
                        )
                    else:
                        final_summary = combine_chain.run(article_text=combined_text)

                    # Log final ChatGPT response
                    if final_summary:
                        logger.info(
                            f"🎯 Final ChatGPT response: {final_summary.strip()[:300]}..."
                        )
                    else:
                        logger.warning("❌ Final ChatGPT response was empty")

                except Exception as e:
                    logger.warning(
                        f"Error creating specialized combine chain: {e}, falling back to regular chain"
                    )
                    # Fallback to original method
                    if estimate_tokens(combined_text) > 12000:
                        final_summary = chain.run(
                            article_text=combined_text[: 12000 * 4]
                        )
                    else:
                        final_summary = chain.run(article_text=combined_text)

                if not final_summary:
                    return "Error: Final summary generation failed"

                # Post-process to ensure line limit
                lines = final_summary.strip().split("\n")
                if len(lines) > max_lines:
                    final_summary = "\n".join(lines[:max_lines])

                return final_summary.strip()
            else:
                # Post-process single summary to ensure line limit
                lines = summaries[0].split("\n")
                if len(lines) > max_lines:
                    summary = "\n".join(lines[:max_lines])
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
            lines = summary.strip().split("\n")
            if len(lines) > max_lines:
                summary = "\n".join(lines[:max_lines])

            logger.info(f"Summary generated successfully ({len(lines)} lines)")
            return summary.strip()

    except Exception as e:
        logger.error(f"Error during line-limited summarization: {str(e)}")
        return f"Error generating summary: {str(e)}"








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
        return {
            "summary": "No content available to summarize.",
            "method": "error",
            "intent": intent,
            "confidence": confidence,
        }

    # Check if content is too short to need summarization
    if len(article_text.split()) < 50:
        return {
            "summary": article_text,
            "method": "content_too_short",
            "intent": intent,
            "confidence": confidence,
        }

    # Try intent-aware OpenAI summarization first
    if LANGCHAIN_AVAILABLE:
        api_key = get_openai_api_key()
        if api_key:
            try:
                # Estimate tokens
                estimated_tokens = estimate_tokens(article_text)
                logger.info(f"Article estimated tokens: {estimated_tokens}")

                if estimated_tokens > 12000:  # Leave room for prompt and response
                    # Handle long articles with chunking
                    chunks = chunk_text_for_openai(article_text, max_chunk_tokens=10000)
                    logger.info(
                        f"Article too long ({estimated_tokens} tokens), splitting into {len(chunks)} chunks"
                    )

                    summaries = []
                    chain = create_intent_aware_chain(intent, confidence)
                    if chain:
                        for i, chunk in enumerate(
                            chunks[:2]
                        ):  # Limit to first 2 chunks
                            try:
                                log_chatgpt_request(chain.prompt.template, chunk, i + 1)
                                safe_chunk = sanitize_article_text(chunk)
                                chunk_summary = chain.run(article_text=safe_chunk)
                                summaries.append(chunk_summary)
                                logger.info(f"✅ Chunk {i+1} summarized successfully")
                            except Exception as e:
                                logger.error(f"Error summarizing chunk {i+1}: {str(e)}")
                                continue

                        if summaries:
                            combined_summary = " ".join(summaries)
                            return {
                                "summary": combined_summary,
                                "method": f"intent_aware_chunked_openai_{intent.lower()}",
                                "intent": intent,
                                "confidence": confidence,
                                "chunks_processed": len(summaries),
                            }
                else:
                    # Handle normal-sized articles
                    chain = create_intent_aware_chain(intent, confidence)
                    if chain:
                        log_chatgpt_request(chain.prompt.template, article_text)
                        safe_article_text = sanitize_article_text(article_text)
                        summary = chain.run(article_text=safe_article_text)
                        logger.info(
                            "✅ Intent-aware OpenAI summarization completed successfully"
                        )
                        return {
                            "summary": summary,
                            "method": f"intent_aware_openai_{intent.lower()}",
                            "intent": intent,
                            "confidence": confidence,
                        }

            except Exception as e:
                logger.error(f"Error in intent-aware OpenAI summarization: {str(e)}")

    # Fallback to regular summarization
    logger.info("Falling back to regular summarization")
    regular_result = summarize_article_with_limit(article_text, title)
    if isinstance(regular_result, dict):
        regular_result["intent"] = intent
        regular_result["confidence"] = confidence
        return regular_result
    else:
        return {
            "summary": regular_result,
            "method": "fallback_basic",
            "intent": intent,
            "confidence": confidence,
        }



