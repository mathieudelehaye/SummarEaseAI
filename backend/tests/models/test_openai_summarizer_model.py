"""
Unit tests for OpenAI Summarizer Model
Tests the data access layer for OpenAI API interactions
"""

import os
from unittest.mock import Mock, patch

from backend.models.openai_summarizer_model import (
    chunk_text_for_openai,
    create_intent_aware_chain,
    create_line_limited_chain,
    create_summarization_chain,
    estimate_tokens,
    get_openai_api_key,
    sanitize_article_text,
)


class TestUtilityFunctions:
    """Test utility functions in the model"""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    def test_get_openai_api_key_exists(self):
        """Test getting API key when it exists"""
        key = get_openai_api_key()
        assert key == "test-key-123"

    @patch.dict("os.environ", {}, clear=True)
    def test_get_openai_api_key_missing(self):
        """Test getting API key when it doesn't exist"""
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        key = get_openai_api_key()
        assert key is None

    def test_estimate_tokens_calculation(self):
        """Test token estimation calculation"""
        # Test known cases
        assert estimate_tokens("test") == 1  # 4 chars / 4 = 1
        assert estimate_tokens("testing text") == 3  # 12 chars / 4 = 3
        assert estimate_tokens("") == 0

    def test_sanitize_article_text_curly_braces(self):
        """Test sanitization of curly braces"""
        text = "This has {single} and {{double}} braces"
        result = sanitize_article_text(text)
        assert result == "This has (single) and ((double)) braces"

    def test_sanitize_article_text_empty(self):
        """Test sanitization of empty text"""
        assert sanitize_article_text("") == ""
        assert sanitize_article_text(None) == ""

    @patch("backend.models.openai_summarizer_model.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.openai_summarizer_model.RecursiveCharacterTextSplitter")
    def test_chunk_text_for_openai_with_langchain(self, mock_splitter_class):
        """Test text chunking with LangChain available"""
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = ["chunk1", "chunk2"]
        mock_splitter_class.return_value = mock_splitter

        result = chunk_text_for_openai("long text here", 1000)
        assert result == ["chunk1", "chunk2"]

    @patch("backend.models.openai_summarizer_model.LANGCHAIN_AVAILABLE", False)
    def test_chunk_text_for_openai_fallback(self):
        """Test text chunking fallback when LangChain unavailable"""
        text = "word " * 1000  # Long text
        result = chunk_text_for_openai(text, 100)
        assert isinstance(result, list)
        assert len(result) > 0


class TestChainCreation:
    """Test LangChain chain creation functions"""

    @patch("backend.models.openai_summarizer_model.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.openai_summarizer_model.get_openai_api_key")
    @patch("backend.models.openai_summarizer_model.ChatOpenAI")
    @patch("backend.models.openai_summarizer_model.LLMChain")
    @patch("backend.models.openai_summarizer_model.PromptTemplate")
    def test_create_summarization_chain_success(
        self, mock_prompt_template, mock_llm_chain, mock_chat_openai, mock_get_key
    ):
        """Test successful chain creation"""
        mock_get_key.return_value = "test-key"
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        mock_prompt = Mock()
        mock_prompt_template.return_value = mock_prompt
        mock_chain = Mock()
        mock_llm_chain.return_value = mock_chain

        result = create_summarization_chain()
        assert result == mock_chain

    @patch("backend.models.openai_summarizer_model.LANGCHAIN_AVAILABLE", False)
    def test_create_summarization_chain_no_langchain(self):
        """Test chain creation when LangChain is not available"""
        result = create_summarization_chain()
        assert result is None

    @patch("backend.models.openai_summarizer_model.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.openai_summarizer_model.get_openai_api_key")
    def test_create_summarization_chain_no_api_key(self, mock_get_key):
        """Test chain creation when API key is not available"""
        mock_get_key.return_value = None
        result = create_summarization_chain()
        assert result is None

    @patch("backend.models.openai_summarizer_model.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.openai_summarizer_model.get_openai_api_key")
    @patch("backend.models.openai_summarizer_model.ChatOpenAI")
    @patch("backend.models.openai_summarizer_model.LLMChain")
    @patch("backend.models.openai_summarizer_model.PromptTemplate")
    def test_create_line_limited_chain(self, mock_prompt_template, mock_llm_chain, mock_chat_openai, mock_get_key):
        """Test line-limited chain creation"""
        mock_get_key.return_value = "test-key"
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        mock_prompt = Mock()
        mock_prompt_template.return_value = mock_prompt
        mock_chain = Mock()
        mock_llm_chain.return_value = mock_chain

        result = create_line_limited_chain(20)
        assert result == mock_chain

    @patch("backend.models.openai_summarizer_model.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.openai_summarizer_model.get_openai_api_key")
    @patch("backend.models.openai_summarizer_model.ChatOpenAI")
    @patch("backend.models.openai_summarizer_model.LLMChain")
    @patch("backend.models.openai_summarizer_model.PromptTemplate")
    def test_create_intent_aware_chain_science(self, mock_prompt_template, mock_llm_chain, mock_chat_openai, mock_get_key):
        """Test intent-aware chain creation for Science"""
        mock_get_key.return_value = "test-key"
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        mock_prompt = Mock()
        mock_prompt_template.return_value = mock_prompt
        mock_chain = Mock()
        mock_llm_chain.return_value = mock_chain

        result = create_intent_aware_chain("Science", 0.8)
        assert result == mock_chain

    @patch("backend.models.openai_summarizer_model.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.openai_summarizer_model.get_openai_api_key")
    @patch("backend.models.openai_summarizer_model.ChatOpenAI")
    @patch("backend.models.openai_summarizer_model.LLMChain")
    @patch("backend.models.openai_summarizer_model.PromptTemplate")
    def test_create_intent_aware_chain_low_confidence(
        self, mock_prompt_template, mock_llm_chain, mock_chat_openai, mock_get_key
    ):
        """Test intent-aware chain creation with low confidence falls back to general"""
        mock_get_key.return_value = "test-key"
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        mock_prompt = Mock()
        mock_prompt_template.return_value = mock_prompt
        mock_chain = Mock()
        mock_llm_chain.return_value = mock_chain

        result = create_intent_aware_chain("Science", 0.3)  # Low confidence
        assert result == mock_chain


class TestErrorHandling:
    """Test error handling in the model"""

    @patch("backend.models.openai_summarizer_model.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.openai_summarizer_model.RecursiveCharacterTextSplitter")
    def test_chunk_text_splitter_exception(self, mock_splitter_class):
        """Test chunking when text splitter raises exception"""
        mock_splitter_class.side_effect = Exception("Splitter error")

        # Should fall back to simple chunking
        result = chunk_text_for_openai("test text", 1000)
        
        # Should return a list with the original text
        assert isinstance(result, list)
        assert len(result) > 0
        assert "test text" in result[0]

    def test_sanitize_text_with_none_input(self):
        """Test sanitization with None input"""
        result = sanitize_article_text(None)
        assert result == ""

    def test_estimate_tokens_with_none_input(self):
        """Test token estimation with None input"""
        # This should handle gracefully or raise appropriate error
        try:
            result = estimate_tokens(None)
            assert result == 0
        except (TypeError, AttributeError):
            # Either outcome is acceptable for None input
            pass
