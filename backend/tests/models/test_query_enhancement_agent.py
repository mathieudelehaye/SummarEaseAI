"""
Unit tests for QueryEnhancementAgent
Tests the query enhancement functionality with proper mocking
"""

from unittest.mock import Mock, patch

import pytest

from backend.models.agents.query_enhancement_agent import QueryEnhancementAgent


class TestQueryEnhancementAgent:
    """Test cases for QueryEnhancementAgent"""

    @patch("backend.models.agents.query_enhancement_agent.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.agents.query_enhancement_agent.initialize_agent")
    @patch("backend.models.agents.query_enhancement_agent.ConversationBufferMemory")
    @patch("backend.models.agents.query_enhancement_agent.Tool")
    def test_initialization_success(
        self, mock_tool, mock_memory, mock_initialize_agent
    ):
        """Test successful agent initialization"""
        # Mock LLM
        mock_llm = Mock()

        # Mock agent
        mock_agent = Mock()
        mock_initialize_agent.return_value = mock_agent

        # Initialize agent
        agent = QueryEnhancementAgent(mock_llm)

        # Verify initialization
        assert agent.llm == mock_llm
        assert agent.agent == mock_agent
        mock_initialize_agent.assert_called_once()
        mock_memory.assert_called_once()
        mock_tool.assert_called_once()

    @patch("backend.models.agents.query_enhancement_agent.LANGCHAIN_AVAILABLE", False)
    def test_initialization_langchain_unavailable(self):
        """Test agent initialization when LangChain is unavailable"""
        mock_llm = Mock()

        with pytest.raises(ImportError, match="LangChain not available"):
            QueryEnhancementAgent(mock_llm)

    @patch("backend.models.agents.query_enhancement_agent.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.agents.query_enhancement_agent.initialize_agent")
    @patch("backend.models.agents.query_enhancement_agent.ConversationBufferMemory")
    @patch("backend.models.agents.query_enhancement_agent.Tool")
    def test_enhance_query_success(self, mock_tool, mock_memory, mock_initialize_agent):
        """Test successful query enhancement"""
        # Mock LLM
        mock_llm = Mock()

        # Mock agent with successful response
        mock_agent = Mock()
        mock_agent.run.return_value = "artificial intelligence"
        mock_initialize_agent.return_value = mock_agent

        # Initialize agent
        agent = QueryEnhancementAgent(mock_llm)

        # Test query enhancement
        result = agent.enhance_query("What is AI?")

        # Verify result
        assert result["original_query"] == "What is AI?"
        assert result["enhanced_query"] == "artificial intelligence"
        assert result["enhancement_method"] == "langchain_agent"
        assert result["confidence"] == 0.9
        mock_agent.run.assert_called_once()

    @patch("backend.models.agents.query_enhancement_agent.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.agents.query_enhancement_agent.initialize_agent")
    @patch("backend.models.agents.query_enhancement_agent.ConversationBufferMemory")
    @patch("backend.models.agents.query_enhancement_agent.Tool")
    def test_enhance_query_agent_failure(
        self, mock_tool, mock_memory, mock_initialize_agent
    ):
        """Test query enhancement when agent fails"""
        # Mock LLM
        mock_llm = Mock()

        # Mock agent that raises exception
        mock_agent = Mock()
        mock_agent.run.side_effect = ValueError("Agent execution failed")
        mock_initialize_agent.return_value = mock_agent

        # Initialize agent
        agent = QueryEnhancementAgent(mock_llm)

        # Test query enhancement with failure
        result = agent.enhance_query("Who was Albert Einstein?")

        # Verify fallback result
        assert result["original_query"] == "Who was Albert Einstein?"
        assert result["enhanced_query"] == "Albert Einstein"
        assert result["enhancement_method"] == "rule_based_fallback"
        assert result["confidence"] == 0.6

    @patch("backend.models.agents.query_enhancement_agent.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.agents.query_enhancement_agent.initialize_agent")
    @patch("backend.models.agents.query_enhancement_agent.ConversationBufferMemory")
    @patch("backend.models.agents.query_enhancement_agent.Tool")
    def test_extract_query_from_response(
        self, mock_tool, mock_memory, mock_initialize_agent
    ):
        """Test query extraction from agent response"""
        # Mock LLM and agent
        mock_llm = Mock()
        mock_agent = Mock()
        mock_initialize_agent.return_value = mock_agent

        # Initialize agent
        agent = QueryEnhancementAgent(mock_llm)

        # Test various response formats
        test_cases = [
            (
                "Thought: Let me enhance this\nAction: search\nObservation: Found results\nquantum mechanics",
                "quantum mechanics",
            ),
            ('"machine learning algorithms"', "machine learning algorithms"),
            ("artificial intelligence\nSome other text", "artificial intelligence"),
            ("", "original query"),  # Empty response falls back to original
        ]

        for response, expected in test_cases:
            result = agent._extract_query_from_response(response, "original query")
            assert result == expected

    @patch("backend.models.agents.query_enhancement_agent.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.agents.query_enhancement_agent.initialize_agent")
    @patch("backend.models.agents.query_enhancement_agent.ConversationBufferMemory")
    @patch("backend.models.agents.query_enhancement_agent.Tool")
    def test_fallback_enhancement_various_queries(
        self, mock_tool, mock_memory, mock_initialize_agent
    ):
        """Test fallback enhancement for various query types"""
        # Mock LLM and agent
        mock_llm = Mock()
        mock_agent = Mock()
        mock_initialize_agent.return_value = mock_agent

        # Initialize agent
        agent = QueryEnhancementAgent(mock_llm)

        # Test various query patterns
        test_cases = [
            ("Who were the Beatles?", "the Beatles"),
            ("Who was Marie Curie?", "Marie Curie"),
            ("What is quantum physics?", "quantum physics"),
            (
                "Tell me about climate change",
                "Tell me about climate change",
            ),  # No change
        ]

        for original, expected in test_cases:
            result = agent._fallback_enhancement(original)
            assert result["original_query"] == original
            assert result["enhanced_query"] == expected
            assert result["enhancement_method"] == "rule_based_fallback"
            assert result["confidence"] == 0.6
