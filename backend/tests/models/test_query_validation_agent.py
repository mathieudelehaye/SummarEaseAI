"""
Unit tests for QueryValidationAgent
Tests the query validation functionality with proper mocking
"""

from unittest.mock import Mock, patch

import pytest

from backend.models.agents.query_validation_agent import QueryValidationAgent


class TestQueryValidationAgent:
    """Test cases for QueryValidationAgent"""

    @patch("backend.models.agents.query_validation_agent.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.agents.query_validation_agent.WIKIPEDIA_AVAILABLE", True)
    @patch("backend.models.agents.query_validation_agent.initialize_agent")
    @patch("backend.models.agents.query_validation_agent.ConversationBufferMemory")
    @patch("backend.models.agents.query_validation_agent.Tool")
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
        agent = QueryValidationAgent(mock_llm)

        # Verify initialization
        assert agent.llm == mock_llm
        assert agent.agent == mock_agent
        mock_initialize_agent.assert_called_once()
        mock_memory.assert_called_once()
        # Should create 3 tools: wikipedia_search, wikipedia_suggest, get_article_preview
        assert mock_tool.call_count == 3

    @patch("backend.models.agents.query_validation_agent.LANGCHAIN_AVAILABLE", False)
    def test_initialization_langchain_unavailable(self):
        """Test agent initialization when LangChain is unavailable"""
        mock_llm = Mock()

        with pytest.raises(ImportError, match="LangChain not available"):
            QueryValidationAgent(mock_llm)

    @patch("backend.models.agents.query_validation_agent.WIKIPEDIA_AVAILABLE", False)
    def test_initialization_wikipedia_unavailable(self):
        """Test agent initialization when Wikipedia is unavailable"""
        mock_llm = Mock()

        with pytest.raises(ImportError, match="Wikipedia not available"):
            QueryValidationAgent(mock_llm)

    @patch("backend.models.agents.query_validation_agent.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.agents.query_validation_agent.WIKIPEDIA_AVAILABLE", True)
    @patch("backend.models.agents.query_validation_agent.initialize_agent")
    @patch("backend.models.agents.query_validation_agent.ConversationBufferMemory")
    @patch("backend.models.agents.query_validation_agent.Tool")
    def test_validate_query_very_likely(
        self, mock_tool, mock_memory, mock_initialize_agent
    ):
        """Test query validation returning 'Very likely'"""
        # Mock LLM
        mock_llm = Mock()

        # Mock agent with successful response
        mock_agent = Mock()
        mock_agent.run.return_value = (
            "Based on my search, this query is Very likely to find relevant articles."
        )
        mock_initialize_agent.return_value = mock_agent

        # Initialize agent
        agent = QueryValidationAgent(mock_llm)

        # Test query validation
        result = agent.validate_query("Albert Einstein")

        # Verify result
        assert result["query"] == "Albert Einstein"
        assert result["viability"] == "Very likely"
        assert result["validation_method"] == "langchain_agent_with_rag"
        assert result["confidence"] == 0.95
        mock_agent.run.assert_called_once()

    @patch("backend.models.agents.query_validation_agent.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.agents.query_validation_agent.WIKIPEDIA_AVAILABLE", True)
    @patch("backend.models.agents.query_validation_agent.initialize_agent")
    @patch("backend.models.agents.query_validation_agent.ConversationBufferMemory")
    @patch("backend.models.agents.query_validation_agent.Tool")
    def test_validate_query_very_unlikely(
        self, mock_tool, mock_memory, mock_initialize_agent
    ):
        """Test query validation returning 'Very unlikely'"""
        # Mock LLM
        mock_llm = Mock()

        # Mock agent with negative response
        mock_agent = Mock()
        mock_agent.run.return_value = (
            "After searching, this is Very unlikely to find relevant content."
        )
        mock_initialize_agent.return_value = mock_agent

        # Initialize agent
        agent = QueryValidationAgent(mock_llm)

        # Test query validation
        result = agent.validate_query("xyzabc123nonexistent")

        # Verify result
        assert result["query"] == "xyzabc123nonexistent"
        assert result["viability"] == "Very unlikely"
        assert result["validation_method"] == "langchain_agent_with_rag"
        assert result["confidence"] == 0.95

    @patch("backend.models.agents.query_validation_agent.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.agents.query_validation_agent.WIKIPEDIA_AVAILABLE", True)
    @patch("backend.models.agents.query_validation_agent.initialize_agent")
    @patch("backend.models.agents.query_validation_agent.ConversationBufferMemory")
    @patch("backend.models.agents.query_validation_agent.Tool")
    def test_validate_query_agent_failure(
        self, mock_tool, mock_memory, mock_initialize_agent
    ):
        """Test query validation when agent fails"""
        # Mock LLM
        mock_llm = Mock()

        # Mock agent that raises exception
        mock_agent = Mock()
        mock_agent.run.side_effect = ValueError("Agent execution failed")
        mock_initialize_agent.return_value = mock_agent

        # Initialize agent
        agent = QueryValidationAgent(mock_llm)

        # Test query validation with failure
        result = agent.validate_query("test query")

        # Verify fallback result
        assert result["query"] == "test query"
        assert (
            result["viability"] == "Very likely"
        )  # Long enough query defaults to likely
        assert result["validation_method"] == "fallback_heuristic"
        assert result["confidence"] == 0.5

    @patch("backend.models.agents.query_validation_agent.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.agents.query_validation_agent.WIKIPEDIA_AVAILABLE", True)
    @patch("backend.models.agents.query_validation_agent.initialize_agent")
    @patch("backend.models.agents.query_validation_agent.ConversationBufferMemory")
    @patch("backend.models.agents.query_validation_agent.Tool")
    def test_extract_viability_from_response(
        self, mock_tool, mock_memory, mock_initialize_agent
    ):
        """Test viability extraction from agent response"""
        # Mock LLM and agent
        mock_llm = Mock()
        mock_agent = Mock()
        mock_initialize_agent.return_value = mock_agent

        # Initialize agent
        agent = QueryValidationAgent(mock_llm)

        # Test various response formats
        test_cases = [
            ("This query is Very likely to succeed", "Very likely"),
            ("VERY LIKELY to find articles", "Very likely"),
            ("The result is very unlikely", "Very unlikely"),
            ("VERY UNLIKELY to find content", "Very unlikely"),
            ("Unclear response without keywords", "Very likely"),  # Default to likely
            ("", "Very likely"),  # Empty response defaults to likely
        ]

        for response, expected in test_cases:
            result = agent._extract_viability_from_response(response)
            assert result == expected

    @patch("backend.models.agents.query_validation_agent.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.agents.query_validation_agent.WIKIPEDIA_AVAILABLE", True)
    @patch("backend.models.agents.query_validation_agent.initialize_agent")
    @patch("backend.models.agents.query_validation_agent.ConversationBufferMemory")
    @patch("backend.models.agents.query_validation_agent.Tool")
    def test_fallback_validation(self, mock_tool, mock_memory, mock_initialize_agent):
        """Test fallback validation logic"""
        # Mock LLM and agent
        mock_llm = Mock()
        mock_agent = Mock()
        mock_initialize_agent.return_value = mock_agent

        # Initialize agent
        agent = QueryValidationAgent(mock_llm)

        # Test various query lengths
        test_cases = [
            ("x", "Very unlikely"),  # Too short
            ("xy", "Very unlikely"),  # Too short
            ("normal query", "Very likely"),  # Normal length
            ("this is a longer query with more words", "Very likely"),  # Long query
        ]

        for query, expected in test_cases:
            result = agent._fallback_validation(query)
            assert result["query"] == query
            assert result["viability"] == expected
            assert result["validation_method"] == "fallback_heuristic"
            assert result["confidence"] == 0.5
            assert result["agent_response"] == "Agent failed, using fallback"
