"""
Unit tests for ArticleSelectionAgent
Tests the article selection functionality with proper mocking
"""

from unittest.mock import Mock, patch

import pytest

from backend.models.agents.article_selection_agent import ArticleSelectionAgent


class TestArticleSelectionAgent:
    """Test cases for ArticleSelectionAgent"""

    @patch("backend.models.agents.article_selection_agent.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.agents.article_selection_agent.initialize_agent")
    @patch("backend.models.agents.article_selection_agent.ConversationBufferMemory")
    @patch("backend.models.agents.article_selection_agent.Tool")
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
        agent = ArticleSelectionAgent(mock_llm)

        # Verify initialization
        assert agent.llm == mock_llm
        assert agent.agent == mock_agent
        mock_initialize_agent.assert_called_once()
        mock_memory.assert_called_once()
        mock_tool.assert_called_once()

    @patch("backend.models.agents.article_selection_agent.LANGCHAIN_AVAILABLE", False)
    def test_initialization_langchain_unavailable(self):
        """Test agent initialization when LangChain is unavailable"""
        mock_llm = Mock()

        with pytest.raises(ImportError, match="LangChain not available"):
            ArticleSelectionAgent(mock_llm)

    @patch("backend.models.agents.article_selection_agent.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.agents.article_selection_agent.initialize_agent")
    @patch("backend.models.agents.article_selection_agent.ConversationBufferMemory")
    @patch("backend.models.agents.article_selection_agent.Tool")
    def test_select_best_article_success(
        self, mock_tool, mock_memory, mock_initialize_agent
    ):
        """Test successful article selection"""
        # Mock LLM
        mock_llm = Mock()

        # Mock agent with successful response
        mock_agent = Mock()
        mock_agent.run.return_value = "I recommend: Artificial Intelligence"
        mock_initialize_agent.return_value = mock_agent

        # Initialize agent
        agent = ArticleSelectionAgent(mock_llm)

        # Test article selection
        article_options = [
            "Artificial Intelligence",
            "Machine Learning",
            "Deep Learning",
        ]
        result = agent.select_best_article("What is AI?", article_options)

        # Verify result
        assert result["selected_article"] == "Artificial Intelligence"
        assert result["method"] == "langchain_agent"
        assert result["confidence"] == 0.9
        mock_agent.run.assert_called_once()

    @patch("backend.models.agents.article_selection_agent.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.agents.article_selection_agent.initialize_agent")
    @patch("backend.models.agents.article_selection_agent.ConversationBufferMemory")
    @patch("backend.models.agents.article_selection_agent.Tool")
    def test_select_best_article_no_options(
        self, mock_tool, mock_memory, mock_initialize_agent
    ):
        """Test article selection with no options provided"""
        # Mock LLM and agent
        mock_llm = Mock()
        mock_agent = Mock()
        mock_initialize_agent.return_value = mock_agent

        # Initialize agent
        agent = ArticleSelectionAgent(mock_llm)

        # Test with empty options
        result = agent.select_best_article("What is AI?", [])

        # Verify result
        assert result["selected_article"] is None
        assert result["method"] == "no_options"

    @patch("backend.models.agents.article_selection_agent.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.agents.article_selection_agent.initialize_agent")
    @patch("backend.models.agents.article_selection_agent.ConversationBufferMemory")
    @patch("backend.models.agents.article_selection_agent.Tool")
    def test_select_best_article_agent_failure(
        self, mock_tool, mock_memory, mock_initialize_agent
    ):
        """Test article selection when agent fails"""
        # Mock LLM
        mock_llm = Mock()

        # Mock agent that raises exception
        mock_agent = Mock()
        mock_agent.run.side_effect = ValueError("Agent execution failed")
        mock_initialize_agent.return_value = mock_agent

        # Initialize agent
        agent = ArticleSelectionAgent(mock_llm)

        # Test article selection with failure
        article_options = ["Option 1", "Option 2", "Option 3"]
        result = agent.select_best_article("test query", article_options)

        # Verify fallback result (should return first option)
        assert result["selected_article"] == "Option 1"
        assert result["method"] == "fallback_first"
        assert result["confidence"] == 0.5

    @patch("backend.models.agents.article_selection_agent.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.agents.article_selection_agent.initialize_agent")
    @patch("backend.models.agents.article_selection_agent.ConversationBufferMemory")
    @patch("backend.models.agents.article_selection_agent.Tool")
    def test_extract_article_from_response(
        self, mock_tool, mock_memory, mock_initialize_agent
    ):
        """Test article extraction from agent response"""
        # Mock LLM and agent
        mock_llm = Mock()
        mock_agent = Mock()
        mock_initialize_agent.return_value = mock_agent

        # Initialize agent
        agent = ArticleSelectionAgent(mock_llm)

        # Test article options
        options = ["Quantum Physics", "Classical Mechanics", "Thermodynamics"]

        # Test various response formats
        test_cases = [
            ("I recommend Quantum Physics for this query", "Quantum Physics"),
            (
                "The best choice is classical mechanics",
                "Classical Mechanics",
            ),  # Case insensitive
            ("THERMODYNAMICS is the answer", "Thermodynamics"),  # Case insensitive
            ("None of these match", "Quantum Physics"),  # Falls back to first option
            ("", "Quantum Physics"),  # Empty response falls back to first option
        ]

        for response, expected in test_cases:
            result = agent._extract_article_from_response(response, options)
            assert result == expected

    @patch("backend.models.agents.article_selection_agent.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.agents.article_selection_agent.initialize_agent")
    @patch("backend.models.agents.article_selection_agent.ConversationBufferMemory")
    @patch("backend.models.agents.article_selection_agent.Tool")
    def test_tool_configuration(self, mock_tool, mock_memory, mock_initialize_agent):
        """Test that tools are configured correctly"""
        # Mock LLM and agent
        mock_llm = Mock()
        mock_agent = Mock()
        mock_initialize_agent.return_value = mock_agent

        # Initialize agent
        agent = ArticleSelectionAgent(mock_llm)

        # Verify tool was called with correct parameters
        mock_tool.assert_called_once()
        call_args = mock_tool.call_args

        # Check tool configuration
        assert call_args[1]["name"] == "get_article_preview"
        assert (
            "Get a preview of a Wikipedia article by title"
            in call_args[1]["description"]
        )
        # Verify the func is the WikipediaSearchTool.get_article_preview method
        assert callable(call_args[1]["func"])
