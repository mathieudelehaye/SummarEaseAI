"""
Unit tests for MultiSourceAgentService
Tests the core multi-source intelligence functionality
"""

from unittest.mock import Mock

# Import the class under test
from backend.services.multi_source_summary_service import (
    MultiSourceAgentService,
    RateLimitConfig,
)


class TestRateLimitConfig:
    """Test rate limiting configuration"""

    def test_get_limits_minimal_mode(self):
        """Test minimal mode rate limits"""
        limits = RateLimitConfig.get_limits_for_mode("MINIMAL")
        assert limits["max_articles"] == 1
        assert limits["max_secondary_queries"] == 0  # Changed from 1 to 0
        assert limits["max_wikipedia_searches"] == 1
        assert limits["enable_openai"] is False
        assert limits["enable_agents"] is False

    def test_get_limits_balanced_mode(self):
        """Test balanced mode rate limits"""
        limits = RateLimitConfig.get_limits_for_mode("BALANCED")
        assert limits["max_articles"] == 3
        assert limits["max_secondary_queries"] == 3  # Changed from 4 to 3
        assert limits["max_wikipedia_searches"] == 6
        assert limits["enable_openai"] is True
        assert limits["enable_agents"] is True

    def test_get_limits_comprehensive_mode(self):
        """Test comprehensive mode rate limits"""
        limits = RateLimitConfig.get_limits_for_mode("COMPREHENSIVE")
        assert limits["max_articles"] == 6  # Changed from 5 to 6
        assert limits["max_secondary_queries"] == 6
        assert limits["max_wikipedia_searches"] == 12
        assert limits["enable_openai"] is True
        assert limits["enable_agents"] is True

    def test_get_limits_invalid_mode(self):
        """Test invalid mode defaults to balanced"""
        limits = RateLimitConfig.get_limits_for_mode("INVALID")
        assert limits["max_articles"] == 3
        assert limits["max_secondary_queries"] == 3  # Changed from 4 to 3
        assert limits["max_wikipedia_searches"] == 6
        assert limits["enable_openai"] is True
        assert limits["enable_agents"] is True


class TestMultiSourceAgent:
    """Test cases for MultiSourceAgentService"""

    def test_init_balanced_mode(
        self, mock_agent_system, mock_query_gen, mock_get_classifier
    ):
        """Test agent initialization in balanced mode"""
        agent = MultiSourceAgentService(cost_mode="BALANCED")

        assert agent.config["cost_mode"] == "BALANCED"
        assert agent.config["limits"]["max_articles"] == 3
        assert (
            agent.config["limits"]["max_secondary_queries"] == 3
        )  # Changed from 4 to 3
        assert agent.config["limits"]["max_wikipedia_searches"] == 6
        assert agent.config["limits"]["enable_openai"] is True
        assert agent.config["limits"]["enable_agents"] is True

    def test_init_minimal_mode(
        self, mock_agent_system, mock_query_gen, mock_get_classifier
    ):
        """Test agent initialization in minimal mode"""
        agent = MultiSourceAgentService(cost_mode="MINIMAL")

        assert agent.config["cost_mode"] == "MINIMAL"
        assert agent.config["limits"]["max_articles"] == 1
        assert (
            agent.config["limits"]["max_secondary_queries"] == 0
        )  # Changed from 1 to 0
        assert agent.config["limits"]["max_wikipedia_searches"] == 1
        assert agent.config["limits"]["enable_openai"] is False
        assert agent.config["limits"]["enable_agents"] is False

    def test_plan_search_strategy_with_openai(
        self, mock_agent_system, mock_query_gen_class, mock_get_classifier
    ):
        """Test search strategy planning with OpenAI"""
        # Setup mocks
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_get_classifier.return_value = mock_classifier

        mock_query_gen = Mock()
        mock_query_gen_class.return_value = mock_query_gen

        # Mock OpenAI response
        mock_search_plan = {
            "secondary_queries": {
                "contextual": ["AI history", "AI principles"],
                "related": ["machine learning"],
            },
            "synthesis_focus": "comprehensive",
            "search_strategy": "openai_generated",
        }
        mock_query_gen.generate_comprehensive_search_plan.return_value = (
            mock_search_plan
        )

        agent = MultiSourceAgentService(cost_mode="BALANCED")

        # Test the actual method that exists
        queries = agent._generate_search_queries(
            "artificial intelligence", "Technology"
        )
        assert isinstance(queries, list)
        assert len(queries) > 0

    def test_plan_search_strategy_rate_limited(
        self, mock_agent_system, mock_query_gen_class, mock_get_classifier
    ):
        """Test search strategy planning when rate limited"""
        agent = MultiSourceAgentService(
            cost_mode="MINIMAL"
        )  # This mode disables OpenAI
        strategy = agent._generate_search_queries("test query", "General")
        assert isinstance(strategy, list)
        assert len(strategy) > 0

    def test_generate_fallback_search_plan_biography(
        self, mock_agent_system, mock_query_gen, mock_get_classifier
    ):
        """Test fallback search plan generation for biography intent"""
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_get_classifier.return_value = mock_classifier

        agent = MultiSourceAgentService()

        plan = agent._generate_search_queries("Albert Einstein", "Biography")
        assert isinstance(plan, list)
        assert len(plan) > 0

    def test_generate_fallback_search_plan_science(
        self, mock_agent_system, mock_query_gen, mock_get_classifier
    ):
        """Test fallback search plan generation for science intent"""
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_get_classifier.return_value = mock_classifier

        agent = MultiSourceAgentService()

        plan = agent._generate_search_queries("quantum physics", "Science")
        assert isinstance(plan, list)
        assert len(plan) > 0


    def test_gather_articles_rate_limited(
        self, mock_agent_system_class, mock_query_gen, mock_get_classifier
    ):
        """Test article gathering when rate limited"""
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_get_classifier.return_value = mock_classifier

        mock_agent_system = Mock()
        mock_agent_system_class.return_value = mock_agent_system

        agent = MultiSourceAgentService(cost_mode="MINIMAL")

        strategy = ["query1", "query2", "query3"]

        # Mock successful agent response
        mock_agent_result = {
            "article_info": {"title": "Test Article", "content": "Test content"},
            "enhancement_result": {
                "enhanced_query": "enhanced query",
                "agent_reasoning": "test reasoning",
            },
            "search_results": ["result1"],
        }
        mock_agent_system.intelligent_wikipedia_search.return_value = mock_agent_result

        articles = agent._search_and_gather_articles(strategy, "test query")
        assert isinstance(articles, list)

    def test_analyze_intent_with_bert(
        self, mock_agent_system, mock_query_gen, mock_get_classifier
    ):
        """Test intent analysis with BERT classifier"""
        # Mock BERT classifier
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_classifier.classify_text.return_value = {"intent": "Technology"}
        mock_get_classifier.return_value = mock_classifier

        agent = MultiSourceAgentService()
        result = agent._classify_intent("artificial intelligence")
        assert isinstance(result, str)
        # The actual categories are different from BERT categories
        assert result in [
            "general_knowledge",
            "biography",
            "definition",
            "historical_event",
            "technology",
        ]

    def test_analyze_intent_fallback(
        self, mock_agent_system, mock_query_gen, mock_get_classifier
    ):
        """Test intent analysis fallback when BERT fails"""
        # Mock BERT classifier to fail
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = False
        mock_get_classifier.return_value = mock_classifier

        agent = MultiSourceAgentService()
        result = agent._classify_intent("test query")
        assert isinstance(result, str)
        # The actual categories are different from BERT categories
        assert result in ["general_knowledge", "biography", "definition", "finance"]


class TestMultiSourceAgentIntegration:
    """Integration tests for MultiSourceAgentService"""

    def test_cost_mode_consistency(
        self, mock_agent_system, mock_query_gen, mock_get_classifier
    ):
        """Test that cost modes are applied consistently"""
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_get_classifier.return_value = mock_classifier

        modes = ["MINIMAL", "BALANCED", "COMPREHENSIVE"]

        for mode in modes:
            agent = MultiSourceAgentService(cost_mode=mode)
            limits = RateLimitConfig.get_limits_for_mode(mode)

            assert agent.config["cost_mode"] == mode
            assert agent.config["limits"] == limits

    def test_tracking_variables_initialization(
        self, mock_agent_system, mock_query_gen, mock_get_classifier
    ):
        """Test that tracking variables are properly initialized"""
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_get_classifier.return_value = mock_classifier

        agent = MultiSourceAgentService()

        assert agent.usage["openai_calls_made"] == 0
        assert agent.usage["wikipedia_calls_made"] == 0
        assert agent.usage["articles_processed"] == 0

    def test_error_handling_in_planning(
        self, mock_agent_system, mock_query_gen_class, mock_get_classifier
    ):
        """Test error handling during search strategy planning"""
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_get_classifier.return_value = mock_classifier

        mock_query_gen = Mock()
        mock_query_gen_class.return_value = mock_query_gen
        mock_query_gen.generate_secondary_queries.side_effect = Exception(
            "OpenAI API error"
        )

        agent = MultiSourceAgentService()

        # Should not raise exception, should fall back to rule-based planning
        strategy = agent._generate_search_queries("test query", "Technology")
        assert isinstance(strategy, list)
        assert len(strategy) > 0
