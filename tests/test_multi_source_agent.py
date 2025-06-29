"""
Unit tests for MultiSourceAgent
Tests the core multi-source intelligence functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

# Import the class under test
from utils.multi_source_agent import MultiSourceAgent, RateLimitConfig


class TestRateLimitConfig:
    """Test rate limiting configuration"""
    
    def test_get_limits_minimal_mode(self):
        """Test minimal mode rate limits"""
        limits = RateLimitConfig.get_limits_for_mode("MINIMAL")
        assert limits['max_articles'] == 1
        assert limits['max_secondary_queries'] == 0  # Changed from 1 to 0
        assert limits['max_wikipedia_searches'] == 1
        assert limits['enable_openai'] is False
        assert limits['enable_agents'] is False
    
    def test_get_limits_balanced_mode(self):
        """Test balanced mode rate limits"""
        limits = RateLimitConfig.get_limits_for_mode("BALANCED")
        assert limits['max_articles'] == 3
        assert limits['max_secondary_queries'] == 3  # Changed from 4 to 3
        assert limits['max_wikipedia_searches'] == 6
        assert limits['enable_openai'] is True
        assert limits['enable_agents'] is True
    
    def test_get_limits_comprehensive_mode(self):
        """Test comprehensive mode rate limits"""
        limits = RateLimitConfig.get_limits_for_mode("COMPREHENSIVE")
        assert limits['max_articles'] == 6  # Changed from 5 to 6
        assert limits['max_secondary_queries'] == 6
        assert limits['max_wikipedia_searches'] == 12
        assert limits['enable_openai'] is True
        assert limits['enable_agents'] is True
    
    def test_get_limits_invalid_mode(self):
        """Test invalid mode defaults to balanced"""
        limits = RateLimitConfig.get_limits_for_mode("INVALID")
        assert limits['max_articles'] == 3
        assert limits['max_secondary_queries'] == 3  # Changed from 4 to 3
        assert limits['max_wikipedia_searches'] == 6
        assert limits['enable_openai'] is True
        assert limits['enable_agents'] is True


class TestMultiSourceAgent:
    """Test cases for MultiSourceAgent"""
    
    @patch('utils.multi_source_agent.get_gpu_classifier')
    @patch('utils.multi_source_agent.OpenAIQueryGenerator')
    @patch('utils.multi_source_agent.WikipediaAgentSystem')
    def test_init_balanced_mode(self, mock_agent_system, mock_query_gen, mock_get_classifier):
        """Test agent initialization in balanced mode"""
        agent = MultiSourceAgent(cost_mode="BALANCED")
        
        assert agent.cost_mode == "BALANCED"
        assert agent.limits['max_articles'] == 3
        assert agent.limits['max_secondary_queries'] == 3  # Changed from 4 to 3
        assert agent.limits['max_wikipedia_searches'] == 6
        assert agent.limits['enable_openai'] is True
        assert agent.limits['enable_agents'] is True
    
    @patch('utils.multi_source_agent.get_gpu_classifier')
    @patch('utils.multi_source_agent.OpenAIQueryGenerator')
    @patch('utils.multi_source_agent.WikipediaAgentSystem')
    def test_init_minimal_mode(self, mock_agent_system, mock_query_gen, mock_get_classifier):
        """Test agent initialization in minimal mode"""
        agent = MultiSourceAgent(cost_mode="MINIMAL")
        
        assert agent.cost_mode == "MINIMAL"
        assert agent.limits['max_articles'] == 1
        assert agent.limits['max_secondary_queries'] == 0  # Changed from 1 to 0
        assert agent.limits['max_wikipedia_searches'] == 1
        assert agent.limits['enable_openai'] is False
        assert agent.limits['enable_agents'] is False
    
    @patch('utils.multi_source_agent.get_gpu_classifier')
    @patch('utils.multi_source_agent.OpenAIQueryGenerator')
    @patch('utils.multi_source_agent.WikipediaAgentSystem')
    def test_plan_search_strategy_with_openai(self, mock_agent_system, mock_query_gen_class, mock_get_classifier):
        """Test search strategy planning with OpenAI"""
        # Setup mocks
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_get_classifier.return_value = mock_classifier
        
        mock_query_gen = Mock()
        mock_query_gen_class.return_value = mock_query_gen
        
        # Mock OpenAI response
        mock_search_plan = {
            'secondary_queries': {
                'contextual': ['AI history', 'AI principles'],
                'related': ['machine learning']
            },
            'synthesis_focus': 'comprehensive',
            'search_strategy': 'openai_generated'
        }
        mock_query_gen.generate_comprehensive_search_plan.return_value = mock_search_plan
        
        agent = MultiSourceAgent(cost_mode="BALANCED")
        
        strategy = agent.plan_search_strategy("artificial intelligence", "Technology", 0.9)
        
        assert strategy['openai_used'] is True
        assert len(strategy['secondary_queries']) == 3  # Changed from 4 to 3
        assert strategy['synthesis_focus'] == 'comprehensive'
        assert strategy['cost_mode'] == 'BALANCED'
    
    @patch('utils.multi_source_agent.get_gpu_classifier')
    @patch('utils.multi_source_agent.OpenAIQueryGenerator')
    @patch('utils.multi_source_agent.WikipediaAgentSystem')
    def test_plan_search_strategy_rate_limited(self, mock_agent_system, mock_query_gen_class, mock_get_classifier):
        """Test search strategy planning when rate limited"""
        agent = MultiSourceAgent(cost_mode="MINIMAL")  # This mode disables OpenAI
        strategy = agent.plan_search_strategy("test query", "General", 0.5)
        
        assert strategy['openai_used'] is False
        assert len(strategy['secondary_queries']) == 0  # Changed from > 0 to == 0 for MINIMAL mode
        assert strategy['cost_mode'] == 'MINIMAL'
    
    @patch('utils.multi_source_agent.get_gpu_classifier')
    @patch('utils.multi_source_agent.OpenAIQueryGenerator')
    @patch('utils.multi_source_agent.WikipediaAgentSystem')
    def test_generate_fallback_search_plan_biography(self, mock_agent_system, mock_query_gen, mock_get_classifier):
        """Test fallback search plan generation for biography intent"""
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_get_classifier.return_value = mock_classifier
        
        agent = MultiSourceAgent()
        
        plan = agent._generate_fallback_search_plan("Albert Einstein", "Biography")
        
        assert 'secondary_queries' in plan
        assert 'contextual' in plan['secondary_queries']
        assert 'related' in plan['secondary_queries']
        assert 'biography' in plan['secondary_queries']['contextual'][0].lower()
        assert plan['search_strategy'] == 'fallback_rule_based'
    
    @patch('utils.multi_source_agent.get_gpu_classifier')
    @patch('utils.multi_source_agent.OpenAIQueryGenerator')
    @patch('utils.multi_source_agent.WikipediaAgentSystem')
    def test_generate_fallback_search_plan_science(self, mock_agent_system, mock_query_gen, mock_get_classifier):
        """Test fallback search plan generation for science intent"""
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_get_classifier.return_value = mock_classifier
        
        agent = MultiSourceAgent()
        
        plan = agent._generate_fallback_search_plan("quantum physics", "Science")
        
        assert 'secondary_queries' in plan
        assert 'contextual' in plan['secondary_queries']
        assert 'related' in plan['secondary_queries']
        assert any('explanation' in query.lower() or 'principles' in query.lower() 
                  for query in plan['secondary_queries']['contextual'])
    
    @patch('utils.multi_source_agent.get_gpu_classifier')
    @patch('utils.multi_source_agent.OpenAIQueryGenerator')
    @patch('utils.multi_source_agent.WikipediaAgentSystem')
    def test_gather_articles_with_agents_success(self, mock_agent_system_class, mock_query_gen, mock_get_classifier):
        """Test article gathering with LangChain agents"""
        # Setup mocks
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_get_classifier.return_value = mock_classifier
        
        mock_agent_system = Mock()
        mock_agent_system_class.return_value = mock_agent_system
        
        # Mock agent response with proper dictionary that supports 'in' operator
        # The actual method called is intelligent_wikipedia_search, not search_and_select_articles
        mock_agent_result = {
            'article_info': {
                'title': 'Test Article', 
                'content': 'Test content', 
                'url': 'test.com'
            },
            'enhancement_result': {
                'enhanced_query': 'enhanced test query',
                'agent_reasoning': 'Query enhanced for better search'
            },
            'selection_result': {
                'agent_reasoning': 'Selected based on relevance',
                'confidence_score': 0.9
            },
            'search_results': ['Test Article', 'Alternative Article']
        }
        
        # Return the actual dictionary, not a Mock - use the correct method name
        mock_agent_system.intelligent_wikipedia_search.return_value = mock_agent_result
        
        agent = MultiSourceAgent(cost_mode="BALANCED")
        
        strategy = {
            'primary_queries': ['test query'],
            'secondary_queries': ['related query'],
            'max_articles': 3,
            'synthesis_focus': 'comprehensive'
        }
        
        articles = agent.gather_articles_with_agents(strategy)
        
        assert len(articles) > 0
        assert articles[0]['title'] == 'Test Article'
        assert articles[0]['selection_method'] == 'langchain_agent'
    
    @patch('utils.multi_source_agent.get_gpu_classifier')
    @patch('utils.multi_source_agent.OpenAIQueryGenerator')
    @patch('utils.multi_source_agent.WikipediaAgentSystem')
    def test_gather_articles_rate_limited(self, mock_agent_system_class, mock_query_gen, mock_get_classifier):
        """Test article gathering when rate limited"""
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_get_classifier.return_value = mock_classifier
        
        mock_agent_system = Mock()
        mock_agent_system_class.return_value = mock_agent_system
        
        agent = MultiSourceAgent(cost_mode="MINIMAL")
        
        strategy = {
            'primary_queries': ['query1', 'query2', 'query3'],
            'secondary_queries': ['query4', 'query5'],
            'max_articles': 1  # MINIMAL mode limit
        }
        
        # Mock successful agent response
        mock_agent_result = {
            'article_info': {
                'title': 'Test Article',
                'content': 'Test content'
            },
            'enhancement_result': {
                'enhanced_query': 'enhanced query',
                'agent_reasoning': 'test reasoning'
            },
            'search_results': ['result1']
        }
        mock_agent_system.intelligent_wikipedia_search.return_value = mock_agent_result
        
        articles = agent.gather_articles_with_agents(strategy)
        
        # Should stop at max_articles limit
        assert len(articles) <= 1
    
    @patch('utils.multi_source_agent.get_gpu_classifier')
    @patch('utils.multi_source_agent.OpenAIQueryGenerator')
    @patch('utils.multi_source_agent.WikipediaAgentSystem')
    def test_analyze_intent_with_bert(self, mock_agent_system, mock_query_gen, mock_get_classifier):
        """Test intent analysis with BERT classifier"""
        # Mock BERT classifier
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_classifier.predict.return_value = ('Technology', 0.9)
        mock_get_classifier.return_value = mock_classifier
        
        agent = MultiSourceAgent()
        result = agent._analyze_intent("artificial intelligence")
        
        assert result['intent'] == 'Technology'
        assert result['confidence'] == 0.9
        assert result['method'] == 'gpu_bert'  # Changed from 'GPU BERT' to 'gpu_bert'
    
    @patch('utils.multi_source_agent.get_gpu_classifier')
    @patch('utils.multi_source_agent.OpenAIQueryGenerator')
    @patch('utils.multi_source_agent.WikipediaAgentSystem')
    def test_analyze_intent_fallback(self, mock_agent_system, mock_query_gen, mock_get_classifier):
        """Test intent analysis fallback when BERT fails"""
        # Mock BERT classifier to fail
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = False
        mock_get_classifier.return_value = mock_classifier
        
        agent = MultiSourceAgent()
        result = agent._analyze_intent("test query")
        
        assert result['intent'] == 'General'
        assert result['confidence'] == 0.5
        assert result['method'] == 'fallback'  # Changed from 'keyword_fallback' to 'fallback'
    
    @patch('utils.multi_source_agent.get_gpu_classifier')
    @patch('utils.multi_source_agent.OpenAIQueryGenerator')
    @patch('utils.multi_source_agent.WikipediaAgentSystem')
    def test_rank_articles(self, mock_agent_system, mock_query_gen, mock_get_classifier):
        """Test article ranking functionality"""
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_get_classifier.return_value = mock_classifier
        
        agent = MultiSourceAgent()
        
        articles = [
            {'title': 'Artificial Intelligence', 'content': 'AI content', 'relevance_score': 0.9},
            {'title': 'Machine Learning', 'content': 'ML content', 'relevance_score': 0.7},
            {'title': 'Neural Networks', 'content': 'NN content', 'relevance_score': 0.8}
        ]
        
        ranked = agent.rank_articles(articles, "artificial intelligence", "Technology")
        
        # Should be sorted by relevance score (descending)
        assert ranked[0]['title'] == 'Artificial Intelligence'
        assert ranked[1]['title'] == 'Neural Networks'
        assert ranked[2]['title'] == 'Machine Learning'


class TestMultiSourceAgentIntegration:
    """Integration tests for MultiSourceAgent"""
    
    @patch('utils.multi_source_agent.get_gpu_classifier')
    @patch('utils.multi_source_agent.OpenAIQueryGenerator')
    @patch('utils.multi_source_agent.WikipediaAgentSystem')
    def test_cost_mode_consistency(self, mock_agent_system, mock_query_gen, mock_get_classifier):
        """Test that cost modes are applied consistently"""
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_get_classifier.return_value = mock_classifier
        
        modes = ["MINIMAL", "BALANCED", "COMPREHENSIVE"]
        
        for mode in modes:
            agent = MultiSourceAgent(cost_mode=mode)
            limits = RateLimitConfig.get_limits_for_mode(mode)
            
            assert agent.cost_mode == mode
            assert agent.limits == limits
    
    @patch('utils.multi_source_agent.get_gpu_classifier')
    @patch('utils.multi_source_agent.OpenAIQueryGenerator')
    @patch('utils.multi_source_agent.WikipediaAgentSystem')
    def test_tracking_variables_initialization(self, mock_agent_system, mock_query_gen, mock_get_classifier):
        """Test that tracking variables are properly initialized"""
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_get_classifier.return_value = mock_classifier
        
        agent = MultiSourceAgent()
        
        assert agent.openai_calls_made == 0
        assert agent.wikipedia_calls_made == 0
        assert agent.articles_processed == 0
    
    @patch('utils.multi_source_agent.get_gpu_classifier')
    @patch('utils.multi_source_agent.OpenAIQueryGenerator')
    @patch('utils.multi_source_agent.WikipediaAgentSystem')
    def test_error_handling_in_planning(self, mock_agent_system, mock_query_gen_class, mock_get_classifier):
        """Test error handling during search strategy planning"""
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = True
        mock_get_classifier.return_value = mock_classifier
        
        mock_query_gen = Mock()
        mock_query_gen_class.return_value = mock_query_gen
        mock_query_gen.generate_comprehensive_search_plan.side_effect = Exception("OpenAI API error")
        
        agent = MultiSourceAgent()
        
        # Should not raise exception, should fall back to rule-based planning
        strategy = agent.plan_search_strategy("test query", "Technology", 0.8)
        
        assert strategy['openai_used'] is False
        assert 'synthesis_focus' in strategy
        assert len(strategy['secondary_queries']) > 0 