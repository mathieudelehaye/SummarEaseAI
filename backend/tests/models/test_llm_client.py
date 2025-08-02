"""
Unit tests for LLM Client Model
Tests the OpenAI client wrapper and abstraction
"""

from unittest.mock import Mock, patch

from backend.models.llm_client import LLMClient, get_llm_client


class TestLLMClient:
    """Test cases for LLMClient"""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    def test_check_openai_availability_with_key(self):
        """Test OpenAI availability check when API key exists"""
        client = LLMClient()
        assert client.check_openai_availability() is True

    @patch.dict("os.environ", {}, clear=True)
    def test_check_openai_availability_no_key(self):
        """Test OpenAI availability check when API key missing"""
        client = LLMClient()
        assert client.check_openai_availability() is False

    @patch("backend.models.llm_client.LLMClient")  # patch where it's used
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_get_llm_client_success(self, mock_llm_class):
        """Test successful LLM client creation"""
        mock_instance = Mock()
        mock_llm_class.return_value = mock_instance  # make constructor return mock

        # Act
        llm_client = get_llm_client()

        # Assert
        assert llm_client == mock_instance
        mock_llm_class.assert_called_once()

    @patch("backend.models.llm_client.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.llm_client.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_call_openai_chat_success(self, mock_chat_openai):
        """Test successful OpenAI chat call"""
        # Setup mock response
        mock_response = Mock()
        mock_response.content = "This is the AI response"

        mock_openai_instance = Mock()
        mock_openai_instance.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_openai_instance

        client = LLMClient()
        response = client.call_openai_chat("Test prompt")

        assert response == "This is the AI response"
        mock_openai_instance.invoke.assert_called_once_with("Test prompt")

    @patch("backend.models.llm_client.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.llm_client.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_call_openai_chat_with_custom_params(self, mock_chat_openai):
        """Test OpenAI chat call with custom parameters"""
        mock_response = Mock()
        mock_response.content = "Custom response"

        mock_openai_instance = Mock()
        mock_openai_instance.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_openai_instance

        client = LLMClient()
        response = client.call_openai_chat(
            prompt="Custom prompt", temperature=0.8, max_tokens=500
        )

        assert response == "Custom response"

    @patch.dict("os.environ", {}, clear=True)
    def test_call_openai_chat_no_api_key(self):
        """Test OpenAI chat call when API key missing"""
        client = LLMClient()
        response = client.call_openai_chat("Test prompt")

        assert "Error: OpenAI API key not configured" in response

    @patch("backend.models.llm_client.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.llm_client.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_call_openai_chat_exception(self, mock_chat_openai):
        """Test OpenAI chat call when API raises exception"""
        mock_openai_instance = Mock()
        mock_openai_instance.invoke.side_effect = Exception("API call failed")
        mock_chat_openai.return_value = mock_openai_instance

        client = LLMClient()
        response = client.call_openai_chat("Test prompt")

        assert "Error calling OpenAI" in response
        assert "API call failed" in response

    def test_get_model_info(self):
        """Test model information retrieval"""
        client = LLMClient(model="gpt-4", temperature=0.5, max_tokens=1500)

        info = client.get_model_info()

        assert info["model"] == "gpt-4"
        assert info["temperature"] == 0.5
        assert info["max_tokens"] == 1500
        assert "openai_available" in info

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_get_model_info_with_api_key(self):
        """Test model information when API key available"""
        client = LLMClient()
        info = client.get_model_info()

        assert info["openai_available"] is True

    @patch.dict("os.environ", {}, clear=True)
    def test_get_model_info_without_api_key(self):
        """Test model information when API key unavailable"""
        client = LLMClient()
        info = client.get_model_info()

        assert info["openai_available"] is False


class TestLLMClientSingleton:
    """Test singleton pattern for LLMClient"""

    def test_get_llm_client_singleton(self):
        """Test that get_llm_client returns same instance"""
        # Reset singleton instance
        from backend.models.llm_client import _LLMClientSingleton

        _LLMClientSingleton._instance = None

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client1 = get_llm_client()
            client2 = get_llm_client()

            assert client1 is client2

    def test_singleton_instance_type(self):
        """Test that singleton returns correct type"""
        # Reset singleton instance
        from backend.models.llm_client import _LLMClientSingleton

        _LLMClientSingleton._instance = None

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = get_llm_client()
            assert isinstance(client, LLMClient)


class TestLLMClientConfigurationVariations:
    """Test different configuration scenarios"""

    @patch("backend.models.llm_client.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_client_reuses_instance(self, mock_chat_openai):
        """Test that client reuses the same instance"""
        mock_response = Mock()
        mock_response.content = "Response"

        mock_openai_instance = Mock()
        mock_openai_instance.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_openai_instance

        client1 = LLMClient()
        client2 = LLMClient()

        # Both should work with the same configuration
        response1 = client1.call_openai_chat("Test")
        response2 = client2.call_openai_chat("Test")

        assert response1 == response2


class TestLLMClientErrorHandling:
    """Test error handling scenarios"""

    @patch("backend.models.llm_client.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.llm_client.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_call_openai_chat_timeout_error(self, mock_chat_openai):
        """Test handling of timeout errors"""
        mock_openai_instance = Mock()
        mock_openai_instance.invoke.side_effect = Exception("Request timeout")
        mock_chat_openai.return_value = mock_openai_instance

        client = LLMClient()
        response = client.call_openai_chat("Test prompt")

        assert "Error calling OpenAI" in response
        assert "Request timeout" in response

    @patch("backend.models.llm_client.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.llm_client.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_call_openai_chat_connection_error(self, mock_chat_openai):
        """Test handling of connection errors"""
        mock_openai_instance = Mock()
        mock_openai_instance.invoke.side_effect = Exception("Connection failed")
        mock_chat_openai.return_value = mock_openai_instance

        client = LLMClient()
        response = client.call_openai_chat("Test prompt")

        assert "Error calling OpenAI" in response
        assert "Connection failed" in response

    @patch("backend.models.llm_client.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.llm_client.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_call_openai_chat_value_error(self, mock_chat_openai):
        """Test handling of value errors"""
        mock_openai_instance = Mock()
        mock_openai_instance.invoke.side_effect = ValueError("Invalid parameter")
        mock_chat_openai.return_value = mock_openai_instance

        client = LLMClient()
        response = client.call_openai_chat("Test prompt")

        assert "Error calling OpenAI" in response
        assert "Invalid parameter" in response

    def test_call_openai_chat_empty_prompt(self):
        """Test handling of empty prompt"""
        client = LLMClient()

        # Test with empty string
        response = client.call_openai_chat("")
        assert "Error: Empty prompt provided" in response

        # Test with whitespace only
        response = client.call_openai_chat("   ")
        assert "Error: Empty prompt provided" in response


class TestLLMClientIntegration:
    """Test integration scenarios"""

    @patch("backend.models.llm_client.LANGCHAIN_AVAILABLE", True)
    @patch("backend.models.llm_client.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_full_workflow(self, mock_chat_openai):
        """Test complete workflow with API key"""
        mock_response = Mock()
        mock_response.content = "Workflow response"

        mock_openai_instance = Mock()
        mock_openai_instance.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_openai_instance

        client = LLMClient()

        # Check availability
        assert client.check_openai_availability() is True

        # Get model info
        info = client.get_model_info()
        assert info["openai_available"] is True

        # Make API call
        response = client.call_openai_chat("Test workflow")
        assert response == "Workflow response"

    @patch.dict("os.environ", {}, clear=True)
    def test_workflow_without_api_key(self):
        """Test workflow when API key is not available"""
        client = LLMClient()

        # Check availability
        assert client.check_openai_availability() is False

        # Get model info
        info = client.get_model_info()
        assert info["openai_available"] is False

        # Try API call
        response = client.call_openai_chat("Test")
        assert "Error: OpenAI API key not configured" in response
