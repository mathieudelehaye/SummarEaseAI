"""
Unit tests for LLM Client Model
Tests the OpenAI client wrapper and abstraction
"""

from unittest.mock import Mock, patch

from backend.models.llm_client import LLMClient, get_llm_client


class TestLLMClient:
    """Test cases for LLMClient"""

    def test_init_default(self):
        """Test LLM client initialization with defaults"""
        client = LLMClient()
        assert client.model_name == "gpt-3.5-turbo"
        assert client.temperature == 0.3
        assert client.max_tokens == 1000

    def test_init_custom_parameters(self):
        """Test LLM client initialization with custom parameters"""
        client = LLMClient(model_name="gpt-4", temperature=0.7, max_tokens=2000)
        assert client.model_name == "gpt-4"
        assert client.temperature == 0.7
        assert client.max_tokens == 2000

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

    @patch("backend.models.llm_client.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_get_llm_client_success(self, mock_chat_openai):
        """Test successful LLM client creation"""
        mock_openai_instance = Mock()
        mock_chat_openai.return_value = mock_openai_instance

        client = LLMClient()
        llm_client = client.get_llm_client()

        assert llm_client == mock_openai_instance
        mock_chat_openai.assert_called_once_with(
            openai_api_key="test-key",
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=1000,
        )

    @patch.dict("os.environ", {}, clear=True)
    def test_get_llm_client_no_api_key(self):
        """Test LLM client creation when API key missing"""
        client = LLMClient()
        llm_client = client.get_llm_client()

        assert llm_client is None

    @patch("backend.models.llm_client.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_get_llm_client_exception(self, mock_chat_openai):
        """Test LLM client creation when ChatOpenAI raises exception"""
        mock_chat_openai.side_effect = Exception("OpenAI initialization error")

        client = LLMClient()
        llm_client = client.get_llm_client()

        assert llm_client is None

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
        client = LLMClient(model_name="gpt-4", temperature=0.5, max_tokens=1500)

        info = client.get_model_info()

        assert info["model_name"] == "gpt-4"
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
        client1 = get_llm_client()
        client2 = get_llm_client()
        assert client1 is client2

    def test_singleton_instance_type(self):
        """Test that singleton returns correct type"""
        client = get_llm_client()
        assert isinstance(client, LLMClient)


class TestLLMClientConfigurationVariations:
    """Test different configuration scenarios"""

    def test_different_model_names(self):
        """Test client with different model names"""
        models = ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"]

        for model in models:
            client = LLMClient(model_name=model)
            assert client.model_name == model

    def test_temperature_bounds(self):
        """Test client with different temperature values"""
        temperatures = [0.0, 0.5, 1.0]

        for temp in temperatures:
            client = LLMClient(temperature=temp)
            assert client.temperature == temp

    def test_max_tokens_variations(self):
        """Test client with different max_tokens values"""
        token_limits = [100, 1000, 4000]

        for limit in token_limits:
            client = LLMClient(max_tokens=limit)
            assert client.max_tokens == limit

    @patch("backend.models.llm_client.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_client_reuses_instance(self, mock_chat_openai):
        """Test that LLM client reuses OpenAI instance"""
        mock_openai_instance = Mock()
        mock_chat_openai.return_value = mock_openai_instance

        client = LLMClient()

        # First call creates instance
        llm1 = client.get_llm_client()
        # Second call should return same instance
        llm2 = client.get_llm_client()

        assert llm1 is llm2
        # ChatOpenAI should only be called once
        assert mock_chat_openai.call_count == 1


class TestLLMClientErrorHandling:
    """Test error handling scenarios"""

    @patch("backend.models.llm_client.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_call_openai_chat_timeout_error(self, mock_chat_openai):
        """Test handling of timeout errors"""
        mock_openai_instance = Mock()
        mock_openai_instance.invoke.side_effect = TimeoutError("Request timeout")
        mock_chat_openai.return_value = mock_openai_instance

        client = LLMClient()
        response = client.call_openai_chat("Test prompt")

        assert "Error calling OpenAI" in response
        assert "Request timeout" in response

    @patch("backend.models.llm_client.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_call_openai_chat_connection_error(self, mock_chat_openai):
        """Test handling of connection errors"""
        mock_openai_instance = Mock()
        mock_openai_instance.invoke.side_effect = ConnectionError("Network error")
        mock_chat_openai.return_value = mock_openai_instance

        client = LLMClient()
        response = client.call_openai_chat("Test prompt")

        assert "Error calling OpenAI" in response
        assert "Network error" in response

    @patch("backend.models.llm_client.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_call_openai_chat_value_error(self, mock_chat_openai):
        """Test handling of value errors"""
        mock_openai_instance = Mock()
        mock_openai_instance.invoke.side_effect = ValueError("Invalid input")
        mock_chat_openai.return_value = mock_openai_instance

        client = LLMClient()
        response = client.call_openai_chat("Test prompt")

        assert "Error calling OpenAI" in response
        assert "Invalid input" in response

    def test_call_openai_chat_empty_prompt(self):
        """Test handling of empty prompt"""
        client = LLMClient()

        # Test with empty string
        response = client.call_openai_chat("")
        assert "Error calling OpenAI" in response or response == ""

        # Test with None
        response = client.call_openai_chat(None)
        assert "Error calling OpenAI" in response or response is None


class TestLLMClientIntegration:
    """Integration tests for LLMClient"""

    @patch("backend.models.llm_client.ChatOpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_full_workflow(self, mock_chat_openai):
        """Test complete workflow from initialization to API call"""
        # Setup mock
        mock_response = Mock()
        mock_response.content = "AI generated response"

        mock_openai_instance = Mock()
        mock_openai_instance.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_openai_instance

        # Test workflow
        client = LLMClient(model_name="gpt-4", temperature=0.7)

        # Check availability
        assert client.check_openai_availability() is True

        # Get model info
        info = client.get_model_info()
        assert info["openai_available"] is True
        assert info["model_name"] == "gpt-4"

        # Make API call
        response = client.call_openai_chat("Generate a summary")
        assert response == "AI generated response"

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
        response = client.call_openai_chat("Test prompt")
        assert "Error: OpenAI API key not configured" in response
