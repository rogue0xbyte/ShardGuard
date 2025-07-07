"""Tests for ShardGuard planning functionality."""

import json
from unittest.mock import Mock, patch

import httpx
import pytest

from shardguard.core.planning import PlanningLLM


class MockPlanningLLM:
    """Mock implementation of PlanningLLMProtocol for testing."""

    def __init__(self, response: str | None = None):
        self.response = response or '{"original_prompt": "test", "sub_prompts": []}'

    def generate_plan(self, prompt: str) -> str:
        return self.response


class TestPlanningLLMProtocol:
    """Test cases for PlanningLLMProtocol."""

    def test_protocol_implementation(self):
        """Test that MockPlanningLLM implements the protocol."""
        mock_llm = MockPlanningLLM()

        # Should be able to call generate_plan
        result = mock_llm.generate_plan("test prompt")
        assert isinstance(result, str)

    def test_protocol_with_custom_response(self):
        """Test protocol implementation with custom response."""
        custom_response = '{"original_prompt": "custom", "sub_prompts": [{"id": 1, "content": "task"}]}'
        mock_llm = MockPlanningLLM(custom_response)

        result = mock_llm.generate_plan("test")
        assert result == custom_response


class TestPlanningLLM:
    """Test cases for PlanningLLM class."""

    @pytest.mark.parametrize(
        "model, base_url, expected_model, expected_url",
        [
            (
                "llama3.2",
                "http://localhost:11434",
                "llama3.2",
                "http://localhost:11434",
            ),
            (
                "custom-model",
                "http://custom:8080",
                "custom-model",
                "http://custom:8080",
            ),
        ],
    )
    def test_initialization(self, model, base_url, expected_model, expected_url):
        """Test PlanningLLM initialization with various parameters."""
        llm = PlanningLLM(model=model, base_url=base_url)

        assert llm.model == expected_model
        assert llm.base_url == expected_url

    @patch("httpx.Client.post")
    @pytest.mark.parametrize(
        "response_json, expected_result",
        [
            (
                {
                    "response": '{"original_prompt": "test prompt", "sub_prompts": [{"id": 1, "content": "subtask", "opaque_values": {}}]}'
                },
                '{"original_prompt": "test prompt", "sub_prompts": [{"id": 1, "content": "subtask", "opaque_values": {}}]}',
            ),
        ],
    )
    def test_generate_plan_success(self, mock_post, response_json, expected_result):
        """Test successful plan generation."""
        mock_response = Mock()
        mock_response.json.return_value = response_json
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        llm = PlanningLLM()
        result = llm.generate_plan("test prompt")

        assert result == expected_result

    @patch("httpx.Client.post")
    def test_generate_plan_request_error(self, mock_post):
        """Test handling of request errors with fallback response."""
        mock_post.side_effect = httpx.RequestError("Connection failed")

        llm = PlanningLLM()

        result = llm.generate_plan("test prompt")

        # Should return fallback JSON response instead of raising
        assert "Error occurred" in result
        parsed = json.loads(result)
        assert "test prompt" in parsed["original_prompt"]
        assert len(parsed["sub_prompts"]) > 0

    @patch("httpx.Client.post")
    def test_generate_plan_http_error(self, mock_post):
        """Test handling of HTTP status errors with fallback response."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.side_effect = httpx.HTTPStatusError(
            "Server error", request=Mock(), response=mock_response
        )

        llm = PlanningLLM()

        result = llm.generate_plan("test prompt")

        # Should return fallback JSON response instead of raising
        assert "Error occurred" in result
        parsed = json.loads(result)
        assert "test prompt" in parsed["original_prompt"]
        assert len(parsed["sub_prompts"]) > 0

    @patch("httpx.Client.post")
    def test_generate_plan_invalid_json_from_api(self, mock_post):
        """Test handling of invalid JSON from API with fallback response."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        llm = PlanningLLM()

        result = llm.generate_plan("test prompt")

        # Should return fallback JSON response instead of raising
        assert "Error occurred" in result
        parsed = json.loads(result)
        assert "test prompt" in parsed["original_prompt"]
        assert len(parsed["sub_prompts"]) > 0

    def test_extract_json_from_response_valid_json(self):
        """Test JSON extraction from response with valid JSON."""
        llm = PlanningLLM()

        response_text = 'Here is the plan: {"original_prompt": "test", "sub_prompts": []} That\'s it!'
        result = llm._extract_json_from_response(response_text)

        expected = '{"original_prompt": "test", "sub_prompts": []}'
        assert result == expected

    def test_extract_json_from_response_complex_json(self):
        """Test JSON extraction with complex nested JSON."""
        llm = PlanningLLM()

        complex_json = '{"original_prompt": "complex task", "sub_prompts": [{"id": 1, "content": "step 1", "opaque_values": {"key": "value"}}]}'
        response_text = f"The analysis shows: {complex_json} End of response."

        result = llm._extract_json_from_response(response_text)
        assert result == complex_json

    def test_extract_json_from_response_multiple_json_blocks(self):
        """Test JSON extraction when multiple JSON blocks exist."""
        llm = PlanningLLM()

        response_text = """
        First attempt: {"wrong": "json"}

        Corrected version: {"original_prompt": "test", "sub_prompts": [{"id": 1, "content": "task", "opaque_values": {}}]}

        That's the final answer.
        """

        result = llm._extract_json_from_response(response_text)

        # Should return the longest valid JSON
        assert "original_prompt" in result
        assert "sub_prompts" in result

    def test_extract_json_from_response_invalid_json(self):
        """Test JSON extraction when no valid JSON exists."""
        llm = PlanningLLM()

        response_text = "Sorry, I cannot process this request. No JSON here."
        result = llm._extract_json_from_response(response_text)

        # Should return fallback JSON response when no valid JSON found
        parsed = json.loads(result)
        assert "original_prompt" in parsed
        assert "sub_prompts" in parsed
        assert response_text in parsed["original_prompt"]

    def test_extract_json_from_response_malformed_json(self):
        """Test JSON extraction with malformed JSON."""
        llm = PlanningLLM()

        response_text = (
            'Result: {"original_prompt": "test", "sub_prompts": [missing_bracket}'
        )
        result = llm._extract_json_from_response(response_text)

        # Should return fallback JSON response since JSON is malformed
        parsed = json.loads(result)
        assert "original_prompt" in parsed
        assert "sub_prompts" in parsed
        assert "Result:" in parsed["original_prompt"]

    def test_context_manager_usage(self):
        """Test using PlanningLLM as a context manager."""
        with patch("httpx.Client.close") as mock_close:
            with PlanningLLM() as llm:
                assert isinstance(llm, PlanningLLM)

            # Should call close on the client
            mock_close.assert_called_once()

    @patch("httpx.Client.post")
    def test_generate_plan_with_custom_options(self, mock_post):
        """Test that custom options are passed to the API."""
        mock_response = Mock()
        mock_response.json.return_value = {"response": '{"test": "response"}'}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        llm = PlanningLLM()
        llm.generate_plan("test")

        # Check that options are set correctly
        call_args = mock_post.call_args
        request_data = call_args[1]["json"]
        options = request_data["options"]

        assert options["temperature"] == 0.1
        assert options["top_p"] == 0.9
        assert options["num_predict"] == 2048

    @patch("httpx.Client.post")
    def test_generate_plan_with_different_base_url(self, mock_post):
        """Test generate_plan with different base URL."""
        mock_response = Mock()
        mock_response.json.return_value = {"response": '{"test": "response"}'}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        llm = PlanningLLM(base_url="http://remote:8080")
        llm.generate_plan("test")

        # Verify correct URL is used
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://remote:8080/api/generate"


class TestPlanningLLMWithMCP:
    """Test cases for PlanningLLM with MCP server integration."""

    def test_planning_llm_has_mcp_integration(self):
        """Test that PlanningLLM has MCP integration capability."""
        llm = PlanningLLM()

        # Should be able to get tools description
        description = llm.get_available_tools_description()
        assert isinstance(description, str)

    @patch("httpx.Client.post")
    def test_generate_plan_basic_functionality(self, mock_post):
        """Test basic generate_plan functionality."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '{"original_prompt": "test", "sub_prompts": []}'
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        llm = PlanningLLM()
        result = llm.generate_plan("test prompt")

        # Should return a JSON string
        assert isinstance(result, str)

        # Verify the API was called
        mock_post.assert_called_once()


class TestMockPlanningLLM:
    """Tests for the MockPlanningLLM class."""
