"""Tests for ShardGuard coordination service."""

import json
from io import StringIO
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from shardguard.core.coordination import CoordinationService
from shardguard.core.models import Plan
from shardguard.core.sanitization import InputSanitizer
from tests.test_helpers import MockPlanningLLM


class TestCoordinationService:
    """Test cases for CoordinationService."""

    def test_init_with_planner(self):
        """Test CoordinationService initialization."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        assert service.planner is planner
        assert service.console is not None
        assert service.sanitizer is not None
        assert isinstance(service.sanitizer, InputSanitizer)

    def test_handle_prompt_basic_flow(self):
        """Test the complete flow of handling a prompt."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        with patch("sys.stdout", new_callable=StringIO):  # Suppress console output
            result = service.handle_prompt("backup files at 3 AM")

        assert isinstance(result, Plan)
        assert result.original_prompt is not None
        assert len(result.sub_prompts) == 1
        assert result.sub_prompts[0].id == 1

    def test_handle_prompt_with_mock_planner(self):
        """Test handle_prompt with mock planner to verify integration."""
        mock_planner = Mock()
        mock_plan_json = json.dumps(
            {
                "original_prompt": "test prompt",
                "sub_prompts": [
                    {
                        "id": 1,
                        "content": "task 1",
                        "opaque_values": {"[VALUE_1]": "val1"},
                    }
                ],
            }
        )
        mock_planner.generate_plan.return_value = mock_plan_json

        service = CoordinationService(mock_planner)

        with patch("sys.stdout", new_callable=StringIO):
            result = service.handle_prompt("test input")

        # Verify planner was called with formatted prompt
        mock_planner.generate_plan.assert_called_once()
        call_args = mock_planner.generate_plan.call_args[0][0]
        assert "test input" in call_args
        assert "ShardGuard" in call_args

        # Verify result
        assert isinstance(result, Plan)
        assert result.original_prompt == "test prompt"


class TestCoordinationServiceSanitization:
    """Test cases for sanitization integration in CoordinationService."""

    def test_handle_prompt_calls_sanitizer(self):
        """Test that handle_prompt uses the sanitizer."""
        mock_planner = Mock()
        mock_plan_json = json.dumps(
            {
                "original_prompt": "test prompt",
                "sub_prompts": [
                    {
                        "id": 1,
                        "content": "task 1",
                        "opaque_values": {"[VALUE_1]": "val1"},
                    }
                ],
            }
        )
        mock_planner.generate_plan.return_value = mock_plan_json

        service = CoordinationService(mock_planner)

        with patch.object(service.sanitizer, "sanitize") as mock_sanitize:
            mock_sanitize.return_value.sanitized_input = "sanitized input"

            with patch("sys.stdout", new_callable=StringIO):
                service.handle_prompt("test input")

            # Verify sanitizer was called
            mock_sanitize.assert_called_once_with("test input")

    def test_handle_prompt_uses_sanitized_input(self):
        """Test that the sanitized input is used for prompt formatting."""
        mock_planner = Mock()
        mock_plan_json = json.dumps(
            {
                "original_prompt": "test prompt",
                "sub_prompts": [{"id": 1, "content": "task 1", "opaque_values": {}}],
            }
        )
        mock_planner.generate_plan.return_value = mock_plan_json

        service = CoordinationService(mock_planner)

        with patch.object(service.sanitizer, "sanitize") as mock_sanitize:
            mock_sanitize.return_value.sanitized_input = "cleaned input"

            with patch("sys.stdout", new_callable=StringIO):
                service.handle_prompt("dirty input")

            # Verify the sanitized input was used in the prompt
            call_args = mock_planner.generate_plan.call_args[0][0]
            assert "cleaned input" in call_args
            assert "dirty input" not in call_args

    def test_handle_prompt_sanitization_error_propagates(self):
        """Test that sanitization errors are propagated."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        with patch.object(service.sanitizer, "sanitize") as mock_sanitize:
            mock_sanitize.side_effect = ValueError("Empty input")

            with pytest.raises(ValueError, match="Empty input"):
                service.handle_prompt("")


class TestCoordinationServicePromptFormatting:
    """Test cases for prompt formatting in CoordinationService."""

    def test_format_prompt_basic(self):
        """Test basic prompt formatting."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        result = service._format_prompt("test input")

        assert "test input" in result
        assert "ShardGuard" in result
        assert "Input" in result

    def test_format_prompt_uses_template(self):
        """Test that _format_prompt uses the PLANNING_PROMPT template."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        with patch("shardguard.core.coordination.PLANNING_PROMPT") as mock_prompt:
            mock_prompt.format.return_value = "formatted result"

            result = service._format_prompt("test")

            mock_prompt.format.assert_called_once_with(user_prompt="test")
            assert result == "formatted result"

    def test_format_prompt_with_special_characters(self):
        """Test prompt formatting with special characters."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        special_input = 'input with "quotes" and \n newlines'
        result = service._format_prompt(special_input)

        assert special_input in result


class TestCoordinationServiceIntegration:
    """Integration tests for CoordinationService."""

    def test_full_flow_with_mock_planner_basic_structure(self):
        """Test complete flow with MockPlanningLLM returns valid structure."""
        service = CoordinationService(MockPlanningLLM())

        with patch("sys.stdout", new_callable=StringIO):
            result = service.handle_prompt("backup at 3 AM with 5 retries")

        assert isinstance(result, Plan)
        assert result.original_prompt is not None
        assert len(result.sub_prompts) >= 1
        assert isinstance(result.sub_prompts[0].opaque_values, dict)

    def test_error_handling_invalid_json_from_planner(self):
        """Test error handling when planner returns invalid JSON."""
        mock_planner = Mock()
        mock_planner.generate_plan.return_value = "invalid json"

        service = CoordinationService(mock_planner)

        with patch("sys.stdout", new_callable=StringIO):
            with pytest.raises((json.JSONDecodeError, ValidationError)):
                service.handle_prompt("test")

    def test_error_handling_invalid_plan_structure(self):
        """Test error handling when planner returns invalid plan structure."""
        mock_planner = Mock()
        mock_planner.generate_plan.return_value = json.dumps({"invalid": "structure"})

        service = CoordinationService(mock_planner)

        with patch("sys.stdout", new_callable=StringIO):
            with pytest.raises(Exception):  # Pydantic validation error
                service.handle_prompt("test")

    def test_multiple_calls_produce_valid_plans(self):
        """Test that multiple calls to handle_prompt produce valid plans."""
        service = CoordinationService(MockPlanningLLM())

        test_prompts = [
            "first prompt with 1 number",
            "second prompt with 2 and 3",
            "third prompt",
        ]

        with patch("sys.stdout", new_callable=StringIO):
            for prompt in test_prompts:
                result = service.handle_prompt(prompt)
                assert isinstance(result, Plan)
                assert result.original_prompt is not None
                assert len(result.sub_prompts) >= 1
