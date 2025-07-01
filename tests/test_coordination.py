"""Tests for ShardGuard coordination service."""

import json
from io import StringIO
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from shardguard.core.coordination import CoordinationService
from shardguard.core.models import Plan
from tests.test_helpers import MockPlanningLLM


class TestCoordinationService:
    """Test cases for CoordinationService."""

    def test_init_with_planner(self):
        """Test CoordinationService initialization."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        assert service.planner is planner
        assert service.console is not None

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
    """Test cases for input sanitization in CoordinationService."""

    def test_sanitize_input_basic_clean_input(self):
        """Test sanitization with clean input."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        with patch("sys.stdout", new_callable=StringIO):
            result = service._sanitize_input("clean input")

        assert result == "clean input"

    def test_sanitize_input_whitespace_normalization(self):
        """Test whitespace normalization."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        with patch("sys.stdout", new_callable=StringIO):
            result = service._sanitize_input("  multiple   spaces  \n\t  ")

        assert result == "multiple spaces"

    def test_sanitize_input_empty_input_raises_error(self):
        """Test that empty input raises ValueError."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        with patch("sys.stdout", new_callable=StringIO):
            with pytest.raises(ValueError, match="User input cannot be empty"):
                service._sanitize_input("")

            with pytest.raises(ValueError, match="User input cannot be empty"):
                service._sanitize_input("   ")

    def test_sanitize_input_control_character_removal(self):
        """Test removal of dangerous control characters."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        # Test with various control characters
        dangerous_input = "test\x00\x01\x08\x0b\x0c\x0e\x1f\x7fdata"

        with patch("sys.stdout", new_callable=StringIO):
            result = service._sanitize_input(dangerous_input)

        # Control characters should be removed but whitespace normalization might add spaces
        assert "test" in result
        assert "data" in result
        assert len(result) < len(dangerous_input)  # Should be shorter due to removal

    def test_sanitize_input_length_truncation(self):
        """Test truncation of extremely long inputs."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        long_input = "x" * 15000  # Longer than 10000 char limit

        with patch("sys.stdout", new_callable=StringIO):
            result = service._sanitize_input(long_input)

        assert len(result) == 10000 + len("... [truncated]")
        assert result.endswith("... [truncated]")

    def test_sanitize_input_script_tag_removal(self):
        """Test removal of script tags."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        malicious_input = 'backup <script>alert("xss")</script> files'

        with patch("sys.stdout", new_callable=StringIO):
            result = service._sanitize_input(malicious_input)

        assert result == "backup [REMOVED] files"
        assert "<script>" not in result

    def test_sanitize_input_javascript_url_removal(self):
        """Test removal of javascript URLs."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        malicious_input = "visit javascript:alert('xss') this site"

        with patch("sys.stdout", new_callable=StringIO):
            result = service._sanitize_input(malicious_input)

        assert result == "visit [REMOVED]alert('xss') this site"

    def test_sanitize_input_data_html_removal(self):
        """Test removal of HTML data URLs."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        malicious_input = "load data:text/html,<h1>test</h1> content"

        with patch("sys.stdout", new_callable=StringIO):
            result = service._sanitize_input(malicious_input)

        assert result == "load [REMOVED],<h1>test</h1> content"

    def test_sanitize_input_preserves_legitimate_content(self):
        """Test that sanitization preserves legitimate content."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        legitimate_inputs = [
            "backup /home/user/files to /backup/location",
            "connect to database://server:5432/mydb",
            "process data from file.txt at 3:30 AM",
            "user@domain.com needs access to folder_123",
            "run command with --flag=value --other-flag",
        ]

        for test_input in legitimate_inputs:
            with patch("sys.stdout", new_callable=StringIO):
                result = service._sanitize_input(test_input)

            # Should be unchanged (maybe whitespace normalized)
            assert test_input.strip() == result or test_input == result

    def test_sanitize_input_logging_output(self):
        """Test that sanitization produces appropriate logging output."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            service._sanitize_input("  test   input  ")
            output = mock_stdout.getvalue()

        # Should contain sanitization process indicators
        assert "Input Sanitization Process" in output
        assert "Original Input" in output

    def test_sanitize_input_case_insensitive_removal(self):
        """Test that pattern removal is case insensitive."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        test_cases = [
            "test <SCRIPT>alert('test')</SCRIPT> data",
            "test <Script>alert('test')</Script> data",
            "test JAVASCRIPT:void(0) data",
            "test JavaScript:void(0) data",
        ]

        for test_input in test_cases:
            with patch("sys.stdout", new_callable=StringIO):
                result = service._sanitize_input(test_input)

            assert "[REMOVED]" in result
            assert "script" not in result.lower() or "javascript" not in result.lower()


class TestCoordinationServicePromptFormatting:
    """Test cases for prompt formatting in CoordinationService."""

    def test_format_prompt_basic(self):
        """Test basic prompt formatting."""
        planner = MockPlanningLLM()
        service = CoordinationService(planner)

        result = service._format_prompt("test input")

        assert "test input" in result
        assert "ShardGuard" in result
        assert "USER_PROMPT:" in result
        assert "END." in result

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

    def test_full_flow_with_real_mock_planner(self):
        """Test complete flow with MockPlanningLLM."""
        service = CoordinationService(MockPlanningLLM())

        with patch("sys.stdout", new_callable=StringIO):
            result = service.handle_prompt("backup at 3 AM with 5 retries")

        assert isinstance(result, Plan)
        extracted_numbers = list(result.sub_prompts[0].opaque_values.values())
        assert "3" in extracted_numbers
        assert "5" in extracted_numbers

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

    def test_multiple_calls_independence(self):
        """Test that multiple calls to handle_prompt are independent."""
        service = CoordinationService(MockPlanningLLM())

        with patch("sys.stdout", new_callable=StringIO):
            result1 = service.handle_prompt("first prompt with 1 number")
            result2 = service.handle_prompt("second prompt with 2 and 3")

        # Results should be independent - just check that different numbers are extracted
        extracted_numbers_1 = list(result1.sub_prompts[0].opaque_values.values())
        extracted_numbers_2 = list(result2.sub_prompts[0].opaque_values.values())
        assert "1" in extracted_numbers_1
        assert "2" in extracted_numbers_2
        assert "3" in extracted_numbers_2
