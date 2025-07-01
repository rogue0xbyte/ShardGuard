"""Tests for ShardGuard CLI interface."""

import json
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from shardguard.cli import app
from shardguard.core.models import Plan, SubPrompt


def extract_json_from_cli_output(output: str) -> dict:
    """Extract JSON from CLI output that may contain logging and other text."""
    lines = output.strip().split("\n")

    # Find the start of JSON (first line starting with '{')
    json_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("{"):
            json_start = i
            break

    if json_start == -1:
        raise ValueError("No JSON found in output")

    # Take everything from the first '{' to the end
    json_lines = lines[json_start:]
    json_text = "\n".join(json_lines)

    return json.loads(json_text)


class TestCLIBasic:
    """Basic CLI functionality tests."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_help_command(self):
        """Test CLI help command."""
        result = self.runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "ShardGuard CLI" in result.stdout
        assert "plan" in result.stdout

    def test_cli_no_command_shows_welcome(self):
        """Test CLI without command shows welcome message."""
        result = self.runner.invoke(app, [])

        assert result.exit_code == 0
        assert "Welcome to ShardGuard" in result.stdout

    def test_plan_command_help(self):
        """Test plan command help."""
        result = self.runner.invoke(app, ["plan", "--help"])

        assert result.exit_code == 0
        assert (
            "Generate a safe execution plan" in result.stdout
            or "prompt" in result.stdout
        )


class TestPlanCommand:
    """Test cases for the plan command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("shardguard.cli.CoordinationService")
    def test_plan_command_basic_execution(self, mock_coordination_class):
        """Test basic execution of plan command."""
        # Mock the coordination service
        mock_coordination = Mock()
        mock_coordination_class.return_value = mock_coordination

        # Create a mock plan result
        mock_plan = Plan(
            original_prompt="test prompt",
            sub_prompts=[
                SubPrompt(id=1, content="task 1", opaque_values={"[VALUE_1]": "val1"})
            ],
        )
        mock_coordination.handle_prompt.return_value = mock_plan

        # Run the command
        result = self.runner.invoke(app, ["plan", "backup my files"])

        assert result.exit_code == 0
        mock_coordination.handle_prompt.assert_called_once_with("backup my files")

    @patch("shardguard.cli.CoordinationService")
    def test_plan_command_json_output(self, mock_coordination_class):
        """Test that plan command outputs valid JSON."""
        # Mock coordination service
        mock_coordination = Mock()
        mock_coordination_class.return_value = mock_coordination

        mock_plan = Plan(
            original_prompt="test",
            sub_prompts=[SubPrompt(id=1, content="task", opaque_values={})],
        )
        mock_coordination.handle_prompt.return_value = mock_plan

        result = self.runner.invoke(app, ["plan", "test prompt"])

        assert result.exit_code == 0

        # Should output valid JSON
        try:
            output_data = json.loads(result.stdout.strip())
            assert "original_prompt" in output_data
            assert "sub_prompts" in output_data
        except json.JSONDecodeError:
            pytest.fail("CLI output is not valid JSON")

    def test_plan_command_missing_prompt(self):
        """Test plan command without prompt argument."""
        result = self.runner.invoke(app, ["plan"])

        assert result.exit_code != 0
        # Error output might be in stderr or stdout
        error_output = result.stdout + (result.stderr or "")
        assert "Missing argument" in error_output or result.exit_code == 2

    @patch("shardguard.cli.CoordinationService")
    def test_plan_command_with_complex_prompt(self, mock_coordination_class):
        """Test plan command with complex prompt containing special characters."""
        mock_coordination = Mock()
        mock_coordination_class.return_value = mock_coordination

        mock_plan = Plan(
            original_prompt="complex prompt",
            sub_prompts=[SubPrompt(id=1, content="task", opaque_values={})],
        )
        mock_coordination.handle_prompt.return_value = mock_plan

        complex_prompt = 'backup "user files" & logs @3AM #priority'
        result = self.runner.invoke(app, ["plan", complex_prompt])

        assert result.exit_code == 0
        mock_coordination.handle_prompt.assert_called_once_with(complex_prompt)

    @patch("shardguard.cli.CoordinationService")
    def test_plan_command_empty_prompt(self, mock_coordination_class):
        """Test plan command with empty prompt."""
        mock_coordination = Mock()
        mock_coordination_class.return_value = mock_coordination

        mock_plan = Plan(original_prompt="", sub_prompts=[])
        mock_coordination.handle_prompt.return_value = mock_plan

        result = self.runner.invoke(app, ["plan", ""])

        assert result.exit_code == 0
        mock_coordination.handle_prompt.assert_called_once_with("")

    @patch("shardguard.cli.CoordinationService")
    def test_plan_command_uses_mock_planning_llm(self, mock_coordination_class):
        """Test that plan command initializes CoordinationService with MockPlanningLLM."""
        mock_coordination = Mock()
        mock_coordination_class.return_value = mock_coordination

        mock_plan = Plan(
            original_prompt="test",
            sub_prompts=[SubPrompt(id=1, content="task", opaque_values={})],
        )
        mock_coordination.handle_prompt.return_value = mock_plan

        with patch("shardguard.dev_utils.MockPlanningLLM") as mock_planner_class:
            mock_planner = Mock()
            mock_planner_class.return_value = mock_planner

            result = self.runner.invoke(app, ["plan", "test"])

            assert result.exit_code == 0
            mock_planner_class.assert_called_once()
            mock_coordination_class.assert_called_once_with(mock_planner)


class TestCLIErrorHandling:
    """Test error handling in CLI."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("shardguard.cli.CoordinationService")
    def test_plan_command_coordination_error(self, mock_coordination_class):
        """Test plan command when coordination service raises an error."""
        mock_coordination = Mock()
        mock_coordination_class.return_value = mock_coordination
        mock_coordination.handle_prompt.side_effect = ValueError("Test error")

        result = self.runner.invoke(app, ["plan", "test prompt"])

        # Should not crash, but might have non-zero exit code
        assert "Test error" in result.stdout or result.exit_code != 0

    @patch("shardguard.cli.CoordinationService")
    def test_plan_command_json_serialization_error(self, mock_coordination_class):
        """Test plan command when JSON serialization fails."""
        mock_coordination = Mock()
        mock_coordination_class.return_value = mock_coordination

        # Create a mock plan that might cause serialization issues
        mock_plan = Mock()
        mock_plan.model_dump_json.side_effect = Exception("Serialization error")
        mock_coordination.handle_prompt.return_value = mock_plan

        result = self.runner.invoke(app, ["plan", "test"])

        # Should handle error gracefully
        assert result.exit_code != 0 or "error" in result.stdout.lower()

    @patch("shardguard.cli.CoordinationService")
    def test_plan_command_import_error_handling(self, mock_coordination_class):
        """Test plan command handles import errors gracefully."""
        # Simulate import error
        mock_coordination_class.side_effect = ImportError("Module not found")

        result = self.runner.invoke(app, ["plan", "test"])

        # Should not crash with unhandled exception
        assert result.exit_code != 0


class TestCLIIntegration:
    """Integration tests for CLI with real components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_plan_command_real_integration(self):
        """Test plan command with real MockPlanningLLM integration."""
        result = self.runner.invoke(app, ["plan", "backup files at 3 AM"])

        assert result.exit_code == 0

        # Extract and parse JSON from output
        try:
            output_data = extract_json_from_cli_output(result.stdout)
            assert isinstance(output_data, dict)
            assert "original_prompt" in output_data
            assert "sub_prompts" in output_data
        except (json.JSONDecodeError, ValueError):
            pytest.fail(f"CLI output does not contain valid JSON: {result.stdout}")

    def test_plan_command_real_number_extraction(self):
        """Test that real integration properly extracts numbers."""
        result = self.runner.invoke(app, ["plan", "schedule at 15:30 with 5 retries"])

        assert result.exit_code == 0

        # Extract and parse JSON from output
        output_data = extract_json_from_cli_output(result.stdout)
        opaque_values = output_data["sub_prompts"][0]["opaque_values"]

        # Should extract numbers: 15, 30, 5
        extracted_numbers = list(opaque_values.values())
        assert "15" in extracted_numbers
        assert "30" in extracted_numbers
        assert "5" in extracted_numbers

    def test_multiple_plan_commands(self):
        """Test multiple plan commands work independently."""
        test_cases = [
            ("backup at 1 AM", ["1"]),
            ("sync every 2 hours", ["2"]),
            ("cleanup after 3 days", ["3"]),
        ]

        for prompt, expected_numbers in test_cases:
            result = self.runner.invoke(app, ["plan", prompt])

            assert result.exit_code == 0

            # Extract and parse JSON from output
            output_data = extract_json_from_cli_output(result.stdout)
            opaque_values = output_data["sub_prompts"][0]["opaque_values"]
            extracted_numbers = list(opaque_values.values())

            for num in expected_numbers:
                assert num in extracted_numbers


class TestCLIOutput:
    """Test CLI output formatting and presentation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("shardguard.cli.CoordinationService")
    def test_plan_command_json_formatting(self, mock_coordination_class):
        """Test that plan command outputs properly formatted JSON."""
        mock_coordination = Mock()
        mock_coordination_class.return_value = mock_coordination

        mock_plan = Plan(
            original_prompt="test prompt",
            sub_prompts=[
                SubPrompt(
                    id=1,
                    content="first task",
                    opaque_values={"[VALUE_1]": "val1", "[VALUE_2]": "val2"},
                ),
                SubPrompt(id=2, content="second task", opaque_values={}),
            ],
        )
        mock_coordination.handle_prompt.return_value = mock_plan

        result = self.runner.invoke(app, ["plan", "test"])

        assert result.exit_code == 0

        # Check JSON structure
        output_data = json.loads(result.stdout.strip())
        assert output_data["original_prompt"] == "test prompt"
        assert len(output_data["sub_prompts"]) == 2
        assert output_data["sub_prompts"][0]["opaque_values"] == {
            "[VALUE_1]": "val1",
            "[VALUE_2]": "val2",
        }
        assert output_data["sub_prompts"][1]["opaque_values"] == {}

    def test_plan_command_preserves_unicode(self):
        """Test that plan command properly handles Unicode characters."""
        result = self.runner.invoke(app, ["plan", "backup café files at 3€"])

        assert result.exit_code == 0

        # Extract and parse JSON from output
        output_data = extract_json_from_cli_output(result.stdout)
        assert isinstance(output_data, dict)
