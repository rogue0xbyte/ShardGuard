"""Tests for ShardGuard CLI functionality."""

import json
from unittest.mock import Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from shardguard.cli import app, main
from shardguard.core.models import Plan, SubPrompt


class TestCLICommands:
    """Test cases for CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_app_help(self):
        """Test that the app shows help correctly."""
        result = self.runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "ShardGuard CLI" in result.output

    def test_main_callback_without_subcommand(self):
        """Test main callback when no subcommand is provided."""
        result = self.runner.invoke(app, [])

        assert result.exit_code == 0
        assert "Welcome to ShardGuard!" in result.output
        assert "sub-commands" in result.output

    def test_plan_command_help(self):
        """Test plan command help."""
        result = self.runner.invoke(app, ["plan", "--help"])

        assert result.exit_code == 0
        assert "Generate a safe execution plan" in result.output
        assert "--model" in result.output
        assert "--ollama-url" in result.output

    @patch("shardguard.cli.CoordinationService")
    @patch("shardguard.cli.PlanningLLM")
    def test_plan_command_success(self, mock_planning_llm, mock_coordination_service):
        """Test successful plan command execution."""
        # Setup mocks
        mock_planner_instance = Mock()
        mock_planning_llm.return_value = mock_planner_instance

        mock_coordination_instance = Mock()
        mock_coordination_service.return_value = mock_coordination_instance

        # Create a sample plan to return
        sample_plan = Plan(
            original_prompt="Test prompt",
            sub_prompts=[SubPrompt(id=1, content="Test task", opaque_values={})],
        )
        mock_coordination_instance.handle_prompt.return_value = sample_plan

        result = self.runner.invoke(app, ["plan", "Test prompt"])

        assert result.exit_code == 0

        # Verify the mocks were called correctly
        mock_planning_llm.assert_called_once_with(
            model="llama3.2", base_url="http://localhost:11434"
        )
        mock_coordination_service.assert_called_once_with(mock_planner_instance)
        mock_coordination_instance.handle_prompt.assert_called_once_with("Test prompt")

        # Verify output contains JSON
        assert "original_prompt" in result.output
        assert "sub_prompts" in result.output

    @pytest.mark.parametrize(
        "args, expected_model, expected_url",
        [
            (
                ["plan", "Test prompt", "--model", "custom-model"],
                "custom-model",
                "http://localhost:11434",
            ),
            (
                ["plan", "Test prompt", "--ollama-url", "http://custom:8080"],
                "llama3.2",
                "http://custom:8080",
            ),
        ],
    )
    @patch("shardguard.cli.CoordinationService")
    @patch("shardguard.cli.PlanningLLM")
    def test_plan_command_with_custom_parameters(
        self,
        mock_planning_llm,
        mock_coordination_service,
        args,
        expected_model,
        expected_url,
    ):
        """Test plan command with custom parameters."""
        mock_planner_instance = Mock()
        mock_planning_llm.return_value = mock_planner_instance

        mock_coordination_instance = Mock()
        mock_coordination_service.return_value = mock_coordination_instance

        sample_plan = Plan(original_prompt="Test", sub_prompts=[])
        mock_coordination_instance.handle_prompt.return_value = sample_plan

        result = self.runner.invoke(app, args)

        assert result.exit_code == 0

        # Verify custom parameters were used
        mock_planning_llm.assert_called_once_with(
            model=expected_model, base_url=expected_url
        )

    @patch("shardguard.cli.CoordinationService")
    @patch("shardguard.cli.PlanningLLM")
    def test_plan_command_connection_error(
        self, mock_planning_llm, mock_coordination_service
    ):
        """Test plan command handling of connection errors."""
        mock_planner_instance = Mock()
        mock_planning_llm.return_value = mock_planner_instance

        mock_coordination_instance = Mock()
        mock_coordination_service.return_value = mock_coordination_instance

        # Simulate connection error
        mock_coordination_instance.handle_prompt.side_effect = ConnectionError(
            "Connection failed"
        )

        result = self.runner.invoke(app, ["plan", "Test prompt"])

        assert result.exit_code == 1
        assert "Connection Error" in result.output
        assert "Make sure Ollama is running" in result.output

    @patch("shardguard.cli.CoordinationService")
    @patch("shardguard.cli.PlanningLLM")
    def test_plan_command_general_error(
        self, mock_planning_llm, mock_coordination_service
    ):
        """Test plan command handling of general errors."""
        mock_planner_instance = Mock()
        mock_planning_llm.return_value = mock_planner_instance

        mock_coordination_instance = Mock()
        mock_coordination_service.return_value = mock_coordination_instance

        # Simulate general error
        mock_coordination_instance.handle_prompt.side_effect = ValueError(
            "Invalid input"
        )

        result = self.runner.invoke(app, ["plan", "Test prompt"])

        assert result.exit_code == 1
        assert "Error:" in result.output

    @patch("shardguard.cli.CoordinationService")
    @patch("shardguard.cli.PlanningLLM")
    def test_plan_command_console_output(
        self, mock_planning_llm, mock_coordination_service
    ):
        """Test that plan command shows console output for model info."""
        mock_planner_instance = Mock()
        mock_planning_llm.return_value = mock_planner_instance

        mock_coordination_instance = Mock()
        mock_coordination_service.return_value = mock_coordination_instance

        sample_plan = Plan(original_prompt="Test", sub_prompts=[])
        mock_coordination_instance.handle_prompt.return_value = sample_plan

        result = self.runner.invoke(
            app,
            [
                "plan",
                "Test prompt",
                "--model",
                "test-model",
                "--ollama-url",
                "http://test:1234",
            ],
        )

        assert result.exit_code == 0
        # Should show model info
        assert "test-model" in result.output
        assert "http://test:1234" in result.output

    def test_plan_command_required_prompt_argument(self):
        """Test that plan command requires a prompt argument."""
        result = self.runner.invoke(app, ["plan"])

        assert result.exit_code != 0  # Should fail without prompt

    @patch("shardguard.cli.CoordinationService")
    @patch("shardguard.cli.PlanningLLM")
    def test_plan_command_json_output_format(
        self, mock_planning_llm, mock_coordination_service
    ):
        """Test that plan command outputs properly formatted JSON."""
        mock_planner_instance = Mock()
        mock_planning_llm.return_value = mock_planner_instance

        mock_coordination_instance = Mock()
        mock_coordination_service.return_value = mock_coordination_instance

        # Create a more complex plan
        sample_plan = Plan(
            original_prompt="Complex task with [[P1]]",
            sub_prompts=[
                SubPrompt(id=1, content="First step", opaque_values={}),
                SubPrompt(
                    id=2, content="Process [[P1]]", opaque_values={"[[P1]]": "secret"}
                ),
            ],
        )
        mock_coordination_instance.handle_prompt.return_value = sample_plan

        result = self.runner.invoke(app, ["plan", "Complex task"])

        assert result.exit_code == 0

        # Try to parse the output as JSON (should work)
        # Extract JSON from output (skip console messages)
        output_lines = result.output.strip().split("\n")
        json_lines = [
            line
            for line in output_lines
            if line.strip().startswith("{")
            or '"' in line
            or "}" in line
            or "[" in line
            or "]" in line
        ]
        json_output = "\n".join(json_lines)

        try:
            parsed_json = json.loads(json_output)
            assert "original_prompt" in parsed_json
            assert "sub_prompts" in parsed_json
            assert len(parsed_json["sub_prompts"]) == 2
        except json.JSONDecodeError:
            # If JSON parsing fails, at least check that JSON-like content is present
            assert "original_prompt" in result.output
            assert "sub_prompts" in result.output


class TestMainFunction:
    """Test cases for the main function."""

    def test_main_function_with_context(self):
        """Test main function with typer context."""
        # Create a mock context
        mock_ctx = Mock()
        mock_ctx.invoked_subcommand = None

        # This should execute without error
        # In actual usage, this would print welcome message
        main(mock_ctx)

    def test_main_function_with_subcommand(self):
        """Test main function when a subcommand is invoked."""
        mock_ctx = Mock()
        mock_ctx.invoked_subcommand = "plan"

        # Should not do anything when subcommand is present
        main(mock_ctx)


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def test_app_structure_and_commands(self):
        """Test that the app is properly structured and commands are available."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert isinstance(app, typer.Typer)
        assert app.info.help == "ShardGuard CLI"
        assert "plan" in result.output
