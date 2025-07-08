"""Tests for ShardGuard CLI functionality."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
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
        assert (
            "Available commands:" in result.output
            or "execute, list-tools, plan" in result.output
        )

    def test_plan_command_help(self):
        """Test plan command help."""
        result = self.runner.invoke(app, ["plan", "--help"])

        assert result.exit_code == 0
        assert "Generate a safe execution plan" in result.output
        assert "--model" in result.output
        assert "--ollama-url" in result.output

    @patch("shardguard.core.coordination.AsyncCoordinationService")
    @patch("shardguard.cli.get_mcp_planner")
    def test_plan_command_success(
        self, mock_get_mcp_planner, mock_coordination_service
    ):
        """Test successful plan command execution."""
        # Setup mcp planner mock
        mock_planner_instance = AsyncMock()
        mock_planner_instance.get_available_tools_description.return_value = (
            "No MCP tools available."
        )
        mock_get_mcp_planner.return_value = mock_planner_instance

        mock_coordination_instance = AsyncMock()
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
        mock_get_mcp_planner.assert_called_once()
        mock_coordination_service.assert_called_once_with(mock_planner_instance)

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
    @patch("shardguard.core.coordination.AsyncCoordinationService")
    @patch("shardguard.cli.get_mcp_planner")
    def test_plan_command_with_custom_parameters(
        self,
        mock_get_mcp_planner,
        mock_coordination_service,
        args,
        expected_model,
        expected_url,
    ):
        """Test plan command with custom parameters."""
        mock_planner_instance = AsyncMock()
        mock_planner_instance.get_available_tools_description.return_value = (
            "No MCP tools available."
        )
        mock_get_mcp_planner.return_value = mock_planner_instance

        mock_coordination_instance = AsyncMock()
        mock_coordination_service.return_value = mock_coordination_instance

        sample_plan = Plan(original_prompt="Test", sub_prompts=[])
        mock_coordination_instance.handle_prompt.return_value = sample_plan

        result = self.runner.invoke(app, args)

        assert result.exit_code == 0
        mock_get_mcp_planner.assert_called_once()

    @patch("shardguard.core.coordination.AsyncCoordinationService")
    @patch("shardguard.cli.get_mcp_planner")
    def test_plan_command_connection_error(
        self, mock_get_mcp_planner, mock_coordination_service
    ):
        """Test plan command handling of connection errors."""
        mock_planner_instance = AsyncMock()
        mock_planner_instance.get_available_tools_description.return_value = (
            "No MCP tools available."
        )
        mock_get_mcp_planner.return_value = mock_planner_instance

        mock_coordination_instance = AsyncMock()
        mock_coordination_service.return_value = mock_coordination_instance

        # Simulate connection error
        mock_coordination_instance.handle_prompt.side_effect = ConnectionError(
            "Connection failed"
        )

        result = self.runner.invoke(app, ["plan", "Test prompt"])

        assert result.exit_code == 1
        assert "Connection Error" in result.output

    @patch("shardguard.core.coordination.AsyncCoordinationService")
    @patch("shardguard.cli.get_mcp_planner")
    def test_plan_command_general_error(
        self, mock_get_mcp_planner, mock_coordination_service
    ):
        """Test plan command handling of general errors."""
        mock_planner_instance = AsyncMock()
        mock_planner_instance.get_available_tools_description.return_value = (
            "No MCP tools available."
        )
        mock_get_mcp_planner.return_value = mock_planner_instance

        mock_coordination_instance = AsyncMock()
        mock_coordination_service.return_value = mock_coordination_instance

        # Simulate general error
        mock_coordination_instance.handle_prompt.side_effect = ValueError("Some error")

        result = self.runner.invoke(app, ["plan", "Test prompt"])

        assert result.exit_code == 1
        assert "Error:" in result.output

    @patch("shardguard.core.coordination.AsyncCoordinationService")
    @patch("shardguard.cli.get_mcp_planner")
    def test_plan_command_console_output(
        self, mock_get_mcp_planner, mock_coordination_service
    ):
        """Test that plan command shows console output for model info."""
        mock_planner_instance = AsyncMock()
        mock_planner_instance.get_available_tools_description.return_value = (
            "No MCP tools available."
        )
        mock_get_mcp_planner.return_value = mock_planner_instance

        mock_coordination_instance = AsyncMock()
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
        assert "Using Ollama model: test-model at http://test:1234" in result.output

    def test_plan_command_required_prompt_argument(self):
        """Test that plan command requires a prompt argument."""
        result = self.runner.invoke(app, ["plan"])

        assert result.exit_code == 2  # Click error code for missing argument
        assert "Missing argument" in result.output

    @patch("shardguard.core.coordination.AsyncCoordinationService")
    @patch("shardguard.cli.get_mcp_planner")
    def test_plan_command_json_output_format(
        self, mock_get_mcp_planner, mock_coordination_service
    ):
        """Test that plan command outputs properly formatted JSON."""
        mock_planner_instance = AsyncMock()
        mock_planner_instance.get_available_tools_description.return_value = (
            "No MCP tools available."
        )
        mock_get_mcp_planner.return_value = mock_planner_instance

        mock_coordination_instance = AsyncMock()
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

        # Parse the JSON output to ensure it's valid
        lines = result.output.split("\n")
        json_lines = [
            line for line in lines if line.strip() and not line.startswith("Using")
        ]
        json_text = "\n".join(json_lines)

        parsed_plan = json.loads(json_text)
        assert parsed_plan["original_prompt"] == "Complex task with [[P1]]"
        assert len(parsed_plan["sub_prompts"]) == 2

    def test_list_tools_command(self):
        """Test the list-tools command."""
        result = self.runner.invoke(app, ["list-tools"])

        assert result.exit_code == 0
        assert "Available MCP Tools:" in result.output
        # Either MCP tools are available or not, both are valid
        assert "No MCP tools available." in result.output or "Server:" in result.output

    def test_main_function_with_context(self):
        """Test main function with context that has subcommand."""
        # This test validates the callback behavior when subcommands are invoked
        # The main function should not run the welcome message
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Welcome to ShardGuard!" not in result.output


class TestMainCallback:
    """Test cases for the main callback function."""

    def test_main_callback_returns_none_with_subcommand(self):
        """Test main callback when invoked_subcommand is not None."""
        # Create a mock context with a subcommand
        mock_context = Mock()
        mock_context.invoked_subcommand = "plan"

        # This should not raise any exceptions and should return None
        result = main(mock_context)
        assert result is None

    def test_main_callback_behavior_without_subcommand(self):
        """Test main callback when no subcommand is invoked."""
        # This is tested indirectly through the CLI runner test above
        pass
