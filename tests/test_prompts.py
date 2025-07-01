"""Tests for ShardGuard prompt templates."""

from shardguard.core.prompts import ERROR_HANDLING_PROMPT, PLANNING_PROMPT


class TestPlanningPrompt:
    """Test cases for PLANNING_PROMPT template."""

    def test_planning_prompt_format_basic(self):
        """Test basic formatting of the planning prompt."""
        user_input = "backup my files"
        formatted = PLANNING_PROMPT.format(user_prompt=user_input)

        assert "backup my files" in formatted
        assert "ShardGuard" in formatted
        assert "USER_PROMPT:" in formatted
        assert "END." in formatted

    def test_planning_prompt_contains_instructions(self):
        """Test that prompt contains necessary instructions."""
        user_input = "test prompt"
        formatted = PLANNING_PROMPT.format(user_prompt=user_input)

        # Check for key instruction elements
        assert "sensitive and private information" in formatted
        assert "reference placeholders" in formatted
        assert "[USERNAME]" in formatted
        assert "[PASSWORD]" in formatted
        assert "JSON document" in formatted

    def test_planning_prompt_contains_json_schema(self):
        """Test that prompt contains proper JSON response schema."""
        user_input = "test"
        formatted = PLANNING_PROMPT.format(user_prompt=user_input)

        # Check for JSON schema elements
        assert '"original_prompt"' in formatted
        assert '"sub_prompts"' in formatted
        assert '"id"' in formatted
        assert '"content"' in formatted
        assert '"opaque_values"' in formatted

    def test_planning_prompt_with_special_characters(self):
        """Test formatting with special characters in user input."""
        user_input = 'test with "quotes" and \n newlines'
        formatted = PLANNING_PROMPT.format(user_prompt=user_input)

        assert user_input in formatted
        assert "USER_PROMPT:" in formatted

    def test_planning_prompt_with_empty_input(self):
        """Test formatting with empty user input."""
        user_input = ""
        formatted = PLANNING_PROMPT.format(user_prompt=user_input)

        assert "USER_PROMPT:\n\nEND." in formatted

    def test_planning_prompt_with_long_input(self):
        """Test formatting with very long user input."""
        user_input = "x" * 1000
        formatted = PLANNING_PROMPT.format(user_prompt=user_input)

        assert user_input in formatted
        assert len(formatted) > 1000


class TestErrorHandlingPrompt:
    """Test cases for ERROR_HANDLING_PROMPT template."""

    def test_error_handling_prompt_format_basic(self):
        """Test basic formatting of the error handling prompt."""
        error = "JSON parsing failed"
        original = "backup files"
        formatted = ERROR_HANDLING_PROMPT.format(error=error, original_prompt=original)

        assert "JSON parsing failed" in formatted
        assert "backup files" in formatted
        assert "error occurred" in formatted

    def test_error_handling_prompt_instructions(self):
        """Test that error prompt contains retry instructions."""
        error = "test error"
        original = "test prompt"
        formatted = ERROR_HANDLING_PROMPT.format(error=error, original_prompt=original)

        assert "retry" in formatted.lower()
        assert "subtasks" in formatted
        assert "opaque values" in formatted

    def test_error_handling_prompt_with_complex_error(self):
        """Test formatting with complex error message."""
        error = 'ValidationError: {"field": "missing required field"}'
        original = "complex user request with multiple steps"
        formatted = ERROR_HANDLING_PROMPT.format(error=error, original_prompt=original)

        assert error in formatted
        assert original in formatted

    def test_error_handling_prompt_with_empty_values(self):
        """Test formatting with empty error and prompt."""
        formatted = ERROR_HANDLING_PROMPT.format(error="", original_prompt="")

        assert "error occurred" in formatted
        assert "Original prompt:" in formatted


class TestPromptConstants:
    """Test cases for prompt template constants."""

    def test_planning_prompt_is_string(self):
        """Test that PLANNING_PROMPT is a string."""
        assert isinstance(PLANNING_PROMPT, str)
        assert len(PLANNING_PROMPT) > 0

    def test_error_handling_prompt_is_string(self):
        """Test that ERROR_HANDLING_PROMPT is a string."""
        assert isinstance(ERROR_HANDLING_PROMPT, str)
        assert len(ERROR_HANDLING_PROMPT) > 0

    def test_planning_prompt_has_placeholder(self):
        """Test that PLANNING_PROMPT has the required placeholder."""
        assert "{user_prompt}" in PLANNING_PROMPT

    def test_error_handling_prompt_has_placeholders(self):
        """Test that ERROR_HANDLING_PROMPT has required placeholders."""
        assert "{error}" in ERROR_HANDLING_PROMPT
        assert "{original_prompt}" in ERROR_HANDLING_PROMPT

    def test_prompts_not_empty_after_formatting(self):
        """Test that prompts are not empty after formatting."""
        planning_formatted = PLANNING_PROMPT.format(user_prompt="test")
        error_formatted = ERROR_HANDLING_PROMPT.format(
            error="test_error", original_prompt="test_prompt"
        )

        assert len(planning_formatted.strip()) > 0
        assert len(error_formatted.strip()) > 0
