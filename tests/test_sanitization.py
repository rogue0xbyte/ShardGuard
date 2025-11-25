"""Tests for ShardGuard input sanitization functionality."""

import pytest
from rich.console import Console

from shardguard.core.sanitization import InputSanitizer, SanitizationResult


class TestSanitizationResult:
    """Test cases for SanitizationResult class."""

    def test_sanitization_result_creation(self):
        """Test creating a SanitizationResult."""
        changes = ["✓ Normalized whitespace", "✓ Removed control characters"]
        result = SanitizationResult("clean input", changes, 15)

        assert result.sanitized_input == "clean input"
        assert result.changes_made == changes
        assert result.original_length == 15
        assert result.final_length == 11  # len("clean input")

    def test_sanitization_result_no_changes(self):
        """Test SanitizationResult with no changes made."""
        result = SanitizationResult("unchanged", [], 9)

        assert result.sanitized_input == "unchanged"
        assert result.changes_made == []
        assert result.original_length == 9
        assert result.final_length == 9


class TestInputSanitizer:
    """Test cases for InputSanitizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use a console that doesn't output to avoid cluttering test output
        console = Console(file=open("/dev/null", "w"), stderr=False)
        self.sanitizer = InputSanitizer(console=console)

    def test_sanitizer_initialization_default(self):
        """Test InputSanitizer initialization with defaults."""
        sanitizer = InputSanitizer()

        assert sanitizer.max_length == 10000
        assert sanitizer.console is not None
        assert len(sanitizer.dangerous_patterns) > 0

    def test_sanitizer_initialization_custom(self):
        """Test InputSanitizer initialization with custom parameters."""
        console = Console()
        sanitizer = InputSanitizer(console=console, max_length=5000)

        assert sanitizer.max_length == 5000
        assert sanitizer.console is console

    @pytest.mark.parametrize(
        "input_text, expected_output, expected_changes",
        [
            ("Hello world", "Hello world", []),
            ("   \n\t  ", None, ["User input cannot be empty"]),
            ("Hello\n\n   world\t\ttest   ", "Hello world test", ["whitespace"]),
            ("A" * 15000, None, ["truncated"]),
            ("Hello <script>alert('xss')</script> world", "Hello  world", ["script"]),
        ],
    )
    def test_sanitize_input(self, input_text, expected_output, expected_changes):
        """Test sanitizing various inputs."""
        if expected_output is None:
            with pytest.raises(ValueError, match=expected_changes[0]):
                self.sanitizer.sanitize(input_text, show_progress=False)
        else:
            result = self.sanitizer.sanitize(input_text, show_progress=False)

            assert result.sanitized_input == expected_output
            for change in expected_changes:
                assert any(change in c.lower() for c in result.changes_made)
