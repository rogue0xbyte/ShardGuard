"""Tests for ShardGuard input sanitization module."""

from io import StringIO
from unittest.mock import patch

import pytest

from shardguard.core.sanitization import InputSanitizer, SanitizationResult


class TestSanitizationResult:
    """Test cases for SanitizationResult."""

    def test_sanitization_result_creation(self):
        """Test basic SanitizationResult creation."""
        result = SanitizationResult("cleaned", ["change1", "change2"], 100)

        assert result.sanitized_input == "cleaned"
        assert result.changes_made == ["change1", "change2"]
        assert result.original_length == 100
        assert result.final_length == 7  # len("cleaned")


class TestInputSanitizer:
    """Test cases for InputSanitizer."""

    def test_sanitizer_initialization_default(self):
        """Test InputSanitizer initialization with defaults."""
        sanitizer = InputSanitizer()

        assert sanitizer.max_length == 10000
        assert sanitizer.console is not None
        assert len(sanitizer.dangerous_patterns) == 3

    def test_sanitizer_initialization_custom(self):
        """Test InputSanitizer initialization with custom parameters."""
        from rich.console import Console

        custom_console = Console()

        sanitizer = InputSanitizer(console=custom_console, max_length=5000)

        assert sanitizer.max_length == 5000
        assert sanitizer.console is custom_console

    def test_sanitize_clean_input(self):
        """Test sanitization with clean input."""
        sanitizer = InputSanitizer()

        with patch("sys.stdout", new_callable=StringIO):
            result = sanitizer.sanitize("clean input")

        assert result.sanitized_input == "clean input"
        assert result.changes_made == []
        assert result.original_length == 11
        assert result.final_length == 11

    def test_sanitize_empty_input_raises_error(self):
        """Test that empty input raises ValueError."""
        sanitizer = InputSanitizer()

        with patch("sys.stdout", new_callable=StringIO):
            with pytest.raises(ValueError, match="User input cannot be empty"):
                sanitizer.sanitize("")

            with pytest.raises(ValueError, match="User input cannot be empty"):
                sanitizer.sanitize("   ")

    def test_sanitize_whitespace_normalization(self):
        """Test whitespace normalization."""
        sanitizer = InputSanitizer()

        with patch("sys.stdout", new_callable=StringIO):
            result = sanitizer.sanitize("  multiple   spaces  \n\t  ")

        assert result.sanitized_input == "multiple spaces"
        assert "✓ Normalized whitespace and line endings" in result.changes_made

    def test_sanitize_control_character_removal(self):
        """Test removal of dangerous control characters."""
        sanitizer = InputSanitizer()

        # Test with various control characters
        dangerous_input = "test\x00\x01\x08\x0b\x0c\x0e\x1f\x7fdata"

        with patch("sys.stdout", new_callable=StringIO):
            result = sanitizer.sanitize(dangerous_input)

        # Control characters should be removed
        assert "test" in result.sanitized_input
        assert "data" in result.sanitized_input
        assert "✓ Removed dangerous control characters" in result.changes_made
        assert len(result.sanitized_input) < len(dangerous_input)

    def test_sanitize_length_truncation(self):
        """Test truncation of extremely long inputs."""
        sanitizer = InputSanitizer(max_length=100)

        long_input = "x" * 150  # Longer than max_length

        with patch("sys.stdout", new_callable=StringIO):
            result = sanitizer.sanitize(long_input)

        assert len(result.sanitized_input) == 100 + len("... [truncated]")
        assert result.sanitized_input.endswith("... [truncated]")
        assert "✓ Truncated input to 100 characters" in result.changes_made

    @pytest.mark.parametrize(
        "malicious_input,expected_pattern",
        [
            ('backup <script>alert("xss")</script> files', "Script tags"),
            ("visit javascript:alert('xss') this site", "JavaScript URLs"),
            ("load data:text/html,<h1>test</h1> content", "HTML data URLs"),
            ("test <SCRIPT>alert('test')</SCRIPT> data", "Script tags"),
            ("test JavaScript:void(0) data", "JavaScript URLs"),
        ],
    )
    def test_sanitize_dangerous_patterns(self, malicious_input, expected_pattern):
        """Test removal of various malicious content patterns."""
        sanitizer = InputSanitizer()

        with patch("sys.stdout", new_callable=StringIO):
            result = sanitizer.sanitize(malicious_input)

        assert "[REMOVED]" in result.sanitized_input
        assert f"✓ Removed {expected_pattern}" in result.changes_made

    def test_sanitize_preserves_legitimate_content(self):
        """Test that sanitization preserves legitimate content."""
        sanitizer = InputSanitizer()

        legitimate_inputs = [
            "backup /home/user/files to /backup/location",
            "connect to database://server:5432/mydb",
            "process data from file.txt at 3:30 AM",
            "user@domain.com needs access to folder_123",
            "run command with --flag=value --other-flag",
        ]

        for test_input in legitimate_inputs:
            with patch("sys.stdout", new_callable=StringIO):
                result = sanitizer.sanitize(test_input)

            # Should be unchanged (maybe whitespace normalized)
            assert (
                test_input.strip() == result.sanitized_input
                or test_input == result.sanitized_input
            )

    def test_sanitize_no_progress_display(self):
        """Test sanitization without progress display."""
        sanitizer = InputSanitizer()

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = sanitizer.sanitize("test input", show_progress=False)
            output = mock_stdout.getvalue()

        assert result.sanitized_input == "test input"
        assert output == ""  # No console output

    def test_sanitize_with_progress_display(self):
        """Test sanitization with progress display."""
        sanitizer = InputSanitizer()

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            sanitizer.sanitize("  test   input  ", show_progress=True)
            output = mock_stdout.getvalue()

        # Should contain sanitization process indicators
        assert "Input Sanitization Process" in output
        assert "Original Input" in output

    def test_sanitize_multiple_changes(self):
        """Test sanitization with multiple types of changes."""
        sanitizer = InputSanitizer()

        complex_input = "  <script>alert('test')</script>   with\x00\x01control chars  "

        with patch("sys.stdout", new_callable=StringIO):
            result = sanitizer.sanitize(complex_input)

        # Should have multiple changes
        assert len(result.changes_made) > 1
        assert any("whitespace" in change.lower() for change in result.changes_made)
        assert any(
            "control characters" in change.lower() for change in result.changes_made
        )
        assert any("Script tags" in change for change in result.changes_made)

    def test_add_dangerous_pattern(self):
        """Test adding custom dangerous patterns."""
        sanitizer = InputSanitizer()
        original_count = len(sanitizer.dangerous_patterns)

        sanitizer.add_dangerous_pattern(r"eval\(", "Eval calls")

        assert len(sanitizer.dangerous_patterns) == original_count + 1

        with patch("sys.stdout", new_callable=StringIO):
            result = sanitizer.sanitize("test eval('code') here")

        assert "[REMOVED]" in result.sanitized_input
        assert "✓ Removed Eval calls" in result.changes_made

    def test_remove_dangerous_pattern(self):
        """Test removing dangerous patterns."""
        sanitizer = InputSanitizer()
        original_count = len(sanitizer.dangerous_patterns)

        # Remove the script pattern
        sanitizer.remove_dangerous_pattern(r"<script[^>]*>.*?</script>")

        assert len(sanitizer.dangerous_patterns) == original_count - 1

        with patch("sys.stdout", new_callable=StringIO):
            result = sanitizer.sanitize("<script>alert('test')</script>")

        # Script should not be removed now
        assert "[REMOVED]" not in result.sanitized_input
        assert "Script tags" not in " ".join(result.changes_made)


class TestInputSanitizerPrivateMethods:
    """Test cases for private methods of InputSanitizer."""

    def test_normalize_whitespace(self):
        """Test whitespace normalization method."""
        sanitizer = InputSanitizer()

        result, changes = sanitizer._normalize_whitespace("  test   input  \n\t")

        assert result == "test input"
        assert "✓ Normalized whitespace and line endings" in changes

    def test_normalize_whitespace_no_change(self):
        """Test whitespace normalization when no change is needed."""
        sanitizer = InputSanitizer()

        result, changes = sanitizer._normalize_whitespace("test input")

        assert result == "test input"
        assert changes == []

    def test_remove_control_characters(self):
        """Test control character removal method."""
        sanitizer = InputSanitizer()

        result, changes = sanitizer._remove_control_characters("test\x00\x01data")

        assert result == "testdata"
        assert "✓ Removed dangerous control characters" in changes

    def test_remove_control_characters_no_change(self):
        """Test control character removal when no change is needed."""
        sanitizer = InputSanitizer()

        result, changes = sanitizer._remove_control_characters("clean text")

        assert result == "clean text"
        assert changes == []

    def test_truncate_long_input(self):
        """Test input truncation method."""
        sanitizer = InputSanitizer(max_length=10)

        result, changes = sanitizer._truncate_long_input("x" * 20)

        assert len(result) == 10 + len("... [truncated]")
        assert result.endswith("... [truncated]")
        assert "✓ Truncated input to 10 characters" in changes

    def test_truncate_long_input_no_change(self):
        """Test input truncation when no change is needed."""
        sanitizer = InputSanitizer(max_length=100)

        result, changes = sanitizer._truncate_long_input("short")

        assert result == "short"
        assert changes == []

    def test_remove_dangerous_patterns(self):
        """Test dangerous pattern removal method."""
        sanitizer = InputSanitizer()

        result, changes = sanitizer._remove_dangerous_patterns(
            "test <script>alert('x')</script> data"
        )

        assert "[REMOVED]" in result
        assert "✓ Removed Script tags" in changes

    def test_remove_dangerous_patterns_no_change(self):
        """Test dangerous pattern removal when no change is needed."""
        sanitizer = InputSanitizer()

        result, changes = sanitizer._remove_dangerous_patterns("safe text")

        assert result == "safe text"
        assert changes == []
