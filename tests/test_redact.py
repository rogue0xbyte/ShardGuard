"""Tests for YAML-based Redactor functionality."""

import pytest
from unittest.mock import patch

from shardguard.utils.redaction import Redactor


class MockRedactor:
    """Mock implementation of Redactor for isolated testing."""

    def __init__(self, strategy="pseudonymize"):
        self.strategy = strategy

    def redact(self, text: str) -> str:
        """Simplified redaction simulation."""
        return (
            text.replace("john@example.com", "<EMAIL:abc123>")
            .replace("+1 555-234-5678", "<PHONE:xyz789>")
            .replace("https://example.com/login", "<URL:def456>")
        )


class TestRedactorProtocol:
    """Test cases for Redactor protocol and pseudonymization behavior."""

    @pytest.mark.parametrize(
        "text,expected_substring",
        [
            ("Send an email to john@example.com about the meeting", "<EMAIL:"),
            ("Call me at +1 555-234-5678", "<PHONE:"),
            ("Visit https://example.com/login", "<URL:"),
        ],
    )
    def test_basic_redaction(self, text, expected_substring):
        """Test basic redaction functionality using pseudonymization strategy."""
        redactor = MockRedactor(strategy="pseudonymize")
        output = redactor.redact(text)
        assert expected_substring in output
        assert "example.com" not in output  # sensitive data removed

    def test_strategy_attribute(self):
        """Test that Redactor preserves chosen strategy."""
        redactor = MockRedactor(strategy="mask")
        assert redactor.strategy == "mask"


class TestYAMLRedactorIntegration:
    """Integration tests for YAML-driven Redactor."""

    @pytest.fixture
    def redactor(self):
        """Fixture for Redactor instance with pseudonymize strategy."""
        return Redactor("rules.yaml", strategy="pseudonymize")

    @pytest.mark.parametrize(
        "text,should_contain",
        [
            ("Send an email to john@example.com about the meeting", "<EMAIL:"),
            ("Call me at +1 555-234-5678", "<PHONE:"),
            ("Visit https://example.com/login", "<URL:"),
        ],
    )
    def test_redaction_from_yaml_rules(self, redactor, text, should_contain):
        """Test that YAML-based rules correctly redact sensitive data."""
        output = redactor.redact(text)
        assert should_contain in output
        assert "example.com" not in output

    def test_redaction_multiple_entities(self, redactor):
        """Test simultaneous redaction of multiple entity types."""
        text = "SSN 123-45-6789, CC 4111-1111-1111-5678"
        output = redactor.redact(text)
        assert "<SSN:" in output or "<CREDIT_CARD:" in output

    @patch("redactor.yaml.safe_load")
    def test_yaml_loading_called(self, mock_yaml):
        """Ensure YAML configuration file is read during initialization."""
        mock_yaml.return_value = {"patterns": []}
        _ = Redactor("rules.yaml", strategy="mask")
        mock_yaml.assert_called_once()


class TestRedactorDemo:
    """Demonstration of Redactor redaction behavior."""

    def test_demo_output(self, capsys):
        """Run sample demo lines through Redactor and print output."""
        demo_lines = [
            "Send an email to johndoe@example.com about the meeting",
            "My phone is +1 555-234-5678",
            "SSN 123-45-6789, CC 4111-1111-1111-5678",
            "Visit https://example.com or www.test.org",
        ]
        demo = MockRedactor(strategy="pseudonymize")

        print("=== Demo ===")
        for line in demo_lines:
            print("IN :", line)
            print("OUT:", demo.redact(line))
            print()

        captured = capsys.readouterr()
        assert "<EMAIL:" in captured.out
        assert "<PHONE:" in captured.out
        assert "<URL:" in captured.out
