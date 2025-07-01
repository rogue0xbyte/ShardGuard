"""Pytest configuration and shared fixtures for ShardGuard tests."""

import json
from io import StringIO
from unittest.mock import Mock

import pytest

from shardguard.core.models import Plan, SubPrompt
from tests.test_helpers import MockPlanningLLM


@pytest.fixture
def mock_planning_llm():
    """Fixture providing a mock planning LLM."""
    return Mock(spec=MockPlanningLLM)


@pytest.fixture
def sample_plan():
    """Fixture providing a sample Plan object for testing."""
    return Plan(
        original_prompt="Sample user request",
        sub_prompts=[
            SubPrompt(id=1, content="First task", opaque_values=["val1", "val2"]),
            SubPrompt(id=2, content="Second task", opaque_values=[]),
            SubPrompt(id=3, content="Third task", opaque_values=["secret123"]),
        ],
    )


@pytest.fixture
def sample_subprompt():
    """Fixture providing a sample SubPrompt object for testing."""
    return SubPrompt(
        id=1,
        content="Sample task with sensitive data",
        opaque_values=["user123", "password456", "192.168.1.1"],
    )


@pytest.fixture
def real_mock_planning_llm():
    """Fixture providing a real MockPlanningLLM instance."""
    return MockPlanningLLM()


@pytest.fixture
def captured_console_output():
    """Fixture to capture console output during tests."""
    with pytest.MonkeyPatch().context() as m:
        captured_output = StringIO()
        m.setattr("sys.stdout", captured_output)
        yield captured_output


@pytest.fixture
def sample_user_inputs():
    """Fixture providing various user input samples for testing."""
    return {
        "clean": "backup my files",
        "with_numbers": "backup at 3 AM with 5 retries",
        "with_whitespace": "  backup   files   ",
        "with_special_chars": 'backup "user files" & logs @3AM',
        "with_malicious": '<script>alert("xss")</script> backup files',
        "empty": "",
        "long": "x" * 15000,
        "unicode": "backup café files at 3€",
    }


@pytest.fixture
def mock_console():
    """Fixture providing a mock Rich console."""
    return Mock()


# Test markers for different test categories
pytest_markers = [
    "unit: marks tests as unit tests",
    "integration: marks tests as integration tests",
    "cli: marks tests as CLI tests",
    "slow: marks tests as slow running",
]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)


# Custom assertions for common test patterns
class ShardGuardAssertions:
    """Custom assertion helpers for ShardGuard tests."""

    @staticmethod
    def assert_valid_plan(plan):
        """Assert that a Plan object is valid and well-formed."""
        assert isinstance(plan, Plan)
        assert isinstance(plan.original_prompt, str)
        assert isinstance(plan.sub_prompts, list)

        for i, subprompt in enumerate(plan.sub_prompts):
            assert isinstance(subprompt, SubPrompt)
            assert isinstance(subprompt.id, int)
            assert isinstance(subprompt.content, str)
            assert isinstance(subprompt.opaque_values, list)
            assert subprompt.id > 0, f"SubPrompt {i} has invalid id: {subprompt.id}"

    @staticmethod
    def assert_sanitized_input(original, sanitized):
        """Assert that input has been properly sanitized."""
        # Should not contain dangerous patterns
        dangerous_patterns = ["<script", "javascript:", "data:text/html"]
        for pattern in dangerous_patterns:
            assert pattern.lower() not in sanitized.lower()

        # Should not have excessive whitespace
        assert not sanitized.startswith(" ")
        assert not sanitized.endswith(" ")
        assert "  " not in sanitized  # No double spaces

    @staticmethod
    def assert_valid_json_output(json_string):
        """Assert that a string is valid JSON with expected structure."""
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON: {e}")

        assert isinstance(data, dict)
        assert "original_prompt" in data
        assert "sub_prompts" in data
        assert isinstance(data["sub_prompts"], list)


# Make assertions available as pytest fixture
@pytest.fixture
def assert_shardguard():
    """Fixture providing ShardGuard-specific assertions."""
    return ShardGuardAssertions()


# Parametrized test data
@pytest.fixture(
    params=[
        "backup my files",
        "sync data at 3 AM",
        "cleanup logs older than 30 days",
        "restore from backup_2024.tar.gz",
    ]
)
def various_prompts(request):
    """Fixture providing various prompt examples."""
    return request.param


@pytest.fixture(
    params=[
        ("  spaces  ", "spaces"),
        ("tabs\t\there", "tabs here"),
        ("newlines\n\nhere", "newlines here"),
        ("mixed \t\n spaces", "mixed spaces"),
    ]
)
def whitespace_test_cases(request):
    """Fixture providing whitespace normalization test cases."""
    return request.param


@pytest.fixture(
    params=[
        ('<script>alert("xss")</script>', "[REMOVED]"),
        ("javascript:void(0)", "[REMOVED]void(0)"),
        ("data:text/html,<h1>test</h1>", "[REMOVED],<h1>test</h1>"),
    ]
)
def malicious_input_test_cases(request):
    """Fixture providing malicious input test cases."""
    return request.param
