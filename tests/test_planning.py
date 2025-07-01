"""Tests for ShardGuard planning LLM implementations."""

import json

from tests.test_helpers import MockPlanningLLM


class TestMockPlanningLLM:
    """Test cases for MockPlanningLLM - should be simple and not contain business logic."""

    def test_generate_plan_basic(self):
        """Test basic plan generation returns valid JSON."""
        llm = MockPlanningLLM()
        prompt = "backup files at 3 AM"

        result = llm.generate_plan(prompt)

        # Should return valid JSON
        plan_data = json.loads(result)
        assert isinstance(plan_data, dict)
        assert "original_prompt" in plan_data
        assert "sub_prompts" in plan_data

    def test_generate_plan_returns_consistent_structure(self):
        """Test that MockPlanningLLM always returns consistent structure."""
        llm = MockPlanningLLM()

        for prompt in ["test 1", "another test 2", "final test 3"]:
            result = llm.generate_plan(prompt)
            plan_data = json.loads(result)

            # Check consistent structure
            assert "original_prompt" in plan_data
            assert "sub_prompts" in plan_data
            assert isinstance(plan_data["sub_prompts"], list)
            assert len(plan_data["sub_prompts"]) >= 1

            sub_prompt = plan_data["sub_prompts"][0]
            assert "id" in sub_prompt
            assert "content" in sub_prompt
            assert "opaque_values" in sub_prompt

    def test_generate_plan_returns_string(self):
        """Test that generate_plan always returns a string."""
        llm = MockPlanningLLM()
        prompt = "any prompt"

        result = llm.generate_plan(prompt)

        assert isinstance(result, str)
        # Should be valid JSON string
        json.loads(result)  # Should not raise exception

    def test_generate_plan_handles_various_inputs(self):
        """Test that MockPlanningLLM handles various input types without crashing."""
        llm = MockPlanningLLM()

        test_cases = [
            "",
            "simple prompt",
            'complex "quoted" prompt with special chars @#$%',
            "very " * 100 + "long prompt",
            "unicode café test 🚀",
        ]

        for prompt in test_cases:
            result = llm.generate_plan(prompt)
            plan_data = json.loads(result)
            assert isinstance(plan_data, dict)
            assert "original_prompt" in plan_data
            assert "sub_prompts" in plan_data
