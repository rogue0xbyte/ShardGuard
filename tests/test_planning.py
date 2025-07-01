"""Tests for ShardGuard planning LLM implementations."""

import json

from tests.test_helpers import MockPlanningLLM


class TestMockPlanningLLM:
    """Test cases for MockPlanningLLM."""

    def test_generate_plan_basic(self):
        """Test basic plan generation."""
        llm = MockPlanningLLM()
        prompt = "backup files at 3 AM"

        result = llm.generate_plan(prompt)

        # Should return valid JSON
        plan_data = json.loads(result)
        assert isinstance(plan_data, dict)
        assert "original_prompt" in plan_data
        assert "sub_prompts" in plan_data

    def test_generate_plan_contains_original_prompt(self):
        """Test that generated plan contains the original prompt."""
        llm = MockPlanningLLM()
        prompt = "test prompt with specific content"

        result = llm.generate_plan(prompt)
        plan_data = json.loads(result)

        assert plan_data["original_prompt"] == prompt

    def test_generate_plan_extracts_numbers(self):
        """Test that the mock LLM extracts numbers from the prompt."""
        llm = MockPlanningLLM()
        prompt = "backup at 3 AM on day 15 using port 8080"

        result = llm.generate_plan(prompt)
        plan_data = json.loads(result)

        # Should extract numbers: 3, 15, 8080
        opaque_values = plan_data["sub_prompts"][0]["opaque_values"]
        extracted_numbers = list(opaque_values.values())
        assert "3" in extracted_numbers
        assert "15" in extracted_numbers
        assert "8080" in extracted_numbers

    def test_generate_plan_no_numbers(self):
        """Test plan generation with prompt containing no numbers."""
        llm = MockPlanningLLM()
        prompt = "backup all files to cloud storage"

        result = llm.generate_plan(prompt)
        plan_data = json.loads(result)

        opaque_values = plan_data["sub_prompts"][0]["opaque_values"]
        assert opaque_values == {}

    def test_generate_plan_sub_prompts_structure(self):
        """Test that sub_prompts have the correct structure."""
        llm = MockPlanningLLM()
        prompt = "test with number 42"

        result = llm.generate_plan(prompt)
        plan_data = json.loads(result)

        sub_prompts = plan_data["sub_prompts"]
        assert len(sub_prompts) == 1

        sub_prompt = sub_prompts[0]
        assert "id" in sub_prompt
        assert "content" in sub_prompt
        assert "opaque_values" in sub_prompt
        assert sub_prompt["id"] == 1
        assert sub_prompt["content"] == "Sub-task 1"

    def test_generate_plan_with_multiple_numbers(self):
        """Test extraction of multiple numbers from complex prompt."""
        llm = MockPlanningLLM()
        prompt = "Schedule backup for 2024-12-31 at 23:59:30 using 5 retries"

        result = llm.generate_plan(prompt)
        plan_data = json.loads(result)

        opaque_values = plan_data["sub_prompts"][0]["opaque_values"]
        # Should extract: 2024, 12, 31, 23, 59, 30, 5
        assert len(opaque_values) == 7
        extracted_numbers = list(opaque_values.values())
        assert "2024" in extracted_numbers
        assert "5" in extracted_numbers

    def test_generate_plan_with_empty_prompt(self):
        """Test plan generation with empty prompt."""
        llm = MockPlanningLLM()
        prompt = ""

        result = llm.generate_plan(prompt)
        plan_data = json.loads(result)

        assert plan_data["original_prompt"] == ""
        assert len(plan_data["sub_prompts"]) == 1
        assert plan_data["sub_prompts"][0]["opaque_values"] == {}

    def test_generate_plan_with_special_characters(self):
        """Test plan generation with special characters in prompt."""
        llm = MockPlanningLLM()
        prompt = 'backup "user files" & logs @3AM #priority1'

        result = llm.generate_plan(prompt)
        plan_data = json.loads(result)

        assert plan_data["original_prompt"] == prompt
        opaque_values = plan_data["sub_prompts"][0]["opaque_values"]
        extracted_numbers = list(opaque_values.values())
        assert "3" in extracted_numbers
        assert "1" in extracted_numbers

    def test_generate_plan_returns_string(self):
        """Test that generate_plan always returns a string."""
        llm = MockPlanningLLM()
        prompt = "any prompt"

        result = llm.generate_plan(prompt)

        assert isinstance(result, str)
        # Should be valid JSON string
        json.loads(result)  # Should not raise exception

    def test_generate_plan_consistent_format(self):
        """Test that generated plans have consistent format across calls."""
        llm = MockPlanningLLM()

        for prompt in ["test 1", "another test 2", "final test 3"]:
            result = llm.generate_plan(prompt)
            plan_data = json.loads(result)

            # Check consistent structure
            assert "original_prompt" in plan_data
            assert "sub_prompts" in plan_data
            assert isinstance(plan_data["sub_prompts"], list)
            assert len(plan_data["sub_prompts"]) == 1

            sub_prompt = plan_data["sub_prompts"][0]
            assert sub_prompt["id"] == 1
            assert sub_prompt["content"] == "Sub-task 1"
            assert isinstance(sub_prompt["opaque_values"], dict)

    def test_generate_plan_number_extraction_edge_cases(self):
        """Test number extraction with various edge cases."""
        llm = MockPlanningLLM()

        test_cases = [
            ("floating point 3.14", ["3", "14"]),
            ("negative -5 numbers", ["5"]),
            ("mixed 1a2b3c", ["1", "2", "3"]),
            ("version 1.2.3", ["1", "2", "3"]),
            ("no digits here!", []),
        ]

        for prompt, expected_numbers in test_cases:
            result = llm.generate_plan(prompt)
            plan_data = json.loads(result)
            opaque_values = plan_data["sub_prompts"][0]["opaque_values"]
            extracted_numbers = list(opaque_values.values())

            assert extracted_numbers == expected_numbers, f"Failed for prompt: {prompt}"
