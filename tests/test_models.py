"""Tests for ShardGuard data models."""

import pytest
from pydantic import ValidationError

from shardguard.core.models import Plan, SubPrompt


class TestSubPrompt:
    """Test cases for SubPrompt model."""

    def test_create_subprompt_with_required_fields(self):
        """Test creating a SubPrompt with only required fields."""
        subprompt = SubPrompt(id=1, content="Test content")

        assert subprompt.id == 1
        assert subprompt.content == "Test content"
        assert subprompt.opaque_values == {}

    def test_create_subprompt_with_opaque_values(self):
        """Test creating a SubPrompt with opaque values."""
        opaque_values = {"[VALUE_1]": "value1", "[SECRET]": "secret123"}
        subprompt = SubPrompt(
            id=2, content="Content with secrets", opaque_values=opaque_values
        )

        assert subprompt.id == 2
        assert subprompt.content == "Content with secrets"
        assert subprompt.opaque_values == opaque_values

    def test_subprompt_validation_missing_id(self):
        """Test that SubPrompt validation fails when id is missing."""
        with pytest.raises(ValidationError) as exc_info:
            SubPrompt(content="Test content")

        assert "id" in str(exc_info.value)

    def test_subprompt_validation_missing_content(self):
        """Test that SubPrompt validation fails when content is missing."""
        with pytest.raises(ValidationError) as exc_info:
            SubPrompt(id=1)

        assert "content" in str(exc_info.value)

    def test_subprompt_validation_invalid_id_type(self):
        """Test that SubPrompt validation fails for invalid id type."""
        with pytest.raises(ValidationError) as exc_info:
            SubPrompt(id="not_an_int", content="Test content")

        assert "id" in str(exc_info.value)

    def test_subprompt_validation_invalid_content_type(self):
        """Test that SubPrompt validation fails for invalid content type."""
        with pytest.raises(ValidationError) as exc_info:
            SubPrompt(id=1, content=123)

        assert "content" in str(exc_info.value)

    def test_subprompt_validation_invalid_opaque_values_type(self):
        """Test that SubPrompt validation fails for invalid opaque_values type."""
        with pytest.raises(ValidationError) as exc_info:
            SubPrompt(id=1, content="Test", opaque_values="not_a_list")

        assert "opaque_values" in str(exc_info.value)


class TestPlan:
    """Test cases for Plan model."""

    def test_create_plan_with_required_fields(self):
        """Test creating a Plan with required fields."""
        subprompts = [
            SubPrompt(id=1, content="First task"),
            SubPrompt(
                id=2, content="Second task", opaque_values={"[SECRET]": "secret"}
            ),
        ]
        plan = Plan(original_prompt="Original user request", sub_prompts=subprompts)

        assert plan.original_prompt == "Original user request"
        assert len(plan.sub_prompts) == 2
        assert plan.sub_prompts[0].id == 1
        assert plan.sub_prompts[1].opaque_values == {"[SECRET]": "secret"}

    def test_create_plan_empty_subprompts(self):
        """Test creating a Plan with empty sub_prompts list."""
        plan = Plan(original_prompt="Simple request", sub_prompts=[])

        assert plan.original_prompt == "Simple request"
        assert plan.sub_prompts == []

    def test_plan_validation_missing_original_prompt(self):
        """Test that Plan validation fails when original_prompt is missing."""
        with pytest.raises(ValidationError) as exc_info:
            Plan(sub_prompts=[])

        assert "original_prompt" in str(exc_info.value)

    def test_plan_validation_missing_sub_prompts(self):
        """Test that Plan validation fails when sub_prompts is missing."""
        with pytest.raises(ValidationError) as exc_info:
            Plan(original_prompt="Test prompt")

        assert "sub_prompts" in str(exc_info.value)

    def test_plan_validation_invalid_sub_prompts_type(self):
        """Test that Plan validation fails for invalid sub_prompts type."""
        with pytest.raises(ValidationError) as exc_info:
            Plan(original_prompt="Test", sub_prompts="not_a_list")

        assert "sub_prompts" in str(exc_info.value)

    def test_plan_json_serialization(self):
        """Test that Plan can be serialized to JSON."""
        plan = Plan(
            original_prompt="Test prompt",
            sub_prompts=[
                SubPrompt(id=1, content="Task 1", opaque_values={"[VALUE_1]": "val1"}),
                SubPrompt(id=2, content="Task 2"),
            ],
        )

        json_str = plan.model_dump_json()
        assert "Test prompt" in json_str
        assert "Task 1" in json_str
        assert "val1" in json_str

    def test_plan_json_deserialization(self):
        """Test that Plan can be deserialized from JSON."""
        json_data = """
        {
            "original_prompt": "Test prompt",
            "sub_prompts": [
                {
                    "id": 1,
                    "content": "Task 1",
                    "opaque_values": {"[VALUE_1]": "val1"}
                },
                {
                    "id": 2,
                    "content": "Task 2",
                    "opaque_values": {}
                }
            ]
        }
        """

        plan = Plan.model_validate_json(json_data)
        assert plan.original_prompt == "Test prompt"
        assert len(plan.sub_prompts) == 2
        assert plan.sub_prompts[0].opaque_values == {"[VALUE_1]": "val1"}
        assert plan.sub_prompts[1].opaque_values == {}
