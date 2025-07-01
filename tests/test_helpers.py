"""Test helper classes and utilities."""

import json


class MockPlanningLLM:
    """Simple mock LLM that returns valid JSON without business logic."""

    def generate_plan(self, prompt: str) -> str:
        """Return a simple, valid JSON response for testing."""
        # Just return a minimal valid plan structure
        # The real business logic should be in the actual LLM or coordination service
        plan = {
            "original_prompt": "test prompt",
            "sub_prompts": [{"id": 1, "content": "Sub-task 1", "opaque_values": {}}],
        }
        return json.dumps(plan)
