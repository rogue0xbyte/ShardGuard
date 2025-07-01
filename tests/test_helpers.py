"""Test helper classes and utilities."""

import json
import re


class MockPlanningLLM:
    """Rule-based fake for tests / offline dev."""

    def generate_plan(self, prompt: str) -> str:
        """Generate a mock plan for testing purposes."""
        # Extract numbers from prompt for testing
        extracted = re.findall(r"\d+", prompt)

        # Create opaque_values dict for extracted numbers
        opaque_values = {}
        for i, num in enumerate(extracted, 1):
            opaque_values[f"[VALUE_{i}]"] = num

        plan = {
            "original_prompt": prompt,
            "sub_prompts": [
                {"id": 1, "content": "Sub-task 1", "opaque_values": opaque_values}
            ],
        }
        return json.dumps(plan)
