from typing import Protocol

from .mcp_integration import MCPIntegratedPlanningLLM


class PlanningLLMProtocol(Protocol):
    def generate_plan(self, prompt: str) -> str: ...


class PlanningLLM:
    """Synchronous wrapper for MCPIntegratedPlanningLLM that implements PlanningLLMProtocol."""

    def __init__(
        self, model: str = "llama3.2", base_url: str = "http://localhost:11434"
    ):
        self._impl = MCPIntegratedPlanningLLM(model=model, base_url=base_url)
        # Expose attributes for compatibility with tests
        self.model = model
        self.base_url = base_url

    def generate_plan(self, prompt: str) -> str:
        """Generate a plan using the sync version of the underlying implementation."""
        return self._impl.generate_plan_sync(prompt)

    def get_available_tools_description(self) -> str:
        """Get formatted description of all available MCP tools."""
        return self._impl.get_available_tools_description_sync()

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from LLM response that might contain extra text."""
        result = self._impl._extract_json_from_response(response)
        # If the original response was returned (not a valid JSON), return a fallback
        if result == response:
            import json

            return json.dumps(
                {
                    "original_prompt": response,
                    "sub_prompts": [
                        {
                            "id": 1,
                            "content": f"Failed to parse JSON from response: {response[:100]}...",
                            "opaque_values": {},
                            "suggested_tools": [],
                        }
                    ],
                }
            )
        return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self._impl, "client") and self._impl.client:
            self._impl.client.close()

    async def close(self):
        """Close any open connections."""
        await self._impl.close()
