import json
import re
from typing import Protocol

import httpx


class PlanningLLMProtocol(Protocol):
    def generate_plan(self, prompt: str) -> str: ...


class PlanningLLM:
    """LLM implementation using Ollama API."""

    def __init__(
        self, model: str = "llama3.2", base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url
        self.client = httpx.Client(timeout=60.0)

    def generate_plan(self, prompt: str) -> str:
        """Generate a plan using Ollama LLM."""
        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for more consistent JSON output
                        "top_p": 0.9,
                        "num_predict": 2048,  # Max tokens
                    },
                },
            )
            response.raise_for_status()

            result = response.json()
            raw_response = result.get("response", "")

            # Extract JSON from the response (LLMs often add explanatory text)
            json_response = self._extract_json_from_response(raw_response)
            return json_response

        except httpx.RequestError as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.base_url}: {e}"
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama API error {e.response.status_code}: {e.response.text}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from Ollama: {e}")

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from LLM response that might contain extra text."""
        # Try to find JSON block enclosed in curly braces
        json_pattern = r"\{.*\}"
        matches = re.findall(json_pattern, response, re.DOTALL)

        if matches:
            # Return the longest JSON-like match
            json_candidate = max(matches, key=len)

            # Validate that it's actually valid JSON
            try:
                json.loads(json_candidate)
                return json_candidate
            except json.JSONDecodeError:
                pass

        # If no valid JSON found, return the original response
        # and let the calling code handle the validation error
        return response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
