from rich.console import Console

from shardguard.core.models import Plan
from shardguard.core.planning import PlanningLLMProtocol
from shardguard.core.prompts import PLANNING_PROMPT
from shardguard.core.sanitization import InputSanitizer


class CoordinationService:
    """Receives raw user prompt, performs safety injection, calls PlanningLLM."""

    def __init__(self, planner: PlanningLLMProtocol):
        self.planner = planner
        self.console = Console()
        self.sanitizer = InputSanitizer(self.console)

    def handle_prompt(self, user_input: str) -> Plan:
        sanitization_result = self.sanitizer.sanitize(user_input)
        formatted_prompt = self._format_prompt(sanitization_result.sanitized_input)
        plan_json = self.planner.generate_plan(formatted_prompt)
        return Plan.model_validate_json(plan_json)

    def _format_prompt(self, user_input: str) -> str:
        """Format the user input using the planning prompt template."""
        return PLANNING_PROMPT.format(user_prompt=user_input)
