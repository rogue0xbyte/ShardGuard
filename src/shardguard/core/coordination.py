import re

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from shardguard.core.models import Plan
from shardguard.core.planning import PlanningLLMProtocol
from shardguard.core.prompts import PLANNING_PROMPT


class CoordinationService:
    """Receives raw user prompt, performs safety injection, calls PlanningLLM."""

    def __init__(self, planner: PlanningLLMProtocol):
        self.planner = planner
        self.console = Console()

    def handle_prompt(self, user_input: str) -> Plan:
        sanitized_input = self._sanitize_input(user_input)
        formatted_prompt = self._format_prompt(sanitized_input)
        plan_json = self.planner.generate_plan(formatted_prompt)
        return Plan.model_validate_json(plan_json)

    def _sanitize_input(self, user_input: str) -> str:
        """Basic sanitization without removing important data."""
        self.console.print("\n[bold blue]🔍 Input Sanitization Process[/bold blue]")

        # Show original input
        original_panel = Panel(
            user_input[:200] + ("..." if len(user_input) > 200 else ""),
            title="[bold]Original Input[/bold]",
            border_style="dim",
        )
        self.console.print(original_panel)

        if not user_input or not user_input.strip():
            self.console.print(
                "[bold red]❌ Error: User input cannot be empty[/bold red]"
            )
            raise ValueError("User input cannot be empty")

        changes_made = []
        original_length = len(user_input)

        # Normalize whitespace and line endings
        sanitized = re.sub(r"\s+", " ", user_input.strip())
        if sanitized != user_input.strip():
            changes_made.append("✓ Normalized whitespace and line endings")

        # Remove potentially dangerous control characters but preserve important ones
        before_control_removal = sanitized
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", sanitized)
        if sanitized != before_control_removal:
            changes_made.append("✓ Removed dangerous control characters")

        # Basic length check to prevent extremely long inputs
        if len(sanitized) > 10000:
            sanitized = sanitized[:10000] + "... [truncated]"
            changes_made.append("✓ Truncated input to 10,000 characters")

        # Remove obvious injection attempts while preserving legitimate content
        # TODO: Should we add more ? Should this be extracted out ?
        patterns_to_clean = [
            (r"<script[^>]*>.*?</script>", "Script tags"),
            (r"javascript:", "JavaScript URLs"),
            (r"data:text/html", "HTML data URLs"),
        ]

        for pattern, description in patterns_to_clean:
            before_pattern_removal = sanitized
            sanitized = re.sub(
                pattern, "[REMOVED]", sanitized, flags=re.IGNORECASE | re.DOTALL
            )
            if sanitized != before_pattern_removal:
                changes_made.append(f"✓ Removed {description}")

        # Show sanitization results
        if changes_made:
            changes_text = Text()
            for change in changes_made:
                changes_text.append(change + "\n", style="green")

            changes_panel = Panel(
                changes_text,
                title="[bold green]Sanitization Changes[/bold green]",
                border_style="green",
            )
            self.console.print(changes_panel)
        else:
            self.console.print(
                "[green]✓ No sanitization needed - input is clean[/green]"
            )

        # Show final sanitized input if different
        if sanitized != user_input:
            sanitized_panel = Panel(
                sanitized[:200] + ("..." if len(sanitized) > 200 else ""),
                title="[bold]Sanitized Input[/bold]",
                border_style="green",
            )
            self.console.print(sanitized_panel)

        # Show length comparison
        final_length = len(sanitized)
        if final_length != original_length:
            self.console.print(
                f"[dim]Length: {original_length} → {final_length} characters[/dim]"
            )

        return sanitized

    def _format_prompt(self, user_input: str) -> str:
        """Format the user input using the planning prompt template."""
        return PLANNING_PROMPT.format(user_prompt=user_input)
