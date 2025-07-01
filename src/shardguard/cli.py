import typer
from rich.console import Console

from shardguard.core.coordination import CoordinationService
from shardguard.core.planning import PlanningLLM

app = typer.Typer(help="ShardGuard CLI")
console = Console()


@app.command()
def plan(
    prompt: str,
    use_ollama: bool = typer.Option(
        False, "--ollama", help="Use Ollama instead of mock LLM"
    ),
    model: str = typer.Option("llama3.2", "--model", help="Ollama model to use"),
    ollama_url: str = typer.Option(
        "http://localhost:11434", "--ollama-url", help="Ollama base URL"
    ),
):
    """Generate a safe execution plan for a user prompt."""
    try:
        if use_ollama:
            planner = PlanningLLM(model=model, base_url=ollama_url)
            console.print(f"[dim]Using Ollama model: {model} at {ollama_url}[/dim]")
        else:
            # Import MockPlanningLLM only when needed for development/testing
            from shardguard.dev_utils import MockPlanningLLM

            planner = MockPlanningLLM()
            console.print("[dim]Using MockPlanningLLM for development[/dim]")

        coord = CoordinationService(planner)
        plan_obj = coord.handle_prompt(prompt)
        typer.echo(plan_obj.model_dump_json(indent=2))

    except ConnectionError as e:
        console.print(f"[bold red]Connection Error:[/bold red] {e}")
        console.print("Make sure Ollama is running: `ollama serve`")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


# Add a callback to ensure we always have subcommands
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        typer.echo("Welcome to ShardGuard! Use --help to see available commands.")
        typer.echo("We support only sub-commands at this moment")


if __name__ == "__main__":
    app()
