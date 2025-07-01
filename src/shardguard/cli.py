import typer

from shardguard.core.coordination import CoordinationService
from shardguard.core.planning import PlanningLLM

app = typer.Typer(help="ShardGuard CLI")


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
            typer.echo(
                f"[dim]Using Ollama model: {model} at {ollama_url}[/dim]", err=True
            )
        else:
            # Import MockPlanningLLM only when needed for development/testing
            from shardguard.dev_utils import MockPlanningLLM

            planner = MockPlanningLLM()
            typer.echo("[dim]Using MockPlanningLLM for development[/dim]", err=True)

        coord = CoordinationService(planner)
        plan_obj = coord.handle_prompt(prompt)
        typer.echo(plan_obj.model_dump_json(indent=2))

    except ConnectionError as e:
        typer.echo(f"[bold red]Connection Error:[/bold red] {e}", err=True)
        typer.echo("Make sure Ollama is running: `ollama serve`", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"[bold red]Error:[/bold red] {e}", err=True)
        raise typer.Exit(1)


# Add a callback to ensure we always have subcommands
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        typer.echo("Welcome to ShardGuard! Use --help to see available commands.")
        typer.echo("We support only sub-commands at this moment")


if __name__ == "__main__":
    app()
