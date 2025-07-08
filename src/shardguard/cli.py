import asyncio
import os

import typer
from rich.console import Console

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # python-dotenv not available

from shardguard.core.mcp_integration import MCPIntegratedPlanningLLM

app = typer.Typer(help="ShardGuard CLI")
console = Console()

# Global MCP planner instance for connection reuse
_mcp_planner = None


async def get_mcp_planner(
    provider_type: str = "ollama",
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    api_key: str | None = None,
):
    """Get or create the global MCP planner instance."""
    global _mcp_planner  # noqa: PLW0603
    if _mcp_planner is None:
        console.print(
            f"[dim]🔌 Initializing MCP server connections with {provider_type} provider...[/dim]"
        )
        _mcp_planner = MCPIntegratedPlanningLLM(
            provider_type=provider_type, model=model, base_url=base_url, api_key=api_key
        )

        # Test connectivity by checking tools
        try:
            tools_description = await _mcp_planner.get_available_tools_description()
            if "No MCP tools available." not in tools_description:
                # Count tools
                tool_lines = [
                    line
                    for line in tools_description.split("\n")
                    if line.strip().startswith("•")
                ]
                server_lines = [
                    line
                    for line in tools_description.split("\n")
                    if line.strip().startswith("Server:")
                ]
                console.print(
                    f"[dim]✅ Connected to {len(server_lines)} MCP servers with {len(tool_lines)} tools[/dim]"
                )
            else:
                console.print("[dim]⚠️  No MCP tools available[/dim]")
        except Exception as e:
            console.print(f"[dim]⚠️  MCP connection issue: {e}[/dim]")

    return _mcp_planner


def _validate_gemini_api_key(provider: str, api_key: str | None) -> None:
    """Validate Gemini API key if required."""
    if provider == "gemini" and not api_key:
        console.print(
            "[bold red]Error:[/bold red] Gemini API key required. Set GEMINI_API_KEY env var or use --gemini-api-key"
        )
        raise typer.Exit(1)


def _get_model_for_provider(provider: str, model: str | None) -> str:
    """Get the model name for the provider, auto-detecting if not specified."""
    if model is not None:
        return model

    if provider == "gemini":
        return "gemini-2.0-flash-exp"
    return "llama3.2"


def _print_provider_info(provider: str, model: str, ollama_url: str) -> None:
    """Print information about the selected provider and model."""
    if provider == "ollama":
        console.print(f"[dim]Using Ollama model: {model} at {ollama_url}[/dim]")
    else:
        console.print(f"[dim]Using Gemini model: {model}[/dim]")


def _print_tools_info(tools_description: str) -> None:
    """Print information about available tools."""
    if "No MCP tools available." in tools_description:
        return

    tool_lines = [
        line for line in tools_description.split("\n") if line.strip().startswith("•")
    ]
    server_lines = [
        line
        for line in tools_description.split("\n")
        if line.strip().startswith("Server:")
    ]
    console.print(
        f"[dim]Available tools: {len(tool_lines)} tools from {len(server_lines)} MCP servers[/dim]"
    )


def _handle_errors(e: Exception, provider: str) -> None:
    """Handle and display errors appropriately."""
    if isinstance(e, ConnectionError):
        console.print(f"[bold red]Connection Error:[/bold red] {e}")
        if provider == "ollama":
            console.print("Make sure Ollama is running: `ollama serve`")
        else:
            console.print("Check your Gemini API key and internet connection")
    else:
        console.print(f"[bold red]Error:[/bold red] {e}")
    raise typer.Exit(1)


@app.command()
def list_tools(
    provider: str = typer.Option(
        "ollama", "--provider", help="LLM provider (ollama or gemini)"
    ),
    model: str = typer.Option(
        None,
        "--model",
        help="Model to use. Ollama: llama3.2, llama3.1, etc. Gemini: gemini-2.0-flash-exp, gemini-1.5-pro, etc. Auto-detected if not specified.",
    ),
    ollama_url: str = typer.Option(
        "http://localhost:11434", "--ollama-url", help="Ollama base URL"
    ),
    gemini_api_key: str = typer.Option(
        None, "--gemini-api-key", help="Gemini API key (or set GEMINI_API_KEY env var)"
    ),
):
    """List all available MCP tools."""

    async def _list_tools():
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        _validate_gemini_api_key(provider, api_key)

        # Auto-detect model based on provider if not specified
        detected_model = _get_model_for_provider(provider, model)

        planner = await get_mcp_planner(
            provider_type=provider,
            model=detected_model,
            base_url=ollama_url,
            api_key=api_key,
        )
        try:
            tools_description = await planner.get_available_tools_description()
            console.print("[bold blue]Available MCP Tools:[/bold blue]")
            console.print(tools_description)
        finally:
            await planner.close()

    asyncio.run(_list_tools())


@app.command()
def plan(
    prompt: str,
    provider: str = typer.Option(
        "ollama", "--provider", help="LLM provider (ollama or gemini)"
    ),
    model: str = typer.Option(
        None,
        "--model",
        help="Model to use. Ollama: llama3.2, llama3.1, etc. Gemini: gemini-2.0-flash-exp, gemini-1.5-pro, etc. Auto-detected if not specified.",
    ),
    ollama_url: str = typer.Option(
        "http://localhost:11434", "--ollama-url", help="Ollama base URL"
    ),
    gemini_api_key: str = typer.Option(
        None, "--gemini-api-key", help="Gemini API key (or set GEMINI_API_KEY env var)"
    ),
):
    """Generate a safe execution plan for a user prompt."""

    async def _plan():
        try:
            api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

            _validate_gemini_api_key(provider, api_key)

            # Auto-detect model based on provider if not specified
            detected_model = _get_model_for_provider(provider, model)

            # Use MCP-integrated planner with specified provider
            planner = await get_mcp_planner(
                provider_type=provider,
                model=detected_model,
                base_url=ollama_url,
                api_key=api_key,
            )

            _print_provider_info(provider, detected_model, ollama_url)

            # Show available tools
            tools_description = await planner.get_available_tools_description()
            _print_tools_info(tools_description)

            # Create async-aware coordination service
            from shardguard.core.coordination import AsyncCoordinationService

            coord = AsyncCoordinationService(planner)
            plan_obj = await coord.handle_prompt(prompt)
            typer.echo(plan_obj.model_dump_json(indent=2))

        except Exception as e:
            _handle_errors(e, provider)

    asyncio.run(_plan())


# Add a callback to ensure we always have subcommands
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        # Initialize MCP connections when CLI starts
        async def _init():
            console.print("🛡️  [bold blue]Welcome to ShardGuard![/bold blue]")
            await get_mcp_planner()
            console.print("\n[dim]Use --help to see available commands.[/dim]")
            console.print("[dim]Available commands: list-tools, plan[/dim]")
            console.print("[dim]Supported providers: ollama (default), gemini[/dim]")
            console.print(
                "[dim]For Gemini: Set GEMINI_API_KEY environment variable[/dim]"
            )

        asyncio.run(_init())


if __name__ == "__main__":
    app()
