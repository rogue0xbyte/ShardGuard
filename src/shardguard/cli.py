import asyncio

import typer
from rich.console import Console

from shardguard.core.mcp_integration import MCPIntegratedPlanningLLM

app = typer.Typer(help="ShardGuard CLI")
console = Console()

# Global MCP planner instance for connection reuse
_mcp_planner = None


async def get_mcp_planner():
    """Get or create the global MCP planner instance."""
    global _mcp_planner  # noqa: PLW0603
    if _mcp_planner is None:
        console.print("[dim]🔌 Initializing MCP server connections...[/dim]")
        _mcp_planner = MCPIntegratedPlanningLLM()

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


@app.command()
def list_tools():
    """List all available MCP tools."""

    async def _list_tools():
        planner = await get_mcp_planner()
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
    model: str = typer.Option("llama3.2", "--model", help="Ollama model to use"),
    ollama_url: str = typer.Option(
        "http://localhost:11434", "--ollama-url", help="Ollama base URL"
    ),
):
    """Generate a safe execution plan for a user prompt."""

    async def _plan():
        try:
            # Use MCP-integrated planner by default
            planner = await get_mcp_planner()
            console.print(f"[dim]Using Ollama model: {model} at {ollama_url}[/dim]")

            # Show available tools
            tools_description = await planner.get_available_tools_description()
            if "No MCP tools available." not in tools_description:
                # Count tools by parsing the description
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
                    f"[dim]Available tools: {len(tool_lines)} tools from {len(server_lines)} MCP servers[/dim]"
                )

            # Create async-aware coordination service
            from shardguard.core.coordination import AsyncCoordinationService

            coord = AsyncCoordinationService(planner)
            plan_obj = await coord.handle_prompt(prompt)
            typer.echo(plan_obj.model_dump_json(indent=2))

        except ConnectionError as e:
            console.print(f"[bold red]Connection Error:[/bold red] {e}")
            console.print("Make sure Ollama is running: `ollama serve`")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(1)

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
            console.print("[dim]We support sub-commands: list-tools, plan[/dim]")

        asyncio.run(_init())


if __name__ == "__main__":
    app()
