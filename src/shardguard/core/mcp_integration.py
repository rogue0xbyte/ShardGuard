"""MCP client integration for ShardGuard using the official Python SDK."""

import asyncio
import json
import logging
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for communicating with MCP servers."""

    def __init__(self):
        """Initialize the MCP client."""
        import os

        # Get the absolute path to the servers directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        servers_dir = os.path.join(os.path.dirname(current_dir), "mcp_servers")

        self.server_configs = {
            "file-operations": {
                "command": sys.executable,
                "args": [os.path.join(servers_dir, "file_server.py")],
                "description": "File operations with security controls",
            },
            "email-operations": {
                "command": sys.executable,
                "args": [os.path.join(servers_dir, "email_server.py")],
                "description": "Email operations with privacy controls",
            },
            "database-operations": {
                "command": sys.executable,
                "args": [os.path.join(servers_dir, "database_server.py")],
                "description": "Database operations with security controls",
            },
            "web-operations": {
                "command": sys.executable,
                "args": [os.path.join(servers_dir, "web_server.py")],
                "description": "Web operations with security controls",
            },
        }

    async def _execute_with_server(self, server_name: str, operation):
        """Execute an operation with a server connection."""
        if server_name not in self.server_configs:
            return None

        try:
            config = self.server_configs[server_name]
            server_params = StdioServerParameters(
                command=config["command"], args=config["args"]
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await operation(session)
        except Exception as e:
            logger.debug(
                "Error connecting to %s: %s: %s", server_name, type(e).__name__, e
            )
            if hasattr(e, "__cause__") and e.__cause__:
                logger.debug(
                    "  Caused by: %s: %s", type(e.__cause__).__name__, e.__cause__
                )
            if hasattr(e, "exceptions"):
                logger.debug("  Sub-exceptions: %d", len(e.exceptions))
                for i, sub_e in enumerate(e.exceptions):
                    logger.debug("    %d: %s: %s", i, type(sub_e).__name__, sub_e)
            return None

    async def list_tools(self, server_name: str | None = None) -> dict[str, list[Any]]:
        """List available tools from one or all servers."""
        tools_by_server = {}

        servers_to_check = (
            [server_name] if server_name else list(self.server_configs.keys())
        )

        for server in servers_to_check:

            async def get_tools(session):
                tools_response = await session.list_tools()
                return tools_response.tools

            tools = await self._execute_with_server(server, get_tools)
            tools_by_server[server] = tools or []

        return tools_by_server

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> str | None:
        """Call a tool on a specific server."""

        async def call_tool_op(session):
            result = await session.call_tool(tool_name, arguments)

            # Extract text content from the response
            if result.content:
                return "\n".join(
                    item.text for item in result.content if hasattr(item, "text")
                )
            return "Tool executed successfully (no content returned)"

        return await self._execute_with_server(server_name, call_tool_op)

    async def get_tools_description(self) -> str:
        """Get a formatted description of all available tools."""
        tools_by_server = await self.list_tools()

        if not any(tools_by_server.values()):
            return "No MCP tools available."

        description = "Available MCP Tools:\n\n"

        for server_name, tools in tools_by_server.items():
            if tools:
                config = self.server_configs.get(server_name, {})
                server_desc = config.get("description", "MCP Server")
                description += f"Server: {server_name} - {server_desc}\n"

                for tool in tools:
                    description += f"  • {tool.name}: {tool.description}\n"

                    # Add input schema details
                    if hasattr(tool, "inputSchema") and tool.inputSchema:
                        schema = tool.inputSchema
                        if isinstance(schema, dict) and "properties" in schema:
                            required = schema.get("required", [])
                            for prop_name, prop_info in schema["properties"].items():
                                req_marker = (
                                    " (required)" if prop_name in required else ""
                                )
                                prop_desc = prop_info.get(
                                    "description", "No description"
                                )
                                description += (
                                    f"    - {prop_name}: {prop_desc}{req_marker}\n"
                                )

                description += "\n"

        description += "When suggesting tools for tasks, include the tool names in your sub-task 'suggested_tools' field."
        return description

    def get_available_servers(self) -> dict[str, str]:
        """Get list of available servers and their descriptions."""
        return {
            name: config["description"] for name, config in self.server_configs.items()
        }


class MCPIntegratedPlanningLLM:
    """Enhanced PlanningLLM with MCP integration."""

    def __init__(
        self, model: str = "llama3.2", base_url: str = "http://localhost:11434"
    ):
        """Initialize with MCP client integration."""
        self.model = model
        self.base_url = base_url
        self.mcp_client = MCPClient()

        # Import httpx here to avoid import errors if not available
        try:
            import httpx

            self.client = httpx.Client(timeout=60.0)
        except ImportError:
            self.client = None

    async def get_available_tools_description(self) -> str:
        """Get formatted description of all available MCP tools."""
        return await self.mcp_client.get_tools_description()

    def get_available_tools_description_sync(self) -> str:
        """Get formatted description of all available MCP tools (sync version)."""
        try:
            loop = asyncio.get_running_loop()
            # If in async context, return mock description
            return "Available MCP Tools:\n\nServer: file-operations - File operations with security controls\nServer: email-operations - Email operations with privacy controls\nServer: database-operations - Database operations with security controls\nServer: web-operations - Web operations with security controls\n\nWhen suggesting tools for tasks, include the tool names in your sub-task 'suggested_tools' field."
        except RuntimeError:
            # No running loop, we can run async code
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Call the async method directly through the async machinery
                return loop.run_until_complete(self.mcp_client.get_tools_description())
            finally:
                loop.close()

    async def list_available_tools(self) -> dict[str, list[Any]]:
        """List all available tools from all servers."""
        return await self.mcp_client.list_tools()

    async def call_mcp_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> str | None:
        """Call an MCP tool directly."""
        return await self.mcp_client.call_tool(server_name, tool_name, arguments)

    async def generate_plan(self, prompt: str) -> str:
        """Generate a plan using Ollama LLM, including MCP tool descriptions."""
        # Call the async version directly to avoid method override issues
        tools_description = await self.mcp_client.get_tools_description()

        # Create enhanced prompt with tools
        if tools_description != "No MCP tools available.":
            enhanced_prompt = f"{prompt}\n\n{tools_description}"
        else:
            enhanced_prompt = prompt

        # Log the full prompt being sent to the model for debugging
        logger.debug("Full prompt sent to model:\n%s", enhanced_prompt)

        if not self.client:
            # Return mock response if httpx not available
            return json.dumps(
                {
                    "original_prompt": prompt,
                    "sub_prompts": [
                        {
                            "id": 1,
                            "content": "This is a mock response - httpx not available",
                            "opaque_values": {},
                            "suggested_tools": ["file-operations.read_file"],
                        }
                    ],
                }
            )

        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": enhanced_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 2048,
                    },
                },
            )
            response.raise_for_status()

            result = response.json()
            raw_response = result.get("response", "")

            # Extract JSON from the response
            json_response = self._extract_json_from_response(raw_response)
            return json_response

        except Exception as e:
            # Return mock response on error
            return json.dumps(
                {
                    "original_prompt": prompt,
                    "sub_prompts": [
                        {
                            "id": 1,
                            "content": f"Error occurred: {e!s}",
                            "opaque_values": {},
                            "suggested_tools": [],
                        }
                    ],
                }
            )

    def generate_plan_sync(self, prompt: str) -> str:
        """Generate a plan using Ollama LLM (sync version)."""
        tools_description = self.get_available_tools_description_sync()

        # Create enhanced prompt with tools
        if "No MCP tools available" not in tools_description:
            enhanced_prompt = f"{prompt}\n\n{tools_description}"
        else:
            enhanced_prompt = prompt

        # Log the full prompt being sent to the model for debugging
        logger.debug("Full prompt sent to model (sync):\n%s", enhanced_prompt)

        if not self.client:
            # Return mock response if httpx not available
            return json.dumps(
                {
                    "original_prompt": prompt,
                    "sub_prompts": [
                        {
                            "id": 1,
                            "content": "This is a mock response - httpx not available",
                            "opaque_values": {},
                            "suggested_tools": ["file-operations.read_file"],
                        }
                    ],
                }
            )

        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": enhanced_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 2048,
                    },
                },
            )
            response.raise_for_status()

            result = response.json()
            raw_response = result.get("response", "")

            # Extract JSON from the response
            json_response = self._extract_json_from_response(raw_response)
            return json_response

        except Exception as e:
            # Return mock response on error
            return json.dumps(
                {
                    "original_prompt": prompt,
                    "sub_prompts": [
                        {
                            "id": 1,
                            "content": f"Error occurred: {e!s}",
                            "opaque_values": {},
                            "suggested_tools": [],
                        }
                    ],
                }
            )

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from LLM response that might contain extra text."""
        import re

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
        return response

    async def close(self):
        """Close connections."""
        # No persistent connections to close in this implementation
        if self.client:
            self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience function to create the integrated planning LLM
def create_mcp_planning_llm(
    model: str = "llama3.2", base_url: str = "http://localhost:11434"
) -> MCPIntegratedPlanningLLM:
    """Create an MCP-integrated planning LLM."""
    return MCPIntegratedPlanningLLM(model=model, base_url=base_url)
