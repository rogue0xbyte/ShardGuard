from typing import Any, List, Optional, Mapping

from langchain.llms.base import LLM
from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain.agents import AgentType

import json

def GenericExecutionLLMWrapper(generic_llm: "GenericExecutionLLM") -> LLM:
    """Return a LangChain-compatible LLM from a GenericExecutionLLM."""

    class _Wrapper(LLM):
        _generic_llm = generic_llm

        @property
        def _llm_type(self) -> str:
            return "generic_execution_llm"

        def _call(self, *args, **kwargs):
            raise NotImplementedError("Use async _acall() instead")

        async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
            try:
                resp = await self._generic_llm.llm_provider.generate_response(prompt)
                # Ensure always a string
                if not isinstance(resp, str):
                    resp = json.dumps(resp)
                return resp
            except Exception as e:
                return f"[ERROR]: {e}"

        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            return {
                "provider_type": self._generic_llm.provider_type,
                "model": self._generic_llm.model,
            }

    return _Wrapper()


def ToolsWrapper(tools_list: List[str]) -> List[Tool]:
    """
    Convert a list of tool dicts to LangChain Tool objects.
    """
    langchain_tools = []

    for tool_entry in tools_list:
        if "." in tool_entry:
            server_name, tool_name = tool_entry.split(".", 1)
        else:
            server_name, tool_name = "unknown-server", tool_entry

        # Create a LangChain Tool
        langchain_tools.append(
            Tool(
                name=tool_name,
                description= "",
                func=lambda *args, **kwargs: {
                    "server": server_name,
                    "tool": tool_name,
                    "args": kwargs,
                },
            )
        )

    return langchain_tools


EXEC_SYSTEM_PROMPT = """
You are the Execution LLM for ShardGuard.

Rules:
- Only use tools provided in the suggested_tools list.
- Return valid JSON for args.
- Do not invent tools or servers.
- Validate output internally before returning.
"""

def log_tool_call(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        print(f"[Tool Called] {func.__name__} with args={args}, kwargs={kwargs}")
        result = await func(*args, **kwargs) if callable(func) else None
        print(f"[Tool Result] {func.__name__} => {result}")
        return result
    return wrapper

def make_langchain_tools(suggested_tools: List[Tool]):
    """
    Takes a list of Tool objects and returns them for the agent.
    Each Tool must have a name and a callable function.
    """
    return suggested_tools


def make_execution_agent(llm, suggested_tools: List[Tool]):
    """
    Creates a self-contained LangChain agent that:
    - Uses the LLM
    - Has the suggested tools
    - Chooses which tool to run
    - Returns validated JSON
    """
    tools = make_langchain_tools(suggested_tools)
    
    agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )
    print("[-] Using LangChain Agent")

    return agent