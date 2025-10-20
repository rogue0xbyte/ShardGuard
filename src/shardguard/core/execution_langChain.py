"""
execution.py rewritten to support langChain implementations
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

from langchain.tools import BaseTool
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

@dataclass
class ToolCall:
    """
    Represents a single tool call to a given server/tool combo.
    Args are not included since the LLM executor is considered untrusted.
    """
    server: str
    tool: str

@dataclass
class LLMStepResponse:
    """
    Wraps one or more ToolCalls that the LLM has suggested for this step.
    """
    tool_calls: List[ToolCall] = field(default_factory=list)


class LangchainStepExecutor(Runnable):
    """
    LangChain-compatible step executor.
    
    Converts model output or structured step data into normalized ToolCall objects.
    """

    def __init__(self, tools: Optional[List[BaseTool]] = None):
        """
        Initialize the step executor with an optional set of LangChain tools.
        """
        super().__init__()
        self.tools = {t.name: t for t in tools} if tools else {}

    async def invoke(self, input: Any, **kwargs) -> LLMStepResponse:
        """
        Process a single step (like one returned by an agent or chain)
        and normalize it into a structured LLMStepResponse.
        """

        # Accept both 'suggested_tools' and 'calls'
        calls_spec = (
            input.get("suggested_tools")
            or input.get("calls")
            or []
        )

        # Normalize to list
        if isinstance(calls_spec, (Mapping, str)) or hasattr(calls_spec, "__dict__"):
            calls_spec = [calls_spec]
        elif not isinstance(calls_spec, list):
            raise ValueError("suggested_tools must be a list, mapping, or string")

        tool_calls: List[ToolCall] = []

        for i, spec in enumerate(calls_spec):
            server: str
            tool: str

            if isinstance(spec, str):
                try:
                    server, tool = spec.strip().split(".", 1)
                except ValueError:
                    raise ValueError(f"Invalid tool spec '{spec}' — must be 'server.tool'")
            elif isinstance(spec, Mapping):
                server = spec.get("server")
                tool = spec.get("tool")
            elif hasattr(spec, "__dict__"):
                spec = vars(spec)
                server = spec.get("server")
                tool = spec.get("tool")
            else:
                raise ValueError(f"suggested_tools[{i}] must be 'server.tool' or mapping")

            tool_calls.append(ToolCall(server=server, tool=tool))

        return LLMStepResponse(tool_calls=tool_calls)


# ---------------- MAIN / TEST -----------------
if __name__ == "__main__":
    import asyncio

    async def main():
        executor = LangchainStepExecutor()
        step = {"suggested_tools": ["email_server.list_emails", "email_server.send_email"]}
        response = await executor.invoke(step)
        print(response)

    asyncio.run(main())
