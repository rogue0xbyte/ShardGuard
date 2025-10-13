"""
This file handles the whole coordination of the prompt from Planning to
Execution, it is the middleware for everything. Every step executed goes
from here and comes back here to be further processed. This is because we 
trust that the coordination service is the trusted source for ShardGuard.
"""

from rich.console import Console
from dataclasses import is_dataclass, asdict
from typing import Any, Dict, Mapping, Optional

from shardguard.core.models import Plan
from shardguard.core.planning import PlanningLLM
from shardguard.core.prompts import PLANNING_PROMPT
from shardguard.core.execution import StepExecutor, LLMStepResponse, make_execution_llm
from shardguard.core.mcp_integration import MCPClient
from shardguard.utils.validator import _validate_output

class CoordinationService:
    """Coordination service for planning."""

    def __init__(self, planner: PlanningLLM):
        self.planner = planner
        self.console = Console()
        # Saving all the args (opaque values) from the prompt into this dictionary 
        self.args: Dict[str, Any] = {}

    def _to_dict(self, obj: Any) -> Dict[str, Any]:
        """
        Normalize SubPrompt into a real dict.
        When the prompt goes to the LLM, it returns a Pydantic model, 
        which makes it difficult to process for python, instead to make 
        it more generalized have added this middleware to make any kind of 
        data that comes in, return as a dict for simplicity purposes. 
        So that, even if someone changes the structure of the SubPrompt in 
        the future and some other datatype comes in, we would not have to 
        make changes again to this.
        """
        if isinstance(obj, dict):
            return obj
        if is_dataclass(obj):
            return asdict(obj)
        # Added this as most of the tool objects are being referenced as Sharguard Model Schema
        if hasattr(obj, "model_dump") and callable(obj.model_dump): 
            return obj.model_dump()  # Pydantic v2
        if hasattr(obj, "dict") and callable(obj.dict):
            return obj.dict()        # Pydantic v1
        if isinstance(obj, Mapping):
            return dict(obj)
        if hasattr(obj, "__dict__"):
            return dict(vars(obj))
        raise TypeError(f"Unsupported step type: {type(obj)!r}. Provide a dict-like object.")
    
    async def check_tool(self, suggested_tools) -> bool:
        """Check whether the Planning LLM gave the tools only from those present with us and not hallucinate"""
        mcp = MCPClient()
        tools = await mcp.list_tool_names()
        for tool in suggested_tools:
            if tool in tools:
                return True
        return False

    async def handle_prompt(self, user_input: str) -> Plan:
        """Prepare the prompt by adding predefined context to design the plan of execution"""
        formatted_prompt = self._format_prompt(user_input)
        plan_json = await self.planner.generate_plan(formatted_prompt)
        
        # Set the plan to a valid json for processing
        plan_tool_check = Plan.model_validate_json(plan_json).model_dump(exclude_none=True)
        # Looping into subprompts to get suggested tools, and check the tool exists in the system before execution starts
        tool_check = [] # this is an array as we want all the Sub Prompts to have the tools only existing in the system
        for items in plan_tool_check["sub_prompts"]:
            tool_check.append(await self.check_tool(items["suggested_tools"]))
        
        # Validating if all are True in the array, else PlanningLLM is retried
        if(not(False in tool_check)):
            return Plan.model_validate_json(plan_json)
        else:
            print("Retrying PlanningLLM, tool suggestions invalid!")
            await self.handle_prompt(user_input)

    def _format_prompt(self, user_input: str) -> str:
        """Format the user input using the planning prompt template."""
        return PLANNING_PROMPT.format(user_prompt=user_input)
    
    # Have kept the following function commented cause it maybe used in future
    # def extract_arguments(self, task):
    #     """
    #     Extracting arguments from the prompt for both cases:
    #         1. Getting both key-value pairs for the system as a whole 
    #         (system args that will be known only to the coordination service)
    #         2. Getting only the key for the parameter to be obfuscated
    #         (args that can be used and referenced by any subprompt cause this is opaque and obfuscated)
    #     """
    #     opaque = task.get("opaque_values") or {}
    #     if not isinstance(opaque, Mapping):
    #         return []
    #     for k, v in opaque.items():
    #         self.args[k] = v
    #     return list(opaque.keys())

    async def _execute_step_tools(self, step: Dict[str, Any], resp: LLMStepResponse):
        """
        After the Execution LLM processes the subprompt, it prepares the tool call
        and this tool call schema is also validated to make the responses from the LLM
        as deterministic as possible.
        """
        mcp = MCPClient()
        output_schema: Optional[Dict[str, Any]] = step.get("output_schema")

        for call in resp.tool_calls:
            # Build per-call args
            per_tool_args: Dict[str, Any] = {}
            if call.args:
                per_tool_args.update(call.args)
            print("server", call.server)
            print("tool", call.tool)
            result = await mcp.call_tool(call.server, call.tool, per_tool_args)
            # Validating the result from the tool call with the expected schema
            _validate_output(result, output_schema, where="Tool Call")
            print()
    
    async def handle_subtasks(self, tasks, provider, detected_model, api_key):
        """Sends the subtasks to ExecutionLLM"""
        for task in tasks:
            # Instantiating a new ExecutionLLM for each task so that none of them have each others context
            exec_llm = make_execution_llm(provider, detected_model, api_key=api_key)
            executor = StepExecutor(exec_llm)
            task = self._to_dict(task)
            # argument_dicts = self.extract_arguments(task) -- kept this for future use, can be discarded later if not needed after discussion
            # Sends the task to process for execution
            print(task)
            resp = await executor.run_step(task)
            await self._execute_step_tools(task, resp)
        return

