"""
Schemas for Planning and Tool Intents
-------------------------------------

This module defines **strict JSON Schemas** that are used by ShardGuard’s
Planning LLM and its tool execution system. These schemas ensure:
- Deterministic outputs from the planning LLM.
- Structural and security validation before execution.
- Predictable, minimal shape for tool call intents and plan payloads.

By validating against these, ShardGuard guarantees that every step conforms
to a safe, controlled data model, preventing malformed or injected responses.
"""

# This schema defines the structure of tool call "intents" produced
# by the ExecutionLLM. Each element represents a call
# to a specific tool on a specific MCP server.
#
# Example of a valid object:
# [
#     {
#         "server": "nuke_server",
#         "tool": "launch_missile",
#         "args": { "arm_warhead": "WH-07" }
#     }
# ]

TOOL_INTENTS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["server", "tool"],
        "properties": {
            "server": {"type": "string", "minLength": 1},
            "tool":   {"type": "string", "minLength": 1},
            "args":   {"type": "object"}
        },
        "additionalProperties": False       # No extra values permitted 
    }
}

# Defines the top-level structure of a "Plan" JSON returned by the
# Planning LLM. The plan describes how the system should break down
# the user's original input prompt into multiple sub-prompts that can
# be executed deterministically and safely.
#
# Example:
# {
#   "original_prompt": "Get me data for [[P_1]]",
#   "sub_prompts": [
#       {
#           "id": 1,
#           "content": "Fetch data for [[P_1]]",
#           "opaque_values": { "[[P_1]]": "President of US" }
#       }
#   ]
# }

PLANNING_LLM_SCHEMA = {
    "type": "object",
    "required": ["original_prompt", "sub_prompts"],
    "properties": {
        "original_prompt": {
            "type": "string",
            "minLength": 1,
            "description": "Original input with sensitive data replaced by [[Pn]] tokens"
        },
        "sub_prompts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "content", "suggested_tools"],
                "properties": {
                    "id": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Unique identifier for the sub-prompt"
                    },
                    "content": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Subtask prompt text with [[Pn]] tokens"
                    },
                    "opaque_values": {
                        "type": "object",
                        "patternProperties": {
                            r"\[\[P\d+\]\]": {"type": "string"}
                        },
                        "additionalProperties": {},
                        "description": "Optional mapping of placeholder tokens to corresponding sensitive data"
                    }, 
                    "suggested_tools": {
                        "type": "array",
                        "items":{
                            "type":"string",
                            "minLength": 1,
                            "description": "Each tool with server name in the following format server_name.tool_name"
                        },
                        "minLength": 1,
                        "description": "List of suggested tools for the subprompt to be executed"
                    }
                },
                "additionalProperties": {}
            },
            "minItems": 1,
            "description": "List of sub-prompts derived from the original prompt"
        }
    },
    "additionalProperties": {}
}