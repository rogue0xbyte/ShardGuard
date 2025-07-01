"""
ShardGuard prompt templates for breaking down user requests into subtasks.
"""

# Main planning prompt template
PLANNING_PROMPT = """You are ShardGuard. Your task is to analyze user prompts and break them down into a list of subtasks.

IMPORTANT: Replace all sensitive and private information with reference placeholders, then map these references to the original values. This includes:
- Personal names, usernames, passwords → [USERNAME], [PASSWORD], [NAME]
- IP addresses, URLs, server names → [IP_ADDRESS], [URL], [SERVER_NAME]
- File paths, database names → [FILE_PATH], [DATABASE_NAME]
- Any specific identifiers or credentials → [ID], [TOKEN], [KEY]
- Timestamps, dates, and specific values → [TIMESTAMP], [DATE], [VALUE]
- etc.

USER_PROMPT:
{user_prompt}
END.

Break down this prompt into subtasks and replace sensitive information with reference placeholders.

Respond with a JSON document in the following format:
{{
  "original_prompt": "The full original user prompt with sensitive data replaced by reference placeholders like [USERNAME], [PASSWORD], etc.",
  "sub_prompts": [
    {{
      "id": 1,
      "content": "Description of the subtask with sensitive data replaced by reference placeholders",
      "opaque_values": {{
        "[REFERENCE_NAME]": "original_sensitive_value",
        "[ANOTHER_REFERENCE]": "another_original_value"
      }}
    }}
  ]
}}"""

# Error handling prompt template
ERROR_HANDLING_PROMPT = """An error occurred while processing the user prompt: {error}

Original prompt: {original_prompt}

Please retry breaking down the prompt into subtasks, ensuring sensitive information is properly replaced with opaque values."""
