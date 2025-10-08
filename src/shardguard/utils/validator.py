from jsonschema import validate as jsonschema_validate, ValidationError
from typing import Any, Dict, Optional

def _validate_output(output: Any, schema: Optional[Dict[str, Any]], where) -> None:
    """
    Added a schema validation for the output generated so as 
    to make the LLM responses as deterministic as possible
    """
    if not schema:
        return
    try:
        jsonschema_validate(instance=output, schema=schema)
    except ValidationError as e:
        raise RuntimeError(f"{where} output failed schema validation: {e.message}") from e