from pydantic import BaseModel, Field


class SubPrompt(BaseModel):
    id: int
    content: str
    opaque_values: dict[str, str] = Field(default_factory=dict)


class Plan(BaseModel):
    original_prompt: str
    sub_prompts: list[SubPrompt]
