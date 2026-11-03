"""Token usage recorded for individual Claude API calls."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ClaudeTokenUsage(BaseModel):
    call_id: str
    run_id: str | None = None
    agent_name: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    usage_available: bool = True
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
