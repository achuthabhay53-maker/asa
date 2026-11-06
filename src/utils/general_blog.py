"""Shared helpers for General Blog agents."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import anthropic
from pydantic import BaseModel

from src.observability.token_usage import record_claude_response
from src.utils.structured_output import extract_tool_input, model_input_schema

logger = logging.getLogger(__name__)


def call_structured_agent(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    payload: dict[str, Any],
    response_model: type[BaseModel],
    tool_name: str,
    *,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    tool = {
        "name": tool_name,
        "description": "Submit the requested structured response.",
        "input_schema": model_input_schema(response_model),
    }
    logger.debug(
        "Calling General Blog structured agent (agent=%s, model=%s, max_tokens=%d)",
        tool_name,
        model,
        max_tokens,
    )
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            }
        ],
    )
    record_claude_response(response, model)
    if response.stop_reason != "tool_use":
        logger.error(
            "General Blog structured response missing (agent=%s, model=%s, stop_reason=%s)",
            tool_name,
            model,
            response.stop_reason or "unknown",
        )
        raise RuntimeError(
            "General Blog agent did not return its required structured response "
            f"(stop reason: {response.stop_reason or 'unknown'})."
        )
    result = extract_tool_input(response, tool_name)
    logger.debug(
        "General Blog structured agent response received (agent=%s, fields=%d)",
        tool_name,
        len(result),
    )
    return result


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def matching_quote(quote: str, source_text: str) -> bool:
    cleaned = normalized_text(quote)
    return len(cleaned) >= 30 and cleaned in normalized_text(source_text)
