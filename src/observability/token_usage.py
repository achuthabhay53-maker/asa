"""Capture Claude token usage and associate calls with workflow runs."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import Any
from uuid import uuid4

from src.models.token_usage import ClaudeTokenUsage

logger = logging.getLogger(__name__)

_ActiveUsage = tuple[str, str, list[ClaudeTokenUsage]]
_active_usage: ContextVar[_ActiveUsage | None] = ContextVar("active_claude_usage", default=None)
_records_by_run: dict[str, dict[str, ClaudeTokenUsage]] = {}
_records_lock = Lock()


@contextmanager
def token_usage_scope(run_id: str, agent_name: str) -> Iterator[list[ClaudeTokenUsage]]:
    """Collect calls made in one workflow stage or direct UI agent action."""
    records: list[ClaudeTokenUsage] = []
    token = _active_usage.set((run_id, agent_name, records))
    try:
        yield records
    finally:
        _active_usage.reset(token)


def record_claude_response(response: Any, model: str) -> ClaudeTokenUsage:
    """Record token counts from an Anthropic SDK response or decoded API payload."""
    active = _active_usage.get()
    run_id, agent_name = (active[0], active[1]) if active else (None, "unscoped")

    response_data = _value(response, "usage")
    if response_data is None and hasattr(response, "json"):
        try:
            response_data = _value(response.json(), "usage")
        except ValueError:
            response_data = None

    input_tokens = _value(response_data, "input_tokens")
    output_tokens = _value(response_data, "output_tokens")
    usage_available = input_tokens is not None and output_tokens is not None
    if usage_available:
        input_tokens = int(input_tokens)
        output_tokens = int(output_tokens)
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Claude returned negative token usage.")
    else:
        logger.warning(
            "Claude response did not include token usage (run_id=%s, agent=%s).",
            run_id or "unavailable",
            agent_name,
        )
        input_tokens = output_tokens = 0

    record = ClaudeTokenUsage(
        call_id=uuid4().hex,
        run_id=run_id,
        agent_name=agent_name,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usage_available=usage_available,
    )
    if active:
        active[2].append(record)
    if run_id:
        with _records_lock:
            _records_by_run.setdefault(run_id, {})[record.call_id] = record

    logger.info(
        "run_id=%s agent=%s model=%s input_tokens=%d output_tokens=%d usage_available=%s",
        run_id or "unavailable",
        agent_name,
        model,
        record.input_tokens,
        record.output_tokens,
        record.usage_available,
    )
    return record


def merge_token_usage(
    existing: list[ClaudeTokenUsage],
    new_records: list[ClaudeTokenUsage],
) -> list[ClaudeTokenUsage]:
    """Append new usage records without duplicating checkpointed calls."""
    merged = {record.call_id: record for record in existing}
    merged.update((record.call_id, record) for record in new_records)
    return list(merged.values())


def take_run_token_usage(
    run_id: str | None,
    persisted_records: list[ClaudeTokenUsage] | None = None,
) -> list[ClaudeTokenUsage]:
    """Combine checkpointed records with recent calls, then release the memory cache."""
    records = list(persisted_records or [])
    if run_id:
        with _records_lock:
            recent = list(_records_by_run.pop(run_id, {}).values())
        records = merge_token_usage(records, recent)
    return records


def token_usage_totals(records: list[ClaudeTokenUsage]) -> tuple[int, int, int]:
    """Return input total, output total, and number of records without usage data."""
    available = [record for record in records if record.usage_available]
    missing_count = len(records) - len(available)
    return (
        sum(record.input_tokens for record in available),
        sum(record.output_tokens for record in available),
        missing_count,
    )


def _value(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None) if obj is not None else None
