"""Persist output from the content workflow agents."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
AGENT1_DEBUG_OUTPUT_DIR = DEFAULT_OUTPUT_DIR.parent / "outputs2" / "agent1"
GENERAL_BLOG_DEBUG_OUTPUT_DIR = DEFAULT_OUTPUT_DIR.parent / "outputs2" / "general_blog"


def save_general_blog_stage_output(
    run_id: str,
    stage: str,
    output: dict[str, Any],
) -> Path:
    if not re.fullmatch(r"[a-fA-F0-9-]{1,64}", run_id):
        raise ValueError("run_id must contain only hexadecimal characters and hyphens.")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", stage):
        raise ValueError("stage must contain only letters, numbers, underscores, and hyphens.")

    path = GENERAL_BLOG_DEBUG_OUTPUT_DIR / run_id / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def save_agent_output(
    agent_number: int,
    output: Any,
    output_dir: str | Path,
    run_id: str | None = None,
) -> Path:
    if not 1 <= agent_number <= 6:
        raise ValueError("agent_number must be between 1 and 6")

    if hasattr(output, "model_dump"):
        output = output.model_dump(mode="json")
    if run_id is not None:
        if not isinstance(output, dict):
            raise TypeError("output must be a JSON object when run_id is provided")
        output = {**output, "_run_id": run_id}
    path = Path(output_dir) / f"agent{agent_number}_output.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def clear_agent_outputs(output_dir: str | Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for agent_number in range(1, 7):
        (directory / f"agent{agent_number}_output.json").unlink(missing_ok=True)


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")
