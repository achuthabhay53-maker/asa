"""Persist output from the content workflow agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def clear_agent_outputs(output_dir: str | Path) -> None:
    directory = Path(output_dir)
    for agent_number in range(1, 7):
        (directory / f"agent{agent_number}_output.json").unlink(missing_ok=True)
