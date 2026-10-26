import json

import pytest

from scripts.run_agent6 import load_review_state
from tests.agent_fixtures import (
    agent1_output,
    agent2_output,
    agent3_output,
    agent4_output,
    agent5_output,
)


def _write_outputs(tmp_path, run_ids=None):
    outputs = {
        "agent1": agent1_output(),
        "agent2": agent2_output(),
        "agent3": agent3_output(),
        "agent4": agent4_output(),
        "agent5": agent5_output(),
    }
    paths = {}
    for name, output in outputs.items():
        data = output.model_dump(mode="json")
        if run_ids is not None:
            data["_run_id"] = run_ids[name]
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        paths[name] = path
    return paths


def test_load_review_state_reuses_saved_agent1_through_agent5_outputs(tmp_path):
    run_id = "run-123"
    paths = _write_outputs(tmp_path, {f"agent{i}": run_id for i in range(1, 6)})

    state = load_review_state(paths, website="https://example.com")

    assert state.run_id == run_id
    assert state.review_requested is True
    assert state.written_article == agent5_output()
    assert state.website == "https://example.com"


def test_load_review_state_rejects_mixed_saved_runs(tmp_path):
    run_ids = {f"agent{i}": "run-a" for i in range(1, 6)}
    run_ids["agent5"] = "run-b"
    paths = _write_outputs(tmp_path, run_ids)

    with pytest.raises(ValueError, match="same _run_id"):
        load_review_state(paths)


def test_load_review_state_requires_all_five_agent_files(tmp_path):
    paths = _write_outputs(tmp_path)
    paths["agent4"] = tmp_path / "missing-agent4.json"

    with pytest.raises(FileNotFoundError, match="Agent 4"):
        load_review_state(paths)
