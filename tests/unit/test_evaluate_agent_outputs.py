import json

import pytest

from scripts.evaluate_agent_outputs import _load_agent_outputs
from ..agent_fixtures import (
    agent1_output,
    agent2_output,
    agent3_output,
    agent4_output,
    agent5_output,
)


def test_evaluator_rejects_outputs_from_different_runs(tmp_path):
    outputs = {
        "agent1": agent1_output(),
        "agent2": agent2_output(),
        "agent3": agent3_output(),
        "agent4": agent4_output(),
        "agent5": agent5_output(),
    }
    paths = {}
    for agent, output in outputs.items():
        data = output.model_dump(mode="json")
        data["_run_id"] = "run-2" if agent == "agent5" else "run-1"
        path = tmp_path / f"{agent}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        paths[agent] = path

    with pytest.raises(ValueError, match="different _run_id"):
        _load_agent_outputs(paths)
