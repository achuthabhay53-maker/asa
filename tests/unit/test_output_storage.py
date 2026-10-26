import json

import pytest

from aca.output_storage import clear_agent_outputs, save_agent_output


def test_saves_and_overwrites_agent_output(tmp_path):
    path = save_agent_output(1, {"prompt_id": "first"}, tmp_path)
    save_agent_output(1, {"prompt_id": "second"}, tmp_path)

    assert path.name == "agent1_output.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"prompt_id": "second"}


def test_clears_only_current_agent_output_files(tmp_path):
    for number in range(1, 7):
        save_agent_output(number, {"run": "old"}, tmp_path)
    unrelated = tmp_path / "agent5_alamar_test.json"
    unrelated.write_text("keep", encoding="utf-8")

    clear_agent_outputs(tmp_path)

    assert not any((tmp_path / f"agent{number}_output.json").exists() for number in range(1, 7))
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_rejects_agent_numbers_outside_workflow(tmp_path):
    with pytest.raises(ValueError, match="between 1 and 6"):
        save_agent_output(7, {}, tmp_path)


def test_saves_reviewer_output(tmp_path):
    path = save_agent_output(6, {"summary": "review complete"}, tmp_path)

    assert path.name == "agent6_output.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"summary": "review complete"}


def test_persists_run_id_with_agent_output(tmp_path):
    path = save_agent_output(3, {"topic": "assay research"}, tmp_path, run_id="run-123")

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "topic": "assay research",
        "_run_id": "run-123",
    }
