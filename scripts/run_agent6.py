"""Run Agent 6 from saved Agent 1-5 JSON outputs without rerunning the graph.

Example:
    python scripts/run_agent6.py
    python scripts/run_agent6.py --agent5 path\\to\\agent5_output.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from aca.agents.reviewer import ReviewerAgent  # noqa: E402
from aca.models.content_brief import ContentBrief  # noqa: E402
from aca.models.ground_truth import GroundTruthGuide  # noqa: E402
from aca.models.prompt_intelligence import PromptAnalysis  # noqa: E402
from aca.models.source_intelligence import SourceIntelligence  # noqa: E402
from aca.models.writer import WriterOutput  # noqa: E402
from aca.orchestrator.pipeline import PipelineState, _build_writer_input  # noqa: E402


def load_review_state(paths: dict[str, Path], website: str = "") -> PipelineState:
    """Validate saved agent files and reconstruct the state needed by Agent 6."""
    models = {
        "agent1": PromptAnalysis,
        "agent2": SourceIntelligence,
        "agent3": GroundTruthGuide,
        "agent4": ContentBrief,
        "agent5": WriterOutput,
    }
    outputs: dict[str, Any] = {}
    run_ids: list[str | None] = []
    for name, model in models.items():
        path = paths[name].resolve()
        if not path.is_file():
            label = name.replace("agent", "Agent ")
            raise FileNotFoundError(f"Missing {label} output JSON: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{name} output must contain a JSON object: {path}")
        run_ids.append(raw.get("_run_id"))
        outputs[name] = model.model_validate(raw)

    present_run_ids = {run_id for run_id in run_ids if run_id is not None}
    if present_run_ids and (len(present_run_ids) != 1 or any(run_id is None for run_id in run_ids)):
        raise ValueError(
            "Agent 1-5 files must all have the same _run_id; " "these files may come from different runs."
        )
    run_id = next(iter(present_run_ids), uuid4().hex)

    prompt_id = str(outputs["agent1"].prompt_id)
    if prompt_id != str(outputs["agent2"].prompt_id):
        raise ValueError("Agent 1 and Agent 2 outputs have different prompt IDs.")

    guide = outputs["agent3"]
    if guide.verification is None or not guide.verification.is_verified:
        raise ValueError("Agent 3 verification did not pass; there is no approved run to review.")
    handoff = guide.agent4_handoff
    company = handoff.company if handoff else None
    resolved_website = website.strip() or (company.website if company else "")
    if not resolved_website:
        raise ValueError("Customer website was not saved in Agent 3 output. Supply it with --website.")
    customer_name = company.name if company else outputs["agent1"].customer_name

    state = PipelineState(
        run_id=run_id,
        campaign_id=outputs["agent2"].campaign_id,
        prompt_id=prompt_id,
        website=resolved_website,
        company_name=customer_name,
        prompt_analysis=outputs["agent1"],
        source_analysis=outputs["agent2"],
        guide=guide,
        content_brief=outputs["agent4"],
        written_article=outputs["agent5"],
        review_requested=True,
    )
    if not state.content_brief.sections:
        raise ValueError("Agent 4 output has no sections; cannot review the draft.")
    _build_writer_input(state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run only Agent 6 using saved Agent 1-5 JSON outputs.")
    for number in range(1, 6):
        parser.add_argument(
            f"--agent{number}",
            type=Path,
            default=ROOT / f"agent{number}_output.json",
            help=f"Agent {number} output JSON (default: agent{number}_output.json)",
        )
    parser.add_argument(
        "--website",
        default="",
        help="Customer website if it is missing from the Agent 3 output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "agent6_output.json",
        help="Reviewer result JSON path.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=ROOT / "agent6_reviewed.md",
        help="Revised Markdown article path.",
    )
    args = parser.parse_args()

    paths = {f"agent{number}": getattr(args, f"agent{number}") for number in range(1, 6)}
    state = load_review_state(paths, website=args.website)
    output_path = args.output.resolve()
    markdown_path = args.markdown_output.resolve()
    if output_path == markdown_path:
        parser.error("--output and --markdown-output must be different files")
    input_paths = {path.resolve() for path in paths.values()}
    if output_path in input_paths or markdown_path in input_paths:
        parser.error("Output paths must not overwrite Agent 1-5 input files")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    writer_input = _build_writer_input(state)
    if state.written_article is None:
        raise ValueError("Agent 5 output is missing.")
    result = ReviewerAgent().review(writer_input, state.written_article)

    output = result.model_dump(mode="json")
    output["_run_id"] = state.run_id
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        result.revised_blog.article_markdown.rstrip() + "\n",
        encoding="utf-8",
    )
    print(f"Agent 6 review JSON: {output_path}")
    print(f"Revised blog Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
