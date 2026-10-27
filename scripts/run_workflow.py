"""Single source of input workflow runner for Agents 1, 2, and 3.

Runs:
    Agent 1 (Prompt Intelligence) ->
    Agent 2 (Source Intelligence) [Independent source citations] ->
    Agent 3 (Ground Truth & Verification) [Web verification -> Capabilities -> Blog Guidance]

Input:
    - campaign_id: e.g. 81
    - prompt_id: e.g. 1409
    - website / company_link: e.g. https://alamarbio.com
    - customer_name: (optional, inferred from Agent 1 / DB if omitted)

Outputs:
    - agent1_output.json
    - agent2_output.json
    - agent3_output.json
    - workflow_output.json (All outputs consolidated, ready for Agent 4)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aca.agents.ground_truth import MAX_QUESTIONS  # noqa: E402
from aca.agents.source_intelligence import TOP_N_SOURCES  # noqa: E402
from aca.orchestrator.checkpointer import postgres_checkpointer, thread_config  # noqa: E402
from aca.orchestrator.pipeline import PipelineState, build_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ACA end-to-end workflow (Agent 1, Agent 2, Agent 3) from a single source of input."
    )
    parser.add_argument("--campaign-id", type=int, required=True, help="Campaign ID (e.g. 81)")
    parser.add_argument("--prompt-id", type=str, required=True, help="Prompt ID (e.g. 1409)")
    parser.add_argument(
        "--website",
        "--company-link",
        dest="website",
        required=True,
        help="Customer website link (e.g. https://alamarbio.com)",
    )
    parser.add_argument(
        "--customer",
        "--company-name",
        dest="company_name",
        default="",
        help="Optional customer name override (inferred by Agent 1 if omitted)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N_SOURCES,
        help=f"Number of cited sources for Agent 2 to analyze (default: {TOP_N_SOURCES})",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=MAX_QUESTIONS,
        help=f"Maximum capability questions for Agent 3 to test (default: {MAX_QUESTIONS})",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save agent outputs (default: current directory)",
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ACA WORKFLOW: SINGLE SOURCE OF INPUT")
    print(f"  Campaign ID : {args.campaign_id}")
    print(f"  Prompt ID   : {args.prompt_id}")
    print(f"  Company Link: {args.website}")
    if args.company_name:
        print(f"  Customer    : {args.company_name}")
    print("=" * 60)

    initial_state = PipelineState(
        campaign_id=args.campaign_id,
        prompt_id=str(args.prompt_id).strip(),
        website=args.website.strip(),
        company_name=args.company_name.strip(),
        top_n=args.top_n,
        max_questions=args.max_questions,
    )

    print("\n[1/3] Executing Workflow Pipeline...")
    with postgres_checkpointer() as checkpointer:
        pipeline = build_pipeline(checkpointer)
        final_dict = pipeline.invoke(initial_state, config=thread_config(initial_state.run_id))
    state = PipelineState.model_validate(final_dict)

    # 1. Save Agent 1 output
    if state.prompt_analysis is not None:
        p1 = out_dir / "agent1_output.json"
        p1.write_text(
            json.dumps(state.prompt_analysis.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  [Agent 1: Prompt Intelligence] Completed -> Saved to {p1.resolve()}")
        print(f"    - Customer Inferred : {state.prompt_analysis.customer_name}")
        print(f"    - Prompt Intent     : {state.prompt_analysis.prompt_understanding.intent}")
        print(f"    - Visibility Status : {state.prompt_analysis.customer_visibility.status}")
    else:
        print("  [Agent 1] Failed or not executed.")

    # 2. Save Agent 2 output
    if state.source_analysis is not None:
        p2 = out_dir / "agent2_output.json"
        p2.write_text(
            json.dumps(state.source_analysis.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  [Agent 2: Source Intelligence] Completed -> Saved to {p2.resolve()}")
        print(f"    - Unique Sources    : {state.source_analysis.citation_summary.unique_sources}")
        print(f"    - Pages Analyzed    : {len(state.source_analysis.content_analyses)}")
    else:
        print("  [Agent 2] Failed or not executed.")

    # 3. Save Agent 3 output
    p3 = out_dir / "agent3_output.json"
    if state.guide is not None:
        p3.write_text(
            json.dumps(state.guide.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  [Agent 3: Ground Truth & Research] Completed -> Saved to {p3.resolve()}")
        if state.guide.verification:
            print(f"    - Verification      : {'VERIFIED' if state.guide.verification.is_verified else 'FAILED'}")
            print(f"    - Reason            : {state.guide.verification.reason}")
        print(f"    - Main Topic        : {state.guide.main_topic}")
        print(f"    - Search Intent     : {state.guide.search_intent}")
        print(f"    - Target Audience   : {state.guide.target_audience}")
        print(f"    - Prompt Scope      : {state.guide.prompt_scope}")
        print(f"    - Candidate Subtopics: {len(state.guide.candidate_subtopics)}")
        for sub in state.guide.candidate_subtopics:
            print(f"        * [{sub.subtopic_id}] {sub.title} ({sub.topic_angle} | {sub.evidence_status})")
        print(f"    - Verified Facts    : {len(state.guide.verified_facts)}")
        for vf in state.guide.verified_facts[:4]:
            print(f"        * [{vf.fact_id}] ({vf.evidence_type}) {vf.statement}")
        print(f"    - Evidence Items    : {len(state.guide.evidence)}")
        print(f"    - Research Gaps     : {len(state.guide.research_gaps)}")
        for gap in state.guide.research_gaps[:3]:
            print(f"        * {gap.topic_or_question} ({gap.reason})")
        print(f"    - Limitations       : {len(state.guide.limitations)}")
    else:
        p3.write_text("null\n", encoding="utf-8")
        err_msg = state.errors[0] if state.errors else "Customer irrelevant or intent invalid."
        print(f"  [Agent 3: Ground Truth & Research] Returned null -> Saved null to {p3.resolve()}")
        print(f"    - STOPPED: {err_msg}")

    # 4. Save consolidated workflow output for Agent 4
    consolidated = {
        "campaign_id": state.campaign_id,
        "prompt_id": state.prompt_id,
        "customer_name": state.company_name or (state.prompt_analysis.customer_name if state.prompt_analysis else ""),
        "website": state.website,
        "workflow_status": "ready_for_agent4" if state.guide is not None else "stopped_no_blog_content",
        "agent1_prompt_intelligence": state.prompt_analysis.model_dump() if state.prompt_analysis else None,
        "agent2_source_intelligence": state.source_analysis.model_dump() if state.source_analysis else None,
        "agent3_ground_truth": state.guide.model_dump() if state.guide else None,
        "errors": state.errors,
    }
    pw = out_dir / "workflow_output.json"
    pw.write_text(json.dumps(consolidated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[Consolidated] Saved full workflow state to {pw.resolve()}")

    if state.guide is None:
        print("\nWorkflow stopped: Intent is invalid or customer is not relevant. No blog content will be written.")
    else:
        print("\nWorkflow completed successfully! Output is ready to connect directly to Agent 4 (Brief).")


if __name__ == "__main__":
    main()
