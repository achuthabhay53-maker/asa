"""Runner script for Agent 4: Strategy and Blog Brief Builder.

Usage:
    python scripts/run_agent4.py ^
        --agent1 agent1_output.json ^
        --agent2 agent2_output.json ^
        --agent3 agent3_output.json ^
        --output agent4_output.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aca.agents.content_brief import ContentBriefAgent


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Input file not found: {p.resolve()}")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agent 4 to generate an evidence-backed blog brief.")
    parser.add_argument("--agent1", default="agent1_output.json", help="Path to Agent 1 JSON")
    parser.add_argument("--agent2", default="agent2_output.json", help="Path to Agent 2 JSON")
    parser.add_argument("--agent3", default="agent3_output.json", help="Path to Agent 3 JSON")
    parser.add_argument("--output", default="agent4_output.json", help="Path to save Agent 4 brief JSON")
    parser.add_argument("--model", default=None, help="Anthropic model name override")
    args = parser.parse_args()

    agent1_data = _load(args.agent1)
    agent2_data = _load(args.agent2)
    agent3_data = _load(args.agent3)

    print("=" * 60)
    print("RUNNING AGENT 4: STRATEGY & BLOG BRIEF BUILDER")
    print("=" * 60)
    print(f"Agent 1: {args.agent1}")
    print(f"Agent 2: {args.agent2}")
    print(f"Agent 3: {args.agent3}")

    agent = ContentBriefAgent(model=args.model)
    brief = agent.build_brief(agent1_data, agent2_data, agent3_data)

    payload = brief.model_dump(mode="json")
    out_path = Path(args.output)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print("AGENT 4 BRIEF GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"Recommended Title : {brief.recommended_title}")
    print(f"Target Audience   : {brief.target_audience}")
    print(f"Article Objective : {brief.article_objective}")
    print(f"\nSections ({len(brief.sections)}):")
    for i, sec in enumerate(brief.sections, 1):
        ev_str = f" [Evidence: {', '.join(sec.evidence_ids)}]" if sec.evidence_ids else ""
        print(f"  {i}. [{sec.level}] {sec.heading}{ev_str}")
        if sec.section_objective:
            print(f"     Objective: {sec.section_objective}")
        if sec.topics_and_questions:
            print(f"     Key Questions: {' | '.join(sec.topics_and_questions[:2])}")
    print(f"\nWriting Tone      : {brief.writing_style.tone}")
    print(f"Paragraph Style   : {brief.writing_style.paragraph_style}")
    print(f"Formatting        : {', '.join(brief.writing_style.formatting_elements)}")
    print(f"\nUnsupported Claims to Avoid ({len(brief.unsupported_claims_to_avoid)}):")
    for b in brief.unsupported_claims_to_avoid[:4]:
        print(f"  - {b}")
    print("=" * 60)
    print(f"Brief saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
