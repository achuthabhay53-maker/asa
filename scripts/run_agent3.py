"""Manual runner for Agent 3.

The website is typed here. The company name and prompt context come from Agent 1.

    python scripts/run_agent3.py ^
        --website https://alamarbio.com ^
        --agent1 agent1_output.json ^
        --agent2 agent2_output.json
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

from aca.agents.ground_truth import MAX_QUESTIONS
from aca.models.ground_truth import GroundTruthGuide
from aca.models.prompt_intelligence import PromptAnalysis
from aca.models.source_intelligence import SourceIntelligence
from aca.orchestrator.pipeline import PipelineState, agent3_node


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agent 3 on saved Agent 1 and Agent 2 JSON.")
    parser.add_argument("--website", required=True, help="Client site, for example https://alamarbio.com")
    parser.add_argument("--company-name", help="Overrides the customer name from Agent 1")
    parser.add_argument("--agent1", required=True, help="Path to Agent 1 JSON")
    parser.add_argument("--agent2", required=True, help="Path to Agent 2 JSON")
    parser.add_argument("--max-questions", type=int, default=MAX_QUESTIONS)
    parser.add_argument("--output", default="agent3_output.json")
    args = parser.parse_args()

    prompt_analysis = _load(args.agent1)
    source_analysis = _load(args.agent2)
    company_name = (args.company_name or prompt_analysis.get("customer_name") or "").strip()
    if not company_name:
        raise SystemExit("Agent 1 JSON has no customer_name. Pass --company-name.")
    state = PipelineState(
        campaign_id=int(prompt_analysis.get("campaign_id") or source_analysis.get("campaign_id") or 0),
        prompt_id=str(prompt_analysis.get("prompt_id") or ""),
        website=args.website,
        company_name=company_name,
        max_questions=args.max_questions,
        prompt_analysis=PromptAnalysis.model_validate(prompt_analysis),
        source_analysis=SourceIntelligence.model_validate(source_analysis),
    )
    result = agent3_node(state)
    guide = result.get("guide")
    errors = result.get("errors") or []
    if guide is None:
        msg = errors[0] if errors else "Prompt intent is invalid or customer is not relevant to requested topic."
        print(f"Company: {company_name}")
        print(f"Website: {args.website}")
        print(f"Prompt: {prompt_analysis.get('prompt', '')}")
        print(f"Agent 3 Result: null ({msg})")
        Path(args.output).write_text("null\n", encoding="utf-8")
        print(f"Output saved to: {Path(args.output).resolve()}")
        return
    if isinstance(guide, dict):
        guide = GroundTruthGuide.model_validate(guide)

    payload = guide.model_dump(mode="json")
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Company: {company_name}")
    print(f"Website: {args.website}")
    print(f"Prompt: {prompt_analysis.get('prompt', '')}")
    if guide.verification:
        print(f"Verification: {'VERIFIED' if guide.verification.is_verified else 'FAILED'} (intent_valid={guide.verification.intent_valid}, customer_relevant={guide.verification.customer_relevant})")
        print(f"Verification Reason: {guide.verification.reason}")
    print("\n--- STRUCTURED INPUT FOR AGENT 4 ---")
    print(f"Main Topic     : {guide.main_topic}")
    print(f"Search Intent  : {guide.search_intent}")
    print(f"Target Audience: {guide.target_audience}")
    print(f"Prompt Scope   : {guide.prompt_scope}")
    print(f"\nCandidate Subtopics ({len(guide.candidate_subtopics)}):")
    for sub in guide.candidate_subtopics:
        status = sub.support_status
        fact_refs = f" facts={sub.verified_fact_ids}" if sub.verified_fact_ids else ""
        ev_refs = f" ev={sub.evidence_ids}" if sub.evidence_ids else ""
        print(f"  [{sub.subtopic_id}] {sub.title}")
        print(f"      angle={sub.topic_angle} | relevance={sub.relevance} | status={status}{fact_refs}{ev_refs}")
        if sub.research_needed:
            print(f"      research_needed: {sub.research_needed[0]}")
        if sub.suggested_questions:
            print(f"      questions: {' | '.join(sub.suggested_questions[:2])}")
    print(f"\nVerified Facts ({len(guide.verified_facts)}):")
    for vf in guide.verified_facts:
        source_lbl = "[Company]" if vf.evidence_type == "company_website" else "[External]"
        boundary = f" (Boundary: {vf.wording_boundary})" if vf.wording_boundary else ""
        ev_lbl = f" ev={vf.evidence_ids}" if vf.evidence_ids else ""
        print(f"  [{vf.fact_id}] {source_lbl} {vf.statement}{boundary}{ev_lbl}")
    print(f"\nResearched Evidence ({len(guide.evidence)}):")
    for ev in guide.evidence[:6]:
        print(f"  [{ev.evidence_id}] ({ev.source_type}) {ev.url}")
        print(f"      \"{ev.quote[:110]}...\"")
    print(f"\nResearch Gaps ({len(guide.research_gaps)}):")
    for gap in guide.research_gaps:
        print(f"  [{gap.gap_id}] {gap.topic_or_question}")
    print(f"\nLimitations ({len(guide.limitations)}):")
    for lim in guide.limitations[:5]:
        print(f"  - {lim}")

    # ── Agent 4 handoff summary ──────────────────────────────────────────
    h = guide.agent4_handoff
    if h:
        print("\n" + "=" * 60)
        print("AGENT 4 RESEARCH HANDOFF")
        print("=" * 60)
        print(f"1. TOPIC: {h.topic}")
        print(f"   Target Audience: {h.target_audience}")
        print(f"   Search Intent  : {h.search_intent}")
        print(f"\n2. RELEVANT FINDINGS ({len(h.relevant_findings)}):")
        for pt in h.relevant_findings[:6]:
            status_str = f" [{pt.verification_status}]" if pt.verification_status else ""
            url_str = f" ({pt.source_urls[0]})" if pt.source_urls else ""
            print(f"   • {pt.claim}{status_str}{url_str}")
        print(f"\n3. EVIDENCE ({len(h.evidence)} items):")
        for ev in h.evidence[:4]:
            print(f"   [{ev.evidence_id}] {ev.url}")
            print(f"       \"{ev.quote[:90]}...\"")
        print(f"\n4. INFORMATION STILL NEEDED ({len(h.information_still_needed)}):")
        for gap in h.information_still_needed[:4]:
            print(f"   - {gap.topic_or_question}")
        print(f"\n5. CANDIDATE SUBTOPICS ({len(h.candidate_subtopics)}):")
        for sub in h.candidate_subtopics[:5]:
            fids = f" facts={sub.verified_fact_ids}" if sub.verified_fact_ids else ""
            print(f"   - {sub.title} [{sub.support_status}]{fids}")
        print(f"\n6. BOUNDARIES ({len(h.boundaries)}):")
        for b in h.boundaries:
            print(f"   - {b}")
        print("=" * 60)
    else:
        print("\n[!] agent4_handoff is None — check GroundTruthAgent output")

    print(f"\nOutput saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
