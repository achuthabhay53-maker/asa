import asyncio

from aca.agents.analyzer import run
from aca.orchestrator.state import WorkflowState
from aca.tools.aimpact import (
    get_source_analyzer_input,
)


async def main():
    project_id = 29
    campaign_id = 81
    score_threshold = 70
    target_prompt_id = "1379"

    project, prompts = get_source_analyzer_input(
        project_id=project_id,
        campaign_id=campaign_id,
        score_threshold=score_threshold,
    )

    print()
    print("=" * 70)
    print("PROJECT")
    print("=" * 70)
    print(project)

    print()
    print("=" * 70)
    print("UNDERPERFORMING PROMPTS")
    print("=" * 70)

    print(
        f"Found {len(prompts)} prompts."
    )

    for prompt in prompts:
        print()
        print(
            f"Prompt ID: {prompt.prompt_id}"
        )
        print(
            f"Prompt: {prompt.prompt_text}"
        )
        print(
            f"Models: {list(prompt.per_model.keys())}"
        )

    if not prompts:
        print(
            f"No underperforming prompts found for campaign "
            f"{campaign_id} at score threshold {score_threshold}. "
            "Nothing to analyze."
        )
        return

    target_prompt = (
        prompts[0]
        if target_prompt_id is None
        else next(
            (
                prompt
                for prompt in prompts
                if prompt.prompt_id == target_prompt_id
            ),
            None,
        )
    )

    if target_prompt is None:
        available_ids = ", ".join(
            prompt.prompt_id
            for prompt in prompts
        )
        print(
            f"Prompt {target_prompt_id} was not found among "
            f"the underperforming prompts: {available_ids}. "
            "Nothing to analyze."
        )
        return

    state = WorkflowState(
        job_id="manual-source-analyzer-test",
        tenant_id=1,
        campaign_id=campaign_id,
        company_name=project["company_name"],
        prompts=[target_prompt],
    )

    print()
    print("=" * 70)
    print("RUNNING SOURCE ANALYZER")
    print("=" * 70)

    state = await run(state)

    print()
    print("=" * 70)
    print("CONTENT GAP")
    print("=" * 70)

    if state.gap is None:
        raise RuntimeError(
            "Source Analyzer did not produce a ContentGap."
        )

    print()
    print("VERIFICATION STATUS:")
    print(state.gap.verification_status)
    print(state.gap.verification_summary)
    print("Target-site coverage:", state.gap.target_site_coverage)
    print("Content-gap status:", state.gap.content_gap_status)
    print("Opportunity:", state.gap.opportunity)

    print()
    print("Prompt ID:")
    print(state.gap.prompt_id)

    print()
    print("Prompt:")
    print(state.gap.prompt_text)

    print()
    print("1. SOURCES CITED:")
    for index, source in enumerate(
        state.gap.cited_sources,
        start=1,
    ):
        print(
            f"{index}. Model={source.model}; URL={source.url}; "
            f"Domain={source.domain}; "
            f"Source verification={source.verification_status}; "
            f"Competitor verification={source.competitor_verification_status}"
        )
        if source.verified_url:
            print("   Fetched URL:", source.verified_url)
        if source.verification_note:
            print("   Note:", source.verification_note)
        for evidence_quote in source.verification_evidence:
            print("   Evidence:", evidence_quote)
        for claim in source.verified_claims:
            print(
                f"   Claim [{claim.status}]: {claim.claim}"
            )

    print()
    print("2. COMPETITORS REPRESENTED:")
    for source in state.gap.cited_sources:
        print(
            f"- Verified: {source.competitor or 'Not identified'}; "
            f"Candidate: {source.candidate_competitor or 'Not identified'} "
            f"(source: {source.url or source.domain or 'unknown'})"
        )

    print()
    print("3. SOURCE CONTRIBUTIONS TO THE ANSWERS:")
    for source in state.gap.cited_sources:
        print(
            f"- {source.url or source.domain or 'Unknown source'} "
            f"[{source.model}]: {source.contribution}"
        )

    print()
    print("4. VERIFIED THEMES:")
    for theme in state.gap.themes:
        print("-", theme)

    print()
    print("CANDIDATE THEMES:")
    for theme in state.gap.candidate_themes:
        print("-", theme)

    print()
    print("THEME EVIDENCE:")
    for evidence in state.gap.theme_evidence:
        print(
            f"- {evidence.theme} [{evidence.source_id}] "
            f"{evidence.source_url}: {evidence.evidence_quote}"
        )

    print()
    print("THEME ASSESSMENTS:")
    for assessment in state.gap.theme_assessments:
        print(
            f"- {assessment.theme}: {assessment.status}; "
            f"verified AI usage={', '.join(assessment.cited_by) or 'none'}; "
            f"verified competitors={', '.join(assessment.competitors) or 'none'}; "
            f"candidate AI usage={', '.join(assessment.candidate_cited_by) or 'none'}; "
            f"candidate competitors={', '.join(assessment.candidate_competitors) or 'none'}; "
            f"target-site coverage={assessment.target_site_coverage}; "
            f"content-gap status={assessment.content_gap_status}; "
            f"opportunity={assessment.opportunity}"
        )

    print()
    print("CANDIDATE ANGLE:")
    print(state.gap.candidate_target_angle)

    print()
    print("5. CONTENT ANGLE:")
    print(state.gap.target_angle)

    print()
    print("HYPOTHESIS:")
    print(state.gap.hypothesis)


if __name__ == "__main__":
    asyncio.run(main())