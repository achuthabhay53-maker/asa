import asyncio
from urllib.parse import urlsplit

from aca.agents.analyzer import run
from aca.orchestrator.state import WorkflowState
from aca.tools.aimpact import (
    get_source_analyzer_input,
)


def _section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _display_url(url: str | None) -> str:
    if not url:
        return "URL unavailable"

    try:
        parsed = urlsplit(url)
    except ValueError:
        return url[:100] + ("..." if len(url) > 100 else "")

    hostname = parsed.hostname or ""
    if (
        hostname.endswith("vertexaisearch.cloud.google.com")
        and "grounding-api-redirect" in parsed.path
    ):
        return "Google Search grounding redirect (opaque citation URL)"

    visible_url = f"{hostname}{parsed.path}"
    if parsed.query:
        visible_url += "?..."

    if len(visible_url) > 110:
        visible_url = visible_url[:107] + "..."

    return visible_url or url[:110]


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

    _section("SOURCE ANALYZER | INPUT")
    print(
        f"Project: {project.get('company_name', 'Unknown company')}  |  "
        f"Region: {project.get('region', 'Unknown')}"
    )
    print(
        f"Campaign: {campaign_id}  |  Score threshold: {score_threshold}  |  "
        f"Underperforming prompts: {len(prompts)}"
    )

    for prompt in prompts:
        models = ", ".join(prompt.per_model.keys()) or "none"
        print(
            f"  {prompt.prompt_id}: {prompt.prompt_text} "
            f"(models: {models})"
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

    _section(f"ANALYZING PROMPT {target_prompt.prompt_id}")
    print(target_prompt.prompt_text)

    state = await run(state)

    if state.gap is None:
        raise RuntimeError(
            "Source Analyzer did not produce a ContentGap."
        )

    gap = state.gap
    assessments = {
        assessment.theme: assessment
        for assessment in gap.theme_assessments
    }

    _section("ANALYSIS SUMMARY")
    print(f"Prompt: {gap.prompt_id} | {gap.prompt_text}")
    print(f"Verification: {gap.verification_status}")
    print(gap.verification_summary)
    print(
        f"Target-site coverage: {gap.target_site_coverage}  |  "
        f"Content gap: {gap.content_gap_status}"
    )

    _section(f"VERIFIED THEMES ({len(gap.themes)})")
    if not gap.themes:
        print("No themes met the evidence threshold.")
    for theme in gap.themes:
        assessment = assessments.get(theme)
        print(f"\n{theme}")
        if assessment is not None:
            print(
                f"  Sources: {', '.join(assessment.source_ids) or 'none'}  |  "
                f"Cited by: {', '.join(assessment.cited_by) or 'none'}  |  "
                f"Competitors: {', '.join(assessment.competitors) or 'none'}"
            )
            for evidence in assessment.evidence:
                print(
                    f"  Evidence [{evidence.source_id}] "
                    f"({_display_url(evidence.source_url)}):"
                )
                print(f'    "{evidence.evidence_quote}"')

    _section(f"CANDIDATE THEMES ({len(gap.candidate_themes)})")
    if not gap.candidate_themes:
        print("None.")
    for theme in gap.candidate_themes:
        assessment = assessments.get(theme)
        if assessment is None:
            print(f"- {theme}")
            continue
        print(
            f"- {theme} | sources: "
            f"{', '.join(assessment.candidate_source_ids) or 'none'} | "
            f"cited by: {', '.join(assessment.candidate_cited_by) or 'none'}"
        )

    _section(f"CITED SOURCES ({len(gap.cited_sources)})")
    for source in gap.cited_sources:
        source_id = source.source_id or "?"
        print(
            f"\n[{source_id}] {source.model} | "
            f"page: {source.verification_status} | "
            f"competitor: {source.competitor or 'not verified'}"
        )
        print(f"  URL: {_display_url(source.verified_url or source.url)}")
        if source.candidate_competitor:
            print(f"  Competitor candidate: {source.candidate_competitor}")
        if source.verification_note:
            print(f"  Note: {source.verification_note}")
        for claim in source.verified_claims:
            print(f"  Claim [{claim.status}]: {claim.claim}")
            if claim.evidence_quote:
                print(f'    Evidence: "{claim.evidence_quote}"')

    _section("OPPORTUNITY AND INTERPRETATION")
    print("Opportunity:")
    print(f"  {gap.opportunity}")
    print("Content angle:")
    print(f"  {gap.target_angle}")
    print("Candidate angle:")
    print(f"  {gap.candidate_target_angle or 'Not provided.'}")
    print("Hypothesis:")
    print(f"  {gap.hypothesis}")


if __name__ == "__main__":
    asyncio.run(main())