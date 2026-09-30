"""
ACA Source Analyzer.

Responsibilities
----------------
1. Read underperforming AImpact prompt evidence.
2. Analyze cited sources with Claude Haiku.
3. Fetch cited source pages through the SourceFetcher abstraction.
4. Verify candidate claims and candidate themes against fetched page text.
5. Deterministically validate evidence quotes.
6. Verify candidate competitors conservatively.
7. Produce an evidence-based ContentGap.

Important boundary
------------------
This agent does NOT establish that the target company's website has
a content gap.

Target-site coverage remains "unknown" unless a separate target-site
content inventory is supplied and analyzed.

The Source Analyzer answers:

    "What does the AI-visible source evidence tell us?"

It does not answer:

    "What exact article should we write?"

That is the Brief Agent's responsibility.
"""

from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from aca.orchestrator.state import (
    AnalyzedSource,
    ContentGap,
    CitedSource,
    PromptEvidence,
    SourceClaimAssessment,
    ThemeAssessment,
    ThemeEvidence,
    WorkflowState,
)
from aca.tools.source_fetcher import FetchedPage, get_source_fetcher

load_dotenv()


# ---------------------------------------------------------------------------
# LLM output schemas
# ---------------------------------------------------------------------------


class SourceAnalysis(BaseModel):
    """First-pass analysis of one cited source."""

    source_id: str

    competitor: str | None = None

    contribution: str

    claims_to_verify: list[str] = Field(
        default_factory=list,
        max_length=3,
    )

    themes: list[str] = Field(
        default_factory=list,
        max_length=5,
    )


class SourceAnalyzerOutput(BaseModel):
    """First-pass Source Analyzer output."""

    source_analyses: list[SourceAnalysis] = Field(
        default_factory=list,
        max_length=100,
    )

    themes: list[str] = Field(
        default_factory=list,
        max_length=20,
    )

    visibility_gap: str

    editorial_angle: str

    hypothesis: str


class ClaimCheck(BaseModel):
    """Verification result for one candidate claim."""

    source_id: str

    claim: str

    verdict: Literal[
        "supported",
        "contradicted",
        "not_found",
        "inconclusive",
        "unavailable",
    ]

    evidence_quote: str | None = None


class ThemeCheck(BaseModel):
    """Verification result for one candidate theme."""

    source_id: str

    theme: str

    verdict: Literal[
        "supported",
        "contradicted",
        "not_found",
        "inconclusive",
        "unavailable",
    ]

    evidence_quote: str | None = None


class SourceVerificationOutput(BaseModel):
    """Second-pass verification output."""

    claim_checks: list[ClaimCheck] = Field(
        default_factory=list,
        max_length=100,
    )

    theme_checks: list[ThemeCheck] = Field(
        default_factory=list,
        max_length=100,
    )


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _normalize(value: str) -> str:
    """
    Normalize text for conservative matching.

    This is intentionally simple. We do not use fuzzy semantic matching
    here because verification should be deterministic.
    """

    value = value.lower().strip()

    value = re.sub(
        r"[^a-z0-9\s-]",
        "",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def _theme_key(theme: str) -> str:
    """
    Conservative theme canonicalization.

    Only collapses obvious wording variants. It does not attempt
    semantic clustering because that could incorrectly merge
    distinct research concepts.
    """

    normalized = _normalize(theme)

    replacements = (
        (" platforms and technologies", " platforms"),
        (" platform and technologies", " platform"),
        (" technologies and platforms", " platforms"),
        (" technologies", " technology"),
    )

    for old, new in replacements:
        if normalized.endswith(old):
            normalized = normalized[: -len(old)] + new

    return normalized.strip()


def _dedupe_themes(themes: list[str]) -> list[str]:
    """
    Preserve the first human-readable form of each normalized theme.
    """

    seen: set[str] = set()
    result: list[str] = []

    for theme in themes:
        theme = theme.strip()

        if not theme:
            continue

        key = _theme_key(theme)

        if not key or key in seen:
            continue

        seen.add(key)
        result.append(theme)

    return result


def _select_candidate_themes(
    *,
    global_themes: list[str],
    analyses: list[SourceAnalysis],
    limit: int = 8,
) -> list[str]:
    """Select a compact, source-backed theme shortlist for verification."""

    source_counts: dict[str, int] = {}
    candidates: list[str] = []

    for analysis in analyses:
        source_keys = {
            _theme_key(theme)
            for theme in analysis.themes
            if _theme_key(theme)
        }
        for key in source_keys:
            source_counts[key] = source_counts.get(key, 0) + 1
        candidates.extend(analysis.themes)

    candidates = _dedupe_themes(candidates)
    global_rank = {
        _theme_key(theme): index
        for index, theme in enumerate(global_themes)
    }
    original_rank = {
        _theme_key(theme): index
        for index, theme in enumerate(candidates)
    }

    candidates.sort(
        key=lambda theme: (
            -source_counts.get(_theme_key(theme), 0),
            global_rank.get(_theme_key(theme), len(global_rank)),
            original_rank[_theme_key(theme)],
        )
    )

    return candidates[:limit]


def _build_hypothesis(
    *,
    company_name: str,
    verified_themes: list[str],
    candidate_themes: list[str],
) -> str:
    if verified_themes:
        citation_observation = (
            "The sampled AI responses cite pages with verified coverage of: "
            + ", ".join(verified_themes)
            + "."
        )
    elif candidate_themes:
        citation_observation = (
            "The cited sources proposed these topics for investigation: "
            + ", ".join(candidate_themes)
            + "; none met the page-verification threshold."
        )
    else:
        citation_observation = (
            "The cited-source evidence did not establish a verified theme."
        )

    return (
        f"{citation_observation} This describes the sampled citation "
        f"set only. {company_name}'s actual site coverage has not been "
        "assessed, so the absence of the company from these citations "
        "is not evidence of missing website content or a content gap."
    )


def _quote_is_present(
    quote: str | None,
    page_text: str,
) -> bool:
    """
    Deterministically verify that the LLM's evidence quote actually
    occurs in fetched page text.

    Matching is whitespace-normalized and case-insensitive.
    """

    if not quote or not quote.strip():
        return False

    normalized_quote = _normalize_whitespace(quote)
    normalized_page = _normalize_whitespace(page_text)

    if not normalized_quote:
        return False

    return normalized_quote.lower() in normalized_page.lower()


def _normalize_whitespace(value: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def _safe_domain(url: str | None) -> str | None:
    if not url:
        return None

    try:
        hostname = urlparse(url).hostname
    except ValueError:
        return None

    if not hostname:
        return None

    return hostname.lower()


# ---------------------------------------------------------------------------
# Competitor verification
# ---------------------------------------------------------------------------


def _competitor_appears_in_page(
    candidate: str,
    page: FetchedPage,
) -> bool:
    """
    Conservative competitor verification.

    The candidate must actually appear in the fetched page content
    or in the fetched URL/domain.

    We intentionally do NOT maintain a large alias map because aliases
    can silently turn brands into companies or parent companies.
    """

    if not candidate:
        return False

    candidate_normalized = _normalize(candidate)

    if not candidate_normalized:
        return False

    page_content = _normalize(
        page.content,
    )

    page_url = _normalize(
        page.url,
    )

    if candidate_normalized in page_content:
        return True

    # Domain check is deliberately conservative.
    domain = _safe_domain(page.url)

    if domain:
        domain_normalized = _normalize(domain)

        candidate_tokens = [
            token
            for token in candidate_normalized.split()
            if len(token) >= 4
        ]

        if candidate_tokens and all(
            token in domain_normalized
            for token in candidate_tokens
        ):
            return True

    if candidate_normalized in page_url:
        return True

    return False


def _verify_competitor(
    candidate: str | None,
    page: FetchedPage | None,
    company_name: str,
    relevant: bool,
) -> tuple[
    str | None,
    Literal[
        "verified",
        "unverified",
        "unavailable",
        "not_identified",
    ],
]:
    """
    Return the verified competitor and verification status.

    Candidate names remain separate from verified names.

    A candidate is never promoted to verified merely because:
      - Haiku suggested it
      - the source URL fetched successfully
      - the domain is generally related to the candidate
    """

    if not candidate:
        return None, "not_identified"

    candidate = candidate.strip()

    if not candidate:
        return None, "not_identified"

    # Do not classify the target company itself as a competitor.
    if _normalize(candidate) == _normalize(company_name):
        return None, "not_identified"

    if page is None:
        return None, "unavailable"

    if not page.success:
        return None, "unavailable"

    if not relevant:
        return None, "unverified"

    if _competitor_appears_in_page(
        candidate=candidate,
        page=page,
    ):
        return candidate, "verified"

    return None, "unverified"


# ---------------------------------------------------------------------------
# Source contribution
# ---------------------------------------------------------------------------


def _build_verified_contribution(
    *,
    candidate_contribution: str,
    claim_checks: list[ClaimCheck],
    page: FetchedPage | None,
) -> str:
    """
    Build source contribution from verification results.

    We avoid presenting an unverified LLM interpretation as factual.
    """

    if page is None or not page.success:
        return (
            "Source page was unavailable, so the proposed contribution "
            "could not be verified."
        )

    supported = [
        check.claim
        for check in claim_checks
        if check.verdict == "supported"
    ]

    contradicted = [
        check.claim
        for check in claim_checks
        if check.verdict == "contradicted"
    ]

    not_found = [
        check.claim
        for check in claim_checks
        if check.verdict == "not_found"
    ]

    inconclusive = [
        check.claim
        for check in claim_checks
        if check.verdict == "inconclusive"
    ]

    parts: list[str] = []

    if supported:
        parts.append(
            "Verified claims: "
            + "; ".join(supported)
        )

    if contradicted:
        parts.append(
            "Contradicted claims: "
            + "; ".join(contradicted)
        )

    if not_found:
        parts.append(
            "Claims not found: "
            + "; ".join(not_found)
        )

    if inconclusive:
        parts.append(
            "Inconclusive claims: "
            + "; ".join(inconclusive)
        )

    if parts:
        return " ".join(parts)

    return (
        "The source was fetched successfully, but no proposed claims "
        "were verified against the fetched page."
    )


# ---------------------------------------------------------------------------
# Verification status
# ---------------------------------------------------------------------------


def _overall_verification_status(
    *,
    fetched_count: int,
    total_sources: int,
    claim_checks: list[ClaimCheck],
    theme_evidence: list[ThemeEvidence],
) -> Literal[
    "verified",
    "partially_verified",
    "unverified",
    "unavailable",
]:
    if total_sources == 0:
        return "unavailable"

    if fetched_count == 0:
        return "unavailable"

    supported_claims = sum(
        1
        for check in claim_checks
        if check.verdict == "supported"
    )

    supported_themes = len(theme_evidence)

    if (
        fetched_count == total_sources
        and claim_checks
        and all(
            check.verdict == "supported"
            for check in claim_checks
        )
    ):
        return "verified"

    if supported_claims > 0 or supported_themes > 0:
        return "partially_verified"

    return "unverified"


def _verification_summary(
    *,
    total_sources: int,
    fetched_count: int,
    claim_checks: list[ClaimCheck],
    theme_checks: list[ThemeCheck],
    theme_evidence: list[ThemeEvidence],
) -> str:
    verified_claims = sum(
        1
        for check in claim_checks
        if check.verdict == "supported"
    )

    verified_themes = len(
        {
            _theme_key(evidence.theme)
            for evidence in theme_evidence
        }
    )

    candidate_theme_count = len(
        {
            _theme_key(check.theme)
            for check in theme_checks
        }
    )

    unavailable_sources = total_sources - fetched_count

    summary = (
        f"Fetched {fetched_count} of {total_sources} cited sources. "
        f"Verified {verified_claims} of {len(claim_checks)} candidate claims "
        f"and {verified_themes} of {candidate_theme_count} candidate themes "
        f"against fetched page text."
    )

    if unavailable_sources:
        summary += (
            f" {unavailable_sources} cited sources were unavailable."
        )

    summary += (
        " Target-site coverage remains unknown and no content gap "
        "has been established."
    )

    return summary


# ---------------------------------------------------------------------------
# Source Analyzer
# ---------------------------------------------------------------------------


class SourceAnalyzer:
    """
    Source-driven AI visibility analyzer.

    Uses:
        Haiku -> candidate analysis
        Tavily -> page extraction
        Haiku -> claim/theme verification
        Python -> deterministic evidence validation
    """

    def __init__(self) -> None:
        self.model = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=0,
        )

        self.fetcher = get_source_fetcher()

    # ------------------------------------------------------------------
    # First pass
    # ------------------------------------------------------------------

    def _analyze_candidates(
        self,
        *,
        company: dict,
        prompt_evidence: PromptEvidence,
        sources: list[tuple[str, CitedSource]],
    ) -> SourceAnalyzerOutput:
        source_text = []

        for source_id, source in sources:
            source_text.append(
                f"""
SOURCE {source_id}
Model: {source.model}
URL: {source.url}
Domain: {source.domain}
Existing competitor label: {source.competitor}
Excerpt:
{source.excerpt}
""".strip()
            )

        joined_sources = "\n\n".join(source_text)

        system_prompt = """
You are the Source Analyzer for an AI visibility/content system.

Your job is to analyze the evidence that AI systems cited when answering
a user's research prompt.

You are NOT a keyword researcher.

You must NOT invent:
- competitors
- claims
- URLs
- products
- capabilities
- target-site content gaps

You may identify:
- what each cited source contributes
- candidate competitors
- candidate themes
- evidence-based visibility patterns
- an editorial angle
- a hypothesis about why the cited sources are useful

Important:
A target-site content gap cannot be established from competitor/source
evidence alone.

The target company website has NOT been inspected by this agent.

Candidate themes should describe meaningful subject areas that are
actually represented in the cited material.

Keep themes concise and specific.

Examples:
- Neurology biomarker panels
- Immunology multiplex assays
- Luminex-based multiplex platforms
- Multiplex immunofluorescence
- Proximity extension assay technology

Do not create generic themes such as:
- scientific information
- product information
- research
- useful sources

For each source:
- explain its contribution to the AI answer
- propose at most 3 claims that can be checked against the source page
- propose at most 5 themes directly represented by that source
- identify a candidate competitor only when the source clearly represents
    a company's product, platform, or service that is relevant to the user's
    prompt; a passing mention or citation is not enough

Do not infer hidden model reasoning or infer website absence from missing
citations. Describe the citation set; do not claim why a model selected it.
"""

        user_prompt = f"""
TARGET COMPANY
Name: {company.get("company_name")}
Website: {company.get("company_url")}
Region: {company.get("region")}

AI PROMPT
{prompt_evidence.prompt_text}

AI MODEL EVIDENCE
{joined_sources}

Produce the structured Source Analyzer output.

The visibility_gap should describe an evidence-based AI visibility
observation, not a confirmed website content gap.

The editorial_angle should describe what a future content strategy
could investigate based on the cited evidence.

The hypothesis should clearly distinguish observed evidence from
inference.
"""

        structured = self.model.with_structured_output(
            SourceAnalyzerOutput
        )

        return structured.invoke(
            [
                (
                    "system",
                    system_prompt,
                ),
                (
                    "human",
                    user_prompt,
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Verification pass
    # ------------------------------------------------------------------

    def _verify_sources(
        self,
        *,
        prompt_evidence: PromptEvidence,
        analyses: list[SourceAnalysis],
        fetched_pages: dict[str, FetchedPage],
    ) -> SourceVerificationOutput:
        verification_items: list[str] = []

        analysis_by_source = {
            analysis.source_id: analysis
            for analysis in analyses
        }

        for source_id, analysis in analysis_by_source.items():
            page = fetched_pages.get(source_id)

            if page is None:
                verification_items.append(
                    f"""
SOURCE {source_id}
STATUS: unavailable

CANDIDATE CLAIMS:
{chr(10).join(f"- {claim}" for claim in analysis.claims_to_verify)}

CANDIDATE THEMES:
{chr(10).join(f"- {theme}" for theme in analysis.themes)}
""".strip()
                )
                continue

            verification_items.append(
                f"""
SOURCE {source_id}
URL: {page.url}
FETCH STATUS: {"success" if page.success else "failed"}

CANDIDATE CLAIMS:
{chr(10).join(f"- {claim}" for claim in analysis.claims_to_verify)}

CANDIDATE THEMES:
{chr(10).join(f"- {theme}" for theme in analysis.themes)}

FETCHED PAGE TEXT:
{page.content[:30000]}
""".strip()
            )

        joined = "\n\n".join(verification_items)

        system_prompt = """
You are the verification stage of an evidence-grounded Source Analyzer.

Verify ONLY the supplied candidate claims and candidate themes.

Do not add new claims.
Do not add new themes.
Do not use outside knowledge.
Do not infer that a theme is supported merely because one word from
the theme appears on the page.

CLAIM VERIFICATION
------------------
supported:
    The page directly supports the claim.

contradicted:
    The page directly conflicts with the claim.

not_found:
    The page was fetched successfully but the claim could not be found.

inconclusive:
    The page contains related material but does not establish the
    supplied claim clearly enough.

unavailable:
    The page could not be fetched.

THEME VERIFICATION
------------------
A theme is supported only when the fetched page directly represents
that subject as a meaningful part of its content.

For example, a page discussing:
    "multiplex immunoassays for neurobiology and neuroscience research"

can support:
    "Neurology biomarker panels"

if the evidence clearly describes those applications.

But a generic mention of "technology" does not support:
    "Luminex-based multiplex platforms"

unless the page actually discusses that platform/application.

EVIDENCE QUOTES
---------------
For every supported or contradicted claim/theme, provide a SHORT,
VERBATIM quote copied from the fetched page.

Never paraphrase the quote.

For not_found and inconclusive:
    evidence_quote may be null.

For unavailable:
    evidence_quote must be null.

Do not invent evidence.
"""

        user_prompt = f"""
PROMPT
{prompt_evidence.prompt_text}

SOURCE VERIFICATION MATERIAL
{joined}

Return structured claim checks and theme checks.
"""

        structured = self.model.with_structured_output(
            SourceVerificationOutput
        )

        return structured.invoke(
            [
                (
                    "system",
                    system_prompt,
                ),
                (
                    "human",
                    user_prompt,
                ),
            ]
        )

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        *,
        project: dict,
        prompt_evidence: PromptEvidence,
    ) -> ContentGap:
        """
        Analyze one underperforming prompt.
        """

        company_name = (
            project.get("company_name")
            or "Unknown company"
        )

        # --------------------------------------------------------------
        # Flatten all cited sources into stable S1/S2/S3 IDs.
        # --------------------------------------------------------------

        sources: list[tuple[str, CitedSource]] = []

        source_counter = 1

        for model_name, model_evidence in prompt_evidence.per_model.items():
            for source in model_evidence.sources:
                source_id = f"S{source_counter}"

                sources.append(
                    (
                        source_id,
                        source.model or model_name,
                    )
                    if False
                    else (
                        source_id,
                        source,
                    )
                )

                source_counter += 1

        # --------------------------------------------------------------
        # No cited sources.
        # --------------------------------------------------------------

        if not sources:
            return ContentGap(
                prompt_id=prompt_evidence.prompt_id,
                prompt_text=prompt_evidence.prompt_text,
                cited_sources=[],
                candidate_themes=[],
                themes=[],
                candidate_target_angle="",
                theme_evidence=[],
                theme_assessments=[],
                verification_status="unavailable",
                verification_summary=(
                    "No cited sources were available for this prompt. "
                    "Target-site coverage remains unknown and no content "
                    "gap has been established."
                ),
                target_site_coverage="unknown",
                content_gap_status="not_established",
                opportunity=(
                    "No source evidence was available to identify an "
                    "AI-visible content opportunity."
                ),
                target_angle=(
                    "No target content angle can be established "
                    "without source evidence."
                ),
                hypothesis=(
                    "No hypothesis can be established because the "
                    "prompt contained no cited sources."
                ),
            )

        # --------------------------------------------------------------
        # First-pass Haiku analysis.
        # --------------------------------------------------------------

        first_pass = self._analyze_candidates(
            company=project,
            prompt_evidence=prompt_evidence,
            sources=sources,
        )

        candidate_themes = _select_candidate_themes(
            global_themes=first_pass.themes,
            analyses=first_pass.source_analyses,
        )
        selected_theme_keys = {
            _theme_key(theme)
            for theme in candidate_themes
        }
        for analysis in first_pass.source_analyses:
            analysis.themes = [
                theme
                for theme in analysis.themes
                if _theme_key(theme) in selected_theme_keys
            ]

        # --------------------------------------------------------------
        # Fetch every cited source.
        # --------------------------------------------------------------

        source_url_map: dict[str, str] = {
            source_id: source.url
            for source_id, source in sources
            if source.url
        }

        fetched_pages = self.fetcher.fetch_pages(
            source_url_map
        )

        # --------------------------------------------------------------
        # Second-pass verification.
        # --------------------------------------------------------------

        verification = self._verify_sources(
            prompt_evidence=prompt_evidence,
            analyses=first_pass.source_analyses,
            fetched_pages=fetched_pages,
        )

        analyses_by_source = {
            analysis.source_id: analysis
            for analysis in first_pass.source_analyses
        }

        source_by_id = {
            source_id: source
            for source_id, source in sources
        }

        # --------------------------------------------------------------
        # Index verification results.
        # --------------------------------------------------------------

        claims_by_source: dict[str, list[ClaimCheck]] = {}

        for check in verification.claim_checks:
            claims_by_source.setdefault(
                check.source_id,
                [],
            ).append(check)

        theme_checks_by_source: dict[str, list[ThemeCheck]] = {}

        for check in verification.theme_checks:
            theme_checks_by_source.setdefault(
                check.source_id,
                [],
            ).append(check)

        # --------------------------------------------------------------
        # Deterministically validate theme evidence.
        # --------------------------------------------------------------

        verified_theme_evidence: list[ThemeEvidence] = []

        verified_theme_keys: set[str] = set()

        for check in verification.theme_checks:
            source_id = check.source_id

            source = source_by_id.get(source_id)
            analysis = analyses_by_source.get(source_id)
            page = fetched_pages.get(source_id)

            if source is None:
                continue

            if analysis is None:
                continue

            if page is None or not page.success:
                continue

            if check.verdict != "supported":
                continue

            if not check.evidence_quote:
                continue

            if not _quote_is_present(
                check.evidence_quote,
                page.content,
            ):
                continue

            # The theme must have been proposed for this source.
            source_theme_keys = {
                _theme_key(theme)
                for theme in analysis.themes
            }

            if _theme_key(check.theme) not in source_theme_keys:
                continue

            theme_key = _theme_key(check.theme)

            if theme_key in verified_theme_keys:
                # Keep multiple pieces of evidence even when the theme
                # is already verified.
                pass
            else:
                verified_theme_keys.add(theme_key)

            verified_theme_evidence.append(
                ThemeEvidence(
                    theme=check.theme.strip(),
                    source_id=source_id,
                    source_url=page.url,
                    evidence_quote=check.evidence_quote.strip(),
                )
            )

        # --------------------------------------------------------------
        # Canonical verified theme names.
        #
        # Preserve the first candidate spelling rather than allowing
        # multiple LLM variants to become separate themes.
        # --------------------------------------------------------------

        canonical_theme_name: dict[str, str] = {}

        for theme in candidate_themes:
            canonical_theme_name.setdefault(
                _theme_key(theme),
                theme,
            )

        verified_theme_names: list[str] = []

        for evidence in verified_theme_evidence:
            key = _theme_key(evidence.theme)

            canonical = canonical_theme_name.get(
                key,
                evidence.theme,
            )

            if key not in {
                _theme_key(theme)
                for theme in verified_theme_names
            }:
                verified_theme_names.append(canonical)

        assessed_theme_names = candidate_themes
        verified_theme_keys = {
            _theme_key(theme)
            for theme in verified_theme_names
        }
        candidate_themes = [
            theme
            for theme in assessed_theme_names
            if _theme_key(theme) not in verified_theme_keys
        ]

        # --------------------------------------------------------------
        # Theme assessments.
        # --------------------------------------------------------------

        theme_assessments: list[ThemeAssessment] = []

        for theme in assessed_theme_names:
            theme_key = _theme_key(theme)

            theme_source_ids: set[str] = set()
            theme_cited_by: set[str] = set()
            theme_candidate_competitors: set[str] = set()
            theme_verified_competitors: set[str] = set()
            theme_evidence: list[ThemeEvidence] = []

            for source_id, analysis in analyses_by_source.items():
                source_theme_keys = {
                    _theme_key(candidate)
                    for candidate in analysis.themes
                }

                if theme_key not in source_theme_keys:
                    continue

                source = source_by_id.get(source_id)

                if source is None:
                    continue

                theme_source_ids.add(
                    source_id
                )

                theme_cited_by.add(
                    source.model
                )

                if analysis.competitor:
                    theme_candidate_competitors.add(
                        analysis.competitor
                    )

                for evidence in verified_theme_evidence:
                    if (
                        evidence.source_id == source_id
                        and _theme_key(evidence.theme) == theme_key
                    ):
                        theme_evidence.append(
                            evidence
                        )

            # Verified competitors represented by sources supporting
            # this theme.
            for source_id in theme_source_ids:
                analysis = analyses_by_source.get(source_id)
                page = fetched_pages.get(source_id)

                if analysis is None:
                    continue

                competitor, status = _verify_competitor(
                    candidate=analysis.competitor,
                    page=page,
                    company_name=company_name,
                    relevant=any(
                        evidence.source_id == source_id
                        for evidence in theme_evidence
                    ),
                )

                if (
                    status == "verified"
                    and competitor
                ):
                    theme_verified_competitors.add(
                        competitor
                    )

            # Candidate competitors can come from all candidate
            # analyses associated with the theme.
            verified = bool(theme_evidence)

            theme_assessments.append(
                ThemeAssessment(
                    theme=theme,
                    status=(
                        "verified"
                        if verified
                        else "unverified"
                    ),
                    candidate_source_ids=sorted(
                        theme_source_ids
                    ),
                    candidate_cited_by=sorted(
                        theme_cited_by
                    ),
                    candidate_competitors=sorted(
                        theme_candidate_competitors
                    ),
                    source_ids=sorted(
                        {
                            evidence.source_id
                            for evidence in theme_evidence
                        }
                    ),
                    cited_by=sorted(
                        {
                            source_by_id[evidence.source_id].model
                            for evidence in theme_evidence
                            if evidence.source_id in source_by_id
                        }
                    ),
                    competitors=sorted(
                        theme_verified_competitors
                    ),
                    evidence=theme_evidence,
                    target_site_coverage="unknown",
                    content_gap_status="not_established",
                    opportunity=(
                        f"Investigate whether {company_name} already "
                        f"has content that clearly covers {theme}. "
                        "Target-site coverage has not been assessed."
                    ),
                )
            )

        # --------------------------------------------------------------
        # Analyze each cited source into AnalyzedSource.
        # --------------------------------------------------------------

        analyzed_sources: list[AnalyzedSource] = []

        for source_id, source in sources:
            analysis = analyses_by_source.get(
                source_id
            )

            page = fetched_pages.get(
                source_id
            )

            source_claim_checks = claims_by_source.get(
                source_id,
                [],
            )

            candidate_competitor = (
                analysis.competitor
                if analysis
                else None
            )

            verified_competitor, competitor_status = (
                _verify_competitor(
                    candidate=candidate_competitor,
                    page=page,
                    company_name=company_name,
                    relevant=any(
                        evidence.source_id == source_id
                        for evidence in verified_theme_evidence
                    ),
                )
            )

            # Build SourceClaimAssessment objects only from the
            # verification output.
            source_claims: list[
                SourceClaimAssessment
            ] = []

            for check in source_claim_checks:
                source_claims.append(
                    SourceClaimAssessment(
                        claim=check.claim,
                        status=(
                            "verified"
                            if check.verdict == "supported"
                            else check.verdict
                        ),
                        evidence_quote=check.evidence_quote,
                    )
)
            supported_claims = [
                check
                for check in source_claim_checks
                if (
                    check.verdict == "supported"
                    and check.evidence_quote
                    and page is not None
                    and page.success
                    and _quote_is_present(
                        check.evidence_quote,
                        page.content,
                    )
                )
            ]

            contradicted_claims = [
                check
                for check in source_claim_checks
                if check.verdict == "contradicted"
            ]

            not_found_claims = [
                check
                for check in source_claim_checks
                if check.verdict == "not_found"
            ]

            if page is None or not page.success:
                source_status: Literal[
                    "verified",
                    "partially_verified",
                    "unsupported",
                    "unverified",
                    "unavailable",
                ] = "unavailable"

                verification_note = (
                    page.error
                    if page is not None
                    else "Source was not fetched."
                )

            elif supported_claims and not (
                contradicted_claims
                or not_found_claims
            ):
                source_status = "verified"
                verification_note = None

            elif supported_claims:
                source_status = "partially_verified"
                verification_note = (
                    "Some proposed claims were verified, "
                    "while others were not fully supported."
                )

            elif source_claim_checks:
                source_status = "unsupported"
                verification_note = (
                    "The fetched page did not provide sufficient "
                    "evidence for the proposed claims."
                )

            else:
                source_status = "unverified"
                verification_note = (
                    "No claims were proposed for verification."
                )

            contribution = _build_verified_contribution(
                candidate_contribution=(
                    analysis.contribution
                    if analysis
                    else ""
                ),
                claim_checks=source_claim_checks,
                page=page,
            )

            analyzed_sources.append(
                AnalyzedSource(
                    source_id=source_id,
                    model=source.model,
                    url=source.url,
                    domain=source.domain,
                    competitor=verified_competitor,
                    excerpt=source.excerpt,
                    contribution=contribution,
                    candidate_competitor=candidate_competitor,
                    competitor_verification_status=competitor_status,
                    verified_claims=source_claims,
                    verification_status=source_status,
                    verification_evidence=[
                        check.evidence_quote
                        for check in supported_claims
                        if check.evidence_quote
                    ],
                    verified_url=(
                        page.url
                        if page is not None
                        and page.success
                        else None
                    ),
                    verification_note=verification_note,
                )
            )

        # --------------------------------------------------------------
        # Overall status.
        # --------------------------------------------------------------

        fetched_count = sum(
            1
            for page in fetched_pages.values()
            if page.success
        )

        verification_status = _overall_verification_status(
            fetched_count=fetched_count,
            total_sources=len(sources),
            claim_checks=verification.claim_checks,
            theme_evidence=verified_theme_evidence,
        )

        verification_summary = _verification_summary(
            total_sources=len(sources),
            fetched_count=fetched_count,
            claim_checks=verification.claim_checks,
            theme_checks=verification.theme_checks,
            theme_evidence=verified_theme_evidence,
        )

        # --------------------------------------------------------------
        # Candidate opportunity.
        #
        # Important: this is NOT a confirmed content gap.
        # --------------------------------------------------------------

        if verified_theme_names:
            opportunity = (
                "Investigate the verified source themes against the "
                f"{company_name} site inventory before establishing "
                "a content gap. Verified themes include: "
                + ", ".join(
                    verified_theme_names
                )
                + "."
            )
        elif candidate_themes:
            opportunity = (
                "Investigate the candidate source themes against the "
                f"{company_name} site inventory before establishing "
                "a content gap. No candidate theme has yet met the "
                "verification threshold."
            )
        else:
            opportunity = (
                "No sufficiently grounded source themes were identified. "
                "Target-site coverage remains unknown."
            )

        # --------------------------------------------------------------
        # Content angle.
        # --------------------------------------------------------------

        if verified_theme_names:
            target_angle = (
                f"Compare {company_name}'s existing coverage of "
                + ", ".join(
                    verified_theme_names
                )
                + " against the verified source evidence. "
                "Do not treat this as a content gap until the "
                "target-site inventory has been checked."
            )
        elif candidate_themes:
            target_angle = (
                f"Investigate {company_name}'s existing coverage of "
                + ", ".join(candidate_themes)
                + " against the cited source evidence. "
                "No content gap is established."
            )
        else:
            target_angle = (
                "No target content angle can be established until "
                "relevant source evidence is available."
            )

        # --------------------------------------------------------------
        # Final ContentGap.
        # --------------------------------------------------------------

        hypothesis = _build_hypothesis(
            company_name=company_name,
            verified_themes=verified_theme_names,
            candidate_themes=(
                candidate_themes or assessed_theme_names
            ),
        )

        return ContentGap(
            prompt_id=prompt_evidence.prompt_id,
            prompt_text=prompt_evidence.prompt_text,
            cited_sources=analyzed_sources,
            candidate_themes=candidate_themes,
            themes=verified_theme_names,
            candidate_target_angle=first_pass.editorial_angle,
            theme_evidence=verified_theme_evidence,
            theme_assessments=theme_assessments,
            verification_status=verification_status,
            verification_summary=verification_summary,
            target_site_coverage="unknown",
            content_gap_status="not_established",
            opportunity=opportunity,
            target_angle=target_angle,
            hypothesis=hypothesis,
        )


# ---------------------------------------------------------------------------
# LangGraph/module entrypoint
# ---------------------------------------------------------------------------


async def run(
    state: WorkflowState,
) -> WorkflowState:
    """
    Run Source Analyzer for the first prompt in WorkflowState.

    The production orchestrator can later invoke this node once per
    selected prompt or provide a prompt-specific state.
    """

    if not state.prompts:
        raise RuntimeError(
            "Source Analyzer cannot run because no prompt evidence "
            f"was supplied for campaign {state.campaign_id}."
        )

    prompt = state.prompts[0]

    # Development adapter only.
    #
    # Production ACA should replace this with the AImpact REST client.
    from aca.tools.aimpact import (
        get_campaign,
        get_project,
    )

    campaign = get_campaign(
        state.campaign_id
    )

    project = get_project(
        campaign["project_id"]
    )

    analyzer = SourceAnalyzer()

    state.gap = analyzer.analyze(
        project=project,
        prompt_evidence=prompt,
    )

    return state