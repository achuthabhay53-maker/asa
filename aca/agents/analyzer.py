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

import os
import re
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field, field_validator

from aca.orchestrator.state import (
    AnalyzedSource,
    ContentGap,
    CitedSource,
    PromptEvidence,
    SourceClaimAssessment,
    TargetSiteDiagnostic,
    TargetSiteEvidence,
    ThemeAssessment,
    ThemeEvidence,
    WorkflowState,
)
from aca.tools.source_fetcher import (
    FetchedPage,
    SiteSearchCandidates,
    get_source_fetcher,
)

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

    # Specific, checkable target-site coverage questions.
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

    @field_validator("theme", mode="before")
    @classmethod
    def _empty_missing_theme(cls, value):
        return value.strip() if isinstance(value, str) else ""


class SourceVerificationOutput(BaseModel):
    """Second-pass verification output."""

    claim_checks: list[ClaimCheck] = Field(
        default_factory=list,
        max_length=100,
    )

    theme_checks: list[ThemeCheck] = Field(
        default_factory=list,
        max_length=800,
    )


class TargetSiteVerificationOutput(BaseModel):
    """Theme coverage checks against target-site pages."""

    theme_checks: list[ThemeCheck] = Field(
        default_factory=list,
        max_length=800,
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
    target_site_observation: str | None = None,
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

    site_observation = target_site_observation or (
        f"{company_name}'s actual site coverage has not been assessed."
    )

    return (
        f"{citation_observation} This describes the sampled citation "
        f"set only. {site_observation} The absence of the company from "
        "the citations alone is not evidence of missing website content."
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


def _claim_quote_is_valid(
    check: ClaimCheck,
    page: FetchedPage | None,
) -> bool:
    if not (
        check.verdict == "supported"
        and check.evidence_quote
        and page is not None
        and page.success
        and _quote_is_present(
            check.evidence_quote,
            page.content,
        )
    ):
        return False

    claim = _normalize(check.claim)
    quote = _normalize(check.evidence_quote)
    claim_is_normative = any(
        term in claim
        for term in (
            "standard",
            "standards",
            "guideline",
            "guidelines",
            "recommendation",
            "recommendations",
            "requirement",
            "requirements",
        )
    )
    quote_only_states_aim = any(
        phrase in quote
        for phrase in (
            "aim of this review",
            "aim of the review",
            "objective of this review",
            "objective of the review",
            "purpose of this review",
            "purpose of the review",
            "aim of this study",
            "objective of this study",
            "purpose of this study",
        )
    )
    if claim_is_normative and quote_only_states_aim:
        return False

    named_entities = (
        "thermo fisher scientific",
        "thermo fisher",
        "sigma aldrich",
        "milliplex",
        "bio techne",
        "r&d systems",
        "rd systems",
        "bio rad",
        "bio plex",
        "meso scale discovery",
        "mesoscale",
        "raybiotech",
        "creative proteomics",
        "creative biomart",
        "olink",
        "luminex",
        "xmap",
        "u plex",
        "s plex",
        "nulisa",
        "nulisa seq",
        "abcam",
    )
    for entity in named_entities:
        normalized_entity = _normalize(entity)
        if normalized_entity in claim and normalized_entity not in quote:
            return False

    return True


def _classify_target_fetch_failure(
    page: FetchedPage,
) -> str:
    error = (page.error or "").lower()
    path = urlparse(page.url).path.lower()

    if any(
        marker in error
        for marker in (
            "robots.txt",
            "blocked by robots",
            "robots exclusion",
            "forbidden",
            "access denied",
            "status code 401",
            "status code 403",
            "http 401",
            "http 403",
        )
    ):
        return "robots/access restriction"

    if any(
        marker in error
        for marker in (
            "unsupported content type",
            "application/pdf",
            "image/",
            "rss feed",
            "atom feed",
            "not html",
        )
    ) or path.endswith(
        (
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".webp",
            ".svg",
            ".rss",
            ".atom",
            ".xml",
            ".mp4",
        )
    ):
        return "non-HTML content"

    if any(
        marker in error
        for marker in (
            "redirect",
            "timed out",
            "timeout",
            "connection",
            "dns",
            "tls",
            "ssl",
            "status code 3",
            "status code 4",
            "status code 5",
            "http 3",
            "http 4",
            "http 5",
        )
    ):
        return "HTTP/redirect failure"

    if page.failure_stage == "tavily_batch":
        return "Tavily batch/extraction behavior"

    if page.failure_stage in {
        "tavily_failed_result",
        "tavily_empty_content",
    }:
        return "Tavily extraction failure"

    return "other error"


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


_KNOWN_DOMAIN_ENTITIES = {
    "thermofisher.com": "Thermo Fisher Scientific",
    "sigmaaldrich.com": "Sigma-Aldrich",
    "bio-techne.com": "Bio-Techne",
    "rndsystems.com": "Bio-Techne",
    "biorad.com": "Bio-Rad",
    "mesoscale.com": "Meso Scale Discovery",
    "meso-scale.com": "Meso Scale Discovery",
    "raybiotech.com": "RayBiotech",
    "creative-proteomics.com": "Creative Proteomics",
    "olink.com": "Olink",
    "abcam.com": "Abcam",
    "evotec.com": "Evotec",
}


def _entity_from_domain(url: str | None) -> str | None:
    domain = _safe_domain(url)
    if not domain:
        return None

    return next(
        (
            entity
            for known_domain, entity in _KNOWN_DOMAIN_ENTITIES.items()
            if domain == known_domain or domain.endswith(f".{known_domain}")
        ),
        None,
    )


def _classify_source_type(
    *,
    url: str | None,
    title: str | None = None,
) -> Literal["commercial", "academic", "educational", "unknown"]:
    domain = _safe_domain(url) or ""
    path = urlparse(url or "").path.lower()
    normalized_title = _normalize(title or "")

    if (
        domain.endswith(".edu")
        or domain.endswith(".gov")
        or domain in {
            "pubmed.ncbi.nlm.nih.gov",
            "ncbi.nlm.nih.gov",
            "pmc.ncbi.nlm.nih.gov",
            "arxiv.org",
            "doi.org",
            "frontiersin.org",
            "nature.com",
            "jci.org",
        }
        or "/pmc/articles/" in path
        or "journal" in path
        or "review" in normalized_title
    ):
        return "academic"

    if any(
        marker in path
        for marker in (
            "/knowledge-center/",
            "/technical-documents/",
            "/education/",
            "/learn/",
            "/blog/",
            "/publication/",
            "/publications/",
        )
    ):
        return "educational"

    if _entity_from_domain(url) or any(
        marker in path
        for marker in (
            "/products/",
            "/product/",
            "/services/",
            "/solutions/",
            "/assays/",
            "/panels/",
            "/platform/",
            "/sciencepool/",
        )
    ):
        return "commercial"

    return "unknown"


def _classify_target_page_type(
    *,
    url: str,
    title: str | None,
) -> str:
    path = urlparse(url).path.lower()
    normalized_title = _normalize(title or "")

    if any(
        marker in path
        for marker in ("/publication/", "/publications/", "/papers/", "/journal/")
    ) or any(
        marker in normalized_title
        for marker in ("clinical study", "research study", "published in", "journal article")
    ):
        return "publication"

    if any(marker in path for marker in ("/blog/", "/news/", "/insights/")):
        return "blog/news"

    if any(
        marker in path
        for marker in (
            "/product/",
            "/products/",
            "/service/",
            "/services/",
            "/technology/",
            "/platform/",
            "/assays/",
            "/panels/",
        )
    ):
        return "product/service/technology"

    if any(marker in path for marker in ("/event/", "/events/", "/webinar/")):
        return "event"

    return "general page"


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
    source_type: str | None = None,
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

    if page is None:
        return None, "unavailable"

    if not page.success:
        return None, "unavailable"

    source_type = source_type or _classify_source_type(
        url=page.url,
        title=page.title,
    )
    if source_type != "commercial":
        return None, "not_identified"

    candidate = (
        candidate.strip()
        if candidate and candidate.strip()
        else _entity_from_domain(page.url)
    )
    if not candidate:
        return None, "not_identified"

    # Do not classify the target company itself as a competitor.
    if _normalize(candidate) == _normalize(company_name):
        return None, "not_identified"

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
        if _claim_quote_is_valid(check, page)
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


def _complete_theme_checks(
    *,
    analyses: list[SourceAnalysis],
    fetched_pages: dict[str, FetchedPage],
    returned_checks: list[ThemeCheck],
) -> list[ThemeCheck]:
    """Return one evidence-validated result for every expected source/theme pair."""

    expected: dict[tuple[str, str], str] = {}
    for analysis in analyses:
        for theme in analysis.themes:
            key = _theme_key(theme)
            if key:
                expected.setdefault(
                    (analysis.source_id, key),
                    theme.strip(),
                )

    returned: dict[tuple[str, str], list[ThemeCheck]] = {}
    for check in returned_checks:
        key = _theme_key(check.theme)
        if not key:
            continue
        pair = (check.source_id, key)
        if pair in expected:
            returned.setdefault(pair, []).append(check)

    completed: list[ThemeCheck] = []
    for (source_id, theme_key), theme in expected.items():
        page = fetched_pages.get(source_id)
        if page is None or not page.success:
            completed.append(
                ThemeCheck(
                    source_id=source_id,
                    theme=theme,
                    verdict="unavailable",
                )
            )
            continue

        checks = returned.get((source_id, theme_key), [])
        supported = next(
            (
                check
                for check in checks
                if check.verdict == "supported"
                and check.evidence_quote
                and _quote_is_present(
                    check.evidence_quote,
                    page.content,
                )
            ),
            None,
        )
        if supported is not None:
            completed.append(
                ThemeCheck(
                    source_id=source_id,
                    theme=theme,
                    verdict="supported",
                    evidence_quote=supported.evidence_quote,
                )
            )
            continue

        contradicted = next(
            (
                check
                for check in checks
                if check.verdict == "contradicted"
                and check.evidence_quote
                and _quote_is_present(
                    check.evidence_quote,
                    page.content,
                )
            ),
            None,
        )
        if contradicted is not None:
            completed.append(
                ThemeCheck(
                    source_id=source_id,
                    theme=theme,
                    verdict="contradicted",
                    evidence_quote=contradicted.evidence_quote,
                )
            )
            continue

        non_support_result = next(
            (
                check
                for check in checks
                if check.verdict
                not in {"supported", "contradicted", "unavailable"}
            ),
            None,
        )
        completed.append(
            ThemeCheck(
                source_id=source_id,
                theme=theme,
                verdict=(
                    non_support_result.verdict
                    if non_support_result is not None
                    else "inconclusive"
                ),
            )
        )

    return completed


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

Candidate themes must be specific, checkable content-coverage questions
grounded in the cited material. They are not broad subject labels.

Each question must be narrow enough that a target-site page can clearly
answer it with evidence. Prefer a concrete application, audience need,
comparison, workflow, or capability over a generic topic name.

Good examples:
- Does the site explain neurology biomarker panels for CSF or plasma?
- Does the site compare NULISA with Luminex-based multiplex assays or Olink/PEA?
- Does the site explain multiplex assay options for oncology immune profiling?
- Does the site describe how multiplex panels are used for inflammatory biomarkers?

Bad examples:
- Multiplex protein assays
- Immunology
- Proteomics
- Research
- Scientific information

Keep each question concise enough to use as a website search query.
Do not invent facts about the target company; the question must come from
the cited source evidence.

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

For the themes field, return a short list of specific, checkable
content-coverage questions. Do not return broad topic labels. Each
question must be grounded in the cited evidence and narrow enough that
a target-site page can answer it with a yes/no-style evidence check.

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
        claim_items: list[str] = []
        theme_items: list[str] = []

        analysis_by_source = {
            analysis.source_id: analysis
            for analysis in analyses
        }

        for source_id, analysis in analysis_by_source.items():
            page = fetched_pages.get(source_id)
            if page is None:
                page_context = "STATUS: unavailable"
            else:
                fetch_status = "success" if page.success else "failed"
                page_context = (
                    f"URL: {page.url}\n"
                    f"FETCH STATUS: {fetch_status}"
                )
                if page.success:
                    page_context += (
                        f"\nFETCHED PAGE TEXT:\n{page.content[:30000]}"
                    )

            claim_items.append(
                f"SOURCE {source_id}\n{page_context}\n"
                "CANDIDATE CLAIMS:\n"
                + "\n".join(
                    f"- {claim}"
                    for claim in analysis.claims_to_verify
                )
            )
            theme_items.append(
                f"SOURCE {source_id}\n{page_context}\n"
                "CANDIDATE THEMES:\n"
                + "\n".join(
                    f"- {theme}"
                    for theme in analysis.themes
                )
            )

        joined_claims = "\n\n".join(claim_items)
        joined_themes = "\n\n".join(theme_items)

        structured = self.model.with_structured_output(
            SourceVerificationOutput
        )

        claim_result = structured.invoke(
            [
                (
                    "system",
                    """Verify only the candidate claims against their corresponding
fetched source pages. Do not add claims or use outside knowledge. A claim
is supported only when the page directly supports it. For supported or
contradicted claims, return a short verbatim quote from that page. Do not
paraphrase or invent quotes. The quote must support the entire claim,
not merely contain a related generic term. If a claim names a company,
product, or platform, the quote must connect that entity to the claimed
fact. A paper's stated aim or scope is not evidence that its intended
result, standard, or recommendation exists. Return claim_checks only;
leave theme_checks empty.""",
                ),
                (
                    "human",
                    f"""PROMPT
{prompt_evidence.prompt_text}

SOURCE CLAIM VERIFICATION MATERIAL
{joined_claims}

Return one claim check for each supplied candidate claim.
""",
                ),
            ]
        )

        theme_result = structured.invoke(
            [
                (
                    "system",
                    """Verify only the listed candidate source-theme pairs against
their corresponding fetched pages. Return exactly one ThemeCheck for
every listed pair, preserving its source_id and theme. Do not add themes
or use outside knowledge. A theme is supported only when the page
directly represents it as a meaningful subject. A page about multiplex
immunoassays for neuroscience can support a neurology biomarker-panel
theme when the passage establishes that application; a generic mention
of technology cannot support a specific platform theme. For supported
or contradicted results, include a short verbatim quote from that source
page. For inconclusive or not_found, a quote may be null. Never invent
or paraphrase evidence.""",
                ),
                (
                    "human",
                    f"""PROMPT
{prompt_evidence.prompt_text}

SOURCE THEME VERIFICATION MATERIAL
{joined_themes}

Return one theme_check per candidate theme listed under each source.
Do not omit a pair. If the page is unavailable, use verdict 'unavailable'.
""",
                ),
            ]
        )

        return SourceVerificationOutput(
            claim_checks=claim_result.claim_checks,
            theme_checks=theme_result.theme_checks,
        )

    def _verify_target_site_themes(
        self,
        *,
        themes: list[str],
        pages: dict[str, FetchedPage],
    ) -> list[ThemeCheck]:
        """Verify target-site theme coverage against the fetched sitemap inventory."""

        if not themes or not pages:
            return []

        page_items = list(pages.items())
        expected_themes = {
            _theme_key(theme): theme
            for theme in themes
        }

        returned_checks: list[ThemeCheck] = []
        structured = self.model.with_structured_output(
            TargetSiteVerificationOutput
        )

        # Keep each request bounded. Missing checks are handled below as
        # inconclusive; they are never silently interpreted as not_found.
        batch_size = 8

        for start in range(0, len(page_items), batch_size):
            batch = page_items[start:start + batch_size]
            page_blocks: list[str] = []

            for source_id, page in batch:
                if page.success:
                    page_status = "success"
                    page_text = page.content[:10000]
                else:
                    page_status = "failed"
                    page_text = ""

                page_blocks.append(
                    f"TARGET PAGE {source_id}\n"
                    f"URL: {page.url}\n"
                    f"TITLE: {page.title or 'unknown'}\n"
                    f"PAGE TYPE: {_classify_target_page_type(url=page.url, title=page.title)}\n"
                    f"FETCH STATUS: {page_status}\n"
                    f"PAGE TEXT:\n{page_text}"
                )

            try:
                result = structured.invoke(
                    [
                        (
                            "system",
                            """You are verifying whether specific, checkable content-coverage
questions are answered by pages from a target company's website.

For EVERY target page and EVERY listed coverage question, return exactly
one theme_check.

Rules:
1. Preserve source_id exactly.
2. Preserve the supplied coverage-question wording.
3. Return exactly one check for every page/question pair.
4. Use "supported" only when the page's main subject or a substantive
   section directly answers the question. A passing mention is not enough.
5. A product, service, or technology page is strong evidence when it
   directly answers the question. A publication, event, poster, or PDF
   is supporting evidence only when the page itself answers the question
   substantively.
6. A "supported" result MUST contain a short verbatim quote copied from
   the supplied page text.
7. Use "not_found" when the supplied page text does not answer the question.
8. Use "inconclusive" when the page is tangential or the text is insufficient.
9. Use "unavailable" when the page fetch status is failed.
10. Do not infer from company name, URL alone, reputation, outside
    knowledge, or other pages.
11. Never invent or paraphrase evidence.
12. Do not omit a page/question pair.

The Python application independently verifies every supporting quote.
""",
                        ),
                        (
                            "human",
                            f"""TARGET COMPANY WEBSITE

VERIFIED SOURCE COVERAGE QUESTIONS:
{chr(10).join(f'- {theme}' for theme in themes)}

TARGET PAGES:
{chr(10).join(page_blocks)}

Return exactly one theme_check for every combination of:
TARGET PAGE x VERIFIED SOURCE COVERAGE QUESTION

For failed pages use "unavailable".
For successful pages that do not substantively answer the question use
"not_found". For insufficient text use "inconclusive". For supported
questions provide a short verbatim evidence_quote that directly answers
the question.""",
                        ),
                    ]
                )
                returned_checks.extend(result.theme_checks)

            except Exception:
                # A failed verification call is not evidence of absence.
                for source_id, page in batch:
                    for theme in themes:
                        returned_checks.append(
                            ThemeCheck(
                                source_id=source_id,
                                theme=theme,
                                verdict=(
                                    "unavailable"
                                    if not page.success
                                    else "inconclusive"
                                ),
                            )
                        )

        checks_by_pair: dict[tuple[str, str], list[ThemeCheck]] = {}

        for check in returned_checks:
            theme_key = _theme_key(check.theme)

            if theme_key not in expected_themes:
                continue

            if check.source_id not in pages:
                continue

            checks_by_pair.setdefault(
                (check.source_id, theme_key),
                [],
            ).append(check)

        completed: list[ThemeCheck] = []

        # Complete the expected page x theme matrix deterministically.
        # Missing model output becomes inconclusive, never not_found.
        for source_id, page in page_items:
            for theme_key, theme in expected_themes.items():

                if not page.success:
                    completed.append(
                        ThemeCheck(
                            source_id=source_id,
                            theme=theme,
                            verdict="unavailable",
                        )
                    )
                    continue

                checks = checks_by_pair.get(
                    (source_id, theme_key),
                    [],
                )

                supported = next(
                    (
                        check
                        for check in checks
                        if (
                            check.verdict == "supported"
                            and check.evidence_quote
                            and _quote_is_present(
                                check.evidence_quote,
                                page.content,
                            )
                        )
                    ),
                    None,
                )

                if supported is not None:
                    completed.append(
                        ThemeCheck(
                            source_id=source_id,
                            theme=theme,
                            verdict="supported",
                            evidence_quote=supported.evidence_quote,
                        )
                    )
                    continue

                if any(
                    check.verdict == "not_found"
                    for check in checks
                ):
                    completed.append(
                        ThemeCheck(
                            source_id=source_id,
                            theme=theme,
                            verdict="not_found",
                        )
                    )
                    continue

                completed.append(
                    ThemeCheck(
                        source_id=source_id,
                        theme=theme,
                        verdict="inconclusive",
                    )
                )

        return completed

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

        verification = SourceVerificationOutput(
            claim_checks=verification.claim_checks,
            theme_checks=_complete_theme_checks(
                analyses=first_pass.source_analyses,
                fetched_pages=fetched_pages,
                returned_checks=verification.theme_checks,
            ),
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

        target_site_url = project.get("company_url")
        target_site_pages: dict[str, FetchedPage] = {}
        target_site_pages_discovered = 0
        target_site_urls_discovered = 0
        target_site_pages_attempted = 0
        target_site_pages_fetched = 0
        target_site_diagnostics: list[TargetSiteDiagnostic] = []
        target_site_inventory_complete = False
        target_site_note = "No target company URL was provided."
        target_theme_coverage: dict[str, str] = {}
        target_theme_evidence: dict[str, list[TargetSiteEvidence]] = {}

        if target_site_url:
            try:
                max_target_urls = int(
                    os.getenv("ACA_TARGET_SITE_MAX_URLS", "15")
                )
                if max_target_urls < 1:
                    max_target_urls = 1
                pages_per_theme = int(
                    os.getenv("ACA_TARGET_PAGES_PER_THEME", "3")
                )
                if pages_per_theme < 1:
                    pages_per_theme = 1

                site_inventory: SiteSearchCandidates = (
                    self.fetcher.search_site_pages(
                        target_site_url,
                        verified_theme_names,
                        per_theme_limit=pages_per_theme,
                        total_limit=max_target_urls,
                    )
                )

                target_site_pages_discovered = len(site_inventory.urls)
                target_site_urls_discovered = (
                    site_inventory.discovered_url_count
                )

                for skipped in site_inventory.skipped_urls or []:
                    target_site_diagnostics.append(
                        TargetSiteDiagnostic(
                            status="skipped",
                            url=skipped.url,
                            reason_category=skipped.reason_category,
                            reason=skipped.reason,
                        )
                    )

                target_site_pages = self.fetcher.fetch_pages(
                    {
                        f"TARGET-{index}": url
                        for index, url in enumerate(
                            site_inventory.urls,
                            start=1,
                        )
                    }
                )

                target_site_pages_attempted = len(target_site_pages)
                target_site_pages_fetched = sum(
                    1
                    for page in target_site_pages.values()
                    if page.success
                )

                for page in target_site_pages.values():
                    if page.success:
                        continue

                    target_site_diagnostics.append(
                        TargetSiteDiagnostic(
                            status="failed",
                            url=page.url,
                            reason_category=_classify_target_fetch_failure(
                                page
                            ),
                            reason=page.error or "No error detail returned.",
                        )
                    )

                target_site_note = site_inventory.note or ""
                target_site_note = (
                    f"{target_site_note} Targeted search checked at most "
                    f"{pages_per_theme} URLs per coverage question and "
                    f"{max_target_urls} total. This is not an exhaustive "
                    "site inventory, so unanswered questions are potential "
                    "gap signals only."
                ).strip()

                if target_site_pages_fetched < target_site_pages_discovered:
                    failed_count = (
                        target_site_pages_discovered
                        - target_site_pages_fetched
                    )
                    target_site_note = (
                        f"{target_site_note} "
                        f"{failed_count} target-site page(s) could not be "
                        f"extracted. Negative coverage findings remain unknown."
                    ).strip()

                target_site_inventory_complete = False

            except Exception as exc:
                target_site_note = f"Target-site discovery failed: {exc}"
                target_site_inventory_complete = False

        target_site_coverage = (
            "assessed"
            if target_site_inventory_complete
            else "partial"
            if target_site_pages_fetched
            else "unknown"
        )

        if verified_theme_names and target_site_pages:
            target_checks = self._verify_target_site_themes(
                themes=verified_theme_names,
                pages=target_site_pages,
            )

            for theme in verified_theme_names:
                theme_key = _theme_key(theme)
                checks = [
                    check
                    for check in target_checks
                    if _theme_key(check.theme) == theme_key
                ]

                evidence: list[TargetSiteEvidence] = []

                for check in checks:
                    if len(evidence) >= 3:
                        break

                    page = target_site_pages.get(check.source_id)

                    if (
                        check.verdict == "supported"
                        and check.evidence_quote
                        and page is not None
                        and page.success
                        and _quote_is_present(
                            check.evidence_quote,
                            page.content,
                        )
                    ):
                        evidence.append(
                            TargetSiteEvidence(
                                theme=theme,
                                source_url=page.url,
                                evidence_quote=check.evidence_quote,
                                page_type=_classify_target_page_type(
                                    url=page.url,
                                    title=page.title,
                                ),
                            )
                        )

                target_theme_evidence[theme_key] = evidence

                if evidence:
                    target_theme_coverage[theme_key] = "covered"

                elif (
                    target_site_inventory_complete
                    and len(checks) == target_site_pages_discovered
                    and all(
                        check.verdict == "not_found"
                        for check in checks
                    )
                ):
                    target_theme_coverage[theme_key] = "not_covered"

                else:
                    target_theme_coverage[theme_key] = "unknown"

        assessed_theme_names = list(candidate_themes)
        original_candidate_theme_count = len(assessed_theme_names)

        verified_theme_keys = {
            _theme_key(theme)
            for theme in verified_theme_names
        }

        unresolved_candidate_themes = [
            theme
            for theme in assessed_theme_names
            if _theme_key(theme) not in verified_theme_keys
        ]

        # Keep candidate_themes as unresolved themes for backwards
        # compatibility with the existing ContentGap contract.
        candidate_themes = unresolved_candidate_themes

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

                page = fetched_pages.get(source_id)
                resolved_url = (
                    page.url
                    if page is not None and page.success
                    else source.url
                )
                source_type = _classify_source_type(
                    url=resolved_url,
                    title=(
                        page.title
                        if page is not None and page.success
                        else source.title
                    ),
                )

                theme_source_ids.add(
                    source_id
                )

                theme_cited_by.add(
                    source.model
                )

                source_competitor = (
                    analysis.competitor
                    or _entity_from_domain(resolved_url)
                )
                if source_type == "commercial" and source_competitor:
                    theme_candidate_competitors.add(
                        source_competitor
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
                    candidate=(
                        analysis.competitor
                        or _entity_from_domain(
                            page.url
                            if page is not None
                            else None
                        )
                    ),
                    page=page,
                    company_name=company_name,
                    relevant=any(
                        evidence.source_id == source_id
                        for evidence in theme_evidence
                    ),
                    source_type=_classify_source_type(
                        url=page.url if page is not None else None,
                        title=page.title if page is not None else None,
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
            target_coverage = target_theme_coverage.get(
                theme_key,
                "unknown",
            )
            target_evidence = target_theme_evidence.get(
                theme_key,
                [],
            )
            theme_gap_status = (
                "established"
                if target_coverage == "not_covered"
                else "not_established"
            )
            if target_coverage == "covered":
                theme_opportunity = (
                    f"The target-site inventory contains verified evidence "
                    f"covering {theme}; review whether that coverage is "
                    f"sufficient for the intended audience and use case."
                )
            elif target_coverage == "not_covered":
                theme_opportunity = (
                    f"No verified coverage of {theme} was found across the "
                    f"complete fetched sitemap inventory. This establishes "
                    f"a content-gap signal within the assessed sitemap "
                    f"inventory, not necessarily across every public URL."
                )
            else:
                theme_opportunity = (
                    f"No verified page was found in the bounded target-site "
                    f"search that answers {theme}. Treat this as a potential "
                    f"content-gap signal only; targeted search is not proof "
                    f"of site-wide absence."
                )

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
                    target_site_coverage=target_coverage,
                    target_site_evidence=target_evidence,
                    content_gap_status=theme_gap_status,
                    opportunity=theme_opportunity,
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

            resolved_url = (
                page.url
                if page is not None and page.success
                else source.url
            )
            source_type = _classify_source_type(
                url=resolved_url,
                title=(
                    page.title
                    if page is not None and page.success
                    else source.title
                ),
            )
            if source_type == "commercial":
                candidate_competitor = (
                    candidate_competitor
                    or _entity_from_domain(resolved_url)
                )
            else:
                candidate_competitor = None

            verified_competitor, competitor_status = (
                _verify_competitor(
                    candidate=candidate_competitor,
                    page=page,
                    company_name=company_name,
                    relevant=any(
                        evidence.source_id == source_id
                        for evidence in verified_theme_evidence
                    ),
                    source_type=source_type,
                )
            )

            # Build SourceClaimAssessment objects only from the
            # verification output.
            source_claims: list[
                SourceClaimAssessment
            ] = []

            for check in source_claim_checks:
                quote_valid = _claim_quote_is_valid(
                    check,
                    page,
                )
                source_claims.append(
                    SourceClaimAssessment(
                        claim=check.claim,
                        status=(
                            "verified"
                            if quote_valid
                            else "model_supported"
                            if check.verdict == "supported"
                            else check.verdict
                        ),
                        evidence_quote=(
                            check.evidence_quote
                            if quote_valid
                            or check.verdict != "supported"
                            else None
                        ),
                    )
)
            supported_claims = [
                check
                for check in source_claim_checks
                if _claim_quote_is_valid(check, page)
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
                    source_type=source_type,
                    url=(
                        page.url
                        if page is not None and page.success
                        else source.url
                    ),
                    domain=(
                        _safe_domain(page.url)
                        if page is not None and page.success
                        else source.domain
                    ),
                    title=(
                        page.title
                        if page is not None and page.success and page.title
                        else source.title
                    ),
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
                    requested_url=(
                        page.requested_url
                        if page is not None
                        else source.url
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

        if target_site_url:
            verification_summary += (
                f" Target-site inventory: {target_site_coverage}; "
                f"{target_site_pages_fetched} of "
                f"{target_site_pages_discovered} pages extracted."
            )
            if target_site_note:
                verification_summary += f" {target_site_note}"

        established_gap_themes = [
            assessment.theme
            for assessment in theme_assessments
            if assessment.content_gap_status == "established"
        ]

        # --------------------------------------------------------------
        # Candidate opportunity.
        #
        # Important: this is NOT a confirmed content gap.
        # --------------------------------------------------------------

        if established_gap_themes:
            opportunity = (
                f"The complete extracted sitemap inventory did not show "
                f"coverage for these source-verified themes: "
                + ", ".join(established_gap_themes)
                + ". Review these as potential gaps, noting that the sitemap "
                "may not enumerate every public page."
            )
        elif verified_theme_names and target_site_coverage == "assessed":
            opportunity = (
                f"The target-site inventory was assessed across "
                f"{target_site_pages_fetched} extracted pages. Compare the "
                f"page-level coverage evidence for the verified themes before "
                "choosing a content action."
            )
        elif verified_theme_names and target_site_pages_fetched:
            potential_gap_themes = [
                theme
                for theme in verified_theme_names
                if target_theme_coverage.get(_theme_key(theme)) == "unknown"
            ]
            if potential_gap_themes:
                opportunity = (
                    "Targeted website search found no verified coverage for these "
                    "specific questions: "
                    + ", ".join(potential_gap_themes[:5])
                    + ". Treat them as potential content-gap signals, not "
                    "proof of site-wide absence."
                )
            else:
                opportunity = (
                    "Targeted website search found verified coverage for the "
                    "selected questions. No site-wide content gap is established."
                )
        elif verified_theme_names:
            opportunity = (
                "Investigate the verified coverage questions against the "
                f"{company_name} site; target-site inspection is "
                f"{target_site_coverage}."
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

        if established_gap_themes:
            target_angle = (
                f"Review or develop {company_name} content for "
                + ", ".join(established_gap_themes)
                + ", using the verified cited-source evidence as the reference."
            )
        elif verified_theme_names and target_site_coverage == "assessed":
            target_angle = (
                f"Use the assessed target-site pages and verified source evidence "
                f"to prioritize coverage of: {', '.join(verified_theme_names)}."
            )
        elif verified_theme_names:
            target_angle = (
                f"Compare {company_name}'s existing coverage of "
                + ", ".join(
                    verified_theme_names
                )
                + f" against the verified source evidence. Target-site "
                f"inspection is {target_site_coverage}; do not infer missing coverage."
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

        if target_site_coverage == "assessed":
            target_site_observation = (
                f"The sitemap-derived inventory for {company_name} was "
                f"fully extracted ({target_site_pages_fetched} pages). "
                + (
                    "The checked themes not covered were: "
                    + ", ".join(established_gap_themes)
                    + "."
                    if established_gap_themes
                    else "No verified source theme was classified as absent from the inventory."
                )
            )
        elif target_site_coverage == "partial":
            target_site_observation = (
                f"Only {target_site_pages_fetched} of "
                f"{target_site_pages_discovered} discovered pages were "
                "sufficiently inspected; missing-theme findings remain unknown."
            )
        else:
            target_site_observation = None

        hypothesis = _build_hypothesis(
            company_name=company_name,
            verified_themes=verified_theme_names,
            candidate_themes=(
                candidate_themes or assessed_theme_names
            ),
            target_site_observation=target_site_observation,
        )

        covered_target_themes = [
            theme
            for theme in verified_theme_names
            if target_theme_coverage.get(_theme_key(theme)) == "covered"
        ]
        potential_gap_themes = [
            theme
            for theme in verified_theme_names
            if (
                target_site_pages_fetched > 0
                and target_theme_coverage.get(_theme_key(theme)) == "unknown"
            )
        ]
        if established_gap_themes:
            candidate_target_angle = (
                "Review whether new or improved resources are appropriate for: "
                + ", ".join(established_gap_themes)
                + ". These are potential gaps in the checked sitemap pages, "
                "not proof that the entire website lacks this content."
            )
        elif potential_gap_themes:
            candidate_target_angle = (
                "Potential content-gap signals from targeted website search: "
                + ", ".join(potential_gap_themes[:5])
                + ". These are specific unanswered coverage questions in the "
                "selected pages, not proof of site-wide absence."
            )
        elif covered_target_themes:
            candidate_target_angle = (
                "Review the existing pages covering "
                + ", ".join(covered_target_themes[:5])
                + ". Consider additional content only if a specific unmet "
                "audience need remains; no product advantage is inferred."
            )
        elif verified_theme_names:
            candidate_target_angle = (
                "Review the verified coverage questions against the selected "
                "target-site pages: "
                + ", ".join(verified_theme_names[:5])
                + ". No site-wide gap is established."
            )
        else:
            candidate_target_angle = (
                "A target content angle cannot be recommended until source "
                "themes are verified and relevant target-site pages are reviewed."
            )

        return ContentGap(
            prompt_id=prompt_evidence.prompt_id,
            prompt_text=prompt_evidence.prompt_text,
            cited_sources=analyzed_sources,
            candidate_themes=candidate_themes,
            themes=verified_theme_names,
            candidate_target_angle=candidate_target_angle,
            theme_evidence=verified_theme_evidence,
            theme_assessments=theme_assessments,
            verification_status=verification_status,
            verification_summary=verification_summary,
            target_site_url=target_site_url,
            target_site_coverage=target_site_coverage,
            target_site_pages_discovered=target_site_pages_discovered,
            target_site_urls_discovered=target_site_urls_discovered,
            target_site_pages_attempted=target_site_pages_attempted,
            target_site_pages_fetched=target_site_pages_fetched,
            target_site_diagnostics=target_site_diagnostics,
            target_site_inventory_complete=target_site_inventory_complete,
            target_site_note=target_site_note,
            content_gap_status=(
                "established"
                if established_gap_themes
                else "not_established"
            ),
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
    if state.company_url:
        project["company_url"] = state.company_url

    analyzer = SourceAnalyzer()

    state.gap = analyzer.analyze(
        project=project,
        prompt_evidence=prompt,
    )

    return state