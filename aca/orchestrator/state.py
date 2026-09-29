"""Typed workflow state passed between ACA agents."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class CitedSource(BaseModel):
    """A source cited by an AI model when answering an AImpact prompt."""

    model: str
    url: str | None = None
    domain: str | None = None
    competitor: str | None = None
    excerpt: str
    contribution: str | None = None


class AnalyzedSource(CitedSource):
    """A cited source enriched with its observed contribution to an answer."""

    contribution: str
    source_id: str = ""
    candidate_competitor: str | None = None
    competitor_verification_status: Literal[
        "verified",
        "unverified",
        "unavailable",
        "not_identified",
    ] = "not_identified"
    verified_claims: list["SourceClaimAssessment"] = Field(
        default_factory=list
    )
    verification_status: Literal[
        "verified",
        "partially_verified",
        "unsupported",
        "unverified",
        "unavailable",
    ] = "unverified"
    verification_evidence: list[str] = Field(
        default_factory=list
    )
    verified_url: str | None = None
    verification_note: str | None = None


class ThemeEvidence(BaseModel):
    theme: str
    source_id: str
    source_url: str
    evidence_quote: str


class SourceClaimAssessment(BaseModel):
    claim: str
    status: Literal[
        "verified",
        "contradicted",
        "not_found",
        "inconclusive",
        "unavailable",
    ]
    evidence_quote: str | None = None


class ThemeAssessment(BaseModel):
    theme: str
    status: Literal["verified", "unverified"]
    candidate_source_ids: list[str] = Field(default_factory=list)
    candidate_cited_by: list[str] = Field(default_factory=list)
    candidate_competitors: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    cited_by: list[str] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list)
    evidence: list[ThemeEvidence] = Field(default_factory=list)
    target_site_coverage: Literal["unknown"] = "unknown"
    content_gap_status: Literal["not_established"] = "not_established"
    opportunity: str


class ModelEvidence(BaseModel):
    """One AI model's answer and cited sources for a prompt."""

    response_text: str | None = None
    sources: list[CitedSource] = Field(default_factory=list)


class PromptEvidence(BaseModel):
    """All AI evidence collected for one AImpact prompt."""

    prompt_id: str
    prompt_text: str

    per_model: dict[str, ModelEvidence] = Field(
        default_factory=dict
    )


class ContentGap(BaseModel):
    """
    Source Analyzer evidence and opportunity assessment.

    Page verification does not establish a target-site gap without
    the target site's content inventory.
    """

    prompt_id: str
    prompt_text: str

    cited_sources: list[AnalyzedSource] = Field(
        default_factory=list
    )

    candidate_themes: list[str] = Field(
        default_factory=list
    )

    themes: list[str] = Field(
        default_factory=list
    )

    candidate_target_angle: str = ""

    theme_evidence: list[ThemeEvidence] = Field(
        default_factory=list
    )

    theme_assessments: list[ThemeAssessment] = Field(
        default_factory=list
    )

    verification_status: Literal[
        "verified",
        "partially_verified",
        "unverified",
        "unavailable",
    ] = "unverified"

    verification_summary: str = ""

    target_site_coverage: Literal["unknown"] = "unknown"

    content_gap_status: Literal["not_established"] = "not_established"

    opportunity: str = ""

    target_angle: str

    hypothesis: str


class Brief(BaseModel):
    title: str
    slug: str
    meta: str
    outline: list[dict[str, Any]]
    faqs: list[dict[str, str]]
    internal_links: list[str]


class Draft(BaseModel):
    markdown: str
    word_count: int


class Verdict(BaseModel):
    decision: str
    notes: list[dict[str, Any]] = Field(
        default_factory=list
    )
    brand_voice_score: float


class WorkflowState(BaseModel):
    """Complete state passed through the ACA workflow."""

    job_id: str
    tenant_id: int
    campaign_id: int

    # Target company / brand being analyzed.
    company_name: str

    # Input evidence supplied to Source Analyzer.
    prompts: list[PromptEvidence] = Field(
        default_factory=list
    )

    # Agent outputs.
    gap: ContentGap | None = None
    brief: Brief | None = None
    verdict: Verdict | None = None

    # Reviewer -> Writer loop.
    revise_count: int = 0