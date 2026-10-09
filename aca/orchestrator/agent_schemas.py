"""Typed inputs and outputs for the six ACA agents.

Naming note: ``state.py`` already defines ``Brief``, ``Draft`` and
``Verdict`` for the earlier workflow. The new schemas use different names
(``ContentBrief``, ``ArticleDraft``, ``ReviewVerdict``) so both can live in
the same package while you migrate.

Agents pass these objects (references such as URLs and fact IDs, not raw page
text) through the LangGraph state defined in ``workflow.py``.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ACAModel(BaseModel):
    """Base model. Forbidding extra fields keeps LLM structured output strict."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Shared vocabularies
# ---------------------------------------------------------------------------

SourceType = Literal[
    "client_owned",
    "competitor_owned",
    "academic",
    "community",
    "media_review",
    "aggregator",
    "reference",
    "unknown",
]

FactStatus = Literal[
    "verified",
    "contradicted",
    "not_found",
    "inconclusive",
]

SourceTier = Literal[
    "first_party",  # the organization's own site or documents
    "primary",  # journals, regulators, standards bodies
    "trusted_third_party",
    "other",
]

Confidence = Literal["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Agent 1: Prompt + answer intelligence
# Input: list[PromptEvidence] from state.py (AImpact prompts, answers, citations)
# ---------------------------------------------------------------------------


class PromptRef(ACAModel):
    prompt_id: str
    prompt_text: str
    intent: str | None = None


class ConceptStat(ACAModel):
    name: str
    # Share of sampled answers (models x runs) that mention the concept.
    frequency: float = Field(ge=0.0, le=1.0)


class CompanyMention(ACAModel):
    name: str
    rank: int | None = None
    attributes: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)


class CitationRef(ACAModel):
    """A cited URL after redirect resolution."""

    final_url: str
    requested_url: str | None = None
    domain: str | None = None
    models: list[str] = Field(default_factory=list)
    count: int = 1


class PromptUnderstanding(ACAModel):
    cluster: list[PromptRef]
    topic: str
    intent: str | None = None
    entities: list[str] = Field(default_factory=list)
    concepts: list[ConceptStat] = Field(default_factory=list)
    companies_mentioned: list[CompanyMention] = Field(default_factory=list)
    client_presence: Literal["absent", "low_rank", "negative_framing", "present"]
    # Inferred from answers, which vary run to run. Treat as a hypothesis.
    answer_criteria: list[str] = Field(default_factory=list)
    citation_map: list[CitationRef] = Field(default_factory=list)
    # prompt_id -> mean AImpact score at the time of analysis (re-audit baseline).
    baseline_scores: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent 2: Source intelligence
# Input: PromptUnderstanding (mainly citation_map and concepts)
# ---------------------------------------------------------------------------


class SourceClaim(ACAModel):
    text: str
    # Only set when the quote was found verbatim in the fetched page text.
    verified_quote: str | None = None


class CompetitorClaim(ACAModel):
    name: str
    attribute: str
    source_url: str
    evidence_quote: str | None = None
    verified: bool = False


class StructureMetrics(ACAModel):
    """Measured in code, not estimated by the LLM."""

    word_count: int = 0
    heading_depth: int = 0
    h2_count: int = 0
    table_count: int = 0
    has_faq: bool = False
    avg_paragraph_words: float = 0.0


class StyleProfile(ACAModel):
    """Tone and audience come from the LLM; structure comes from metrics."""

    tone: str = ""
    audience: str = ""
    content_pattern: list[str] = Field(default_factory=list)


class RankedSource(ACAModel):
    url: str
    domain: str
    source_type: SourceType = "unknown"
    citation_count: int = 0
    # Hard signals first (citations across models, runs, prompts), then relevance.
    rank_score: float = 0.0
    fetch_status: Literal["fetched", "unavailable", "blocked"] = "fetched"
    topics: list[str] = Field(default_factory=list)
    claims: list[SourceClaim] = Field(default_factory=list)
    competitors: list[CompetitorClaim] = Field(default_factory=list)
    structure: StructureMetrics = Field(default_factory=StructureMetrics)
    style: StyleProfile = Field(default_factory=StyleProfile)


class QuestionItem(ACAModel):
    question: str
    frequency: int = 1
    origin: Literal["source", "answer", "comment"] = "source"


class SourceIntelligence(ACAModel):
    ranked_sources: list[RankedSource] = Field(default_factory=list)
    # Share of citations by source type, summing to about 1.0.
    source_mix: dict[SourceType, float] = Field(default_factory=dict)
    # Competitor name -> attributes they are credited for (each backed by a source).
    competitor_attribute_map: dict[str, list[str]] = Field(default_factory=dict)
    question_map: list[QuestionItem] = Field(default_factory=list)
    consensus_outline: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent 3: Ground truth + verification
# Input: ClaimRequest list + OrgInputs. Also exposed as a verify() tool that
# Agent 6 calls on every claim in a draft.
# ---------------------------------------------------------------------------


class OrgFact(ACAModel):
    """A fact the organization confirmed directly."""

    claim: str
    confirmed_by: str | None = None
    source_url: str | None = None


class OrgInputs(ACAModel):
    domain: str | None = None
    uploads: list[str] = Field(default_factory=list)
    confirmations: list[OrgFact] = Field(default_factory=list)


class ClaimRequest(ACAModel):
    claim: str
    kind: Literal["company", "competitor", "statistic", "scientific"] = "company"


class Evidence(ACAModel):
    url: str
    quote: str
    source_tier: SourceTier = "other"
    published: date | None = None


class Fact(ACAModel):
    fact_id: str
    claim: str
    status: FactStatus
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: Confidence = "low"


class FactSet(ACAModel):
    facts: list[Fact] = Field(default_factory=list)
    # Claims the Writer must not make ("industry leader", "largest", ...).
    banned_claims: list[str] = Field(default_factory=list)
    questions_for_org: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent 4: Strategy + brief
# Input: PromptUnderstanding, SourceIntelligence, FactSet
# ---------------------------------------------------------------------------


class OutlineSection(ACAModel):
    h2: str
    purpose: str
    key_points: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)


class FAQItem(ACAModel):
    question: str
    answer_points: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)


class ContentBrief(ACAModel):
    action: Literal["create", "refresh", "needs_data"]
    channel: str
    format: str
    title: str
    slug: str
    meta: str
    angle: str
    audience: str
    # What this piece adds that existing sources lack (data, comparison, synthesis).
    novelty: str
    outline: list[OutlineSection] = Field(default_factory=list)
    faqs: list[FAQItem] = Field(default_factory=list)
    entities_to_cover: list[str] = Field(default_factory=list)
    internal_links: list[str] = Field(default_factory=list)
    style_spec: StyleProfile | None = None
    # write: proceed, ask: ask the organization, stop: not enough to write.
    readiness: Literal["write", "ask", "stop"]
    questions_for_org: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent 5: Writer
# Input: ContentBrief, FactSet, revision notes (optional)
# ---------------------------------------------------------------------------


class ClaimLink(ACAModel):
    """Ties one sentence in the draft to the fact or source that supports it."""

    sentence_id: str
    fact_id: str | None = None
    source_ref: str | None = None


class ArticleDraft(ACAModel):
    markdown: str
    word_count: int
    revision_no: int = 0
    claim_map: list[ClaimLink] = Field(default_factory=list)
    faq_jsonld: dict | None = None
    meta: str = ""


# ---------------------------------------------------------------------------
# Agent 6: Reviewer
# Input: ArticleDraft, ContentBrief, FactSet, source text refs
# ---------------------------------------------------------------------------


class ClaimCheck(ACAModel):
    claim: str
    status: Literal["supported", "unsupported", "contradicted"]
    fact_id: str | None = None


class OriginalityReport(ACAModel):
    max_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    flagged: list[str] = Field(default_factory=list)


class ReviewVerdict(ACAModel):
    decision: Literal["approve", "revise", "reject"]
    # Only meaningful when decision == "reject".
    reject_target: Literal["strategy", "human"] | None = None
    claim_checks: list[ClaimCheck] = Field(default_factory=list)
    prompt_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    concept_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    style_score: float = Field(default=0.0, ge=0.0, le=1.0)
    originality: OriginalityReport = Field(default_factory=OriginalityReport)
    unsupported_superlatives: list[str] = Field(default_factory=list)
    ai_readiness: dict[str, bool] = Field(default_factory=dict)
    revision_notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# After approval
# ---------------------------------------------------------------------------


class Publication(ACAModel):
    channel: str
    url: str | None = None
    scheduled_for: str | None = None
    export_path: str | None = None


class ReauditPlan(ACAModel):
    prompt_ids: list[str]
    due_after_days: list[int] = Field(default_factory=lambda: [7, 30])
    baseline_scores: dict[str, float] = Field(default_factory=dict)