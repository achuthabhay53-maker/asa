"""Pydantic classes for Agent 3, Ground truth."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

FactStatus = Literal["verified", "contradicted", "not_found", "inconclusive"]
Confidence = Literal["low", "medium", "high"]
Relevance = Literal["high", "medium", "low"]
SourceTier = Literal["first_party", "primary", "trusted_third_party", "other"]
ClaimType = Literal[
    "product",
    "platform_capability",
    "service",
    "certification",
    "geographic_service_availability",
    "other",
]
TopicDecision = Literal["write", "write_limited", "do_not_write", "research_more"]
CheckedFrom = Literal["prompt", "content_opportunity"]


class ClientInput(BaseModel):
    """Website is supplied by the user. Name comes from Agent 1 unless overridden."""

    company_name: str
    website: str


class PromptUnderstandingInput(BaseModel):
    intent: str = ""
    topic: str = ""
    audience: str | None = None
    user_goal: str = ""
    key_questions: list[str] = Field(default_factory=list)


class PromptContext(BaseModel):
    """What Agent 3 verifies against. The original prompt comes first."""

    prompt_id: str
    original_prompt: str
    customer: ClientInput
    prompt_understanding: PromptUnderstandingInput
    content_opportunities: list[str] = Field(default_factory=list)
    answer_topics: list[str] = Field(default_factory=list)
    comparison_criteria: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    url: str
    quote: str
    source_tier: SourceTier = "other"
    published: date | None = None


class Fact(BaseModel):
    fact_id: str
    claim: str
    status: FactStatus
    claim_type: ClaimType = "other"
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: Confidence = "low"
    relevance_to_original_prompt: Relevance = "medium"
    relevance_to_topic: Relevance = "medium"
    checked_from: CheckedFrom = "prompt"
    reason: str | None = None


class TopicVerdict(BaseModel):
    topic: str
    decision: TopicDecision
    fact_ids: list[str] = Field(default_factory=list)
    reason: str | None = None


class DoNotWrite(BaseModel):
    claim_or_topic: str
    reason: str


class UnresolvedClaim(BaseModel):
    """Checked, but not established. Absence is not proof the claim is false."""

    claim: str
    reason: str


class ScopeNote(BaseModel):
    """The typed prompt stays unchanged. Extra topic wording is labeled."""

    original_prompt: str
    inferred_topic: str | None = None
    scope_note: str | None = None


class LimitedStatement(BaseModel):
    """A quote supports this sentence, and a broader wording is not allowed."""

    fact_id: str
    statement: str
    wording_limit: str
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: Confidence = "medium"
    relevance_to_original_prompt: Relevance = "medium"
    relevance_to_topic: Relevance = "medium"


EvidenceSourceType = Literal["company_website", "reputable_external"]
TopicAngle = Literal["company_capability", "industry_context", "comparison_methodology", "educational_guide"]
EvidenceStatus = Literal["supported", "needs_more_research", "unsupported"]
SupportStatus = Literal["fully_supported", "partially_supported", "needs_more_research", "unsupported"]
PromptScope = Literal["broad_industry", "company_specific", "hybrid"]


class EvidenceItem(BaseModel):
    """Detailed evidence quote with attribution and source type."""

    evidence_id: str
    url: str
    quote: str
    source_type: EvidenceSourceType = "company_website"
    domain: str = ""
    page_title: str | None = None
    context: str | None = None
    published: date | None = None


class CandidateSubtopic(BaseModel):
    """Possible blog section and related questions discovered through research."""

    subtopic_id: str = ""
    title: str
    description: str = ""
    relevance: Relevance = "high"
    support_status: SupportStatus = "partially_supported"
    verified_fact_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    research_needed: list[str] = Field(default_factory=list)
    suggested_questions: list[str] = Field(default_factory=list)
    topic_angle: TopicAngle = "company_capability"

    # Compatibility properties
    @property
    def relevance_to_main_topic(self) -> str:
        return self.relevance

    @property
    def evidence_status(self) -> str:
        return self.support_status


class VerifiedFact(BaseModel):
    """A factual claim verified against researched evidence."""

    fact_id: str
    statement: str
    category: str = "company_capability"
    evidence_type: EvidenceSourceType = "company_website"
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: Confidence = "high"
    status: str = "verified"
    wording_boundary: str | None = None


class ResearchGap(BaseModel):
    """A topic or question that needs more evidence before it can be written."""

    gap_id: str
    topic_or_question: str
    missing_claim: str = ""
    """The specific claim or assertion that lacks evidence."""
    reason: str
    suggested_search_direction: str | None = None
    affects_subtopic_ids: list[str] = Field(default_factory=list)
    """Subtopic IDs that cannot be fully written until this gap is resolved."""


class PromptVerification(BaseModel):
    """Verification of the original prompt and intent for the customer."""

    is_verified: bool
    intent_valid: bool
    customer_relevant: bool
    reason: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class EvidenceBackedTopic(BaseModel):
    """An evidence-backed topic suitable for blog content."""

    topic: str
    description: str = ""
    evidence_fact_ids: list[str] = Field(default_factory=list)
    supporting_quotes: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)


class HeadingItem(BaseModel):
    """A heading candidate for blog content outline."""

    level: str = "H2"
    text: str
    intent_hint: str = ""


class CompanyInfo(BaseModel):
    name: str
    website: str


class PromptContextHandoff(BaseModel):
    original_prompt: str
    main_topic: str
    search_intent: str
    target_audience: list[str] = Field(default_factory=list)
    prompt_scope: str = "hybrid"
    scope_note: str = ""


class VerificationHandoff(BaseModel):
    is_verified: bool
    intent_valid: bool
    customer_relevant: bool
    reason: str = ""


class LimitedClaimHandoff(BaseModel):
    statement: str
    boundary: str
    fact_id: str


class BlogOutlineSection(BaseModel):
    """An outline section for the blog brief."""

    heading: str
    description: str = ""
    key_points: list[str] = Field(default_factory=list)


class CompanyTalkingPoint(BaseModel):
    """A researched claim about company capabilities with sources and review status."""

    claim: str
    source_urls: list[str] = Field(default_factory=list)
    verification_status: str = "verified"  # "verified" | "needs_human_review"


class Agent4HandoffPackage(BaseModel):
    """Clean, structured research handoff package from Agent 3 to Agent 4."""

    task: str = "Create evidence-backed content brief and strategy"
    topic: str = ""
    search_intent: str = ""
    target_audience: str = ""
    relevant_findings: list[CompanyTalkingPoint] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    information_still_needed: list[ResearchGap] = Field(default_factory=list)
    candidate_subtopics: list[CandidateSubtopic] = Field(default_factory=list)
    boundaries: list[str] = Field(default_factory=list)
    company: CompanyInfo | None = None
    verification: VerificationHandoff | None = None


class GroundTruthGuide(BaseModel):
    """Agent 3 structured research result and handoff to Agent 4."""

    topic: str = ""
    main_topic: str = ""
    search_intent: str = ""
    target_audience: str = ""
    prompt_scope: str = "hybrid"
    relevant_findings: list[CompanyTalkingPoint] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    information_still_needed: list[ResearchGap] = Field(default_factory=list)
    research_gaps: list[ResearchGap] = Field(default_factory=list)
    candidate_subtopics: list[CandidateSubtopic] = Field(default_factory=list)
    verified_facts: list[VerifiedFact] = Field(default_factory=list)
    boundaries: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    verification: PromptVerification | None = None
    agent4_handoff: Agent4HandoffPackage | None = None

    # Internal checking fields used during pipeline processing (excluded from JSON serialization)
    can_write: list[Fact] = Field(default_factory=list, exclude=True)
    can_write_with_limits: list[LimitedStatement] = Field(default_factory=list, exclude=True)
    cannot_write: list[DoNotWrite] = Field(default_factory=list, exclude=True)
    banned_claims: list[str] = Field(default_factory=list, exclude=True)
    scope: ScopeNote | None = Field(default=None, exclude=True)
    potential_headings: list[str] = Field(default_factory=list, exclude=True)
    structured_headings: list[HeadingItem] = Field(default_factory=list, exclude=True)
    related_topics: list[str] = Field(default_factory=list, exclude=True)
    evidence_backed_topics: list[EvidenceBackedTopic] = Field(default_factory=list, exclude=True)





