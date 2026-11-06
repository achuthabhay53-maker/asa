"""Models for the standalone, non-AImpact blog generation workflow."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field, field_validator

from src.models.reviewer import ReviewOutput
from src.models.token_usage import ClaudeTokenUsage
from src.models.writer import WriterInput, WriterOutput


def normalize_http_url(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("URL must not be empty.")
    if "://" not in value:
        value = f"https://{value}"
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("URL must be a valid public HTTP or HTTPS URL.")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, parsed.path or "/", parsed.query, ""))


class GeneralBlogRequest(BaseModel):
    """User-provided inputs for a general blog."""

    company_name: str = Field(min_length=1, max_length=200)
    company_website: str
    prompt: str = Field(min_length=3, max_length=12_000)
    target_audience: str | None = Field(default=None, max_length=500)
    tone: str | None = Field(default=None, max_length=300)
    reference_urls: list[str] = Field(default_factory=list, max_length=20)
    target_word_count: int | None = Field(default=None, gt=0, le=10_000)

    @field_validator("company_name", "prompt", mode="before")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Value must be text.")
        return value.strip()

    @field_validator("company_website")
    @classmethod
    def normalize_company_website(cls, value: str) -> str:
        return normalize_http_url(value)

    @field_validator("reference_urls")
    @classmethod
    def normalize_reference_urls(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            url = normalize_http_url(value)
            if url not in seen:
                seen.add(url)
                normalized.append(url)
        return normalized


class GeneralBlogPlan(BaseModel):
    """Research and content plan produced from the user request."""

    article_objective: str
    search_intent: str
    target_audience: str
    tone: str
    title: str
    primary_keyword: str
    secondary_keywords: list[str] = Field(default_factory=list)
    related_topics: list[str] = Field(default_factory=list)
    question_based_queries: list[str] = Field(default_factory=list)
    research_queries: list[str] = Field(min_length=1, max_length=6)


EvidenceRelevance = Literal["direct", "related", "background"]
UploadedPDFStatusValue = Literal[
    "parsed_with_evidence",
    "parsed_no_evidence",
    "failed",
]


class GeneralBlogSection(BaseModel):
    """A brief section with research-backed writing guidance."""

    heading: str
    objective: str
    topics: list[str] = Field(default_factory=list)
    questions_to_answer: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    guidelines: list[str] = Field(default_factory=list)


class GeneralBlogBrief(BaseModel):
    """Writer-ready brief compiled from the plan and validated evidence."""

    title: str
    article_objective: str
    target_audience: str
    search_intent: str
    tone: str
    primary_keyword: str
    secondary_keywords: list[str] = Field(default_factory=list)
    related_topics: list[str] = Field(default_factory=list)
    question_based_queries: list[str] = Field(default_factory=list)
    sections: list[GeneralBlogSection] = Field(min_length=1, max_length=10)
    faq_questions: list[str] = Field(default_factory=list, max_length=5)
    unsupported_claims_to_avoid: list[str] = Field(default_factory=list)


class GeneralBlogEvidence(BaseModel):
    """Source evidence; verified means the quote matched, not that the claim is true."""

    evidence_id: str
    claim: str
    status: Literal["verified", "inconclusive", "unavailable"] = Field(
        default="verified",
        description=(
            "For General Blog research, verified means the quote matched the supplied "
            "source text; it does not independently establish the claim's accuracy."
        ),
    )
    relevance: EvidenceRelevance = Field(
        default="direct",
        description="How directly the claim supports the requested blog topic.",
    )
    evidence_quote: str | None = None
    source_url: str | None = None
    wording_boundary: str | None = None
    source_title: str | None = None
    publication_date: str | None = None
    source_type: Literal["company_website", "user_reference"]


class UploadedPDFStatus(BaseModel):
    """Visible processing outcome for a user-uploaded research PDF."""

    file_name: str
    status: UploadedPDFStatusValue
    extracted_text_characters: int = 0
    evidence_count: int = 0
    message: str


class GeneralBlogResult(BaseModel):
    """Complete result from the General Blog workflow."""

    plan: GeneralBlogPlan
    brief: GeneralBlogBrief
    writer_input: WriterInput
    draft: WriterOutput
    review: ReviewOutput
    sources: list[GeneralBlogEvidence] = Field(default_factory=list)
    uploaded_pdf_statuses: list[UploadedPDFStatus] = Field(default_factory=list)
    quality_passed: bool
    review_current: bool = True
    review_cycles: int = 0
    warnings: list[str] = Field(default_factory=list)
    token_usage: list[ClaudeTokenUsage] = Field(default_factory=list)
