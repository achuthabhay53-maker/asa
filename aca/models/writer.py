
"""Pydantic models for Agent 5: Writer Agent."""

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class WriterEvidence(BaseModel):
    """Evidence available to the writer for factual claims."""

    evidence_id: str
    claim: str
    status: Literal[
        "verified",
        "model_supported",
        "contradicted",
        "not_found",
        "inconclusive",
        "unavailable",
    ]
    evidence_quote: str | None = None
    source_url: str | None = None


class WriterSectionInput(BaseModel):
    """One section requested by the content brief."""

    heading: str
    level: int = Field(default=2, ge=1, le=6)
    objective: str = ""
    topics: list[str] = Field(default_factory=list)
    questions_to_answer: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    guidelines: list[str] = Field(default_factory=list)


class WriterInput(BaseModel):
    """All inputs required to write an article."""

    company_name: str
    company_url: str | None = None

    title: str
    meta_description: str = ""
    article_objective: str
    target_audience: str
    search_intent: str = ""

    sections: list[WriterSectionInput] = Field(
        default_factory=list
    )
    faqs: list[dict[str, str]] = Field(default_factory=list)
    internal_links: list[str] = Field(default_factory=list)

    writing_style: dict[str, str] = Field(
        default_factory=dict
    )
    unsupported_claims_to_avoid: list[str] = Field(
        default_factory=list
    )
    evidence: list[WriterEvidence] = Field(
        default_factory=list
    )

    target_word_count: int | None = Field(
        default=None,
        gt=0,
    )


class WriterClaim(BaseModel):
    """A material factual claim used in the article."""

    claim: str
    evidence_ids: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)


class WrittenSection(BaseModel):
    """Generated content mapped back to its requested section."""

    heading: str
    content_markdown: str
    evidence_ids_used: list[str] = Field(default_factory=list)


class WriterOutput(BaseModel):
    """Structured output returned by Agent 5."""

    title: str
    meta_description: str = ""
    article_markdown: str
    word_count: int = Field(ge=0)

    sections: list[WrittenSection] = Field(
        default_factory=list
    )
    claims_used: list[WriterClaim] = Field(
        default_factory=list
    )
    unresolved_items: list[str] = Field(
        default_factory=list
    )
    brief_coverage: list[str] = Field(
        default_factory=list
    )
