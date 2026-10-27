"""Pydantic models for the optional Agent 6 review step."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from aca.models.writer import WriterClaim


class ReviewFinding(BaseModel):
    area: Literal["grammar", "formatting", "headings", "content", "keywords", "seo", "factual_claims"]
    severity: Literal["critical", "high", "medium", "low"]
    issue: str
    recommendation: str


class RevisedBlog(BaseModel):
    title: str
    seo_title: str
    meta_description: str
    article_markdown: str
    claims_used: list[WriterClaim] = Field(default_factory=list)
    unresolved_items: list[str] = Field(default_factory=list)
    brief_coverage: list[str] = Field(default_factory=list)


class ReviewOutput(BaseModel):
    summary: str
    findings: list[ReviewFinding] = Field(default_factory=list)
    unresolved_items: list[str] = Field(default_factory=list)


class ReviewerResponse(BaseModel):
    """Reviewer response containing findings for human follow-up."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    findings: list[ReviewFinding] = Field(default_factory=list, max_length=5)
    unresolved_items: list[str] = Field(default_factory=list, max_length=10)
