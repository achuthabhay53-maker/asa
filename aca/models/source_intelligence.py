"""Pydantic classes for Agent 2, Source Intelligence."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ContentStructure(BaseModel):
    headings: list[str] = Field(default_factory=list)
    has_faq: bool = False
    has_tables: bool = False
    has_lists: bool = False


class WritingStyle(BaseModel):
    tone: str = ""
    format: str = ""
    audience: str = ""
    paragraph_style: str = ""
    evidence_style: str = ""


class ContentAnalysis(BaseModel):
    summary: str = ""
    main_topics: list[str] = Field(default_factory=list)
    findings: list["SourceFinding"] = Field(default_factory=list)
    content_structure: ContentStructure = Field(default_factory=ContentStructure)
    writing_style: WritingStyle = Field(default_factory=WritingStyle)


class SourceFinding(BaseModel):
    finding: str
    evidence_quote: str = ""
    quote_found_on_page: bool = False


class RankedSource(BaseModel):
    source_id: str
    rank: int
    url: str
    title: str | None = None
    citation_count: int
    cited_by_models: list[str] = Field(default_factory=list)


class AnalyzedSource(BaseModel):
    source_id: str
    url: str
    title: str | None = None
    fetch_status: str = "success"
    content_analysis: ContentAnalysis


class RetrySource(BaseModel):
    source_id: str
    url: str
    title: str | None = None
    fetch_status: str


class CitationSummary(BaseModel):
    total_citation_occurrences: int = 0
    unique_sources: int = 0


class SourceIntelligence(BaseModel):
    """Agent 2 result."""

    campaign_id: int | str
    prompt_id: int | str
    citation_summary: CitationSummary
    ranked_sources: list[RankedSource] = Field(default_factory=list)
    content_analyses: list[AnalyzedSource] = Field(default_factory=list)
    retry_sources: list[RetrySource] = Field(default_factory=list)
