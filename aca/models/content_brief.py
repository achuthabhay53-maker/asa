"""Pydantic classes for Agent 4: Strategy & Blog Brief."""
from __future__ import annotations

from pydantic import BaseModel, Field


class BriefSection(BaseModel):
    """A single structured section in the content brief."""

    heading: str
    level: str = "H2"
    section_objective: str = ""
    topics_and_questions: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    guidelines_and_gaps: list[str] = Field(default_factory=list)


class WritingStyleGuide(BaseModel):
    """Writing and formatting guidelines for Agent 5 (Writer)."""

    tone: str = "Professional, authoritative, peer-to-peer scientific"
    audience_level: str = "Technical / scientific (assumes assay & biomarker familiarity)"
    paragraph_style: str = "Concise, capability-focused paragraphs (3-4 sentences) with bulleted lists"
    formatting_elements: list[str] = Field(default_factory=list)
    terminology: list[str] = Field(default_factory=list)


class ContentBrief(BaseModel):
    """The complete structured content brief produced by Agent 4 for Agent 5 (Writer)."""

    article_objective: str = ""
    target_audience: str = ""
    search_intent: str = ""
    primary_keyword: str = ""
    secondary_keywords: list[str] = Field(default_factory=list)
    related_topics: list[str] = Field(default_factory=list)
    question_based_queries: list[str] = Field(default_factory=list)
    faq_questions: list[str] = Field(default_factory=list)
    recommended_title: str = ""
    alternative_titles: list[str] = Field(default_factory=list)
    sections: list[BriefSection] = Field(default_factory=list)
    writing_style: WritingStyleGuide = Field(default_factory=WritingStyleGuide)
    unsupported_claims_to_avoid: list[str] = Field(default_factory=list)
