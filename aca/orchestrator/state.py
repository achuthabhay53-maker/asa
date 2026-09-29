"""Typed workflow state passed between agents."""
from pydantic import BaseModel, Field
from typing import Any


class CitedSource(BaseModel):
    """A source (URL, page, doc) that a model cited when answering a prompt."""
    model: str                  # chatgpt | claude | gemini | google_ai
    url: str | None = None
    domain: str | None = None
    competitor: str | None = None
    excerpt: str


class ContentGap(BaseModel):
    """Analyzer's output — what the sources reveal and where our content is missing."""
    prompt_id: str
    prompt_text: str
    cited_sources: list[CitedSource] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)   # topics competitors cover that we don't
    target_angle: str                                  # one-line editorial angle for our blog
    hypothesis: str                                    # natural-language gap statement


class Brief(BaseModel):
    title: str
    slug: str
    meta: str
    outline: list[dict[str, Any]]
    faqs: list[dict[str, str]]
    internal_links: list[str]


class Draft(BaseModel):
    markdown: str
    version: int
    word_count: int


class Verdict(BaseModel):
    decision: str            # approve | revise | reject
    notes: list[dict[str, Any]] = Field(default_factory=list)
    brand_voice_score: float


class WorkflowState(BaseModel):
    job_id: str
    tenant_id: int
    campaign_id: int
    gap: ContentGap | None = None
    brief: Brief | None = None
    draft: Draft | None = None
    verdict: Verdict | None = None
    revise_count: int = 0
