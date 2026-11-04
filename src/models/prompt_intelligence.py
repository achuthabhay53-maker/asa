"""Pydantic classes for Agent 1, Prompt & Answer Intelligence."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PromptUnderstanding(BaseModel):
    intent: str
    topic: str
    audience: str | None = None
    user_goal: str
    key_questions: list[str] = Field(default_factory=list)


class AnswerUnderstanding(BaseModel):
    summary: str
    key_topics: list[str] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)
    comparison_criteria: list[str] = Field(default_factory=list)


class CompanyMention(BaseModel):
    name: str
    context: str


class CustomerVisibility(BaseModel):
    status: str = "unknown"
    models_mentioning_customer: list[str] = Field(default_factory=list)
    context: str | None = None


class CrossModelPatterns(BaseModel):
    common_topics: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)


class PromptAnalysis(BaseModel):
    """Agent 1 result."""

    prompt_id: str
    prompt: str
    customer_name: str
    prompt_understanding: PromptUnderstanding
    answer_understanding: AnswerUnderstanding
    companies_mentioned: list[CompanyMention] = Field(default_factory=list)
    customer_visibility: CustomerVisibility
    cross_model_patterns: CrossModelPatterns = Field(default_factory=CrossModelPatterns)
    content_opportunities: list[str] = Field(default_factory=list)
