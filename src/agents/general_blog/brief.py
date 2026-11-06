"""Compile a general-blog brief and evidence into the existing Writer input."""

from __future__ import annotations

import os

import anthropic

from src.models.general_blog import (
    GeneralBlogBrief,
    GeneralBlogEvidence,
    GeneralBlogPlan,
    GeneralBlogRequest,
)
from src.models.writer import WriterEvidence, WriterInput, WriterSectionInput
from src.utils.general_blog import call_structured_agent


class GeneralBlogBriefAgent:
    """Create an evidence-mapped brief without using ACA/AImpact context."""

    def __init__(
        self,
        model: str | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if client is None and not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")
        self.client = client or anthropic.Anthropic(api_key=api_key)
        self.model = model or os.getenv("ANTHROPIC_MODEL_NAME") or "claude-haiku-4-5-20251001"

    def build(
        self,
        request: GeneralBlogRequest,
        plan: GeneralBlogPlan,
        evidence: list[GeneralBlogEvidence],
    ) -> tuple[GeneralBlogBrief, WriterInput]:
        data = call_structured_agent(
            self.client,
            self.model,
            """Build a concise, reader-focused blog brief from the user request,
research plan, and evidence package. Map each factual section to only evidence
IDs that support it. Every factual claim about the company must use company-site
evidence. User-provided papers may support research findings but must retain
their population, method, limitations, and wording boundaries. Do not invent
citations, search metrics, company claims, outcomes, or statistics. General
education may be included without company-specific evidence, but clearly keep
it distinct from factual company claims. Use the requested tone and audience.
Prioritize evidence marked direct for the main article claims. Use related
evidence only as a clearly identified supporting example, without implying it
proves the main topic. Use background evidence only when the requested article
explicitly discusses that context, and explain its limits. Evidence status
"verified" means its quote matched the supplied source text; it does not
independently establish that the claim is scientifically correct.
Return between 1 and 10 sections total; never return more than 10. Combine
closely related topics into one section when needed to stay within this limit.
Return 3 to 5 FAQs that can be answered from the planned article.""",
            {
                "request": request.model_dump(mode="json"),
                "plan": plan.model_dump(mode="json"),
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "allowed_evidence_ids": [item.evidence_id for item in evidence],
            },
            GeneralBlogBrief,
            "submit_general_blog_brief",
            max_tokens=5000,
        )
        brief = GeneralBlogBrief.model_validate(data)
        allowed_ids = {item.evidence_id for item in evidence}
        unknown_ids = {
            evidence_id
            for section in brief.sections
            for evidence_id in section.evidence_ids
            if evidence_id not in allowed_ids
        }
        if unknown_ids:
            raise RuntimeError(
                "General Blog brief referenced unknown evidence IDs: "
                + ", ".join(sorted(unknown_ids))
            )
        evidence_by_id = {item.evidence_id: item for item in evidence}
        writer_input = WriterInput(
            company_name=request.company_name,
            company_url=request.company_website,
            title=brief.title,
            article_objective=brief.article_objective,
            target_audience=brief.target_audience,
            search_intent=brief.search_intent,
            primary_keyword=brief.primary_keyword,
            secondary_keywords=brief.secondary_keywords,
            related_topics=brief.related_topics,
            question_based_queries=brief.question_based_queries,
            sections=[
                WriterSectionInput(
                    heading=section.heading,
                    objective=section.objective,
                    topics=section.topics,
                    questions_to_answer=section.questions_to_answer,
                    evidence_ids=section.evidence_ids,
                    guidelines=section.guidelines,
                )
                for section in brief.sections
            ],
            faqs=[{"question": question, "answer": ""} for question in brief.faq_questions],
            writing_style={
                "tone": brief.tone,
                "audience_level": brief.target_audience,
            },
            unsupported_claims_to_avoid=brief.unsupported_claims_to_avoid,
            evidence=[
                WriterEvidence(
                    evidence_id=item.evidence_id,
                    claim=item.claim,
                    status=item.status,
                    relevance=item.relevance,
                    evidence_quote=item.evidence_quote,
                    source_url=item.source_url,
                    wording_boundary=item.wording_boundary,
                )
                for item in evidence_by_id.values()
            ],
            target_word_count=request.target_word_count,
        )
        return brief, writer_input
