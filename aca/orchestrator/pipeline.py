"""LangGraph flow for the five-agent content workflow.

    agent1 -> agent2 -> agent3 -> agent4 -> agent5

The workflow resumes at the first incomplete agent. The website is the
caller's input.
"""
from __future__ import annotations

import re
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from aca.agents.content_brief import ContentBriefAgent
from aca.agents.ground_truth import MAX_QUESTIONS, GroundTruthAgent
from aca.agents.prompt_intelligence import run_agent1
from aca.agents.source_intelligence import TOP_N_SOURCES, run_agent2
from aca.agents.writer import WriterAgent
from aca.models.content_brief import ContentBrief
from aca.models.ground_truth import ClientInput, GroundTruthGuide
from aca.models.prompt_intelligence import PromptAnalysis
from aca.models.source_intelligence import SourceIntelligence
from aca.models.writer import (
    WriterEvidence,
    WriterInput,
    WriterOutput,
    WriterSectionInput,
)


class PipelineState(BaseModel):
    campaign_id: int
    prompt_id: str
    website: str
    company_name: str = ""
    top_n: int = TOP_N_SOURCES
    max_questions: int = MAX_QUESTIONS
    target_word_count: int | None = None
    prompt_analysis: PromptAnalysis | None = None
    source_analysis: SourceIntelligence | None = None
    guide: GroundTruthGuide | None = None
    content_brief: ContentBrief | None = None
    written_article: WriterOutput | None = None
    errors: list[str] = Field(default_factory=list)


def agent1_node(state: PipelineState) -> dict[str, Any]:
    result = run_agent1(
        state.campaign_id,
        state.prompt_id,
        state.company_name.strip() or None,
    )
    return {"prompt_analysis": result, "company_name": result.customer_name}


def agent2_node(state: PipelineState) -> dict[str, Any]:
    result = run_agent2(state.campaign_id, state.prompt_id, top_n=state.top_n)
    return {"source_analysis": result}


def agent3_node(state: PipelineState) -> dict[str, Any]:
    if not state.website.strip():
        return {"errors": ["A website is required."]}
    if state.prompt_analysis is None:
        return {"errors": ["Agent 1 output is missing."]}
    name = state.company_name.strip() or state.prompt_analysis.customer_name
    guide = GroundTruthAgent(max_questions=state.max_questions).analyze(
        state.prompt_analysis,
        state.source_analysis,
        ClientInput(company_name=name, website=state.website),
    )
    if guide is None:
        return {
            "guide": None,
            "errors": [
                "Prompt intent is invalid or customer is not relevant to the requested topic. Stopped as no blog content can be written."
            ],
        }
    if guide.verification is None or not guide.verification.is_verified:
        reason = guide.verification.reason if guide.verification else "Verification result is missing."
        return {
            "guide": guide,
            "errors": [f"Agent 3 verification failed. Agents 4 and 5 were not run. {reason}"],
        }
    return {"guide": guide}


def agent4_node(state: PipelineState) -> dict[str, Any]:
    if state.prompt_analysis is None or state.source_analysis is None or state.guide is None:
        raise RuntimeError("Agents 1–3 outputs are required before running Agent 4.")
    if state.guide.verification is None or not state.guide.verification.is_verified:
        raise RuntimeError("Agent 3 verification must pass before running Agent 4.")

    brief = ContentBriefAgent().build_brief(
        state.prompt_analysis,
        state.source_analysis,
        state.guide,
    )
    if not brief.sections:
        return {
            "content_brief": brief,
            "errors": ["Agent 4 returned no usable sections. Agent 5 was not run."],
        }
    return {"content_brief": brief}


def _build_writer_input(state: PipelineState) -> WriterInput:
    if state.prompt_analysis is None or state.guide is None or state.content_brief is None:
        raise RuntimeError("Agents 1, 3, and 4 outputs are required before running Agent 5.")

    brief = state.content_brief
    guide = state.guide
    facts_by_evidence: dict[str, list[Any]] = {}
    for fact in guide.verified_facts:
        for evidence_id in fact.evidence_ids:
            facts_by_evidence.setdefault(evidence_id, []).append(fact)

    evidence: list[WriterEvidence] = []
    for item in guide.evidence:
        evidence_id = item.evidence_id.strip()
        if not evidence_id:
            continue
        related_facts = facts_by_evidence.get(evidence_id, [])
        fact = related_facts[0] if related_facts else None
        status = fact.status.lower() if fact else "inconclusive"
        if status not in {"verified", "contradicted", "not_found", "inconclusive"}:
            status = "inconclusive"
        evidence.append(
            WriterEvidence(
                evidence_id=evidence_id,
                claim=fact.statement if fact else item.quote,
                status=status,
                evidence_quote=item.quote,
                source_url=item.url,
            )
        )

    sections: list[WriterSectionInput] = []
    for section in brief.sections:
        match = re.search(r"\d+", section.level)
        level = min(max(int(match.group()) if match else 2, 1), 6)
        sections.append(
            WriterSectionInput(
                heading=section.heading,
                level=level,
                objective=section.section_objective,
                topics=section.topics_and_questions,
                questions_to_answer=[
                    question for question in section.topics_and_questions if "?" in question
                ],
                evidence_ids=section.evidence_ids,
                guidelines=section.guidelines_and_gaps,
            )
        )

    writing_style = brief.writing_style.model_dump(mode="json")
    return WriterInput(
        company_name=state.company_name.strip() or state.prompt_analysis.customer_name,
        company_url=state.website,
        title=brief.recommended_title or state.prompt_analysis.prompt,
        article_objective=brief.article_objective,
        target_audience=brief.target_audience or guide.target_audience,
        search_intent=brief.search_intent or guide.search_intent,
        primary_keyword=brief.primary_keyword or guide.main_topic,
        secondary_keywords=brief.secondary_keywords,
        related_topics=brief.related_topics,
        question_based_queries=brief.question_based_queries,
        sections=sections,
        faqs=[{"question": question, "answer": ""} for question in brief.faq_questions],
        writing_style={
            key: ", ".join(value) if isinstance(value, list) else str(value)
            for key, value in writing_style.items()
        },
        unsupported_claims_to_avoid=brief.unsupported_claims_to_avoid,
        evidence=evidence,
        target_word_count=state.target_word_count,
    )


def agent5_node(state: PipelineState) -> dict[str, Any]:
    writer_input = _build_writer_input(state)
    if not writer_input.sections:
        raise RuntimeError("Agent 4 returned no usable sections. Agent 5 was not run.")
    return {"written_article": WriterAgent().write(writer_input)}


def _entry(state: PipelineState) -> str:
    if state.prompt_analysis is None:
        return "agent1"
    if state.source_analysis is None:
        return "agent2"
    if state.guide is None:
        return "agent3"
    if state.guide.verification is None or not state.guide.verification.is_verified:
        return END
    if state.content_brief is None:
        return "agent4"
    if not state.content_brief.sections:
        return END
    if state.written_article is None:
        return "agent5"
    return END


def _after_agent3(state: PipelineState) -> str:
    if (
        state.guide is None
        or state.guide.verification is None
        or not state.guide.verification.is_verified
    ):
        return END
    return "agent4"


def _after_agent4(state: PipelineState) -> str:
    if state.content_brief is None or not state.content_brief.sections:
        return END
    return "agent5"


def build_pipeline():
    """Compile the five-agent content workflow."""
    graph = StateGraph(PipelineState)
    graph.add_node("agent1", agent1_node)
    graph.add_node("agent2", agent2_node)
    graph.add_node("agent3", agent3_node)
    graph.add_node("agent4", agent4_node)
    graph.add_node("agent5", agent5_node)
    graph.add_conditional_edges(
        START,
        _entry,
        ["agent1", "agent2", "agent3", "agent4", "agent5", END],
    )
    graph.add_edge("agent1", "agent2")
    graph.add_edge("agent2", "agent3")
    graph.add_conditional_edges("agent3", _after_agent3, ["agent4", END])
    graph.add_conditional_edges("agent4", _after_agent4, ["agent5", END])
    graph.add_edge("agent5", END)
    return graph.compile()


def run_workflow(
    campaign_id: int,
    prompt_id: str | int,
    website: str,
    company_name: str = "",
    top_n: int = TOP_N_SOURCES,
    max_questions: int = MAX_QUESTIONS,
    target_word_count: int | None = None,
) -> PipelineState:
    """Run the entire workflow from a single input: campaign_id, prompt_id, website."""
    initial_state = PipelineState(
        campaign_id=int(campaign_id),
        prompt_id=str(prompt_id),
        website=website.strip(),
        company_name=company_name.strip(),
        top_n=top_n,
        max_questions=max_questions,
        target_word_count=target_word_count,
    )
    pipeline = build_pipeline()
    final_dict = pipeline.invoke(initial_state)
    return PipelineState.model_validate(final_dict)
