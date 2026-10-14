"""LangGraph flow for Agents 1, 2, and 3.

    agent1 -> agent2 -> agent3

Agent 3 also runs on its own when Agent 1 and Agent 2 results are already
on the state. The website is the caller's input.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from aca.agents.ground_truth import MAX_QUESTIONS, GroundTruthAgent
from aca.agents.prompt_intelligence import run_agent1
from aca.agents.source_intelligence import TOP_N_SOURCES, run_agent2
from aca.models.ground_truth import ClientInput, GroundTruthGuide
from aca.models.prompt_intelligence import PromptAnalysis
from aca.models.source_intelligence import SourceIntelligence


class PipelineState(BaseModel):
    campaign_id: int
    prompt_id: str
    website: str
    company_name: str = ""
    top_n: int = TOP_N_SOURCES
    max_questions: int = MAX_QUESTIONS
    prompt_analysis: PromptAnalysis | None = None
    source_analysis: SourceIntelligence | None = None
    guide: GroundTruthGuide | None = None
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
    return {"guide": guide}


def _entry(state: PipelineState) -> str:
    if state.prompt_analysis is None:
        return "agent1"
    if state.source_analysis is None:
        return "agent2"
    return "agent3"


def build_pipeline():
    """Compile the Agent 1, 2, and 3 graph."""
    graph = StateGraph(PipelineState)
    graph.add_node("agent1", agent1_node)
    graph.add_node("agent2", agent2_node)
    graph.add_node("agent3", agent3_node)
    graph.add_conditional_edges(START, _entry, ["agent1", "agent2", "agent3"])
    graph.add_edge("agent1", "agent2")
    graph.add_edge("agent2", "agent3")
    graph.add_edge("agent3", END)
    return graph.compile()


def run_workflow(
    campaign_id: int,
    prompt_id: str | int,
    website: str,
    company_name: str = "",
    top_n: int = TOP_N_SOURCES,
    max_questions: int = MAX_QUESTIONS,
) -> PipelineState:
    """Run the entire workflow from a single input: campaign_id, prompt_id, website."""
    initial_state = PipelineState(
        campaign_id=int(campaign_id),
        prompt_id=str(prompt_id),
        website=website.strip(),
        company_name=company_name.strip(),
        top_n=top_n,
        max_questions=max_questions,
    )
    pipeline = build_pipeline()
    final_dict = pipeline.invoke(initial_state)
    return PipelineState.model_validate(final_dict)
