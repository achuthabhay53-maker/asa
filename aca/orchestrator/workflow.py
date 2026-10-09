"""LangGraph workflow for the six ACA agents.

Flow::

    START -> agent1 -> (agent2 | agent3 in parallel) -> agent4
    agent4 --write--> agent5 -> agent6
           --ask----> ask_org -> agent3_refresh -> agent4
           --stop---> finish_needs_data -> END
    agent6 --approve--> human_approval -> publish -> schedule_reaudit -> END
           --revise---> agent5            (up to max_revisions)
           --reject---> agent4 | escalate (up to max_revisions)
    escalate -> human_approval

Design rules:
* Routing functions only read state. Anything they depend on (revision
  counters, statuses) is written by nodes.
* Agent 6 verifies claims by calling the Agent 3 verify() tool inside its own
  node. That is a tool call, not a graph edge.
* ``agent3_refresh`` reuses the Agent 3 callable. A separate node is used
  because the (agent2, agent3) -> agent4 join fires once; re-running agent3
  alone through the join would never trigger agent4 a second time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Protocol, Union

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from aca.orchestrator.agent_schemas import (
    ArticleDraft,
    ClaimRequest,
    ContentBrief,
    FactSet,
    OrgInputs,
    Publication,
    PromptUnderstanding,
    ReauditPlan,
    ReviewVerdict,
    SourceIntelligence,
)
from aca.orchestrator.state import PromptEvidence

Status = Literal[
    "running",
    "needs_data",
    "needs_human",
    "approved",
    "published",
    "rejected",
    "failed",
]


# ---------------------------------------------------------------------------
# Workflow state
# ---------------------------------------------------------------------------


class ACAState(BaseModel):
    """State passed between nodes. Nodes return partial dict updates."""

    # Identity and inputs
    job_id: str
    tenant_id: int
    campaign_id: int
    company_name: str
    org: OrgInputs = Field(default_factory=OrgInputs)
    prompts: list[PromptEvidence] = Field(default_factory=list)

    # Agent outputs
    understanding: PromptUnderstanding | None = None  # agent 1
    source_intel: SourceIntelligence | None = None  # agent 2
    facts: FactSet | None = None  # agent 3
    brief: ContentBrief | None = None  # agent 4
    draft: ArticleDraft | None = None  # agent 5
    verdict: ReviewVerdict | None = None  # agent 6

    # Loop controls
    revise_count: int = 0
    org_rounds: int = 0
    max_revisions: int = 2
    max_org_rounds: int = 2

    # Human approval (set by the UI through update_state before resuming)
    approval: Literal["pending", "approved", "rejected"] = "pending"
    approval_notes: str | None = None

    # After approval
    publication: Publication | None = None
    reaudit: ReauditPlan | None = None

    status: Status = "running"
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Node plumbing
# ---------------------------------------------------------------------------

NodeFn = Callable[[ACAState], Union[dict[str, Any], Awaitable[dict[str, Any]]]]


class ClaimVerifier(Protocol):
    """Agent 3's verify() tool, injected into the Agent 6 implementation."""

    def __call__(
        self, claims: list[ClaimRequest], org: OrgInputs
    ) -> FactSet: ...


@dataclass
class AgentNodes:
    """Plug your real implementations in here."""

    agent1: NodeFn  # prompt + answer intelligence
    agent2: NodeFn  # source intelligence
    agent3: NodeFn  # ground truth + verification
    agent4: NodeFn  # strategy + brief
    agent5: NodeFn  # writer
    agent6: NodeFn  # reviewer (use apply_verdict for its return value)
    human_approval: NodeFn  # usually a no-op; the UI sets state.approval
    publish: NodeFn
    schedule_reaudit: NodeFn


def apply_verdict(state: ACAState, verdict: ReviewVerdict) -> dict[str, Any]:
    """Return value for the Agent 6 node.

    Increments ``revise_count`` for both ``revise`` and ``reject`` so a
    reject -> brief -> write -> reject cycle cannot run forever.
    """
    update: dict[str, Any] = {"verdict": verdict}
    if verdict.decision in ("revise", "reject"):
        update["revise_count"] = state.revise_count + 1
    return update


def ask_org_node(state: ACAState) -> dict[str, Any]:
    """Pause point. Compile with interrupt_before=["ask_org"].

    The UI shows ``brief.questions_for_org``, stores the answers in
    ``state.org.confirmations`` with graph.update_state(), then resumes.
    """
    return {"org_rounds": state.org_rounds + 1}


def _set_status(status: Status) -> Callable[[ACAState], dict[str, Any]]:
    def node(state: ACAState) -> dict[str, Any]:
        return {"status": status}

    return node


# ---------------------------------------------------------------------------
# Routing (read-only)
# ---------------------------------------------------------------------------


def route_after_brief(
    state: ACAState,
) -> Literal["agent5", "ask_org", "finish_needs_data"]:
    brief = state.brief
    if brief is None:
        return "finish_needs_data"

    if brief.readiness == "write" and brief.action != "needs_data":
        return "agent5"

    if brief.readiness == "ask" and state.org_rounds < state.max_org_rounds:
        return "ask_org"

    return "finish_needs_data"


def route_after_review(
    state: ACAState,
) -> Literal["human_approval", "agent5", "agent4", "escalate"]:
    verdict = state.verdict
    if verdict is None:
        return "escalate"

    if verdict.decision == "approve":
        return "human_approval"

    # revise_count already includes this verdict (see apply_verdict).
    if state.revise_count > state.max_revisions:
        return "escalate"

    if verdict.decision == "revise":
        return "agent5"

    # reject
    if verdict.reject_target == "strategy":
        return "agent4"
    return "escalate"


def route_after_approval(
    state: ACAState,
) -> Literal["publish", "finish_rejected"]:
    if state.approval == "approved":
        return "publish"
    return "finish_rejected"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph(nodes: AgentNodes, checkpointer: Any | None = None):
    """Build and compile the workflow.

    A checkpointer is required for the human pauses to work (for example
    langgraph.checkpoint.postgres.PostgresSaver in production, MemorySaver in
    tests).
    """
    graph = StateGraph(ACAState)

    graph.add_node("agent1", nodes.agent1)
    graph.add_node("agent2", nodes.agent2)
    graph.add_node("agent3", nodes.agent3)
    graph.add_node("agent3_refresh", nodes.agent3)
    graph.add_node("agent4", nodes.agent4)
    graph.add_node("agent5", nodes.agent5)
    graph.add_node("agent6", nodes.agent6)
    graph.add_node("ask_org", ask_org_node)
    graph.add_node("human_approval", nodes.human_approval)
    graph.add_node("escalate", _set_status("needs_human"))
    graph.add_node("publish", nodes.publish)
    graph.add_node("schedule_reaudit", nodes.schedule_reaudit)
    graph.add_node("finish_needs_data", _set_status("needs_data"))
    graph.add_node("finish_rejected", _set_status("rejected"))

    # Understand: agent 1, then agents 2 and 3 in parallel, joined at agent 4.
    graph.add_edge(START, "agent1")
    graph.add_edge("agent1", "agent2")
    graph.add_edge("agent1", "agent3")
    graph.add_edge(["agent2", "agent3"], "agent4")

    # Brief decides: write, ask the organization, or stop.
    graph.add_conditional_edges(
        "agent4",
        route_after_brief,
        {
            "agent5": "agent5",
            "ask_org": "ask_org",
            "finish_needs_data": "finish_needs_data",
        },
    )
    graph.add_edge("ask_org", "agent3_refresh")
    graph.add_edge("agent3_refresh", "agent4")
    graph.add_edge("finish_needs_data", END)

    # Write and review loop.
    graph.add_edge("agent5", "agent6")
    graph.add_conditional_edges(
        "agent6",
        route_after_review,
        {
            "human_approval": "human_approval",
            "agent5": "agent5",
            "agent4": "agent4",
            "escalate": "escalate",
        },
    )
    graph.add_edge("escalate", "human_approval")

    # Human approval, publish, schedule the re-audit.
    graph.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"publish": "publish", "finish_rejected": "finish_rejected"},
    )
    graph.add_edge("finish_rejected", END)
    graph.add_edge("publish", "schedule_reaudit")
    graph.add_edge("schedule_reaudit", END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["ask_org", "human_approval"],
    )