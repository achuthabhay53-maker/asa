"""Scores draft against brand-voice rubric. Verdict: approve | revise | reject."""
from aca.orchestrator.state import WorkflowState


async def run(state: WorkflowState) -> WorkflowState:
    raise NotImplementedError("implement reviewer agent")
