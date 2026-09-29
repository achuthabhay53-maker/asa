"""Pushes the approved draft to WordPress as a scheduled post."""
from aca.orchestrator.state import WorkflowState


async def run(state: WorkflowState) -> WorkflowState:
    raise NotImplementedError("implement publisher agent")
