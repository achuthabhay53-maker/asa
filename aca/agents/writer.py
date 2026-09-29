"""Expands the brief into Markdown draft + schema.org FAQ JSON-LD."""
from aca.orchestrator.state import WorkflowState


async def run(state: WorkflowState) -> WorkflowState:
    raise NotImplementedError("implement writer agent")
