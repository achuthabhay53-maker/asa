"""Emits structured brief JSON: title, outline, meta, FAQs, internal links."""
from aca.orchestrator.state import WorkflowState


async def run(state: WorkflowState) -> WorkflowState:
    raise NotImplementedError("implement brief agent")
