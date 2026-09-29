"""Agent protocol. Each agent = one function: state -> state."""
from typing import Protocol
from aca.orchestrator.state import WorkflowState


class Agent(Protocol):
    async def run(self, state: WorkflowState) -> WorkflowState: ...
