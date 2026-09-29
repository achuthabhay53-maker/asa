"""Source Analyzer.

For each low-scoring prompt from AImpact, this agent inspects the sources that
ChatGPT / Claude / Gemini / Google AI cite when answering that prompt, identifies
the themes competitors are covering that we're not, and produces a content-gap
hypothesis with an editorial angle the Brief agent can build on.

Input:  state.prompts   (list of prompts + per-model responses + cited sources)
Output: state.gap       (ContentGap: cited_sources, themes, target_angle, hypothesis)
"""
from aca.orchestrator.state import WorkflowState


async def run(state: WorkflowState) -> WorkflowState:
    raise NotImplementedError("implement source analyzer")
