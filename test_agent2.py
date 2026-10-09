"""
Test Agent 2 - Source Intelligence

Uses:
- real AImpact citation data
- Agent 1 output

Prompt:
campaign 81
prompt 1409
"""

from __future__ import annotations

import json
import os

from aca.agents.source_intelligence import (
    SourceIntelligenceAgent,
)
from aca.tools.aimpact import (
    get_prompt_evidence,
)


CAMPAIGN_ID = 81
PROMPT_ID = "1409"

AGENT1_OUTPUT_FILE = (
    "agent1_output.json"
)

OUTPUT_FILE = (
    "agent2_output.json"
)


def main() -> None:

    print("\nLoading Agent 1 output...")

    with open(
        AGENT1_OUTPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        prompt_analysis = json.load(file)

    print(
        "Agent 1 output loaded."
    )

    print("\nLoading raw AImpact citation data...")

    prompt_data = get_prompt_evidence(
        campaign_id=CAMPAIGN_ID,
    )

    # get_prompt_evidence returns PromptEvidence
    # objects for the campaign.

    prompt_evidence = None

    for item in prompt_data:

        if str(item.prompt_id) == PROMPT_ID:
            prompt_evidence = item
            break

    if prompt_evidence is None:

        raise RuntimeError(
            f"Prompt {PROMPT_ID} was not found "
            f"in campaign {CAMPAIGN_ID}"
        )

    print(
        f"Prompt found: "
        f"{prompt_evidence.prompt_text}"
    )

    print(
        "\nModels available:"
    )

    for model_name in prompt_evidence.per_model:

        model_evidence = (
            prompt_evidence.per_model[
                model_name
            ]
        )

        print(
            f"  {model_name}: "
            f"{len(model_evidence.sources)} citations"
        )

    print("\nStarting Agent 2...")

    agent = SourceIntelligenceAgent()

    result = agent.run(
        prompt_analysis=prompt_analysis,
        prompt_evidence=prompt_evidence,
    )

    output = result.model_dump(
        mode="json"
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nAgent 2 output saved to:"
        f"\n{os.path.abspath(OUTPUT_FILE)}"
    )

    print("\nSummary")
    print("-" * 50)

    print(
        "Unique AImpact sources:",
        result.total_unique_cited_sources,
    )

    print(
        "Total citation occurrences:",
        result.total_citation_occurrences,
    )

    print(
        "Sources analyzed:",
        result.sources_analyzed,
    )


if __name__ == "__main__":
    main()