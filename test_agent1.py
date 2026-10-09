import json

from aca.tools.aimpact import get_prompt_evidence
from aca.agents.prompt_intelligence import PromptAnswerAgent


CAMPAIGN_ID = 81
PROMPT_ID = "1409"


def main():
    print("=" * 70)
    print("AGENT 1 TEST")
    print("=" * 70)

    print("\nLoading AImpact data...")

    prompts = get_prompt_evidence(
        campaign_id=CAMPAIGN_ID,
        score_threshold=70,
    )

    print(f"Found {len(prompts)} prompts.")

    prompt = None

    for item in prompts:
        if str(item.prompt_id) == PROMPT_ID:
            prompt = item
            break

    if prompt is None:
        raise RuntimeError(
            f"Prompt {PROMPT_ID} was not found."
        )

    print("\nPrompt:")
    print(prompt.prompt_text)

    print("\nModels:")
    for model_name in prompt.per_model:
        print(f"  - {model_name}")

    print("\nRunning Agent 1...")

    agent = PromptAnswerAgent()

    result = agent.analyze(prompt)

    print("\n" + "=" * 70)
    print("AGENT 1 OUTPUT")
    print("=" * 70)

    print(
        json.dumps(
            result.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
    )

    # Save output so you can inspect it later.
    output_file = "agent1_output.json"

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result.model_dump(),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print(f"Saved output to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()