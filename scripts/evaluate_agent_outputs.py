"""Evaluate saved Agent 1–5 outputs with DeepEval and a Claude judge.

Run from the repository root after creating the five agent output JSON files.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from aca.models.content_brief import ContentBrief  # noqa: E402
from aca.models.ground_truth import GroundTruthGuide  # noqa: E402
from aca.models.prompt_intelligence import PromptAnalysis  # noqa: E402
from aca.models.source_intelligence import SourceIntelligence  # noqa: E402
from aca.models.writer import WriterOutput  # noqa: E402


OUTPUT_FILES = {
    "agent1": ROOT / "agent1_output.json",
    "agent2": ROOT / "agent2_output.json",
    "agent3": ROOT / "agent3_output.json",
    "agent4": ROOT / "agent4_output.json",
    "agent5": ROOT / "agent5_output.json",
}
THRESHOLD = 0.7


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _load_agent_outputs(paths: dict[str, Path]) -> dict[str, Any]:
    models = {
        "agent1": PromptAnalysis,
        "agent2": SourceIntelligence,
        "agent3": GroundTruthGuide,
        "agent4": ContentBrief,
        "agent5": WriterOutput,
    }
    outputs = {}
    run_ids: set[str] = set()
    saw_run_id = False
    run_id_count = 0
    for agent, model in models.items():
        path = paths[agent].resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{agent} output file does not exist: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{agent} output must be a JSON object: {path}")
        run_id = raw.get("_run_id")
        if run_id is not None:
            saw_run_id = True
            run_id_count += 1
            if not isinstance(run_id, str) or not run_id.strip():
                raise ValueError(f"{agent} output has an invalid _run_id: {path}")
            run_ids.add(run_id)
        outputs[agent] = model.model_validate(raw)

    if saw_run_id and run_id_count != len(models):
        raise ValueError(
            "Some Agent output files are missing _run_id; "
            "use outputs from one workflow run."
        )
    if saw_run_id and len(run_ids) != 1:
        raise ValueError(
            "Agent output files contain different _run_id values; "
            "use outputs from one workflow run."
        )

    if str(outputs["agent1"].prompt_id) != str(outputs["agent2"].prompt_id):
        raise ValueError(
            "Agent 1 and Agent 2 outputs have different prompt IDs: "
            f"{outputs['agent1'].prompt_id!r} vs {outputs['agent2'].prompt_id!r}"
        )
    handoff = outputs["agent3"].agent4_handoff
    if handoff and handoff.company:
        if outputs["agent1"].customer_name.casefold() != handoff.company.name.casefold():
            raise ValueError("Agent 3 handoff company does not match Agent 1 customer.")

    return outputs


def _create_claude_judge(model_name: str):
    try:
        from anthropic import Anthropic
        from deepeval.models import DeepEvalBaseLLM
        import httpx
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError(
            'Install the optional evaluation dependencies with '
            '".\\.venv-llm-eval\\Scripts\\python.exe -m pip install -e \".[llm-eval]".'
        ) from exc

    class ClaudeJudge(DeepEvalBaseLLM):
        def __init__(self) -> None:
            super().__init__(model_name=model_name)

        def load_model(self, *args: Any, **kwargs: Any):
            del args, kwargs
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not configured.")
            return Anthropic(api_key=api_key, http_client=httpx.Client())

        def get_model_name(self, *args: Any, **kwargs: Any) -> str:
            del args, kwargs
            return model_name

        def generate(self, prompt: str, schema: type[BaseModel] | None = None):
            client = self.model
            if schema is None:
                response = client.messages.create(
                    model=model_name,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                return self._text(response)

            tool_name = "submit_evaluation"
            response = client.messages.create(
                model=model_name,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                tools=[{
                    "name": tool_name,
                    "description": "Return the requested structured evaluation result.",
                    "input_schema": schema.model_json_schema(),
                }],
                tool_choice={"type": "tool", "name": tool_name},
            )
            tool_result = next(
                (block.input for block in response.content if block.type == "tool_use"),
                None,
            )
            if tool_result is None:
                raise RuntimeError(
                    f"Claude did not return the required {schema.__name__} evaluation result."
                )
            return schema.model_validate(tool_result)

        async def a_generate(
            self,
            prompt: str,
            schema: type[BaseModel] | None = None,
        ):
            return await asyncio.to_thread(self.generate, prompt, schema)

        @staticmethod
        def _text(response: Any) -> str:
            text = "\n".join(
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
            ).strip()
            if not text:
                raise RuntimeError("Claude returned an empty judge response.")
            return text

    return ClaudeJudge()


def _evaluate(outputs: dict[str, Any], judge: Any, threshold: float) -> list[dict[str, Any]]:
    try:
        from deepeval.metrics import FaithfulnessMetric, GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    except ImportError as exc:
        raise RuntimeError(
            'Install the optional evaluation dependencies with '
            '".\\.venv-llm-eval\\Scripts\\python.exe -m pip install -e \".[llm-eval]".'
        ) from exc

    prompt = outputs["agent1"].prompt
    prompt_context = outputs["agent1"].prompt_understanding.model_dump(mode="json")
    source_context = [
        {
            "url": source.url,
            "title": source.title,
            "citation_count": source.citation_count,
            "summary": next(
                (
                    analysis.content_analysis.summary
                    for analysis in outputs["agent2"].content_analyses
                    if analysis.source_id == source.source_id
                ),
                "",
            ),
        }
        for source in outputs["agent2"].ranked_sources[:10]
    ]
    guide = outputs["agent3"]
    evidence_context = [
        {"evidence_id": item.evidence_id, "url": item.url, "quote": item.quote}
        for item in guide.evidence
    ]
    brief_context = {
        "title": outputs["agent4"].recommended_title,
        "objective": outputs["agent4"].article_objective,
        "audience": outputs["agent4"].target_audience,
        "search_intent": outputs["agent4"].search_intent,
        "sections": [section.model_dump(mode="json") for section in outputs["agent4"].sections],
        "faq_questions": outputs["agent4"].faq_questions,
        "unsupported_claims_to_avoid": outputs["agent4"].unsupported_claims_to_avoid,
    }

    evaluations = [
        (
            "Agent 1 — prompt understanding and intent",
            "Assess whether the output accurately captures the user's intent, topic, "
            "audience, goal, and key questions from the original prompt. Do not reward "
            "claims about the company that are not present in the input.",
            _json({
                "original_prompt": prompt,
                "customer_name": outputs["agent1"].customer_name,
            }),
            outputs["agent1"].model_dump_json(),
        ),
        (
            "Agent 2 — source relevance and quality",
            "Assess how relevant the selected cited sources are to the original prompt, "
            "whether source details and page summaries are traceable and useful, and "
            "whether quality is assessed from content rather than citation count alone. "
            "Do not treat citation frequency as proof of factual reliability.",
            _json({"original_prompt": prompt, "prompt_understanding": prompt_context}),
            outputs["agent2"].model_dump_json(),
        ),
        (
            "Agent 3 — company facts and evidence accuracy",
            "Assess whether company facts are accurately and narrowly supported by the "
            "provided evidence quotes and source URLs. Penalize invented claims, claims "
            "broader than their quotes, unsupported verification, and incorrect evidence "
            "links. A source URL alone is not proof.",
            _json({
                "original_prompt": prompt,
                "customer": outputs["agent1"].customer_name,
                "source_analysis": source_context,
                "evidence_quotes": evidence_context,
            }),
            guide.model_dump_json(),
        ),
        (
            "Agent 4 — brief completeness and correctness",
            "Assess whether the brief gives the writer a coherent, audience-appropriate "
            "objective, search intent, title, section outline, useful FAQs, SEO topics, "
            "and clear constraints. Check that factual section evidence IDs exist in "
            "the supplied evidence and that the brief follows the prompt rather than "
            "inventing company facts.",
            _json({
                "original_prompt": prompt,
                "customer": outputs["agent1"].customer_name,
                "verified_topic": guide.main_topic or guide.topic,
                "verified_facts": [fact.model_dump(mode="json") for fact in guide.verified_facts],
                "evidence_quotes": evidence_context,
            }),
            outputs["agent4"].model_dump_json(),
        ),
        (
            "Agent 5 — draft quality and factual grounding",
            "Assess whether the article answers the reader's intent, follows the "
            "provided brief, is clear and well-organized, and includes a useful "
            "conclusion and relevant FAQs. Check material company-specific claims "
            "against the provided evidence, penalizing unsupported or overstated claims.",
            _json({
                "original_prompt": prompt,
                "brief": brief_context,
                "verified_facts": [fact.model_dump(mode="json") for fact in guide.verified_facts],
                "evidence_quotes": evidence_context,
            }),
            outputs["agent5"].article_markdown,
        ),
    ]
    results: list[dict[str, Any]] = []
    fixed_steps = [
        "Identify the output's required task and relevant input constraints.",
        "Compare the output with the provided input, brief, and evidence as applicable.",
        "Penalize missing requirements, unsupported assertions, and factual overreach.",
        "Assign a score from 0 to 1 based on how well the output meets the criterion.",
    ]
    for name, criteria, input_text, actual_output in evaluations:
        metric = GEval(
            name=name,
            criteria=criteria,
            evaluation_steps=fixed_steps,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=threshold,
            model=judge,
            async_mode=False,
        )
        metric.measure(
            LLMTestCase(input=input_text, actual_output=actual_output),
            _show_indicator=False,
        )
        results.append({
            "name": name,
            "score": metric.score,
            "threshold": threshold,
            "passed": bool(metric.score is not None and metric.score >= threshold),
            "reason": metric.reason,
        })

    claims = [
        claim.model_dump(mode="json")
        for claim in outputs["agent5"].claims_used
    ]
    if claims:
        faithfulness = FaithfulnessMetric(
            threshold=threshold,
            model=judge,
            include_reason=True,
            async_mode=False,
        )
        faithfulness.measure(
            LLMTestCase(
                input="Check whether the tracked factual claims in the Agent 5 draft are supported.",
                actual_output=_json(claims),
                retrieval_context=[
                    f"{item.url}\n{item.quote}"
                    for item in guide.evidence
                    if item.quote.strip()
                ],
            ),
            _show_indicator=False,
        )
        results.append({
            "name": "Agent 5 — tracked claim faithfulness",
            "score": faithfulness.score,
            "threshold": threshold,
            "passed": bool(
                faithfulness.score is not None and faithfulness.score >= threshold
            ),
            "reason": faithfulness.reason,
        })
    else:
        results.append({
            "name": "Agent 5 — tracked claim faithfulness",
            "score": None,
            "threshold": threshold,
            "passed": False,
            "reason": "Agent 5 output has no claims_used entries to evaluate.",
        })

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score real saved Agent 1–5 JSON outputs with a Claude judge."
    )
    for agent, default in OUTPUT_FILES.items():
        parser.add_argument(
            f"--{agent}",
            type=Path,
            default=default,
            help=f"JSON output from {agent} (default: {default.name})",
        )
    parser.add_argument(
        "--judge-model",
        default=(
            os.getenv("DEEPEVAL_MODEL")
            or os.getenv("ANTHROPIC_MODEL_NAME")
            or "claude-haiku-4-5-20251001"
        ),
        help="Claude model used to judge outputs (defaults to ANTHROPIC_MODEL_NAME).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="Minimum score to pass each evaluation (default: 0.7).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "agent_quality_report.json",
        help="Where to save metric scores and reasons.",
    )
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")

    paths = {agent: getattr(args, agent) for agent in OUTPUT_FILES}
    outputs = _load_agent_outputs(paths)
    judge = _create_claude_judge(args.judge_model)
    try:
        results = _evaluate(outputs, judge, args.threshold)
    finally:
        judge.model.close()

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": args.judge_model,
        "threshold": args.threshold,
        "prompt_id": outputs["agent1"].prompt_id,
        "customer_name": outputs["agent1"].customer_name,
        "input_files": {agent: str(path.resolve()) for agent, path in paths.items()},
        "evaluations": results,
        "passed": all(result["passed"] for result in results),
    }
    report_path = args.report.resolve()
    if report_path in {path.resolve() for path in paths.values()}:
        parser.error("--report must not overwrite one of the agent output files.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_json(report) + "\n", encoding="utf-8")

    print(f"Claude judge: {args.judge_model}")
    print(f"Prompt: {outputs['agent1'].prompt_id} · {outputs['agent1'].customer_name}")
    for result in results:
        score = result["score"]
        score_text = f"{score:.2f}" if score is not None else "n/a"
        status = "PASS" if result["passed"] else "REVIEW"
        print(f"{status:6} {score_text:>4}  {result['name']}")
        if result["reason"]:
            print(f"       {result['reason']}")
    print(f"Report: {report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
