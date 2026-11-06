"""ACA Agent 1: Prompt & Answer Intelligence for AImpact.

Understands one user prompt and the AI answers already stored for it:
intent, topics, concepts, mentioned companies, and whether the customer
appears in those answers.

Reads evidence through the temporary AImpact PostgreSQL reader in
``src.tools.aimpact``. It does not fetch citation pages (Agent 2), verify
customer capabilities (Agent 3), or write content. Citation URLs are not
sent to the model and are not part of the output.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from src.models.prompt_intelligence import PromptAnalysis
from src.observability.token_usage import record_claude_response
from src.utils.output_storage import DEFAULT_OUTPUT_DIR
from src.utils.structured_output import extract_tool_input

load_dotenv()


def _value(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


_PROMPT_ANALYSIS_TOOL = {
    "name": "submit_prompt_analysis",
    "description": "Submit structured prompt and answer analysis.",
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt_understanding": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "topic": {"type": "string"},
                    "audience": {"type": ["string", "null"]},
                    "user_goal": {"type": "string"},
                    "key_questions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["intent", "topic", "audience", "user_goal", "key_questions"],
                "additionalProperties": False,
            },
            "answer_understanding": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "key_topics": {"type": "array", "items": {"type": "string"}},
                    "key_concepts": {"type": "array", "items": {"type": "string"}},
                    "comparison_criteria": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary", "key_topics", "key_concepts", "comparison_criteria"],
                "additionalProperties": False,
            },
            "companies_mentioned": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "context": {"type": "string"}},
                    "required": ["name", "context"],
                    "additionalProperties": False,
                },
            },
            "cross_model_patterns": {
                "type": "object",
                "properties": {
                    "common_topics": {"type": "array", "items": {"type": "string"}},
                    "differences": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["common_topics", "differences"],
                "additionalProperties": False,
            },
            "content_opportunities": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "prompt_understanding",
            "answer_understanding",
            "companies_mentioned",
            "cross_model_patterns",
            "content_opportunities",
        ],
        "additionalProperties": False,
    },
}


def _get_answers(prompt_evidence: Any) -> tuple[str, list[dict[str, str]]]:
    prompt = str(_value(prompt_evidence, "prompt_text", "") or "").strip()
    per_model = _value(prompt_evidence, "per_model", {}) or {}
    answers = []
    for model_name, evidence in per_model.items():
        answer = str(_value(evidence, "response_text", "") or "").strip()
        if answer:
            answers.append({"model": str(model_name), "answer": answer})
    if not prompt:
        raise ValueError("Selected AImpact prompt has no prompt_text.")
    if not answers:
        raise ValueError("No non-empty model answers were found for this prompt.")
    return prompt, answers


def _mentioned(name: str, text: str) -> bool:
    needle = name.strip()
    if len(needle) < 2:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", text, flags=re.I) is not None


def _mention_snippet(answer: str, customer_name: str) -> str | None:
    match = re.search(rf"(?<!\w){re.escape(customer_name.strip())}(?!\w)", answer, flags=re.I)
    if not match:
        return None
    start = max(0, match.start() - 90)
    end = min(len(answer), match.end() + 90)
    snippet = re.sub(r"\s+", " ", answer[start:end]).strip()
    return snippet or None


def _visibility(customer_name: str, answers: list[dict[str, str]]) -> dict[str, Any]:
    mentioning: list[str] = []
    contexts: list[str] = []
    for answer in answers:
        if not _mentioned(customer_name, answer["answer"]):
            continue
        mentioning.append(answer["model"])
        snippet = _mention_snippet(answer["answer"], customer_name)
        if snippet and snippet not in contexts:
            contexts.append(snippet)
    total = len(answers)
    if total == 0:
        status = "unknown"
    elif not mentioning:
        status = "not_mentioned"
    elif len(mentioning) == total:
        status = "mentioned"
    else:
        status = "mixed"
    return {
        "status": status,
        "models_mentioning_customer": mentioning,
        "context": "; ".join(contexts) if contexts else None,
    }


def _companies(raw: Any, answers: list[dict[str, str]]) -> list[dict[str, str]]:
    blob = "\n".join(item["answer"] for item in answers)
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw or []:
        if isinstance(item, str):
            name, context = item.strip(), "Mentioned in the AI answers"
        elif isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            context = str(item.get("context") or "").strip() or "Mentioned in the AI answers"
        else:
            continue
        key = name.lower()
        if not name or key in seen or not _mentioned(name, blob):
            continue
        seen.add(key)
        cleaned.append({"name": name, "context": context})
    return cleaned


def _string_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    values: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            values.append(text)
    return values


class PromptIntelligenceAgent:
    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self.api_key = api_key
        self.model = (
            model
            or os.getenv("ANTHROPIC_MODEL_NAME")
            or "claude-haiku-4-5-20251001"
        )

    def analyze(
        self,
        prompt_evidence: Any,
        customer_name: str,
        campaign_id: int | str | None = None,
    ) -> PromptAnalysis:
        del campaign_id  # kept so callers can pass the campaign they loaded
        prompt_id = str(_value(prompt_evidence, "prompt_id", "")).strip()
        if not prompt_id:
            raise ValueError("Selected AImpact prompt has no prompt_id.")
        customer_name = customer_name.strip()
        if not customer_name:
            raise ValueError("customer_name cannot be empty.")
        prompt, answers = _get_answers(prompt_evidence)

        system_prompt = """You are Agent 1: Prompt & Answer Intelligence for AImpact.
Analyze only the provided prompt and AI answers. Submit your analysis using the
required submit_prompt_analysis tool.
Rules:
- prompt_understanding describes what the user is asking, not what the AI answered.
- answer_understanding summarizes what the AI answers actually say: topics, concepts, and how they compare options.
- companies_mentioned lists only organizations, brands, or products named in the answers. context is a short phrase about how the answer uses that name (for example "Recommended running shoe brand"). Do not include companies that are not in the answers.
- cross_model_patterns.common_topics are ideas shared across answers. differences are real disagreements. Use an empty list when answers do not disagree.
- content_opportunities are short hypotheses about what a later agent might investigate for the named customer. They are not confirmed gaps and must not claim the customer's real products, rankings, or capabilities.
- Do not browse, fetch citation URLs, output citations, or verify any company's real-world capabilities.
- Do not add fields. Do not wrap the JSON in Markdown."""

        payload = {
            "customer_name": customer_name,
            "prompt_id": prompt_id,
            "prompt": prompt,
            "answers": answers,
        }
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 4000,
                "system": system_prompt,
                "tools": [_PROMPT_ANALYSIS_TOOL],
                "tool_choice": {"type": "tool", "name": _PROMPT_ANALYSIS_TOOL["name"]},
                "messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            },
            timeout=90.0,
        )
        record_claude_response(response, self.model)
        response.raise_for_status()
        response_data = response.json()
        if response_data.get("stop_reason") != "tool_use":
            raise RuntimeError(
                "Claude did not return Agent 1's required structured response "
                f"(stop reason: {response_data.get('stop_reason') or 'unknown'})."
            )
        data = extract_tool_input(response_data, _PROMPT_ANALYSIS_TOOL["name"])

        understanding = data.get("prompt_understanding") or {}
        answers_view = data.get("answer_understanding") or {}
        patterns = data.get("cross_model_patterns") or {}
        public = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "customer_name": customer_name,
            "prompt_understanding": {
                "intent": str(understanding.get("intent") or "").strip(),
                "topic": str(understanding.get("topic") or "").strip(),
                "audience": (str(understanding.get("audience")).strip() if understanding.get("audience") else None),
                "user_goal": str(understanding.get("user_goal") or "").strip(),
                "key_questions": _string_list(understanding.get("key_questions")),
            },
            "answer_understanding": {
                "summary": str(answers_view.get("summary") or "").strip(),
                "key_topics": _string_list(answers_view.get("key_topics")),
                "key_concepts": _string_list(answers_view.get("key_concepts")),
                "comparison_criteria": _string_list(answers_view.get("comparison_criteria")),
            },
            "companies_mentioned": _companies(data.get("companies_mentioned"), answers),
            "customer_visibility": _visibility(customer_name, answers),
            "cross_model_patterns": {
                "common_topics": _string_list(patterns.get("common_topics")),
                "differences": _string_list(patterns.get("differences")),
            },
            "content_opportunities": _string_list(data.get("content_opportunities")),
        }
        return PromptAnalysis.model_validate(public)


def run_agent1(
    campaign_id: int,
    prompt_id: int | str,
    customer_name: str | None = None,
    score_threshold: float = 70,
) -> PromptAnalysis:
    """Run Agent 1 using the temporary AImpact PostgreSQL reader."""
    from src.tools.aimpact import get_campaign, get_project, get_prompt_evidence

    records = get_prompt_evidence(campaign_id=campaign_id, score_threshold=score_threshold)
    selected = next((p for p in records if str(_value(p, "prompt_id", "")) == str(prompt_id)), None)
    if selected is None:
        raise ValueError(
            f"Prompt {prompt_id} not found in campaign {campaign_id} "
            f"with score_threshold={score_threshold}. Check the adapter's score filter."
        )

    if not customer_name:
        campaign = get_campaign(campaign_id)
        project_id = campaign.get("project_id")
        if project_id is None:
            raise ValueError("get_campaign() did not return project_id. Pass --customer-name explicitly.")
        project = get_project(project_id)
        customer_name = project.get("company_name") or project.get("name")
        if not customer_name:
            raise ValueError("Could not get customer name from get_project(); pass --customer-name explicitly.")

    return PromptIntelligenceAgent().analyze(selected, customer_name, campaign_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AImpact Agent 1: Prompt & Answer Intelligence")
    parser.add_argument("--campaign-id", type=int, required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--customer-name")
    parser.add_argument("--score-threshold", type=float, default=70)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR / "agent1_output.json", type=os.fspath)
    args = parser.parse_args()

    result = run_agent1(args.campaign_id, args.prompt_id, args.customer_name, args.score_threshold)
    payload = result.model_dump()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("=" * 68)
    print("AIMPACT AGENT 1 — PROMPT & ANSWER INTELLIGENCE")
    print("=" * 68)
    print(f"Prompt ID: {result.prompt_id}")
    print(f"Prompt: {result.prompt}")
    print(f"Customer: {result.customer_name}")
    print(f"Topic: {result.prompt_understanding.topic}")
    print(f"Companies mentioned: {len(result.companies_mentioned)}")
    print(f"Customer visibility: {result.customer_visibility.status}")
    print(f"Output saved to: {output_path.resolve()}")
