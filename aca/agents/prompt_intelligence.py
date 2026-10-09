"""
Agent 1: Prompt + Answer Intelligence

Purpose:
    Understand one AImpact prompt and the AI answers generated for it.

This agent does NOT:
    - fetch web pages
    - research competitors
    - verify company claims
    - write content
    - create a content brief

It only answers:

    1. What is the user asking?
    2. What topic is involved?
    3. What concepts matter?
    4. What are AI models currently saying?
    5. Which companies are mentioned?
    6. What citation patterns appear?
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any
from urllib.parse import urlsplit

from anthropic import Anthropic
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# OUTPUT MODELS
# ---------------------------------------------------------------------------


class CitationPattern(BaseModel):
    domain: str
    citation_count: int
    why_relevant: str = ""


class PromptAnalysis(BaseModel):
    """
    Structured output produced by Agent 1.
    """

    prompt_id: str
    prompt: str

    intent: str
    topic: str

    entities: list[str] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)

    answer_summary: str

    answer_patterns: list[str] = Field(default_factory=list)

    companies_mentioned: list[str] = Field(default_factory=list)

    citation_patterns: list[CitationPattern] = Field(default_factory=list)


# Google AI citations are redirect links. The real domain is stored in title.
_GOOGLE_REDIRECT_HOST = "vertexaisearch.cloud.google.com"
_DOMAIN_RE = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)+$")


def _citation_domain(source: Any) -> str | None:
    """Domain for a citation. AImpact often leaves domain empty and puts it in the URL or title."""
    host = ""
    if getattr(source, "url", None):
        host = (urlsplit(source.url).hostname or "").lower().removeprefix("www.")

    if host and not host.endswith(_GOOGLE_REDIRECT_HOST):
        return host

    for candidate in (getattr(source, "domain", None), getattr(source, "title", None)):
        if candidate and _DOMAIN_RE.match(str(candidate).strip().lower()):
            return str(candidate).strip().lower().removeprefix("www.")

    return host or None


# ---------------------------------------------------------------------------
# AGENT
# ---------------------------------------------------------------------------


class PromptAnswerAgent:
    """
    Agent 1 — Prompt + Answer Intelligence.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not configured."
            )

        self.client = Anthropic(api_key=self.api_key)
        self.model = (
            model
            or os.getenv("ANTHROPIC_MODEL_HAIKU")
            or os.getenv("ANTHROPIC_MODEL_NAME")
            or "claude-haiku-4-5-20251001"
        )

    # -----------------------------------------------------------------------
    # PUBLIC METHOD
    # -----------------------------------------------------------------------

    def analyze(self, prompt_data: Any) -> PromptAnalysis:
        """
        Analyze one AImpact PromptEvidence object.

        Expected input:

            PromptEvidence(
                prompt_id="1409",
                prompt_text="Immunology biomarker assay provider",
                per_model={
                    "ChatGPT": ModelEvidence(...),
                    "Claude": ModelEvidence(...),
                    ...
                }
            )
        """

        prompt_id = str(prompt_data.prompt_id)
        prompt_text = prompt_data.prompt_text

        # Convert Pydantic object to normal Python data.
        payload = self._build_input_payload(prompt_data)

        # First extract deterministic citation information.
        citation_patterns = self._build_citation_patterns(prompt_data)

        # Ask Claude to understand the prompt + answers.
        llm_result = self._ask_claude(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            payload=payload,
        )

        # Combine Claude's interpretation with deterministic citation data.
        result = PromptAnalysis(
            prompt_id=prompt_id,
            prompt=prompt_text,

            intent=llm_result.get(
                "intent",
                "",
            ),

            topic=llm_result.get(
                "topic",
                "",
            ),

            entities=self._clean_string_list(
                llm_result.get("entities", [])
            ),

            key_concepts=self._clean_string_list(
                llm_result.get("key_concepts", [])
            ),

            answer_summary=llm_result.get(
                "answer_summary",
                "",
            ),

            answer_patterns=self._clean_string_list(
                llm_result.get("answer_patterns", [])
            ),

            companies_mentioned=self._clean_string_list(
                llm_result.get("companies_mentioned", [])
            ),

            citation_patterns=citation_patterns,
        )

        return result

    # -----------------------------------------------------------------------
    # BUILD INPUT
    # -----------------------------------------------------------------------

    def _build_input_payload(
        self,
        prompt_data: Any,
    ) -> dict[str, Any]:
        """
        Convert PromptEvidence into a compact JSON payload for Claude.
        """

        models: dict[str, Any] = {}

        for model_name, model_evidence in prompt_data.per_model.items():

            sources = []

            for source in model_evidence.sources:

                sources.append(
                    {
                        "url": source.url,
                        "domain": source.domain,
                        "title": source.title,
                        "excerpt": source.excerpt,
                        "competitor": source.competitor,
                    }
                )

            models[model_name] = {
                "response": model_evidence.response_text or "",
                "sources": sources,
            }

        return {
            "prompt_id": str(prompt_data.prompt_id),
            "prompt": prompt_data.prompt_text,
            "models": models,
        }

    # -----------------------------------------------------------------------
    # CITATION ANALYSIS
    # -----------------------------------------------------------------------

    def _build_citation_patterns(
        self,
        prompt_data: Any,
    ) -> list[CitationPattern]:
        """
        Count how frequently each domain is cited.

        This is deterministic and therefore does not depend on the LLM.
        """

        domain_counts: Counter[str] = Counter()

        for model_evidence in prompt_data.per_model.values():

            for source in model_evidence.sources:

                domain = _citation_domain(source)

                if domain:
                    domain_counts[domain] += 1

        patterns: list[CitationPattern] = []

        for domain, count in domain_counts.most_common(20):

            patterns.append(
                CitationPattern(
                    domain=domain,
                    citation_count=count,
                )
            )

        return patterns

    # -----------------------------------------------------------------------
    # CLAUDE
    # -----------------------------------------------------------------------

    def _ask_claude(
        self,
        prompt_id: str,
        prompt_text: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:

        system_prompt = """
You are Agent 1 in an AI visibility/content intelligence system.

Your job is ONLY:

1. Understand what the user prompt is asking.
2. Identify the main intent.
3. Identify the main topic.
4. Identify important entities.
5. Identify important concepts that appear in the AI answers.
6. Summarize what the AI models currently say.
7. Identify recurring answer patterns.
8. Identify companies mentioned by the AI models.

Do NOT:
- research the web
- fetch URLs
- verify company claims
- recommend content
- create a content strategy
- write an article
- invent company capabilities
- assume that a cited source is authoritative

Use ONLY the supplied prompt and AI responses.

Return ONLY valid JSON with exactly these fields:

{
  "intent": "...",
  "topic": "...",
  "entities": [],
  "key_concepts": [],
  "answer_summary": "...",
  "answer_patterns": [],
  "companies_mentioned": []
}

Be concise and evidence-based.
"""

        user_message = f"""
Analyze this AImpact prompt and its AI responses.

Prompt ID:
{prompt_id}

Prompt:
{prompt_text}

AImpact data:

{json.dumps(payload, indent=2, ensure_ascii=False)}
"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
        )

        text = self._extract_text(response)

        return self._parse_json(text)

    # -----------------------------------------------------------------------
    # RESPONSE HELPERS
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        """
        Extract text from Anthropic response.
        """

        parts: list[str] = []

        for block in response.content:

            if getattr(block, "type", None) == "text":

                parts.append(
                    block.text
                )

        return "\n".join(parts).strip()

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """
        Parse Claude's JSON response.

        Handles accidental markdown fences such as:

        ```json
        {...}
        ```
        """

        cleaned = text.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "Agent 1 received invalid JSON from Claude.\n\n"
                f"Claude response:\n{cleaned}"
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                "Agent 1 expected a JSON object from Claude."
            )

        return data

    @staticmethod
    def _clean_string_list(value: Any) -> list[str]:
        """
        Ensure list fields contain only non-empty strings.
        """

        if not isinstance(value, list):
            return []

        result: list[str] = []

        for item in value:

            if isinstance(item, str):

                cleaned = item.strip()

                if cleaned:
                    result.append(cleaned)

        return result


# ---------------------------------------------------------------------------
# CONVENIENCE FUNCTION
# ---------------------------------------------------------------------------


def analyze_prompt(prompt_data: Any) -> PromptAnalysis:
    """
    Simple function for the LangGraph node.

    Usage:

        result = analyze_prompt(prompt_evidence)
    """

    agent = PromptAnswerAgent()

    return agent.analyze(prompt_data)