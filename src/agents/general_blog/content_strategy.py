"""Interpret a General Blog request and produce a research plan."""

from __future__ import annotations

import os
import re

import anthropic

from src.models.general_blog import GeneralBlogPlan, GeneralBlogRequest, normalize_http_url
from src.utils.general_blog import call_structured_agent

_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE)
_URL_TRAILING_PUNCTUATION = ".,!?;:)]}"


def _find_prompt_urls(prompt: str) -> list[str]:
    found = []
    for match in _URL_PATTERN.findall(prompt):
        candidate = match.rstrip(_URL_TRAILING_PUNCTUATION)
        if candidate and candidate not in found:
            found.append(candidate)
    return found


class GeneralBlogIntakeAgent:
    """Create the research plan and collect explicit reference links."""

    def __init__(
        self,
        model: str | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if client is None and not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")
        self.client = client or anthropic.Anthropic(api_key=api_key)
        self.model = model or os.getenv("ANTHROPIC_MODEL_NAME") or "claude-haiku-4-5-20251001"

    def prepare(
        self,
        request: GeneralBlogRequest,
    ) -> tuple[GeneralBlogPlan, list[str]]:
        reference_urls = []
        seen = {request.company_website}
        for value in [*request.reference_urls, *_find_prompt_urls(request.prompt)]:
            url = normalize_http_url(value)
            if url not in seen:
                seen.add(url)
                reference_urls.append(url)

        payload = {
            "company_name": request.company_name,
            "company_website": request.company_website,
            "user_prompt": request.prompt,
            "target_audience": request.target_audience,
            "tone": request.tone,
            "reference_urls": reference_urls,
        }
        data = call_structured_agent(
            self.client,
            self.model,
            """You plan a standalone company blog using only the supplied request.
Create a focused article objective, reader search intent, title, keyword ideas,
and up to six web research queries. Use the user's topic as the anchor. Do not
make claims about company capabilities or invent citations. Incorporate the
requested audience and tone when provided. Search queries should target the
company's website for company-specific information and the topic for needed
context. Return only the required structured tool response.""",
            payload,
            GeneralBlogPlan,
            "submit_general_blog_plan",
            max_tokens=2200,
        )
        plan = GeneralBlogPlan.model_validate(data)
        if request.target_audience or request.tone:
            plan = plan.model_copy(
                update={
                    "target_audience": request.target_audience or plan.target_audience,
                    "tone": request.tone or plan.tone,
                }
            )
        return plan, reference_urls
