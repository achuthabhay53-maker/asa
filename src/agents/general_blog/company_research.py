"""Research and validate evidence from the supplied company website."""

from __future__ import annotations

import logging
import os
from urllib.parse import urlsplit

import anthropic

from src.models.general_blog import GeneralBlogEvidence, GeneralBlogPlan, GeneralBlogRequest
from src.tools.reference_fetcher import MAX_SOURCE_CHARS, RetrievedSource
from src.tools.web_research import WebResearch, get_web_research
from src.utils.general_blog_research import extract_source_evidence

MAX_COMPANY_PAGES = 8
logger = logging.getLogger(__name__)


class CompanyWebsiteResearchAgent:
    """Find relevant pages on the supplied company's website."""

    def __init__(
        self,
        *,
        model: str | None = None,
        client: anthropic.Anthropic | None = None,
        research: WebResearch | None = None,
    ) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if client is None and not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")
        self.client = client or anthropic.Anthropic(api_key=api_key)
        self.model = model or os.getenv("ANTHROPIC_MODEL_NAME") or "claude-haiku-4-5-20251001"
        self.research = research or get_web_research()

    def research_company(
        self,
        request: GeneralBlogRequest,
        plan: GeneralBlogPlan,
        evidence_offset: int = 0,
    ) -> tuple[list[GeneralBlogEvidence], list[str]]:
        host = urlsplit(request.company_website).hostname
        if not host:
            raise ValueError("The company website must include a hostname.")
        pages: dict[str, RetrievedSource] = {}
        warnings: list[str] = []
        for query in plan.research_queries:
            result = self.research.search(
                f"{request.company_name} {query}",
                max_results=3,
                include_domains=[host],
            )
            if result.failed:
                logger.warning(
                    "Company website search failed (company_host=%s, query=%s)",
                    host,
                    query,
                )
                warnings.append(f"Company-site search failed for: {query}")
                continue
            for item in result.results:
                item_host = (urlsplit(item.url).hostname or "").lower()
                allowed_host = host.lower()
                if item_host != allowed_host and not item_host.endswith(f".{allowed_host}"):
                    continue
                if item.url in pages:
                    continue
                page = self.research.extract(item.url, max_chars=MAX_SOURCE_CHARS)
                if page is None or not page.text.strip():
                    warnings.append(f"Could not extract company website page: {item.url}")
                    continue
                page_host = (urlsplit(page.url).hostname or "").lower()
                if page_host != allowed_host and not page_host.endswith(f".{allowed_host}"):
                    warnings.append(
                        f"Skipped company page after it redirected outside the supplied website: {item.url}"
                    )
                    continue
                pages[page.url] = RetrievedSource(
                    url=page.url,
                    title=page.title or item.title,
                    text=page.text[:MAX_SOURCE_CHARS],
                    publication_date=item.published,
                )
                if len(pages) >= MAX_COMPANY_PAGES:
                    break
            if len(pages) >= MAX_COMPANY_PAGES:
                break

        if not pages:
            warnings.append("No relevant company website pages were retrieved.")
        evidence: list[GeneralBlogEvidence] = []
        for page in pages.values():
            evidence.extend(
                extract_source_evidence(
                    self.client,
                    self.model,
                    page,
                    "company_website",
                    evidence_offset + len(evidence),
                    request.prompt,
                )
            )
        if pages and not evidence:
            warnings.append("Company pages were retrieved, but no exact-quote evidence was validated.")
        return evidence, warnings
