"""
Agent 2 - Source Intelligence

Takes:
1. Agent 1 PromptAnalysis
2. Raw AImpact PromptEvidence

Then:
- ranks real AImpact-cited sources
- fetches the actual URLs
- analyzes their content with Claude
- extracts topics, claims, competitors, structure and style

Important:
This agent does NOT invent citations.
A source is considered an "AImpact cited source" only when it
comes from the raw citation records supplied by AImpact.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any
from urllib.parse import urlparse

import anthropic
import httpx
from pydantic import BaseModel, Field

from aca.orchestrator.state import CitedSource, PromptEvidence


# ============================================================
# Configuration
# ============================================================

MAX_SOURCES = 10
FETCH_TIMEOUT = 20.0
MAX_PAGE_CHARS = 30000


# ============================================================
# Output models
# ============================================================


class SourceAnalysisItem(BaseModel):
    """
    Intelligence extracted from one real AImpact-cited source.
    """

    source_id: str

    url: str
    domain: str

    title: str | None = None

    citation_count: int = 0

    cited_by_models: list[str] = Field(default_factory=list)

    source_type: str = "unknown"

    summary: str = ""

    topics: list[str] = Field(default_factory=list)

    important_claims: list[str] = Field(default_factory=list)

    competitors: list[str] = Field(default_factory=list)

    competitor_context: list[str] = Field(default_factory=list)

    content_structure: list[str] = Field(default_factory=list)

    style_profile: list[str] = Field(default_factory=list)

    audience: str = ""

    evidence_density: str = ""

    why_relevant_to_prompt: str = ""

    fetch_status: str = "not_attempted"

    fetch_error: str | None = None


class SourceAnalysis(BaseModel):
    """
    Complete Agent 2 output for one prompt.
    """

    prompt_id: str

    prompt: str

    total_unique_cited_sources: int = 0

    total_citation_occurrences: int = 0

    sources_analyzed: int = 0

    top_sources: list[SourceAnalysisItem] = Field(default_factory=list)


# ============================================================
# Agent
# ============================================================


class SourceIntelligenceAgent:
    """
    Agent 2.

    Input:
        PromptAnalysis from Agent 1
        PromptEvidence from AImpact

    Output:
        SourceAnalysis
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5",
    ) -> None:

        self.client = anthropic.Anthropic(
            api_key=api_key,
        )

        self.model = model

    # --------------------------------------------------------
    # Public entry point
    # --------------------------------------------------------

    def run(
        self,
        prompt_analysis: dict[str, Any],
        prompt_evidence: PromptEvidence,
    ) -> SourceAnalysis:

        print("\n" + "=" * 70)
        print("AGENT 2 - SOURCE INTELLIGENCE")
        print("=" * 70)

        print(f"Prompt ID: {prompt_evidence.prompt_id}")
        print(f"Prompt: {prompt_evidence.prompt_text}")

        # ----------------------------------------------------
        # Step 1: Collect raw AImpact citations
        # ----------------------------------------------------

        sources = self._collect_sources(prompt_evidence)

        print(f"\nRaw unique AImpact sources: {len(sources)}")

        total_citations = sum(
            source["citation_count"]
            for source in sources
        )

        print(f"Total citation occurrences: {total_citations}")

        # ----------------------------------------------------
        # Step 2: Rank sources
        # ----------------------------------------------------

        ranked_sources = self._rank_sources(
            sources=sources,
            prompt_analysis=prompt_analysis,
        )

        print(
            f"Selected top sources for analysis: "
            f"{len(ranked_sources)}"
        )

        # ----------------------------------------------------
        # Step 3: Analyze each source
        # ----------------------------------------------------

        analyzed_sources: list[SourceAnalysisItem] = []

        for index, source in enumerate(ranked_sources, start=1):

            print(
                f"\n[{index}/{len(ranked_sources)}] "
                f"{source['domain']}"
            )

            print(f"URL: {source['url']}")
            print(
                f"Citations: {source['citation_count']}"
            )

            analysis = self._analyze_source(
                source=source,
                prompt_analysis=prompt_analysis,
                prompt_text=prompt_evidence.prompt_text,
            )

            analyzed_sources.append(analysis)

        # ----------------------------------------------------
        # Step 4: Build final output
        # ----------------------------------------------------

        result = SourceAnalysis(
            prompt_id=prompt_evidence.prompt_id,
            prompt=prompt_evidence.prompt_text,
            total_unique_cited_sources=len(sources),
            total_citation_occurrences=total_citations,
            sources_analyzed=len(analyzed_sources),
            top_sources=analyzed_sources,
        )

        print("\n" + "=" * 70)
        print("AGENT 2 COMPLETE")
        print("=" * 70)

        print(
            f"Unique AImpact sources: "
            f"{result.total_unique_cited_sources}"
        )

        print(
            f"Total citation occurrences: "
            f"{result.total_citation_occurrences}"
        )

        print(
            f"Sources analyzed: "
            f"{result.sources_analyzed}"
        )

        return result

    # --------------------------------------------------------
    # Collect raw citations
    # --------------------------------------------------------

    def _collect_sources(
        self,
        prompt_evidence: PromptEvidence,
    ) -> list[dict[str, Any]]:

        source_map: dict[str, dict[str, Any]] = {}

        for model_name, model_evidence in prompt_evidence.per_model.items():

            for citation in model_evidence.sources:

                url = citation.url

                if not url:
                    continue

                url = url.strip()

                if not url:
                    continue

                # Normalize URL for deduplication.
                normalized_url = self._normalize_url(url)

                if not normalized_url:
                    continue

                if normalized_url not in source_map:

                    domain = (
                        citation.domain
                        or self._extract_domain(normalized_url)
                    )

                    source_map[normalized_url] = {
                        "url": normalized_url,
                        "domain": domain,
                        "title": citation.title,
                        "excerpt": citation.excerpt,
                        "competitor": citation.competitor,
                        "citation_count": 0,
                        "cited_by_models": set(),
                        "citations": [],
                    }

                source = source_map[normalized_url]

                source["citation_count"] += 1

                source["cited_by_models"].add(
                    model_name
                )

                source["citations"].append(
                    citation
                )

        # Convert sets to lists.
        results = []

        for source in source_map.values():

            source["cited_by_models"] = sorted(
                source["cited_by_models"]
            )

            results.append(source)

        return results

    # --------------------------------------------------------
    # Rank sources
    # --------------------------------------------------------

    def _rank_sources(
        self,
        sources: list[dict[str, Any]],
        prompt_analysis: dict[str, Any],
    ) -> list[dict[str, Any]]:

        key_concepts = {
            self._clean_text(concept)
            for concept in prompt_analysis.get(
                "key_concepts",
                [],
            )
        }

        companies = {
            self._clean_text(company)
            for company in prompt_analysis.get(
                "companies_mentioned",
                [],
            )
        }

        scored_sources = []

        for source in sources:

            score = 0.0

            # ------------------------------------------------
            # Citation frequency
            # ------------------------------------------------

            score += source["citation_count"] * 10

            # ------------------------------------------------
            # Multiple AI models cited it
            # ------------------------------------------------

            score += len(
                source["cited_by_models"]
            ) * 5

            # ------------------------------------------------
            # Title/excerpt relevance
            # ------------------------------------------------

            searchable_text = " ".join(
                [
                    source.get("title") or "",
                    source.get("excerpt") or "",
                    source.get("domain") or "",
                ]
            ).lower()

            for concept in key_concepts:

                if concept and concept.lower() in searchable_text:
                    score += 3

            # ------------------------------------------------
            # Mentioned company relevance
            # ------------------------------------------------

            competitor = (
                source.get("competitor")
                or ""
            )

            for company in companies:

                if company and company.lower() in (
                    competitor.lower()
                ):
                    score += 2

            source["_ranking_score"] = score

            scored_sources.append(source)

        scored_sources.sort(
            key=lambda item: item["_ranking_score"],
            reverse=True,
        )

        return scored_sources[:MAX_SOURCES]

    # --------------------------------------------------------
    # Fetch webpage
    # --------------------------------------------------------

    def _fetch_page(
        self,
        url: str,
    ) -> tuple[str | None, str | None]:

        try:

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "Chrome/154.0 Safari/537.36"
                )
            }

            with httpx.Client(
                timeout=FETCH_TIMEOUT,
                follow_redirects=True,
                headers=headers,
            ) as client:

                response = client.get(url)

                response.raise_for_status()

                content_type = response.headers.get(
                    "content-type",
                    "",
                ).lower()

                if (
                    "text/html" not in content_type
                    and "application/xhtml" not in content_type
                ):
                    return (
                        None,
                        f"Unsupported content type: {content_type}",
                    )

                html = response.text

                text = self._html_to_text(html)

                if not text.strip():
                    return (
                        None,
                        "Page contained no readable text",
                    )

                return (
                    text[:MAX_PAGE_CHARS],
                    None,
                )

        except Exception as exc:

            return (
                None,
                str(exc),
            )

    # --------------------------------------------------------
    # Convert HTML into readable text
    # --------------------------------------------------------

    def _html_to_text(
        self,
        html: str,
    ) -> str:

        # Remove scripts/styles.
        html = re.sub(
            r"<script\b[^>]*>.*?</script>",
            " ",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        html = re.sub(
            r"<style\b[^>]*>.*?</style>",
            " ",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        html = re.sub(
            r"<noscript\b[^>]*>.*?</noscript>",
            " ",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Preserve headings as separate text.
        html = re.sub(
            r"<h[1-6][^>]*>",
            "\n",
            html,
            flags=re.IGNORECASE,
        )

        html = re.sub(
            r"</h[1-6]>",
            "\n",
            html,
            flags=re.IGNORECASE,
        )

        # Paragraphs / list items.
        html = re.sub(
            r"<(?:p|li|div|section|article|br)[^>]*>",
            "\n",
            html,
            flags=re.IGNORECASE,
        )

        # Remove remaining tags.
        html = re.sub(
            r"<[^>]+>",
            " ",
            html,
        )

        # Decode common entities.
        replacements = {
            "&nbsp;": " ",
            "&amp;": "&",
            "&quot;": '"',
            "&#39;": "'",
            "&lt;": "<",
            "&gt;": ">",
        }

        for old, new in replacements.items():
            html = html.replace(old, new)

        # Normalize whitespace.
        html = re.sub(
            r"[ \t]+",
            " ",
            html,
        )

        html = re.sub(
            r"\n\s*\n+",
            "\n",
            html,
        )

        return html.strip()

    # --------------------------------------------------------
    # Claude source analysis
    # --------------------------------------------------------

    def _analyze_source(
        self,
        source: dict[str, Any],
        prompt_analysis: dict[str, Any],
        prompt_text: str,
    ) -> SourceAnalysisItem:

        url = source["url"]
        domain = source["domain"]

        page_text, fetch_error = self._fetch_page(url)

        if page_text is None:

            print(
                f"  Fetch failed: {fetch_error}"
            )

            return SourceAnalysisItem(
                source_id=self._make_source_id(url),
                url=url,
                domain=domain,
                title=source.get("title"),
                citation_count=source["citation_count"],
                cited_by_models=source[
                    "cited_by_models"
                ],
                fetch_status="failed",
                fetch_error=fetch_error,
            )

        print(
            f"  Fetched {len(page_text):,} characters"
        )

        prompt_analysis_json = json.dumps(
            prompt_analysis,
            ensure_ascii=False,
            indent=2,
        )

        analysis_prompt = f"""
You are Agent 2 of an agentic content intelligence system.

Your job is to analyze a REAL WEBPAGE that was cited by an AI
response recorded by AImpact.

IMPORTANT:
- The URL below came from AImpact's citation data.
- Do not invent facts about the source.
- Only extract information supported by the supplied webpage text.
- Do not copy long passages.
- Summarize in your own words.
- "style_profile" means structural/editorial characteristics,
  NOT instructions to copy the source.
- Competitors must only be listed when the webpage actually
  supports that they are competitors or comparable providers.
- If something cannot be determined, return an empty list or
  a conservative statement.

USER PROMPT:
{prompt_text}

AGENT 1 PROMPT ANALYSIS:
{prompt_analysis_json}

SOURCE URL:
{url}

SOURCE DOMAIN:
{domain}

SOURCE TITLE:
{source.get("title") or ""}

AIMPACT CITATION COUNT:
{source["citation_count"]}

AIMPACT MODELS THAT CITED THIS SOURCE:
{json.dumps(source["cited_by_models"])}

WEBPAGE TEXT:
{page_text}

Return ONLY valid JSON with exactly this structure:

{{
  "title": "source title",
  "source_type": "commercial|academic|educational|directory|news|government|unknown",
  "summary": "short factual summary",
  "topics": [
    "topic 1",
    "topic 2"
  ],
  "important_claims": [
    "claim 1",
    "claim 2"
  ],
  "competitors": [
    "company or provider"
  ],
  "competitor_context": [
    "Company X is discussed in relation to ..."
  ],
  "content_structure": [
    "service overview",
    "technical capabilities",
    "applications",
    "FAQ"
  ],
  "style_profile": [
    "technical",
    "authoritative",
    "evidence-oriented"
  ],
  "audience": "description of intended audience",
  "evidence_density": "low|medium|high",
  "why_relevant_to_prompt": "why this source helps answer the user prompt"
}}
"""

        try:

            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": analysis_prompt,
                    }
                ],
            )

            raw_text = self._extract_response_text(
                response
            )

            parsed = self._parse_json(
                raw_text
            )

            return SourceAnalysisItem(
                source_id=self._make_source_id(url),
                url=url,
                domain=domain,
                title=(
                    parsed.get("title")
                    or source.get("title")
                ),
                citation_count=source[
                    "citation_count"
                ],
                cited_by_models=source[
                    "cited_by_models"
                ],
                source_type=parsed.get(
                    "source_type",
                    "unknown",
                ),
                summary=parsed.get(
                    "summary",
                    "",
                ),
                topics=self._ensure_list(
                    parsed.get("topics")
                ),
                important_claims=self._ensure_list(
                    parsed.get("important_claims")
                ),
                competitors=self._ensure_list(
                    parsed.get("competitors")
                ),
                competitor_context=self._ensure_list(
                    parsed.get("competitor_context")
                ),
                content_structure=self._ensure_list(
                    parsed.get("content_structure")
                ),
                style_profile=self._ensure_list(
                    parsed.get("style_profile")
                ),
                audience=parsed.get(
                    "audience",
                    "",
                ),
                evidence_density=parsed.get(
                    "evidence_density",
                    "",
                ),
                why_relevant_to_prompt=parsed.get(
                    "why_relevant_to_prompt",
                    "",
                ),
                fetch_status="success",
            )

        except Exception as exc:

            print(
                f"  Claude analysis failed: {exc}"
            )

            return SourceAnalysisItem(
                source_id=self._make_source_id(url),
                url=url,
                domain=domain,
                title=source.get("title"),
                citation_count=source[
                    "citation_count"
                ],
                cited_by_models=source[
                    "cited_by_models"
                ],
                fetch_status="analysis_failed",
                fetch_error=str(exc),
            )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    @staticmethod
    def _normalize_url(
        url: str,
    ) -> str:

        url = url.strip()

        # Remove fragments.
        url = url.split("#", 1)[0]

        # Remove trailing slash except for root.
        parsed = urlparse(url)

        if not parsed.scheme:
            return url

        path = parsed.path.rstrip("/")

        normalized = (
            f"{parsed.scheme}://"
            f"{parsed.netloc}"
            f"{path}"
        )

        if parsed.query:
            normalized += f"?{parsed.query}"

        return normalized

    @staticmethod
    def _extract_domain(
        url: str,
    ) -> str:

        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""

    @staticmethod
    def _make_source_id(
        url: str,
    ) -> str:

        import hashlib

        return hashlib.sha1(
            url.encode("utf-8")
        ).hexdigest()[:12]

    @staticmethod
    def _clean_text(
        value: str,
    ) -> str:

        return " ".join(
            str(value).lower().split()
        )

    @staticmethod
    def _ensure_list(
        value: Any,
    ) -> list[str]:

        if isinstance(value, list):
            return [
                str(item)
                for item in value
                if item
            ]

        if isinstance(value, str) and value.strip():
            return [value.strip()]

        return []

    @staticmethod
    def _extract_response_text(
        response: Any,
    ) -> str:

        parts = []

        for block in response.content:

            if getattr(block, "type", None) == "text":
                parts.append(block.text)

        return "\n".join(parts).strip()

    @staticmethod
    def _parse_json(
        text: str,
    ) -> dict[str, Any]:

        text = text.strip()

        # Remove markdown code fences.
        if text.startswith("```"):
            text = re.sub(
                r"^```(?:json)?",
                "",
                text,
                flags=re.IGNORECASE,
            )

            text = re.sub(
                r"```$",
                "",
                text,
            )

            text = text.strip()

        try:
            result = json.loads(text)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

        # Try extracting the first JSON object.
        match = re.search(
            r"\{.*\}",
            text,
            flags=re.DOTALL,
        )

        if match:

            result = json.loads(
                match.group(0)
            )

            if isinstance(result, dict):
                return result

        raise ValueError(
            "Claude did not return valid JSON"
        )


# ============================================================
# Convenience function
# ============================================================


def run_source_intelligence(
    prompt_analysis: dict[str, Any],
    prompt_evidence: PromptEvidence,
) -> SourceAnalysis:

    agent = SourceIntelligenceAgent()

    return agent.run(
        prompt_analysis=prompt_analysis,
        prompt_evidence=prompt_evidence,
    )