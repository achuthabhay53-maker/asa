"""Agent 2: Source Intelligence for AImpact.

Independently analyzes citations already stored for one prompt. Ranking is
separate from content analysis: the top sources stay in ``ranked_sources``
even when a page cannot be fetched, and those pages are listed in
``retry_sources``.

It does not search the web for new sources, treat a page's claims as
verified facts about the customer, or write a content strategy.
"""
from __future__ import annotations

import json
import os
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from dotenv import load_dotenv

from aca.models.source_intelligence import (
    AnalyzedSource,
    CitationSummary,
    ContentAnalysis,
    ContentStructure,
    RankedSource,
    RetrySource,
    SourceFinding,
    SourceIntelligence,
    WritingStyle,
)

load_dotenv()


# How many highest-cited sources to fetch and analyze.
# Override at runtime with --top-n or SourceIntelligenceAgent(top_n=...).
TOP_N_SOURCES = 5

FETCH_TIMEOUT = 20.0
MAX_HTML_CHARS = 1_000_000
MAX_PAGE_CHARS = 24_000
MAX_HEADINGS = 40

_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "gclid",
    "fbclid",
    "mc_cid",
    "mc_eid",
    "igshid",
    "si",
}
_REDIRECT_HOSTS = {
    "vertexaisearch.cloud.google.com",
    "vertexaisearch.googleapis.com",
    "www.google.com",
    "google.com",
}
_BLOCKED_STATUS = {401, 403, 407, 429, 451}


class _PageStructure(HTMLParser):
    """Measures headings, lists, tables, and FAQ markers from one page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.headings: list[str] = []
        self.has_tables = False
        self.has_lists = False
        self.has_faq = False
        self.title = ""
        self.text_parts: list[str] = []
        self._ignore = 0
        self._chrome = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._heading_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attr = {key.lower(): (value or "") for key, value in attrs}
        blob = " ".join(
            [
                attr.get("id", ""),
                attr.get("class", ""),
                attr.get("itemtype", ""),
                attr.get("role", ""),
            ]
        ).lower()
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignore += 1
        if tag in {"nav", "header", "footer"}:
            self._chrome += 1
        if tag == "title":
            self._in_title = True
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self._ignore == 0:
            self._heading_parts = []
        if tag == "table" and self._chrome == 0:
            self.has_tables = True
        if tag in {"ul", "ol"} and self._chrome == 0 and self._ignore == 0:
            self.has_lists = True
        if "faq" in blob or "frequentlyasked" in blob.replace(" ", ""):
            self.has_faq = True
        if tag in {"p", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"} and self._ignore:
            self._ignore -= 1
        if tag in {"nav", "header", "footer"} and self._chrome:
            self._chrome -= 1
        if tag == "title":
            self._in_title = False
            self.title = _clean_whitespace(" ".join(self._title_parts))
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self._heading_parts is not None:
            heading = _clean_whitespace(" ".join(self._heading_parts))
            if heading and heading not in self.headings and len(self.headings) < MAX_HEADINGS:
                self.headings.append(heading)
                if re.search(r"\bfaq\b|frequently asked", heading, flags=re.I):
                    self.has_faq = True
            self._heading_parts = None
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignore:
            return
        if self._in_title:
            self._title_parts.append(data)
        if self._heading_parts is not None:
            self._heading_parts.append(data)
        if data.strip():
            self.text_parts.append(data)


def _clean_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _value(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _public_id(value: Any) -> int | str:
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def _normalize_url(url: str) -> str:
    raw = url.strip()
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMS
    ]
    path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", urlencode(query), ""))


def _is_redirect_host(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in _REDIRECT_HOSTS or "grounding-api-redirect" in url


def _extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^\s*```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Claude did not return a valid JSON object.")
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from Claude.")
    return data


def _string_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    values: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = _clean_whitespace(str(item))
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            values.append(text)
    return values


def _looks_like_challenge(text: str, html: str) -> bool:
    sample = f"{text[:1500]}\n{html[:4000]}".lower()
    markers = (
        "cf-challenge",
        "just a moment",
        "attention required",
        "enable javascript and cookies",
        "captcha",
        "access denied",
    )
    return len(text) < 800 and any(marker in sample for marker in markers)


def _parse_page(html: str) -> tuple[ContentStructure, str, str]:
    parser = _PageStructure()
    parser.feed(html[:MAX_HTML_CHARS])
    parser.close()
    text = _clean_whitespace(" ".join(parser.text_parts))
    structure = ContentStructure(
        headings=parser.headings,
        has_faq=parser.has_faq,
        has_tables=parser.has_tables,
        has_lists=parser.has_lists,
    )
    return structure, parser.title, text


def _collect_sources(prompt_evidence: Any) -> list[dict[str, Any]]:
    per_model = _value(prompt_evidence, "per_model", {}) or {}
    grouped: dict[str, dict[str, Any]] = {}
    for model_name, evidence in per_model.items():
        for citation in _value(evidence, "sources", []) or []:
            raw_url = str(_value(citation, "url", "") or "").strip()
            normalized = _normalize_url(raw_url)
            if not normalized:
                continue
            source = grouped.get(normalized)
            if source is None:
                title = _value(citation, "title", None)
                source = {
                    "url": normalized,
                    "title": str(title).strip() if title else None,
                    "citation_count": 0,
                    "cited_by_models": [],
                }
                grouped[normalized] = source
            source["citation_count"] += 1
            model = str(model_name)
            if model not in source["cited_by_models"]:
                source["cited_by_models"].append(model)
    return list(grouped.values())


def _rank_sources(sources: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    ranked = sorted(
        sources,
        key=lambda source: (
            -int(source["citation_count"]),
            -len(source["cited_by_models"]),
            1 if _is_redirect_host(source["url"]) else 0,
            source["url"],
        ),
    )
    return ranked[:top_n]


class SourceIntelligenceAgent:
    def __init__(self, model: str | None = None, top_n: int | None = None) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        if top_n is None:
            top_n = TOP_N_SOURCES
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        self.top_n = top_n
        self.api_key = api_key
        self.model = model or os.getenv("ANTHROPIC_MODEL_NAME") or "claude-haiku-4-5-20251001"

    def analyze(self, campaign_id: int, prompt_evidence: Any) -> SourceIntelligence:
        prompt_id = _public_id(_value(prompt_evidence, "prompt_id", ""))
        if prompt_id == "":
            raise ValueError("Selected AImpact prompt has no prompt_id.")
        sources = _collect_sources(prompt_evidence)
        occurrences = sum(int(source["citation_count"]) for source in sources)
        selected = _rank_sources(sources, self.top_n)
        ranked_sources: list[RankedSource] = []
        content_analyses: list[AnalyzedSource] = []
        retry_sources: list[RetrySource] = []
        for index, source in enumerate(selected, start=1):
            source_id = f"source_{index:03d}"
            ranked_sources.append(
                RankedSource(
                    source_id=source_id,
                    rank=index,
                    url=source["url"],
                    title=source.get("title"),
                    citation_count=source["citation_count"],
                    cited_by_models=list(source["cited_by_models"]),
                )
            )
            analysis, retry = self._analyze_source(source_id, source)
            if analysis is not None:
                content_analyses.append(analysis)
            if retry is not None:
                retry_sources.append(retry)
        return SourceIntelligence(
            campaign_id=_public_id(campaign_id),
            prompt_id=prompt_id,
            citation_summary=CitationSummary(
                total_citation_occurrences=occurrences,
                unique_sources=len(sources),
            ),
            ranked_sources=ranked_sources,
            content_analyses=content_analyses,
            retry_sources=retry_sources,
        )

    def _analyze_source(
        self,
        source_id: str,
        source: dict[str, Any],
    ) -> tuple[AnalyzedSource | None, RetrySource | None]:
        fetched = _fetch_page(source["url"])
        if fetched["status"] != "success":
            return None, RetrySource(
                source_id=source_id,
                url=source["url"],
                title=source.get("title"),
                fetch_status=fetched["status"],
            )

        structure: ContentStructure = fetched["structure"]
        try:
            described = self._describe_page(
                url=fetched["final_url"],
                title=fetched["title"] or source.get("title") or "",
                page_text=fetched["text"],
                structure=structure,
            )
        except Exception as exc:
            print(f"  Style analysis failed for {source_id}: {exc}")
            described = {
                "summary": "The page was fetched, but its writing style could not be analyzed.",
                "main_topics": [],
                "findings": [],
                "writing_style": {},
            }
        style = described.get("writing_style") or {}
        findings = []
        normalized_page_text = _clean_whitespace(fetched["text"]).casefold()
        for item in described.get("findings") or []:
            if not isinstance(item, dict):
                continue
            finding = _clean_whitespace(str(item.get("finding") or ""))
            quote = _clean_whitespace(str(item.get("evidence_quote") or ""))
            if finding:
                findings.append(
                    SourceFinding(
                        finding=finding,
                        evidence_quote=quote,
                        quote_found_on_page=(
                            len(quote) >= 25
                            and quote.casefold() in normalized_page_text
                        ),
                    )
                )
        return AnalyzedSource(
            source_id=source_id,
            url=fetched["final_url"],
            title=fetched["title"] or source.get("title"),
            fetch_status="success",
            content_analysis=ContentAnalysis(
                summary=str(described.get("summary") or "").strip(),
                main_topics=_string_list(described.get("main_topics")),
                findings=findings,
                content_structure=structure,
                writing_style=WritingStyle(
                    tone=str(style.get("tone") or "").strip(),
                    format=str(style.get("format") or "").strip(),
                    audience=str(style.get("audience") or "").strip(),
                    paragraph_style=str(style.get("paragraph_style") or "").strip(),
                    evidence_style=str(style.get("evidence_style") or "").strip(),
                ),
            ),
        ), None

    def _describe_page(
        self,
        url: str,
        title: str,
        page_text: str,
        structure: ContentStructure,
    ) -> dict[str, Any]:
        system_prompt = """You describe one fetched web page for Agent 2: Source Intelligence.
Return one JSON object and nothing else:
{
  "summary": "",
  "main_topics": [],
  "findings": [
    {
      "finding": "One concrete observation about this page",
      "evidence_quote": "Exact contiguous quote copied from page_text"
    }
  ],
  "writing_style": {
    "tone": "",
    "format": "",
    "audience": "",
    "paragraph_style": "",
    "evidence_style": ""
  }
}
Rules:
- Describe only the supplied page. Do not say a style, topic, or structure is common, typical, or shared across sources.
- summary is a short account of what this page itself says.
- main_topics are topics present on this page.
- findings are concrete factual observations about the page's content. Include
  an exact contiguous evidence_quote copied from page_text for each finding.
  Quotes must be at least 25 characters. Do not create a finding if no quote
  directly supports it.
- writing_style describes this page's tone, format, audience, paragraph style, and how it uses evidence.
- Do not treat companies or claims on the page as verified facts about the customer.
- Do not propose a blog, outline, or content strategy.
- Do not add sources, URLs, or fields. JSON only."""
        payload = {
            "url": url,
            "title": title,
            "measured_structure": structure.model_dump(),
            "page_text": page_text[:MAX_PAGE_CHARS],
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
                "max_tokens": 1200,
                "system": system_prompt,
                "messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        blocks = response.json().get("content") or []
        text_blocks = [block.get("text", "") for block in blocks if block.get("type") == "text"]
        if not text_blocks:
            raise RuntimeError("Claude returned no text for Agent 2.")
        return _extract_json("\n".join(text_blocks))


def _fetch_page(url: str) -> dict[str, Any]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/154.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    }
    try:
        with httpx.Client(timeout=FETCH_TIMEOUT, follow_redirects=True, headers=headers) as client:
            response = client.get(url)
            if response.status_code in _BLOCKED_STATUS:
                return {"status": "blocked"}
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "html" not in content_type and "xml" not in content_type and content_type:
                return {"status": "unavailable"}
            html = response.text[:MAX_HTML_CHARS]
            structure, title, text = _parse_page(html)
            if _looks_like_challenge(text, html):
                return {"status": "blocked"}
            if not text:
                return {"status": "unavailable"}
            return {
                "status": "success",
                "final_url": str(response.url),
                "title": title or None,
                "text": text,
                "structure": structure,
            }
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        return {"status": "blocked" if status in _BLOCKED_STATUS else "unavailable"}
    except httpx.HTTPError:
        return {"status": "unavailable"}


def run_agent2(
    campaign_id: int,
    prompt_id: int | str,
    top_n: int | None = None,
    score_threshold: float = 70,
) -> SourceIntelligence:
    """Run Agent 2 using the temporary AImpact PostgreSQL reader."""
    from aca.tools.aimpact import get_prompt_evidence

    records = get_prompt_evidence(campaign_id=campaign_id, score_threshold=score_threshold)
    selected = next((item for item in records if str(_value(item, "prompt_id", "")) == str(prompt_id)), None)
    if selected is None:
        raise ValueError(
            f"Prompt {prompt_id} not found in campaign {campaign_id} "
            f"with score_threshold={score_threshold}."
        )
    return SourceIntelligenceAgent(top_n=top_n).analyze(campaign_id, selected)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AImpact Agent 2: Source Intelligence")
    parser.add_argument("--campaign-id", type=int, required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--top-n", type=int, default=TOP_N_SOURCES)
    parser.add_argument("--score-threshold", type=float, default=70)
    parser.add_argument("--output", default="agent2_output.json")
    args = parser.parse_args()

    result = run_agent2(args.campaign_id, args.prompt_id, args.top_n, args.score_threshold)
    payload = result.model_dump()
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("=" * 68)
    print("AIMPACT AGENT 2 — SOURCE INTELLIGENCE")
    print("=" * 68)
    print(f"Campaign ID: {result.campaign_id}")
    print(f"Prompt ID: {result.prompt_id}")
    print(f"Citation occurrences: {result.citation_summary.total_citation_occurrences}")
    print(f"Unique sources: {result.citation_summary.unique_sources}")
    print(f"Ranked sources: {len(result.ranked_sources)}")
    print(f"Content analyses: {len(result.content_analyses)}")
    print(f"Needs retry: {len(result.retry_sources)}")
    for source in result.retry_sources:
        print(f"- {source.source_id} [{source.fetch_status}] {source.url}")
    print(f"Output saved to: {os.path.abspath(args.output)}")
