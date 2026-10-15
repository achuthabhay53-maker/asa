"""Agent 3 web search through Vertex AI and direct page extraction."""
from __future__ import annotations

import os
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import quote, urlparse

import httpx
from dotenv import load_dotenv

load_dotenv()

SEARCH_TIMEOUT_SECONDS = 30
EXTRACT_TIMEOUT_SECONDS = 60
_ATTEMPTS = 2
_MAX_HTML_CHARS = 1_000_000
_BLOCK_ELEMENTS = {
    "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt",
    "h1", "h2", "h3", "h4", "h5", "h6", "li", "main", "ol", "p",
    "section", "table", "td", "th", "tr", "ul",
}
_IGNORED_ELEMENTS = {"footer", "header", "nav", "noscript", "script", "style", "svg"}


@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str | None = None
    published: str | None = None


@dataclass(frozen=True)
class SearchResponse:
    results: list[SearchResult]
    failed: bool = False


@dataclass(frozen=True)
class ExtractedPage:
    url: str
    text: str
    title: str | None = None


class _VisibleTextParser(HTMLParser):
    """Extract readable page text and its title without a third-party service."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.title_parts: list[str] = []
        self._ignored: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in _IGNORED_ELEMENTS:
            self._ignored.append(tag)
            return
        if self._ignored:
            return
        if tag == "title":
            self._in_title = True
        if tag in _BLOCK_ELEMENTS:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._ignored:
            self._ignored.remove(tag)
            return
        if self._ignored:
            return
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_ELEMENTS:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored:
            return
        if self._in_title:
            self.title_parts.append(data)
        else:
            self.text_parts.append(data)


class WebResearch:
    """Search with Vertex AI Google Search grounding and extract result pages."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("VERTEX_API_KEY")
        if not self.api_key:
            raise RuntimeError("VERTEX_API_KEY is not set.")
        self.model = model or os.getenv("VERTEX_MODEL_NAME") or "gemini-2.5-flash-lite"
        self.client = client or httpx.Client(follow_redirects=True)

    def search(
        self,
        query: str,
        *,
        max_results: int = 2,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> SearchResponse:
        """Return Google-grounded result URLs for a query."""
        if max_results < 1:
            raise ValueError("max_results must be at least 1.")

        search_query = query.strip()
        if include_domains:
            search_query += " " + " OR ".join(f"site:{domain}" for domain in include_domains)
        if exclude_domains:
            search_query += " " + " ".join(f"-site:{domain}" for domain in exclude_domains)

        url = (
            "https://aiplatform.googleapis.com/v1/publishers/google/models/"
            f"{quote(self.model, safe='-_.')}:generateContent"
        )
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{
                    "text": (
                        "Use Google Search to find webpages relevant to this query. "
                        "Return a brief response grounded in those pages.\n\n"
                        f"Search query: {search_query}"
                    ),
                }],
            }],
            "tools": [{"googleSearch": {}}],
            "generationConfig": {"temperature": 1.0, "maxOutputTokens": 512},
        }
        response = self._call(
            "Vertex AI Google Search",
            lambda: self.client.post(
                url,
                params={"key": self.api_key},
                json=payload,
                timeout=SEARCH_TIMEOUT_SECONDS,
            ),
        )
        if response is None:
            return SearchResponse(results=[], failed=True)

        body = response.json()
        candidates = body.get("candidates") or []
        grounding = candidates[0].get("groundingMetadata") or {} if candidates else {}
        chunks = grounding.get("groundingChunks") or []
        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for chunk in chunks:
            web = chunk.get("web") or {}
            result_url = web.get("uri") or web.get("url")
            if not isinstance(result_url, str) or not result_url.startswith(("http://", "https://")):
                continue
            if result_url in seen_urls or not self._matches_domains(
                result_url, include_domains, exclude_domains
            ):
                continue
            seen_urls.add(result_url)
            title = web.get("title")
            results.append(
                SearchResult(
                    url=result_url,
                    title=str(title) if title else None,
                )
            )
            if len(results) >= max_results:
                break
        return SearchResponse(results=results)

    def extract(self, url: str, *, max_chars: int = 12_000) -> ExtractedPage | None:
        """Fetch and extract readable text from one search result URL."""
        response = self._call(
            "Page extraction",
            lambda: self.client.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/154.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
                },
                timeout=EXTRACT_TIMEOUT_SECONDS,
            ),
        )
        if response is None:
            return None

        content_type = response.headers.get("content-type", "").lower()
        if content_type and not any(
            kind in content_type for kind in ("html", "xml", "text/plain")
        ):
            return None

        raw = response.text[:_MAX_HTML_CHARS]
        if "html" not in content_type and "xml" not in content_type:
            text = raw.strip()
            title = None
        else:
            parser = _VisibleTextParser()
            parser.feed(raw)
            text = " ".join(" ".join(parser.text_parts).split())
            title = " ".join(" ".join(parser.title_parts).split()) or None
        if not text:
            return None
        return ExtractedPage(
            url=str(response.url),
            text=text[:max_chars],
            title=title,
        )

    @staticmethod
    def _matches_domains(
        url: str,
        include_domains: list[str] | None,
        exclude_domains: list[str] | None,
    ) -> bool:
        hostname = (urlparse(url).hostname or "").lower()

        def matches(domain: str) -> bool:
            expected = domain.lower().lstrip(".")
            return hostname == expected or hostname.endswith(f".{expected}")

        if include_domains and not any(matches(domain) for domain in include_domains):
            return False
        return not exclude_domains or not any(matches(domain) for domain in exclude_domains)

    @staticmethod
    def _call(label: str, action) -> httpx.Response | None:
        for attempt in range(_ATTEMPTS):
            try:
                response = action()
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                if attempt == _ATTEMPTS - 1:
                    # Do not include exception text: request URLs may contain API keys.
                    print(f"  {label} failed after {_ATTEMPTS} attempts ({type(exc).__name__}).")
        return None


def get_web_research() -> WebResearch:
    """Create the Vertex AI-backed research client from environment settings."""
    return WebResearch()
