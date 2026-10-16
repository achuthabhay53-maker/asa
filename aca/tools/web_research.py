"""Agent 3 web search through Exa and direct page extraction."""
from __future__ import annotations

import os
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse

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
    """Search with Exa and extract result pages."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        api_key: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise RuntimeError("EXA_API_KEY is not set.")
        self.client = client or httpx.Client(follow_redirects=True)
        self._search_contents: dict[str, tuple[str, str | None]] = {}

    def search(
        self,
        query: str,
        *,
        max_results: int = 2,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> SearchResponse:
        """Return Exa result URLs for a query."""
        if max_results < 1:
            raise ValueError("max_results must be at least 1.")
        self._search_contents.clear()

        payload = {
            "query": query.strip(),
            "type": "fast",
            "numResults": max_results,
            "contents": {"text": {"maxCharacters": 12_000}},
        }
        if include_domains:
            payload["includeDomains"] = include_domains
        if exclude_domains:
            payload["excludeDomains"] = exclude_domains

        response = self._call(
            "Exa search",
            lambda: self.client.post(
                "https://api.exa.ai/search",
                headers={"x-api-key": self.api_key},
                json=payload,
                timeout=SEARCH_TIMEOUT_SECONDS,
            ),
        )
        if response is None:
            return SearchResponse(results=[], failed=True)

        body = response.json()
        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for item in body.get("results") or []:
            result_url = item.get("url")
            if not isinstance(result_url, str) or not result_url.startswith(("http://", "https://")):
                continue
            if result_url in seen_urls or not self._matches_domains(
                result_url, include_domains, exclude_domains
            ):
                continue
            seen_urls.add(result_url)
            title = item.get("title")
            published = item.get("publishedDate")
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                self._search_contents[result_url] = (
                    text.strip(),
                    str(title) if title else None,
                )
            results.append(
                SearchResult(
                    url=result_url,
                    title=str(title) if title else None,
                    published=str(published) if published else None,
                )
            )
            if len(results) >= max_results:
                break
        return SearchResponse(results=results)

    def extract(self, url: str, *, max_chars: int = 12_000) -> ExtractedPage | None:
        """Return Exa-extracted text, falling back to a direct HTTP page fetch."""
        cached = self._search_contents.get(url)
        if cached:
            text, title = cached
            return ExtractedPage(url=url, text=text[:max_chars], title=title)

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
                    status = (
                        f"HTTP {exc.response.status_code}"
                        if isinstance(exc, httpx.HTTPStatusError)
                        else type(exc).__name__
                    )
                    print(f"  {label} failed after {_ATTEMPTS} attempts ({status}).")
        return None


def get_web_research() -> WebResearch:
    """Create the Exa-backed research client from environment settings."""
    return WebResearch()
