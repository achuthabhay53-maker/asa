"""Shared web research tool.

Agent 3 calls this module for search and page extract. The Tavily client
stays here, so a later provider can replace it without changing the agent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

SEARCH_TIMEOUT_SECONDS = 30
EXTRACT_TIMEOUT_SECONDS = 60
_ATTEMPTS = 2


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


class WebResearch:
    """Search the web and extract page text."""

    def __init__(self, client: TavilyClient | None = None, api_key: str | None = None) -> None:
        if client is None:
            key = api_key if api_key is not None else os.getenv("TAVILY_API_KEY")
            if not key:
                raise RuntimeError("TAVILY_API_KEY is not set.")
            client = TavilyClient(api_key=key)
        self.client = client

    def search(
        self,
        query: str,
        *,
        max_results: int = 2,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> SearchResponse:
        """Return result URLs. ``failed`` is true when the search call itself fails."""
        kwargs: dict[str, object] = {
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_raw_content": False,
            "include_images": False,
            "timeout": SEARCH_TIMEOUT_SECONDS,
        }
        if include_domains:
            kwargs["include_domains"] = include_domains
        if exclude_domains:
            kwargs["exclude_domains"] = exclude_domains
        response = self._call("Search", lambda: self.client.search(**kwargs))
        if response is None:
            return SearchResponse(results=[], failed=True)
        results: list[SearchResult] = []
        for item in response.get("results") or []:
            url = item.get("url")
            if not isinstance(url, str) or not url:
                continue
            title = item.get("title")
            published = item.get("published_date")
            results.append(SearchResult(
                url=url,
                title=str(title) if title else None,
                published=str(published) if published else None,
            ))
        return SearchResponse(results=results, failed=False)

    def extract(self, url: str, *, max_chars: int = 12_000) -> ExtractedPage | None:
        """Extract one page. Returns None when the page has no usable text."""
        response = self._call(
            "Extract",
            lambda: self.client.extract(urls=[url], format="markdown", extract_depth="advanced", timeout=EXTRACT_TIMEOUT_SECONDS),
        )
        if response is None:
            return None
        for item in response.get("results") or []:
            text = str(item.get("raw_content") or "").strip()
            if not text:
                continue
            title = item.get("title")
            resolved = item.get("final_url") or item.get("resolved_url") or item.get("url") or url
            return ExtractedPage(url=str(resolved), text=text[:max_chars], title=str(title) if title else None)
        return None

    def _call(self, label: str, action):
        error: Exception | None = None
        for _ in range(_ATTEMPTS):
            try:
                return action()
            except Exception as exc:
                error = exc
        print(f"  {label} failed: {error}")
        return None


def get_web_research() -> WebResearch:
    """Factory used by agents. Reads TAVILY_API_KEY."""
    return WebResearch()
