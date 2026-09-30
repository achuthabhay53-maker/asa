"""
Source fetching abstraction for ACA.

The Source Analyzer should not know whether pages are fetched using
Tavily, Firecrawl, Exa, or another provider.

For v0.2, Tavily Extract is the implementation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()


@dataclass
class FetchedPage:
    """
    Normalized webpage content returned to the Source Analyzer.
    """

    source_id: str
    url: str
    content: str
    success: bool
    error: str | None = None


class SourceFetcher(Protocol):
    """
    Provider-neutral interface.

    The analyzer depends on this interface, not Tavily directly.
    """

    def fetch_pages(
        self,
        sources: dict[str, str],
    ) -> dict[str, FetchedPage]:
        """
        Fetch multiple known URLs.

        Args:
            sources:
                Mapping of source_id -> URL.

        Returns:
            Mapping of source_id -> FetchedPage.
        """
        ...


class TavilySourceFetcher:
    """
    Tavily implementation of SourceFetcher.
    """

    def __init__(self) -> None:
        api_key = os.getenv("TAVILY_API_KEY")

        # Tavily currently permits keyless extract usage, so we don't
        # require the key during development.
        self.client = TavilyClient(
            api_key=api_key or None,
        )

    def fetch_pages(
        self,
        sources: dict[str, str],
    ) -> dict[str, FetchedPage]:

        if not sources:
            return {}

        results: dict[str, FetchedPage] = {}

        # Tavily Extract accepts multiple URLs in one request.
        items = list(sources.items())

        for start in range(0, len(items), 20):
            batch = items[start:start + 20]

            source_ids = [source_id for source_id, _ in batch]
            urls = [url for _, url in batch]

            try:
                response = self.client.extract(
                    urls=urls,
                    format="markdown",
                    extract_depth="advanced",
                )

                extracted_by_url = {
                    item["url"]: item
                    for item in response.get("results", [])
                }

                failed_by_url = {
                    item["url"]: item
                    for item in response.get("failed_results", [])
                }

                for source_id, url in batch:

                    extracted = extracted_by_url.get(url)

                    if extracted:
                        content = extracted.get("raw_content") or ""

                        results[source_id] = FetchedPage(
                            source_id=source_id,
                            url=url,
                            content=content,
                            success=bool(content.strip()),
                        )
                        continue

                    failed = failed_by_url.get(url)

                    error = None

                    if failed:
                        error = (
                            failed.get("error")
                            or failed.get("reason")
                            or "Tavily extraction failed."
                        )

                    results[source_id] = FetchedPage(
                        source_id=source_id,
                        url=url,
                        content="",
                        success=False,
                        error=error or "No content returned.",
                    )

            except Exception as exc:
                for source_id, url in batch:
                    results[source_id] = FetchedPage(
                        source_id=source_id,
                        url=url,
                        content="",
                        success=False,
                        error=str(exc),
                    )

        return results


def get_source_fetcher() -> SourceFetcher:
    """
    Factory used by the rest of ACA.

    This lets us replace Tavily later without changing the Analyzer.
    """

    return TavilySourceFetcher()