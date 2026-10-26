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
from urllib.parse import urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

from dotenv import load_dotenv
import httpx
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
    failure_stage: str | None = None
    requested_url: str | None = None
    title: str | None = None


@dataclass
class SiteUrlSkip:
    url: str
    reason_category: str
    reason: str


@dataclass
class SiteUrlInventory:
    """Sitemap-derived URLs and whether discovery reached its page cap."""

    urls: list[str]
    complete: bool
    sitemap_count: int
    discovered_url_count: int = 0
    skipped_urls: list[SiteUrlSkip] | None = None
    note: str | None = None


@dataclass
class SiteSearchCandidates:
    """Theme-searched target URLs selected for Tavily extraction."""

    urls: list[str]
    theme_urls: dict[str, list[str]]
    discovered_url_count: int
    skipped_urls: list[SiteUrlSkip]
    note: str | None = None


def discover_site_urls(
    site_url: str,
    max_urls: int = 2000,
) -> SiteUrlInventory:
    """Discover same-host pages from robots.txt and XML sitemaps."""

    parsed_site = urlsplit(site_url.strip())
    if (
        parsed_site.scheme not in {"http", "https"}
        or not parsed_site.hostname
        or parsed_site.username
        or parsed_site.password
    ):
        return SiteUrlInventory(
            urls=[],
            complete=False,
            sitemap_count=0,
            discovered_url_count=0,
            skipped_urls=[],
            note="The supplied company URL is invalid.",
        )

    origin = urlunsplit(
        (
            parsed_site.scheme,
            parsed_site.netloc,
            "",
            "",
            "",
        )
    )
    allowed_host = parsed_site.hostname.lower()

    def same_host(url: str) -> bool:
        try:
            parsed = urlsplit(url)
        except ValueError:
            return False
        return (
            parsed.scheme in {"http", "https"}
            and parsed.hostname is not None
            and parsed.hostname.lower() == allowed_host
            and parsed.username is None
            and parsed.password is None
        )

    sitemap_candidates: list[str] = []
    incomplete = False
    skipped_urls: list[SiteUrlSkip] = []

    try:
        robots_response = httpx.get(
            f"{origin}/robots.txt",
            timeout=10.0,
            follow_redirects=False,
        )
        if robots_response.is_success:
            for line in robots_response.text.splitlines():
                directive, separator, value = line.partition(":")
                if separator and directive.strip().lower() == "sitemap":
                    sitemap_url = urljoin(origin, value.strip())
                    if same_host(sitemap_url):
                        sitemap_candidates.append(sitemap_url)
                    else:
                        incomplete = True
                        skipped_urls.append(
                            SiteUrlSkip(
                                url=sitemap_url,
                                reason_category="URL filtering",
                                reason="robots.txt declared a sitemap on a different host.",
                            )
                        )
        elif robots_response.status_code not in {404, 410}:
            incomplete = True
    except httpx.HTTPError:
        incomplete = True

    if not sitemap_candidates:
        sitemap_candidates.append(f"{origin}/sitemap.xml")

    sitemap_queue = list(sitemap_candidates)
    visited_sitemaps: set[str] = set()
    discovered_urls: list[str] = []
    discovered_url_count = 0
    max_sitemaps = 10

    while sitemap_queue and len(visited_sitemaps) < max_sitemaps:
        sitemap_url = sitemap_queue.pop(0)
        if sitemap_url in visited_sitemaps:
            continue
        visited_sitemaps.add(sitemap_url)

        try:
            response = httpx.get(
                sitemap_url,
                timeout=10.0,
                follow_redirects=False,
            )
            if not response.is_success:
                incomplete = True
                continue
            root = ElementTree.fromstring(response.content)
        except (httpx.HTTPError, ElementTree.ParseError):
            incomplete = True
            continue

        root_kind = root.tag.rsplit("}", 1)[-1].lower()
        locations = [
            element.text.strip()
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1].lower() == "loc"
            and element.text
            and element.text.strip()
        ]

        if root_kind == "sitemapindex":
            for location in locations:
                child_url = urljoin(sitemap_url, location)
                if same_host(child_url):
                    sitemap_queue.append(child_url)
                else:
                    incomplete = True
                    skipped_urls.append(
                        SiteUrlSkip(
                            url=child_url,
                            reason_category="URL filtering",
                            reason="Sitemap index referenced a different host.",
                        )
                    )
        elif root_kind == "urlset":
            for location in locations:
                page_url = urljoin(sitemap_url, location)
                discovered_url_count += 1
                if same_host(page_url):
                    if page_url in discovered_urls:
                        skipped_urls.append(
                            SiteUrlSkip(
                                url=page_url,
                                reason_category="duplicate URL",
                                reason="URL appeared more than once in sitemap data.",
                            )
                        )
                    elif len(discovered_urls) >= max_urls:
                        incomplete = True
                        skipped_urls.append(
                            SiteUrlSkip(
                                url=page_url,
                                reason_category="URL filtering",
                                reason=f"Page limit of {max_urls} URLs reached.",
                            )
                        )
                    else:
                        discovered_urls.append(page_url)
                else:
                    incomplete = True
                    skipped_urls.append(
                        SiteUrlSkip(
                            url=page_url,
                            reason_category="URL filtering",
                            reason="Sitemap page URL was not on the supplied host.",
                        )
                    )
        else:
            incomplete = True

    if sitemap_queue:
        incomplete = True

    unique_urls = list(dict.fromkeys(discovered_urls))
    if not unique_urls:
        return SiteUrlInventory(
            urls=[site_url.strip()],
            complete=False,
            sitemap_count=len(visited_sitemaps),
            discovered_url_count=discovered_url_count,
            skipped_urls=skipped_urls,
            note="No usable sitemap was found; only the supplied URL can be checked.",
        )

    return SiteUrlInventory(
        urls=unique_urls[:max_urls],
        complete=not incomplete,
        sitemap_count=len(visited_sitemaps),
        discovered_url_count=discovered_url_count,
        skipped_urls=skipped_urls,
        note=(
            "Sitemap discovery was incomplete or truncated."
            if incomplete
            else None
        ),
    )


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

    def search_site_pages(
        self,
        site_url: str,
        themes: list[str],
        per_theme_limit: int = 3,
        total_limit: int = 15,
    ) -> SiteSearchCandidates:
        """Find a small set of same-host pages relevant to verified themes."""
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

    def search_site_pages(
        self,
        site_url: str,
        themes: list[str],
        per_theme_limit: int = 3,
        total_limit: int = 15,
    ) -> SiteSearchCandidates:
        parsed_site = urlsplit(site_url.strip())
        if (
            parsed_site.scheme not in {"http", "https"}
            or not parsed_site.hostname
            or parsed_site.username
            or parsed_site.password
        ):
            return SiteSearchCandidates(
                urls=[],
                theme_urls={},
                discovered_url_count=0,
                skipped_urls=[],
                note="The supplied company URL is invalid.",
            )

        allowed_host = parsed_site.hostname.lower()
        selected_urls: list[str] = []
        theme_urls: dict[str, list[str]] = {}
        skipped_urls: list[SiteUrlSkip] = []
        discovered_url_count = 0
        search_errors: list[str] = []

        for theme in themes:
            try:
                response = self.client.search(
                    query=theme,
                    search_depth="basic",
                    max_results=per_theme_limit,
                    include_domains=[allowed_host],
                    include_raw_content=False,
                    include_images=False,
                    timeout=30,
                )
            except Exception as exc:
                search_errors.append(f"{theme}: {exc}")
                theme_urls[theme] = []
                continue

            matching_urls: list[str] = []
            for result in response.get("results", []):
                url = result.get("url")
                if not isinstance(url, str) or not url:
                    continue
                discovered_url_count += 1

                try:
                    parsed_result = urlsplit(url)
                except ValueError:
                    parsed_result = None

                if (
                    parsed_result is None
                    or parsed_result.scheme not in {"http", "https"}
                    or not parsed_result.hostname
                    or parsed_result.hostname.lower() != allowed_host
                    or parsed_result.username
                    or parsed_result.password
                ):
                    skipped_urls.append(
                        SiteUrlSkip(
                            url=url,
                            reason_category="URL filtering",
                            reason="Search result was not on the supplied host.",
                        )
                    )
                    continue

                if url in matching_urls:
                    skipped_urls.append(
                        SiteUrlSkip(
                            url=url,
                            reason_category="duplicate URL",
                            reason=f"Search returned this URL more than once for {theme}.",
                        )
                    )
                    continue

                matching_urls.append(url)

                if url not in selected_urls:
                    if len(selected_urls) >= total_limit:
                        skipped_urls.append(
                            SiteUrlSkip(
                                url=url,
                                reason_category="URL filtering",
                                reason=f"Target page limit of {total_limit} reached.",
                            )
                        )
                        continue
                    selected_urls.append(url)

            theme_urls[theme] = [
                url
                for url in matching_urls
                if url in selected_urls
            ][:per_theme_limit]

        return SiteSearchCandidates(
            urls=selected_urls,
            theme_urls=theme_urls,
            discovered_url_count=discovered_url_count,
            skipped_urls=skipped_urls,
            note=(
                "; ".join(search_errors)
                if search_errors
                else "Targeted theme search is not an exhaustive site inventory."
            ),
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
                            url=(
                                extracted.get("final_url")
                                or extracted.get("resolved_url")
                                or extracted.get("url")
                                or url
                            ),
                            content=content,
                            success=bool(content.strip()),
                            error=(
                                None
                                if content.strip()
                                else "Tavily returned empty extracted content."
                            ),
                            failure_stage=(
                                None
                                if content.strip()
                                else "tavily_empty_content"
                            ),
                            requested_url=url,
                            title=extracted.get("title"),
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
                        failure_stage="tavily_failed_result",
                        requested_url=url,
                    )

            except Exception as exc:
                for source_id, url in batch:
                    results[source_id] = FetchedPage(
                        source_id=source_id,
                        url=url,
                        content="",
                        success=False,
                        error=str(exc),
                        failure_stage="tavily_batch",
                        requested_url=url,
                    )

        return results


def get_source_fetcher() -> SourceFetcher:
    """
    Factory used by the rest of ACA.

    This lets us replace Tavily later without changing the Analyzer.
    """

    return TavilySourceFetcher()