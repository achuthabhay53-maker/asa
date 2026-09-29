"""Bounded public-web fetches for cited-source verification."""

import asyncio
import ipaddress
import re
import socket
from html.parser import HTMLParser
from typing import Literal, Mapping
from urllib.parse import urljoin, urlsplit

import httpx
from pydantic import BaseModel


MAX_FETCHES = 30
MAX_REDIRECTS = 4
MAX_PAGE_BYTES = 1_000_000
FETCH_CONCURRENCY = 4


class FetchedPage(BaseModel):
    """Bounded page content fetched for source verification."""

    source_id: str
    status: Literal["fetched", "unavailable", "skipped_limit"]
    final_url: str | None = None
    title: str | None = None
    text: str = ""
    note: str | None = None


class _VisiblePageParser(HTMLParser):
    _IGNORED_TAGS = {"script", "style", "noscript", "svg", "template"}
    _SEPARATOR_TAGS = {
        "address", "article", "br", "div", "h1", "h2", "h3", "li", "p", "section"
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self.title: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag in self._IGNORED_TAGS:
            self._ignored_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self._SEPARATOR_TAGS and not self._ignored_depth:
            self.text.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self._SEPARATOR_TAGS and not self._ignored_depth:
            self.text.append(" ")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._in_title:
            self.title.append(data)
        self.text.append(data)


async def _validate_public_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Only public HTTP(S) URLs are allowed.")

    expected_port = 443 if parsed.scheme == "https" else 80
    port = parsed.port or expected_port
    if port != expected_port:
        raise ValueError("Non-standard URL ports are not allowed.")

    hostname = parsed.hostname.rstrip(".")
    try:
        addresses = {ipaddress.ip_address(hostname)}
    except ValueError:
        records = await asyncio.get_running_loop().getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
        )
        addresses = {
            ipaddress.ip_address(record[4][0])
            for record in records
        }

    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("URL host does not resolve to a public address.")


def _extract_page_text(body: bytes, content_type: str) -> tuple[str, str | None]:
    charset_match = re.search(
        r"charset=([\w.-]+)",
        content_type,
        re.IGNORECASE,
    )
    encoding = charset_match.group(1) if charset_match else "utf-8"
    try:
        decoded = body.decode(encoding, errors="replace")
    except LookupError:
        decoded = body.decode("utf-8", errors="replace")

    if "html" not in content_type.lower() and "xhtml" not in content_type.lower():
        return re.sub(r"\s+", " ", decoded).strip(), None

    parser = _VisiblePageParser()
    parser.feed(decoded)
    page_text = re.sub(r"\s+", " ", " ".join(parser.text)).strip()
    title = re.sub(r"\s+", " ", " ".join(parser.title)).strip()
    return page_text, title or None


async def _fetch_page(
    client: httpx.AsyncClient,
    source_id: str,
    url: str | None,
) -> FetchedPage:
    if not url:
        return FetchedPage(
            source_id=source_id,
            status="unavailable",
            note="No source URL was supplied.",
        )

    current_url = url
    for redirect_count in range(MAX_REDIRECTS + 1):
        try:
            await _validate_public_url(current_url)
            async with client.stream(
                "GET",
                current_url,
                follow_redirects=False,
                headers={"Accept": "text/html,application/xhtml+xml,text/plain"},
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location or redirect_count == MAX_REDIRECTS:
                        return FetchedPage(
                            source_id=source_id,
                            status="unavailable",
                            note="Too many redirects or missing redirect target.",
                        )
                    current_url = urljoin(current_url, location)
                    continue

                content_type = response.headers.get("content-type", "")
                if response.status_code != 200:
                    return FetchedPage(
                        source_id=source_id,
                        status="unavailable",
                        note=f"Source returned HTTP {response.status_code}.",
                    )
                if not any(
                    media_type in content_type.lower()
                    for media_type in (
                        "text/html",
                        "application/xhtml+xml",
                        "text/plain",
                    )
                ):
                    return FetchedPage(
                        source_id=source_id,
                        status="unavailable",
                        note="Source did not return HTML or plain text.",
                    )

                declared_size = response.headers.get("content-length")
                if declared_size and int(declared_size) > MAX_PAGE_BYTES:
                    return FetchedPage(
                        source_id=source_id,
                        status="unavailable",
                        note="Source page exceeded the fetch size limit.",
                    )

                chunks = []
                total_bytes = 0
                async for chunk in response.aiter_bytes():
                    total_bytes += len(chunk)
                    if total_bytes > MAX_PAGE_BYTES:
                        return FetchedPage(
                            source_id=source_id,
                            status="unavailable",
                            note="Source page exceeded the fetch size limit.",
                        )
                    chunks.append(chunk)

                page_text, title = _extract_page_text(
                    b"".join(chunks),
                    content_type,
                )
                if not page_text:
                    return FetchedPage(
                        source_id=source_id,
                        status="unavailable",
                        note="No readable page text was extracted.",
                    )
                return FetchedPage(
                    source_id=source_id,
                    status="fetched",
                    final_url=current_url,
                    title=title,
                    text=page_text,
                )
        except (httpx.HTTPError, OSError, ValueError, UnicodeError):
            return FetchedPage(
                source_id=source_id,
                status="unavailable",
                note="The cited page could not be safely fetched.",
            )

    return FetchedPage(
        source_id=source_id,
        status="unavailable",
        note="Too many redirects.",
    )


async def fetch_public_pages(
    urls_by_source: Mapping[str, str | None],
) -> dict[str, FetchedPage]:
    """Fetch bounded public HTML/text pages for the supplied citations."""

    source_items = list(urls_by_source.items())
    pages = {
        source_id: FetchedPage(
            source_id=source_id,
            status="skipped_limit",
            note="The per-prompt source verification limit was reached.",
        )
        for source_id, _ in source_items[MAX_FETCHES:]
    }
    fetchable = source_items[:MAX_FETCHES]
    timeout = httpx.Timeout(12.0, connect=5.0)
    limits = httpx.Limits(
        max_connections=FETCH_CONCURRENCY,
        max_keepalive_connections=FETCH_CONCURRENCY,
    )
    semaphore = asyncio.Semaphore(FETCH_CONCURRENCY)

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        follow_redirects=False,
        trust_env=False,
        headers={"User-Agent": "ACA-SourceVerifier/0.1"},
    ) as client:
        async def fetch_one(source_id: str, url: str | None) -> FetchedPage:
            async with semaphore:
                return await _fetch_page(client, source_id, url)

        fetched = await asyncio.gather(
            *(fetch_one(source_id, url) for source_id, url in fetchable)
        )

    pages.update((page.source_id, page) for page in fetched)
    return pages