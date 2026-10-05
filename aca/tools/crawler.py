"""
ACA Website Evidence Crawler
=============================

Purpose
-------
Crawl an entire website and build a reusable evidence inventory for ACA.

This is a deterministic crawling TOOL, not an AI agent.

The crawler preserves:
- HTML page content as Markdown
- page title
- metadata
- requested URL
- final URL
- redirect URL when available
- discovered links
- PDFs
- PDF extracted text
- crawl status/errors
- checkpoint/resume state

Architecture
------------
Sitemap / robots.txt
        |
        v
   URL scheduler
        |
        v
  Concurrent Crawl4AI workers
        |
        +---- HTML -> Markdown
        |
        +---- PDF  -> pypdf text extraction
        |
        v
 Evidence inventory JSON
        |
        v
 Checkpoint for resume

Current implementation:
- Crawl4AI 0.9.4
- asyncio concurrency
- default 8 concurrent workers
- unlimited pages
- no Source Analyzer integration yet
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import (
    parse_qsl,
    urlencode,
    urljoin,
    urlparse,
    urlunparse,
)
from urllib.robotparser import RobotFileParser
from html.parser import HTMLParser

from dotenv import load_dotenv
from pypdf import PdfReader

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()


# Website to crawl.
START_URL = os.getenv(
    "ACA_CRAWL_URL",
    "https://alamarbio.com/",
)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
#
# Start with 8.
#
# If the website remains stable and you want more speed:
#
#   MAX_CONCURRENT = 12
#
# Do not immediately use 50/100 workers.
#
MAX_CONCURRENT = int(
    os.getenv("ACA_CRAWL_CONCURRENCY", "12")
)


# ---------------------------------------------------------------------------
# Crawl limits
# ---------------------------------------------------------------------------
#
# None = unlimited.
#
# The crawler will continue until the queue is empty.
#
MAX_PAGES: int | None = None


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

OUTPUT_FILE = Path(
    os.getenv(
        "ACA_CRAWL_OUTPUT",
        "crawl4ai_site_output.json",
    )
)

CHECKPOINT_FILE = Path(
    os.getenv(
        "ACA_CRAWL_CHECKPOINT",
        "crawl4ai_site_checkpoint.json",
    )
)


# ---------------------------------------------------------------------------
# PDF settings
# ---------------------------------------------------------------------------

PDF_TIMEOUT_SECONDS = 60


# ---------------------------------------------------------------------------
# Request settings
# ---------------------------------------------------------------------------

ROBOTS_TIMEOUT_SECONDS = 15
SITEMAP_TIMEOUT_SECONDS = 30


# Tracking parameters that should not create separate pages.
TRACKING_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
}


# Binary/static resources that should not enter the HTML crawl queue.
BINARY_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".avif",
    ".svg",
    ".ico",
    ".bmp",
    ".tif",
    ".tiff",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".mp3",
    ".wav",
    ".ogg",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".css",
    ".js",
    ".map",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
}


# =============================================================================
# HELPERS
# =============================================================================


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def normalize_url(url: str) -> str:
    """
    Normalize a URL for deduplication.

    Rules:
    - lower-case scheme
    - lower-case hostname
    - remove www.
    - remove default ports
    - remove fragments
    - collapse repeated slashes in path
    - normalize trailing slash
    - remove common tracking parameters
    - sort query parameters
    """
    if not url:
        return ""

    url = url.strip()

    if not url:
        return ""

    parsed = urlparse(url)

    if parsed.scheme.lower() not in {"http", "https"}:
        return ""

    scheme = parsed.scheme.lower()

    hostname = parsed.hostname
    if not hostname:
        return ""

    hostname = hostname.lower()

    if hostname.startswith("www."):
        hostname = hostname[4:]

    port = parsed.port

    if port is None:
        netloc = hostname
    elif (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    ):
        netloc = hostname
    else:
        netloc = f"{hostname}:{port}"

    path = parsed.path or "/"

    # Collapse repeated slashes.
    path = re.sub(r"/{2,}", "/", path)

    # Normalize trailing slash.
    if path != "/":
        path = path.rstrip("/")

    # Remove tracking parameters.
    query_pairs = parse_qsl(
        parsed.query,
        keep_blank_values=True,
    )

    filtered_pairs = [
        (key, value)
        for key, value in query_pairs
        if key.lower() not in TRACKING_QUERY_PARAMS
    ]

    filtered_pairs.sort()

    query = urlencode(
        filtered_pairs,
        doseq=True,
    )

    return urlunparse(
        (
            scheme,
            netloc,
            path,
            "",
            query,
            "",
        )
    )


def is_pdf(url: str) -> bool:
    """Return True when URL appears to point to a PDF."""
    parsed = urlparse(url)

    path = parsed.path.lower()

    return (
        path.endswith(".pdf")
        or ".pdf" in path
    )


def is_binary_asset(url: str) -> bool:
    """Return True for non-HTML binary/static resources."""
    parsed = urlparse(url)

    path = parsed.path.lower()

    if is_pdf(url):
        return False

    return any(
        path.endswith(extension)
        for extension in BINARY_EXTENSIONS
    )


def is_same_site(url: str, root_host: str) -> bool:
    """Check whether URL belongs to the target website."""
    try:
        hostname = urlparse(url).hostname

        if not hostname:
            return False

        hostname = hostname.lower()

        if hostname.startswith("www."):
            hostname = hostname[4:]

        return hostname == root_host
    except Exception:
        return False


def should_queue_url(
    url: str,
    root_host: str,
) -> bool:
    """
    Decide whether an URL should enter the crawl queue.
    """
    normalized = normalize_url(url)

    if not normalized:
        return False

    if not is_same_site(
        normalized,
        root_host,
    ):
        return False

    if is_binary_asset(normalized):
        return False

    return True


# =============================================================================
# HTML TITLE EXTRACTION
# =============================================================================


class TitleParser(HTMLParser):
    """Small HTML title extractor."""

    def __init__(self) -> None:
        super().__init__()

        self.in_title = False
        self.parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() == "title":
            self.in_title = True

    def handle_endtag(
        self,
        tag: str,
    ) -> None:
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(
        self,
        data: str,
    ) -> None:
        if self.in_title:
            self.parts.append(data)

    @property
    def title(self) -> str:
        return " ".join(
            part.strip()
            for part in self.parts
            if part.strip()
        ).strip()


def extract_html_title(
    html: str | None,
) -> str | None:
    """Extract <title> from raw HTML."""
    if not html:
        return None

    try:
        parser = TitleParser()
        parser.feed(html)

        return parser.title or None
    except Exception:
        return None


# =============================================================================
# ROBOTS.TXT
# =============================================================================


@dataclass
class RobotsPolicy:
    """
    robots.txt policy.

    If robots.txt cannot be reached because of a network/server problem,
    we fail closed and do not crawl.

    If robots.txt returns 404, crawling is allowed.
    """

    root_url: str
    user_agent: str = "ACA-Crawler"

    def __post_init__(self) -> None:
        self.parser = RobotFileParser()
        self.available = False
        self.allow_all = False
        self.sitemaps: list[str] = []

    def load(self) -> None:
        parsed = urlparse(self.root_url)

        robots_url = urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                "/robots.txt",
                "",
                "",
                "",
            )
        )

        print(
            f"[ROBOTS] Loading: {robots_url}"
        )

        request = urllib.request.Request(
            robots_url,
            headers={
                "User-Agent": self.user_agent,
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=ROBOTS_TIMEOUT_SECONDS,
            ) as response:
                status = getattr(
                    response,
                    "status",
                    200,
                )

                raw = response.read()

            if status == 404:
                print(
                    "[ROBOTS] robots.txt not found; "
                    "allowing crawl."
                )

                self.allow_all = True
                self.available = True
                return

            if status >= 400:
                raise RuntimeError(
                    f"robots.txt returned HTTP {status}"
                )

            text = raw.decode(
                "utf-8",
                errors="replace",
            )

            self.parser.parse(
                text.splitlines()
            )

            self.available = True

            for line in text.splitlines():
                stripped = line.strip()

                if not stripped.lower().startswith(
                    "sitemap:"
                ):
                    continue

                sitemap = stripped.split(
                    ":",
                    1,
                )[1].strip()

                if sitemap:
                    self.sitemaps.append(
                        normalize_url(sitemap)
                    )

            self.sitemaps = list(
                dict.fromkeys(
                    sitemap
                    for sitemap in self.sitemaps
                    if sitemap
                )
            )

            print(
                f"[ROBOTS] Loaded successfully. "
                f"Sitemaps declared: {len(self.sitemaps)}"
            )

        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print(
                    "[ROBOTS] robots.txt returned 404; "
                    "allowing crawl."
                )

                self.allow_all = True
                self.available = True
                return

            print(
                f"[ROBOTS] HTTP error: {exc}"
            )

            raise RuntimeError(
                "Could not safely determine robots.txt policy."
            ) from exc

        except Exception as exc:
            print(
                f"[ROBOTS] Failed to load robots.txt: {exc}"
            )

            raise RuntimeError(
                "Could not safely determine robots.txt policy."
            ) from exc

    def can_fetch(
        self,
        url: str,
    ) -> bool:
        """Check whether robots.txt permits crawling."""
        if self.allow_all:
            return True

        if not self.available:
            return False

        try:
            return self.parser.can_fetch(
                self.user_agent,
                url,
            )
        except Exception:
            return False


# =============================================================================
# SITEMAP DISCOVERY
# =============================================================================


def download_url_bytes(
    url: str,
    timeout: int,
) -> bytes:
    """Download bytes using urllib."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ACA-Crawler/1.0",
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=timeout,
    ) as response:
        return response.read()


def parse_sitemap_urls(
    content: bytes,
) -> tuple[list[str], list[str]]:
    """
    Parse sitemap XML.

    Returns:
        urls
        nested_sitemaps
    """
    try:
        if content[:2] == b"\x1f\x8b":
            content = gzip.decompress(content)

        root = ET.fromstring(content)

        tag = root.tag.lower()

        urls: list[str] = []
        nested: list[str] = []

        for element in root.iter():
            local_name = (
                element.tag.split("}")[-1]
                .lower()
            )

            if local_name != "loc":
                continue

            if not element.text:
                continue

            value = normalize_url(
                element.text.strip()
            )

            if not value:
                continue

            if tag.endswith("sitemapindex"):
                nested.append(value)
            else:
                urls.append(value)

        return urls, nested

    except Exception as exc:
        print(
            f"[SITEMAP] XML parsing failed: {exc}"
        )

        return [], []


def discover_sitemaps(
    start_url: str,
    robots: RobotsPolicy,
) -> list[str]:
    """
    Discover sitemap URLs from:
    - robots.txt
    - /sitemap.xml
    - /sitemap_index.xml
    - nested sitemap indexes
    """
    parsed = urlparse(start_url)

    base = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            "",
            "",
            "",
            "",
        )
    )

    candidates: list[str] = []

    candidates.extend(
        robots.sitemaps
    )

    candidates.extend(
        [
            normalize_url(
                urljoin(
                    base,
                    "/sitemap.xml",
                )
            ),
            normalize_url(
                urljoin(
                    base,
                    "/sitemap_index.xml",
                )
            ),
        ]
    )

    candidates = list(
        dict.fromkeys(
            value
            for value in candidates
            if value
        )
    )

    print(
        f"[SITEMAP] Initial candidates: "
        f"{len(candidates)}"
    )

    discovered_urls: list[str] = []

    visited_sitemaps: set[str] = set()

    queue = deque(candidates)

    max_sitemaps = 100

    while queue and len(visited_sitemaps) < max_sitemaps:
        sitemap_url = queue.popleft()

        sitemap_url = normalize_url(
            sitemap_url
        )

        if not sitemap_url:
            continue

        if sitemap_url in visited_sitemaps:
            continue

        visited_sitemaps.add(
            sitemap_url
        )

        if not is_same_site(
            sitemap_url,
            urlparse(
                normalize_url(start_url)
            ).hostname.replace("www.", ""),
        ):
            continue

        print(
            f"[SITEMAP] Reading: {sitemap_url}"
        )

        try:
            content = download_url_bytes(
                sitemap_url,
                SITEMAP_TIMEOUT_SECONDS,
            )

            urls, nested = parse_sitemap_urls(
                content
            )

            for url in urls:
                discovered_urls.append(url)

            for nested_sitemap in nested:
                if (
                    nested_sitemap
                    not in visited_sitemaps
                ):
                    queue.append(
                        nested_sitemap
                    )

            print(
                f"[SITEMAP] "
                f"urls={len(urls)} "
                f"nested={len(nested)}"
            )

        except Exception as exc:
            print(
                f"[SITEMAP] Failed: "
                f"{sitemap_url} -> {exc}"
            )

    unique_urls = list(
        dict.fromkeys(
            discovered_urls
        )
    )

    print(
        f"[SITEMAP] Total discovered URLs: "
        f"{len(unique_urls)}"
    )

    return unique_urls


# =============================================================================
# CRAWL4AI RESULT EXTRACTION
# =============================================================================


def extract_markdown(
    result: Any,
) -> str:
    """
    Extract Markdown safely across Crawl4AI versions.

    Crawl4AI can expose markdown as:
    - plain string
    - MarkdownGenerationResult with raw_markdown
    """
    markdown_result = getattr(
        result,
        "markdown",
        None,
    )

    if markdown_result is None:
        return ""

    if hasattr(
        markdown_result,
        "raw_markdown",
    ):
        return (
            getattr(
                markdown_result,
                "raw_markdown",
                "",
            )
            or ""
        )

    if isinstance(
        markdown_result,
        str,
    ):
        return markdown_result

    return str(
        markdown_result
    )


def extract_metadata(
    result: Any,
) -> dict[str, Any]:
    """Safely extract Crawl4AI metadata."""
    metadata = getattr(
        result,
        "metadata",
        None,
    )

    if metadata is None:
        return {}

    if isinstance(
        metadata,
        dict,
    ):
        return metadata

    try:
        return dict(metadata)
    except Exception:
        return {
            "raw": str(metadata)
        }


def extract_links(
    result: Any,
    root_host: str,
) -> list[str]:
    """Extract and normalize internal links."""
    raw_links = getattr(
        result,
        "links",
        None,
    )

    if not raw_links:
        return []

    candidates: list[str] = []

    if isinstance(
        raw_links,
        dict,
    ):
        for value in raw_links.values():
            if isinstance(
                value,
                list,
            ):
                for item in value:
                    if isinstance(
                        item,
                        str,
                    ):
                        candidates.append(
                            item
                        )

                    elif isinstance(
                        item,
                        dict,
                    ):
                        href = (
                            item.get("href")
                            or item.get("url")
                        )

                        if href:
                            candidates.append(
                                href
                            )

    elif isinstance(
        raw_links,
        list,
    ):
        for item in raw_links:
            if isinstance(
                item,
                str,
            ):
                candidates.append(item)

            elif isinstance(
                item,
                dict,
            ):
                href = (
                    item.get("href")
                    or item.get("url")
                )

                if href:
                    candidates.append(
                        href
                    )

    normalized: list[str] = []

    for link in candidates:
        link = normalize_url(link)

        if not link:
            continue

        if not is_same_site(
            link,
            root_host,
        ):
            continue

        if is_binary_asset(link):
            continue

        normalized.append(link)

    return list(
        dict.fromkeys(
            normalized
        )
    )


# =============================================================================
# PDF EXTRACTION
# =============================================================================


def extract_pdf_text(
    content: bytes,
) -> dict[str, Any]:
    """
    Extract PDF text page by page.

    One page failure does not fail the entire document.
    """
    result: dict[str, Any] = {
        "page_count": 0,
        "extraction_status": "success",
        "extracted_text_length": 0,
        "text_extractable": False,
        "text": "",
        "extraction_error": None,
    }

    try:
        reader = PdfReader(
            BytesIO(content),
            strict=False,
        )

        page_count = len(
            reader.pages
        )

        result["page_count"] = page_count

        page_texts: list[str] = []

        page_failures: list[str] = []

        for index, page in enumerate(
            reader.pages,
            start=1,
        ):
            try:
                text = page.extract_text() or ""

                text = text.strip()

                page_texts.append(
                    f"--- PDF PAGE {index} ---\n"
                    f"{text}"
                )

            except Exception as exc:
                page_failures.append(
                    f"Page {index}: {exc}"
                )

                page_texts.append(
                    f"--- PDF PAGE {index} ---\n"
                    "[TEXT EXTRACTION FAILED]"
                )

        full_text = "\n\n".join(
            page_texts
        )

        result["text"] = full_text
        result["extracted_text_length"] = len(
            full_text
        )

        result["text_extractable"] = bool(
            full_text.strip()
        )

        if page_failures:
            result["extraction_status"] = (
                "partial"
            )

            result["extraction_error"] = (
                "; ".join(page_failures)
            )

        elif not full_text.strip():
            result["extraction_status"] = (
                "no_text"
            )

        return result

    except Exception as exc:
        result["extraction_status"] = (
            "extraction_failed"
        )

        result["extraction_error"] = str(
            exc
        )

        return result


def crawl_pdf_sync(
    requested_url: str,
) -> dict[str, Any]:
    """
    Download and extract a PDF.

    This is synchronous, so it must run via asyncio.to_thread()
    to prevent blocking other Crawl4AI workers.
    """
    started = time.perf_counter()

    record: dict[str, Any] = {
        "type": "pdf",
        "requested_url": requested_url,
        "final_url": requested_url,
        "redirected_url": None,
        "success": False,
        "status_code": None,
        "content_type": None,
        "content_length": None,
        "download_bytes": 0,
        "download_status": None,
        "download_error": None,
        "filename": Path(
            urlparse(requested_url).path
        ).name,
        "page_count": 0,
        "extraction_status": None,
        "extracted_text_length": 0,
        "text_extractable": False,
        "text": "",
        "extraction_error": None,
        "started_at": utc_now(),
        "duration_seconds": None,
    }

    request = urllib.request.Request(
        requested_url,
        headers={
            "User-Agent": "ACA-Crawler/1.0",
            "Accept": "application/pdf,*/*",
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=PDF_TIMEOUT_SECONDS,
        ) as response:
            content = response.read()

            final_url = response.geturl()

            status_code = getattr(
                response,
                "status",
                None,
            )

            content_type = (
                response.headers.get(
                    "Content-Type"
                )
            )

            content_length = (
                response.headers.get(
                    "Content-Length"
                )
            )

        record["final_url"] = normalize_url(
            final_url
        ) or requested_url

        record["status_code"] = status_code
        record["content_type"] = content_type
        record["content_length"] = (
            int(content_length)
            if content_length
            and content_length.isdigit()
            else None
        )

        record["download_bytes"] = len(
            content
        )

        record["download_status"] = (
            "success"
        )

        extraction = extract_pdf_text(
            content
        )

        record.update(
            extraction
        )

        record["success"] = (
            extraction["extraction_status"]
            in {
                "success",
                "partial",
                "no_text",
            }
        )

        return record

    except Exception as exc:
        record["download_status"] = (
            "failed"
        )

        record["download_error"] = str(
            exc
        )

        return record

    finally:
        record["duration_seconds"] = round(
            time.perf_counter()
            - started,
            3,
        )


# =============================================================================
# CHECKPOINT
# =============================================================================


def save_json(
    path: Path,
    data: Any,
) -> None:
    """
    Atomically write JSON.

    Writes to a temporary file first so an interrupted write is less likely
    to corrupt the main file.
    """
    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            data,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    temp_path.replace(path)


def save_checkpoint(
    *,
    start_url: str,
    queue: deque[str],
    queued: set[str],
    visited: set[str],
    records: list[dict[str, Any]],
) -> None:
    """Save crawler state."""
    data = {
        "version": 2,
        "saved_at": utc_now(),
        "start_url": start_url,
        "queue": list(queue),
        "queued": sorted(queued),
        "visited": sorted(visited),
        "records": records,
    }

    save_json(
        CHECKPOINT_FILE,
        data,
    )


def load_checkpoint(
    start_url: str,
) -> tuple[
    deque[str],
    set[str],
    set[str],
    list[dict[str, Any]],
]:
    """Load checkpoint if it belongs to this crawl."""
    if not CHECKPOINT_FILE.exists():
        return (
            deque(),
            set(),
            set(),
            [],
        )

    try:
        with CHECKPOINT_FILE.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(handle)

        saved_start_url = normalize_url(
            data.get("start_url", "")
        )

        current_start_url = normalize_url(
            start_url
        )

        if saved_start_url != current_start_url:
            print(
                "[CHECKPOINT] Existing checkpoint "
                "belongs to another website. "
                "Ignoring it."
            )

            return (
                deque(),
                set(),
                set(),
                [],
            )

        queue = deque(
            data.get(
                "queue",
                [],
            )
        )

        queued = set(
            data.get(
                "queued",
                [],
            )
        )

        visited = set(
            data.get(
                "visited",
                [],
            )
        )

        records = data.get(
            "records",
            [],
        )

        print(
            "[CHECKPOINT] Resuming previous crawl:"
        )

        print(
            f"    visited={len(visited)}"
        )

        print(
            f"    queued={len(queue)}"
        )

        print(
            f"    records={len(records)}"
        )

        return (
            queue,
            queued,
            visited,
            records,
        )

    except Exception as exc:
        print(
            f"[CHECKPOINT] Failed to load: {exc}"
        )

        print(
            "[CHECKPOINT] Starting fresh."
        )

        return (
            deque(),
            set(),
            set(),
            [],
        )


# =============================================================================
# EVIDENCE INVENTORY
# =============================================================================


def build_inventory(
    *,
    start_url: str,
    root_host: str,
    records: list[dict[str, Any]],
    remaining_queue: deque[str],
    started_at: str,
    finished_at: str | None,
    complete: bool,
) -> dict[str, Any]:
    """Build final JSON inventory."""
    html_records = [
        record
        for record in records
        if record.get("type") == "html"
    ]

    pdf_records = [
        record
        for record in records
        if record.get("type") == "pdf"
    ]

    successful_html = [
        record
        for record in html_records
        if record.get("success")
    ]

    failed_html = [
        record
        for record in html_records
        if not record.get("success")
    ]

    successful_pdf_downloads = [
        record
        for record in pdf_records
        if record.get(
            "download_status"
        )
        == "success"
    ]

    successful_pdf_extractions = [
        record
        for record in pdf_records
        if record.get(
            "extraction_status"
        )
        in {
            "success",
            "partial",
            "no_text",
        }
    ]

    thin_pages = [
        record
        for record in successful_html
        if len(
            record.get(
                "markdown",
                "",
            )
        ) < 500
    ]

    return {
        "crawler": {
            "name": "ACA Crawl4AI Website Evidence Crawler",
            "version": 2,
            "crawl4ai_version": "0.9.4",
            "concurrency": MAX_CONCURRENT,
            "max_pages": MAX_PAGES,
        },
        "website": {
            "requested_url": start_url,
            "normalized_url": normalize_url(
                start_url
            ),
            "root_host": root_host,
        },
        "crawl": {
            "started_at": started_at,
            "finished_at": finished_at,
            "complete": complete,
            "remaining_queue": len(
                remaining_queue
            ),
        },
        "summary": {
            "records": len(records),
            "html_pages": len(
                html_records
            ),
            "successful_html_pages": len(
                successful_html
            ),
            "failed_html_pages": len(
                failed_html
            ),
            "thin_html_pages": len(
                thin_pages
            ),
            "pdfs_found": len(
                pdf_records
            ),
            "pdf_downloads_successful": len(
                successful_pdf_downloads
            ),
            "pdf_extractions_successful": len(
                successful_pdf_extractions
            ),
        },
        "pages": records,
        "remaining_queue": list(
            remaining_queue
        ),
    }


# =============================================================================
# CONCURRENT CRAWLER
# =============================================================================


class ConcurrentCrawler:
    """
    Controlled asynchronous crawler.

    Multiple Crawl4AI browser requests run concurrently.

    Shared state is updated by the coordinator, rather than allowing each
    worker to mutate the queue directly.
    """

    def __init__(
        self,
        *,
        start_url: str,
        root_host: str,
        robots: RobotsPolicy,
        initial_queue: deque[str],
        initial_queued: set[str],
        initial_visited: set[str],
        initial_records: list[dict[str, Any]],
    ) -> None:
        self.start_url = start_url

        self.root_host = root_host

        self.robots = robots

        self.queue = initial_queue

        self.queued = initial_queued

        self.visited = initial_visited

        self.records = initial_records

        self.semaphore = asyncio.Semaphore(
            MAX_CONCURRENT
        )

        self.state_lock = asyncio.Lock()

        self.last_checkpoint = time.monotonic()

        self.pages_since_checkpoint = 0

    # -------------------------------------------------------------------------
    # Queue management
    # -------------------------------------------------------------------------

    def add_url(
        self,
        url: str,
    ) -> bool:
        """Add URL to queue if it has not been queued/visited."""
        normalized = normalize_url(url)

        if not normalized:
            return False

        if not should_queue_url(
            normalized,
            self.root_host,
        ):
            return False

        if normalized in self.visited:
            return False

        if normalized in self.queued:
            return False

        if not self.robots.can_fetch(
            normalized
        ):
            return False

        self.queue.append(
            normalized
        )

        self.queued.add(
            normalized
        )

        return True

    def add_urls(
        self,
        urls: list[str],
    ) -> int:
        """Add multiple URLs and return number added."""
        added = 0

        for url in urls:
            if self.add_url(url):
                added += 1

        return added

    # -------------------------------------------------------------------------
    # PDF worker
    # -------------------------------------------------------------------------

    async def process_pdf(
        self,
        url: str,
    ) -> dict[str, Any]:
        """Process PDF outside the Crawl4AI browser."""
        async with self.semaphore:
            return await asyncio.to_thread(
                crawl_pdf_sync,
                url,
            )

    # -------------------------------------------------------------------------
    # HTML worker
    # -------------------------------------------------------------------------

    async def process_html(
        self,
        crawler: AsyncWebCrawler,
        url: str,
    ) -> dict[str, Any]:
        """Crawl one HTML page with Crawl4AI."""
        started = time.perf_counter()

        record: dict[str, Any] = {
            "type": "html",
            "requested_url": url,
            "final_url": url,
            "redirected_url": None,
            "success": False,
            "status_code": None,
            "title": None,
            "metadata": {},
            "markdown": "",
            "markdown_length": 0,
            "discovered_links": [],
            "discovered_link_count": 0,
            "error": None,
            "started_at": utc_now(),
            "duration_seconds": None,
        }

        try:
            config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                word_count_threshold=0,
            )

            async with self.semaphore:
                result = await crawler.arun(
                    url=url,
                    config=config,
                )

            markdown = extract_markdown(
                result
            )

            metadata = extract_metadata(
                result
            )

            raw_html = getattr(
                result,
                "html",
                None,
            )

            title = (
                metadata.get("title")
                if isinstance(
                    metadata,
                    dict,
                )
                else None
            )

            if not title:
                title = extract_html_title(
                    raw_html
                )

            final_url_raw = (
                getattr(
                    result,
                    "url",
                    None,
                )
                or url
            )

            redirected_url = getattr(
                result,
                "redirected_url",
                None,
            )

            final_url = normalize_url(
                final_url_raw
            ) or url

            if redirected_url:
                redirected_url = (
                    normalize_url(
                        redirected_url
                    )
                    or redirected_url
                )

            discovered_links = extract_links(
                result,
                self.root_host,
            )

            status_code = getattr(
                result,
                "status_code",
                None,
            )

            success = bool(
                getattr(
                    result,
                    "success",
                    False,
                )
            )

            record.update(
                {
                    "final_url": final_url,
                    "redirected_url": redirected_url,
                    "success": success,
                    "status_code": status_code,
                    "title": title,
                    "metadata": metadata,
                    "markdown": markdown,
                    "markdown_length": len(
                        markdown
                    ),
                    "discovered_links": discovered_links,
                    "discovered_link_count": len(
                        discovered_links
                    ),
                }
            )

            return record

        except Exception as exc:
            record["error"] = str(
                exc
            )

            return record

        finally:
            record["duration_seconds"] = round(
                time.perf_counter()
                - started,
                3,
            )

    # -------------------------------------------------------------------------
    # One URL
    # -------------------------------------------------------------------------

    async def process_one(
        self,
        crawler: AsyncWebCrawler,
        url: str,
    ) -> dict[str, Any]:
        """
        Process a single URL.

        PDF and HTML take different paths.
        """
        if is_pdf(url):
            return await self.process_pdf(
                url
            )

        return await self.process_html(
            crawler,
            url,
        )

    # -------------------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------------------

    async def checkpoint(
        self,
    ) -> None:
        """Persist crawler state."""
        async with self.state_lock:
            save_checkpoint(
                start_url=self.start_url,
                queue=self.queue,
                queued=self.queued,
                visited=self.visited,
                records=self.records,
            )

            inventory = build_inventory(
                start_url=self.start_url,
                root_host=self.root_host,
                records=self.records,
                remaining_queue=self.queue,
                started_at="",
                finished_at=None,
                complete=False,
            )

            save_json(
                OUTPUT_FILE,
                inventory,
            )

            self.pages_since_checkpoint = 0
            self.last_checkpoint = (
                time.monotonic()
            )

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    async def run(
        self,
    ) -> None:
        """Run the concurrent crawl."""
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
        )

        started_at = utc_now()

        async with AsyncWebCrawler(
            config=browser_config,
        ) as crawler:

            while self.queue:
                # Respect unlimited/specified page limit.
                if (
                    MAX_PAGES is not None
                    and len(self.records)
                    >= MAX_PAGES
                ):
                    print(
                        "[LIMIT] Maximum page limit reached."
                    )

                    break

                batch: list[str] = []

                while (
                    self.queue
                    and len(batch)
                    < MAX_CONCURRENT
                ):
                    if (
                        MAX_PAGES is not None
                        and (
                            len(self.records)
                            + len(batch)
                        )
                        >= MAX_PAGES
                    ):
                        break

                    url = self.queue.popleft()

                    self.queued.discard(
                        url
                    )

                    if url in self.visited:
                        continue

                    self.visited.add(
                        url
                    )

                    batch.append(
                        url
                    )

                if not batch:
                    continue

                print()
                print(
                    "=" * 78
                )

                print(
                    f"[BATCH] Starting "
                    f"{len(batch)} workers"
                )

                print(
                    f"[BATCH] Queue before: "
                    f"{len(self.queue)}"
                )

                for index, url in enumerate(
                    batch,
                    start=1,
                ):
                    print(
                        f"    [{index}/{len(batch)}] "
                        f"{url}"
                    )

                # Run batch concurrently.
                tasks = [
                    asyncio.create_task(
                        self.process_one(
                            crawler,
                            url,
                        )
                    )
                    for url in batch
                ]

                results = await asyncio.gather(
                    *tasks,
                    return_exceptions=True,
                )

                total_new_urls = 0

                for url, result in zip(
                    batch,
                    results,
                ):
                    if isinstance(
                        result,
                        Exception,
                    ):
                        record = {
                            "type": (
                                "pdf"
                                if is_pdf(url)
                                else "html"
                            ),
                            "requested_url": url,
                            "final_url": url,
                            "success": False,
                            "error": str(
                                result
                            ),
                            "started_at": utc_now(),
                        }

                    else:
                        record = result

                    self.records.append(
                        record
                    )

                    if record.get(
                        "type"
                    ) == "html":
                        discovered = record.get(
                            "discovered_links",
                            [],
                        )

                        added = self.add_urls(
                            discovered
                        )

                        total_new_urls += added

                self.pages_since_checkpoint += len(
                    batch
                )

                # -----------------------------------------------------------------
                # Batch summary
                # -----------------------------------------------------------------

                successful = sum(
                    1
                    for record in results
                    if not isinstance(
                        record,
                        Exception,
                    )
                    and record.get(
                        "success"
                    )
                )

                failed = len(batch) - successful

                html_count = sum(
                    1
                    for record in results
                    if not isinstance(
                        record,
                        Exception,
                    )
                    and record.get(
                        "type"
                    )
                    == "html"
                )

                pdf_count = len(batch) - html_count

                print()
                print(
                    f"[BATCH] Finished: "
                    f"{successful}/{len(batch)} successful"
                )

                print(
                    f"[BATCH] Failed: "
                    f"{failed}"
                )

                print(
                    f"[BATCH] HTML: "
                    f"{html_count}"
                )

                print(
                    f"[BATCH] PDFs: "
                    f"{pdf_count}"
                )

                print(
                    f"[BATCH] New URLs discovered: "
                    f"{total_new_urls}"
                )

                print(
                    f"[BATCH] Queue after: "
                    f"{len(self.queue)}"
                )

                print(
                    f"[BATCH] Total records: "
                    f"{len(self.records)}"
                )

                # -----------------------------------------------------------------
                # Checkpoint every batch.
                #
                # This is deliberately frequent because the user wants reliable
                # evidence preservation.
                # -----------------------------------------------------------------

                await self.checkpoint()

                if (
                    not self.queue
                ):
                    print(
                        "[QUEUE] No URLs remaining."
                    )

            # Final inventory.
            finished_at = utc_now()

            inventory = build_inventory(
                start_url=self.start_url,
                root_host=self.root_host,
                records=self.records,
                remaining_queue=self.queue,
                started_at=started_at,
                finished_at=finished_at,
                complete=not bool(
                    self.queue
                ),
            )

            save_json(
                OUTPUT_FILE,
                inventory,
            )

            save_checkpoint(
                start_url=self.start_url,
                queue=self.queue,
                queued=self.queued,
                visited=self.visited,
                records=self.records,
            )


# =============================================================================
# MAIN
# =============================================================================


async def main() -> None:
    """Main crawler entry point."""
    normalized_start = normalize_url(
        START_URL
    )

    if not normalized_start:
        raise RuntimeError(
            f"Invalid ACA_CRAWL_URL: {START_URL}"
        )

    parsed = urlparse(
        normalized_start
    )

    root_host = parsed.hostname

    if not root_host:
        raise RuntimeError(
            f"Could not determine host from: "
            f"{normalized_start}"
        )

    root_host = root_host.lower()

    if root_host.startswith("www."):
        root_host = root_host[4:]

    print()
    print(
        "=" * 78
    )
    print(
        "ACA CRAWL4AI CONCURRENT WEBSITE EVIDENCE CRAWLER"
    )
    print(
        "=" * 78
    )

    print(
        f"Website:        {START_URL}"
    )

    print(
        f"Normalized:     {normalized_start}"
    )

    print(
        f"Root host:      {root_host}"
    )

    print(
        f"Concurrency:    {MAX_CONCURRENT}"
    )

    print(
        f"Max pages:      "
        f"{MAX_PAGES if MAX_PAGES is not None else 'UNLIMITED'}"
    )

    print(
        f"Output:         {OUTPUT_FILE.resolve()}"
    )

    print(
        f"Checkpoint:     {CHECKPOINT_FILE.resolve()}"
    )

    print()

    # -------------------------------------------------------------------------
    # Robots
    # -------------------------------------------------------------------------

    robots = RobotsPolicy(
        root_url=normalized_start
    )

    robots.load()

    # -------------------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------------------

    (
        queue,
        queued,
        visited,
        records,
    ) = load_checkpoint(
        normalized_start
    )

    # -------------------------------------------------------------------------
    # Fresh crawl
    # -------------------------------------------------------------------------

    if not queue and not records:
        print(
            "[SITEMAP] Discovering sitemap URLs..."
        )

        sitemap_urls = discover_sitemaps(
            normalized_start,
            robots,
        )

        queue = deque()

        queued = set()

        visited = set()

        records = []

        # Always crawl root first.
        if robots.can_fetch(
            normalized_start
        ):
            queue.append(
                normalized_start
            )

            queued.add(
                normalized_start
            )

        added_from_sitemap = 0

        for url in sitemap_urls:
            if should_queue_url(
                url,
                root_host,
            ) and robots.can_fetch(
                url
            ):
                if url not in queued:
                    queue.append(
                        url
                    )

                    queued.add(
                        url
                    )

                    added_from_sitemap += 1

        print()
        print(
            f"[QUEUE] Initial queue size: "
            f"{len(queue)}"
        )

        print(
            f"[QUEUE] Sitemap URLs added: "
            f"{added_from_sitemap}"
        )

    else:
        print()
        print(
            "[QUEUE] Resuming from checkpoint."
        )

        # Re-queue any records that were marked visited but never produced
        # a record. This protects against interruption between claiming a URL
        # and writing its result.
        recorded_urls = {
            normalize_url(
                record.get(
                    "requested_url",
                    "",
                )
            )
            for record in records
        }

        interrupted_urls = (
            visited - recorded_urls
        )

        if interrupted_urls:
            print(
                f"[CHECKPOINT] Re-queueing "
                f"{len(interrupted_urls)} "
                f"interrupted URLs."
            )

            for url in sorted(
                interrupted_urls
            ):
                if should_queue_url(
                    url,
                    root_host,
                ) and robots.can_fetch(
                    url
                ):
                    queue.append(
                        url
                    )

                    queued.add(
                        url
                    )

                    visited.discard(
                        url
                    )

    if not queue:
        print()
        print(
            "[QUEUE] Nothing to crawl."
        )

        return

    # -------------------------------------------------------------------------
    # Run concurrent crawler
    # -------------------------------------------------------------------------

    crawler = ConcurrentCrawler(
        start_url=normalized_start,
        root_host=root_host,
        robots=robots,
        initial_queue=queue,
        initial_queued=queued,
        initial_visited=visited,
        initial_records=records,
    )

    try:
        await crawler.run()

    except KeyboardInterrupt:
        print()
        print(
            "[STOP] Crawl interrupted by user."
        )

        await crawler.checkpoint()

        print(
            "[STOP] Checkpoint saved."
        )

        raise

    except Exception:
        # Save everything possible before propagating the error.
        await crawler.checkpoint()

        raise

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    final_records = crawler.records

    html_records = [
        record
        for record in final_records
        if record.get("type") == "html"
    ]

    pdf_records = [
        record
        for record in final_records
        if record.get("type") == "pdf"
    ]

    successful_html = [
        record
        for record in html_records
        if record.get("success")
    ]

    failed_html = [
        record
        for record in html_records
        if not record.get("success")
    ]

    successful_pdf_downloads = [
        record
        for record in pdf_records
        if record.get(
            "download_status"
        )
        == "success"
    ]

    successful_pdf_extractions = [
        record
        for record in pdf_records
        if record.get(
            "extraction_status"
        )
        in {
            "success",
            "partial",
            "no_text",
        }
    ]

    print()
    print(
        "=" * 78
    )
    print(
        "CRAWL FINISHED"
    )
    print(
        "=" * 78
    )

    print(
        f"HTML pages crawled:       "
        f"{len(html_records)}"
    )

    print(
        f"Successful HTML pages:    "
        f"{len(successful_html)}"
    )

    print(
        f"Failed HTML pages:        "
        f"{len(failed_html)}"
    )

    print(
        f"PDFs found:               "
        f"{len(pdf_records)}"
    )

    print(
        f"PDF downloads successful: "
        f"{len(successful_pdf_downloads)}"
    )

    print(
        f"PDF extractions successful:"
        f" {len(successful_pdf_extractions)}"
    )

    print(
        f"Remaining queue:           "
        f"{len(crawler.queue)}"
    )

    print(
        f"Inventory:                 "
        f"{'COMPLETE' if not crawler.queue else 'PARTIAL'}"
    )

    print(
        f"Output:                    "
        f"{OUTPUT_FILE.resolve()}"
    )

    print(
        f"Checkpoint:                "
        f"{CHECKPOINT_FILE.resolve()}"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    asyncio.run(
        main()
    )