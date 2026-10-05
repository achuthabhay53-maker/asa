from __future__ import annotations

import argparse
import hashlib
import io
import re
from collections import defaultdict
from typing import Any
from urllib.parse import (
    parse_qsl,
    unquote,
    urlencode,
    urldefrag,
    urljoin,
    urlparse,
    urlunparse,
)

import scrapy
from scrapy import signals
from scrapy.crawler import CrawlerProcess
import trafilatura
from pypdf import PdfReader
from scrapy.http import Response
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import Spider


class WebsiteEvidenceSpider(Spider):
    """
    Scrapy v2 evidence crawler for ACA.

    Purpose:
        Build a website evidence inventory that can later be consumed
        by the ACA Source Analyzer.

    This is intentionally independent from the Source Analyzer.
    """

    name = "website_evidence"
    allowed_domains: list[str] = []
    start_urls: list[str] = []

    custom_settings = {
        # ---------------------------------------------------------
        # Crawl limits
        # ---------------------------------------------------------
        "CLOSESPIDER_PAGECOUNT": 10000,

        # ---------------------------------------------------------
        # Concurrency
        # ---------------------------------------------------------
        "CONCURRENT_REQUESTS": 16,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 16,

        # Small delay to avoid hammering the site.
        "DOWNLOAD_DELAY": 0.15,

        # ---------------------------------------------------------
        # Robots
        # ---------------------------------------------------------
        "ROBOTSTXT_OBEY": True,

        # ---------------------------------------------------------
        # Retry
        # ---------------------------------------------------------
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 3,
        "RETRY_HTTP_CODES": [
            408,
            429,
            500,
            502,
            503,
            504,
        ],

        # ---------------------------------------------------------
        # Auto throttling
        # ---------------------------------------------------------
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 0.15,
        "AUTOTHROTTLE_MIN_DELAY": 0.10,
        "AUTOTHROTTLE_MAX_DELAY": 5.0,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 8.0,

        # ---------------------------------------------------------
        # Request headers
        # ---------------------------------------------------------
        "DEFAULT_REQUEST_HEADERS": {
            "User-Agent": "ACA-Website-Evidence-Crawler/2.0",
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,"
                "application/pdf;q=0.8,"
                "*/*;q=0.5"
            ),
        },

        # ---------------------------------------------------------
        # Output
        # ---------------------------------------------------------
        "FEED_EXPORT_ENCODING": "utf-8",

        # ---------------------------------------------------------
        # Logging
        # ---------------------------------------------------------
        "LOG_LEVEL": "INFO",
    }

    # Tracking/query parameters that should not create separate
    # evidence URLs.
    TRACKING_PARAMS = {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_name",
        "gclid",
        "fbclid",
        "msclkid",
        "dclid",
        "mc_cid",
        "mc_eid",
        "_ga",
        "_gl",
    }

    # URL patterns that frequently represent archive/taxonomy
    # duplication. We still allow them, but this classification
    # lets ACA distinguish them from substantive pages.
    ARCHIVE_PATTERNS = (
        "/tag/",
        "/category/",
        "/author/",
        "/journal-name/",
        "/page/",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        start_url = kwargs.pop("start_url", None) or kwargs.pop("url", None)
        if not isinstance(start_url, str) or not start_url.strip():
            raise ValueError(
                "Provide a website URL with -a start_url=https://example.com"
            )

        parsed_start = urlparse(start_url.strip())
        if (
            parsed_start.scheme.lower() not in {"http", "https"}
            or not parsed_start.hostname
            or parsed_start.username
            or parsed_start.password
        ):
            raise ValueError(
                "start_url must be an absolute HTTP or HTTPS URL without credentials."
            )

        super().__init__(*args, **kwargs)

        self.site_domain = parsed_start.hostname.lower().rstrip(".")
        if self.site_domain.startswith("www."):
            self.site_domain = self.site_domain[4:]
        self.site_origin = f"{parsed_start.scheme.lower()}://{self.site_domain}"
        self.start_url = self.normalize_url(start_url.strip())
        if not self.start_url:
            raise ValueError("Could not normalize the supplied start_url.")

        self.robots_url = urljoin(self.site_origin + "/", "robots.txt")
        self.common_sitemap_urls = [
            urljoin(self.site_origin + "/", "sitemap.xml"),
            urljoin(self.site_origin + "/", "wp-sitemap.xml"),
        ]
        self.allowed_domains = [self.site_domain]
        self.start_urls = [self.start_url]

        self.seen_content_hashes: dict[str, str] = {}
        self.url_aliases: dict[str, str] = {}

        self.stats = defaultdict(int)
        self.sitemap_diagnostics = defaultdict(int)
        self.robots_sitemap_declarations: list[str] = []
        self.unique_sitemap_urls_queued: list[str] = []
        self.sitemap_page_urls: set[str] = set()

        self.link_extractor = LinkExtractor(
            allow_domains=self.allowed_domains,
            unique=True,
            deny_extensions=(),
        )

    @classmethod
    def from_crawler(cls, crawler, *args: Any, **kwargs: Any):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(
            spider._on_request_scheduled,
            signal=signals.request_scheduled,
        )
        crawler.signals.connect(
            spider._on_request_dropped,
            signal=signals.request_dropped,
        )
        return spider

    def _on_request_scheduled(self, request, spider) -> None:
        if spider is not self:
            return
        kind = request.meta.get("sitemap_diagnostic_kind")
        if kind:
            self.sitemap_diagnostics[f"{kind}_schedule_attempts"] += 1

    def _on_request_dropped(self, request, spider) -> None:
        if spider is not self:
            return
        kind = request.meta.get("sitemap_diagnostic_kind")
        if kind:
            self.sitemap_diagnostics[f"{kind}_duplicate_drops"] += 1

    # ============================================================
    # URL NORMALIZATION
    # ============================================================

    def normalize_url(self, url: str, base_url: str | None = None) -> str | None:
        if not url:
            return None

        if base_url:
            url = urljoin(base_url, url)

        url, _fragment = urldefrag(url)

        parsed = urlparse(url)

        if parsed.scheme.lower() not in {"http", "https"}:
            return None

        host = (parsed.hostname or "").lower().rstrip(".")
        if not host:
            return None
        if host.startswith("www."):
            host = host[4:]

        try:
            port = parsed.port
        except ValueError:
            return None
        if port is not None and not (
            (parsed.scheme.lower() == "http" and port == 80)
            or (parsed.scheme.lower() == "https" and port == 443)
        ):
            host = f"{host}:{port}"

        # Normalize path.
        path = parsed.path or "/"
        decoded_path = unquote(path)

        if path != "/" and not decoded_path.strip():
            return None

        # Remove duplicate slashes.
        path = re.sub(r"/{2,}", "/", path)

        # Remove ;jsessionid style suffixes.
        path = re.sub(
            r";jsessionid=[^/?#]+",
            "",
            path,
            flags=re.IGNORECASE,
        )

        # Normalize query parameters.
        query_pairs = []

        for key, value in parse_qsl(
            parsed.query,
            keep_blank_values=True,
        ):
            key_lower = key.lower()

            # Generic UTM support, including misspelled or future UTM
            # variants such as utm_souce.
            if key_lower.startswith("utm_"):
                continue

            if key_lower in self.TRACKING_PARAMS:
                continue

            # Remove common session parameters.
            if key_lower in {
                "sessionid",
                "session_id",
                "sid",
                "phpsessid",
                "jsessionid",
            }:
                continue

            query_pairs.append((key, value))

        query = urlencode(query_pairs, doseq=True)

        return urlunparse(
            (
                parsed.scheme.lower(),
                host,
                path,
                "",
                query,
                "",
            )
        )

    def is_allowed_url(self, url: str) -> bool:
        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"}:
            return False

        host = (parsed.hostname or "").lower().rstrip(".")
        if host.startswith("www."):
            host = host[4:]

        return host == self.site_domain or host.endswith(
            f".{self.site_domain}"
        )

    # ============================================================
    # START
    # ============================================================

    async def start(self):
        yield scrapy.Request(
            self.robots_url,
            callback=self.parse_robots,
            errback=self.handle_error,
            priority=100,
        )

        yield scrapy.Request(
            self.start_url,
            callback=self.parse,
            errback=self.handle_error,
        )

    def start_requests(self):
        yield scrapy.Request(
            self.robots_url,
            callback=self.parse_robots,
            errback=self.handle_error,
            priority=100,
        )

        yield scrapy.Request(
            self.start_url,
            callback=self.parse,
            errback=self.handle_error,
        )

    # ============================================================
    # ROBOTS / SITEMAP
    # ============================================================

    def parse_robots(self, response: Response):
        self.sitemap_diagnostics["robots_callbacks"] += 1
        self.sitemap_diagnostics["robots_status"] = response.status
        self.logger.info(
            "ROBOTS RESPONSE: status=%s url=%s",
            response.status,
            response.url,
        )

        sitemap_urls = []

        for line in response.text.splitlines():
            line = line.strip()

            if line.lower().startswith("sitemap:"):
                sitemap_url = line.split(":", 1)[1].strip()
                self.robots_sitemap_declarations.append(sitemap_url)
                self.sitemap_diagnostics[
                    "robots_sitemap_declarations"
                ] += 1

                normalized = self.normalize_url(sitemap_url)

                if normalized:
                    sitemap_urls.append(normalized)
                else:
                    self.sitemap_diagnostics[
                        "robots_sitemap_urls_rejected_normalization"
                    ] += 1
                    self.logger.warning(
                        "ROBOTS SITEMAP REJECTED: raw=%s",
                        sitemap_url,
                    )

        # Always try the common sitemap locations as well.
        self.common_sitemap_urls = [
            urljoin(self.site_origin + "/", "sitemap.xml"),
            urljoin(self.site_origin + "/", "wp-sitemap.xml"),
        ]
        self.sitemap_diagnostics["common_sitemap_urls"] = len(
            self.common_sitemap_urls
        )
        sitemap_urls.extend(self.common_sitemap_urls)

        seen = set()

        for sitemap_url in sitemap_urls:
            if sitemap_url in seen:
                continue

            seen.add(sitemap_url)
            self.unique_sitemap_urls_queued.append(sitemap_url)
            self.sitemap_diagnostics["sitemap_requests_yielded"] += 1

            yield scrapy.Request(
                sitemap_url,
                callback=self.parse_sitemap,
                errback=self.handle_error,
                priority=90,
                meta={"sitemap_diagnostic_kind": "robots_sitemap"},
            )

    def parse_sitemap(self, response: Response):
        """
        Parse both sitemap indexes and regular URL sitemaps.
        """

        self.sitemap_diagnostics["sitemap_responses"] += 1
        content_type = (
            response.headers.get(
                "Content-Type",
                b"",
            )
            .decode(
                "latin1",
                errors="ignore",
            )
            .lower()
        )

        if response.status != 200:
            self.sitemap_diagnostics[
                "sitemap_non_200_responses"
            ] += 1
            self.logger.warning(
                "Sitemap returned %s: %s",
                response.status,
                response.url,
            )
            return

        text = response.text

        # Sitemap index.
        if "<sitemapindex" in text.lower():
            self.sitemap_diagnostics["sitemap_indexes"] += 1
            urls = re.findall(
                r"<loc>\s*(.*?)\s*</loc>",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )

            for sitemap_url in urls:
                normalized = self.normalize_url(
                    sitemap_url.strip(),
                    response.url,
                )

                if not normalized:
                    self.sitemap_diagnostics[
                        "sitemap_index_urls_rejected_normalization"
                    ] += 1
                    continue

                self.sitemap_diagnostics[
                    "sitemap_index_requests_yielded"
                ] += 1

                yield scrapy.Request(
                    normalized,
                    callback=self.parse_sitemap,
                    errback=self.handle_error,
                    priority=80,
                    meta={"sitemap_diagnostic_kind": "sitemap_index"},
                )

            return

        # Regular sitemap.
        if "<urlset" in text.lower():
            self.sitemap_diagnostics["sitemap_urlsets"] += 1
            urls = re.findall(
                r"<loc>\s*(.*?)\s*</loc>",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )

            for page_url in urls:
                self.sitemap_diagnostics["sitemap_loc_values"] += 1
                normalized = self.normalize_url(
                    page_url.strip(),
                    response.url,
                )

                if not normalized:
                    self.sitemap_diagnostics[
                        "sitemap_page_urls_rejected_normalization"
                    ] += 1
                    continue

                if not self.is_allowed_url(normalized):
                    self.sitemap_diagnostics[
                        "sitemap_page_urls_rejected_domain"
                    ] += 1
                    self.logger.info(
                        "SITEMAP URL REJECTED BY ALLOW-LIST: %s",
                        normalized,
                    )
                    continue

                self.sitemap_page_urls.add(normalized)
                self.sitemap_diagnostics[
                    "sitemap_page_urls_accepted"
                ] += 1
                self.sitemap_diagnostics[
                    "sitemap_page_requests_yielded"
                ] += 1

                yield scrapy.Request(
                    normalized,
                    callback=self.parse,
                    errback=self.handle_error,
                    priority=10,
                    meta={"sitemap_diagnostic_kind": "sitemap_page"},
                )

            return

        self.logger.warning(
            "Unrecognized sitemap format: %s (%s)",
            response.url,
            content_type,
        )

    # ============================================================
    # MAIN PAGE PARSER
    # ============================================================

    def parse(self, response: Response):
        normalized_url = self.normalize_url(response.url)

        if not normalized_url:
            return

        self.stats["responses_seen"] += 1

        content_type = (
            response.headers.get(
                "Content-Type",
                b"",
            )
            .decode(
                "latin1",
                errors="ignore",
            )
            .lower()
        )

        # --------------------------------------------------------
        # PDF
        # --------------------------------------------------------

        if (
            "application/pdf" in content_type
            or response.url.lower().endswith(".pdf")
        ):
            item = self.extract_pdf(response, normalized_url)

            if item:
                yield item

            return

        # --------------------------------------------------------
        # Only process HTML/XML-like responses.
        # --------------------------------------------------------

        if (
            "text/html" not in content_type
            and "application/xhtml" not in content_type
            and "application/xml" not in content_type
        ):
            self.stats["non_html_skipped"] += 1
            return

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------

        title = self.clean_text(
            response.css("title::text").get()
        )

        description = self.clean_text(
            response.css(
                'meta[name="description"]::attr(content)'
            ).get()
        )

        canonical_url = self.normalize_url(
            response.css(
                'link[rel="canonical"]::attr(href)'
            ).get(),
            response.url,
        )

        if not canonical_url:
            canonical_url = self.normalize_url(
                response.css(
                    'meta[property="og:url"]::attr(content)'
                ).get(),
                response.url,
            )

        # --------------------------------------------------------
        # Text extraction
        # --------------------------------------------------------

        raw_html = response.text

        cleaned_text = self.extract_clean_text(
            raw_html,
            response.url,
        )

        if not cleaned_text:
            cleaned_text = self.extract_body_text(response)

        text_length = len(cleaned_text)

        # --------------------------------------------------------
        # Content hash
        # --------------------------------------------------------

        content_hash = hashlib.sha256(
            cleaned_text.encode(
                "utf-8",
                errors="ignore",
            )
        ).hexdigest()

        duplicate_of = self.seen_content_hashes.get(
            content_hash
        )

        if duplicate_of is None:
            self.seen_content_hashes[
                content_hash
            ] = normalized_url

        # --------------------------------------------------------
        # Page classification
        # --------------------------------------------------------

        page_type = self.classify_page(
            normalized_url,
            title,
            cleaned_text,
        )

        # --------------------------------------------------------
        # Redirect information
        # --------------------------------------------------------

        redirect_chain = response.meta.get(
            "redirect_urls",
            [],
        )

        final_url = self.normalize_url(response.url)

        # --------------------------------------------------------
        # Evidence record
        # --------------------------------------------------------

        item = {
            "record_type": "html",

            "requested_url": (
                response.request.url
            ),

            "final_url": final_url,

            "canonical_url": canonical_url,

            "redirected": bool(redirect_chain),

            "redirect_chain": [
                self.normalize_url(url)
                for url in redirect_chain
                if self.normalize_url(url)
            ],

            "http_status": response.status,

            "content_type": content_type,

            "title": title,

            "meta_description": description,

            "page_type": page_type,

            "text_length": text_length,

            "content_hash": content_hash,

            "duplicate_of": duplicate_of,

            "text": cleaned_text,

            "crawl_source": "scrapy",

        }

        self.stats["html_items"] += 1

        if duplicate_of:
            self.stats["duplicates"] += 1

        if text_length < 500:
            self.stats["thin_pages"] += 1

        yield item

        # --------------------------------------------------------
        # Discover links
        # --------------------------------------------------------

        for link in self.link_extractor.extract_links(
            response
        ):
            normalized = self.normalize_url(
                link.url,
                response.url,
            )

            if not normalized:
                continue

            if not self.is_allowed_url(normalized):
                continue

            yield scrapy.Request(
                normalized,
                callback=self.parse,
                errback=self.handle_error,
                priority=0,
            )

    # ============================================================
    # PDF EXTRACTION
    # ============================================================

    def extract_pdf(
        self,
        response: Response,
        normalized_url: str,
    ):
        self.stats["pdfs_seen"] += 1

        try:
            reader = PdfReader(
                io.BytesIO(response.body),
                strict=False,
            )

            pages = []

            for page_number, page in enumerate(
                reader.pages,
                start=1,
            ):
                try:
                    text = page.extract_text() or ""
                except Exception as exc:
                    self.logger.warning(
                        "PDF page extraction failed: %s page=%s error=%s",
                        normalized_url,
                        page_number,
                        exc,
                    )
                    text = ""

                text = self.clean_text(text)

                if text:
                    pages.append(
                        f"[Page {page_number}]\n{text}"
                    )

            full_text = "\n\n".join(pages)

            if full_text:
                extraction_status = "success"
            else:
                extraction_status = "no_text"

            content_hash = hashlib.sha256(
                full_text.encode(
                    "utf-8",
                    errors="ignore",
                )
            ).hexdigest()

            duplicate_of = self.seen_content_hashes.get(
                content_hash
            )

            if duplicate_of is None:
                self.seen_content_hashes[
                    content_hash
                ] = normalized_url

            self.stats["pdf_items"] += 1

            if extraction_status == "success":
                self.stats[
                    "pdf_extractions_successful"
                ] += 1

            return {
                "record_type": "pdf",

                "requested_url": response.request.url,

                "final_url": normalized_url,

                "canonical_url": None,

                "redirected": bool(
                    response.meta.get(
                        "redirect_urls",
                        [],
                    )
                ),

                "redirect_chain": [
                    self.normalize_url(url)
                    for url in response.meta.get(
                        "redirect_urls",
                        [],
                    )
                    if self.normalize_url(url)
                ],

                "http_status": response.status,

                "content_type": (
                    response.headers.get(
                        "Content-Type",
                        b"",
                    )
                    .decode(
                        "latin1",
                        errors="ignore",
                    )
                    .lower()
                ),

                "title": normalized_url.rsplit(
                    "/",
                    1,
                )[-1],

                "meta_description": None,

                "page_type": "pdf",

                "text_length": len(full_text),

                "content_hash": content_hash,

                "duplicate_of": duplicate_of,

                "text": full_text,

                "pdf_page_count": len(
                    reader.pages
                ),

                "pdf_extraction_status": extraction_status,

                "crawl_source": "scrapy",

            }

        except Exception as exc:
            self.stats["pdf_extraction_failed"] += 1

            self.logger.error(
                "PDF extraction failed: %s: %s",
                normalized_url,
                exc,
            )

            return {
                "record_type": "pdf",

                "requested_url": response.request.url,

                "final_url": normalized_url,

                "canonical_url": None,

                "redirected": bool(
                    response.meta.get(
                        "redirect_urls",
                        [],
                    )
                ),

                "redirect_chain": [],

                "http_status": response.status,

                "content_type": (
                    response.headers.get(
                        "Content-Type",
                        b"",
                    )
                    .decode(
                        "latin1",
                        errors="ignore",
                    )
                    .lower()
                ),

                "title": normalized_url.rsplit(
                    "/",
                    1,
                )[-1],

                "meta_description": None,

                "page_type": "pdf",

                "text_length": 0,

                "content_hash": None,

                "duplicate_of": None,

                "text": "",

                "pdf_page_count": None,

                "pdf_extraction_status": (
                    "extraction_failed"
                ),

                "pdf_extraction_error": str(exc),

                "crawl_source": "scrapy",
            }

    # ============================================================
    # TEXT EXTRACTION
    # ============================================================

    @staticmethod
    def extract_clean_text(
        html: str,
        url: str,
    ) -> str:
        try:
            text = trafilatura.extract(
                html,
                include_links=True,
                include_tables=True,
                include_images=False,
                favor_precision=False,
                favor_recall=True,
                output_format="txt",
            )

            if text:
                return text.strip()

        except Exception:
            pass

        return ""

    @staticmethod
    def extract_body_text(
        response: Response,
    ) -> str:
        parts = response.xpath(
            "//body//text()[normalize-space()]"
        ).getall()

        return WebsiteEvidenceSpider.clean_text(
            " ".join(parts)
        )

    @staticmethod
    def clean_text(value: str | None) -> str:
        if not value:
            return ""

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip()

    # ============================================================
    # PAGE CLASSIFICATION
    # ============================================================

    def classify_page(
        self,
        url: str,
        title: str,
        text: str,
    ) -> str:
        parsed = urlparse(url)

        path = parsed.path.lower()

        if path.endswith(".pdf"):
            return "pdf"

        if any(
            pattern in path
            for pattern in self.ARCHIVE_PATTERNS
        ):
            return "archive_or_taxonomy"

        if "/publication/" in path:
            return "publication"

        if "/resources/" in path:
            return "resource"

        if "/event/" in path or "/events/" in path:
            return "event"

        if "/company/" in path:
            return "company"

        if "/products" in path:
            return "product"

        if "/support" in path:
            return "support"

        if "/about" in path:
            return "about"

        if "/blog" in path:
            return "article"

        if (
            "login" in path
            or "/account/" in path
            or "/auth" in path
        ):
            return "authentication"

        if len(path.strip("/").split("/")) <= 1:
            return "site_page"

        return "page"

    # ============================================================
    # ERROR HANDLING
    # ============================================================

    def handle_error(self, failure):
        request = failure.request

        self.stats["request_errors"] += 1
        diagnostic_kind = request.meta.get("sitemap_diagnostic_kind")
        if diagnostic_kind:
            self.sitemap_diagnostics[
                f"{diagnostic_kind}_request_errors"
            ] += 1
        if request.url.rstrip("/") == self.robots_url.rstrip("/"):
            self.sitemap_diagnostics["robots_request_errors"] += 1

        self.logger.warning(
            "Request failed: %s | %s",
            request.url,
            failure.value,
        )

    # ============================================================
    # CLOSE SUMMARY
    # ============================================================

    def closed(self, reason):
        self.logger.info(
            "=================================================="
        )


        self.logger.info(
            "ACA Scrapy Evidence Crawler finished"
        )

        self.logger.info(
            "Finish reason: %s",
            reason,
        )

        self.logger.info(
            "HTML items: %s",
            self.stats["html_items"],
        )

        self.logger.info(
            "PDFs seen: %s",
            self.stats["pdfs_seen"],
        )

        self.logger.info(
            "PDFs extracted: %s",
            self.stats[
                "pdf_extractions_successful"
            ],
        )

        self.logger.info(
            "Duplicates: %s",
            self.stats["duplicates"],
        )

        self.logger.info(
            "Thin pages: %s",
            self.stats["thin_pages"],
        )

        self.logger.info(
            "Request errors: %s",
            self.stats["request_errors"],
        )

        self.logger.info("SITEMAP DISCOVERY DIAGNOSTICS")
        self.logger.info(
            "Robots callbacks/status: %s / %s",
            self.sitemap_diagnostics["robots_callbacks"],
            self.sitemap_diagnostics.get("robots_status", "not received"),
        )
        self.logger.info(
            "Robots request errors: %s",
            self.sitemap_diagnostics["robots_request_errors"],
        )
        self.logger.info(
            "Robots sitemap declarations: %s",
            self.robots_sitemap_declarations,
        )
        self.logger.info(
            "Common sitemap URLs: %s",
            self.common_sitemap_urls,
        )
        self.logger.info(
            "Unique sitemap URLs queued: %s",
            self.unique_sitemap_urls_queued,
        )
        self.logger.info(
            "Sitemap responses / non-200: %s / %s",
            self.sitemap_diagnostics["sitemap_responses"],
            self.sitemap_diagnostics["sitemap_non_200_responses"],
        )
        self.logger.info(
            "Sitemap indexes / URL sets: %s / %s",
            self.sitemap_diagnostics["sitemap_indexes"],
            self.sitemap_diagnostics["sitemap_urlsets"],
        )
        self.logger.info(
            "Sitemap page URLs extracted / unique accepted: %s / %s",
            self.sitemap_diagnostics["sitemap_loc_values"],
            len(self.sitemap_page_urls),
        )
        self.logger.info(
            "Sitemap page URL rejections (normalization/domain): %s / %s",
            self.sitemap_diagnostics[
                "sitemap_page_urls_rejected_normalization"
            ],
            self.sitemap_diagnostics[
                "sitemap_page_urls_rejected_domain"
            ],
        )

        yielded_keys = {
            "robots_sitemap": "sitemap_requests_yielded",
            "sitemap_index": "sitemap_index_requests_yielded",
            "sitemap_page": "sitemap_page_requests_yielded",
        }
        for kind, yielded_key in yielded_keys.items():
            schedule_attempts = self.sitemap_diagnostics[
                f"{kind}_schedule_attempts"
            ]
            duplicate_drops = self.sitemap_diagnostics[
                f"{kind}_duplicate_drops"
            ]
            scheduler_accepted = max(
                0,
                schedule_attempts - duplicate_drops,
            )
            self.logger.info(
                "%s requests yielded/schedule attempts/accepted/duplicate-dropped/errors: %s/%s/%s/%s/%s",
                kind,
                self.sitemap_diagnostics[yielded_key],
                schedule_attempts,
                scheduler_accepted,
                duplicate_drops,
                self.sitemap_diagnostics[f"{kind}_request_errors"],
            )

        self.logger.info(
            "=================================================="
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl a website and export its evidence inventory."
    )
    parser.add_argument(
        "--url",
        help="Website URL to crawl. If omitted, the spider prompts for it.",
    )
    parser.add_argument(
        "--output",
        default="website_evidence.json",
        help="JSON output path (default: website_evidence.json).",
    )
    args = parser.parse_args()

    start_url = (args.url or input("Website URL: ")).strip()
    if not start_url:
        parser.error("A website URL is required.")

    process = CrawlerProcess(
        settings={
            "FEEDS": {
                args.output: {
                    "format": "json",
                    "encoding": "utf-8",
                    "overwrite": True,
                }
            }
        }
    )
    process.crawl(
        WebsiteEvidenceSpider,
        start_url=start_url,
    )
    process.start()


if __name__ == "__main__":
    main()