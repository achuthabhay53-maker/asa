"""Safe retrieval and text extraction for user-provided General Blog references."""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_SOURCE_CHARS = 24_000
MAX_REFERENCE_BYTES = 15 * 1024 * 1024
MAX_REFERENCE_PAGES = 80
MAX_UPLOADED_PDFS = 5
MAX_REDIRECTS = 5


@dataclass(frozen=True)
class RetrievedSource:
    url: str | None
    title: str | None
    text: str
    publication_date: str | None = None


class SourceFetchError(RuntimeError):
    """A user-provided source could not be safely retrieved or extracted."""


class _ReferenceHTMLParser(HTMLParser):
    _BLOCK_TAGS = {
        "article", "blockquote", "br", "div", "h1", "h2", "h3", "h4",
        "li", "main", "p", "section", "td", "th", "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self.title: list[str] = []
        self.meta: dict[str, str] = {}
        self._ignored = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = {key.lower(): value for key, value in attrs if value}
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored += 1
            return
        if self._ignored:
            return
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            key = (
                attributes.get("name")
                or attributes.get("property")
                or attributes.get("itemprop")
                or ""
            ).lower()
            content = attributes.get("content")
            if key and content:
                self.meta[key] = content.strip()
        if tag in self._BLOCK_TAGS:
            self.text.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"} and self._ignored:
            self._ignored -= 1
            return
        if self._ignored:
            return
        if tag == "title":
            self._in_title = False
        if tag in self._BLOCK_TAGS:
            self.text.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored:
            return
        if self._in_title:
            self.title.append(data)
        else:
            self.text.append(data)


def _validate_public_url(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise SourceFetchError("Only public HTTP and HTTPS source URLs are supported.")
    try:
        parsed.port
    except ValueError as exc:
        raise SourceFetchError("Source URL contains an invalid port.") from exc
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
        raise SourceFetchError("Local and internal network URLs are not allowed.")
    try:
        literal_address = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            addresses = {
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
            }
        except OSError as exc:
            raise SourceFetchError(f"Could not resolve source host {hostname!r}.") from exc
        if not addresses or any(not address.is_global for address in addresses):
            raise SourceFetchError("Source URL must resolve only to public IP addresses.")
    else:
        if not literal_address.is_global:
            raise SourceFetchError("Private and reserved IP addresses are not allowed.")
    return url


def _read_reference_bytes(url: str) -> tuple[str, bytes, str]:
    current_url = url
    with httpx.Client(timeout=30.0, follow_redirects=False) as client:
        for redirect_count in range(MAX_REDIRECTS + 1):
            _validate_public_url(current_url)
            try:
                with client.stream(
                    "GET",
                    current_url,
                    headers={"User-Agent": "ACA-GeneralBlogResearch/1.0"},
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location or redirect_count >= MAX_REDIRECTS:
                            raise SourceFetchError("Source exceeded the allowed redirects.")
                        current_url = urljoin(current_url, location)
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_REFERENCE_BYTES:
                            raise SourceFetchError("Source exceeds the 15 MB download limit.")
                    return str(response.url), bytes(body), content_type
            except httpx.HTTPError as exc:
                raise SourceFetchError(
                    f"Source retrieval failed ({type(exc).__name__})."
                ) from exc
    raise SourceFetchError("Source could not be retrieved.")


def fetch_reference(url: str) -> RetrievedSource:
    final_url, content, content_type = _read_reference_bytes(url)
    if not content:
        raise SourceFetchError("The reference source is empty.")
    if "pdf" in content_type or content.startswith(b"%PDF-"):
        return _extract_pdf_source(content, final_url, None)

    if content_type and not any(
        kind in content_type for kind in ("html", "xml", "text/plain", "text/")
    ):
        raise SourceFetchError(f"Unsupported reference content type: {content_type}.")
    decoded = content.decode("utf-8", errors="replace")
    if "html" in content_type or "<html" in decoded[:1000].lower():
        parser = _ReferenceHTMLParser()
        parser.feed(decoded)
        title = " ".join(" ".join(parser.title).split()) or None
        published = (
            parser.meta.get("article:published_time")
            or parser.meta.get("citation_publication_date")
            or parser.meta.get("date")
        )
        text = " ".join(" ".join(parser.text).split())
        if not text:
            raise SourceFetchError("The reference page contains no extractable text.")
        return RetrievedSource(
            url=final_url,
            title=title,
            text=text[:MAX_SOURCE_CHARS],
            publication_date=published,
        )
    text = decoded[:MAX_SOURCE_CHARS].strip()
    if not text:
        raise SourceFetchError("The reference source contains no extractable text.")
    return RetrievedSource(url=final_url, title=None, text=text)


def fetch_uploaded_pdf(file_name: str, content: bytes) -> RetrievedSource:
    """Extract text from an uploaded PDF without writing it to disk."""
    safe_name = Path(file_name).name.strip()
    if not safe_name or Path(safe_name).suffix.lower() != ".pdf":
        raise SourceFetchError("Uploaded research documents must be PDF files.")
    if not content:
        raise SourceFetchError("The uploaded PDF is empty.")
    if len(content) > MAX_REFERENCE_BYTES:
        raise SourceFetchError("The uploaded PDF exceeds the 15 MB limit.")
    if not content.startswith(b"%PDF-"):
        raise SourceFetchError("The uploaded file is not a valid PDF.")
    return _extract_pdf_source(content, None, safe_name)


def _extract_pdf_source(
    content: bytes,
    source_url: str | None,
    fallback_title: str | None,
) -> RetrievedSource:
    try:
        reader = PdfReader(BytesIO(content), strict=False)
        if reader.is_encrypted:
            raise SourceFetchError("Password-protected PDFs are not supported.")
        page_text = [
            (page.extract_text() or "").strip()
            for page in reader.pages[:MAX_REFERENCE_PAGES]
        ]
    except (PdfReadError, ValueError, OSError) as exc:
        raise SourceFetchError("The PDF could not be parsed.") from exc
    text = "\n".join(value for value in page_text if value)
    if not text:
        raise SourceFetchError(
            "The PDF contains no extractable text; scanned PDFs require OCR and are not supported."
        )
    metadata = reader.metadata
    title = str(metadata.title).strip() if metadata and metadata.title else fallback_title
    date_value = str(metadata.creation_date) if metadata and metadata.creation_date else None
    if date_value:
        date_match = re.search(r"\d{4}", date_value)
        date_value = date_match.group(0) if date_match else None
    return RetrievedSource(
        url=source_url,
        title=title,
        text=text[:MAX_SOURCE_CHARS],
        publication_date=date_value,
    )
