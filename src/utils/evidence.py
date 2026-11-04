"""Helpers for comparing registered evidence provenance."""
from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def canonicalize_evidence_url(url: str) -> str:
    """Normalize URL components without merging distinct paths or queries."""
    try:
        parsed = urlsplit(url.strip())
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return ""
        if parsed.username or parsed.password:
            return ""
        host = parsed.hostname.lower().rstrip(".")
        if ":" in host:
            host = f"[{host}]"
        port = parsed.port
        if port and not (
            (parsed.scheme.lower() == "http" and port == 80)
            or (parsed.scheme.lower() == "https" and port == 443)
        ):
            host = f"{host}:{port}"
        path = parsed.path.rstrip("/") or "/"
        return urlunsplit(
            (
                parsed.scheme.lower(),
                host,
                path,
                parsed.query,
                "",
            )
        )
    except ValueError:
        return ""
