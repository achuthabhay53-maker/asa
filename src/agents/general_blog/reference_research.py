"""Retrieve user-provided references and extract supported evidence."""

from __future__ import annotations

import logging
import os
from urllib.parse import urlsplit

import anthropic

from src.models.general_blog import (
    GeneralBlogEvidence,
    GeneralBlogRequest,
    UploadedPDFStatus,
)
from src.tools.reference_fetcher import (
    MAX_UPLOADED_PDFS,
    RetrievedSource,
    SourceFetchError,
    fetch_reference,
    fetch_uploaded_pdf,
)
from src.utils.general_blog_research import extract_source_evidence

logger = logging.getLogger(__name__)


class UserReferenceResearchAgent:
    """Retrieve user-supplied papers and references and extract evidence."""

    def __init__(
        self,
        *,
        model: str | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if client is None and not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")
        self.client = client or anthropic.Anthropic(api_key=api_key)
        self.model = model or os.getenv("ANTHROPIC_MODEL_NAME") or "claude-haiku-4-5-20251001"

    def research_references(
        self,
        request: GeneralBlogRequest,
        reference_urls: list[str],
        evidence_offset: int = 0,
        uploaded_pdfs: list[tuple[str, bytes]] | None = None,
    ) -> tuple[list[GeneralBlogEvidence], list[str], list[UploadedPDFStatus]]:
        evidence: list[GeneralBlogEvidence] = []
        warnings: list[str] = []
        pdf_statuses: list[UploadedPDFStatus] = []
        uploaded_pdfs = uploaded_pdfs or []
        if len(uploaded_pdfs) > MAX_UPLOADED_PDFS:
            raise ValueError(f"Upload no more than {MAX_UPLOADED_PDFS} PDFs per run.")

        for url in reference_urls:
            try:
                source = fetch_reference(url)
            except SourceFetchError as exc:
                logger.warning(
                    "User reference retrieval failed (source_host=%s, error_type=%s, reason=%s)",
                    urlsplit(url).hostname,
                    type(exc).__name__,
                    exc,
                )
                warnings.append(f"Reference could not be incorporated: {url} ({exc})")
                continue
            self._extract_evidence(
                source,
                request,
                evidence_offset,
                evidence,
                warnings,
                warning_label=url,
            )
        for file_name, content in uploaded_pdfs:
            try:
                source = fetch_uploaded_pdf(file_name, content)
            except SourceFetchError as exc:
                logger.warning(
                    "Uploaded reference PDF could not be extracted (file_name=%s, error_type=%s)",
                    file_name,
                    type(exc).__name__,
                )
                warnings.append(f"Uploaded PDF could not be incorporated: {file_name} ({exc})")
                pdf_statuses.append(
                    UploadedPDFStatus(
                        file_name=file_name,
                        status="failed",
                        message=f"PDF could not be parsed: {exc}",
                    )
                )
                continue
            evidence_before = len(evidence)
            self._extract_evidence(
                source,
                request,
                evidence_offset,
                evidence,
                warnings,
                warning_label=file_name,
            )
            extracted_count = len(evidence) - evidence_before
            pdf_statuses.append(
                UploadedPDFStatus(
                    file_name=file_name,
                    status=(
                        "parsed_with_evidence"
                        if extracted_count
                        else "parsed_no_evidence"
                    ),
                    extracted_text_characters=len(source.text),
                    evidence_count=extracted_count,
                    message=(
                        f"PDF parsed successfully; {extracted_count} evidence item(s) extracted."
                        if extracted_count
                        else "PDF parsed successfully, but no relevant exact-quote evidence was extracted."
                    ),
                )
            )
        return evidence, warnings, pdf_statuses

    def _extract_evidence(
        self,
        source: RetrievedSource,
        request: GeneralBlogRequest,
        evidence_offset: int,
        evidence: list[GeneralBlogEvidence],
        warnings: list[str],
        *,
        warning_label: str,
    ) -> None:
        extracted = extract_source_evidence(
            self.client,
            self.model,
            source,
            "user_reference",
            evidence_offset + len(evidence),
            request.prompt,
        )
        if not extracted:
            warnings.append(
                "Reference was retrieved but no topic-relevant exact-quote evidence "
                f"was validated: {warning_label}"
            )
        evidence.extend(extracted)
