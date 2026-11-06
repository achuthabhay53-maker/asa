"""Shared evidence extraction for General Blog research agents."""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

import anthropic
from pydantic import BaseModel, Field

from src.models.general_blog import EvidenceRelevance, GeneralBlogEvidence
from src.tools.reference_fetcher import MAX_SOURCE_CHARS, RetrievedSource
from src.utils.general_blog import call_structured_agent, matching_quote

logger = logging.getLogger(__name__)


class EvidenceCandidate(BaseModel):
    claim: str
    quote: str
    wording_boundary: str
    relevance: EvidenceRelevance


class SourceAnalysis(BaseModel):
    summary: str
    evidence: list[EvidenceCandidate] = Field(max_length=12)


def extract_source_evidence(
    client: anthropic.Anthropic,
    model: str,
    source: RetrievedSource,
    source_type: str,
    evidence_offset: int,
    prompt: str,
) -> list[GeneralBlogEvidence]:
    data = call_structured_agent(
        client,
        model,
        """Extract concise, topic-relevant factual claims from this source.
Use only the supplied source text. For every claim, provide an exact contiguous
quote copied from it and a wording boundary that preserves the source's scope
and limitations. Keep numerical findings, study populations, methods, and
conditions attached to the claims they qualify. Do not extrapolate or convert
correlation into causation. Exclude claims without a direct source quote.
Classify each claim's relevance to the user's requested blog topic:
- direct: directly supports the main topic (for example, VR training for
  medical device operation);
- related: a useful adjacent example, but not direct evidence for the main
  topic (for example, VR hand-hygiene training in healthcare);
- background: broader context that is useful only when the article discusses
  that context (for example, general 3D production costs).
Do not classify relevance as a measure of truth or scientific quality.
Return a short source summary and the required structured response.""",
        {
            "topic_prompt": prompt,
            "source_type": source_type,
            "source_url": source.url,
            "source_title": source.title,
            "publication_date": source.publication_date,
            "source_text": source.text[:MAX_SOURCE_CHARS],
        },
        SourceAnalysis,
        "submit_source_evidence",
        max_tokens=3000,
    )
    analysis = SourceAnalysis.model_validate(data)
    evidence = []
    rejected_quotes = 0
    for candidate in analysis.evidence:
        if not candidate.claim.strip() or not matching_quote(candidate.quote, source.text):
            rejected_quotes += 1
            continue
        evidence.append(
            GeneralBlogEvidence(
                evidence_id=f"ev_{evidence_offset + len(evidence) + 1:03d}",
                claim=candidate.claim.strip(),
                status="verified",
                relevance=candidate.relevance,
                evidence_quote=candidate.quote.strip(),
                source_url=source.url,
                wording_boundary=candidate.wording_boundary.strip(),
                source_title=source.title,
                publication_date=source.publication_date,
                source_type=source_type,
            )
        )
    if rejected_quotes:
        logger.warning(
            "Discarded source evidence without exact quotes (source_type=%s, "
            "evidence_rejected=%d, source_host=%s)",
            source_type,
            rejected_quotes,
            urlsplit(source.url).hostname if source.url else None,
        )
    return evidence
