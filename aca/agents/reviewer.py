"""Agent 6: Review a generated blog and report issues for human review."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import anthropic
from pydantic import ValidationError

from aca.evidence import canonicalize_evidence_url
from aca.models.reviewer import ReviewOutput, ReviewerResponse
from aca.models.writer import (
    WriterEvidence,
    WriterInput,
    WriterOutput,
    WrittenSection,
)
logger = logging.getLogger(__name__)


class ReviewerAgent:
    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 4096,
    ) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def review(self, writer_input: WriterInput, draft: WriterOutput) -> ReviewOutput:
        """Review the draft and return findings without changing the article."""
        payload = {
            "company_name": writer_input.company_name,
            "company_url": writer_input.company_url,
            "search_intent": writer_input.search_intent,
            "article_objective": writer_input.article_objective,
            "target_audience": writer_input.target_audience,
            "primary_keyword": writer_input.primary_keyword,
            "secondary_keywords": writer_input.secondary_keywords,
            "related_topics": writer_input.related_topics,
            "question_based_queries": writer_input.question_based_queries,
            "sections": [
                {
                    "heading": section.heading,
                    "objective": section.objective,
                    "topics": section.topics,
                    "questions_to_answer": section.questions_to_answer,
                    "evidence_ids": section.evidence_ids,
                }
                for section in writer_input.sections
            ],
            "approved_evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "status": item.status,
                    "quote": item.evidence_quote,
                    "source_url": item.source_url,
                    "wording_boundary": item.wording_boundary,
                }
                for item in writer_input.evidence
            ],
            "draft_unresolved_items": draft.unresolved_items,
            "brief_coverage": draft.brief_coverage,
            "draft": {
                "title": draft.title,
                "seo_title": draft.seo_title,
                "meta_description": draft.meta_description,
                "article_markdown": draft.article_markdown,
                "claims_used": [
                    {
                        "claim": claim.claim,
                        "claim_scope": claim.claim_scope,
                        "evidence_ids": claim.evidence_ids,
                        "support_status": claim.support_status,
                    }
                    for claim in draft.claims_used
                ],
            },
        }
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                }
            ],
        )
        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                f"Reviewer response was truncated at {self.max_tokens:,} output tokens. "
                "Reduce the findings to concise issues and try again."
            )
        if response.stop_reason not in {"end_turn", "stop_sequence"}:
            raise RuntimeError(
                "Reviewer did not finish its response " f"(stop reason: {response.stop_reason or 'unknown'})."
            )
        raw_text = "\n".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()
        if not raw_text:
            raise RuntimeError("Reviewer Agent returned an empty response.")
        try:
            result = ReviewerResponse.model_validate(self._parse_json(raw_text))
        except (ValidationError, ValueError) as exc:
            raise RuntimeError(f"Reviewer Agent returned invalid output: {exc}") from exc

        return ReviewOutput(
            summary=result.summary,
            findings=result.findings,
            unresolved_items=result.unresolved_items,
        )

    @staticmethod
    def _normalize_revised_metadata(blog: WriterOutput) -> WriterOutput:
        """Keep the review usable when generated SEO metadata violates limits."""
        updates = {}
        unresolved = list(blog.unresolved_items)
        meta_description = blog.meta_description.strip()
        if not meta_description:
            meta_description = blog.title.strip()
            if len(meta_description) > 155:
                meta_description = meta_description[:155]
                if " " in meta_description:
                    meta_description = meta_description.rsplit(" ", 1)[0]
            updates["meta_description"] = meta_description
            unresolved.append("Reviewer returned an empty meta description; a shortened title was used.")
        elif len(meta_description) > 155:
            shortened = meta_description[:155]
            if " " in shortened:
                shortened = shortened.rsplit(" ", 1)[0]
            updates["meta_description"] = shortened.rstrip(" ,;:-")
            unresolved.append("Reviewer meta description exceeded 155 characters and was shortened.")
        elif meta_description != blog.meta_description:
            updates["meta_description"] = meta_description

        seo_title = blog.seo_title.strip()
        if not seo_title or len(seo_title) > 60:
            title = blog.title.strip()
            seo_title = title[:60]
            if len(title) > 60 and " " in seo_title:
                seo_title = seo_title.rsplit(" ", 1)[0]
            updates["seo_title"] = seo_title
            unresolved.append("Reviewer SEO title was missing or exceeded 60 characters and was normalized.")
        elif seo_title != blog.seo_title:
            updates["seo_title"] = seo_title
        if updates:
            updates["unresolved_items"] = list(dict.fromkeys(unresolved))
            return blog.model_copy(update=updates)
        return blog

    @staticmethod
    def _build_writer_output(
        draft: WriterOutput,
        revision: Any,
    ) -> WriterOutput:
        """Restore the standard writer model and derive sections from revised Markdown."""
        lines = revision.article_markdown.splitlines()
        headings = [
            (index, line.strip()[3:].strip())
            for index, line in enumerate(lines)
            if re.match(r"^##\s+", line.strip())
        ]
        sections = []
        for position, (start, heading) in enumerate(headings):
            end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
            content = "\n".join(lines[start + 1 : end]).strip()
            sections.append(
                WrittenSection(
                    heading=heading,
                    content_markdown=content,
                    evidence_ids_used=[
                        evidence_id
                        for claim in revision.claims_used
                        if claim.claim.casefold() in content.casefold()
                        for evidence_id in claim.evidence_ids
                    ],
                )
            )
        return draft.model_copy(
            update={
                "title": revision.title,
                "seo_title": revision.seo_title,
                "meta_description": revision.meta_description,
                "article_markdown": revision.article_markdown,
                "word_count": len(re.findall(r"\b[\w’'-]+\b", revision.article_markdown)),
                "sections": sections,
                "claims_used": revision.claims_used,
                "unresolved_items": revision.unresolved_items,
                "brief_coverage": revision.brief_coverage,
            }
        )

    @staticmethod
    def _sanitize_revised_blog(
        blog: WriterOutput,
        evidence: list[WriterEvidence],
    ) -> WriterOutput:
        evidence_by_id = {item.evidence_id: item for item in evidence}
        allowed_urls = {
            canonicalize_evidence_url(item.source_url)
            for item in evidence
            if item.source_url and canonicalize_evidence_url(item.source_url)
        }
        unresolved = list(blog.unresolved_items)
        claims = []
        for claim in blog.claims_used:
            unknown_ids = sorted(set(claim.evidence_ids) - evidence_by_id.keys())
            if unknown_ids:
                note = (
                    f"Reviewer referenced unknown evidence IDs for claim "
                    f"{claim.claim!r}; removed: {', '.join(unknown_ids)}."
                )
                unresolved.append(note)
            references = [
                evidence_by_id[evidence_id]
                for evidence_id in dict.fromkeys(claim.evidence_ids)
                if evidence_id in evidence_by_id
            ]
            linked_urls = {
                canonicalize_evidence_url(item.source_url): item.source_url
                for item in references
                if item.source_url and canonicalize_evidence_url(item.source_url)
            }
            bad_claim_urls = [
                url for url in claim.source_urls if canonicalize_evidence_url(url) not in linked_urls
            ]
            if bad_claim_urls:
                unresolved.append(
                    f"Reviewer returned URLs not linked to supplied evidence for "
                    f"claim {claim.claim!r}; removed: {', '.join(bad_claim_urls)}."
                )
            support_status = claim.support_status
            if claim.claim_scope == "company_specific" and not references:
                if support_status in {"supported", "partially_supported"}:
                    support_status = "unsupported"
                    unresolved.append(
                        "Reviewer did not attach approved evidence to this "
                        f"company-specific claim: {claim.claim}"
                    )
                elif support_status == "contradicted":
                    support_status = "not_assessed"
                    unresolved.append(
                        "Reviewer marked this claim contradicted without attaching "
                        f"approved evidence; human verification is required: {claim.claim}"
                    )
            if claim.claim_scope == "company_specific" and support_status in {
                "partially_supported",
                "unsupported",
                "contradicted",
                "inconclusive",
                "not_assessed",
            }:
                status_text = support_status.replace("_", " ")
                unresolved.append(
                    f"Company-specific claim is {status_text} and needs human " f"verification: {claim.claim}"
                )
            claims.append(
                claim.model_copy(
                    update={
                        "evidence_ids": [
                            evidence_id
                            for evidence_id in dict.fromkeys(claim.evidence_ids)
                            if evidence_id in evidence_by_id
                        ],
                        "source_urls": list(dict.fromkeys(linked_urls.values())),
                        "supporting_evidence": references,
                        "support_status": support_status,
                        "citation_status": (
                            "mismatch"
                            if unknown_ids or bad_claim_urls
                            else "linked"
                            if references
                            else "missing"
                        ),
                    }
                )
            )

        sections = []
        for section in blog.sections:
            unknown_ids = sorted(set(section.evidence_ids_used) - evidence_by_id.keys())
            if unknown_ids:
                unresolved.append(
                    f"Reviewer referenced unknown evidence IDs in section "
                    f"{section.heading!r}; removed: {', '.join(unknown_ids)}."
                )
            sections.append(
                section.model_copy(
                    update={
                        "evidence_ids_used": [
                            evidence_id
                            for evidence_id in dict.fromkeys(section.evidence_ids_used)
                            if evidence_id in evidence_by_id
                        ]
                    }
                )
            )

        article = blog.article_markdown

        def remove_unsupported_link(match: re.Match[str]) -> str:
            url = match.group("url")
            if canonicalize_evidence_url(url) in allowed_urls:
                return match.group(0)
            unresolved.append(
                f"Reviewer added a source URL not present in approved evidence; "
                f"the link was removed: {url}"
            )
            return match.group("label")

        article = re.sub(
            r"\[(?P<label>[^\]]+)\]\((?P<url>https?://[^)\s]+)\)",
            remove_unsupported_link,
            article,
        )
        return blog.model_copy(
            update={
                "article_markdown": article,
                "sections": sections,
                "claims_used": claims,
                "unresolved_items": list(dict.fromkeys(unresolved)),
                "word_count": len(re.findall(r"\b[\w’'-]+\b", article)),
            }
        )

    @staticmethod
    def _parse_json(raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        decoder = json.JSONDecoder()
        last_json_error: json.JSONDecodeError | None = None
        last_validation_error: ValidationError | None = None
        search_from = 0

        while (start := text.find("{", search_from)) != -1:
            try:
                payload, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError as exc:
                last_json_error = exc
                search_from = start + 1
                continue

            search_from = start + 1
            if not isinstance(payload, dict):
                continue
            try:
                ReviewerResponse.model_validate(payload)
            except ValidationError as exc:
                last_validation_error = exc
                continue
            trailing_kind = ReviewerAgent._classify_trailing_content(text[end:])
            if trailing_kind:
                logger.warning(
                    "Reviewer response had %s after its first valid JSON object "
                    "(starting at character %d); extra content was ignored.",
                    trailing_kind,
                    end,
                )
            return payload

        if last_validation_error is not None:
            missing_or_invalid_fields = [
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in last_validation_error.errors(include_input=False)[:5]
            ]
            raise RuntimeError(
                "Reviewer Agent response contained JSON, but no object matched "
                "the required review response schema. "
                f"Field errors: {'; '.join(missing_or_invalid_fields)}."
            ) from last_validation_error
        if last_json_error is not None:
            raise RuntimeError(
                "Reviewer Agent response contained no complete JSON object. "
                f"First syntax error at character {last_json_error.pos}: "
                f"{last_json_error.msg} (response length: {len(text)} characters)."
            ) from last_json_error
        raise RuntimeError(
            "Reviewer Agent output contained no JSON object "
            f"(response length: {len(text)} characters)."
        )

    @staticmethod
    def _classify_trailing_content(trailing_text: str) -> str | None:
        trailing = trailing_text.strip()
        if not trailing or re.fullmatch(r"```(?:json)?\s*```", trailing, flags=re.IGNORECASE):
            return None

        decoder = json.JSONDecoder()
        search_from = 0
        while (start := trailing.find("{", search_from)) != -1:
            try:
                payload, _ = decoder.raw_decode(trailing, start)
            except json.JSONDecodeError:
                search_from = start + 1
                continue
            if isinstance(payload, dict):
                try:
                    ReviewerResponse.model_validate(payload)
                except ValidationError:
                    return "another JSON object"
                return "a second valid reviewer JSON object"
            search_from = start + 1
        return "non-JSON trailing text"

    @staticmethod
    def _system_prompt() -> str:
        return """You are Agent 6, the publication reviewer. You only inspect the
Agent 5 draft and report problems for a human to review. Do not rewrite, edit,
correct, or return any part of the article. The human in the loop decides whether
to apply your recommendations.

Review the draft for grammar and spelling errors, unnatural or repetitive writing,
formatting and heading problems, missing brief requirements, weak SEO or keyword
use, and factual claims that are unsupported, misattributed, or contradicted by
the supplied evidence. Assess company-specific claims against the supplied
approved evidence. Do not call a claim false unless evidence directly contradicts
it. Mark uncertainty for human verification where evidence is incomplete. Generic
industry explanations do not need company evidence.

Check for generic openings, unsupported vendor claims, repeated paragraph endings,
awkward headings, and language that conflicts with the writing brief. Treat
Agent 5's unresolved items as leads for review, not as proof that a claim is wrong.
Flag generic openers such as "this guide will explore", inflated wording such as
"leverage" or "cutting-edge", forbidden filler phrases, more than two em dashes,
and paragraphs that restate their own point. Do not silently correct any of them.

Be specific and actionable. Identify the location using a heading or a short
quoted phrase when useful. For each finding, set target_text to a short,
verbatim excerpt from the article that uniquely identifies the issue. The excerpt
must be contiguous plain text as it appears in the article, without Markdown
markers. If no exact unique excerpt identifies the issue, set target_text to null;
do not guess a location. Explain why it needs attention and recommend what a
human should check or change. Do not invent facts, URLs, evidence IDs, or errors.
Do not flag stylistic preferences as errors unless they conflict with the brief.
If you find no material issues, say so in the summary and return empty findings
and unresolved_items arrays.

Return JSON only, matching this schema:
{
  "summary": "One short sentence describing the review result",
  "findings": [
    {
      "area": "grammar|formatting|headings|content|keywords|seo|factual_claims",
      "severity": "critical|high|medium|low",
      "issue": "Specific issue and where it appears",
      "recommendation": "Action or verification for the human reviewer",
      "target_text": "Exact unique excerpt from article, or null"
    }
  ],
  "unresolved_items": ["Material evidence or brief questions a human must resolve"]
}
Return at most 5 findings and 10 unresolved items. Never include a revised article,
replacement wording, or a full article excerpt. Human approval is required."""
