"""Agent 6: Review and revise a generated blog before human publication review."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import anthropic
from pydantic import ValidationError

from aca.evidence import canonicalize_evidence_url
from aca.models.reviewer import ReviewOutput, ReviewerResponse, RevisedBlog
from aca.models.writer import (
    WriterEvidence,
    WriterInput,
    WriterOutput,
    WrittenSection,
)
from aca.agents.writer import WriterAgent


class ReviewerAgent:
    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 16384,
    ) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def review(self, writer_input: WriterInput, draft: WriterOutput) -> ReviewOutput:
        """Review the draft, revise it, and preserve evidence-aware review findings."""
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
                "Shorten the draft or reduce its target word count and try again."
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

        revision = RevisedBlog(
            title=result.title,
            seo_title=result.seo_title,
            meta_description=result.meta_description,
            article_markdown=result.article_markdown,
            claims_used=draft.claims_used,
            unresolved_items=list(dict.fromkeys([*draft.unresolved_items, *result.unresolved_items])),
            brief_coverage=draft.brief_coverage,
        )
        revised = self._build_writer_output(draft, revision)
        revised = self._sanitize_revised_blog(revised, writer_input.evidence)
        revised = self._normalize_revised_metadata(revised)
        try:
            WriterAgent._validate_seo_requirements(revised, writer_input)
        except RuntimeError as exc:
            raise RuntimeError(f"Reviewer Agent returned an invalid blog: {exc}") from exc
        return ReviewOutput(
            summary=result.summary,
            findings=result.findings,
            revised_blog=revised,
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
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Reviewer Agent returned invalid JSON at character {exc.pos}: {exc.msg}."
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Reviewer Agent output must be a JSON object.")
        return payload

    @staticmethod
    def _system_prompt() -> str:
        return """You are Agent 6, a light-touch publication reviewer. Check the draft for
major formatting, headings, completeness, clarity, keyword, SEO, and company-fact
issues. Make only necessary improvements. Return the complete final blog in
article_markdown, even when no changes are needed.

Use only the supplied draft and evidence. Do not invent facts, URLs, evidence IDs,
metrics, capabilities, or certifications. Focus evidence checks on the named company,
its products, services, capabilities, processes, and results; generic industry or
educational statements do not need company evidence. Preserve uncertain claims and
report them for human verification instead of asserting they are false. Do not add
source URLs unless present in approved evidence. Preserve article order: introduction,
main content, conclusion, FAQs; retain 3–5 FAQs and do not add content without need.
Keep SEO title within 60 characters and meta description within 155 characters.

Return JSON only:
{
  "summary": "One short sentence",
  "findings": [
    {
      "area": "formatting|headings|content|keywords|seo|factual_claims",
      "severity": "critical|high|medium|low",
      "issue": "Short issue",
      "recommendation": "Short action or verification note"
    }
  ],
  "title": "Final H1 title",
  "seo_title": "Final SEO title",
  "meta_description": "Final meta description",
  "article_markdown": "The complete final Markdown blog",
  "unresolved_items": ["Only material items needing human verification"]
}
Return at most 5 findings and 10 unresolved items. Do not repeat the article anywhere
else in the response. Human approval is still required."""
