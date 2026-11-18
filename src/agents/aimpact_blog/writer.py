"""ACA Agent 5: Writer Agent.

Consumes a validated writer input and produces an evidence-grounded blog
for Agent 6.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import quote, urlsplit

import anthropic
from pydantic import ValidationError

from src.models.writer import WriterEvidence, WriterInput, WriterOutput, WrittenSection
from src.observability.token_usage import record_claude_response
from src.utils.evidence import canonicalize_evidence_url
from src.utils.reviewer_lint import lint_article
from src.utils.structured_output import model_input_schema


_WRITER_OUTPUT_SCHEMA = model_input_schema(WriterOutput)
_WRITER_OUTPUT_SCHEMA["properties"].pop("sections")
_WRITER_OUTPUT_SCHEMA["required"].remove("sections")
_claim_schema = _WRITER_OUTPUT_SCHEMA["properties"]["claims_used"]["items"]
for _field in ("source_urls", "supporting_evidence", "validation_notes"):
    _claim_schema["properties"].pop(_field, None)


# Anthropic rejects "minimum"; Pydantic still validates the parsed response.
def _strip_unsupported_json_schema_constraints(schema: Any) -> None:
    if isinstance(schema, dict):
        schema.pop("minimum", None)
        for value in schema.values():
            _strip_unsupported_json_schema_constraints(value)
    elif isinstance(schema, list):
        for value in schema:
            _strip_unsupported_json_schema_constraints(value)


_strip_unsupported_json_schema_constraints(_WRITER_OUTPUT_SCHEMA)

_WRITER_PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "metadata": {
            "type": "object",
            "properties": {
                "title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "seo_title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "meta_description": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["title", "seo_title", "meta_description"],
            "additionalProperties": False,
        },
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["old_text", "new_text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["metadata", "edits"],
    "additionalProperties": False,
}


class WriterAgent:
    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        max_tokens: int = 11000,
    ) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def write(self, writer_input: WriterInput) -> WriterOutput:
        """Generate a reader-focused, evidence-grounded blog post."""
        user_prompt = self._build_user_prompt(writer_input)
        result = self._generate_output(
            user_prompt,
            system_prompt=self._system_prompt(),
        )

        result = self._sanitize_evidence_references(result, writer_input.evidence)
        result = self._normalize_metadata(result)
        seo_repaired = False
        try:
            self._validate_seo_requirements(result, writer_input)
        except RuntimeError as exc:
            error = str(exc)
            style_issues = lint_article(
                result.article_markdown,
                title=result.title,
                seo_title=result.seo_title,
                meta_description=result.meta_description,
            )
            if style_issues:
                error += "\nStyle issues to fix in the same repair:\n" + "\n".join(
                    f"- {issue}" for issue in style_issues
                )

            repair_prompt = self._build_repair_prompt(result, [error])
            result = self._generate_targeted_update(result, repair_prompt)
            result = self._sanitize_evidence_references(result, writer_input.evidence)
            result = self._normalize_metadata(result)
            self._validate_seo_requirements(result, writer_input)
            seo_repaired = True

        # Style lint runs on the model's prose before code adds reference metadata.
        result, style_notes = self._enforce_style(
            result,
            writer_input,
            allow_retry=not seo_repaired,
        )

        result = self._attach_supporting_evidence(result, writer_input.evidence)
        result = self._flag_unsupported_numeric_claims(result, writer_input.evidence)
        result = self._audit_claims_and_faqs(result)
        if style_notes:
            result = result.model_copy(
                update={"unresolved_items": list(dict.fromkeys([*result.unresolved_items, *style_notes]))}
            )
        result = self._append_references(result, writer_input.evidence)
        return result

    def revise(
        self,
        writer_input: WriterInput,
        draft: WriterOutput,
        instructions: str,
    ) -> WriterOutput:
        """Create a proposed revision from human instructions without changing the saved draft."""
        instructions = instructions.strip()
        if not instructions:
            raise ValueError("Revision instructions must not be empty.")

        payload = {
            "revision_context": {
                "company_name": writer_input.company_name,
                "title": writer_input.title,
                "article_objective": writer_input.article_objective,
                "target_audience": writer_input.target_audience,
                "primary_keyword": writer_input.primary_keyword,
                "secondary_keywords": writer_input.secondary_keywords,
                "unsupported_claims_to_avoid": writer_input.unsupported_claims_to_avoid,
                "evidence": [
                    item.model_dump(mode="json", exclude_none=True)
                    for item in self._evidence_for_draft(writer_input, draft)
                ],
            },
            "current_draft": {
                "title": draft.title,
                "seo_title": draft.seo_title,
                "meta_description": draft.meta_description,
                "article_markdown": draft.article_markdown,
            },
            "human_revision_instructions": instructions,
        }
        prompt = (
            "Make only the requested human revision. Keep unrelated text unchanged. "
            "Use only supplied evidence; do not invent facts or remove qualifications. "
            "Preserve the article structure, source links, table, and FAQs. Return a JSON "
            "patch with `metadata` and `edits` keys, not a rewritten article. Metadata "
            "contains only changed title/seo_title/meta_description strings. Each edit "
            "contains an exact `old_text` and replacement `new_text`; old_text must occur "
            "once, and insertions must preserve their unique anchor in new_text. "
            "This is a proposal, not an accepted change.\n\n"
            "<revision_request>\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\n</revision_request>"
        )
        result = self._generate_targeted_update(draft, prompt)
        result = self._sanitize_evidence_references(result, writer_input.evidence)
        result = self._normalize_metadata(result)
        try:
            self._validate_seo_requirements(result, writer_input)
            seo_issues = []
        except RuntimeError as exc:
            seo_issues = [str(exc)]

        style_issues = lint_article(
            result.article_markdown,
            title=result.title,
            seo_title=result.seo_title,
            meta_description=result.meta_description,
        )
        existing_style_issues = set(
            lint_article(
                draft.article_markdown,
                title=draft.title,
                seo_title=draft.seo_title,
                meta_description=draft.meta_description,
            )
        )
        new_style_issues = [
            issue for issue in style_issues if issue not in existing_style_issues
        ]
        actionable_style_issues = self._actionable_style_issues(new_style_issues)
        issues = seo_issues + [f"Style: {issue}" for issue in actionable_style_issues]
        style_notes = [
            f"Style check: {issue}"
            for issue in new_style_issues
            if issue not in actionable_style_issues
        ]
        if issues:
            repair_prompt = self._build_repair_prompt(result, issues)
            result = self._generate_targeted_update(result, repair_prompt)
            result = self._sanitize_evidence_references(result, writer_input.evidence)
            result = self._normalize_metadata(result)
            self._validate_seo_requirements(result, writer_input)
            remaining = lint_article(
                result.article_markdown,
                title=result.title,
                seo_title=result.seo_title,
                meta_description=result.meta_description,
            )
            style_notes = [
                f"Style check: {issue}"
                for issue in [*remaining, *new_style_issues]
                if issue not in existing_style_issues
            ]
            style_notes = list(dict.fromkeys(style_notes))

        result = self._attach_supporting_evidence(result, writer_input.evidence)
        result = self._flag_unsupported_numeric_claims(result, writer_input.evidence)
        result = self._audit_claims_and_faqs(result)
        if style_notes:
            result = result.model_copy(
                update={"unresolved_items": list(dict.fromkeys([*result.unresolved_items, *style_notes]))}
            )
        result = self._append_references(result, writer_input.evidence)
        return result

    @staticmethod
    def _evidence_for_draft(
        writer_input: WriterInput,
        draft: WriterOutput,
    ) -> list[WriterEvidence]:
        """Return evidence that supports claims already present in the draft."""
        evidence_ids = {
            evidence_id
            for claim in draft.claims_used
            for evidence_id in claim.evidence_ids
        }
        if not evidence_ids:
            evidence_ids = {
                evidence_id
                for section in writer_input.sections
                for evidence_id in section.evidence_ids
            }
        if not evidence_ids:
            return writer_input.evidence
        return [
            item for item in writer_input.evidence if item.evidence_id in evidence_ids
        ]

    @staticmethod
    def _append_references(
        result: WriterOutput,
        evidence: list[WriterEvidence],
    ) -> WriterOutput:
        """Convert evidence-ID citations and append linked references after the FAQs."""
        references_heading = re.compile(r"(?im)^##\s+References\s*$")
        existing_references = references_heading.search(result.article_markdown)
        article = (
            result.article_markdown[:existing_references.start()].rstrip()
            if existing_references
            else result.article_markdown.rstrip()
        )
        url_link_pattern = re.compile(
            r"\[(?P<label>[^\]]*)\]\(\s*<?https?://[^)\s>]+>?\s*\)",
            re.IGNORECASE,
        )
        article, linked_url_count = url_link_pattern.subn(
            lambda match: match.group("label"),
            article,
        )
        bare_url_pattern = re.compile(r"https?://[^\s<>()\]]+", re.IGNORECASE)

        def remove_bare_url(match: re.Match[str]) -> str:
            url = match.group(0)
            trailing = ""
            while url and url[-1] in ".,;:!?":
                trailing = url[-1] + trailing
                url = url[:-1]
            return trailing

        article, bare_url_count = bare_url_pattern.subn(remove_bare_url, article)
        unresolved = list(result.unresolved_items)
        if linked_url_count or bare_url_count:
            warning = (
                "Writer included source URLs in article text; the URLs were removed. "
                "Use numbered citations and the References section."
            )
            if warning not in unresolved:
                unresolved.append(warning)
        citation_pattern = re.compile(
            r"\[(?P<token>ev[_-][A-Za-z0-9_-]+|\d+(?:\s*,\s*\d+)*)\]"
        )
        inline_ids = [
            evidence_id
            for match in citation_pattern.finditer(article)
            for evidence_id in match.group("token").split(",")
            if evidence_id.startswith(("ev_", "ev-"))
        ]
        requested_ids = {
            evidence_id
            for claim in result.claims_used
            for evidence_id in claim.evidence_ids
        } | set(inline_ids)
        evidence_by_id = {item.evidence_id: item for item in evidence}
        ordered_ids = [
            item.evidence_id
            for item in evidence
            if item.evidence_id in requested_ids
        ]
        ordered_ids.extend(
            evidence_id
            for evidence_id in [*inline_ids, *(id for claim in result.claims_used for id in claim.evidence_ids)]
            if evidence_id not in evidence_by_id and evidence_id not in ordered_ids
        )

        reference_numbers: dict[str, int] = {}
        references: list[str] = []
        number_by_source: dict[str, int] = {}
        citation_number = 0
        for evidence_id in ordered_ids:
            item = evidence_by_id.get(evidence_id)
            if item is None:
                warning = f"Writer cited unknown evidence ID {evidence_id!r}; the citation was removed."
                if warning not in unresolved:
                    unresolved.append(warning)
                continue

            source_url = canonicalize_evidence_url(item.source_url or "")
            doi = (item.doi or "").strip()
            doi = re.sub(
                r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)",
                "",
                doi,
                flags=re.IGNORECASE,
            ).rstrip(".,;")
            doi_url = (
                f"https://doi.org/{quote(doi, safe='/-._;()')}"
                if doi
                else ""
            )
            source_url = source_url or canonicalize_evidence_url(doi_url)
            source_title = " ".join((item.source_title or "").split())
            if not source_title and not source_url:
                warning = (
                    f"Evidence {evidence_id!r} has no source title or valid URL "
                    "and cannot be included in References."
                )
                if warning not in unresolved:
                    unresolved.append(warning)
                continue
            if not source_url:
                warning = (
                    f"Reference {evidence_id!r} has no valid source URL or DOI "
                    "and cannot be linked."
                )
                if warning not in unresolved:
                    unresolved.append(warning)

            source_key = (
                doi.casefold()
                or source_url
                or f"{source_title.casefold()}|{(item.source_author or item.source_organization or '').casefold()}"
            )
            if source_key in number_by_source:
                number = number_by_source[source_key]
                reference_numbers[evidence_id] = number
                continue

            citation_number += 1
            number_by_source[source_key] = citation_number
            reference_numbers[evidence_id] = citation_number
            author = " ".join((item.source_author or "").split())
            organization = " ".join((item.source_organization or "").split())
            creator = author or organization
            journal = " ".join((item.journal or "").split())
            label_parts = [value for value in (creator, source_title, journal) if value]
            label = ". ".join(label_parts) or urlsplit(source_url).hostname or "Source"
            escaped_label = label.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
            if source_url:
                destination = quote(source_url, safe=":/?#@!$&'*+,;=%-._~")
                rendered_label = f"[{escaped_label}]({destination})"
            else:
                rendered_label = escaped_label
            year_match = re.search(r"\b(?:19|20)\d{2}\b", item.publication_date or "")
            year = f" ({year_match.group()})" if year_match else ""
            doi_label = f" DOI: {doi}." if doi else ""
            references.append(
                f"{citation_number}. {rendered_label}{year}.{doi_label}"
            )

        def render_citation_group(matches: list[re.Match[str]]) -> str:
            numbers = []
            for match in matches:
                for token in match.group("token").split(","):
                    token = token.strip()
                    if token.startswith(("ev_", "ev-")):
                        number = reference_numbers.get(token)
                        if number is None:
                            continue
                    else:
                        number = int(token)
                        if number < 1 or number > citation_number:
                            warning = (
                                f"Writer used numeric citation [{number}], but only "
                                f"{citation_number} references were generated; the citation was removed."
                            )
                            if warning not in unresolved:
                                unresolved.append(warning)
                            continue
                    if number not in numbers:
                        numbers.append(number)

            rendered = []
            for number in numbers:
                rendered.append(f"[\\[{number}\\]](#reference-{number})")
            return ", ".join(rendered)

        citation_matches = list(citation_pattern.finditer(article))
        if citation_matches:
            parts = []
            cursor = 0
            index = 0
            while index < len(citation_matches):
                group = [citation_matches[index]]
                next_index = index + 1
                while next_index < len(citation_matches):
                    gap = article[group[-1].end():citation_matches[next_index].start()]
                    if not re.fullmatch(r"\s*,?\s*", gap):
                        break
                    group.append(citation_matches[next_index])
                    next_index += 1
                parts.append(article[cursor:group[0].start()])
                parts.append(render_citation_group(group))
                cursor = group[-1].end()
                index = next_index
            parts.append(article[cursor:])
            article = "".join(parts)
        if references:
            article += "\n\n## References\n\n" + "\n".join(references)
        return result.model_copy(
            update={
                "article_markdown": article,
                "word_count": len(re.findall(r"\b[\w’'-]+\b", article)),
                "sections": [
                    section
                    for section in result.sections
                    if section.heading.strip().casefold() != "references"
                ],
                "unresolved_items": list(dict.fromkeys(unresolved)),
            }
        )

    @staticmethod
    def _sections_from_markdown(
        article_markdown: str,
        existing_sections: list[WrittenSection] | None = None,
    ) -> list[WrittenSection]:
        """Derive section content from Markdown so the model need not repeat the article."""
        existing_by_heading = {
            section.heading.casefold(): section.evidence_ids_used
            for section in existing_sections or []
        }
        lines = article_markdown.splitlines()
        headings = [
            (index, len(match.group(1)), match.group(2).strip())
            for index, line in enumerate(lines)
            if (match := re.match(r"^(#{2,3})\s+(.+?)\s*#*\s*$", line.strip()))
        ]
        sections = []
        for position, (start, level, heading) in enumerate(headings):
            end = len(lines)
            for next_start, next_level, _ in headings[position + 1 :]:
                if next_level <= level:
                    end = next_start
                    break
            sections.append(
                WrittenSection(
                    heading=heading,
                    content_markdown="\n".join(lines[start + 1 : end]).strip(),
                    evidence_ids_used=existing_by_heading.get(heading.casefold(), []),
                )
            )
        return sections

    def _generate_targeted_update(self, draft: WriterOutput, prompt: str) -> WriterOutput:
        """Apply exact-text edits without asking the model to regenerate the complete article."""
        repair_token_limit = min(self.max_tokens, 9096)
        with self.client.messages.stream(
            model=self.model,
            max_tokens=repair_token_limit,
            system=self._repair_system_prompt(),
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": _WRITER_PATCH_SCHEMA,
                }
            },
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            response = stream.get_final_message()
        record_claude_response(response, self.model)
        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                f"Writer repair was truncated at {repair_token_limit:,} output tokens."
            )

        payload = self._extract_structured_payload(response, "Writer repair")
        metadata = payload.get("metadata", {})
        edits = payload.get("edits", [])
        if not isinstance(metadata, dict) or not isinstance(edits, list):
            raise RuntimeError("Writer repair must return metadata and edits in the expected JSON format.")
        if set(metadata) - {"title", "seo_title", "meta_description"}:
            raise RuntimeError("Writer repair returned unsupported metadata fields.")

        article = draft.article_markdown
        for edit in edits:
            if not isinstance(edit, dict):
                raise RuntimeError("Writer repair edits must be JSON objects.")
            old_text = edit.get("old_text")
            new_text = edit.get("new_text")
            if not isinstance(old_text, str) or not old_text or not isinstance(new_text, str):
                raise RuntimeError("Each Writer repair edit needs non-empty old_text and string new_text.")
            occurrences = article.count(old_text)
            if occurrences != 1:
                raise RuntimeError(
                    "Writer repair anchor must occur exactly once in the current draft; "
                    f"found {occurrences} occurrences."
                )
            article = article.replace(old_text, new_text, 1)

        updates = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if not isinstance(value, str):
                raise RuntimeError(f"Writer repair metadata field {key!r} must be a string.")
            if value != getattr(draft, key):
                updates[key] = value
        if article == draft.article_markdown and not updates:
            raise RuntimeError("Writer repair returned no changes.")

        updates.update({
            "article_markdown": article,
            "word_count": len(re.findall(r"\b[\w’'-]+\b", article)),
            "sections": self._sections_from_markdown(article, draft.sections),
        })
        return draft.model_copy(update=updates)

    def _enforce_style(
        self,
        result: WriterOutput,
        writer_input: WriterInput,
        allow_retry: bool = True,
    ) -> tuple[WriterOutput, list[str]]:
        """Lint the draft; if it fails, ask the model to fix exactly those issues once."""
        issues = lint_article(
            result.article_markdown,
            title=result.title,
            seo_title=result.seo_title,
            meta_description=result.meta_description,
        )
        if not issues:
            return result, []
        actionable_issues = WriterAgent._actionable_style_issues(issues)
        if not actionable_issues:
            return result, [f"Style check: {issue}" for issue in issues]
        if not allow_retry:
            return result, [f"Style check: {issue}" for issue in issues]

        retry_prompt = self._build_repair_prompt(
            result,
            [f"Style: {issue}" for issue in actionable_issues],
        )
        try:
            retried = self._generate_targeted_update(result, retry_prompt)
            retried = self._sanitize_evidence_references(retried, writer_input.evidence)
            retried = self._normalize_metadata(retried)
            self._validate_seo_requirements(retried, writer_input)
        except RuntimeError as exc:
            # The retry broke the structure or output; keep the first draft and report the issues.
            notes = [f"Style check: {issue}" for issue in issues]
            notes.append(f"Style repair failed: {exc}")
            return result, notes

        remaining = lint_article(
            retried.article_markdown,
            title=retried.title,
            seo_title=retried.seo_title,
            meta_description=retried.meta_description,
        )
        return retried, [f"Style check: {issue}" for issue in remaining]

    @staticmethod
    def _actionable_style_issues(issues: list[str]) -> list[str]:
        """Exclude optional word-choice guidance from automatic repair calls."""
        optional_patterns = (
            re.compile(r"^'[^']+' appears \d+ times; use it at most \d+\."),
            re.compile(r"^'Confirm/discuss with the vendor' appears \d+ times;"),
        )
        return [
            issue for issue in issues
            if not any(pattern.match(issue) for pattern in optional_patterns)
        ]

    @staticmethod
    def _build_repair_prompt(result: WriterOutput, issues: list[str]) -> str:
        issue_list = "\n".join(f"- {issue}" for issue in issues)
        requirements = []
        if any("Markdown table" in issue for issue in issues):
            requirements.append(
                "For a table issue, include at least one useful, valid Markdown table "
                "using only facts already supported by the draft."
            )
        if any("FAQ" in issue for issue in issues):
            requirements.append(
                "For an FAQ issue, include 3 to 5 relevant FAQ questions with answers."
            )
        targeted_requirements = "\n".join(requirements)
        return (
            "Fix only these validation issues in the supplied draft:\n"
            f"{issue_list}\n\n"
            f"{targeted_requirements}\n"
            "Preserve unrelated wording, facts, evidence references, and structure. "
            "Do not add facts, claims, sources, or URLs. Do not return the full article. "
            "Return JSON with `metadata` (only changed title/seo_title/meta_description "
            "values) and `edits` (objects with exact `old_text` and replacement `new_text`). "
            "Each old_text must occur exactly once. For insertions, keep the unique anchor "
            "in new_text and add the new text around it.\n\n"
            "<previous_draft>\n"
            f"{json.dumps(result.model_dump(include={'title', 'seo_title', 'meta_description', 'article_markdown'}), ensure_ascii=False, separators=(',', ':'))}"
            "\n</previous_draft>"
        )

    @staticmethod
    def _repair_system_prompt() -> str:
        return (
            "You are Agent 5. Make the smallest requested change to the supplied article. "
            "Use only its existing facts and source links; do not invent or strengthen claims. "
            "Return only a JSON patch with metadata and exact-text edits. Never return the "
            "complete article; each edit's old_text must match one unique span in the draft."
        )

    @staticmethod
    def _flag_unsupported_numeric_claims(
        result: WriterOutput,
        evidence: list[WriterEvidence],
    ) -> WriterOutput:
        """Record numerical claim concerns as review notes, not article text."""
        verify_pattern = re.compile(r"\[VERIFY:[^\]]*\]")
        number_pattern = re.compile(
            r"(?<![\w])(?:[$€£]\s*)?(\d[\d,]*(?:\.\d+)?)"
            r"\s*\+?\s*(%|percent\b|employees?\b|users?\b|weeks?\b|months?\b|"
            r"years?\b|days?\b|hours?\b|million\b|billion\b|times\b|locations?\b|"
            r"countries\b|clients?\b|customers?\b)?(?![A-Za-z])",
            re.IGNORECASE,
        )
        stop_words = {
            "about",
            "after",
            "and",
            "are",
            "as",
            "at",
            "by",
            "for",
            "from",
            "has",
            "have",
            "in",
            "into",
            "is",
            "it",
            "more",
            "of",
            "on",
            "or",
            "the",
            "their",
            "this",
            "to",
            "was",
            "were",
            "with",
        }

        def number_keys(text: str) -> set[tuple[str, str]]:
            keys = set()
            for match in number_pattern.finditer(text):
                digits = match.group(1).replace(",", "")
                unit = (match.group(2) or "").casefold()
                if not unit and re.fullmatch(r"(19|20)\d\d", digits):
                    continue  # a bare year is not a numerical claim
                keys.add((digits, unit))
            return keys

        verified = {
            item.evidence_id: item for item in evidence if item.status == "verified" and item.evidence_quote
        }
        lines = result.article_markdown.splitlines()
        unresolved = list(result.unresolved_items)
        clean_lines = []
        for original_line in lines:
            for marker in verify_pattern.finditer(original_line):
                note = marker.group()[8:-1].strip()
                if note:
                    warning = f"Writer verification note: {note}"
                    if warning not in unresolved:
                        unresolved.append(warning)
            line = verify_pattern.sub("", original_line).rstrip()
            scan_line = re.sub(r"^\s*\d+[.)]\s+", "", line)
            if scan_line.lstrip().startswith("### "):
                clean_lines.append(line)
                continue
            numbers = number_keys(scan_line)
            if not numbers:
                clean_lines.append(line)
                continue

            tokens = {
                token.casefold()
                for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", scan_line)
                if token.casefold() not in stop_words
            }
            unsupported = []
            limitations = []
            for number in numbers:
                supported = False
                for claim in result.claims_used:
                    if number not in number_keys(claim.claim):
                        continue
                    claim_tokens = {
                        token.casefold()
                        for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", claim.claim)
                        if token.casefold() not in stop_words
                    }
                    for evidence_id in claim.evidence_ids:
                        item = verified.get(evidence_id)
                        if item is None or number not in number_keys(item.evidence_quote or ""):
                            continue
                        quote_tokens = {
                            token.casefold()
                            for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", item.evidence_quote or "")
                            if token.casefold() not in stop_words
                        }
                        if tokens & claim_tokens & quote_tokens:
                            supported = True
                            if item.wording_boundary:
                                limitations.append(item.wording_boundary)
                            break
                    if supported:
                        break
                if not supported:
                    unsupported.append(number[0] + (number[1] or ""))

            if unsupported:
                for value in unsupported:
                    warning = (
                        "Numerical claim "
                        f"{value!r} in the article needs a cited verified evidence "
                        "quote and matching context."
                    )
                    if warning not in unresolved:
                        unresolved.append(warning)
            for limitation in dict.fromkeys(limitations):
                warning = (
                    "Preserve source limitation for numerical claim(s) "
                    f"{', '.join(unsupported) or 'in the article'}: {limitation}"
                )
                if warning not in unresolved:
                    unresolved.append(warning)
            clean_lines.append(line)

        article = "\n".join(clean_lines)
        word_count = len(re.findall(r"\b[\w’'-]+\b", article))
        return result.model_copy(
            update={
                "article_markdown": article,
                "word_count": word_count,
                "unresolved_items": unresolved,
            }
        )

    @staticmethod
    def _audit_claims_and_faqs(result: WriterOutput) -> WriterOutput:
        unresolved = list(result.unresolved_items)
        for claim in result.claims_used:
            evidence = claim.supporting_evidence
            if claim.claim_scope != "company_specific":
                continue
            support_messages = {
                "partially_supported": (
                    "Supplied evidence supports only part of this claim; narrow or " "qualify it"
                ),
                "unsupported": (
                    "Supplied evidence does not support this claim; that does not "
                    "by itself mean the claim is false"
                ),
                "contradicted": (
                    "Supplied evidence directly contradicts this claim; resolve "
                    "the conflict before publication"
                ),
                "inconclusive": (
                    "Supplied evidence is incomplete or ambiguous and cannot settle " "this claim"
                ),
                "not_assessed": "Claim-level evidence support was not assessed",
            }
            message = support_messages.get(claim.support_status)
            if message:
                warning = f"{message}: {claim.claim}"
                if warning not in unresolved:
                    unresolved.append(warning)
            if claim.citation_status in {"mismatch", "missing"}:
                for note in claim.validation_notes:
                    if note not in unresolved:
                        unresolved.append(note)
            for item in evidence:
                if item.wording_boundary:
                    warning = (
                        f"Preserve evidence limitation for claim {claim.claim!r}: " f"{item.wording_boundary}"
                    )
                    if warning not in unresolved:
                        unresolved.append(warning)

        lines = result.article_markdown.splitlines()
        faq_heading = re.compile(r"^##\s+Frequently Asked Questions\s*$", re.IGNORECASE)
        faq_index = next(
            (index for index, line in enumerate(lines) if faq_heading.match(line.strip())),
            None,
        )
        if faq_index is not None:
            faq_end = next(
                (
                    index
                    for index in range(faq_index + 1, len(lines))
                    if re.match(r"^##\s+", lines[index].strip())
                ),
                len(lines),
            )
            faq_lines = lines[faq_index + 1 : faq_end]
            questions = [index for index, line in enumerate(faq_lines) if re.match(r"^###\s+", line.strip())]
            for position, start in enumerate(questions):
                end = questions[position + 1] if position + 1 < len(questions) else len(faq_lines)
                answer = "\n".join(faq_lines[start + 1 : end]).strip()
                words = len(re.findall(r"\b[\w’'-]+\b", answer))
                if not 40 <= words <= 50:
                    question = faq_lines[start].strip()[4:].strip()
                    warning = (
                        f"FAQ answer for {question!r} has {words} words; "
                        "review it against the 40–50 word target."
                    )
                    if warning not in unresolved:
                        unresolved.append(warning)

        return result.model_copy(update={"unresolved_items": unresolved})

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        """Trim text to a limit at a sentence end if possible, otherwise at a word boundary."""
        text = text.strip()
        if len(text) <= limit:
            return text
        cut = text[:limit]
        ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", cut)]
        if ends and ends[-1] >= int(limit * 0.6):
            return cut[: ends[-1]].strip()
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        return cut.rstrip(" ,;:-")

    @staticmethod
    def _first_paragraph(article_markdown: str, limit: int) -> str:
        """First body paragraph after the H1, with link URLs removed, trimmed to a limit."""
        for block in re.split(r"\n\s*\n", article_markdown):
            text = block.strip()
            if text and not text.startswith(("#", "|", "-", "*", ">", "[VERIFY")):
                plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
                return WriterAgent._shorten(plain, limit)
        return ""

    @staticmethod
    def _normalize_metadata(result: WriterOutput) -> WriterOutput:
        """Keep a usable blog when Claude returns metadata outside SEO limits.

        A too-long meta description is trimmed, not replaced with the title. An empty one
        falls back to the article's first paragraph.
        """
        seo_title = result.seo_title.strip()
        if not seo_title or len(seo_title) > 60:
            seo_title = WriterAgent._shorten(result.title, 60) or result.title.strip()[:60]

        meta_description = result.meta_description.strip()
        if not meta_description:
            meta_description = WriterAgent._first_paragraph(
                result.article_markdown, 155
            ) or WriterAgent._shorten(result.title, 155)
        elif len(meta_description) > 155:
            meta_description = WriterAgent._shorten(meta_description, 155)

        return result.model_copy(update={"seo_title": seo_title, "meta_description": meta_description})

    def _generate_output(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
    ) -> WriterOutput:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt or self._system_prompt(),
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": _WRITER_OUTPUT_SCHEMA,
                }
            },
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            response = stream.get_final_message()
        record_claude_response(response, self.model)
        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                "Writer response was truncated at "
                f"{self.max_tokens:,} output tokens. "
                "Shorten the article requirements or reduce the requested scope."
            )

        payload = self._extract_structured_payload(response, "Writer")
        payload, support_status_warnings = self._normalize_claim_support_status(payload)

        try:
            result = WriterOutput.model_validate(payload)
        except ValidationError as exc:
            raise RuntimeError(f"Writer returned invalid output: {exc}") from exc

        actual_word_count = len(re.findall(r"\b[\w’'-]+\b", result.article_markdown))
        sections = self._sections_from_markdown(result.article_markdown, result.sections)
        return result.model_copy(
            update={
                "word_count": actual_word_count,
                "sections": sections,
                "unresolved_items": list(dict.fromkeys([*result.unresolved_items, *support_status_warnings])),
            }
        )

    @staticmethod
    def _extract_structured_payload(
        response: Any,
        operation: str,
    ) -> dict[str, Any]:
        if response.stop_reason != "end_turn":
            raise RuntimeError(
                f"{operation} did not complete its structured JSON response "
                f"(stop reason: {response.stop_reason or 'unknown'})."
            )

        text = "\n".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        if not text:
            raise RuntimeError(f"{operation} returned no structured JSON text.")

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{operation} returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"{operation} must return a JSON object.")
        return payload

    @staticmethod
    def _normalize_claim_support_status(
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """Default invalid model-generated claim assessments to the safe unknown state."""
        allowed_statuses = {
            "not_assessed",
            "supported",
            "partially_supported",
            "unsupported",
            "contradicted",
            "inconclusive",
        }
        normalized = dict(payload)
        raw_claims = payload.get("claims_used")
        if not isinstance(raw_claims, list):
            return normalized, []

        claims = []
        warnings = []
        for index, raw_claim in enumerate(raw_claims):
            if not isinstance(raw_claim, dict):
                claims.append(raw_claim)
                continue
            claim = dict(raw_claim)
            status = claim.get("support_status")
            if "support_status" in claim and (not isinstance(status, str) or status not in allowed_statuses):
                claim_text = str(claim.get("claim") or f"claim #{index + 1}")
                warnings.append(
                    f"Writer returned invalid support_status {status!r} for "
                    f"{claim_text!r}; set to not_assessed because evidence support "
                    "was not established by that invalid label."
                )
                claim["support_status"] = "not_assessed"
            claims.append(claim)
        normalized["claims_used"] = claims
        return normalized, warnings

    @staticmethod
    def _validate_evidence_references(
        result: WriterOutput,
        allowed_ids: set[str],
    ) -> None:
        for claim in result.claims_used:
            unknown_ids = set(claim.evidence_ids) - allowed_ids
            if unknown_ids:
                raise RuntimeError(
                    "Writer referenced unknown evidence IDs: " + ", ".join(sorted(unknown_ids))
                )

        for section in result.sections:
            unknown_ids = set(section.evidence_ids_used) - allowed_ids
            if unknown_ids:
                raise RuntimeError(
                    "Writer section referenced unknown evidence IDs: " + ", ".join(sorted(unknown_ids))
                )

    @staticmethod
    def _sanitize_evidence_references(
        result: WriterOutput,
        evidence: list[WriterEvidence],
    ) -> WriterOutput:
        """Drop model-invented evidence IDs while keeping the draft reviewable."""
        allowed_ids = {item.evidence_id for item in evidence}
        unresolved = list(result.unresolved_items)
        claims = []
        for claim in result.claims_used:
            unknown_ids = sorted(set(claim.evidence_ids) - allowed_ids)
            validation_notes = list(claim.validation_notes)
            if unknown_ids:
                warning = (
                    "Writer referenced evidence IDs that were not supplied; "
                    f"they were removed from claim {claim.claim!r}: " + ", ".join(unknown_ids)
                )
                if warning not in unresolved:
                    unresolved.append(warning)
                validation_notes.append(warning)
            claims.append(
                claim.model_copy(
                    update={
                        "evidence_ids": [
                            evidence_id
                            for evidence_id in dict.fromkeys(claim.evidence_ids)
                            if evidence_id in allowed_ids
                        ],
                        "citation_status": ("mismatch" if unknown_ids else "not_checked"),
                        "validation_notes": list(dict.fromkeys(validation_notes)),
                    }
                )
            )

        sections = []
        for section in result.sections:
            unknown_ids = sorted(set(section.evidence_ids_used) - allowed_ids)
            if unknown_ids:
                warning = (
                    "Writer section "
                    f"{section.heading!r} referenced evidence IDs that were not "
                    "supplied; they were removed: " + ", ".join(unknown_ids)
                )
                if warning not in unresolved:
                    unresolved.append(warning)
            sections.append(
                section.model_copy(
                    update={
                        "evidence_ids_used": [
                            evidence_id
                            for evidence_id in dict.fromkeys(section.evidence_ids_used)
                            if evidence_id in allowed_ids
                        ]
                    }
                )
            )

        return result.model_copy(
            update={
                "claims_used": claims,
                "sections": sections,
                "unresolved_items": unresolved,
            }
        )

    @staticmethod
    def _attach_supporting_evidence(
        result: WriterOutput,
        evidence: list[WriterEvidence],
    ) -> WriterOutput:
        evidence_by_id = {item.evidence_id: item for item in evidence}
        claims = []
        unresolved = list(result.unresolved_items)
        for claim in result.claims_used:
            referenced = [evidence_by_id[evidence_id] for evidence_id in dict.fromkeys(claim.evidence_ids)]
            referenced_urls = {
                canonicalize_evidence_url(item.source_url): item.source_url
                for item in referenced
                if item.source_url and canonicalize_evidence_url(item.source_url)
            }
            unlinked_urls = {
                url
                for url in claim.source_urls
                if not canonicalize_evidence_url(url) or canonicalize_evidence_url(url) not in referenced_urls
            }
            validation_notes = list(claim.validation_notes)
            if unlinked_urls:
                warning = (
                    "Writer returned source URLs not linked to supplied evidence "
                    f"for claim {claim.claim!r}; those URLs were removed: " + ", ".join(sorted(unlinked_urls))
                )
                if warning not in unresolved:
                    unresolved.append(warning)
                validation_notes.append(warning)
            elif not referenced and claim.claim_scope == "company_specific":
                warning = (
                    "Writer claim has no registered evidence IDs; treat it as "
                    f"unsupported pending human verification: {claim.claim}"
                )
                if warning not in unresolved:
                    unresolved.append(warning)
                validation_notes.append(warning)
            citation_status = (
                "mismatch"
                if unlinked_urls or claim.citation_status == "mismatch"
                else "linked"
                if referenced
                else "missing"
            )
            support_status = claim.support_status
            if (
                not referenced
                and claim.claim_scope == "company_specific"
                and support_status == "not_assessed"
            ):
                support_status = "unsupported"
            if referenced:
                validation_notes.append(
                    (
                        "No claim-level evidence assessment was provided."
                        if support_status == "not_assessed"
                        else "Claim support is an AI assessment of the supplied "
                        "evidence, not independent verification."
                    )
                )
            elif claim.claim_scope == "company_specific" and support_status == "unsupported":
                validation_notes.append(
                    "No supplied evidence was attached to this company-specific claim; "
                    "unsupported here does not mean the claim is false."
                )
            claims.append(
                claim.model_copy(
                    update={
                        "source_urls": list(dict.fromkeys(referenced_urls.values())),
                        "supporting_evidence": referenced,
                        "citation_status": citation_status,
                        "support_status": support_status,
                        "validation_notes": list(dict.fromkeys(validation_notes)),
                    }
                )
            )
        return result.model_copy(update={"claims_used": claims, "unresolved_items": unresolved})

    @staticmethod
    def _system_prompt() -> str:
        return """You are Agent 5, the BLOG WRITER. Write a reader-focused article in the supplied company's voice. Lead with the reader's challenge or direct answer, and answer each heading directly.
Follow the brief, structure, SEO limits, and output schema. Keep the writing clear, specific, and concise. Avoid filler, repetition, generic copy, banned jargon, bold-label lists, and more than two dashes or lists.
Use only supplied evidence for company-specific facts. Never invent capabilities, outcomes, numbers, citations, or URLs. Respect evidence status and wording boundaries; qualify uncertain claims and do not present unsupported figures as facts.
Use source links only when they support the adjacent claim. Do not imply independent verification of company-reported claims.
Return only the requested JSON object, with no Markdown fence or planning."""

    @staticmethod
    def _build_user_prompt(data: WriterInput) -> str:
        payload = data.model_dump(mode="json", exclude_defaults=True, exclude_none=True)
        section_evidence_ids = {
            evidence_id
            for section in payload.get("sections", [])
            for evidence_id in section.get("evidence_ids", [])
        }
        if section_evidence_ids:
            payload["evidence"] = [
                item
                for item in payload.get("evidence", [])
                if item["evidence_id"] in section_evidence_ids
            ]
        brief_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        evidence_guidance = ""
        if any(item.relevance is not None for item in data.evidence):
            evidence_guidance = """
EVIDENCE RELEVANCE
- Prioritize direct evidence for the main topic.
- Use related evidence only as a clearly identified supporting example.
- Do not present background evidence as direct support.
- For General Blog evidence, "verified" means the quote matched the source
  text; it does not independently establish scientific correctness.
"""

        return f"""
Write a complete blog post using the writer input below.

<writer_input>
{brief_json}
</writer_input>
{evidence_guidance}

Write as the company's knowledgeable content team, speaking directly to
prospective customers. Lead with the customer's challenge, then explain
verified capabilities and supported benefits confidently. Use clear,
relevant calls to action. Do not write as an outside investigator or auditor.
Never invent customer outcomes, capabilities, case-study details, or
guarantees. Present limitations and unknowns constructively while preserving
evidence qualifications.

Before drafting, internally determine:
- the intended reader and their main problem or question;
- the article's central angle and takeaway;
- a logical progression from introduction to conclusion;
- which supplied evidence supports each material factual claim;
- which requested claims must be qualified or omitted.

Do not output this planning.

ARTICLE STRUCTURE

- Write an engaging, topic-focused H1 title matching the title field.
- Write a separate SEO title of no more than 60 characters and a concise,
  accurate meta description of 120 to 155 characters that summarizes the post
  and does not repeat the title.
- Begin with an introduction that establishes the reader's situation and why
  the topic matters. State the direct answer to the main question within the
  first 60 words after the H1. Do not describe what the article covers.
- Develop the topic in a logical sequence using useful H2 and H3 headings.
- Include at least one useful Markdown table built only from supported
  information in the writer input. Do not invent table facts or comparisons.
- target_word_count in the writer input is a rough guide (within 10%). Never
  reach it by adding generic filler, repeated points, or lists of what a good
  vendor "should" offer. If the evidence supports less, write less and record
  the gap in unresolved_items.
- Introduce the company's verified capabilities where they help address the
  customer's challenge, without turning the article into a company profile.
- Include concrete, evidence-supported details where available.
- Cite factual and research-statistic claims inline using the exact supplied
  evidence ID in square brackets, for example [ev_001]. Never create numeric
  citations or bare reference numbers such as 1, [1], or [22]. Use only IDs
  present in the evidence list and include those IDs in claims_used. The app
  converts these markers to bracketed numbered citations that link to the
  matching entry in References.
- Do not print raw URLs in the article. The app adds References entries with
  clickable source titles and links each in-text citation to its matching entry.
- Explain supported practical benefits confidently without inventing outcomes.
- End with a concise conclusion and a clear, relevant call to action.
- Include exactly one H2 titled "Frequently Asked Questions" with 3 to 5
  generic, relevant H3 questions a reader might naturally ask. Base them on
  the article's content, search intent, and supplied keyword research. Each
  answer must be 40–50 words, self-contained, and supported by supplied
  evidence. Do not ask about
  measurable improvements, ROI, performance changes, or outcomes unless
  supplied evidence explicitly reports them. Avoid unsupported company-
  specific facts, capabilities, or statistics. Pick a different question if
  the article and supplied evidence cannot answer it reliably.
  FAQs should complement the article, not repeat whole paragraphs or insert
  keywords unnaturally. Keep answers grounded in the supplied evidence and
  include relevant evidence IDs in claims_used when available. Paraphrasing
  is fine.
- Place the conclusion before the FAQ section. Keep the conclusion's final
  ~60 words focused on the takeaway and one relevant call to action.
- Do not write a References section; the app adds one after the FAQs from
  the evidence IDs actually used.
- Follow all required brief topics. Do not add sections merely
  to increase length.
- For claims_used, include only distinct material factual claims. Keep each
  claim concise and combine claims that use the same evidence where accurate.
  The app adds source URLs, evidence details, and validation notes.
- Keep unresolved_items concise. Report only distinct, material evidence gaps
  or risks. Keep each item focused. Combine related gaps only when they share
  the same cause. Do not repeat limitations that the article already handles
  correctly, and do not omit a material warning about evidence or claims.
- Avoid a generic opening such as "{data.company_name} is a leading company"
  unless the brief specifically requires that approach and the claim is
  supported by evidence.

The blog must fit this particular topic. Do not force every article into
the same formula or use unsupported anecdotes, hypothetical customer
stories presented as real, or invented statistics.
Use primary_keyword, secondary_keywords, related_topics, and
question_based_queries from the writer input naturally; never add
unsupported search-volume or ranking metrics.

Return compact JSON with these fields:
- title: string
- seo_title: string (60 characters maximum)
- meta_description: string
- article_markdown: string
- word_count: integer (approximate; it will be recalculated)
- claims_used: array of objects with claim, evidence_ids,
  claim_scope="company_specific|industry_context|general_education",
  citation_status="not_checked",
  support_status="supported|partially_supported|unsupported|contradicted|inconclusive|not_assessed"
- unresolved_items: array of strings
- brief_coverage: array of strings

Do not return sections; they are derived from article_markdown.
For claims_used, record important factual claims that actually appear in
the article and their supporting evidence IDs and URLs. Use an empty
evidence_ids list if no supplied evidence supports a claim. Do not invent
supporting references. Set citation_status="not_checked"; assess
support_status against the supplied claim, quote, source context, and
evidence status. Use:
- supported: the supplied evidence directly supports the whole claim.
- partially_supported: it supports a narrower version; state the limit.
- contradicted: supplied evidence directly conflicts with the claim.
- unsupported: the available evidence does not support the claim.
- inconclusive: evidence is incomplete or ambiguous and cannot settle it.
- not_assessed: there is not enough supplied information to assess it.
A citation URL/ID mismatch is a citation_status issue, not evidence that a
claim is unsupported or contradicted. This assessment is an AI judgment,
not independent verification. The app creates source_urls, supporting_evidence,
and validation_notes from the supplied evidence; do not return those fields.
Validation code resolves citation links separately. Mark claims about the named
customer, its products, services, capabilities, processes, or results as
company_specific. Mark broad industry observations as industry_context and
plain explanations as general_education. Do not turn generic context into
a company claim or a company-evidence research gap.

For sections, reflect the actual H2/H3 sections in the finished article.
The article_markdown field must contain the complete Markdown article,
including its H1 title and inline source links where supported.

The meta_description should accurately describe the article in a concise,
reader-focused way, within 155 characters. Do not add unsupported promises
or claims.
"""

    @staticmethod
    def _validate_seo_requirements(
        result: WriterOutput,
        writer_input: WriterInput,
    ) -> None:
        """Enforce article and FAQ structure; evidence IDs are checked separately."""
        lines = result.article_markdown.splitlines()
        h1_lines = [line.strip() for line in lines if re.match(r"^#\s+", line)]
        if len(h1_lines) != 1 or h1_lines[0][2:].strip() != result.title.strip():
            raise RuntimeError("Writer article must contain one H1 matching its title.")
        if not result.seo_title.strip() or len(result.seo_title.strip()) > 60:
            raise RuntimeError("Writer SEO title must contain 1 to 60 characters.")
        if not result.meta_description.strip() or len(result.meta_description.strip()) > 155:
            raise RuntimeError("Writer meta description must contain 1 to 155 characters.")

        faq_heading = re.compile(r"^##\s+Frequently Asked Questions\s*$", re.IGNORECASE)
        faq_index = next(
            (index for index, line in enumerate(lines) if faq_heading.match(line.strip())),
            None,
        )
        if faq_index is None:
            raise RuntimeError("Writer article is missing its Frequently Asked Questions section.")
        if not any(
            section.heading.strip().casefold() == "frequently asked questions" for section in result.sections
        ):
            raise RuntimeError("Writer sections must include the FAQ section.")
        faq_end = next(
            (
                index
                for index in range(faq_index + 1, len(lines))
                if re.match(r"^##\s+", lines[index].strip())
            ),
            len(lines),
        )
        faq_lines = lines[faq_index + 1 : faq_end]
        question_indexes = [
            index for index, line in enumerate(faq_lines) if re.match(r"^###\s+", line.strip())
        ]
        if not 3 <= len(question_indexes) <= 5:
            raise RuntimeError("Writer FAQ section must contain 3 to 5 questions.")

        for position, start in enumerate(question_indexes):
            heading = faq_lines[start].strip()[4:].strip()
            if not heading.endswith("?"):
                raise RuntimeError("Each FAQ heading must be phrased as a question.")
            end = question_indexes[position + 1] if position + 1 < len(question_indexes) else len(faq_lines)
            answer = "\n".join(faq_lines[start + 1 : end]).strip()
            if not answer:
                raise RuntimeError(f"FAQ question has no answer: {heading}")

        has_table = False
        for index in range(len(lines) - 2):
            header = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
            separators = [cell.strip() for cell in lines[index + 1].strip().strip("|").split("|")]
            values = [cell.strip() for cell in lines[index + 2].strip().strip("|").split("|")]
            if (
                len(header) >= 2
                and len(header) == len(separators) == len(values)
                and all(re.fullmatch(r":?-{3,}:?", cell) for cell in separators)
                and all(values)
            ):
                has_table = True
                break
        if not has_table:
            raise RuntimeError("Writer article must include at least one valid Markdown table.")
