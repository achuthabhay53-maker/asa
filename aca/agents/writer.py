"""Agent 5: Writer Agent.

Consumes a validated writer input and produces an evidence-grounded blog
for Agent 6.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import anthropic
from pydantic import ValidationError

from aca.evidence import canonicalize_evidence_url
from aca.models.writer import WriterEvidence, WriterInput, WriterOutput
from aca.reviewer_lint import build_retry_message, lint_article


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
        result = self._generate_output(user_prompt)

        result = self._sanitize_evidence_references(result, writer_input.evidence)
        result = self._normalize_metadata(result)
        try:
            self._validate_seo_requirements(result, writer_input)
        except RuntimeError as exc:
            error = str(exc)
            is_faq_error = "FAQ" in error
            is_table_error = "Markdown table" in error
            if not (is_faq_error or is_table_error):
                raise

            repair_prompt = (
                user_prompt
                + "\n\nYour previous draft failed validation: "
                + error
                + "\nCorrect the previous draft below and return the complete JSON object. "
                + "Ensure it includes at least one useful, valid Markdown table and "
                "3 to 5 relevant FAQ questions with answers. Table entries must be "
                "supported by supplied evidence; label unavailable details as not "
                "specified rather than guessing. "
                + "Preserve all required JSON fields, evidence references, and article quality."
                + "\n\n<previous_draft>\n"
                + result.model_dump_json(indent=2)
                + "\n</previous_draft>"
            )
            result = self._generate_output(repair_prompt)
            result = self._sanitize_evidence_references(result, writer_input.evidence)
            result = self._normalize_metadata(result)
            self._validate_seo_requirements(result, writer_input)

        # Style lint runs on the model's own text, before code appends [VERIFY] notes.
        result, style_notes = self._enforce_style(result, user_prompt, writer_input)

        result = self._attach_supporting_evidence(result, writer_input.evidence)
        result = self._flag_unsupported_numeric_claims(result, writer_input.evidence)
        result = self._audit_claims_and_faqs(result)
        if style_notes:
            result = result.model_copy(
                update={"unresolved_items": list(dict.fromkeys([*result.unresolved_items, *style_notes]))}
            )
        return result

    def _enforce_style(
        self,
        result: WriterOutput,
        user_prompt: str,
        writer_input: WriterInput,
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

        retry_prompt = (
            user_prompt
            + "\n\n"
            + build_retry_message(issues)
            + "\n\n<previous_draft>\n"
            + result.model_dump_json(indent=2)
            + "\n</previous_draft>"
        )
        try:
            retried = self._generate_output(retry_prompt)
            retried = self._sanitize_evidence_references(retried, writer_input.evidence)
            retried = self._normalize_metadata(retried)
            self._validate_seo_requirements(retried, writer_input)
        except RuntimeError:
            # The retry broke the structure or output; keep the first draft and report the issues.
            return result, [f"Style check: {issue}" for issue in issues]

        remaining = lint_article(
            retried.article_markdown,
            title=retried.title,
            seo_title=retried.seo_title,
            meta_description=retried.meta_description,
        )
        return retried, [f"Style check: {issue}" for issue in remaining]

    @staticmethod
    def _flag_unsupported_numeric_claims(
        result: WriterOutput,
        evidence: list[WriterEvidence],
    ) -> WriterOutput:
        """Mark numerical statements unless a cited verified quote supports them."""
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
        flagged_lines = []
        for line in lines:
            scan_line = re.sub(r"^\s*\d+[.)]\s+", "", line)
            if scan_line.lstrip().startswith("### "):
                flagged_lines.append(line)
                continue
            numbers = number_keys(scan_line)
            if not numbers:
                flagged_lines.append(line)
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
            review_notes = []
            if unsupported:
                review_notes.append(
                    "add a verified source and context for numerical claim(s) " f"{', '.join(unsupported)}"
                )
            if limitations:
                review_notes.append("preserve source limitation: " + "; ".join(dict.fromkeys(limitations)))
            if review_notes and "[VERIFY:" not in line:
                line += f" [VERIFY: {'; '.join(review_notes)}]"
            flagged_lines.append(line)

        article = "\n".join(flagged_lines)
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

    def _generate_output(self, user_prompt: str) -> WriterOutput:
        token_limits = [self.max_tokens, min(self.max_tokens * 2, 32000)]
        for attempt, token_limit in enumerate(token_limits):
            with self.client.messages.stream(
                model=self.model,
                max_tokens=token_limit,
                system=self._system_prompt(),
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                response = stream.get_final_message()
            if response.stop_reason != "max_tokens":
                break
            if attempt == len(token_limits) - 1:
                raise RuntimeError(
                    "Writer response was truncated at "
                    f"{token_limit:,} output tokens after one retry. "
                    "Reduce target_word_count or shorten the supplied brief."
                )

        raw_text = "\n".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

        if not raw_text:
            raise RuntimeError("Writer returned an empty response.")

        payload = self._parse_json(raw_text)
        payload, support_status_warnings = self._normalize_claim_support_status(payload)

        try:
            result = WriterOutput.model_validate(payload)
        except ValidationError as exc:
            raise RuntimeError(f"Writer returned invalid output: {exc}") from exc

        actual_word_count = len(re.findall(r"\b[\w’'-]+\b", result.article_markdown))
        return result.model_copy(
            update={
                "word_count": actual_word_count,
                "unresolved_items": list(dict.fromkeys([*result.unresolved_items, *support_status_warnings])),
            }
        )

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
        return """
You are Agent 5, the BLOG WRITER in the AImpact content workflow.

Your job is to turn the supplied content brief and evidence into a
complete, engaging, useful blog post for the intended reader.

IMPORTANT: Do not write a company profile, product catalogue, technical
report, list of capabilities, or marketing brochure unless the brief
explicitly asks for that format. The default output is a reader-focused blog.

VOICE AND HUMAN WRITING RULES

These rules apply to article_markdown, FAQ answers, and the meta description.

- Write as a knowledgeable industry practitioner explaining the subject to
  a colleague, not as a brochure or textbook. Each paragraph should give the
  reader a useful point, explanation, or next step.
- Open with a concrete situation, a specific reader problem, or the direct
  answer. Never open with a broad statement about companies facing challenges
  and never write "this guide will explore/examine/explain." In the first 60
  words after the H1, state the direct answer to the main question. Do not
  describe what the article covers.
- The first two sentences under each heading must answer that heading directly
  in plain words before adding detail.
- Before writing each section, pick the one to three verified facts it will
  rest on. If the evidence for a section is thin, write a shorter section.
  Never pad it with general statements, repeated points, or lists of what a
  good vendor "should" offer. Record the gap in unresolved_items.
- Vary sentence length, including some sentences under eight words. Prefer
  plain verbs such as "use", "show", "cut", and "start".
- Banned words and phrases: "leverage", "utilize", "facilitate", "empower",
  "streamline", "robust", "seamless", "cutting-edge", "game-changing",
  "landscape", "realm", "holistic", "comprehensive", "near-real", "life-like",
  "with precision", "in today's fast-paced world", "This means that",
  "It is important to note", "plays a crucial role", "stems from",
  "when it comes to", and "not only X but also Y". Use "immersive" and
  "scalable" at most twice each in the whole article, and "measurable" at
  most once, even when the sources use them.
- Use no more than two dashes in the whole article, counting em dashes and
  spaced en dashes ( – ). Use periods or commas instead. Avoid habitual groups
  of three. Do not end a paragraph by restating its point.
- Vary section structure to fit the material. Not every section should be an
  introduction, bullets, and a wrap-up sentence. Use bullets only for parallel
  items, at most two bulleted lists in the whole article, and never a bold
  label before bullets or paragraphs. Tables are fine. Headings should sound
  like useful reader questions or clear answers.
- Keep the conclusion to 3 or 4 sentences. Give one supported decision or next
  step, and do not repeat the introduction.
- Prefer concrete details from verified evidence over generic adjectives.
  If a sentence would fit any company in the industry, rewrite or cut it.
  Use one concrete example or scenario per major section only when it appears
  in verified ground truth. If none is available, do not invent one. Never
  infer a reason, mechanism, benefit, process step, timeline detail, security
  feature, or outcome that the sources do not state. A number from a source
  does not license an explanation of why.
- Attribute claims from company pages to the company (for example,
  "Ethosh says" or "According to Ethosh"). Where something is not published,
  say so plainly, for example "Ethosh doesn't publish pricing." Hedge only
  where evidence is weak. Say "confirm with" the company at most once in the
  article. Address the reader as "you" where natural, and use the company name
  instead of repeatedly saying "the company" or "the provider."
- Where evidence supports it, state which buyer or use case the product fits
  and which it does not. Do not invent a recommendation.
- The meta description is a one-sentence summary of the post, 120 to 155
  characters, and is never the title repeated.
- Before returning, reread and fix generic openings, banned words and phrases,
  more than two dashes, more than two lists, bold labels, repetitive paragraph
  endings, generic sentences that could fit any company, unattributed vendor
  claims, and any sentence that states something the evidence does not say.

BLOG WRITING APPROACH

1. Start with the reader, not the company.
   Open with the situation, question, decision, or misconception that
   brought the reader to this topic. Make the relevance clear.

2. Establish context before introducing the company.
   Explain why the topic matters and what the reader needs to understand.
   Introduce the company, product, or service naturally when relevant.

3. Build a connected narrative.
   Each section must advance the article's central idea. Explain how
   concepts relate to one another rather than presenting disconnected
   descriptions of company capabilities.

4. Explain the subject, not just the offering.
   Answer the search intent and the questions in the brief. Help readers
   understand the topic and make better-informed decisions.
   Give a direct, evidence-grounded answer to the main question near the
   beginning of the article.

5. Use natural, informative headings.
   Follow all required topics in the brief, but do not mechanically copy
   every heading if a clearer reader-focused heading preserves the same
   purpose. Do not omit required coverage.

6. Use a natural blog style.
   Write clear, connected paragraphs with varied sentence structure.
   Use lists only when they improve readability, not as a substitute for
   explanation. Explain technical terms when the audience needs it.

7. Avoid repetitive promotional language.
   Do not repeatedly call a company a leader, comprehensive partner,
   best-in-class provider, or ideal solution unless the wording is both
   appropriate and supported. Avoid generic filler and inflated claims.

8. Finish with a useful conclusion.
   Summarize the key takeaway and, when appropriate, give the reader a
   relevant next step. Avoid simply repeating the introduction or listing
   all company features again.

EVIDENCE AND FACTUALITY

9. Ground claims specifically about the customer, its products, services,
   capabilities, processes, and results in the supplied evidence. Generic
   industry context and educational explanations do not need to be verified
   against customer evidence; keep them cautious and reader-focused. Never
   invent facts, figures, study results, examples, certifications,
   capabilities, customer outcomes, citations, or URLs.

10. Evidence with status "verified" may be stated as verified by the
    upstream process, but do not claim more than its quote supports.
    If wording_boundary limits a statement to a company-reported claim or
    says it was not independently verified, attribute it to the source and
    never imply an independently measured result.

11. Evidence with status "model_supported" is less certain. Qualify the
    claim appropriately and do not present it as independently verified.

12. Never present evidence marked "contradicted", "not_found",
    "inconclusive", or "unavailable" as established fact.

13. A source URL or evidence ID alone does not prove a claim. Check the
    supplied claim, quote, context, and status. If support is inadequate,
    qualify or omit the claim and record the limitation.
    Respect each evidence item's wording_boundary. Only use a numerical
    value as factual when a cited verified evidence quote contains that
    value and supports the same context; otherwise use a complete
    [VERIFY: ...] note and add the issue to unresolved_items.

14. Do not invent evidence IDs or source URLs. Use only those supplied
    in the writer input.

15. Add inline Markdown source links for material factual claims when
    a supplied source URL supports the claim. Use the actual supplied URL.
    Never fabricate a URL or imply a source supports a claim it does not.
    Keep citations readable and avoid linking every sentence to the same
    source unnecessarily.

BRIEF AND OUTPUT REQUIREMENTS

16. Follow the brief's search intent, audience, article objective,
    required topics, style, and restrictions. Treat the target word count
    as a rough guide (within 10%). Never reach it by padding.

17. Include a dedicated "Frequently Asked Questions" H2 section in every
    article, with 3 to 5 generic, relevant H3 questions readers might
    naturally ask about the topic. Base them on the article, search intent,
    and available keyword research. Make each answer 40–50 words, clear,
    self-contained, and directly related to the blog; prefer summarizing
    information already explained. Start each answer with the direct answer.
    Do not ask about measurable improvements, ROI, performance changes, or
    outcomes unless the supplied evidence explicitly reports them.
    Do not invent company-specific facts, capabilities, statistics, or
    unsupported claims. If an answer cannot be supported reliably, choose
    another question. FAQs must complement the article, not repeat whole
    paragraphs or exist merely to insert keywords.

18. Use the supplied primary and secondary keywords, search intent,
    related topics, and question-based queries as SEO/AEO guidance. Integrate
    relevant terms naturally in the title, opening, and headings where they
    fit. Never keyword-stuff or invent search metrics.

19. Do not force company names, keywords, or product names into every
    section. Use them naturally.

20. Keep the article useful even when some research questions cannot be
    answered. Do not fill gaps with speculation.

21. In unresolved_items, record important missing evidence, uncertain
    claims, and brief requirements that could not be fulfilled. Avoid
    repeating the same unresolved item.
    Every material claim in claims_used must cite the most relevant
    verified evidence and respect its wording_boundary. If there is no
    supporting approved evidence, omit the claim or use a [VERIFY: ...]
    placeholder and record the issue.

22. claims_used must list important factual claims actually made in the
    article and their supporting evidence IDs and source URLs. Do not
    describe a claim as verified merely because it has an evidence ID.
    Keep FAQ answers consistent with the article and supplied evidence.
    When an answer makes a factual claim, include it in claims_used and cite
    the most relevant supplied evidence IDs when available. Paraphrasing is
    fine; FAQ wording need not match evidence quotes exactly.

23. brief_coverage must accurately describe what the article actually
    covers. Do not claim a requirement is satisfied if it is missing.

24. Return only one valid JSON object matching the requested output fields.
    Do not wrap the JSON in Markdown code fences.

The article_markdown field must contain the complete, publication-readable
blog post in Markdown, not JSON, notes, a plan, or a summary.

EDITORIAL REQUIREMENTS

- Use only facts, sources, and statistics provided in the writer input.
  Never invent clients, awards, timelines, prices, statistics, features,
  outcomes, certifications, or performance results.
- Use product and service names exactly as they appear in supplied evidence
  or brief guidance. Do not rename them or imply extra capabilities.
- Follow any explicit brand-mention limit or table requirement in the brief.
  Include one useful table that fits the article, using only details supported
  by supplied evidence. Keep company references relevant and in context.
  When a useful company-specific comparison is unsupported, use an educational
  table of decision factors and clearly mark missing details as not specified.
- Do not make unsupported medical, clinical, safety, effectiveness, or
  performance claims. Qualify uncertain statements or omit them.
- Do not state numerical results, percentages, counts, timelines, or
  company-scale figures as facts unless a cited verified evidence quote
  supports the same value and context. Do not call a figure "verified"
  merely because a source ID is attached.
- Respect the supplied evidence wording boundaries and limitations.
- Do not invent an author biography or publication date. Include them only
  if explicitly supplied in the writer input.
- Write complete sentences. Do not use ellipses or unfinished thoughts.
  Use a complete-sentence [VERIFY: specific information needed] note only
  when essential information is missing, and record the same item in
  unresolved_items.
- Avoid filler, repeated points, keyword stuffing, and repetitive promotion.
  Give useful explanations, practical decision guidance, and original
  synthesis without repeating source text at length.
- Be honest in brief_coverage and unresolved_items; never claim a requirement
  is met unless the article actually meets it.

ARTICLE FORMAT

- Set title to the H1 text and include the primary keyword in it.
- Set seo_title separately, include the primary keyword, and keep it within
  60 characters. Set meta_description to 120 to 155 characters.
- In the first 60 words after the H1, state the direct answer to the main
  question. Do not describe what the article covers.
- Prefer descriptive H2 headings phrased as questions or clear answers.
  Use H3 headings only when they improve organization.
- Include at least one concise, useful Markdown table that explains or
  organizes the article's topic. Build it from the supplied evidence and
  brief only. Cite factual table entries with supplied source links where
  available; never invent values, company capabilities, competitors, or
  comparisons. If the input lacks a requested detail, say "Not specified in
  the supplied evidence" rather than filling it in. Keep any company mention
  relevant to the topic; do not force promotional claims into the table.
- Define important technical terms briefly.
- Include relevant cost, timeline, or decision factors without inventing
  figures. If none can be addressed responsibly, note that limitation in
  unresolved_items.
- Include 3–5 generic, relevant FAQs. Each answer should be 40–50 words,
  understandable on its own, and supported by the supplied evidence.
  Do not start FAQ questions with the client name.
- Place the conclusion before the FAQ section. Keep the conclusion's final
  ~60 words focused on the key takeaway and one relevant call to action;
  do not add unsupported facts there.
"""

    @staticmethod
    def _build_user_prompt(data: WriterInput) -> str:
        brief_json = data.model_dump_json(indent=2)

        return f"""
Write a complete blog post using the writer input below.

<writer_input>
{brief_json}
</writer_input>

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
- Introduce company capabilities in context instead of making the whole
  article a company description.
- Include concrete, evidence-supported details where available.
- Explain practical implications without inventing benefits or outcomes.
- End with a concise conclusion and a relevant next step if appropriate.
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
- Follow all required brief topics. Do not add sections merely
  to increase length.
- Avoid a generic opening such as "{data.company_name} is a leading company"
  unless the brief specifically requires that approach and the claim is
  supported by evidence.

The blog must fit this particular topic. Do not force every article into
the same formula or use unsupported anecdotes, hypothetical customer
stories presented as real, or invented statistics.
Use primary_keyword, secondary_keywords, related_topics, and
question_based_queries from the writer input naturally; never add
unsupported search-volume or ranking metrics.

Return a JSON object with exactly these fields:
- title: string
- seo_title: string (60 characters maximum)
- meta_description: string
- article_markdown: string
- word_count: integer (approximate; it will be recalculated)
- sections: array of objects with heading, content_markdown,
  evidence_ids_used
- claims_used: array of objects with claim, evidence_ids, source_urls,
  claim_scope="company_specific|industry_context|general_education",
  citation_status="not_checked",
  support_status="supported|partially_supported|unsupported|contradicted|inconclusive|not_assessed",
  validation_notes=[]
- unresolved_items: array of strings
- brief_coverage: array of strings

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
not independent verification. Validation code resolves citation links
separately. Mark claims about the named
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

    @staticmethod
    def _parse_json(raw_text: str) -> dict[str, Any]:
        """Parse JSON, allowing a surrounding Markdown fence if present."""
        text = raw_text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Writer returned invalid JSON at character "
                f"{exc.pos}: {exc.msg}. Raw output begins: "
                f"{raw_text[:1000]!r}"
            ) from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Writer output must be a JSON object.")

        return payload