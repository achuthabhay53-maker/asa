
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

from aca.models.writer import WriterInput, WriterOutput


class WriterAgent:
    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 10000,
    ) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def write(self, writer_input: WriterInput) -> WriterOutput:
        """Generate a reader-focused, evidence-grounded blog post."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": self._build_user_prompt(writer_input),
                }
            ],
        )

        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                "Writer response was truncated. Increase max_tokens "
                "or reduce the requested article length."
            )

        raw_text = "\n".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()

        if not raw_text:
            raise RuntimeError("Writer returned an empty response.")

        payload = self._parse_json(raw_text)

        try:
            result = WriterOutput.model_validate(payload)
        except ValidationError as exc:
            raise RuntimeError(
                f"Writer returned invalid output: {exc}"
            ) from exc

        actual_word_count = len(
            re.findall(r"\b[\w’'-]+\b", result.article_markdown)
        )

        # Deduplicate exact repeated warnings without changing their order.
        unresolved_items = list(dict.fromkeys(result.unresolved_items))

        result = result.model_copy(
            update={
                "word_count": actual_word_count,
                "unresolved_items": unresolved_items,
            }
        )

        # Validate evidence references against the supplied input.
        allowed_ids = {
            item.evidence_id for item in writer_input.evidence
        }

        for claim in result.claims_used:
            unknown_ids = set(claim.evidence_ids) - allowed_ids
            if unknown_ids:
                raise RuntimeError(
                    "Writer referenced unknown evidence IDs: "
                    + ", ".join(sorted(unknown_ids))
                )

        for section in result.sections:
            unknown_ids = set(section.evidence_ids_used) - allowed_ids
            if unknown_ids:
                raise RuntimeError(
                    "Writer section referenced unknown evidence IDs: "
                    + ", ".join(sorted(unknown_ids))
                )

        return result

    @staticmethod
    def _system_prompt() -> str:
        return """
You are Agent 5, the BLOG WRITER in the AImpact content workflow.

Your job is to turn the supplied content brief and evidence into a
complete, engaging, useful blog post for the intended reader.

IMPORTANT: Do not write a company profile, product catalogue, technical
report, list of capabilities, or marketing brochure unless the brief
explicitly asks for that format. The default output is a reader-focused blog.

BLOG WRITING APPROACH

1. Start with the reader, not the company.
   Open with the problem, question, decision, misconception, or challenge
   that brought the reader to this topic. Make the relevance clear.

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

5. Use natural, informative headings.
   Follow all required topics in the brief, but do not mechanically copy
   every heading if a clearer reader-focused heading preserves the same
   purpose. Do not omit required coverage.

6. Use a natural blog style.
   Write clear, connected paragraphs with varied sentence structure.
   Use lists when they improve readability, not as a substitute for
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

9. Use the supplied evidence for company-specific and other material
   factual claims. Never invent facts, figures, study results, examples,
   certifications, capabilities, customer outcomes, citations, or URLs.

10. Evidence with status "verified" may be stated as verified by the
    upstream process, but do not claim more than its quote supports.

11. Evidence with status "model_supported" is less certain. Qualify the
    claim appropriately and do not present it as independently verified.

12. Never present evidence marked "contradicted", "not_found",
    "inconclusive", or "unavailable" as established fact.

13. A source URL or evidence ID alone does not prove a claim. Check the
    supplied claim, quote, context, and status. If support is inadequate,
    qualify or omit the claim and record the limitation.

14. Do not invent evidence IDs or source URLs. Use only those supplied
    in the writer input.

15. Add inline Markdown source links for material factual claims when
    a supplied source URL supports the claim. Use the actual supplied URL.
    Never fabricate a URL or imply a source supports a claim it does not.
    Keep citations readable and avoid linking every sentence to the same
    source unnecessarily.

BRIEF AND OUTPUT REQUIREMENTS

16. Follow the brief's search intent, audience, article objective,
    required topics, style, restrictions, and target word count.

17. Cover every required section and FAQ where appropriate. Optional
    sections are allowed only when useful and consistent with the brief.

18. Do not force company names, keywords, or product names into every
    section. Use them naturally.

19. Keep the article useful even when some research questions cannot be
    answered. Do not fill gaps with speculation.

20. In unresolved_items, record important missing evidence, uncertain
    claims, and brief requirements that could not be fulfilled. Avoid
    repeating the same unresolved item.

21. claims_used must list important factual claims actually made in the
    article and their supporting evidence IDs and source URLs. Do not
    describe a claim as verified merely because it has an evidence ID.
    Use an empty evidence_ids list when no supplied evidence supports it.

22. brief_coverage must accurately describe what the article actually
    covers. Do not claim a requirement is satisfied if it is missing.

23. Return only one valid JSON object matching the requested output fields.
    Do not wrap the JSON in Markdown code fences.

The article_markdown field must contain the complete, publication-readable
blog post in Markdown, not JSON, notes, a plan, or a summary.
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

- Write an engaging, topic-focused H1 title.
- Begin with an introduction that establishes the reader's problem and
  explains why the topic matters.
- Develop the topic in a logical sequence using useful H2 and H3 headings.
- Introduce company capabilities in context instead of making the whole
  article a company description.
- Include concrete, evidence-supported details where available.
- Explain practical implications without inventing benefits or outcomes.
- End with a concise conclusion and a relevant next step if appropriate.
- Follow all required brief topics and FAQs. Do not add sections merely
  to increase length.
- Avoid a generic opening such as "{data.company_name} is a leading company"
  unless the brief specifically requires that approach and the claim is
  supported by evidence.

The blog must fit this particular topic. Do not force every article into
the same formula or use unsupported anecdotes, hypothetical customer
stories presented as real, or invented statistics.

Return a JSON object with exactly these fields:
- title: string
- meta_description: string
- article_markdown: string
- word_count: integer (approximate; it will be recalculated)
- sections: array of objects with heading, content_markdown,
  evidence_ids_used
- claims_used: array of objects with claim, evidence_ids, source_urls
- unresolved_items: array of strings
- brief_coverage: array of strings

For claims_used, record important factual claims that actually appear in
the article and their supporting evidence IDs and URLs. Use an empty
evidence_ids list if no supplied evidence supports a claim. Do not invent
supporting references.

For sections, reflect the actual H2/H3 sections in the finished article.
The article_markdown field must contain the complete Markdown article,
including its H1 title and inline source links where supported.

The meta_description should accurately describe the article in a concise,
reader-focused way. Do not add unsupported promises or claims.
"""

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
