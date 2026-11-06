"""Helpers for human edits and follow-up review in the General Blog workflow."""

from __future__ import annotations

import re

from src.models.general_blog import GeneralBlogResult
from src.models.reviewer import ReviewOutput
from src.models.writer import WriterOutput, WrittenSection


def edited_writer_output(output: WriterOutput, article: str) -> WriterOutput:
    lines = article.splitlines()
    headings = [
        (index, line.strip()[3:].strip())
        for index, line in enumerate(lines)
        if re.match(r"^##\s+", line.strip())
    ]
    sections = []
    for position, (start, heading) in enumerate(headings):
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        sections.append(
            WrittenSection(
                heading=heading,
                content_markdown="\n".join(lines[start + 1 : end]).strip(),
            )
        )
    title = next(
        (line.strip()[2:].strip() for line in lines if re.match(r"^#\s+", line.strip())),
        output.title,
    )
    notes = list(output.unresolved_items)
    for note in (
        "Human-edited article; review factual claims against the supplied evidence.",
        "Review SEO title and meta description after edits.",
    ):
        if note not in notes:
            notes.append(note)
    return output.model_copy(
        update={
            "title": title,
            "article_markdown": article,
            "word_count": len(article.split()),
            "sections": sections,
            "claims_used": [],
            "unresolved_items": notes,
        }
    )


def result_after_draft_change(
    result: GeneralBlogResult,
    draft: WriterOutput,
) -> GeneralBlogResult:
    stale_review = ReviewOutput(
        summary="The draft changed after this review. Review the current version before relying on these findings."
    )
    return result.model_copy(
        update={
            "draft": draft,
            "review": stale_review,
            "review_current": False,
            "quality_passed": False,
        }
    )


def review_passed(review: ReviewOutput) -> bool:
    return not review.unresolved_items and not any(
        finding.severity in {"critical", "high"} or finding.area == "factual_claims"
        for finding in review.findings
    )
