"""Helpers for presenting generated Markdown articles in the Streamlit UI."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ArticleSections:
    introduction: str
    main_content: str
    conclusion: str
    faq_intro: str
    faqs: list[tuple[str, str]]
    faq_fallback: str


_H2 = re.compile(r"^##\s+(.+?)\s*#*\s*$")
_H3_OR_LOWER = re.compile(r"^#{3,6}\s+(.+?)\s*#*\s*$")


def split_article_sections(markdown: str) -> ArticleSections:
    """Separate article parts so the conclusion precedes expandable FAQs."""
    lines = markdown.strip().splitlines()
    if lines and re.match(r"^#\s+", lines[0]):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines.pop(0)

    h2_positions = [
        (index, match.group(1).strip())
        for index, line in enumerate(lines)
        if (match := _H2.match(line))
    ]
    if not h2_positions:
        return ArticleSections(
            introduction="\n".join(lines).strip(),
            main_content="",
            conclusion="",
            faq_intro="",
            faqs=[],
            faq_fallback="",
        )

    introduction = "\n".join(lines[:h2_positions[0][0]]).strip()
    main_sections: list[str] = []
    conclusion = ""
    faq_intro = ""
    faqs: list[tuple[str, str]] = []
    faq_fallback = ""

    for section_index, (start, heading) in enumerate(h2_positions):
        end = (
            h2_positions[section_index + 1][0]
            if section_index + 1 < len(h2_positions)
            else len(lines)
        )
        content = lines[start + 1:end]
        normalized_heading = heading.casefold().strip(" :")

        if normalized_heading in {"faq", "faqs", "frequently asked questions"}:
            question_positions = [
                (index, match.group(1).strip())
                for index, line in enumerate(content)
                if (match := _H3_OR_LOWER.match(line))
            ]
            if not question_positions:
                faq_fallback = "\n".join(content).strip()
                continue

            faq_intro = "\n".join(content[:question_positions[0][0]]).strip()
            for question_index, (question_start, question) in enumerate(question_positions):
                question_end = (
                    question_positions[question_index + 1][0]
                    if question_index + 1 < len(question_positions)
                    else len(content)
                )
                answer = "\n".join(content[question_start + 1:question_end]).strip()
                faqs.append((question, answer))
        elif normalized_heading.startswith("conclusion"):
            conclusion = "\n".join(content).strip()
        else:
            section = f"## {heading}"
            body = "\n".join(content).strip()
            main_sections.append(f"{section}\n\n{body}" if body else section)

    return ArticleSections(
        introduction=introduction,
        main_content="\n\n".join(main_sections).strip(),
        conclusion=conclusion,
        faq_intro=faq_intro,
        faqs=faqs,
        faq_fallback=faq_fallback,
    )
