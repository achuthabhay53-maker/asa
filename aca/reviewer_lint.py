"""Deterministic style checks for Agent 5 output."""

from __future__ import annotations

import re

BANNED_PHRASES = [
    "leverage",
    "utilize",
    "facilitate",
    "empower",
    "streamline",
    "robust",
    "seamless",
    "cutting-edge",
    "game-changing",
    "landscape",
    "holistic",
    "comprehensive",
    "near-real",
    "life-like",
    "with precision",
    "fast-paced world",
    "this means that",
    "it is important to note",
    "plays a crucial role",
    "stems from",
    "when it comes to",
    "not only",
]

# These words occur in source material, so allow limited use instead of banning them.
LIMITED_WORDS = {"immersive": 2, "scalable": 2, "measurable": 1}

OPENER_PATTERNS = [
    r"\bface[s]?\s+(a\s+)?(persistent|growing|significant|common|major)?\s*challenge",
    r"\bthis\s+(guide|article|post)\s+(will\s+)?(explore|examine|explain|cover)",
]

LIST_ITEM = re.compile(r"^\s*([-*]|\d+\.)\s+")
BOLD_LABEL = re.compile(r"^\s*(?:[-*]\s+)?\*\*[^*\n]{2,60}\*\*\s*[:\u2013\u2014-]", re.MULTILINE)


def count_list_blocks(article: str) -> int:
    """Count separate bulleted/numbered lists. Blank lines inside a list don't split it."""
    blocks, in_list = 0, False
    for line in article.splitlines():
        if not line.strip():
            continue
        is_item = bool(LIST_ITEM.match(line))
        if is_item and not in_list:
            blocks += 1
        in_list = is_item
    return blocks


def lint_article(
    article: str,
    title: str = "",
    seo_title: str = "",
    meta_description: str = "",
    max_dashes: int = 2,
    max_lists: int = 2,
) -> list[str]:
    issues: list[str] = []
    lowered = article.lower()

    dashes = article.count("\u2014") + len(re.findall(r"\s\u2013\s", article))
    if dashes > max_dashes:
        issues.append(
            f"{dashes} em dashes or spaced en dashes found; the maximum is {max_dashes}. "
            "Replace the rest with periods or commas."
        )

    hits = sorted({phrase for phrase in BANNED_PHRASES if phrase in lowered})
    if hits:
        issues.append("Remove or replace these words and phrases: " + ", ".join(hits) + ".")

    for word, limit in LIMITED_WORDS.items():
        used = len(re.findall(rf"\b{word}\b", lowered))
        if used > limit:
            issues.append(f"'{word}' appears {used} times; use it at most {limit}. Use plain facts instead.")

    lists = count_list_blocks(article)
    if lists > max_lists:
        issues.append(f"{lists} bulleted lists found; the maximum is {max_lists}. Turn the rest into prose.")

    labels = len(BOLD_LABEL.findall(article))
    if labels:
        issues.append(f"{labels} items start with a bold label. Remove the bold labels.")

    opening = lowered[:700]
    if any(re.search(pattern, opening) for pattern in OPENER_PATTERNS):
        issues.append("The opening is a generic statement. Start with a concrete situation or the direct answer.")

    confirms = len(
        re.findall(
            r"\b(confirm|confirmed|discuss|discussed|clarify|clarified)\b[^.]{0,40}\b(directly )?(with|during)\b",
            lowered,
        )
    )
    if confirms > 1:
        issues.append(f"'Confirm/discuss with the vendor' appears {confirms} times; use it at most once.")

    if seo_title and len(seo_title) > 60:
        issues.append(f"SEO title is {len(seo_title)} characters; the maximum is 60.")

    if meta_description:
        length = len(meta_description)
        if length > 155 or length < 120:
            issues.append(f"Meta description is {length} characters; it should be 120 to 155.")
        if title and meta_description.lower().startswith(title.lower()[:40]):
            issues.append("Meta description repeats the title. Write a one-sentence summary of the post.")

    return issues


def build_retry_message(issues: list[str]) -> str:
    bullets = "\n".join(f"- {issue}" for issue in issues)
    return (
        "A style check found these problems in your blog. Fix every one, change "
        "nothing else, and return the same JSON format with the complete article_markdown.\n"
        + bullets
    )
