"""Render article Markdown with safely anchored reviewer annotations."""

from __future__ import annotations

import copy
import html
import re
from collections import defaultdict
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.token import Token

_VERIFY_PATTERN = re.compile(r"\[VERIFY:[^\]]*\]")
_SEVERITY_CLASS = {
    "critical": "review-critical",
    "high": "review-high",
    "medium": "review-medium",
    "low": "review-low",
}


def _finding_value(finding: Any, key: str) -> Any:
    if isinstance(finding, dict):
        return finding.get(key)
    return getattr(finding, key, None)


def _occurrences(text: str, target: str) -> int:
    count = 0
    start = 0
    while (position := text.find(target, start)) != -1:
        count += 1
        start = position + 1
    return count


def _verify_anchor(content: str, marker_start: int) -> tuple[int, int] | None:
    prefix = content[:marker_start]
    matches = [
        match
        for match in re.finditer(r"[^.!?\n]+[.!?]?", prefix)
        if match.group().strip()
    ]
    if not matches:
        return None
    last = matches[-1]
    start = last.start()
    while start < last.end() and prefix[start].isspace():
        start += 1
    end = last.end()
    while end > start and prefix[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _html_token(content: str) -> Token:
    return Token(
        type="html_inline",
        tag="",
        nesting=0,
        attrs=None,
        map=None,
        level=0,
        children=None,
        content=content,
        markup="",
        info="",
        meta={},
        block=False,
        hidden=False,
    )


def anchorable_finding_indexes(article: str, findings: list[Any]) -> set[int]:
    """Find findings whose exact target occurs once in a plain Markdown text token."""
    parser = MarkdownIt("default", {"html": False})
    text_tokens = [
        child
        for token in parser.parse(article)
        if token.type == "inline"
        for child in token.children or []
        if child.type == "text"
    ]
    anchorable: set[int] = set()
    for finding_index, finding in enumerate(findings):
        target = _finding_value(finding, "target_text")
        if not isinstance(target, str) or not target.strip():
            continue
        target = target.strip()
        count = sum(_occurrences(token.content, target) for token in text_tokens)
        if count == 1:
            anchorable.add(finding_index)
    return anchorable


def render_annotated_markdown(
    article: str,
    findings: list[Any],
    allowed_finding_indexes: set[int] | None = None,
) -> tuple[str, list[int]]:
    """Return sanitized article HTML and indexes of findings that could not be anchored."""
    parser = MarkdownIt("default", {"html": False})
    tokens = parser.parse(article)
    text_tokens: list[Token] = [
        child
        for token in tokens
        if token.type == "inline"
        for child in token.children or []
        if child.type == "text"
    ]

    occurrences: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for finding_index, finding in enumerate(findings):
        if allowed_finding_indexes is not None and finding_index not in allowed_finding_indexes:
            continue
        target = _finding_value(finding, "target_text")
        if not isinstance(target, str) or not target.strip():
            continue
        target = target.strip()
        for token_index, token in enumerate(text_tokens):
            start = 0
            while (position := token.content.find(target, start)) != -1:
                occurrences[finding_index].append(
                    (token_index, position, position + len(target))
                )
                start = position + 1

    candidates: list[tuple[int, int, int, int]] = []
    unmatched: set[int] = set(range(len(findings)))
    for finding_index, matches in occurrences.items():
        if len(matches) == 1:
            token_index, start, end = matches[0]
            candidates.append((token_index, start, end, finding_index))

    conflicting: set[int] = set()
    for index, (token_index, start, end, finding_index) in enumerate(candidates):
        for other_token, other_start, other_end, other_finding in candidates[index + 1 :]:
            if token_index == other_token and start < other_end and other_start < end:
                conflicting.update((finding_index, other_finding))

    candidates = [
        item for item in candidates if item[3] not in conflicting
    ]
    for _, _, _, finding_index in candidates:
        unmatched.discard(finding_index)

    matches_by_token: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for token_index, start, end, finding_index in candidates:
        matches_by_token[token_index].append((start, end, finding_index))

    verify_notes_by_finding: dict[tuple[int, int], list[str]] = defaultdict(list)
    text_token_index = 0
    for token in tokens:
        if token.type != "inline":
            continue
        children: list[Token] = []
        for child in token.children or []:
            if child.type != "text":
                children.append(child)
                continue

            finding_decorations = [
                (start, end, "finding", finding_index, None)
                for start, end, finding_index in matches_by_token.get(text_token_index, [])
            ]
            verify_decorations: list[tuple[int, int, str, int | None, str | None]] = []
            marker_removals: list[tuple[int, int, str, int | None, str | None]] = []
            for match in _VERIFY_PATTERN.finditer(child.content):
                marker_removals.append((match.start(), match.end(), "remove", None, None))
                anchor = _verify_anchor(child.content, match.start())
                note = match.group()[8:-1].strip()
                overlapping_findings = [
                    finding_index
                    for start, end, _, finding_index, _ in finding_decorations
                    if anchor and anchor[0] < end and start < anchor[1]
                ]
                if overlapping_findings:
                    verify_notes_by_finding[
                        (text_token_index, overlapping_findings[0])
                    ].append(note)
                elif anchor:
                    verify_decorations.append(
                        (anchor[0], anchor[1], "verify", None, note)
                    )
            decorations = finding_decorations + verify_decorations + marker_removals
            text_token_index += 1
            decorations.sort(key=lambda item: (item[0], item[1]))
            cursor = 0
            for start, end, kind, finding_index, note in decorations:
                if start < cursor:
                    continue
                if start > cursor:
                    plain = copy.copy(child)
                    plain.content = child.content[cursor:start]
                    children.append(plain)
                if kind == "remove":
                    cursor = end
                    continue
                if kind == "verify":
                    tooltip = html.escape(note or "Source verification note")
                    children.append(
                        _html_token(
                            '<span class="verify-annotation" tabindex="0" '
                            f'aria-label="Source verification note: {tooltip}"><mark>'
                        )
                    )
                    marked = copy.copy(child)
                    marked.content = child.content[start:end]
                    children.append(marked)
                    children.append(
                        _html_token(
                            '</mark><span class="review-tooltip"><strong>Source note</strong>'
                            f'<span>{tooltip}</span></span></span>'
                        )
                    )
                else:
                    assert finding_index is not None
                    finding = findings[finding_index]
                    severity = str(_finding_value(finding, "severity") or "medium").lower()
                    css_class = _SEVERITY_CLASS.get(severity, "review-medium")
                    area = str(_finding_value(finding, "area") or "review").replace("_", " ")
                    issue = str(_finding_value(finding, "issue") or "")
                    recommendation = str(_finding_value(finding, "recommendation") or "")
                    source_notes = verify_notes_by_finding.get(
                        (text_token_index - 1, finding_index), []
                    )
                    source_note_tooltip = "".join(
                        f'<span><b>Source note</b><br>{html.escape(note)}</span>'
                        for note in source_notes
                    )
                    source_note_class = " verify-noted" if source_notes else ""
                    label = html.escape(
                        f"{severity.title()} severity · {area} · {issue}",
                        quote=True,
                    )
                    children.append(
                        _html_token(
                            f'<span class="review-annotation {css_class}{source_note_class}" tabindex="0" '
                            f'aria-label="{label}"><mark>'
                        )
                    )
                    marked = copy.copy(child)
                    marked.content = child.content[start:end]
                    children.append(marked)
                    children.append(
                        _html_token(
                            '</mark><span class="review-tooltip">'
                            f'<strong>{html.escape(severity.title())} · '
                            f'{html.escape(area)} · Finding {finding_index + 1}</strong>'
                            f'<span><b>Reason</b><br>{html.escape(issue)}</span>'
                            f'<span><b>Reviewer recommendation</b><br>'
                            f'{html.escape(recommendation)}</span>'
                            f'{source_note_tooltip}</span></span>'
                        )
                    )
                cursor = end
            if cursor < len(child.content):
                plain = copy.copy(child)
                plain.content = child.content[cursor:]
                children.append(plain)
        token.children = children

    rendered = parser.renderer.render(tokens, parser.options, {})
    styles = """
<style>
.review-article {font-size:1.05rem; line-height:1.8; overflow:visible;}
.review-article h2 {font-size:1.55rem; line-height:1.3; margin-top:1.8rem; padding-bottom:.4rem; border-bottom:1px solid rgba(128,128,128,.25);}
.review-article h3 {font-size:1.2rem; margin-top:1.2rem;}
.review-article table {width:100%; font-size:.95rem; border-collapse:collapse;}
.review-article th,.review-article td {padding:.45rem; border:1px solid rgba(128,128,128,.25);}
.review-article th {background:rgba(128,128,128,.12);}
.review-annotation,.verify-annotation {position:relative; cursor:help; outline:none;}
.review-annotation mark,.verify-annotation mark {border-radius:.18rem; padding:.04rem .1rem; color:inherit;}
.review-critical mark,.review-high mark {background:rgba(220,38,38,.22); border-bottom:2px solid #dc2626;}
.review-medium mark {background:rgba(234,120,20,.2); border-bottom:2px solid #ea7814;}
.review-low mark {background:rgba(234,179,8,.22); border-bottom:2px solid #d4a600;}
.review-annotation.verify-noted mark {box-shadow:0 0 0 2px #d4a600;}
.review-tooltip {display:none; position:absolute; z-index:1000; left:0; bottom:calc(100% + .45rem); width:min(340px,80vw); padding:.75rem; border:1px solid rgba(128,128,128,.45); border-radius:.5rem; background:var(--background-color, white); color:var(--text-color, #222); box-shadow:0 4px 18px rgba(0,0,0,.2); font-size:.9rem; line-height:1.45; white-space:normal;}
.review-tooltip span {display:block; margin-top:.5rem;}
.review-annotation:hover .review-tooltip,.review-annotation:focus .review-tooltip,.review-annotation:focus-within .review-tooltip,
.verify-annotation:hover .review-tooltip,.verify-annotation:focus .review-tooltip,.verify-annotation:focus-within .review-tooltip {display:block;}
.verify-annotation mark {background:rgba(234,179,8,.22); border-bottom:2px solid #d4a600;}
</style>
"""
    return f'{styles}<div class="review-article">{rendered}</div>', sorted(unmatched)
