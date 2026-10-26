"""Render a Markdown revision with changed words highlighted."""

from __future__ import annotations

import difflib
import html
import re
from html.parser import HTMLParser

from markdown_it import MarkdownIt

_TOKEN = re.compile(r"\w+|[^\w\s]+|\s+", re.UNICODE)
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


class _TextCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "br"}:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "br"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


class _HighlightRenderer(HTMLParser):
    def __init__(self, changed_ranges: list[tuple[int, int]]) -> None:
        super().__init__(convert_charrefs=True)
        self.changed_ranges = changed_ranges
        self.text_offset = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "br"}:
            self.text_offset += 1
        rendered_attrs = "".join(f' {name}="{html.escape(value or "", quote=True)}"' for name, value in attrs)
        self.parts.append(f"<{tag}{rendered_attrs}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:
            self.parts.append(f"</{tag}>")

    def handle_endtag(self, tag: str) -> None:
        self.parts.append(f"</{tag}>")
        if tag in {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "br"}:
            self.text_offset += 1

    def handle_data(self, data: str) -> None:
        end = self.text_offset + len(data)
        local_ranges = [
            (max(start, self.text_offset) - self.text_offset, min(stop, end) - self.text_offset)
            for start, stop in self.changed_ranges
            if start < end and stop > self.text_offset
        ]
        cursor = 0
        for start, stop in local_ranges:
            if start > cursor:
                self.parts.append(html.escape(data[cursor:start]))
            self.parts.append(
                '<span style="color: #16803c; font-weight: 600">' + html.escape(data[start:stop]) + "</span>"
            )
            cursor = stop
        if cursor < len(data):
            self.parts.append(html.escape(data[cursor:]))
        self.text_offset = end

    def handle_entityref(self, name: str) -> None:
        self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.parts.append(f"&#{name};")


def render_markdown_diff(original: str, revised: str) -> str:
    """Render revised Markdown and color inserted/replaced text green."""
    renderer = MarkdownIt("default", {"html": False})
    original_html = renderer.render(original)
    revised_html = renderer.render(revised)

    original_text = _visible_text(original_html)
    revised_text = _visible_text(revised_html)
    original_tokens = _TOKEN.findall(original_text)
    revised_tokens = _TOKEN.findall(revised_text)

    matcher = difflib.SequenceMatcher(a=original_tokens, b=revised_tokens, autojunk=False)
    token_offsets: list[tuple[int, int]] = []
    position = 0
    for token in revised_tokens:
        token_offsets.append((position, position + len(token)))
        position += len(token)

    changed_ranges: list[tuple[int, int]] = []
    for operation, _start_a, _end_a, start_b, end_b in matcher.get_opcodes():
        if operation not in {"insert", "replace"}:
            continue
        changed_ranges.extend(
            token_offsets[index] for index in range(start_b, end_b) if not revised_tokens[index].isspace()
        )

    highlighter = _HighlightRenderer(changed_ranges)
    highlighter.feed(revised_html)
    highlighter.close()
    return "".join(highlighter.parts)


def _visible_text(rendered_html: str) -> str:
    collector = _TextCollector()
    collector.feed(rendered_html)
    collector.close()
    return "".join(collector.parts)
