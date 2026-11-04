"""Create a structured PDF from Agent 5 Markdown output."""

from __future__ import annotations

import html
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from xml.sax.saxutils import quoteattr

from markdown_it import MarkdownIt
import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_FONT_REGULAR = "ACAReport"
_FONT_BOLD = "ACAReport-Bold"
_FONT_ITALIC = "ACAReport-Italic"
_BLUE = colors.HexColor("#174A70")
_PALE_BLUE = colors.HexColor("#EEF4F8")
_TEXT = colors.HexColor("#263238")


def _register_fonts() -> None:
    if _FONT_REGULAR in pdfmetrics.getRegisteredFontNames():
        return
    font_dir = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont(_FONT_REGULAR, str(font_dir / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont(_FONT_BOLD, str(font_dir / "VeraBd.ttf")))
    pdfmetrics.registerFont(TTFont(_FONT_ITALIC, str(font_dir / "VeraIt.ttf")))
    pdfmetrics.registerFontFamily(
        _FONT_REGULAR,
        normal=_FONT_REGULAR,
        bold=_FONT_BOLD,
        italic=_FONT_ITALIC,
        boldItalic=_FONT_BOLD,
    )


def _styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ArticleTitle",
            parent=sample["Title"],
            fontName=_FONT_BOLD,
            fontSize=23,
            leading=29,
            textColor=_BLUE,
            spaceAfter=17,
            keepWithNext=True,
        ),
        "h1": ParagraphStyle(
            "ArticleHeading1",
            parent=sample["Heading1"],
            fontName=_FONT_BOLD,
            fontSize=19,
            leading=24,
            textColor=_BLUE,
            spaceBefore=16,
            spaceAfter=9,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "ArticleHeading2",
            parent=sample["Heading2"],
            fontName=_FONT_BOLD,
            fontSize=15,
            leading=20,
            textColor=_BLUE,
            spaceBefore=13,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "ArticleHeading3",
            parent=sample["Heading3"],
            fontName=_FONT_BOLD,
            fontSize=12,
            leading=16,
            textColor=_TEXT,
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "ArticleBody",
            parent=sample["BodyText"],
            fontName=_FONT_REGULAR,
            fontSize=10,
            leading=15,
            textColor=_TEXT,
            alignment=TA_LEFT,
            spaceAfter=8,
            splitLongWords=True,
        ),
        "quote": ParagraphStyle(
            "ArticleQuote",
            parent=sample["BodyText"],
            fontName=_FONT_ITALIC,
            fontSize=10,
            leading=15,
            textColor=colors.HexColor("#425466"),
            leftIndent=8,
            rightIndent=8,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "ArticleCode",
            parent=sample["Code"],
            fontName=_FONT_REGULAR,
            fontSize=8,
            leading=11,
            textColor=_TEXT,
            backColor=_PALE_BLUE,
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=9,
        ),
        "list": ParagraphStyle(
            "ArticleList",
            parent=sample["BodyText"],
            fontName=_FONT_REGULAR,
            fontSize=10,
            leading=15,
            textColor=_TEXT,
            spaceAfter=3,
        ),
        "table": ParagraphStyle(
            "ArticleTable",
            parent=sample["BodyText"],
            fontName=_FONT_REGULAR,
            fontSize=8,
            leading=11,
            textColor=_TEXT,
            spaceAfter=0,
        ),
        "table_header": ParagraphStyle(
            "ArticleTableHeader",
            parent=sample["BodyText"],
            fontName=_FONT_BOLD,
            fontSize=8,
            leading=11,
            textColor=colors.white,
            spaceAfter=0,
        ),
    }


def _inline_text(token: Any) -> str:
    parts: list[str] = []
    stack: list[str] = []
    for child in token.children or []:
        kind = child.type
        if kind in {"text", "html_inline"}:
            parts.append(html.escape(child.content))
        elif kind == "code_inline":
            parts.append(f'<font name="Courier" color="#174A70">{html.escape(child.content)}</font>')
        elif kind in {"softbreak", "hardbreak"}:
            parts.append("<br/>")
        elif kind in {"strong_open", "em_open", "s_open"}:
            tag = {"strong_open": "b", "em_open": "i", "s_open": "strike"}[kind]
            parts.append(f"<{tag}>")
            stack.append(tag)
        elif kind in {"strong_close", "em_close", "s_close"}:
            tag = {"strong_close": "b", "em_close": "i", "s_close": "strike"}[kind]
            if stack and stack[-1] == tag:
                stack.pop()
            parts.append(f"</{tag}>")
        elif kind == "link_open":
            href = str((child.attrs or {}).get("href", ""))
            scheme = urlparse(href).scheme.lower()
            if scheme in {"http", "https", "mailto"}:
                parts.append(f'<link href={quoteattr(href)} color="#1769AA">')
                stack.append("link")
            else:
                stack.append("")
        elif kind == "link_close":
            if stack:
                tag = stack.pop()
                if tag == "link":
                    parts.append("</link>")
        elif kind == "image":
            label = child.content or "image"
            parts.append(f"<i>[Image: {html.escape(label)}]</i>")
    while stack:
        tag = stack.pop()
        if tag:
            parts.append(f"</{tag}>")
    return "".join(parts) or " "


def _find_matching(tokens: list[Any], start: int, open_type: str, close_type: str) -> int:
    depth = 0
    for index in range(start, len(tokens)):
        if tokens[index].type == open_type:
            depth += 1
        elif tokens[index].type == close_type:
            depth -= 1
            if depth == 0:
                return index
    return len(tokens)


def _render_table(tokens: list[Any], start: int, styles: dict[str, ParagraphStyle]) -> tuple[Any, int]:
    end = _find_matching(tokens, start, "table_open", "table_close")
    rows: list[list[Paragraph]] = []
    row: list[Paragraph] | None = None
    for token_index, token in enumerate(tokens[start + 1 : end], start + 1):
        if token.type == "tr_open":
            row = []
        elif token.type in {"th_open", "td_open"} and row is not None:
            cell = next((item for item in tokens[token_index + 1 : end] if item.type == "inline"), None)
            if cell is not None:
                style = styles["table_header"] if token.type == "th_open" else styles["table"]
                row.append(Paragraph(_inline_text(cell), style))
        elif token.type == "tr_close" and row is not None:
            if row:
                rows.append(row)
            row = None
    if not rows:
        return Spacer(1, 1), end + 1

    columns = max(len(row) for row in rows)
    normalized = [row + [Paragraph(" ", styles["table"])] * (columns - len(row)) for row in rows]
    table = Table(normalized, colWidths=[170 * mm / columns] * columns, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _BLUE),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C7D1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _PALE_BLUE]),
            ]
        )
    )
    return table, end + 1


def _render_blocks(
    tokens: list[Any],
    styles: dict[str, ParagraphStyle],
    start: int = 0,
    stop_type: str | None = None,
) -> tuple[list[Any], int]:
    flowables: list[Any] = []
    index = start
    first_h1 = start == 0
    while index < len(tokens):
        token = tokens[index]
        if stop_type and token.type == stop_type:
            break
        if token.type in {"paragraph_open", "heading_open"}:
            inline = tokens[index + 1]
            if token.type == "paragraph_open":
                style = styles["body"]
            else:
                level = int(token.tag[1:])
                if level == 1 and first_h1:
                    style = styles["title"]
                    first_h1 = False
                else:
                    style = styles.get(f"h{level}", styles["h3"])
            flowables.append(Paragraph(_inline_text(inline), style))
            index += 3
        elif token.type in {"bullet_list_open", "ordered_list_open"}:
            ordered = token.type == "ordered_list_open"
            index += 1
            items: list[ListItem] = []
            while index < len(tokens) and tokens[index].type != (
                "ordered_list_close" if ordered else "bullet_list_close"
            ):
                if tokens[index].type == "list_item_open":
                    children, index = _render_blocks(tokens, styles, index + 1, stop_type="list_item_close")
                    items.append(ListItem(children or [Paragraph(" ", styles["list"])]))
                    index += 1
                else:
                    index += 1
            if items:
                flowables.append(
                    ListFlowable(
                        items,
                        bulletType="1" if ordered else "bullet",
                        start=str(token.attrGet("start") or 1),
                        leftIndent=18,
                        bulletFontName=_FONT_REGULAR,
                        bulletFontSize=9,
                        bulletColor=_BLUE,
                    )
                )
                flowables.append(Spacer(1, 4))
            index += 1
        elif token.type == "blockquote_open":
            children, index = _render_blocks(tokens, styles, index + 1, stop_type="blockquote_close")
            for child in children:
                if isinstance(child, Paragraph):
                    child.style = styles["quote"]
            if children:
                quote = Table([[children]], colWidths=[170 * mm])
                quote.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), _PALE_BLUE),
                            ("LINEBEFORE", (0, 0), (0, -1), 2, _BLUE),
                            ("LEFTPADDING", (0, 0), (-1, -1), 9),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ]
                    )
                )
                flowables.extend([quote, Spacer(1, 8)])
            index += 1
        elif token.type == "table_open":
            table, index = _render_table(tokens, index, styles)
            flowables.extend([table, Spacer(1, 10)])
        elif token.type in {"fence", "code_block"}:
            code = html.escape(token.content.rstrip()).replace("\n", "<br/>")
            flowables.extend([Paragraph(code or " ", styles["code"]), Spacer(1, 3)])
            index += 1
        elif token.type == "hr":
            flowables.extend(
                [
                    Spacer(1, 4),
                    HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#B8C7D1")),
                    Spacer(1, 8),
                ]
            )
            index += 1
        else:
            index += 1
    return flowables, index


def _page_footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setFont(_FONT_REGULAR, 8)
    canvas.setFillColor(colors.HexColor("#607D8B"))
    canvas.drawString(document.leftMargin, 11 * mm, "AImpact Content Agent")
    canvas.drawRightString(A4[0] - document.rightMargin, 11 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_article_pdf(article_markdown: str) -> bytes:
    """Render Markdown headings, paragraphs, lists, tables, and quotes to PDF."""
    if not isinstance(article_markdown, str) or not article_markdown.strip():
        raise ValueError("Article Markdown must be a non-empty string.")
    _register_fonts()
    tokens = MarkdownIt("default", {"html": False}).parse(article_markdown)
    story, _ = _render_blocks(tokens, _styles())
    if not story:
        raise ValueError("Article Markdown contains no renderable content.")

    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=19 * mm,
        title="Article",
        author="AImpact Content Agent",
        subject="Formatted article",
    )
    document.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return output.getvalue()
