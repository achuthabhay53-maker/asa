import pytest
from markdown_it import MarkdownIt
from reportlab.platypus import ListFlowable, Paragraph, Table

from aca.pdf_export import _render_blocks, _styles, build_article_pdf


def test_build_article_pdf_preserves_markdown_structure_and_content():
    article = """# Product overview

**Example Bio** provides a useful product for researchers.

## Product options

- First option
- Second option

| Feature | Detail |
| --- | --- |
| Format | Example |

> Evidence should be described accurately.

1. Review the source.
2. Confirm the claim.

`inline code` and [source link](https://example.com).
"""

    pdf = build_article_pdf(article)

    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 1_000
    blocks, _ = _render_blocks(MarkdownIt("default", {"html": False}).parse(article), _styles())
    assert sum(isinstance(block, Paragraph) for block in blocks) >= 4
    assert sum(isinstance(block, ListFlowable) for block in blocks) == 2
    assert sum(isinstance(block, Table) for block in blocks) == 2
    assert any("Product overview" in block.text for block in blocks if isinstance(block, Paragraph))


def test_build_article_pdf_rejects_empty_markdown():
    with pytest.raises(ValueError, match="non-empty"):
        build_article_pdf(" \n ")
