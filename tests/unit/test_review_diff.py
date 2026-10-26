from aca.review_diff import render_markdown_diff


def test_highlights_changed_text_while_preserving_markdown_rendering():
    original = "# Draft\n\nThe system supports training.\n"
    revised = "# Revised Draft\n\nThe system supports interactive training.\n"

    rendered = render_markdown_diff(original, revised)

    assert "<h1>" in rendered
    assert "<p>" in rendered
    assert '<span style="color: #16803c; font-weight: 600">Revised</span>' in rendered
    assert '<span style="color: #16803c; font-weight: 600">interactive</span>' in rendered
    assert "Draft</span>" not in rendered


def test_does_not_interpret_raw_html_from_markdown_as_active_html():
    rendered = render_markdown_diff(
        "Safe text.",
        "<script>alert('x')</script>\n\nSafe text.",
    )

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered.replace(
        '<span style="color: #16803c; font-weight: 600">', ""
    ).replace("</span>", "")
