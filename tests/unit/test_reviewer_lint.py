from aca.reviewer_lint import build_retry_message, count_list_blocks, lint_article


def test_count_list_blocks_keeps_blank_lines_inside_a_list_together():
    article = "- First\n\n- Second\n\nParagraph\n\n1. Third\n2. Fourth"

    assert count_list_blocks(article) == 2


def test_lint_article_reports_style_and_seo_issues():
    article = (
        "Companies face a growing challenge. This guide will explore the landscape.\n\n"
        "- **First**: use an immersive tool\n\n"
        "A short transition.\n\n"
        "- **Second**: use an immersive tool\n\n"
        "Another transition.\n\n"
        "- **Third**: use an immersive tool\n\n"
        "The immersive option is robust — it can be scalable — and measurable — too."
    )
    issues = lint_article(
        article,
        title="A guide to training",
        seo_title="x" * 61,
        meta_description="A guide to training",
    )

    assert any("em dashes" in issue for issue in issues)
    assert any("landscape" in issue for issue in issues)
    assert any("'immersive' appears 4 times" in issue for issue in issues)
    assert any("3 bulleted lists" in issue for issue in issues)
    assert any("bold label" in issue for issue in issues)
    assert any("generic statement" in issue for issue in issues)
    assert any("maximum is 60" in issue for issue in issues)
    assert any("Meta description is" in issue for issue in issues)


def test_build_retry_message_lists_all_issues():
    message = build_retry_message(["Fix the opening.", "Shorten the SEO title."])

    assert "change nothing else" in message
    assert "- Fix the opening." in message
    assert "- Shorten the SEO title." in message
