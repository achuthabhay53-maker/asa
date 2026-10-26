from aca.article_sections import split_article_sections


def test_split_article_sections_places_conclusion_before_expandable_faqs():
    article = """# Example article

Opening paragraph.

## Main section

Main details.

## Frequently Asked Questions

### First question?

First answer.

### Second question?

Second answer.

## Conclusion

Closing paragraph.
"""

    sections = split_article_sections(article)

    assert sections.introduction == "Opening paragraph."
    assert sections.main_content == "## Main section\n\nMain details."
    assert sections.conclusion == "Closing paragraph."
    assert sections.faqs == [
        ("First question?", "First answer."),
        ("Second question?", "Second answer."),
    ]


def test_split_article_sections_keeps_faq_content_when_questions_are_not_headings():
    article = """# Example article

Opening paragraph.

## Conclusion

Closing paragraph.

## FAQs

Questions and answers are provided here.
"""

    sections = split_article_sections(article)

    assert sections.conclusion == "Closing paragraph."
    assert sections.faq_fallback == "Questions and answers are provided here."
