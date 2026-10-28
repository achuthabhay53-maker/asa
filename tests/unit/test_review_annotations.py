from aca.review_annotations import anchorable_finding_indexes, render_annotated_markdown


def test_review_finding_is_highlighted_with_actual_reason_and_recommendation():
    article = "The demo cost fell by more than 80% in the example."
    findings = [
        {
            "area": "factual_claims",
            "severity": "medium",
            "issue": "The numerical claim is not attributed in the sentence.",
            "recommendation": "Add the source attribution and verify it.",
            "target_text": "more than 80%",
        }
    ]

    anchorable = anchorable_finding_indexes(article, findings)
    rendered, unmatched = render_annotated_markdown(
        article,
        findings,
        allowed_finding_indexes=anchorable,
    )

    assert anchorable == {0}
    assert unmatched == []
    assert '<span class="review-annotation review-medium"' in rendered
    assert "<mark>more than 80%</mark>" in rendered
    assert "The numerical claim is not attributed in the sentence." in rendered
    assert "Add the source attribution and verify it." in rendered


def test_ambiguous_or_missing_targets_are_not_highlighted():
    article = "A repeated claim. A repeated claim."
    findings = [
        {
            "area": "content",
            "severity": "high",
            "issue": "Needs review.",
            "recommendation": "Check the wording.",
            "target_text": "repeated claim",
        },
        {
            "area": "seo",
            "severity": "low",
            "issue": "Heading issue.",
            "recommendation": "Review the heading.",
            "target_text": "missing heading",
        },
    ]

    anchorable = anchorable_finding_indexes(article, findings)
    rendered, unmatched = render_annotated_markdown(article, findings, anchorable)

    assert anchorable == set()
    assert unmatched == [0, 1]
    assert '<span class="review-annotation ' not in rendered


def test_article_raw_html_is_escaped_before_annotation_rendering():
    rendered, _ = render_annotated_markdown("<script>alert(1)</script>", [])

    assert "&lt;script&gt;" in rendered
    assert "<script>alert(1)</script>" not in rendered


def test_verify_instruction_is_hidden_and_attached_to_the_preceding_sentence():
    article = (
        "Basic VR experiences may have shorter timelines. "
        "[VERIFY: preserve source limitation: complex projects may take longer.]"
    )

    rendered, _ = render_annotated_markdown(article, [])

    assert "[VERIFY:" not in rendered
    assert "preserve source limitation" in rendered
    assert "complex projects may take longer." in rendered
    assert (
        '<span class="verify-annotation" tabindex="0" '
        'aria-label="Source verification note:' in rendered
    )
    assert "<mark>Basic VR experiences may have shorter timelines.</mark>" in rendered
    assert "verify-annotation:hover .review-tooltip" in rendered


def test_verify_instruction_remains_available_when_review_finding_anchors_same_sentence():
    article = (
        "Basic VR experiences may have shorter timelines. "
        "[VERIFY: preserve source limitation: complex projects may take longer.]"
    )
    findings = [
        {
            "area": "content",
            "severity": "medium",
            "issue": "The limitation should be clear.",
            "recommendation": "Retain the source qualification.",
            "target_text": "shorter timelines",
        }
    ]

    rendered, _ = render_annotated_markdown(article, findings)

    assert "[VERIFY:" not in rendered
    assert "preserve source limitation" in rendered
    assert '<span class="review-annotation review-medium verify-noted"' in rendered
    assert "<b>Source note</b><br>preserve source limitation:" in rendered
