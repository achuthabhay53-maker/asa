import json
from types import SimpleNamespace
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agents.aimpact_blog.writer import WriterAgent
from src.agents.aimpact_blog.reviewer import ReviewerAgent
from src.models.reviewer import ReviewerResponse
from src.models.writer import (
    WriterClaim,
    WriterEvidence,
    WriterInput,
    WriterOutput,
    WriterSectionInput,
    WrittenSection,
)
from src.ui.review_annotations import render_annotated_markdown


_ARTICLE = """# VR Training for Medical Device Teams

VR training can give medical device teams a way to practice procedures through
interactive simulations. The supplied evidence describes virtual training
for medical device teams and procedural practice.

## How the training works

Learners practice procedures through interactive simulation.

| Consideration | What the supplied evidence says |
| --- | --- |
| Training approach | Immersive training supports procedural practice. |

## Frequently Asked Questions

### What is VR training used for?

It supports immersive training for medical device teams and lets learners
practice procedures through interactive simulation.

### How do learners practice?

Learners practice procedures through interactive simulation.

### Who can use this training?

Medical device teams can use VR-based training to practice procedures.
"""


def _writer_input() -> WriterInput:
    return WriterInput(
        company_name="Example",
        title="VR Training for Medical Device Teams",
        article_objective="Explain VR training",
        target_audience="Medical device teams",
        primary_keyword="VR training for medical devices",
        secondary_keywords=["immersive training"],
        related_topics=["procedural simulation"],
        question_based_queries=["How does VR training work?"],
        faqs=[{"question": "How do learners practice?", "answer": ""}],
        evidence=[
            WriterEvidence(
                evidence_id="ev_001",
                claim="Immersive training for medical device teams",
                status="verified",
                evidence_quote=(
                    "Immersive training for medical device teams enables learners "
                    "to practice procedures through interactive simulation."
                ),
                source_url="https://example.com/training",
            )
        ],
    )


def test_independent_writer_example_has_faq_and_enough_synthetic_evidence():
    example_path = Path(__file__).resolve().parents[2] / "scripts" / "agent5_input.example.json"
    writer_input = WriterInput.model_validate_json(example_path.read_text(encoding="utf-8"))

    assert len(writer_input.faqs) == 3
    assert len(writer_input.evidence) >= 3
    assert all(item.evidence_quote for item in writer_input.evidence)


def test_reviewer_removes_unapproved_source_urls_and_flags_claims():
    from tests.agent_fixtures import agent5_output

    draft = agent5_output().model_copy(
        update={
            "article_markdown": (
                "# Draft\n\n[Approved](https://example.com/atlas-panel) "
                "[Unknown](https://not-approved.example/page)"
            ),
            "claims_used": [
                WriterClaim(
                    claim="Example Bio provides an unsupported feature.",
                    claim_scope="company_specific",
                    evidence_ids=["ev-001", "fabricated-id"],
                    source_urls=[
                        "https://example.com/atlas-panel",
                        "https://not-approved.example/page",
                    ],
                    support_status="unsupported",
                )
            ],
        }
    )
    evidence = [
        WriterEvidence(
            evidence_id="ev-001",
            claim="Approved product fact",
            status="verified",
            evidence_quote="Example Bio's Atlas panel quantifies 120 immune-related proteins in serum samples.",
            source_url="https://example.com/atlas-panel",
        )
    ]

    sanitized = ReviewerAgent._sanitize_revised_blog(draft, evidence)

    assert "[Approved](https://example.com/atlas-panel)" in sanitized.article_markdown
    assert "Unknown" in sanitized.article_markdown
    assert "not-approved.example" not in sanitized.article_markdown
    assert sanitized.claims_used[0].evidence_ids == ["ev-001"]
    assert sanitized.claims_used[0].support_status == "unsupported"
    assert any("needs human verification" in item for item in sanitized.unresolved_items)


def test_reviewer_returns_findings_without_rewriting_the_draft():
    draft = _writer_output()
    boundary = "The result applies only to the reported pilot."
    draft = draft.model_copy(
        update={
            "unresolved_items": [
                f"Preserve evidence limitation for claim {draft.claims_used[0].claim!r}: {boundary}",
                "A separate note needs human review.",
            ]
        }
    )
    finding = {
        "area": "grammar",
        "severity": "medium",
        "issue": "The opening paragraph repeats a phrase.",
        "recommendation": "Have a human edit the repeated phrase.",
        "target_text": None,
    }
    response = SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(
                type="tool_use",
                name="submit_review",
                input={
                    "summary": "One writing issue needs human review.",
                    "findings": [finding],
                    "unresolved_items": [],
                },
            )
        ],
    )
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return response

    reviewer = ReviewerAgent.__new__(ReviewerAgent)
    reviewer.model = "test"
    reviewer.max_tokens = 4096
    reviewer.client = SimpleNamespace(messages=SimpleNamespace(create=create))

    writer_input = _writer_input().model_copy(
        update={
            "evidence": [
                _writer_input().evidence[0].model_copy(update={"wording_boundary": boundary}),
                WriterEvidence(
                    evidence_id="ev_002",
                    claim="Uncited background source",
                    status="verified",
                    evidence_quote="This source is not cited by the draft.",
                    source_url="https://example.com/background",
                ),
            ]
        }
    )
    result = reviewer.review(writer_input, draft)

    assert result.summary == "One writing issue needs human review."
    assert [item.model_dump() for item in result.findings] == [{**finding, "target_text": None}]
    assert result.unresolved_items == []
    assert draft.article_markdown == _writer_output().article_markdown
    assert "revised_blog" not in result.model_dump()
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 4096
    assert calls[0]["tool_choice"] == {"type": "tool", "name": "submit_review"}
    assert calls[0]["tools"][0]["input_schema"]["required"] == [
        "summary",
        "findings",
        "unresolved_items",
    ]
    request = json.loads(calls[0]["messages"][0]["content"])
    assert "article_markdown" in request["draft"]
    assert "sections" in request
    assert [item["evidence_id"] for item in request["approved_evidence"]] == ["ev_001"]
    assert request["draft_unresolved_items"] == ["A separate note needs human review."]
    assert set(request["draft"]["claims_used"][0]) == {
        "claim",
        "claim_scope",
        "evidence_ids",
        "support_status",
    }
    assert "do not rewrite" in calls[0]["system"].lower()
    assert "article_markdown" not in calls[0]["system"]


def test_reviewer_normalizes_empty_meta_description_without_failing_review():
    blog = _writer_output().model_copy(update={"meta_description": "   "})

    normalized = ReviewerAgent._normalize_revised_metadata(blog)

    assert normalized.meta_description == blog.title
    assert any("empty meta description" in item for item in normalized.unresolved_items)


def test_reviewer_reports_when_full_blog_response_is_truncated():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stop_reason="max_tokens", content=[])

    reviewer = ReviewerAgent.__new__(ReviewerAgent)
    reviewer.model = "test"
    reviewer.max_tokens = 4096
    reviewer.client = SimpleNamespace(messages=SimpleNamespace(create=create))

    with pytest.raises(RuntimeError, match="truncated at 4,096 output tokens"):
        reviewer.review(_writer_input(), _writer_output())

    assert len(calls) == 1


def test_reviewer_rebuilds_writer_sections_from_compact_markdown_response():
    from src.models.reviewer import RevisedBlog

    draft = _writer_output()
    revision = RevisedBlog(
        title=draft.title,
        seo_title=draft.seo_title,
        meta_description=draft.meta_description,
        article_markdown=draft.article_markdown,
        claims_used=draft.claims_used,
        unresolved_items=draft.unresolved_items,
        brief_coverage=draft.brief_coverage,
    )

    rebuilt = ReviewerAgent._build_writer_output(draft, revision)

    assert rebuilt.article_markdown == draft.article_markdown
    assert [section.heading for section in rebuilt.sections] == [
        "How the training works",
        "Frequently Asked Questions",
    ]
    assert all(section.content_markdown for section in rebuilt.sections)


def _writer_output(
    article: str = _ARTICLE,
    title: str = "VR Training for Medical Device Teams",
) -> WriterOutput:
    return WriterOutput(
        title=title,
        seo_title=title[:60],
        meta_description=(
            "Learn how medical device teams use virtual reality training to practice "
            "procedures, assess training options, and choose an approach that fits their needs."
        ),
        article_markdown=article,
        word_count=75,
        sections=[
            WrittenSection(
                heading="Frequently Asked Questions",
                content_markdown="Three evidence-backed answers.",
                evidence_ids_used=["ev_001"],
            )
        ],
        claims_used=[
            WriterClaim(
                claim=(
                    "Immersive training for medical device teams lets learners "
                    "practice procedures through interactive simulation."
                ),
                evidence_ids=["ev_001"],
                source_urls=["https://example.com/training"],
            )
        ],
    )


class _FakeMessageStream:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def get_final_message(self):
        return self.response


def test_writer_requires_seo_and_faq_structure():
    WriterAgent._validate_seo_requirements(_writer_output(), _writer_input())


def test_writer_does_not_block_faqs_on_exact_evidence_word_overlap():
    writer_input = _writer_input().model_copy(
        update={
            "evidence": [
                WriterEvidence(
                    evidence_id="ev_001",
                    claim="The company was founded in a particular year",
                    status="verified",
                    evidence_quote="The company was founded in a particular year.",
                    source_url="https://example.com/company",
                )
            ]
        }
    )

    WriterAgent._validate_seo_requirements(_writer_output(), writer_input)


def test_writer_accepts_paraphrased_faq_answer_supported_by_article_and_evidence():
    paraphrased = _ARTICLE.replace(
        "Learners practice procedures through interactive simulation.",
        "Learners rehearse procedures in simulated environments.",
        1,
    )
    WriterAgent._validate_seo_requirements(_writer_output(paraphrased), _writer_input())


def test_writer_accepts_five_faqs():
    five_faqs = (
        _ARTICLE
        + """
### What do learners practice?

Learners practice procedures in interactive simulation.

### How does immersive training help?

It gives medical device teams a simulated way to practice procedures.
"""
    )
    WriterAgent._validate_seo_requirements(_writer_output(five_faqs), _writer_input())


def test_writer_flags_numerical_claim_without_verified_evidence():
    result = _writer_output(
        _ARTICLE.replace(
            "Learners practice procedures through interactive simulation.",
            "The program delivers an 80% cost reduction.",
            1,
        )
    )

    audited = WriterAgent._flag_unsupported_numeric_claims(result, _writer_input().evidence)

    assert "[VERIFY:" not in audited.article_markdown
    assert any("80%" in item for item in audited.unresolved_items)


def test_writer_accepts_numerical_claim_linked_to_verified_quote_and_context():
    evidence = WriterEvidence(
        evidence_id="ev_metric",
        claim="An 80% cost reduction was measured in the pilot workflow.",
        status="verified",
        evidence_quote=("The pilot workflow reported an 80% cost reduction in measured costs."),
        source_url="https://example.com/results",
    )
    article = _writer_output("# Results\n\nThe pilot workflow reported an 80% cost reduction.").model_copy(
        update={
            "claims_used": [
                WriterClaim(
                    claim="The pilot workflow reported an 80% cost reduction.",
                    evidence_ids=["ev_metric"],
                )
            ]
        }
    )

    audited = WriterAgent._flag_unsupported_numeric_claims(article, [evidence])

    assert "[VERIFY:" not in audited.article_markdown
    assert not audited.unresolved_items


def test_writer_marks_supported_metric_with_source_limitation():
    evidence = WriterEvidence(
        evidence_id="ev_metric",
        claim="The pilot reported an 80% cost reduction.",
        status="verified",
        evidence_quote="The pilot reported an 80% cost reduction.",
        source_url="https://example.com/results",
        wording_boundary="This is a company-reported claim, not independently verified.",
    )
    result = _writer_output("# Results\n\nThe pilot reported an 80% cost reduction.").model_copy(
        update={
            "claims_used": [
                WriterClaim(
                    claim="The pilot reported an 80% cost reduction.",
                    evidence_ids=["ev_metric"],
                )
            ]
        }
    )

    audited = WriterAgent._flag_unsupported_numeric_claims(result, [evidence])

    assert "[VERIFY:" not in audited.article_markdown
    assert "The pilot reported an 80% cost reduction." in audited.article_markdown
    assert any("not independently verified" in item for item in audited.unresolved_items)


def test_writer_adds_linked_references_after_faq_and_converts_evidence_ids():
    evidence = [
        WriterEvidence(
            evidence_id="ev_001",
            claim="Nineteen of 21 studies showed increased competency.",
            status="verified",
            evidence_quote="19 of 21 studies showed increased competency.",
            source_url="https://papers.example/review",
            source_title="A Review of Training Outcomes",
            source_author="A. Researcher",
            journal="Journal of Training",
            doi="10.1234/vr.2024",
            publication_date="2024-06-01",
        )
    ]
    article = _writer_output(
        _ARTICLE.replace(
            "The supplied evidence describes virtual training",
            "[ev_001] The supplied evidence describes virtual training",
        )
    ).model_copy(
        update={
            "claims_used": [
                WriterClaim(
                    claim="Nineteen of 21 studies showed increased competency.",
                    evidence_ids=["ev_001"],
                )
            ]
        }
    )

    referenced = WriterAgent._append_references(article, evidence)

    assert "[ev_001]" not in referenced.article_markdown
    assert (
        "[\\[1\\]](#reference-1) The supplied evidence"
        in referenced.article_markdown
    )
    assert referenced.article_markdown.index("## Frequently Asked Questions") < (
        referenced.article_markdown.index("## References")
    )
    assert (
        "1. [A. Researcher. A Review of Training Outcomes. Journal of Training](https://papers.example/review) (2024). DOI: 10.1234/vr.2024."
        in referenced.article_markdown
    )
    rendered, _ = render_annotated_markdown(referenced.article_markdown, [])
    assert '<a href="#reference-1">[1]</a>' in rendered
    assert '<li id="reference-1">' in rendered
    assert '<a href="https://papers.example/review">A. Researcher. A Review of Training Outcomes. Journal of Training</a>' in rendered
    assert "10.1234/vr.2024" in referenced.article_markdown


def test_writer_deduplicates_sources_and_strips_old_references_before_rebuilding():
    evidence = [
        WriterEvidence(
            evidence_id="ev_001",
            claim="First claim",
            status="verified",
            source_url="https://papers.example/review#section",
            source_title="Research review",
        ),
        WriterEvidence(
            evidence_id="ev_002",
            claim="Second claim",
            status="verified",
            source_url="https://papers.example/review",
            source_title="Research review",
        ),
    ]
    article = _writer_output(
        "# Results\n\n[ev_001] First claim. [ev_002] Second claim.\n\n"
        "## Frequently Asked Questions\n\nQuestion answer.\n\n"
        "## References\n\nOld, stale reference."
    ).model_copy(
        update={
            "claims_used": [
                WriterClaim(claim="First claim", evidence_ids=["ev_001", "ev_002"])
            ]
        }
    )

    referenced = WriterAgent._append_references(article, evidence)

    assert referenced.article_markdown.count("1. [Research review]") == 1
    assert (
        "[\\[1\\]](#reference-1) First claim. "
        "[\\[1\\]](#reference-1) Second claim."
    ) in referenced.article_markdown
    assert "Old, stale reference" not in referenced.article_markdown


def test_writer_normalizes_numeric_citations_to_generated_reference_numbers():
    evidence = [
        WriterEvidence(
            evidence_id="ev_001",
            claim="First research finding",
            status="verified",
            source_url="https://papers.example/first",
            source_title="First Research Paper",
        ),
        WriterEvidence(
            evidence_id="ev_002",
            claim="Second research finding",
            status="verified",
            source_url="https://papers.example/second",
            source_title="Second Research Paper",
        ),
    ]
    article = _writer_output(
        "# Results\n\nThe findings were reported [22][2][1][2]."
    ).model_copy(
        update={
            "claims_used": [
                WriterClaim(
                    claim="Research findings",
                    evidence_ids=["ev_001", "ev_002"],
                )
            ]
        }
    )

    referenced = WriterAgent._append_references(article, evidence)

    assert (
        "The findings were reported [\\[2\\]](#reference-2), "
        "[\\[1\\]](#reference-1)."
    ) in referenced.article_markdown
    assert "22" not in referenced.article_markdown
    assert "1. [First Research Paper]" in referenced.article_markdown
    assert "2. [Second Research Paper]" in referenced.article_markdown
    assert any("numeric citation [22]" in item for item in referenced.unresolved_items)


def test_writer_removes_raw_urls_from_body_and_keeps_one_clean_reference_link():
    evidence = [
        WriterEvidence(
            evidence_id="ev_001",
            claim="Training finding",
            status="verified",
            source_url="https://papers.example/training",
            source_title="Training Paper",
            source_organization="Research Group",
            publication_date="2025",
        )
    ]
    article = _writer_output(
        "# Results\n\nSee [the company page](https://company.example/page) "
        "and https://papers.example/other for details. [ev_001] The finding."
    ).model_copy(
        update={
            "claims_used": [
                WriterClaim(claim="Training finding", evidence_ids=["ev_001"])
            ]
        }
    )

    referenced = WriterAgent._append_references(article, evidence)
    body, references = referenced.article_markdown.split("## References", 1)

    rendered_body, _ = render_annotated_markdown(body, [])
    assert ">https://" not in rendered_body
    assert "the company page" in body
    assert "[\\[1\\]](#reference-1)" in body
    assert (
        "1. [Research Group. Training Paper](https://papers.example/training) (2025)."
        in references
    )
    assert references.count("https://papers.example/training") == 1
    assert any("source URLs in article text" in item for item in referenced.unresolved_items)


def test_writer_uses_doi_link_when_source_url_is_unavailable():
    evidence = [
        WriterEvidence(
            evidence_id="ev_001",
            claim="Research finding",
            status="verified",
            source_title="Research Paper",
            doi="https://doi.org/10.1234/vr.2025",
        )
    ]
    article = _writer_output(
        "# Results\n\n[ev_001] The research finding."
    ).model_copy(
        update={
            "claims_used": [
                WriterClaim(claim="Research finding", evidence_ids=["ev_001"])
            ]
        }
    )

    referenced = WriterAgent._append_references(article, evidence)

    assert "[\\[1\\]](#reference-1)" in referenced.article_markdown
    assert (
        "1. [Research Paper](https://doi.org/10.1234/vr.2025). "
        "DOI: 10.1234/vr.2025."
    ) in referenced.article_markdown


def test_writer_reports_reference_without_a_source_link():
    evidence = [
        WriterEvidence(
            evidence_id="ev_001",
            claim="Research finding",
            status="verified",
            source_title="Research Paper",
        )
    ]
    article = _writer_output(
        "# Results\n\n[ev_001] The research finding."
    ).model_copy(
        update={
            "claims_used": [
                WriterClaim(claim="Research finding", evidence_ids=["ev_001"])
            ]
        }
    )

    referenced = WriterAgent._append_references(article, evidence)

    assert "1. Research Paper." in referenced.article_markdown
    assert any("has no valid source URL or DOI" in item for item in referenced.unresolved_items)


def test_writer_does_not_add_references_for_unused_evidence():
    evidence = [
        WriterEvidence(
            evidence_id="ev_001",
            claim="Unused fact",
            status="verified",
            source_url="https://papers.example/unused",
            source_title="Unused study",
        )
    ]

    referenced = WriterAgent._append_references(
        _writer_output().model_copy(update={"claims_used": []}),
        evidence,
    )

    assert "## References" not in referenced.article_markdown


def test_writer_removes_legacy_verification_markers_from_article_text():
    result = _writer_output(
        "# Results\n\nThe pilot reported an 80% cost reduction. "
        "[VERIFY: preserve source limitation: only measured in one pilot.]"
    )

    audited = WriterAgent._flag_unsupported_numeric_claims(result, _writer_input().evidence)

    assert "[VERIFY:" not in audited.article_markdown
    assert "only measured in one pilot" not in audited.article_markdown
    assert any("only measured in one pilot" in item for item in audited.unresolved_items)


def test_writer_flags_faq_answers_outside_word_count_target():
    audited = WriterAgent._audit_claims_and_faqs(_writer_output())

    assert any("FAQ answer" in item and "40–50 word target" in item for item in audited.unresolved_items)


def test_writer_surfaces_evidence_wording_boundaries_for_review():
    evidence = (
        _writer_input()
        .evidence[0]
        .model_copy(
            update={"wording_boundary": ("This is a company-reported benefit, not independently verified.")}
        )
    )
    result = WriterAgent._attach_supporting_evidence(_writer_output(), [evidence])

    audited = WriterAgent._audit_claims_and_faqs(result)

    assert any("not independently verified" in item for item in audited.unresolved_items)


def test_writer_does_not_block_faqs_for_outcome_wording():
    article = _ARTICLE.replace(
        "### What is VR training used for?",
        "### What measurable improvements should medical device companies expect from VR training?",
    )
    WriterAgent._validate_seo_requirements(_writer_output(article), _writer_input())


def test_writer_repairs_faq_after_evidence_validation_failure():
    invalid_article = _ARTICLE.replace(
        "It supports immersive training for medical device teams and lets learners\n"
        "practice procedures through interactive simulation.",
        "",
    )
    outputs = iter(
        [
            _writer_output(invalid_article).model_dump(mode="json"),
            {
                "metadata": {
                    "title": None,
                    "seo_title": None,
                    "meta_description": None,
                },
                "edits": [
                    {
                        "old_text": "### What is VR training used for?\n\n",
                        "new_text": (
                            "### What is VR training used for?\n\n"
                            "It supports immersive training for medical device teams and lets "
                            "learners practice procedures through interactive simulation.\n\n"
                        ),
                    }
                ],
            },
        ]
    )
    prompts = []

    class FakeMessages:
        @staticmethod
        def stream(**kwargs):
            prompts.append(kwargs["messages"][0]["content"])
            payload = json.dumps(next(outputs))
            return _FakeMessageStream(
                SimpleNamespace(
                    stop_reason="end_turn",
                    content=[SimpleNamespace(type="text", text=payload)],
                )
            )

    agent = WriterAgent.__new__(WriterAgent)
    agent.client = SimpleNamespace(messages=FakeMessages())
    agent.model = "test"
    agent.max_tokens = 1000

    result = agent.write(_writer_input())

    assert "### What is VR training used for?" in result.article_markdown
    assert len(prompts) == 2
    assert "3 to 5 relevant FAQ questions with answers" in prompts[1]
    assert "Do not return the full article" in prompts[1]


def test_writer_attaches_source_quote_and_status_to_each_claim():
    result = WriterAgent._attach_supporting_evidence(
        _writer_output(),
        _writer_input().evidence,
    )

    claim = result.claims_used[0]
    assert claim.evidence_ids == ["ev_001"]
    assert claim.source_urls == ["https://example.com/training"]
    assert claim.citation_status == "linked"
    assert claim.support_status == "not_assessed"
    assert any("No claim-level evidence assessment" in item for item in claim.validation_notes)
    assert len(claim.supporting_evidence) == 1
    assert claim.supporting_evidence[0].evidence_quote == _writer_input().evidence[0].evidence_quote
    assert claim.supporting_evidence[0].status == "verified"


def test_writer_revision_uses_human_instructions_and_returns_a_proposal():
    draft = _writer_output()
    proposed = draft.model_copy(update={"article_markdown": draft.article_markdown + "\n\nA clearer next step."})
    prompts = []
    agent = WriterAgent.__new__(WriterAgent)
    writer_input = _writer_input().model_copy(
        update={
            "evidence": [
                *_writer_input().evidence,
                WriterEvidence(
                    evidence_id="ev_002",
                    claim="Uncited background source",
                    status="verified",
                    evidence_quote="This source is not cited by the draft.",
                    source_url="https://example.com/background",
                ),
            ]
        }
    )

    def generate(draft, prompt):
        prompts.append(prompt)
        return proposed

    agent._generate_targeted_update = generate

    result = agent.revise(writer_input, draft, "Make the conclusion more direct.")

    assert "A clearer next step." in result.article_markdown
    assert result.article_markdown.endswith(
        "## References\n\n1. [example.com](https://example.com/training)."
    )
    assert draft.article_markdown == _writer_output().article_markdown
    assert draft.article_markdown != result.article_markdown
    assert "Make the conclusion more direct." in prompts[0]
    assert "This is a proposal" in prompts[0]
    request = json.loads(prompts[0].split("<revision_request>\n", 1)[1].split("\n</revision_request>", 1)[0])
    assert "revision_context" in request
    assert request["revision_context"]["evidence"][0]["evidence_id"] == "ev_001"
    assert len(request["revision_context"]["evidence"]) == 1
    assert "sections" not in request["current_draft"]
    assert "claims_used" not in request["current_draft"]
    assert len(prompts) == 1


def test_writer_revision_does_not_retry_for_a_preexisting_style_note():
    article = _ARTICLE + (
        "\n\nImmersive simulations support immersive practice and immersive learning."
    )
    draft = _writer_output(article)
    proposed = draft.model_copy(
        update={"article_markdown": article + "\n\nA clearer next step."}
    )
    prompts = []
    agent = WriterAgent.__new__(WriterAgent)

    def generate(_draft, prompt):
        prompts.append(prompt)
        return proposed

    agent._generate_targeted_update = generate

    agent.revise(_writer_input(), draft, "Make the conclusion more direct.")

    assert len(prompts) == 1


def test_writer_keeps_optional_word_choice_lint_as_a_note():
    article = _ARTICLE + (
        "\n\nImmersive simulations support immersive practice and immersive learning."
    )
    result = _writer_output(article)
    agent = WriterAgent.__new__(WriterAgent)
    agent._generate_targeted_update = lambda *_args, **_kwargs: pytest.fail(
        "Optional word-choice guidance must not start a repair call."
    )

    checked, notes = agent._enforce_style(result, _writer_input())

    assert checked.article_markdown == article
    assert any("immersive" in note for note in notes)


def test_writer_revision_rejects_empty_instructions():
    agent = WriterAgent.__new__(WriterAgent)

    with pytest.raises(ValueError, match="instructions must not be empty"):
        agent.revise(_writer_input(), _writer_output(), "  ")


def test_writer_drops_claim_url_not_linked_to_its_evidence_id_and_reports_it():
    result = _writer_output().model_copy(
        update={
            "claims_used": [
                _writer_output()
                .claims_used[0]
                .model_copy(update={"source_urls": ["https://unrelated.example/source"]})
            ]
        }
    )

    attached = WriterAgent._attach_supporting_evidence(result, _writer_input().evidence)

    assert attached.claims_used[0].source_urls == ["https://example.com/training"]
    assert any(
        "URLs not linked to supplied evidence" in item and "https://unrelated.example/source" in item
        for item in attached.unresolved_items
    )
    assert attached.claims_used[0].citation_status == "mismatch"


def test_citation_mismatch_does_not_change_claim_support_assessment():
    result = _writer_output().model_copy(
        update={
            "claims_used": [
                _writer_output()
                .claims_used[0]
                .model_copy(
                    update={
                        "source_urls": ["https://unrelated.example/source"],
                        "support_status": "partially_supported",
                    }
                )
            ]
        }
    )

    attached = WriterAgent._attach_supporting_evidence(
        result,
        _writer_input().evidence,
    )

    assert attached.claims_used[0].citation_status == "mismatch"
    assert attached.claims_used[0].support_status == "partially_supported"


@pytest.mark.parametrize(
    ("support_status", "expected_warning"),
    [
        ("partially_supported", "supports only part"),
        ("unsupported", "does not support this claim"),
        ("contradicted", "directly contradicts"),
        ("inconclusive", "incomplete or ambiguous"),
    ],
)
def test_writer_reports_distinct_company_claim_evidence_assessments(support_status, expected_warning):
    claim = _writer_output().claims_used[0].model_copy(update={"support_status": support_status})
    audited = WriterAgent._audit_claims_and_faqs(_writer_output().model_copy(update={"claims_used": [claim]}))

    assert any(expected_warning in warning for warning in audited.unresolved_items)


def test_writer_accepts_equivalent_url_formatting_from_registered_evidence():
    result = _writer_output().model_copy(
        update={
            "claims_used": [
                _writer_output()
                .claims_used[0]
                .model_copy(update={"source_urls": ["HTTPS://EXAMPLE.COM:443/training/#procedure"]})
            ]
        }
    )

    attached = WriterAgent._attach_supporting_evidence(
        result,
        _writer_input().evidence,
    )

    assert attached.claims_used[0].citation_status == "linked"
    assert attached.claims_used[0].source_urls == ["https://example.com/training"]
    assert not attached.unresolved_items


def test_writer_marks_claim_without_evidence_as_missing_and_reviewable():
    result = _writer_output().model_copy(
        update={"claims_used": [WriterClaim(claim="An unreferenced claim about company capability.")]}
    )

    attached = WriterAgent._attach_supporting_evidence(
        result,
        _writer_input().evidence,
    )

    assert attached.claims_used[0].citation_status == "missing"
    assert attached.claims_used[0].supporting_evidence == []
    assert any("no registered evidence IDs" in item for item in attached.unresolved_items)


def test_writer_does_not_require_customer_evidence_for_generic_context():
    result = _writer_output("# Industry context\n").model_copy(
        update={
            "claims_used": [
                WriterClaim(
                    claim="Healthcare professionals often benefit from hands-on product demonstrations.",
                    claim_scope="industry_context",
                )
            ]
        }
    )

    attached = WriterAgent._attach_supporting_evidence(result, [])
    audited = WriterAgent._audit_claims_and_faqs(attached)

    assert audited.claims_used[0].citation_status == "missing"
    assert not audited.unresolved_items


def test_writer_keeps_company_evidence_requirements():
    result = _writer_output("# Company capabilities\n").model_copy(
        update={
            "claims_used": [
                WriterClaim(
                    claim="Example provides VR surgical simulations.",
                    claim_scope="company_specific",
                )
            ]
        }
    )

    attached = WriterAgent._attach_supporting_evidence(result, [])
    audited = WriterAgent._audit_claims_and_faqs(attached)

    assert any("no registered evidence IDs" in item for item in audited.unresolved_items)


def test_writer_returns_blog_when_generated_claim_has_unlinked_url():
    generated = _writer_output().model_copy(
        update={
            "claims_used": [
                _writer_output()
                .claims_used[0]
                .model_copy(update={"source_urls": ["https://www.ethosh.com/"]})
            ]
        }
    )

    class FakeMessages:
        @staticmethod
        def stream(**_kwargs):
            payload = json.dumps(generated.model_dump(mode="json"))
            response = SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text=payload)],
            )
            return _FakeMessageStream(response)

    agent = WriterAgent.__new__(WriterAgent)
    agent.client = SimpleNamespace(messages=FakeMessages())
    agent.model = "test"
    agent.max_tokens = 1000

    result = agent.write(_writer_input())

    assert result.article_markdown.startswith(_ARTICLE.strip())
    assert result.article_markdown.endswith(
        "## References\n\n1. [example.com](https://example.com/training)."
    )
    assert "https://www.ethosh.com/" not in result.claims_used[0].source_urls
    assert any("https://www.ethosh.com/" in item for item in result.unresolved_items)


def test_writer_drops_unknown_evidence_ids_and_reports_them():
    result = _writer_output().model_copy(
        update={
            "claims_used": [
                _writer_output().claims_used[0],
                WriterClaim(
                    claim="A separate claim with a fabricated evidence reference.",
                    evidence_ids=["invented-evidence"],
                ),
            ]
        }
    )

    sanitized = WriterAgent._sanitize_evidence_references(
        result,
        _writer_input().evidence,
    )

    assert len(sanitized.claims_used) == 2
    assert sanitized.claims_used[0].evidence_ids == ["ev_001"]
    assert sanitized.claims_used[0].citation_status == "not_checked"
    assert sanitized.claims_used[1].evidence_ids == []
    assert sanitized.claims_used[1].citation_status == "mismatch"
    assert any("invented-evidence" in item for item in sanitized.unresolved_items)


def test_writer_keeps_blog_when_metadata_exceeds_seo_limits():
    invalid = (
        _writer_output()
        .model_copy(
            update={
                "seo_title": "A very long SEO title that exceeds the sixty character limit",
                "meta_description": "x" * 161,
            }
        )
        .model_dump(mode="json")
    )
    calls = []

    class FakeMessages:
        @staticmethod
        def stream(**kwargs):
            calls.append(kwargs)
            return _FakeMessageStream(
                SimpleNamespace(
                    stop_reason="end_turn",
                    content=[SimpleNamespace(type="text", text=json.dumps(invalid))],
                )
            )

    agent = WriterAgent.__new__(WriterAgent)
    agent.client = SimpleNamespace(messages=FakeMessages())
    agent.model = "test"
    agent.max_tokens = 1000

    result = agent.write(_writer_input())

    assert len(result.meta_description) <= 155
    assert len(result.seo_title) <= 60
    assert len(result.meta_description) <= 155
    assert len(calls) == 1


def test_writer_returns_blog_even_when_it_exceeds_word_count_target():
    title = "A Practical Guide to Effective Team Training"
    writer_input = _writer_input().model_copy(update={"target_word_count": 800, "title": title})
    article = (
        f"# {title}\n\n"
        + "Clear guidance helps teams understand processes and make informed decisions. " * 180
        + "\n| Consideration | What to review |\n| --- | --- |\n| Evidence | Check supplied sources. |\n"
        + "\n## Frequently Asked Questions\n\n"
        + "### What does the guide explain?\n\nIt explains practical approaches.\n\n"
        + "### How can teams use it?\n\nTeams can apply the guidance to their work.\n\n"
        + "### Who may find it useful?\n\nTeams seeking clear guidance may find it useful."
    )
    output = _writer_output(article, title).model_dump(mode="json")
    calls = []

    class FakeMessages:
        @staticmethod
        def stream(**kwargs):
            calls.append(kwargs)
            return _FakeMessageStream(
                SimpleNamespace(
                    stop_reason="end_turn",
                    content=[SimpleNamespace(type="text", text=json.dumps(output))],
                )
            )

    agent = WriterAgent.__new__(WriterAgent)
    agent.client = SimpleNamespace(messages=FakeMessages())
    agent.model = "test"
    agent.max_tokens = 1000

    result = agent.write(writer_input)

    assert result.word_count > writer_input.target_word_count
    assert len(calls) == 1


def test_writer_uses_one_generation_when_response_fits():
    output = _writer_output().model_dump(mode="json")
    output.pop("sections")
    responses = [
        SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=json.dumps(output))],
        )
    ]
    token_limits = []
    requests = []

    class FakeMessages:
        @staticmethod
        def stream(**kwargs):
            requests.append(kwargs)
            token_limits.append(kwargs["max_tokens"])
            return _FakeMessageStream(responses.pop(0))

    agent = WriterAgent.__new__(WriterAgent)
    agent.client = SimpleNamespace(messages=FakeMessages())
    agent.model = "test"
    agent.max_tokens = 11000

    result = agent._generate_output("test prompt")

    assert result.title == "VR Training for Medical Device Teams"
    assert token_limits == [11000]
    assert "tool_choice" not in requests[0]
    assert "tools" not in requests[0]
    output_format = requests[0]["output_config"]["format"]
    schema = output_format["schema"]
    assert output_format["type"] == "json_schema"
    assert '"minimum"' not in json.dumps(schema)
    assert "sections" not in schema["properties"]
    assert "sections" not in schema["required"]
    assert set(schema["required"]) >= {
        "title",
        "article_markdown",
        "word_count",
    }
    assert [section.heading for section in result.sections] == [
        "How the training works",
        "Frequently Asked Questions",
        "What is VR training used for?",
        "How do learners practice?",
        "Who can use this training?",
    ]


def test_writer_uses_json_schema_output_instead_of_tool_choice():
    requests = []

    class FakeMessages:
        @staticmethod
        def stream(**kwargs):
            requests.append(kwargs)
            response = SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text=json.dumps(
                    _writer_output().model_dump(mode="json")
                ))],
            )

            class RawMessageStream:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def get_final_message(self):
                    return response

            return RawMessageStream()

    agent = WriterAgent.__new__(WriterAgent)
    agent.client = SimpleNamespace(messages=FakeMessages())
    agent.model = "claude-sonnet-5-5"
    agent.max_tokens = 11000

    result = agent._generate_output("Write a blog.")

    assert result.title == "VR Training for Medical Device Teams"
    assert "tool_choice" not in requests[0]
    assert "tools" not in requests[0]
    assert requests[0]["output_config"]["format"]["type"] == "json_schema"
    assert "tool_choice" not in requests[0]
    assert "tools" not in requests[0]


@pytest.mark.parametrize(
    ("stop_reason", "text", "message"),
    [
        ("end_turn", "A blog draft", "invalid JSON"),
        ("end_turn", "[]", "JSON object"),
        ("tool_use", "{}", "did not complete"),
    ],
)
def test_writer_rejects_responses_that_are_not_structured_json(
    stop_reason,
    text,
    message,
):
    class FakeMessages:
        @staticmethod
        def stream(**_kwargs):
            return _FakeMessageStream(
                SimpleNamespace(
                    stop_reason=stop_reason,
                    content=[SimpleNamespace(type="text", text=text)],
                )
            )

    agent = WriterAgent.__new__(WriterAgent)
    agent.client = SimpleNamespace(messages=FakeMessages())
    agent.model = "claude-sonnet-5-5"
    agent.max_tokens = 11000

    with pytest.raises(RuntimeError, match=message):
        agent._generate_output("Write a blog.")


def test_writer_targeted_repair_rejects_ambiguous_edit_anchors():
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(
                type="text",
                text=json.dumps(
                    {
                        "metadata": {
                            "title": None,
                            "seo_title": None,
                            "meta_description": None,
                        },
                        "edits": [{"old_text": "practice", "new_text": "train"}],
                    }
                ),
            )
        ],
    )

    class FakeMessages:
        @staticmethod
        def stream(**_kwargs):
            return _FakeMessageStream(response)

    agent = WriterAgent.__new__(WriterAgent)
    agent.client = SimpleNamespace(messages=FakeMessages())
    agent.model = "test"
    agent.max_tokens = 11000

    with pytest.raises(RuntimeError, match="anchor must occur exactly once"):
        agent._generate_targeted_update(_writer_output(), "Apply a small edit.")


def test_writer_normalizes_invalid_claim_support_status_without_calling_it_supported():
    output = _writer_output().model_dump(mode="json")
    output["claims_used"][0]["support_status"] = "verified"

    class FakeMessages:
        @staticmethod
        def stream(**_kwargs):
            return _FakeMessageStream(
                SimpleNamespace(
                    stop_reason="end_turn",
                    content=[
                        SimpleNamespace(type="text", text=json.dumps(output)),
                    ],
                )
            )

    agent = WriterAgent.__new__(WriterAgent)
    agent.client = SimpleNamespace(messages=FakeMessages())
    agent.model = "test"
    agent.max_tokens = 1000

    result = agent._generate_output("test prompt")

    assert result.claims_used[0].support_status == "not_assessed"
    assert any(
        "invalid support_status 'verified'" in item and "set to not_assessed" in item
        for item in result.unresolved_items
    )


def test_writer_reports_truncation_without_retrying():
    token_limits = []

    class FakeMessages:
        @staticmethod
        def stream(**kwargs):
            token_limits.append(kwargs["max_tokens"])
            return _FakeMessageStream(SimpleNamespace(stop_reason="max_tokens", content=[]))

    agent = WriterAgent.__new__(WriterAgent)
    agent.client = SimpleNamespace(messages=FakeMessages())
    agent.model = "test"
    agent.max_tokens = 11000

    with pytest.raises(RuntimeError, match="truncated at 11,000 output tokens"):
        agent._generate_output("test prompt")

    assert token_limits == [11000]


@pytest.mark.parametrize(
    ("article", "error"),
    [
        ("# VR Training for Medical Device Teams\n\nNo FAQ.", "missing its Frequently Asked Questions"),
        (
            _ARTICLE.replace(
                "### Who can use this training?\n\nMedical device teams can use VR-based training to practice procedures.\n",
                "",
            ),
            "3 to 5 questions",
        ),
        (
            _ARTICLE.replace(
                "### Who can use this training?\n\nMedical device teams can use VR-based training to practice procedures.",
                "### Who can use this training?\n\nMedical device teams can use VR-based training to practice procedures.\n\n### What equipment is required?\n\nThe article does not specify equipment.\n\n### How long does training take?\n\nThe article does not say.\n\n### How is progress measured?\n\nThe article does not say.",
            ),
            "3 to 5 questions",
        ),
    ],
)
def test_writer_rejects_invalid_faq_structure(article: str, error: str):
    with pytest.raises(RuntimeError, match=error):
        WriterAgent._validate_seo_requirements(_writer_output(article), _writer_input())


def test_writer_input_carries_seo_research_and_prompt_requires_article_standards():
    writer_input = _writer_input()
    prompt = WriterAgent._build_user_prompt(writer_input)

    assert "VR training for medical devices" in prompt
    assert "immersive training" in prompt
    assert "How does VR training work?" in prompt
    assert "3 to 5" in prompt
    assert "40–50 words" in prompt
    assert "search intent" in prompt
    system_prompt = WriterAgent._system_prompt()
    assert "BLOG WRITER" in system_prompt
    assert "Use only supplied evidence" in system_prompt
    assert "Never invent capabilities, outcomes, numbers, citations, or URLs" in system_prompt
    assert len(system_prompt) < 1000
    assert "Do not return sections; they are derived from article_markdown." in prompt
    assert set(WriterOutput.model_fields) == {
        "title",
        "seo_title",
        "meta_description",
        "article_markdown",
        "word_count",
        "sections",
        "claims_used",
        "unresolved_items",
        "brief_coverage",
    }


def test_human_writing_rules_are_in_the_writer_prompt():
    prompt = WriterAgent._system_prompt()

    assert "company's voice" in prompt
    assert "Keep the writing clear" in prompt
    assert "Respect evidence status and wording boundaries" in prompt
    assert "Do not imply independent verification" in prompt


def test_writer_user_prompt_reinforces_company_voice_and_evidence_limits():
    prompt = WriterAgent._build_user_prompt(_writer_input())

    assert "speaking directly to" in prompt
    assert "prospective customers" in prompt
    assert "Lead with the customer's challenge" in prompt
    assert "outside investigator or auditor" in prompt
    assert "Never invent customer outcomes" in prompt
    assert "Present limitations and unknowns constructively" in prompt
    assert "clear, relevant call to action" in prompt
    assert "exact supplied" in prompt
    assert "bracketed numbered citations" in prompt
    assert "matching entry in References" in prompt


def test_writer_prompt_sends_only_evidence_mapped_to_brief_sections():
    writer_input = _writer_input().model_copy(
        update={
            "sections": [
                WriterSectionInput(heading="Evidence", evidence_ids=["ev_001"])
            ],
            "evidence": [
                *_writer_input().evidence,
                WriterEvidence(
                    evidence_id="ev_002",
                    claim="Unmapped background source",
                    status="verified",
                    evidence_quote="This source is not mapped to a brief section.",
                    source_url="https://example.com/background",
                ),
            ],
        }
    )

    prompt = WriterAgent._build_user_prompt(writer_input)
    payload = json.loads(prompt.split("<writer_input>\n", 1)[1].split("\n</writer_input>", 1)[0])

    assert [item["evidence_id"] for item in payload["evidence"]] == ["ev_001"]


def test_writer_response_schema_omits_fields_built_by_application():
    from src.agents.aimpact_blog.writer import _WRITER_OUTPUT_SCHEMA

    claim_properties = _WRITER_OUTPUT_SCHEMA["properties"]["claims_used"]["items"]["properties"]

    assert not {"source_urls", "supporting_evidence", "validation_notes"} & set(claim_properties)


def test_reviewer_prompt_reports_issues_without_rewriting():
    prompt = ReviewerAgent._system_prompt()

    assert "Do not rewrite, edit" in prompt
    assert "for a human to review" in prompt
    assert '"article_markdown"' not in prompt
    assert "grammar and spelling errors" in prompt
    assert "Use low severity for optional style preferences" in prompt


def test_reviewer_requires_exactly_one_submit_review_tool_call():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                name="submit_review",
                input={
                    "summary": "Reviewed the article.",
                    "findings": [],
                    "unresolved_items": [],
                },
            )
        ]
    )

    assert ReviewerAgent._extract_review_tool_input(response) == {
        "summary": "Reviewed the article.",
        "findings": [],
        "unresolved_items": [],
    }

    with pytest.raises(RuntimeError, match="exactly one structured review tool call"):
        ReviewerAgent._extract_review_tool_input(SimpleNamespace(content=[]))


def test_reviewer_rejects_unexpected_tool_call():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name="other_tool", input={})]
    )

    with pytest.raises(RuntimeError, match="unexpected tool call"):
        ReviewerAgent._extract_review_tool_input(response)


def test_reviewer_schema_rejects_invalid_tool_input():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                name="submit_review",
                input={"findings": []},
            )
        ]
    )

    with pytest.raises(ValidationError, match="summary"):
        ReviewerResponse.model_validate(ReviewerAgent._extract_review_tool_input(response))


def test_writer_requires_seo_title_and_155_character_meta_description():
    writer_input = _writer_input()
    valid = _writer_output()
    WriterAgent._validate_seo_requirements(valid, writer_input)

    with pytest.raises(RuntimeError, match="SEO title"):
        WriterAgent._validate_seo_requirements(
            valid.model_copy(update={"seo_title": "x" * 61}),
            writer_input,
        )
    with pytest.raises(RuntimeError, match="meta description"):
        WriterAgent._validate_seo_requirements(
            valid.model_copy(update={"meta_description": "x" * 156}),
            writer_input,
        )


def test_writer_rejects_article_without_markdown_table():
    article_without_table = _ARTICLE.replace(
        "| Consideration | What the supplied evidence says |\n"
        "| --- | --- |\n"
        "| Training approach | Immersive training supports procedural practice. |\n\n",
        "",
    )

    with pytest.raises(RuntimeError, match="at least one valid Markdown table"):
        WriterAgent._validate_seo_requirements(
            _writer_output(article_without_table),
            _writer_input(),
        )


def test_writer_repairs_article_when_table_is_missing():
    article_without_table = _ARTICLE.replace(
        "| Consideration | What the supplied evidence says |\n"
        "| --- | --- |\n"
        "| Training approach | Immersive training supports procedural practice. |\n\n",
        "",
    )
    outputs = iter(
        [
            _writer_output(article_without_table).model_dump(mode="json"),
            {
                "metadata": {
                    "title": None,
                    "seo_title": None,
                    "meta_description": None,
                },
                "edits": [
                    {
                        "old_text": "## Frequently Asked Questions",
                        "new_text": (
                            "| Consideration | What the supplied evidence says |\n"
                            "| --- | --- |\n"
                            "| Training approach | Immersive training supports procedural practice. |\n\n"
                            "## Frequently Asked Questions"
                        ),
                    }
                ],
            },
        ]
    )
    calls = []

    class FakeMessages:
        @staticmethod
        def stream(**kwargs):
            calls.append(kwargs)
            return _FakeMessageStream(
                SimpleNamespace(
                    stop_reason="end_turn",
                    content=[SimpleNamespace(type="text", text=json.dumps(next(outputs)))],
                )
            )

    agent = WriterAgent.__new__(WriterAgent)
    agent.client = SimpleNamespace(messages=FakeMessages())
    agent.model = "test"
    agent.max_tokens = 1000

    result = agent.write(_writer_input())

    assert "| Consideration | What the supplied evidence says |" in result.article_markdown
    expected_claim = _writer_output(article_without_table).claims_used[0]
    assert result.claims_used[0].claim == expected_claim.claim
    assert result.claims_used[0].evidence_ids == expected_claim.evidence_ids
    assert len(calls) == 2
    assert calls[1]["max_tokens"] == 1000
    assert "tool_choice" not in calls[1]
    assert "tools" not in calls[1]
    patch_format = calls[1]["output_config"]["format"]
    assert patch_format["type"] == "json_schema"
    assert patch_format["schema"]["properties"]["metadata"]["required"] == [
        "title",
        "seo_title",
        "meta_description",
    ]
    assert "at least one useful, valid Markdown table" in calls[1]["messages"][0]["content"]


def test_writer_combines_seo_and_style_repairs_in_one_retry():
    article_with_issues = _ARTICLE.replace(
        "can give medical device teams",
        "can leverage medical device teams",
    ).replace(
        "| Consideration | What the supplied evidence says |\n"
        "| --- | --- |\n"
        "| Training approach | Immersive training supports procedural practice. |\n\n",
        "",
    )
    outputs = iter(
        [
            _writer_output(article_with_issues).model_dump(mode="json"),
            {
                "metadata": {
                    "title": None,
                    "seo_title": None,
                    "meta_description": None,
                },
                "edits": [
                    {
                        "old_text": "can leverage medical device teams",
                        "new_text": "can give medical device teams",
                    },
                    {
                        "old_text": "## Frequently Asked Questions",
                        "new_text": (
                            "| Consideration | What the supplied evidence says |\n"
                            "| --- | --- |\n"
                            "| Training approach | Immersive training supports procedural practice. |\n\n"
                            "## Frequently Asked Questions"
                        ),
                    },
                ],
            },
        ]
    )
    calls = []

    class FakeMessages:
        @staticmethod
        def stream(**kwargs):
            calls.append(kwargs)
            return _FakeMessageStream(
                SimpleNamespace(
                    stop_reason="end_turn",
                    content=[SimpleNamespace(type="text", text=json.dumps(next(outputs)))],
                )
            )

    agent = WriterAgent.__new__(WriterAgent)
    agent.client = SimpleNamespace(messages=FakeMessages())
    agent.model = "test"
    agent.max_tokens = 1000

    result = agent.write(_writer_input())

    assert "| Consideration | What the supplied evidence says |" in result.article_markdown
    assert len(calls) == 2
    repair_prompt = calls[1]["messages"][0]["content"]
    assert "Markdown table" in repair_prompt
    assert "Style issues to fix in the same repair" in repair_prompt
    assert "leverage" in repair_prompt
    assert "<previous_draft>" in repair_prompt
    assert "<writer_input>" not in repair_prompt
    assert "Do not return the full article" in repair_prompt
    assert "claims_used" not in repair_prompt
    assert "BLOG WRITER" in calls[0]["system"]
    assert "BLOG WRITER" not in calls[1]["system"]


def test_writer_word_count_target_does_not_block_valid_article():
    writer_input = _writer_input().model_copy(update={"target_word_count": 800})

    long_article = (
        "# VR Training for Medical Device Teams\n\n"
        + "Evidence-based guidance helps teams make informed decisions. " * 160
        + "\n| Factor | Guidance |\n| --- | --- |\n| Evidence | Use supplied sources. |\n"
        + "\n## Frequently Asked Questions\n\n"
        + "### What does training support?\n\nTraining supports practice.\n\n"
        + "### How is it delivered?\n\nIt uses simulation.\n\n"
        + "### Who can take part?\n\nTeams can take part."
    )

    WriterAgent._validate_seo_requirements(_writer_output(long_article), writer_input)


def test_writer_streams_claude_requests_and_accepts_configured_word_count():
    writer_input = _writer_input().model_copy(update={"target_word_count": 800})
    article = (
        "# VR Training for Medical Device Teams\n\n"
        + "Useful evidence-based guidance informs decisions. " * 145
        + "\n| Factor | Guidance |\n| --- | --- |\n| Evidence | Use supplied sources. |\n"
        + "\n## Frequently Asked Questions\n\n"
        + "### What does training support?\n\nTraining supports practice.\n\n"
        + "### How is it delivered?\n\nIt uses simulation.\n\n"
        + "### Who can take part?\n\nTeams can take part."
    )
    valid = _writer_output(article).model_dump(mode="json")
    calls = []

    class FakeMessages:
        @staticmethod
        def stream(**kwargs):
            calls.append(kwargs)
            return _FakeMessageStream(
                SimpleNamespace(
                    stop_reason="end_turn",
                    content=[SimpleNamespace(type="text", text=json.dumps(valid))],
                )
            )

    agent = WriterAgent.__new__(WriterAgent)
    agent.client = SimpleNamespace(messages=FakeMessages())
    agent.model = "test"
    agent.max_tokens = 1000

    result = agent.write(writer_input)

    assert calls
    assert 720 <= result.word_count <= 880
