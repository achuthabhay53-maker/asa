import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from aca.agents.writer import WriterAgent
from aca.agents.reviewer import ReviewerAgent
from aca.models.writer import (
    WriterClaim,
    WriterEvidence,
    WriterInput,
    WriterOutput,
    WrittenSection,
)


_ARTICLE = """# VR Training for Medical Device Teams

VR training can give medical device teams a way to practice procedures through
interactive simulations. The supplied evidence describes immersive training
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

Medical device teams can use immersive training to practice procedures.
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


def test_reviewer_returns_the_complete_edited_blog():
    draft = _writer_output()
    long_meta_description = (
        "Custom VR surgical simulations & clinical training for medical device companies. "
        "Ethosh delivers 8–12 week implementations with enterprise analytics. Learn more."
    )
    revised_article = draft.article_markdown.replace(
        "VR training can give medical device teams",
        "Immersive VR training can give medical device teams",
        1,
    )
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(
                type="text",
                text=json.dumps(
                    {
                        "summary": "Made one clarity improvement.",
                        "findings": [],
                        "title": draft.title,
                        "seo_title": draft.seo_title,
                        "meta_description": long_meta_description,
                        "article_markdown": revised_article,
                        "unresolved_items": [],
                    }
                ),
            )
        ],
    )
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return response

    reviewer = ReviewerAgent.__new__(ReviewerAgent)
    reviewer.model = "test"
    reviewer.max_tokens = 16384
    reviewer.client = SimpleNamespace(messages=SimpleNamespace(create=create))

    result = reviewer.review(_writer_input(), draft)

    assert result.revised_blog.article_markdown == revised_article
    assert len(long_meta_description) > 155
    assert len(result.revised_blog.meta_description) <= 155
    assert result.revised_blog.meta_description.startswith("Custom VR surgical simulations")
    assert any(
        "meta description exceeded 155 characters" in item for item in result.revised_blog.unresolved_items
    )
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 16384
    request = json.loads(calls[0]["messages"][0]["content"])
    assert "article_markdown" in request["draft"]
    assert "sections" in request
    assert set(request["draft"]["claims_used"][0]) == {
        "claim",
        "claim_scope",
        "evidence_ids",
        "support_status",
    }


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
    reviewer.max_tokens = 16384
    reviewer.client = SimpleNamespace(messages=SimpleNamespace(create=create))

    with pytest.raises(RuntimeError, match="truncated at 16,384 output tokens"):
        reviewer.review(_writer_input(), _writer_output())

    assert len(calls) == 1


def test_reviewer_rebuilds_writer_sections_from_compact_markdown_response():
    from aca.models.reviewer import RevisedBlog

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
        meta_description="How VR training helps medical device teams practice procedures.",
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

    assert "[VERIFY:" in audited.article_markdown
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

    assert "preserve source limitation" in audited.article_markdown
    assert any("not independently verified" in item for item in audited.unresolved_items)


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
            _writer_output().model_dump(mode="json"),
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

    assert result.article_markdown == _ARTICLE.strip()
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

    assert result.meta_description == result.title
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


def test_writer_retries_truncated_response_with_larger_token_limit():
    responses = [
        SimpleNamespace(stop_reason="max_tokens", content=[]),
        SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(
                    type="text",
                    text=json.dumps(_writer_output().model_dump(mode="json")),
                )
            ],
        ),
    ]
    token_limits = []

    class FakeMessages:
        @staticmethod
        def stream(**kwargs):
            token_limits.append(kwargs["max_tokens"])
            return _FakeMessageStream(responses.pop(0))

    agent = WriterAgent.__new__(WriterAgent)
    agent.client = SimpleNamespace(messages=FakeMessages())
    agent.model = "test"
    agent.max_tokens = 11000

    result = agent._generate_output("test prompt")

    assert result.title == "VR Training for Medical Device Teams"
    assert token_limits == [11000, 22000]


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


def test_writer_reports_when_retry_is_still_truncated():
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

    with pytest.raises(RuntimeError, match="truncated at 22,000 output tokens"):
        agent._generate_output("test prompt")

    assert token_limits == [11000, 22000]


@pytest.mark.parametrize(
    ("article", "error"),
    [
        ("# VR Training for Medical Device Teams\n\nNo FAQ.", "missing its Frequently Asked Questions"),
        (
            _ARTICLE.replace(
                "### Who can use this training?\n\nMedical device teams can use immersive training to practice procedures.\n",
                "",
            ),
            "3 to 5 questions",
        ),
        (
            _ARTICLE.replace(
                "### Who can use this training?\n\nMedical device teams can use immersive training to practice procedures.",
                "### Who can use this training?\n\nMedical device teams can use immersive training to practice procedures.\n\n### What equipment is required?\n\nThe article does not specify equipment.\n\n### How long does training take?\n\nThe article does not say.\n\n### How is progress measured?\n\nThe article does not say.",
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
    assert "seo_title separately" in system_prompt
    assert "first 60 words" in system_prompt
    assert "40–50 words" in system_prompt
    assert "conclusion before the FAQ section" in system_prompt
    assert "[VERIFY: specific information needed]" in system_prompt
    assert "at least one concise, useful Markdown table" in system_prompt
    assert "never invent values, company capabilities, competitors, or" in system_prompt
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
            _writer_output().model_dump(mode="json"),
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
    assert "at least one useful, valid Markdown table" in calls[1]["messages"][0]["content"]


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
