import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from aca.agents.writer import WriterAgent
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
    example_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "agent5_input.example.json"
    )
    writer_input = WriterInput.model_validate_json(
        example_path.read_text(encoding="utf-8")
    )

    assert len(writer_input.faqs) == 3
    assert len(writer_input.evidence) >= 3
    assert all(item.evidence_quote for item in writer_input.evidence)


def _writer_output(
    article: str = _ARTICLE,
    title: str = "VR Training for Medical Device Teams",
) -> WriterOutput:
    return WriterOutput(
        title=title,
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
    five_faqs = _ARTICLE + """
### What do learners practice?

Learners practice procedures in interactive simulation.

### How does immersive training help?

It gives medical device teams a simulated way to practice procedures.
"""
    WriterAgent._validate_seo_requirements(_writer_output(five_faqs), _writer_input())


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
            return _FakeMessageStream(SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text=payload)],
            ))

    agent = WriterAgent.__new__(WriterAgent)
    agent.client = SimpleNamespace(messages=FakeMessages())
    agent.model = "test"
    agent.max_tokens = 1000

    result = agent.write(_writer_input())

    assert "### What is VR training used for?" in result.article_markdown
    assert len(prompts) == 2
    assert "Do not ask about measurable improvements" in prompts[1]


def test_writer_attaches_source_quote_and_status_to_each_claim():
    result = WriterAgent._attach_supporting_evidence(
        _writer_output(),
        _writer_input().evidence,
    )

    claim = result.claims_used[0]
    assert claim.evidence_ids == ["ev_001"]
    assert claim.source_urls == ["https://example.com/training"]
    assert len(claim.supporting_evidence) == 1
    assert claim.supporting_evidence[0].evidence_quote == _writer_input().evidence[0].evidence_quote
    assert claim.supporting_evidence[0].status == "verified"


def test_writer_rejects_claim_url_not_linked_to_its_evidence_id():
    result = _writer_output().model_copy(
        update={
            "claims_used": [
                _writer_output().claims_used[0].model_copy(
                    update={"source_urls": ["https://unrelated.example/source"]}
                )
            ]
        }
    )

    with pytest.raises(RuntimeError, match="not linked to its evidence IDs"):
        WriterAgent._attach_supporting_evidence(result, _writer_input().evidence)


def test_writer_repairs_invalid_meta_description():
    invalid = _writer_output().model_copy(
        update={"meta_description": "x" * 161}
    ).model_dump(mode="json")
    valid = _writer_output().model_dump(mode="json")
    outputs = iter([invalid, valid])
    prompts = []

    class FakeMessages:
        @staticmethod
        def stream(**kwargs):
            prompts.append(kwargs["messages"][0]["content"])
            return _FakeMessageStream(SimpleNamespace(
                stop_reason="end_turn",
                content=[
                    SimpleNamespace(type="text", text=json.dumps(next(outputs)))
                ],
            ))

    agent = WriterAgent.__new__(WriterAgent)
    agent.client = SimpleNamespace(messages=FakeMessages())
    agent.model = "test"
    agent.max_tokens = 1000

    result = agent.write(_writer_input())

    assert result.meta_description == _writer_output().meta_description
    assert len(prompts) == 2
    assert "between 1 and 160 characters" in prompts[1]


def test_writer_returns_blog_even_when_it_exceeds_word_count_target():
    title = "A Practical Guide to Effective Team Training"
    writer_input = _writer_input().model_copy(
        update={"target_word_count": 800, "title": title}
    )
    article = (
        f"# {title}\n\n"
        + "Clear guidance helps teams understand processes and make informed decisions. " * 180
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


def test_writer_reports_when_retry_is_still_truncated():
    token_limits = []

    class FakeMessages:
        @staticmethod
        def stream(**kwargs):
            token_limits.append(kwargs["max_tokens"])
            return _FakeMessageStream(
                SimpleNamespace(stop_reason="max_tokens", content=[])
            )

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
            _ARTICLE.replace("### Who can use this training?\n\nMedical device teams can use immersive training to practice procedures.\n", ""),
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


def test_writer_input_carries_seo_research_without_changing_output_schema():
    writer_input = _writer_input()
    prompt = WriterAgent._build_user_prompt(writer_input)

    assert "VR training for medical devices" in prompt
    assert "immersive training" in prompt
    assert "How does VR training work?" in prompt
    assert "3 to 5" in prompt
    assert "short and clear" in prompt
    assert "search intent" in prompt
    assert set(WriterOutput.model_fields) == {
        "title",
        "meta_description",
        "article_markdown",
        "word_count",
        "sections",
        "claims_used",
        "unresolved_items",
        "brief_coverage",
    }


def test_writer_word_count_target_does_not_block_valid_article():
    writer_input = _writer_input().model_copy(update={"target_word_count": 800})

    long_article = (
        "# VR Training for Medical Device Teams\n\n"
        + "Evidence-based guidance helps teams make informed decisions. " * 160
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
