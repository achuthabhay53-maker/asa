from io import BytesIO
from types import SimpleNamespace

import pytest
from reportlab.pdfgen import canvas

from src.agents.general_blog.brief import GeneralBlogBriefAgent
from src.agents.general_blog.company_research import CompanyWebsiteResearchAgent
from src.agents.general_blog.content_strategy import GeneralBlogIntakeAgent
from src.agents.general_blog.reference_research import UserReferenceResearchAgent
from src.agents.aimpact_blog.writer import WriterAgent
from src.utils.general_blog_research import extract_source_evidence
from src.utils.general_blog_human import (
    edited_writer_output,
    result_after_draft_change,
    review_passed,
)
from src.observability.token_usage import record_claude_response
from src.orchestrator.general_blog import (
    GeneralBlogWorkflow,
    GeneralBlogWorkflowError,
)
from src.tools.reference_fetcher import (
    RetrievedSource,
    SourceFetchError,
    _validate_public_url,
    fetch_reference,
    fetch_uploaded_pdf,
)
from src.models.general_blog import (
    GeneralBlogBrief,
    GeneralBlogEvidence,
    GeneralBlogPlan,
    GeneralBlogRequest,
    GeneralBlogResult,
    UploadedPDFStatus,
)
from src.models.writer import WriterEvidence
from src.models.reviewer import ReviewFinding, ReviewOutput
from src.models.writer import WriterInput, WriterOutput


def _plan():
    return GeneralBlogPlan(
        article_objective="Explain the requested topic.",
        search_intent="Understand the topic.",
        target_audience="Readers",
        tone="Clear",
        title="A Helpful Guide",
        primary_keyword="helpful guide",
        research_queries=["Example company topic"],
    )


def _evidence(evidence_id="ev_001", source_type="company_website", relevance="direct"):
    return GeneralBlogEvidence(
        evidence_id=evidence_id,
        claim="The source describes the company's capability.",
        relevance=relevance,
        evidence_quote="This exact source quote describes the company's capability in detail.",
        source_url="https://example.com/capability",
        source_title="Capability",
        source_type=source_type,
    )


def test_human_edit_updates_article_sections_and_invalidates_previous_review():
    original_draft = WriterOutput(
        title="Old title",
        article_markdown="# Old title\n\n## First section\n\nOld content.",
        word_count=7,
        sections=[{"heading": "First section", "content_markdown": "Old content."}],
        claims_used=[{"claim": "Old claim"}],
    )
    updated_draft = edited_writer_output(
        original_draft,
        "# New title\n\n## First section\n\nUpdated content.\n\n## Second section\n\nMore details.",
    )
    brief = GeneralBlogBrief(
        title="A Helpful Guide",
        article_objective="Explain the topic.",
        target_audience="Readers",
        search_intent="Understand the topic.",
        tone="Clear",
        primary_keyword="helpful guide",
        sections=[{"heading": "Overview", "objective": "Explain the topic."}],
    )
    result = GeneralBlogResult(
        plan=_plan(),
        brief=brief,
        writer_input=WriterInput(
            company_name="Example",
            title=brief.title,
            article_objective=brief.article_objective,
            target_audience=brief.target_audience,
            sections=[{"heading": "Overview"}],
        ),
        draft=original_draft,
        review=ReviewOutput(summary="Passed"),
        quality_passed=True,
    )

    updated_result = result_after_draft_change(result, updated_draft)

    assert updated_draft.title == "New title"
    assert [section.heading for section in updated_draft.sections] == [
        "First section",
        "Second section",
    ]
    assert updated_draft.claims_used == []
    assert updated_result.review_current is False
    assert updated_result.quality_passed is False
    assert "changed after this review" in updated_result.review.summary


def test_human_review_gate_matches_automatic_blocking_findings():
    assert review_passed(ReviewOutput(summary="Clear"))
    assert not review_passed(
        ReviewOutput(
            summary="Needs work",
            findings=[
                ReviewFinding(
                    area="content",
                    severity="high",
                    issue="Issue",
                    recommendation="Fix it",
                )
            ],
        )
    )
    assert not review_passed(
        ReviewOutput(
            summary="Needs work",
            unresolved_items=["Claim is not assessed."],
        )
    )


def test_general_blog_request_normalizes_websites_and_reference_urls():
    request = GeneralBlogRequest(
        company_name=" Example ",
        company_website="example.com",
        prompt="Write about training https://research.example/paper.pdf.",
        reference_urls=["research.example/paper.pdf", "https://research.example/paper.pdf"],
    )

    assert request.company_website == "https://example.com/"
    assert request.reference_urls == ["https://research.example/paper.pdf"]
    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        GeneralBlogRequest(
            company_name="Example",
            company_website="file:///etc/passwd",
            prompt="Write about training",
        )


def test_intake_agent_collects_reference_urls_in_prompt_and_explicit_field():
    request = GeneralBlogRequest(
        company_name="Example",
        company_website="example.com",
        prompt="Discuss training. Reference: https://papers.example/study.pdf.",
        target_audience="Engineers",
        tone="Friendly",
        reference_urls=[
            "https://papers.example/study.pdf",
            "https://example.net/report",
            "https://example.com/company-paper.pdf",
        ],
    )
    expected_plan = _plan().model_dump(mode="json")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            stop_reason="tool_use",
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="submit_general_blog_plan",
                    input=expected_plan,
                )
            ],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    agent = GeneralBlogIntakeAgent(
        client=SimpleNamespace(messages=SimpleNamespace(create=create)),
        model="test",
    )
    plan, urls = agent.prepare(request)

    assert plan.title == "A Helpful Guide"
    assert urls == [
        "https://papers.example/study.pdf",
        "https://example.net/report",
        "https://example.com/company-paper.pdf",
    ]
    assert plan.target_audience == "Engineers"
    assert plan.tone == "Friendly"
    assert calls[0]["tool_choice"] == {
        "type": "tool",
        "name": "submit_general_blog_plan",
    }


def test_fetch_reference_extracts_html_title_published_date_and_visible_text(monkeypatch):
    html = b"""<html><head><title>Research Paper</title>
    <meta property="article:published_time" content="2025-03-04"></head>
    <body><script>hidden secret</script><h1>Findings</h1>
    <p>The study describes training outcomes in the tested group.</p></body></html>"""
    monkeypatch.setattr(
        "src.tools.reference_fetcher._read_reference_bytes",
        lambda _url: ("https://papers.example/article", html, "text/html"),
    )

    source = fetch_reference("https://papers.example/article")

    assert source.title == "Research Paper"
    assert source.publication_date == "2025-03-04"
    assert "The study describes training outcomes" in source.text
    assert "hidden secret" not in source.text


def test_fetch_reference_extracts_pdf_text_and_metadata(monkeypatch):
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer)
    pdf.setTitle("Research Report")
    pdf.drawString(72, 720, "Research paper evidence about the tested population.")
    pdf.save()
    content = pdf_buffer.getvalue()
    monkeypatch.setattr(
        "src.tools.reference_fetcher._read_reference_bytes",
        lambda _url: ("https://papers.example/report.pdf", content, "application/pdf"),
    )

    source = fetch_reference("https://papers.example/report.pdf")

    assert source.title == "Research Report"
    assert "Research paper evidence about the tested population." in source.text


def test_fetch_uploaded_pdf_extracts_text_without_creating_a_source_url():
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer)
    pdf.setTitle("Uploaded Research")
    pdf.drawString(
        72,
        720,
        "The uploaded study included 80 participants in the training cohort.",
    )
    pdf.save()

    source = fetch_uploaded_pdf("training-study.pdf", pdf_buffer.getvalue())

    assert source.url is None
    assert source.title == "Uploaded Research"
    assert "The uploaded study included 80 participants" in source.text


def test_fetch_uploaded_pdf_rejects_non_pdf_and_oversized_files(monkeypatch):
    with pytest.raises(SourceFetchError, match="valid PDF"):
        fetch_uploaded_pdf("report.pdf", b"not a PDF")
    with pytest.raises(SourceFetchError, match="must be PDF"):
        fetch_uploaded_pdf("report.txt", b"%PDF-1.4")
    monkeypatch.setattr("src.tools.reference_fetcher.MAX_REFERENCE_BYTES", 8)
    with pytest.raises(SourceFetchError, match="15 MB limit"):
        fetch_uploaded_pdf("report.pdf", b"%PDF-12345")


def test_reference_fetch_blocks_local_addresses():
    with pytest.raises(SourceFetchError, match="Private and reserved"):
        _validate_public_url("http://127.0.0.1/private")


def test_shared_source_analysis_discards_claims_without_exact_source_quotes():
    source_text = (
        "The study included 120 participants and reported its measured outcome "
        "after twelve weeks of observation."
    )
    output = {
        "summary": "A study with a defined participant group and follow-up period.",
        "evidence": [
            {
                "claim": "The study included 120 participants.",
                "quote": "The study included 120 participants and reported its measured outcome",
                "wording_boundary": "Only describes the study participant count.",
                "relevance": "direct",
            },
            {
                "claim": "The treatment doubled the outcome.",
                "quote": "An invented quote that is absent from the source material.",
                "wording_boundary": "No limits.",
                "relevance": "related",
            },
            {
                "claim": "The report describes an outcome after twelve weeks.",
                "quote": "reported its measured outcome after twelve weeks of observation.",
                "wording_boundary": "Background detail from the study.",
                "relevance": "background",
            },
        ],
    }

    def create(**kwargs):
        assert kwargs["tool_choice"] == {
            "type": "tool",
            "name": "submit_source_evidence",
        }
        return SimpleNamespace(
            stop_reason="tool_use",
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="submit_source_evidence",
                    input=output,
                )
            ],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    agent = UserReferenceResearchAgent(
        client=SimpleNamespace(messages=SimpleNamespace(create=create)),
        model="test",
    )
    evidence = extract_source_evidence(
        agent.client,
        agent.model,
        RetrievedSource(
            url="https://papers.example/study.pdf",
            title="Study",
            text=source_text,
            publication_date="2025",
        ),
        "user_reference",
        0,
        "Discuss the study.",
    )

    assert len(evidence) == 2
    assert evidence[0].evidence_id == "ev_001"
    assert evidence[0].source_type == "user_reference"
    assert evidence[0].publication_date == "2025"
    assert evidence[0].relevance == "direct"
    assert evidence[0].status == "verified"
    assert evidence[1].evidence_id == "ev_002"
    assert evidence[1].relevance == "background"


def test_company_research_searches_only_company_domain_and_validates_evidence():
    company_text = (
        "Example's training platform supports procedural practice for learners "
        "through repeatable interactive scenarios."
    )
    analysis = {
        "summary": "The company page explains its training platform.",
        "evidence": [
            {
                "claim": "Example's platform supports procedural practice.",
                "quote": "Example's training platform supports procedural practice for learners",
                "wording_boundary": "Only describes the stated platform capability.",
                "relevance": "direct",
            }
        ],
    }
    calls = []

    class FakeResearch:
        def search(self, query, **kwargs):
            calls.append((query, kwargs))
            return SimpleNamespace(
                failed=False,
                results=[
                    SimpleNamespace(
                        url="https://example.com/training",
                        title="Training",
                        published="2025-01-01",
                    ),
                    SimpleNamespace(
                        url="https://unrelated.example/training",
                        title="Unrelated",
                        published=None,
                    ),
                ],
            )

        def extract(self, url, *, max_chars):
            assert url == "https://example.com/training"
            assert max_chars > 0
            return SimpleNamespace(
                url=url,
                title="Training",
                text=company_text,
            )

    def create(**kwargs):
        assert kwargs["tool_choice"] == {
            "type": "tool",
            "name": "submit_source_evidence",
        }
        return SimpleNamespace(
            stop_reason="tool_use",
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="submit_source_evidence",
                    input=analysis,
                )
            ],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    agent = CompanyWebsiteResearchAgent(
        client=SimpleNamespace(messages=SimpleNamespace(create=create)),
        research=FakeResearch(),
        model="test",
    )
    request = GeneralBlogRequest(
        company_name="Example",
        company_website="https://example.com",
        prompt="Explain procedural training.",
    )

    evidence, warnings = agent.research_company(request, _plan(), evidence_offset=2)

    assert len(calls) == 1
    assert calls[0][1]["include_domains"] == ["example.com"]
    assert evidence[0].evidence_id == "ev_003"
    assert evidence[0].source_type == "company_website"
    assert evidence[0].source_url == "https://example.com/training"
    assert warnings == []


def test_reference_agent_returns_evidence_and_reports_unavailable_sources(monkeypatch):
    source_text = (
        "The report describes 80 participants and the follow-up period "
        "used during the study."
    )
    analysis = {
        "summary": "The report summarizes a study.",
        "evidence": [
            {
                "claim": "The study included 80 participants.",
                "quote": "The report describes 80 participants and the follow-up period",
                "wording_boundary": "The quote only states participant count.",
                "relevance": "related",
            }
        ],
    }
    monkeypatch.setattr(
        "src.agents.general_blog.reference_research.fetch_reference",
        lambda url: RetrievedSource(url, "Report", source_text, "2024"),
    )

    def create(**_kwargs):
        return SimpleNamespace(
            stop_reason="tool_use",
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="submit_source_evidence",
                    input=analysis,
                )
            ],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    agent = UserReferenceResearchAgent(
        client=SimpleNamespace(messages=SimpleNamespace(create=create)),
        model="test",
    )
    request = GeneralBlogRequest(
        company_name="Example",
        company_website="https://example.com",
        prompt="Discuss the report.",
    )

    evidence, warnings, pdf_statuses = agent.research_references(
        request,
        ["https://papers.example/report.pdf"],
        evidence_offset=4,
    )

    assert len(evidence) == 1
    assert evidence[0].evidence_id == "ev_005"
    assert evidence[0].source_type == "user_reference"
    assert evidence[0].publication_date == "2024"
    assert warnings == []
    assert pdf_statuses == []

    def fail(_url):
        raise SourceFetchError("The PDF could not be parsed.")

    monkeypatch.setattr("src.agents.general_blog.reference_research.fetch_reference", fail)
    evidence, warnings, pdf_statuses = agent.research_references(
        request,
        ["https://papers.example/broken.pdf"],
    )
    assert evidence == []
    assert "could not be incorporated" in warnings[0]
    assert pdf_statuses == []


def test_reference_agent_extracts_evidence_from_uploaded_pdf():
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer)
    pdf.setTitle("Uploaded Study")
    pdf.drawString(
        72,
        720,
        "The uploaded study included 80 participants in the training cohort.",
    )
    pdf.save()
    source_quote = "The uploaded study included 80 participants in the training cohort."
    analysis = {
        "summary": "The PDF reports a study participant group.",
        "evidence": [
            {
                "claim": "The study included 80 participants.",
                "quote": source_quote,
                "wording_boundary": "Only describes the study participant group.",
                "relevance": "direct",
            }
        ],
    }

    def create(**_kwargs):
        return SimpleNamespace(
            stop_reason="tool_use",
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="submit_source_evidence",
                    input=analysis,
                )
            ],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    agent = UserReferenceResearchAgent(
        client=SimpleNamespace(messages=SimpleNamespace(create=create)),
        model="test",
    )
    request = GeneralBlogRequest(
        company_name="Example",
        company_website="https://example.com",
        prompt="Discuss the study.",
    )

    evidence, warnings, pdf_statuses = agent.research_references(
        request,
        [],
        uploaded_pdfs=[("training-study.pdf", pdf_buffer.getvalue())],
    )

    assert len(evidence) == 1
    assert evidence[0].source_url is None
    assert evidence[0].source_title == "Uploaded Study"
    assert evidence[0].evidence_quote == source_quote
    assert evidence[0].source_type == "user_reference"
    assert warnings == []
    assert pdf_statuses == [
        UploadedPDFStatus(
            file_name="training-study.pdf",
            status="parsed_with_evidence",
            extracted_text_characters=len(source_quote),
            evidence_count=1,
            message="PDF parsed successfully; 1 evidence item(s) extracted.",
        )
    ]


def test_reference_agent_reports_uploaded_pdf_parse_failure(monkeypatch):
    def fail(_file_name, _content):
        raise SourceFetchError("The PDF contains no extractable text.")

    monkeypatch.setattr(
        "src.agents.general_blog.reference_research.fetch_uploaded_pdf",
        fail,
    )
    agent = UserReferenceResearchAgent(
        client=SimpleNamespace(messages=SimpleNamespace(create=lambda **_kwargs: None)),
        model="test",
    )
    request = GeneralBlogRequest(
        company_name="Example",
        company_website="https://example.com",
        prompt="Discuss the study.",
    )

    evidence, warnings, pdf_statuses = agent.research_references(
        request,
        [],
        uploaded_pdfs=[("scanned-study.pdf", b"%PDF-test")],
    )

    assert evidence == []
    assert "could not be incorporated" in warnings[0]
    assert pdf_statuses[0].status == "failed"
    assert "no extractable text" in pdf_statuses[0].message


def test_reference_agent_reports_pdf_parsed_without_evidence(monkeypatch):
    monkeypatch.setattr(
        "src.agents.general_blog.reference_research.fetch_uploaded_pdf",
        lambda file_name, _content: RetrievedSource(
            None,
            file_name,
            "Readable document text with enough words to be processed.",
        ),
    )

    def create(**_kwargs):
        return SimpleNamespace(
            stop_reason="tool_use",
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="submit_source_evidence",
                    input={"summary": "No relevant facts.", "evidence": []},
                )
            ],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    agent = UserReferenceResearchAgent(
        client=SimpleNamespace(messages=SimpleNamespace(create=create)),
        model="test",
    )
    request = GeneralBlogRequest(
        company_name="Example",
        company_website="https://example.com",
        prompt="Discuss the study.",
    )

    evidence, warnings, pdf_statuses = agent.research_references(
        request,
        [],
        uploaded_pdfs=[("background.pdf", b"%PDF-test")],
    )

    assert evidence == []
    assert "no topic-relevant" in warnings[0]
    assert pdf_statuses[0].status == "parsed_no_evidence"
    assert pdf_statuses[0].extracted_text_characters > 0
    assert pdf_statuses[0].evidence_count == 0


def test_brief_agent_maps_evidence_into_existing_writer_input():
    evidence = [
        _evidence(),
        _evidence("ev_002", "user_reference", "related"),
    ]
    output = {
        "title": "Training with Evidence",
        "article_objective": "Explain the topic.",
        "target_audience": "Readers",
        "search_intent": "Learn how it works.",
        "tone": "Clear",
        "primary_keyword": "training evidence",
        "secondary_keywords": ["training"],
        "related_topics": ["procedural practice"],
        "question_based_queries": ["How does training work?"],
        "sections": [
            {
                "heading": "Company approach",
                "objective": "Describe verified capability.",
                "topics": ["training"],
                "questions_to_answer": ["What approach is offered?"],
                "evidence_ids": ["ev_001"],
                "guidelines": ["Do not generalize beyond the quote."],
            }
        ],
        "faq_questions": ["What is the approach?"],
        "unsupported_claims_to_avoid": ["Guaranteed outcomes"],
    }
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            stop_reason="tool_use",
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="submit_general_blog_brief",
                    input=output,
                )
            ],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    agent = GeneralBlogBriefAgent(
        client=SimpleNamespace(messages=SimpleNamespace(create=create)),
        model="test",
    )
    request = GeneralBlogRequest(
        company_name="Example",
        company_website="example.com",
        prompt="Explain training.",
    )
    brief, writer_input = agent.build(request, _plan(), evidence)

    assert brief.title == "Training with Evidence"
    assert writer_input.company_name == "Example"
    assert writer_input.sections[0].evidence_ids == ["ev_001"]
    writer_prompt = WriterAgent._build_user_prompt(writer_input)
    assert '"relevance":"related"' in writer_prompt
    assert "Do not present background evidence as direct support." in writer_prompt
    assert "does not independently establish scientific correctness" in writer_prompt
    assert writer_input.evidence == [
        WriterEvidence(
            evidence_id="ev_001",
            claim=evidence[0].claim,
            status="verified",
            relevance="direct",
            evidence_quote=evidence[0].evidence_quote,
            source_url=evidence[0].source_url,
            wording_boundary=evidence[0].wording_boundary,
        ),
        WriterEvidence(
            evidence_id="ev_002",
            claim=evidence[1].claim,
            status="verified",
            relevance="related",
            evidence_quote=evidence[1].evidence_quote,
            source_url=evidence[1].source_url,
            wording_boundary=evidence[1].wording_boundary,
        ),
    ]
    assert calls[0]["tool_choice"] == {
        "type": "tool",
        "name": "submit_general_blog_brief",
    }


def test_brief_agent_rejects_unknown_evidence_ids():
    output = {
        "title": "Training with Evidence",
        "article_objective": "Explain the topic.",
        "target_audience": "Readers",
        "search_intent": "Learn how it works.",
        "tone": "Clear",
        "primary_keyword": "training evidence",
        "sections": [
            {
                "heading": "Company approach",
                "objective": "Describe capability.",
                "topics": [],
                "questions_to_answer": [],
                "evidence_ids": ["ev_fabricated"],
                "guidelines": [],
            }
        ],
        "faq_questions": [],
        "unsupported_claims_to_avoid": [],
    }

    def create(**_kwargs):
        return SimpleNamespace(
            stop_reason="tool_use",
            content=[
                SimpleNamespace(
                    type="tool_use",
                    name="submit_general_blog_brief",
                    input=output,
                )
            ],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    agent = GeneralBlogBriefAgent(
        client=SimpleNamespace(messages=SimpleNamespace(create=create)),
        model="test",
    )
    request = GeneralBlogRequest(
        company_name="Example",
        company_website="example.com",
        prompt="Explain training.",
    )
    with pytest.raises(RuntimeError, match="unknown evidence IDs"):
        agent.build(request, _plan(), [])


def test_general_blog_workflow_revises_blocking_findings_then_passes():
    request = GeneralBlogRequest(
        company_name="Example",
        company_website="https://example.com",
        prompt="Explain the company's training approach.",
    )
    evidence = _evidence()
    brief = GeneralBlogBrief(
        title="A Helpful Guide",
        article_objective="Explain the topic.",
        target_audience="Readers",
        search_intent="Understand the topic.",
        tone="Clear",
        primary_keyword="helpful guide",
        sections=[{"heading": "Overview", "objective": "Explain the topic."}],
        faq_questions=["What is the approach?"],
    )
    writer_input = WriterInput(
        company_name="Example",
        company_url="https://example.com",
        title=brief.title,
        article_objective=brief.article_objective,
        target_audience=brief.target_audience,
        sections=[{"heading": "Overview"}],
    )
    plan = _plan()
    draft = WriterOutput(
        title="A Helpful Guide",
        article_markdown="# A Helpful Guide",
        word_count=3,
    )
    issue = ReviewFinding(
        area="factual_claims",
        severity="high",
        issue="A factual statement is not supported.",
        recommendation="Remove or qualify the statement.",
    )

    class Intake:
        def prepare(self, _request):
            return plan, []

    class CompanyResearch:
        def research_company(self, *_args):
            return [evidence], []

    class ReferenceResearch:
        def research_references(self, _request, urls, evidence_offset, uploaded_pdfs):
            assert urls == []
            assert evidence_offset == 1
            assert uploaded_pdfs == [("training-study.pdf", b"%PDF-test")]
            return (
                [_evidence("ev_002", "user_reference")],
                [],
                [
                    UploadedPDFStatus(
                        file_name="training-study.pdf",
                        status="parsed_with_evidence",
                        extracted_text_characters=80,
                        evidence_count=1,
                        message="PDF parsed successfully; 1 evidence item(s) extracted.",
                    )
                ],
            )

    class BriefAgent:
        def build(self, _request, _plan, sources):
            assert [item.evidence_id for item in sources] == ["ev_001", "ev_002"]
            return brief, writer_input

    class Writer:
        revisions = []

        def write(self, _writer_input):
            record_claude_response(
                SimpleNamespace(usage=SimpleNamespace(input_tokens=2, output_tokens=3)),
                "test-model",
            )
            return draft

        def revise(self, _writer_input, previous_draft, instructions):
            self.revisions.append(instructions)
            record_claude_response(
                SimpleNamespace(usage=SimpleNamespace(input_tokens=2, output_tokens=3)),
                "test-model",
            )
            return previous_draft

    class Reviewer:
        calls = 0

        def review(self, _writer_input, _draft):
            self.calls += 1
            record_claude_response(
                SimpleNamespace(usage=SimpleNamespace(input_tokens=2, output_tokens=3)),
                "test-model",
            )
            return (
                ReviewOutput(summary="Needs correction", findings=[issue])
                if self.calls == 1
                else ReviewOutput(summary="Clear")
            )

    writer = Writer()
    reviewer = Reviewer()
    progress_events = []
    result = GeneralBlogWorkflow(
        intake_agent=Intake(),
        company_research_agent=CompanyResearch(),
        reference_research_agent=ReferenceResearch(),
        brief_agent=BriefAgent(),
        writer_agent=writer,
        reviewer_agent=reviewer,
        progress_callback=progress_events.append,
    ).run(
        request,
        uploaded_reference_pdfs=[("training-study.pdf", b"%PDF-test")],
    )

    assert result.quality_passed is True
    assert result.review_cycles == 2
    assert len(writer.revisions) == 1
    assert "factual_claims" in writer.revisions[0]
    assert {record.agent_name for record in result.token_usage} == {
        "agent5",
        "agent5_revision",
        "agent6",
    }
    assert sum(record.input_tokens for record in result.token_usage) == 8
    assert sum(record.output_tokens for record in result.token_usage) == 12
    assert [item.source_type for item in result.sources] == [
        "company_website",
        "user_reference",
    ]
    assert result.uploaded_pdf_statuses[0].status == "parsed_with_evidence"
    assert progress_events[-1].stage == "complete"
    assert any(
        event.stage == "reference_research" and event.status == "done"
        for event in progress_events
    )
    assert any(
        event.stage == "revision" and event.status == "done"
        for event in progress_events
    )


def test_general_blog_workflow_reports_stage_error_with_tokens_and_run_id():
    request = GeneralBlogRequest(
        company_name="Example",
        company_website="https://example.com",
        prompt="Explain the company's training approach.",
    )

    class Intake:
        def prepare(self, _request):
            return _plan(), []

    class CompanyResearch:
        def research_company(self, *_args):
            return [_evidence()], []

    class BriefAgent:
        def build(self, *_args):
            return (
                GeneralBlogBrief(
                    title="A Helpful Guide",
                    article_objective="Explain the topic.",
                    target_audience="Readers",
                    search_intent="Understand the topic.",
                    tone="Clear",
                    primary_keyword="helpful guide",
                    sections=[{"heading": "Overview", "objective": "Explain the topic."}],
                ),
                WriterInput(
                    company_name="Example",
                    company_url="https://example.com",
                    title="A Helpful Guide",
                    article_objective="Explain the topic.",
                    target_audience="Readers",
                    sections=[{"heading": "Overview"}],
                ),
            )

    class Writer:
        def write(self, _writer_input):
            record_claude_response(
                SimpleNamespace(usage=SimpleNamespace(input_tokens=5, output_tokens=7)),
                "test-model",
            )
            raise RuntimeError("The writer could not create a draft.")

    progress_events = []
    workflow = GeneralBlogWorkflow(
        intake_agent=Intake(),
        company_research_agent=CompanyResearch(),
        brief_agent=BriefAgent(),
        writer_agent=Writer(),
        progress_callback=progress_events.append,
    )

    with pytest.raises(GeneralBlogWorkflowError) as error:
        workflow.run(request)

    assert error.value.stage == "writer"
    assert error.value.cause.args == ("The writer could not create a draft.",)
    assert error.value.run_id
    assert len(error.value.token_usage) == 1
    assert error.value.token_usage[0].agent_name == "agent5"
    assert error.value.token_usage[0].run_id == error.value.run_id
    assert (progress_events[-1].stage, progress_events[-1].status) == (
        "writer",
        "error",
    )
    assert progress_events[-1].token_usage == error.value.token_usage
    assert any(
        event.stage == "reference_research" and event.status == "skipped"
        for event in progress_events
    )


def test_general_blog_workflow_marks_optional_stages_skipped():
    request = GeneralBlogRequest(
        company_name="Example",
        company_website="https://example.com",
        prompt="Explain the company's training approach.",
    )
    brief = GeneralBlogBrief(
        title="A Helpful Guide",
        article_objective="Explain the topic.",
        target_audience="Readers",
        search_intent="Understand the topic.",
        tone="Clear",
        primary_keyword="helpful guide",
        sections=[{"heading": "Overview", "objective": "Explain the topic."}],
    )
    writer_input = WriterInput(
        company_name="Example",
        company_url="https://example.com",
        title=brief.title,
        article_objective=brief.article_objective,
        target_audience=brief.target_audience,
        sections=[{"heading": "Overview"}],
    )
    draft = WriterOutput(
        title=brief.title,
        article_markdown="# A Helpful Guide",
        word_count=3,
    )

    class Intake:
        def prepare(self, _request):
            return _plan(), []

    class CompanyResearch:
        def research_company(self, *_args):
            return [_evidence()], []

    class BriefAgent:
        def build(self, *_args):
            return brief, writer_input

    class Writer:
        def write(self, _writer_input):
            return draft

    class Reviewer:
        def review(self, _writer_input, _draft):
            return ReviewOutput(summary="No issues found")

    progress_events = []
    result = GeneralBlogWorkflow(
        intake_agent=Intake(),
        company_research_agent=CompanyResearch(),
        brief_agent=BriefAgent(),
        writer_agent=Writer(),
        reviewer_agent=Reviewer(),
        progress_callback=progress_events.append,
    ).run(request)

    assert result.quality_passed is True
    assert any(
        event.stage == "reference_research" and event.status == "skipped"
        for event in progress_events
    )
    assert any(
        event.stage == "revision" and event.status == "skipped"
        for event in progress_events
    )
