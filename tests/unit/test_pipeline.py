from contextlib import contextmanager

from langgraph.checkpoint.memory import MemorySaver

from aca.models.content_brief import BriefSection, ContentBrief
from aca.models.ground_truth import ClientInput, GroundTruthGuide
from aca.models.prompt_intelligence import (
    AnswerUnderstanding,
    PromptAnalysis,
    PromptUnderstanding,
)
from aca.models.writer import WriterInput
from aca.orchestrator import pipeline as workflow
from aca.orchestrator.pipeline import PipelineState, _build_writer_input, build_pipeline
from aca.agents.writer import WriterAgent
from aca.models.reviewer import ReviewOutput
from aca.orchestrator.checkpointer import thread_config
from ..agent_fixtures import (
    agent1_output,
    agent2_output,
    agent3_output,
    agent4_output,
    agent5_output,
)


def _use_memory_checkpointer(monkeypatch):
    @contextmanager
    def memory_checkpointer():
        yield MemorySaver()

    monkeypatch.setattr(workflow, "postgres_checkpointer", memory_checkpointer)


def test_shared_agent_outputs_satisfy_schema_and_evidence_contracts():
    prompt = agent1_output()
    sources = agent2_output()
    guide = agent3_output()
    brief = agent4_output()
    article = agent5_output()

    assert prompt.prompt_understanding.intent
    assert prompt.prompt_understanding.topic
    assert prompt.prompt_id == sources.prompt_id == "prompt-1"
    assert sources.citation_summary.unique_sources == len(sources.ranked_sources)
    assert {source.source_id for source in sources.content_analyses} <= {
        source.source_id for source in sources.ranked_sources
    }

    evidence_ids = {item.evidence_id for item in guide.evidence}
    assert guide.verification is not None and guide.verification.is_verified
    assert all(set(fact.evidence_ids) <= evidence_ids for fact in guide.verified_facts)
    assert 4 <= len(brief.sections) <= 6
    assert all(set(section.evidence_ids) <= evidence_ids for section in brief.sections)

    writer_input = _build_writer_input(
        PipelineState(
            campaign_id=1,
            prompt_id=prompt.prompt_id,
            website="https://example.com",
            company_name=prompt.customer_name,
            prompt_analysis=prompt,
            guide=guide,
            content_brief=brief,
        )
    )
    assert isinstance(writer_input, WriterInput)
    assert writer_input.evidence[0].wording_boundary == (
        "Do not infer clinical outcomes from this assay description."
    )
    assert article.title == brief.recommended_title
    assert len(article.meta_description) <= 160
    WriterAgent._validate_seo_requirements(article, writer_input)
    WriterAgent._validate_evidence_references(
        article,
        {item.evidence_id for item in writer_input.evidence},
    )


def test_pipeline_graph_contains_five_core_agents_and_mandatory_reviewer():
    mermaid = build_pipeline().get_graph().draw_mermaid()

    for agent in range(1, 6):
        assert f"agent{agent}(agent{agent})" in mermaid
    assert "agent6(agent6)" in mermaid

    assert "agent1 --> agent2;" in mermaid
    assert "agent2 --> agent3;" in mermaid
    assert "agent4 -.-> agent5;" in mermaid


def test_pipeline_passes_brief_seo_research_and_faqs_to_writer():
    state = PipelineState(
        campaign_id=1,
        prompt_id="prompt-1",
        website="https://example.com",
        prompt_analysis=PromptAnalysis(
            prompt_id="prompt-1",
            prompt="How does immersive training work?",
            customer_name="Example",
            prompt_understanding=PromptUnderstanding(
                intent="Learn how immersive training works",
                topic="immersive training",
                user_goal="Understand the approach",
            ),
            answer_understanding=AnswerUnderstanding(summary=""),
            customer_visibility={},
        ),
        guide=GroundTruthGuide(main_topic="immersive training"),
        content_brief=ContentBrief(
            recommended_title="Immersive Training",
            primary_keyword="immersive training",
            secondary_keywords=["procedural simulation"],
            related_topics=["training design"],
            question_based_queries=["How does immersive training work?"],
            faq_questions=[
                "What is immersive training?",
                "How is procedural simulation used?",
                "Who uses immersive training?",
            ],
            sections=[
                BriefSection(
                    heading="How it works",
                    level="H2",
                    section_objective="Explain the process to a first-time reader.",
                    topics_and_questions=[
                        "What happens during a training session?",
                        "How is feedback provided?",
                    ],
                    evidence_ids=["ev-001"],
                    guidelines_and_gaps=["Do not claim measured outcomes without evidence."],
                )
            ],
        ),
        target_word_count=1200,
    )

    writer_input = _build_writer_input(state)

    assert writer_input.primary_keyword == "immersive training"
    assert writer_input.secondary_keywords == ["procedural simulation"]
    assert writer_input.related_topics == ["training design"]
    assert writer_input.question_based_queries == ["How does immersive training work?"]
    section = writer_input.sections[0]
    assert section.heading == "How it works"
    assert section.level == 2
    assert section.objective == "Explain the process to a first-time reader."
    assert section.topics == [
        "What happens during a training session?",
        "How is feedback provided?",
    ]
    assert section.questions_to_answer == section.topics
    assert section.evidence_ids == ["ev-001"]
    assert section.guidelines == ["Do not claim measured outcomes without evidence."]
    assert writer_input.target_word_count == 1200
    assert [faq["question"] for faq in writer_input.faqs] == [
        "What is immersive training?",
        "How is procedural simulation used?",
        "Who uses immersive training?",
    ]


def test_pipeline_hands_typed_outputs_through_all_five_agents(monkeypatch):
    outputs = {
        "agent1": agent1_output(),
        "agent2": agent2_output(),
        "agent3": agent3_output(),
        "agent4": agent4_output(),
        "agent5": agent5_output(),
    }
    calls = []

    def run_agent1(campaign_id, prompt_id, company_name):
        calls.append(("agent1", campaign_id, prompt_id, company_name))
        return outputs["agent1"]

    def run_agent2(campaign_id, prompt_id, top_n):
        calls.append(("agent2", campaign_id, prompt_id, top_n))
        return outputs["agent2"]

    class FakeGroundTruthAgent:
        def __init__(self, max_questions):
            calls.append(("agent3_init", max_questions))

        def analyze(self, prompt, client):
            calls.append(("agent3", prompt, client))
            return outputs["agent3"]

    class FakeContentBriefAgent:
        def build_brief(self, prompt, sources, guide):
            calls.append(("agent4", prompt, sources, guide))
            return outputs["agent4"]

    class FakeWriterAgent:
        def write(self, writer_input):
            calls.append(("agent5", writer_input))
            return outputs["agent5"]

    monkeypatch.setattr(workflow, "run_agent1", run_agent1)
    monkeypatch.setattr(workflow, "run_agent2", run_agent2)
    monkeypatch.setattr(workflow, "GroundTruthAgent", FakeGroundTruthAgent)
    monkeypatch.setattr(workflow, "ContentBriefAgent", FakeContentBriefAgent)
    monkeypatch.setattr(workflow, "WriterAgent", FakeWriterAgent)

    snapshots = workflow.build_pipeline().stream(
        PipelineState(
            campaign_id=1,
            prompt_id="prompt-1",
            website="https://example.com",
            top_n=2,
            max_questions=3,
            target_word_count=1250,
        ),
        stream_mode="values",
    )
    result = PipelineState.model_validate(list(snapshots)[-1])

    assert [call[0] for call in calls] == ["agent1", "agent2", "agent3_init", "agent3", "agent4", "agent5"]
    assert calls[0][1:] == (1, "prompt-1", None)
    assert calls[1][1:] == (1, "prompt-1", 2)
    assert calls[3][1] == outputs["agent1"]
    assert calls[3][2] == ClientInput(
        company_name="Example Bio",
        website="https://example.com",
    )
    assert calls[4][1:] == (outputs["agent1"], outputs["agent2"], outputs["agent3"])

    writer_input = calls[5][1]
    assert writer_input.company_name == "Example Bio"
    assert writer_input.sections[0].heading == outputs["agent4"].sections[0].heading
    assert writer_input.sections[0].objective == outputs["agent4"].sections[0].section_objective
    assert writer_input.sections[0].topics == outputs["agent4"].sections[0].topics_and_questions
    assert writer_input.sections[0].guidelines == outputs["agent4"].sections[0].guidelines_and_gaps
    assert writer_input.target_word_count == 1250
    assert writer_input.evidence[0].evidence_id == "ev-001"
    assert writer_input.evidence[0].status == "verified"
    assert writer_input.evidence[0].source_url == "https://example.com/atlas-panel"
    assert result.prompt_analysis == outputs["agent1"]
    assert result.source_analysis == outputs["agent2"]
    assert result.guide == outputs["agent3"]
    assert result.content_brief == outputs["agent4"]
    assert result.written_article == outputs["agent5"]
    assert result.company_name == "Example Bio"
    assert result.errors == []


def test_failed_agent3_verification_stops_before_agents4_and5(monkeypatch):
    _use_memory_checkpointer(monkeypatch)
    guide = agent3_output(verified=False)
    calls = []
    monkeypatch.setattr(workflow, "run_agent1", lambda *args: agent1_output())
    monkeypatch.setattr(workflow, "run_agent2", lambda *args, **kwargs: agent2_output())

    class FakeGroundTruthAgent:
        def __init__(self, **kwargs):
            pass

        def analyze(self, *args):
            return guide

    class UnexpectedAgent4:
        def __init__(self):
            calls.append("agent4")

    monkeypatch.setattr(workflow, "GroundTruthAgent", FakeGroundTruthAgent)
    monkeypatch.setattr(workflow, "ContentBriefAgent", UnexpectedAgent4)
    monkeypatch.setattr(workflow, "WriterAgent", UnexpectedAgent4)

    result = workflow.run_workflow(1, "prompt-1", "https://example.com")

    assert result.guide == guide
    assert result.content_brief is None
    assert result.written_article is None
    assert calls == []
    assert result.errors == [
        "Prompt/customer verification did not pass; Agents 4 and 5 were skipped. "
        "The requested capability was not verified."
    ]


def test_empty_agent4_brief_stops_before_agent5(monkeypatch):
    _use_memory_checkpointer(monkeypatch)
    calls = []
    monkeypatch.setattr(workflow, "run_agent1", lambda *args: agent1_output())
    monkeypatch.setattr(workflow, "run_agent2", lambda *args, **kwargs: agent2_output())

    class FakeGroundTruthAgent:
        def __init__(self, **kwargs):
            pass

        def analyze(self, *args):
            return agent3_output()

    class FakeContentBriefAgent:
        def build_brief(self, *args):
            return agent4_output(include_sections=False)

    class UnexpectedWriter:
        def __init__(self):
            calls.append("agent5")

    monkeypatch.setattr(workflow, "GroundTruthAgent", FakeGroundTruthAgent)
    monkeypatch.setattr(workflow, "ContentBriefAgent", FakeContentBriefAgent)
    monkeypatch.setattr(workflow, "WriterAgent", UnexpectedWriter)

    result = workflow.run_workflow(1, "prompt-1", "https://example.com")

    assert result.content_brief is not None
    assert result.content_brief.sections == []
    assert result.written_article is None
    assert calls == []
    assert result.errors == ["Agent 4 returned no usable sections. Agent 5 was not run."]


def test_agent1_errors_are_propagated_without_running_later_agents(monkeypatch):
    _use_memory_checkpointer(monkeypatch)
    error = ValueError("Prompt was not found.")
    calls = []

    def fail_agent1(*args):
        raise error

    def unexpected_agent2(*args, **kwargs):
        calls.append("agent2")

    monkeypatch.setattr(workflow, "run_agent1", fail_agent1)
    monkeypatch.setattr(workflow, "run_agent2", unexpected_agent2)

    try:
        workflow.run_workflow(1, "prompt-1", "https://example.com")
    except ValueError as exc:
        assert exc is error
    else:
        raise AssertionError("Agent 1 failure must propagate to the workflow caller.")

    assert calls == []


def test_workflow_resumes_at_first_missing_agent_output(monkeypatch):
    outputs = {
        "agent1": agent1_output(),
        "agent2": agent2_output(),
        "agent3": agent3_output(),
        "agent4": agent4_output(),
        "agent5": agent5_output(),
    }
    calls = []

    def unexpected_agent1(*args):
        calls.append("agent1")
        raise AssertionError("A completed Agent 1 output must be reused.")

    monkeypatch.setattr(workflow, "run_agent1", unexpected_agent1)
    monkeypatch.setattr(
        workflow,
        "run_agent2",
        lambda *args, **kwargs: calls.append("agent2") or outputs["agent2"],
    )

    class FakeGroundTruthAgent:
        def __init__(self, **kwargs):
            pass

        def analyze(self, *args):
            calls.append("agent3")
            return outputs["agent3"]

    class FakeContentBriefAgent:
        def build_brief(self, *args):
            calls.append("agent4")
            return outputs["agent4"]

    class FakeWriterAgent:
        def write(self, *args):
            calls.append("agent5")
            return outputs["agent5"]

    monkeypatch.setattr(workflow, "GroundTruthAgent", FakeGroundTruthAgent)
    monkeypatch.setattr(workflow, "ContentBriefAgent", FakeContentBriefAgent)
    monkeypatch.setattr(workflow, "WriterAgent", FakeWriterAgent)

    state = PipelineState(
        campaign_id=1,
        prompt_id="prompt-1",
        website="https://example.com",
        company_name="Example Bio",
        prompt_analysis=outputs["agent1"],
    )
    final = build_pipeline().invoke(state)

    assert calls == ["agent2", "agent3", "agent4", "agent5"]
    assert final["prompt_analysis"] == outputs["agent1"]
    assert final["written_article"] == outputs["agent5"]


def test_requested_review_runs_only_after_existing_five_agent_outputs(monkeypatch):
    draft = agent5_output()
    review = ReviewOutput(summary="Review complete.", findings=[])
    calls = []

    class FakeReviewerAgent:
        def review(self, writer_input, existing_draft):
            calls.append((writer_input, existing_draft))
            return review

    monkeypatch.setattr(workflow, "ReviewerAgent", FakeReviewerAgent)

    checkpointer = MemorySaver()
    graph = build_pipeline(checkpointer)
    config = thread_config("checkpoint-run")
    result = graph.invoke(
        PipelineState(
            run_id="checkpoint-run",
            campaign_id=1,
            prompt_id="prompt-1",
            website="https://example.com",
            company_name="Example Bio",
            prompt_analysis=agent1_output(),
            source_analysis=agent2_output(),
            guide=agent3_output(),
            content_brief=agent4_output(),
            written_article=draft,
            review_requested=False,
        ),
        config=config,
    )

    assert result["review_output"] == review
    assert len(calls) == 1
    assert calls[0][1] == draft
    assert calls[0][0].company_name == "Example Bio"
    saved_state = graph.get_state(config)
    assert saved_state.values["review_output"] == review
    graph.update_state(config, {"draft_approval": {"fingerprint": "approved"}})
    assert graph.get_state(config).values["draft_approval"] == {"fingerprint": "approved"}
