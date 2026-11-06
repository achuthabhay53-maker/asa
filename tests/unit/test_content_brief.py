import json

import pytest

from src.agents.aimpact_blog.content_brief import ContentBriefAgent


def _tool_response(output):
    return {
        "stop_reason": "tool_use",
        "content": [
            {
                "type": "tool_use",
                "name": "submit_content_brief",
                "input": output,
            }
        ],
    }


def _brief_inputs():
    return (
        {
            "prompt": "How is immersive training used?",
            "prompt_understanding": {
                "intent": "Learn about immersive training",
                "topic": "immersive training",
                "key_questions": ["How does immersive training work?"],
            },
        },
        {},
        {
            "main_topic": "immersive training",
            "search_intent": "Learn how the technology works",
            "candidate_subtopics": [
                {
                    "title": "Training design",
                    "description": "Procedural simulations",
                    "suggested_questions": ["How does immersive training work?"],
                }
            ],
        },
    )


def test_content_brief_passes_agent3_search_signals_into_seo_fields(monkeypatch):
    output = {
        "article_objective": "Explain the topic",
        "target_audience": "Practitioners",
        "search_intent": "Learn how the technology works",
        "primary_keyword": "immersive training",
        "secondary_keywords": ["procedural simulation"],
        "related_topics": ["Training design"],
        "question_based_queries": ["How does immersive training work?"],
        "faq_questions": [
            "What is immersive training?",
            "How is procedural simulation used?",
            "Who benefits from immersive training?",
        ],
    }
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        @staticmethod
        def json():
            return _tool_response(output)

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("src.agents.aimpact_blog.content_brief.httpx.post", fake_post)
    brief = ContentBriefAgent(api_key="test").build_brief(*_brief_inputs())

    assert brief.primary_keyword == "immersive training"
    assert brief.secondary_keywords == ["procedural simulation"]
    assert brief.related_topics == ["Training design"]
    assert brief.question_based_queries == ["How does immersive training work?"]
    assert len(brief.faq_questions) == 3
    assert captured["json"]["tool_choice"] == {
        "type": "tool",
        "name": "submit_content_brief",
    }
    assert set(captured["json"]["tools"][0]["input_schema"]["required"]) == set(
        captured["json"]["tools"][0]["input_schema"]["properties"]
    )
    user_payload = json.loads(captured["json"]["messages"][0]["content"])
    assert user_payload["agent3_ground_truth"]["question_based_queries"] == [
        "How does immersive training work?"
    ]


def test_content_brief_fails_explicitly_without_required_tool_response(monkeypatch):
    requests = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"stop_reason": "end_turn", "content": []}

    def fake_post(url, **kwargs):
        requests.append(kwargs["json"])
        return FakeResponse()

    monkeypatch.setattr("src.agents.aimpact_blog.content_brief.httpx.post", fake_post)
    with pytest.raises(RuntimeError, match="required structured response"):
        ContentBriefAgent(api_key="test").build_brief(*_brief_inputs())
    assert len(requests) == 1


def test_content_brief_links_section_urls_to_valid_evidence_ids(monkeypatch):
    output = {
        "article_objective": "Plan article",
        "sections": [
            {
                "heading": "Evidence-based section",
                "evidence_ids": ["ev_001"],
                "source_urls": ["https://example.com/evidence"],
            }
        ]
    }
    class FakeResponse:
        def raise_for_status(self):
            pass

        @staticmethod
        def json():
            return _tool_response(output)

    monkeypatch.setattr(
        "src.agents.aimpact_blog.content_brief.httpx.post",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    agent1, agent2, agent3 = _brief_inputs()
    agent3["evidence"] = [
        {
            "evidence_id": "ev_001",
            "url": "https://example.com/evidence",
            "quote": "A directly sourced evidence excerpt.",
        }
    ]

    brief = ContentBriefAgent(api_key="test").build_brief(agent1, agent2, agent3)

    assert brief.sections[0].evidence_ids == ["ev_001"]
    assert brief.sections[0].source_urls == ["https://example.com/evidence"]


def test_content_brief_drops_urls_not_linked_to_section_evidence(monkeypatch, caplog):
    output = {
        "article_objective": "Plan article",
        "sections": [
            {
                "heading": "Evidence-based section",
                "evidence_ids": ["ev_001"],
                "source_urls": [
                    "https://example.com/unlinked",
                    "https://example.com/evidence",
                ],
            }
        ],
    }
    class FakeResponse:
        def raise_for_status(self):
            pass

        @staticmethod
        def json():
            return _tool_response(output)

    monkeypatch.setattr(
        "src.agents.aimpact_blog.content_brief.httpx.post",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    agent1, agent2, agent3 = _brief_inputs()
    agent3["evidence"] = [
        {
            "evidence_id": "ev_001",
            "url": "https://example.com/evidence",
            "quote": "A directly sourced evidence excerpt.",
        },
        {
            "evidence_id": "ev_002",
            "url": "https://example.com/unlinked",
            "quote": "Evidence not assigned to this section.",
        },
    ]

    brief = ContentBriefAgent(api_key="test").build_brief(agent1, agent2, agent3)

    assert brief.sections[0].evidence_ids == ["ev_001"]
    assert brief.sections[0].source_urls == ["https://example.com/evidence"]
    assert "dropping them" in caplog.text


def test_content_brief_rejects_unknown_evidence_ids(monkeypatch):
    output = {
        "article_objective": "Plan article",
        "sections": [{"heading": "Unsupported", "evidence_ids": ["ev_missing"]}],
    }
    class FakeResponse:
        def raise_for_status(self):
            pass

        @staticmethod
        def json():
            return _tool_response(output)

    monkeypatch.setattr(
        "src.agents.aimpact_blog.content_brief.httpx.post",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    with pytest.raises(RuntimeError, match="unknown evidence IDs"):
        ContentBriefAgent(api_key="test").build_brief(*_brief_inputs())
