import json

import pytest

from aca.agents.content_brief import ContentBriefAgent, _extract_json


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
    encoded = json.dumps(output)
    response_text = encoded[len('{"article_objective":'):]
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        @staticmethod
        def json():
            return {"content": [{"type": "text", "text": response_text}]}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("aca.agents.content_brief.httpx.post", fake_post)
    brief = ContentBriefAgent(api_key="test").build_brief(*_brief_inputs())

    assert brief.primary_keyword == "immersive training"
    assert brief.secondary_keywords == ["procedural simulation"]
    assert brief.related_topics == ["Training design"]
    assert brief.question_based_queries == ["How does immersive training work?"]
    assert len(brief.faq_questions) == 3
    user_payload = json.loads(captured["json"]["messages"][0]["content"])
    assert user_payload["agent3_ground_truth"]["question_based_queries"] == [
        "How does immersive training work?"
    ]


def test_content_brief_retries_once_when_model_returns_malformed_json(monkeypatch):
    valid_brief = {
        "article_objective": "Explain the topic",
        "sections": [],
    }
    responses = [
        {"content": [{"type": "text", "text": "not valid JSON"}]},
        {"content": [{"type": "text", "text": json.dumps(valid_brief)}]},
    ]
    requests = []

    class FakeResponse:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    def fake_post(url, **kwargs):
        requests.append(kwargs["json"])
        return FakeResponse(responses.pop(0))

    monkeypatch.setattr("aca.agents.content_brief.httpx.post", fake_post)
    brief = ContentBriefAgent(api_key="test").build_brief(*_brief_inputs())

    assert brief.article_objective == "Explain the topic"
    assert len(requests) == 2
    assert requests[1]["max_tokens"] == 8192
    assert "syntactically valid JSON" in requests[1]["messages"][0]["content"]


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
    response_text = json.dumps(output)[len('{"article_objective":'):]

    class FakeResponse:
        def raise_for_status(self):
            pass

        @staticmethod
        def json():
            return {"content": [{"type": "text", "text": response_text}]}

    monkeypatch.setattr(
        "aca.agents.content_brief.httpx.post",
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


def test_content_brief_rejects_unknown_evidence_ids(monkeypatch):
    output = {
        "article_objective": "Plan article",
        "sections": [{"heading": "Unsupported", "evidence_ids": ["ev_missing"]}],
    }
    response_text = json.dumps(output)[len('{"article_objective":'):]

    class FakeResponse:
        def raise_for_status(self):
            pass

        @staticmethod
        def json():
            return {"content": [{"type": "text", "text": response_text}]}

    monkeypatch.setattr(
        "aca.agents.content_brief.httpx.post",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    with pytest.raises(RuntimeError, match="unknown evidence IDs"):
        ContentBriefAgent(api_key="test").build_brief(*_brief_inputs())


def test_extract_json_reports_error_location_and_context():
    with pytest.raises(ValueError, match=r"line 1, column"):
        _extract_json('{"article_objective": ]}')
