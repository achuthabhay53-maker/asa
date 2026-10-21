from aca.agents import source_intelligence
from aca.agents.source_intelligence import SourceIntelligenceAgent
from aca.models.source_intelligence import ContentStructure


def test_source_findings_mark_only_exact_page_quotes_as_matched(monkeypatch):
    page_text = "The page says this exact sentence describes the service clearly."
    monkeypatch.setattr(
        "aca.agents.source_intelligence._fetch_page",
        lambda _url: {
            "status": "success",
            "final_url": "https://example.com/page",
            "title": "Example page",
            "text": page_text,
            "structure": ContentStructure(),
        },
    )

    agent = SourceIntelligenceAgent.__new__(SourceIntelligenceAgent)
    agent._describe_page = lambda **_kwargs: {
        "summary": "Page summary",
        "findings": [
            {
                "finding": "The page describes a service.",
                "evidence_quote": "The page says this exact sentence describes the service clearly.",
            },
            {
                "finding": "The page reports a performance result.",
                "evidence_quote": "A quote not present in the fetched text.",
            },
            {
                "finding": "The page mentions a date.",
                "evidence_quote": "short",
            },
        ],
    }

    analyzed, retry = agent._analyze_source(
        "source_001",
        {"url": "https://example.com/page"},
    )

    assert retry is None
    assert analyzed is not None
    findings = analyzed.content_analysis.findings
    assert [item.quote_found_on_page for item in findings] == [True, False, False]
    assert findings[0].evidence_quote == page_text
    assert analyzed.source_id == "source_001"
    assert analyzed.url == "https://example.com/page"


def test_run_agent2_uses_same_low_score_threshold_as_agent1(monkeypatch):
    observed = {}
    prompt_evidence = {"prompt_id": "selected-prompt", "sources": []}

    def get_prompt_evidence(**kwargs):
        observed.update(kwargs)
        return [prompt_evidence]

    class FakeSourceAgent:
        def __init__(self, top_n=None):
            pass

        def analyze(self, campaign_id, selected):
            assert selected is prompt_evidence
            return "analyzed"

    monkeypatch.setattr("aca.tools.aimpact.get_prompt_evidence", get_prompt_evidence)
    monkeypatch.setattr(source_intelligence, "SourceIntelligenceAgent", FakeSourceAgent)

    result = source_intelligence.run_agent2(1, "selected-prompt")

    assert result == "analyzed"
    assert observed == {"campaign_id": 1, "score_threshold": 70}
