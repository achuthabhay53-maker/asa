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
