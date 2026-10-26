import json

import httpx

from aca.tools import web_research
from aca.tools.web_research import WebResearch


def test_search_uses_exa_and_filters_domains():
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/article",
                        "title": "Example",
                        "publishedDate": "2026-10-16",
                        "text": "Exa extracted the page content.",
                    },
                    {"url": "https://other.example/article", "title": "Other"},
                    {"url": "https://excluded.com/article"},
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        research = WebResearch(
            client=client,
            api_key="test-key",
        )
        result = research.search(
            "example query",
            max_results=3,
            include_domains=["example.com"],
            exclude_domains=["excluded.com"],
        )

    assert [item.url for item in result.results] == ["https://example.com/article"]
    assert result.results[0].published == "2026-10-16"
    page = research.extract(result.results[0].url)
    assert page is not None
    assert page.text == "Exa extracted the page content."
    assert len(requests) == 1
    request = requests[0]
    assert request.url == "https://api.exa.ai/search"
    assert request.headers["x-api-key"] == "test-key"
    payload = json.loads(request.read())
    assert payload["includeDomains"] == ["example.com"]
    assert payload["excludeDomains"] == ["excluded.com"]
    assert payload["contents"] == {"text": {"maxCharacters": 12_000}}


def test_extract_fetches_and_parses_page_directly():
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "https://example.com/article"
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=(
                "<html><head><title>Example Article</title>"
                "<script>ignored()</script></head><body>"
                "<nav>Navigation</nav><main><h1>Useful title</h1>"
                "<p>Readable page text.</p></main></body></html>"
            ),
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        research = WebResearch(client=client, api_key="test-key")
        page = research.extract("https://example.com/article")

    assert page is not None
    assert page.url == "https://example.com/article"
    assert page.title == "Example Article"
    assert page.text == "Useful title Readable page text."


def test_web_research_retries_rate_limit_and_respects_retry_after(monkeypatch):
    attempts = []
    delays = []

    def handle(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
                request=request,
            )
        return httpx.Response(200, json={"results": []}, request=request)

    monkeypatch.setattr(web_research.time, "sleep", delays.append)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = WebResearch(client=client, api_key="test-key").search("retry test")

    assert result.failed is False
    assert len(attempts) == 2
    assert delays == [0.0]


def test_web_research_does_not_retry_non_transient_client_error(monkeypatch):
    attempts = []
    delays = []

    def handle(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(400, request=request)

    monkeypatch.setattr(web_research.time, "sleep", delays.append)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = WebResearch(client=client, api_key="test-key").search("bad request")

    assert result.failed is True
    assert len(attempts) == 1
    assert not delays
