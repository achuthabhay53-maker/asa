import pytest
from types import SimpleNamespace

from aca.agents import analyzer as analyzer_module
from aca.agents.analyzer import (
    ClaimCheck,
    SourceAnalysis,
    SourceAnalyzer,
    SourceAnalyzerOutput,
    SourceVerificationOutput,
    TargetSiteVerificationOutput,
    ThemeCheck,
    _classify_target_fetch_failure,
    _classify_source_type,
    _claim_quote_is_valid,
    _complete_theme_checks,
    _verify_competitor,
)
from aca.orchestrator.state import (
    CitedSource,
    ContentGap,
    ModelEvidence,
    PromptEvidence,
    WorkflowState,
)
from aca.tools.source_fetcher import (
    FetchedPage,
    SiteSearchCandidates,
    SiteUrlInventory,
    TavilySourceFetcher,
    discover_site_urls,
)
from aca.tools import source_fetcher


def test_null_theme_result_becomes_inconclusive_expected_check():
    returned_check = ThemeCheck(
        source_id="S1",
        theme=None,
        verdict="supported",
        evidence_quote="The page mentions multiplex panels.",
    )
    analysis = SourceAnalysis(
        source_id="S1",
        contribution="The source discusses assay panels.",
        themes=["Multiplex assay panels"],
    )
    page = FetchedPage(
        source_id="S1",
        url="https://example.com/panels",
        content="The page mentions multiplex panels.",
        success=True,
    )

    completed = _complete_theme_checks(
        analyses=[analysis],
        fetched_pages={"S1": page},
        returned_checks=[returned_check],
    )

    assert returned_check.theme == ""
    assert len(completed) == 1
    assert completed[0].theme == "Multiplex assay panels"
    assert completed[0].verdict == "inconclusive"
    assert completed[0].evidence_quote is None


def test_discover_site_urls_follows_sitemap_index_on_same_host(monkeypatch):
    responses = {
        "https://alamar.example/robots.txt": SimpleNamespace(
            is_success=True,
            status_code=200,
            text="Sitemap: https://alamar.example/sitemap-index.xml",
        ),
        "https://alamar.example/sitemap-index.xml": SimpleNamespace(
            is_success=True,
            status_code=200,
            content=(
                b"<sitemapindex><sitemap><loc>"
                b"https://alamar.example/sitemap-pages.xml"
                b"</loc></sitemap></sitemapindex>"
            ),
        ),
        "https://alamar.example/sitemap-pages.xml": SimpleNamespace(
            is_success=True,
            status_code=200,
            content=(
                b"<urlset>"
                b"<url><loc>https://alamar.example/</loc></url>"
                b"<url><loc>https://alamar.example/products/panels</loc></url>"
                b"</urlset>"
            ),
        ),
    }
    requested_urls = []

    def fake_get(url, **kwargs):
        requested_urls.append(url)
        return responses[url]

    monkeypatch.setattr(source_fetcher.httpx, "get", fake_get)

    inventory = discover_site_urls("https://alamar.example")

    assert inventory.urls == [
        "https://alamar.example/",
        "https://alamar.example/products/panels",
    ]
    assert inventory.complete is True
    assert len(requested_urls) == 3


def test_tavily_extract_preserves_resolved_url_and_title():
    class FixedTavilyClient:
        def extract(self, **kwargs):
            return {
                "results": [
                    {
                        "url": "https://search.example/opaque-redirect",
                        "final_url": "https://publisher.example/final-page",
                        "title": "Multiplex assay panels",
                        "raw_content": "Page content.",
                    }
                ],
                "failed_results": [],
            }

    fetcher = TavilySourceFetcher.__new__(TavilySourceFetcher)
    fetcher.client = FixedTavilyClient()

    result = fetcher.fetch_pages(
        {"S1": "https://search.example/opaque-redirect"}
    )["S1"]

    assert result.success is True
    assert result.requested_url == "https://search.example/opaque-redirect"
    assert result.url == "https://publisher.example/final-page"
    assert result.title == "Multiplex assay panels"


def test_competitor_resolution_requires_commercial_source_type():
    academic_page = FetchedPage(
        source_id="S1",
        url="https://www.ncbi.nlm.nih.gov/pmc/articles/example",
        content="This article mentions Olink and multiplex assays.",
        success=True,
        title="Multiplex assay review",
    )
    olink_page = FetchedPage(
        source_id="S2",
        url="https://olink.com/products/olink-target-96",
        content="Olink Target 96 protein panel.",
        success=True,
        title="Olink Target 96",
    )

    assert _classify_source_type(
        url=academic_page.url,
        title=academic_page.title,
    ) == "academic"
    assert _verify_competitor(
        candidate="Olink",
        page=academic_page,
        company_name="Alamar Biosciences",
        relevant=True,
    ) == (None, "not_identified")
    assert _verify_competitor(
        candidate=None,
        page=olink_page,
        company_name="Alamar Biosciences",
        relevant=True,
    ) == ("Olink", "verified")


@pytest.mark.parametrize(
    ("claim", "quote", "expected_valid"),
    [
        (
            "RayBiotech offers Luminex-based multiplex assays.",
            "Multiplex assays are immunoassays that detect multiple analytes.",
            False,
        ),
        (
            "RayBiotech offers Luminex-based multiplex assays.",
            "RayBiotech Luminex panels measure multiple analytes.",
            True,
        ),
    ],
)
def test_claim_quote_must_support_named_entities(
    claim,
    quote,
    expected_valid,
):
    check = ClaimCheck(
        source_id="S1",
        claim=claim,
        verdict="supported",
        evidence_quote=quote,
    )
    page = FetchedPage(
        source_id="S1",
        url="https://raybiotech.com/panels",
        content=quote,
        success=True,
    )

    assert _claim_quote_is_valid(check, page) is expected_valid


def test_claim_quote_for_review_aim_does_not_verify_unmentioned_standard():
    check = ClaimCheck(
        source_id="S1",
        claim="Procedural standards exist for multiplex immunofluorescence.",
        verdict="supported",
        evidence_quote=(
            "The aim of this review is to discuss requirements for multiplex IF."
        ),
    )
    page = FetchedPage(
        source_id="S1",
        url="https://www.ncbi.nlm.nih.gov/pmc/articles/example",
        content=(
            "The aim of this review is to discuss requirements for multiplex IF."
        ),
        success=True,
    )

    assert _claim_quote_is_valid(check, page) is False


def test_tavily_theme_search_filters_host_deduplicates_and_caps():
    class FixedTavilyClient:
        def __init__(self):
            self.queries = []

        def search(self, **kwargs):
            self.queries.append(kwargs)
            if kwargs["query"] == "immunology panels":
                results = [
                    {"url": "https://alamar.example/immunology"},
                    {"url": "https://alamar.example/shared"},
                    {"url": "https://other.example/offsite"},
                ]
            else:
                results = [
                    {"url": "https://alamar.example/shared"},
                    {"url": "https://alamar.example/oncology"},
                ]
            return {"results": results}

    fetcher = TavilySourceFetcher.__new__(TavilySourceFetcher)
    fetcher.client = FixedTavilyClient()

    candidates = fetcher.search_site_pages(
        site_url="https://alamar.example/",
        themes=["immunology panels", "oncology panels"],
        per_theme_limit=3,
        total_limit=2,
    )

    assert candidates.urls == [
        "https://alamar.example/immunology",
        "https://alamar.example/shared",
    ]
    assert candidates.theme_urls == {
        "immunology panels": [
            "https://alamar.example/immunology",
            "https://alamar.example/shared",
        ],
        "oncology panels": ["https://alamar.example/shared"],
    }
    assert candidates.discovered_url_count == 5
    assert [item.reason_category for item in candidates.skipped_urls] == [
        "URL filtering",
        "URL filtering",
    ]
    assert all(
        query["include_domains"] == ["alamar.example"]
        and query["max_results"] == 3
        for query in fetcher.client.queries
    )


def test_blocked_robots_keeps_sitemap_inventory_incomplete(monkeypatch):
    responses = {
        "https://alamar.example/robots.txt": SimpleNamespace(
            is_success=False,
            status_code=403,
            text="",
        ),
        "https://alamar.example/sitemap.xml": SimpleNamespace(
            is_success=True,
            status_code=200,
            content=(
                b"<urlset>"
                b"<url><loc>https://alamar.example/products/panels</loc></url>"
                b"</urlset>"
            ),
        ),
    }
    monkeypatch.setattr(
        source_fetcher.httpx,
        "get",
        lambda url, **kwargs: responses[url],
    )

    inventory = discover_site_urls("https://alamar.example")

    assert inventory.urls == [
        "https://alamar.example/products/panels"
    ]
    assert inventory.complete is False


def test_discovery_rejects_off_host_urls_and_enforces_page_limit(monkeypatch):
    urls = [
        f"https://alamar.example/page-{index}"
        for index in range(4)
    ]
    locations = "".join(
        f"<url><loc>{url}</loc></url>"
        for url in urls
    )
    locations += "<url><loc>https://outside.example/page</loc></url>"
    responses = {
        "https://alamar.example/robots.txt": SimpleNamespace(
            is_success=False,
            status_code=404,
            text="",
        ),
        "https://alamar.example/sitemap.xml": SimpleNamespace(
            is_success=True,
            status_code=200,
            content=f"<urlset>{locations}</urlset>".encode(),
        ),
    }
    monkeypatch.setattr(
        source_fetcher.httpx,
        "get",
        lambda url, **kwargs: responses[url],
    )

    inventory = discover_site_urls(
        "https://alamar.example",
        max_urls=2,
    )

    assert inventory.urls == urls[:2]
    assert all("outside.example" not in url for url in inventory.urls)
    assert inventory.complete is False


@pytest.mark.parametrize(
    ("returned_checks", "expected_verdict"),
    [
        (
            [
                ThemeCheck(
                    source_id="TARGET-1",
                    theme="Multiplex assay panels",
                    verdict="supported",
                    evidence_quote="This page covers multiplex assay panels.",
                )
            ],
            "supported",
        ),
        (
            [
                ThemeCheck(
                    source_id="TARGET-1",
                    theme="Multiplex assay panels",
                    verdict="supported",
                    evidence_quote="A fabricated quotation.",
                )
            ],
            "inconclusive",
        ),
        ([], "inconclusive"),
    ],
)
def test_target_theme_verifier_requires_matching_quote(
    returned_checks,
    expected_verdict,
):
    class FixedModel:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, _messages):
            return TargetSiteVerificationOutput(
                theme_checks=returned_checks
            )

    analyzer = SourceAnalyzer.__new__(SourceAnalyzer)
    analyzer.model = FixedModel()
    page = FetchedPage(
        source_id="TARGET-1",
        url="https://alamar.example/assays",
        content="This page covers multiplex assay panels.",
        success=True,
    )

    checks = analyzer._verify_target_site_themes(
        themes=["Multiplex assay panels"],
        pages={"TARGET-1": page},
    )

    assert len(checks) == 1
    assert checks[0].verdict == expected_verdict


@pytest.mark.parametrize(
    ("url", "error", "failure_stage", "expected_category"),
    [
        (
            "https://site.example/private",
            "HTTP 403 Forbidden",
            "tavily_failed_result",
            "robots/access restriction",
        ),
        (
            "https://site.example/redirect",
            "Redirect failed with status code 302",
            "tavily_failed_result",
            "HTTP/redirect failure",
        ),
        (
            "https://site.example/catalog.pdf",
            "Unsupported document format",
            "tavily_failed_result",
            "non-HTML content",
        ),
        (
            "https://site.example/page",
            "batch request timed out",
            "tavily_batch",
            "HTTP/redirect failure",
        ),
        (
            "https://site.example/page",
            "Service returned an error",
            "tavily_batch",
            "Tavily batch/extraction behavior",
        ),
        (
            "https://site.example/page",
            "Page extraction failed",
            "tavily_failed_result",
            "Tavily extraction failure",
        ),
        (
            "https://site.example/page",
            "unclassified problem",
            None,
            "other error",
        ),
    ],
)
def test_target_fetch_failure_categories(
    url,
    error,
    failure_stage,
    expected_category,
):
    page = FetchedPage(
        source_id="TARGET-1",
        url=url,
        content="",
        success=False,
        error=error,
        failure_stage=failure_stage,
    )

    assert _classify_target_fetch_failure(page) == expected_category


@pytest.mark.asyncio
async def test_workflow_passes_manual_company_url_to_analyzer(monkeypatch):
    from aca.tools import aimpact

    state = WorkflowState(
        job_id="manual-test",
        tenant_id=1,
        campaign_id=81,
        company_name="Alamar Biosciences",
        company_url="https://entered.example",
        prompts=[
            PromptEvidence(
                prompt_id="prompt-1",
                prompt_text="Which panels are available?",
            )
        ],
    )
    captured_project = {}

    class FakeAnalyzer:
        async def analyze(self, *, project, prompt_evidence):
            captured_project.update(project)
            return ContentGap(
                prompt_id=prompt_evidence.prompt_id,
                prompt_text=prompt_evidence.prompt_text,
                target_angle="",
                hypothesis="",
            )

    monkeypatch.setattr(
        aimpact,
        "get_campaign",
        lambda _campaign_id: {"project_id": 29},
    )
    monkeypatch.setattr(
        aimpact,
        "get_project",
        lambda _project_id: {
            "company_name": "Alamar Biosciences",
            "company_url": "https://database.example",
        },
    )
    monkeypatch.setattr(
        analyzer_module,
        "SourceAnalyzer",
        FakeAnalyzer,
    )

    result = await analyzer_module.run(state)

    assert captured_project["company_url"] == "https://entered.example"
    assert result.gap is not None


class _FixedFetcher:
    def __init__(
        self,
        page,
        target_page_content=None,
        target_page_success=True,
    ):
        self.page = page
        self.target_page_content = target_page_content
        self.target_page_success = target_page_success

    def fetch_pages(self, sources):
        return {
            source_id: FetchedPage(
                source_id=source_id,
                url=url,
                content=(
                    self.target_page_content
                    if source_id.startswith("TARGET-")
                    and self.target_page_content is not None
                    else self.page.content
                ),
                success=(
                    self.target_page_success
                    if source_id.startswith("TARGET-")
                    else True
                ),
            )
            for source_id, url in sources.items()
        }


class _FixedTargetFetcher:
    def __init__(self, page_content, success=True):
        self.page_content = page_content
        self.success = success

    async def fetch_pages(self, sources):
        return {
            source_id: FetchedPage(
                source_id=source_id,
                url=url,
                content=self.page_content,
                success=self.success,
            )
            for source_id, url in sources.items()
        }


@pytest.mark.parametrize(
    (
        "target_page_success",
        "target_verdict",
        "expected_site_coverage",
        "expected_coverage",
    ),
    [
        (True, "supported", "partial", "covered"),
        (True, "not_found", "partial", "unknown"),
        (False, "not_found", "unknown", "unknown"),
    ],
)
@pytest.mark.asyncio
async def test_target_site_coverage_requires_complete_evidence(
    monkeypatch,
    target_page_success,
    target_verdict,
    expected_site_coverage,
    expected_coverage,
):
    citation_quote = "Olink offers multiplex protein assay panels."
    target_quote = "The site describes multiplex protein panels."
    analysis = SourceAnalyzerOutput(
        source_analyses=[
            SourceAnalysis(
                source_id="S1",
                competitor="Olink",
                contribution="The source describes multiplex panels.",
                themes=["Multiplex protein assay panels"],
            )
        ],
        themes=["Multiplex protein assay panels"],
        visibility_gap="Citation observation.",
        editorial_angle="Review the panel coverage.",
        hypothesis="Candidate hypothesis.",
    )
    verification = SourceVerificationOutput(
        theme_checks=[
            ThemeCheck(
                source_id="S1",
                theme="Multiplex protein assay panels",
                verdict="supported",
                evidence_quote=citation_quote,
            )
        ]
    )
    analyzer = SourceAnalyzer.__new__(SourceAnalyzer)
    analyzer.fetcher = _FixedFetcher(
        FetchedPage(
            source_id="S1",
            url="https://source.example/panels",
            content=citation_quote,
            success=True,
        ),
        target_page_content=(
            target_quote
            if target_verdict == "supported"
            else "This target page covers different material."
        ),
        target_page_success=target_page_success,
    )
    analyzer.target_fetcher = _FixedTargetFetcher(
        page_content=(
            target_quote
            if target_verdict == "supported"
            else "This target page covers different material."
        ),
        success=target_page_success,
    )
    analyzer._analyze_candidates = lambda **kwargs: analysis
    analyzer._verify_sources = lambda **kwargs: verification
    analyzer.fetcher.search_site_pages = lambda *args, **kwargs: (
        SiteSearchCandidates(
            urls=["https://alamar.example/assays"],
            theme_urls={
                "Multiplex protein assay panels": [
                    "https://alamar.example/assays"
                ]
            },
            discovered_url_count=1,
            skipped_urls=[],
        )
    )
    analyzer._verify_target_site_themes = lambda themes, pages: [
        ThemeCheck(
            source_id="TARGET-1",
            theme=themes[0],
            verdict=target_verdict,
            evidence_quote=(
                target_quote
                if target_verdict == "supported"
                else None
            ),
        )
    ]
    prompt = PromptEvidence(
        prompt_id="prompt-target-site",
        prompt_text="Which multiplex protein panels are available?",
        per_model={
            "ChatGPT": ModelEvidence(
                sources=[
                    CitedSource(
                        model="ChatGPT",
                        url="https://source.example/panels",
                        excerpt="Multiplex panel source.",
                    )
                ]
            )
        },
    )

    gap = await analyzer.analyze(
        project={
            "company_name": "Alamar Biosciences",
            "company_url": "https://alamar.example",
        },
        prompt_evidence=prompt,
    )

    assert gap.target_site_coverage == expected_site_coverage
    assert gap.target_site_inventory_complete is False
    assert gap.content_gap_status == "not_established"
    assert gap.theme_assessments[0].target_site_coverage == expected_coverage
    assert gap.theme_assessments[0].content_gap_status == "not_established"
    if target_verdict == "supported":
        assert gap.theme_assessments[0].target_site_evidence[0].evidence_quote == target_quote
    else:
        assert gap.theme_assessments[0].target_site_evidence == []


@pytest.mark.asyncio
async def test_analyzer_shortlists_themes_and_separates_verified_output():
    quote = "Olink offers multiplex protein assay panels."
    themes = [
        "Multiplex protein assay panels",
        "Neuro biomarker panels",
        "Immunology protein panels",
        "Oncology protein panels",
        "High-plex assays",
        "Cytokine measurements",
        "Protein profiling",
        "Biomarker validation",
        "Inflammatory markers",
    ]
    first_pass = SourceAnalyzerOutput(
        source_analyses=[
            SourceAnalysis(
                source_id="S1",
                competitor="Olink",
                contribution="The cited page describes multiplex panels.",
                themes=themes[:5],
            ),
            SourceAnalysis(
                source_id="S2",
                contribution="Another cited page covers related assays.",
                themes=themes[5:],
            )
        ],
        themes=themes,
        visibility_gap="Candidate visibility note.",
        editorial_angle=(
            "Alamar's technology offers distinct advantages over competitors."
        ),
        hypothesis="The company is absent, so it may lack content.",
    )
    verification = SourceVerificationOutput(
        claim_checks=[
            ClaimCheck(
                source_id="S1",
                claim="Bio-Techne offers multiplex assay solutions.",
                verdict="supported",
                evidence_quote="This quote is not present in the fetched page.",
            )
        ],
        theme_checks=[
            ThemeCheck(
                source_id="S1",
                theme="Multiplex protein assay panels",
                verdict="supported",
                evidence_quote=quote,
            )
        ]
    )

    analyzer = SourceAnalyzer.__new__(SourceAnalyzer)
    analyzer.fetcher = _FixedFetcher(
        FetchedPage(
            source_id="S1",
            url="https://olink.com/products/panels",
            content=quote,
            success=True,
        )
    )
    analyzer._analyze_candidates = lambda **kwargs: first_pass

    def verify_sources(**kwargs):
        assert sum(
            len(analysis.themes)
            for analysis in kwargs["analyses"]
        ) <= 8
        return verification

    analyzer._verify_sources = verify_sources

    prompt = PromptEvidence(
        prompt_id="prompt-1",
        prompt_text="Which multiplex protein assay panels are available?",
        per_model={
            "ChatGPT": ModelEvidence(
                sources=[
                    CitedSource(
                        model="ChatGPT",
                        url="https://olink.com/products/panels",
                        excerpt="Olink panel citation.",
                    ),
                    CitedSource(
                        model="Claude",
                        url="https://research.example/assays",
                        excerpt="Assay research citation.",
                    ),
                ]
            )
        },
    )

    gap = await analyzer.analyze(
        project={"company_name": "Alamar Biosciences"},
        prompt_evidence=prompt,
    )

    assert len(gap.themes) == 1
    assert gap.themes == ["Multiplex protein assay panels"]
    assert gap.themes[0] not in gap.candidate_themes
    assert len(gap.candidate_themes) <= 7
    assert gap.theme_evidence[0].evidence_quote == quote
    assert gap.theme_assessments[0].competitors == ["Olink"]
    analyzed_source = gap.cited_sources[0]
    assert analyzed_source.verification_status == "unsupported"
    assert analyzed_source.verified_claims[0].status == "model_supported"
    assert analyzed_source.verified_claims[0].evidence_quote is None
    assert "Verified claims:" not in analyzed_source.contribution
    assert gap.target_site_coverage == "unknown"
    assert gap.content_gap_status == "not_established"
    assert "1 of 8 candidate themes" in gap.verification_summary
    assert "not evidence of missing website content" in gap.hypothesis
    assert "offers distinct advantages" not in gap.candidate_target_angle
    assert "No site-wide gap is established" in gap.candidate_target_angle
    assert "absence of the company from the citations alone is not evidence" in (
        gap.hypothesis
    )
    assert "suggests the company" not in gap.hypothesis
    assert _verify_competitor(
        candidate="Olink",
        page=analyzer.fetcher.page,
        company_name="Alamar Biosciences",
        relevant=False,
    ) == (None, "unverified")
