from aca.agents.analyzer import (
    ClaimCheck,
    SourceAnalysis,
    SourceAnalyzer,
    SourceAnalyzerOutput,
    SourceVerificationOutput,
    ThemeCheck,
    _complete_theme_checks,
    _verify_competitor,
)
from aca.orchestrator.state import (
    CitedSource,
    ModelEvidence,
    PromptEvidence,
)
from aca.tools.source_fetcher import FetchedPage


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


class _FixedFetcher:
    def __init__(self, page):
        self.page = page

    def fetch_pages(self, sources):
        return {
            source_id: FetchedPage(
                source_id=source_id,
                url=url,
                content=self.page.content,
                success=True,
            )
            for source_id, url in sources.items()
        }


def test_analyzer_shortlists_themes_and_separates_verified_output():
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
        editorial_angle="Compare assay panels.",
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
            url="https://olink.example/panels",
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
                        url="https://olink.example/panels",
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

    gap = analyzer.analyze(
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
    assert "absence of the company from these citations is not evidence" in (
        gap.hypothesis
    )
    assert "suggests the company" not in gap.hypothesis
    assert _verify_competitor(
        candidate="Olink",
        page=analyzer.fetcher.page,
        company_name="Alamar Biosciences",
        relevant=False,
    ) == (None, "unverified")
