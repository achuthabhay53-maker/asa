"""Small synthetic outputs shared by workflow contract and quality tests."""
from aca.models.content_brief import BriefSection, ContentBrief
from aca.models.ground_truth import (
    EvidenceItem,
    GroundTruthGuide,
    PromptVerification,
    VerifiedFact,
)
from aca.models.prompt_intelligence import (
    AnswerUnderstanding,
    CustomerVisibility,
    PromptAnalysis,
    PromptUnderstanding,
)
from aca.models.source_intelligence import (
    AnalyzedSource,
    CitationSummary,
    ContentAnalysis,
    ContentStructure,
    RankedSource,
    SourceIntelligence,
)
from aca.models.writer import (
    WriterClaim,
    WriterOutput,
    WrittenSection,
)


SOURCE_URL = "https://example.com/atlas-panel"
EVIDENCE_QUOTE = (
    "Example Bio's Atlas panel quantifies 120 immune-related proteins "
    "in serum samples."
)


def agent1_output() -> PromptAnalysis:
    return PromptAnalysis(
        prompt_id="prompt-1",
        prompt="How can researchers profile immune markers in serum?",
        customer_name="Example Bio",
        prompt_understanding=PromptUnderstanding(
            intent="Learn how researchers can profile immune markers in serum",
            topic="serum immune-marker profiling",
            audience="biomedical researchers",
            user_goal="Compare approaches to profiling immune markers",
            key_questions=["Which assay supports multiplex profiling?"],
        ),
        answer_understanding=AnswerUnderstanding(
            summary="The answers describe multiplex assays for immune markers.",
            key_topics=["multiplex assays", "immune markers"],
            key_concepts=["serum profiling"],
        ),
        customer_visibility=CustomerVisibility(status="mentioned"),
    )


def agent2_output() -> SourceIntelligence:
    return SourceIntelligence(
        campaign_id=1,
        prompt_id="prompt-1",
        citation_summary=CitationSummary(
            total_citation_occurrences=3,
            unique_sources=1,
        ),
        ranked_sources=[
            RankedSource(
                source_id="src-1",
                rank=1,
                url=SOURCE_URL,
                title="Atlas Panel Overview",
                citation_count=3,
                cited_by_models=["model-a", "model-b"],
            )
        ],
        content_analyses=[
            AnalyzedSource(
                source_id="src-1",
                url=SOURCE_URL,
                title="Atlas Panel Overview",
                content_analysis=ContentAnalysis(
                    summary="Describes the panel and its sample compatibility.",
                    main_topics=["immune proteins", "serum"],
                    content_structure=ContentStructure(
                        headings=["Panel overview", "Sample compatibility"]
                    ),
                ),
            )
        ],
    )


def agent3_output(*, verified: bool = True) -> GroundTruthGuide:
    return GroundTruthGuide(
        topic="Serum immune-marker profiling",
        main_topic="Serum immune-marker profiling",
        search_intent="Evaluate multiplex profiling approaches",
        target_audience="Biomedical researchers",
        verification=PromptVerification(
            is_verified=verified,
            intent_valid=True,
            customer_relevant=verified,
            reason=None if verified else "The requested capability was not verified.",
        ),
        evidence=[
            EvidenceItem(
                evidence_id="ev-001",
                url=SOURCE_URL,
                quote=EVIDENCE_QUOTE,
                domain="example.com",
            )
        ],
        verified_facts=[
            VerifiedFact(
                fact_id="fact-001",
                statement=EVIDENCE_QUOTE,
                evidence_ids=["ev-001"],
                wording_boundary="Do not infer clinical outcomes from this assay description.",
            )
        ],
    )


def agent4_output(*, include_sections: bool = True) -> ContentBrief:
    sections = [
        BriefSection(
            heading="What multiplex immune profiling measures",
            section_objective="Explain the approach to the reader.",
            topics_and_questions=["What does multiplex profiling measure?"],
            evidence_ids=["ev-001"],
        ),
        BriefSection(
            heading="Why serum compatibility matters",
            section_objective="Explain sample choice.",
            evidence_ids=["ev-001"],
        ),
        BriefSection(
            heading="How to evaluate an assay",
            section_objective="Offer a practical evaluation framework.",
            topics_and_questions=["Which factors should researchers compare?"],
        ),
        BriefSection(
            heading="Frequently Asked Questions",
            section_objective="Answer common reader questions.",
        ),
    ] if include_sections else []
    return ContentBrief(
        article_objective="Help researchers assess serum immune-profiling approaches.",
        target_audience="Biomedical researchers",
        search_intent="Evaluate multiplex profiling approaches",
        primary_keyword="serum immune-marker profiling",
        secondary_keywords=["multiplex immune assay"],
        related_topics=["sample compatibility"],
        question_based_queries=["What does multiplex profiling measure?"],
        faq_questions=[
            "What is multiplex immune profiling?",
            "Can serum be used for immune profiling?",
            "What should researchers compare?",
        ],
        recommended_title="A Guide to Serum Immune-Marker Profiling",
        sections=sections,
    )


def agent5_output() -> WriterOutput:
    article = f"""# A Guide to Serum Immune-Marker Profiling

Researchers comparing immune-profiling approaches need to understand what
each assay measures and which sample types it supports.

## What multiplex immune profiling measures

Multiplex assays measure multiple targets in one analysis. {EVIDENCE_QUOTE} [Source]({SOURCE_URL}).

## Why serum compatibility matters

Sample compatibility is one factor to check when planning a study.

## How to evaluate an assay

Compare the panel scope, sample requirements, and evidence available for
the intended study.

| Decision factor | Evidence-based consideration |
| --- | --- |
| Panel scope | Check whether the measured targets fit the study question. |
| Sample compatibility | Confirm the sample type in the supplied assay evidence. |

## Conclusion

Match the assay and sample type to the study question, then verify important
claims against the supplied evidence.

## Frequently Asked Questions

### What is multiplex immune profiling?

It is an approach for measuring multiple immune targets in one analysis.

### Can serum be used for immune profiling?

The cited Example Bio panel is described as quantifying immune-related
proteins in serum samples.

### What should researchers compare?

Consider panel scope, sample compatibility, and the evidence supporting
claims about the assay.
"""
    return WriterOutput(
        title="A Guide to Serum Immune-Marker Profiling",
        seo_title="Serum Immune-Marker Profiling Guide",
        meta_description="A practical guide to evaluating serum immune-marker profiling approaches.",
        article_markdown=article,
        word_count=len(article.split()),
        sections=[
            WrittenSection(
                heading="What multiplex immune profiling measures",
                content_markdown=EVIDENCE_QUOTE,
                evidence_ids_used=["ev-001"],
            ),
            WrittenSection(
                heading="Frequently Asked Questions",
                content_markdown="Three concise answers.",
                evidence_ids_used=["ev-001"],
            ),
        ],
        claims_used=[
            WriterClaim(
                claim=EVIDENCE_QUOTE,
                evidence_ids=["ev-001"],
                source_urls=[SOURCE_URL],
            )
        ],
        brief_coverage=[
            "Explains multiplex immune profiling.",
            "Covers sample compatibility and practical evaluation factors.",
        ],
    )
