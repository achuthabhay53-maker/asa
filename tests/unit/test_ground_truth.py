"""Unit tests for Agent 3: Ground Truth & Research Discovery."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from aca.agents.ground_truth import (
    GroundTruthAgent,
    CheckedClaim,
    _EvidenceRegistry,
    _apply,
    _assemble,
    _is_superlative,
    _handoff_boundaries,
    _merge_pairs,
    _mismatch_reason,
    _scope_note,
    build_prompt_context,
)
from aca.models.ground_truth import (
    CandidateSubtopic,
    ClientInput,
    Evidence,
    EvidenceItem,
    Fact,
    GroundTruthGuide,
    PromptVerification,
    ResearchGap,
    TopicVerdict,
    VerifiedFact,
)
from aca.models.prompt_intelligence import PromptAnalysis
from aca.orchestrator.pipeline import PipelineState, agent3_node
from aca.tools.web_research import ExtractedPage, SearchResponse, SearchResult


def _mock_agent1_data():
    return {
        "prompt_id": "1409",
        "prompt": "Immunology biomarker assay provider",
        "customer_name": "Alamar Biosciences",
        "prompt_understanding": {
            "intent": "Identify companies that provide immunology biomarker assay services",
            "topic": "Immunology biomarker assay providers in North America",
            "audience": "Biopharma companies and CROs",
            "user_goal": "Find assay providers",
            "key_questions": ["Which companies offer immunology biomarker assay services?"],
        },
        "answer_understanding": {
            "summary": "AI models list immunology biomarker assay providers",
            "key_topics": ["Assay development", "Multiplex biomarker profiling"],
            "key_concepts": ["ELISA", "NULISA", "Simoa"],
            "comparison_criteria": ["Sensitivity", "Multiplexing"],
        },
        "companies_mentioned": [],
        "customer_visibility": {"status": "not_mentioned", "models_mentioning_customer": []},
        "cross_model_patterns": {"common_topics": [], "differences": []},
        "content_opportunities": [
            "Evaluate Alamar's biomarker assay technology on sensitivity and multiplexing capacity"
        ],
    }


def test_ground_truth_returns_typed_negative_verification_when_irrelevant():
    agent = GroundTruthAgent(api_key="mock-key", model="mock-model")

    mock_understanding = {
        "main_topic": "Footwear fashion",
        "search_intent": "Find footwear fashion brands",
        "target_audience": "Consumers",
        "prompt_scope": "company_specific",
        "is_verified": False,
        "intent_valid": False,
        "customer_relevant": False,
        "reason": "Customer is an apparel footwear brand with no relation to immunology biomarker assays.",
        "quote": "",
        "source_url": "",
        "retrieved_pages": [],
    }

    with patch.object(agent, "_understand_prompt_and_verify", return_value=mock_understanding):
        agent1 = _mock_agent1_data()
        client = ClientInput(company_name="Acme Footwear", website="https://acmefootwear.example.com")
        guide = agent.analyze(agent1, client)
        assert guide.verification is not None
        assert guide.verification.is_verified is False
        assert guide.verification.intent_valid is False


def test_verification_keeps_valid_intent_separate_from_customer_relevance():
    class EmptyResearch:
        @staticmethod
        def extract(_url):
            return None

    agent = GroundTruthAgent(
        api_key="mock-key",
        model="mock-model",
        research=EmptyResearch(),
    )
    context = build_prompt_context(
        {
            **_mock_agent1_data(),
            "prompt": "Who provides VR training for medical device companies?",
            "prompt_understanding": {
                "intent": "Find providers of virtual reality training for medical devices",
                "topic": "Medical device training",
                "audience": "Medical device companies",
                "user_goal": "Find training providers",
                "key_questions": [],
            },
        },
        "https://ethosh.com",
        "Ethosh",
    )
    captured_prompt = ""

    def complete(system: str, payload: dict, max_tokens: int) -> dict:
        nonlocal captured_prompt
        captured_prompt = system
        return {
            "main_topic": "VR training for medical device companies",
            "search_intent": "Find relevant training providers",
            "target_audience": "Medical device companies",
            "prompt_scope": "hybrid",
            "is_verified": False,
            "intent_valid": True,
            "customer_relevant": False,
            "reason": "The retrieved pages did not establish this customer's VR training capability.",
        }

    with (
        patch.object(agent, "_collect_pages", return_value=([], False)),
        patch.object(agent, "_complete", side_effect=complete),
    ):
        result = agent._understand_prompt_and_verify(
            context,
            "Ethosh",
            "ethosh.com",
        )

    assert result["intent_valid"] is True
    assert result["customer_relevant"] is False
    assert result["is_verified"] is False
    assert "intent_valid: true and customer_relevant: false" in captured_prompt
    assert "virtual reality / VR, immersive training, XR, and simulation" in captured_prompt


def test_verification_rejects_missing_or_non_boolean_flags():
    invalid = {
        "main_topic": "VR training",
        "search_intent": "Find providers",
        "target_audience": "Medical device teams",
        "prompt_scope": "hybrid",
        "reason": "Evidence was found.",
        "is_verified": "false",
        "intent_valid": True,
        "customer_relevant": True,
        "quote": "A sufficiently long exact quote from a retrieved page.",
        "source_url": "https://example.com/page",
        "retrieved_pages": [],
    }

    try:
        GroundTruthAgent._validate_understanding(invalid)
    except ValueError as exc:
        assert "boolean field 'is_verified'" in str(exc)
    else:
        raise AssertionError("Non-boolean verification flags must be rejected.")


def test_evidence_registry_deduplicates_only_same_quote_and_canonical_source():
    quote = "The platform supports a measured multiplex protein analysis workflow."
    pages = [
        {
            "url": "HTTPS://WWW.Example.com:443/product/#details",
            "text": quote,
            "source_type": "company_website",
        },
        {
            "url": "https://example.org/review",
            "text": quote,
            "source_type": "reputable_external",
        },
    ]
    registry = _EvidenceRegistry("example.com")

    first = registry.register_quote(quote, pages)
    duplicate_same_source = registry.register_quote(
        quote,
        [{"url": "https://www.example.com/product", "text": quote}],
    )
    same_quote_other_source = registry.register_quote(quote, pages[1:])
    fabricated = registry.register_quote(
        "A paraphrased sentence that was never on either page.",
        pages,
    )

    assert first == duplicate_same_source == "ev_001"
    assert same_quote_other_source == "ev_002"
    assert fabricated is None
    assert [item.url for item in registry.items] == [
        "HTTPS://WWW.Example.com:443/product/#details",
        "https://example.org/review",
    ]


def test_evidence_registry_keeps_validated_checked_quote_and_original_source_url():
    from aca.models.ground_truth import Evidence as GroundTruthEvidence

    quote = "The product supports a validated immune assay workflow."
    source_url = "https://example.com/product-page"
    registry = _EvidenceRegistry("example.com")

    evidence_id = registry.register_checked_evidence(
        GroundTruthEvidence(url=source_url, quote=quote, source_tier="first_party")
    )

    assert evidence_id == "ev_001"
    assert registry.items[0].url == source_url
    assert registry.items[0].quote == quote
    assert registry.items[0].source_type == "company_website"


def test_superlative_and_certification_matching_use_word_boundaries():
    assert not _is_superlative("The team can bestow several technical capabilities.")
    assert _is_superlative("The product is the best in its category.")
    assert (
        _mismatch_reason("The product has this capability.", "The platform capabilities include analytics.")
        is None
    )
    assert (
        _mismatch_reason("The laboratory is CAP certified.", "The laboratory is ISO certified.") is not None
    )
    assert _mismatch_reason("The system isolates biological samples.", "The system isolates samples.") is None


def test_positive_and_contradictory_pages_are_preserved_as_a_conflict():
    question = {
        "topic": "Product service",
        "claim": "Example Bio provides the service",
        "claim_type": "service",
        "relevance": "high",
        "relevance_to_topic": "high",
        "checked_from": "prompt",
    }
    supported = _apply(
        "fact_001",
        question["topic"],
        question["claim"],
        [
            {
                "support": "full",
                "quote": "Example Bio provides the service to customers.",
                "url": "https://example.com/services",
                "published": None,
                "client_host": "example.com",
                "narrow_statement": question["claim"],
                "wording_limit": "",
            }
        ],
        False,
        "service",
        "high",
        "high",
        "prompt",
    )
    contradicted = _apply(
        "fact_002",
        question["topic"],
        question["claim"],
        [
            {
                "support": "contradicts",
                "quote": "Example Bio does not provide the service to customers.",
                "url": "https://example.org/review",
                "published": None,
                "client_host": "example.com",
                "narrow_statement": "",
                "wording_limit": "",
            }
        ],
        False,
        "service",
        "high",
        "high",
        "prompt",
    )

    merged, _ = _merge_pairs([(question, supported), (question, contradicted)])[0]

    assert merged.fact.status == "inconclusive"
    assert merged.verdict.decision == "research_more"
    assert len(merged.fact.evidence) == 2
    assert merged.gap is not None


def test_checked_research_gaps_survive_assembly_without_reason_word_matching():
    context = build_prompt_context(
        _mock_agent1_data(),
        "https://alamarbio.com",
        "Alamar Biosciences",
    )
    checked = _apply(
        "fact_001",
        "Capability",
        "Example Bio supports a particular platform feature",
        [],
        False,
        "platform_capability",
        "high",
        "high",
        "prompt",
    )

    guide = _assemble(
        context,
        [({"checked_from": "prompt"}, checked)],
        [],
    )

    assert guide.information_still_needed
    assert guide.information_still_needed[0].missing_claim == checked.asked_claim


def test_checked_and_limited_facts_keep_unique_ids_across_groups():
    context = build_prompt_context(
        _mock_agent1_data(),
        "https://alamarbio.com",
        "Alamar Biosciences",
    )
    first_fact = Fact(
        fact_id="input_a",
        claim="Example Bio offers assay A",
        status="verified",
        evidence=[Evidence(url="https://example.com/a", quote="Example Bio offers assay A.")],
    )
    limited_fact = Fact(
        fact_id="input_b",
        claim="Example Bio supports assay B",
        status="verified",
        evidence=[Evidence(url="https://example.com/b", quote="Example Bio supports assay B.")],
    )
    first = CheckedClaim(
        first_fact,
        TopicVerdict(topic="A", decision="write", fact_ids=["input_a"]),
        None,
        [],
        None,
        None,
        None,
        first_fact.claim,
    )
    limited = CheckedClaim(
        limited_fact,
        TopicVerdict(topic="B", decision="write_limited", fact_ids=["input_b"]),
        None,
        [],
        None,
        limited_fact.claim,
        "State only the named assay.",
        limited_fact.claim,
    )

    guide = _assemble(
        context,
        [
            ({"checked_from": "prompt"}, first),
            ({"checked_from": "prompt"}, limited),
        ],
        [],
    )
    fact_ids = [fact.fact_id for fact in guide.can_write]
    fact_ids.extend(item.fact_id for item in guide.can_write_with_limits)

    assert len(fact_ids) == len(set(fact_ids)) == 2
    assert all(fact_id.startswith("fact_") for fact_id in fact_ids)


def test_scope_note_describes_extra_topic_words_generically():
    context = build_prompt_context(
        _mock_agent1_data(),
        "https://example.com",
        "Example Bio",
    )

    note = _scope_note(context)

    assert note.scope_note
    assert "North America" not in note.scope_note


def test_agent3_deduplicates_repeated_warnings():
    agent = GroundTruthAgent(api_key="mock-key", model="mock-model")

    agent._add_warning("The same retrieval warning.")
    agent._add_warning("The same retrieval warning.")

    assert agent._warnings == ["The same retrieval warning."]


def test_agent3_handoff_includes_prohibited_claims_and_deduplicated_warnings():
    boundaries = _handoff_boundaries(
        "Example Bio",
        ["Claim contradicted by evidence", "best"],
        ["Research retrieval failed", "Research retrieval failed"],
    )

    assert any("Claim contradicted by evidence" in item for item in boundaries)
    assert any("best" in item for item in boundaries)
    assert boundaries.count("Research retrieval failed") == 1


def test_step1_search_failure_is_reported_as_unassessed_not_irrelevant():
    agent = GroundTruthAgent(
        api_key="mock-key",
        model="mock-model",
        research=SimpleNamespace(extract=lambda _url: None),
    )
    context = build_prompt_context(
        _mock_agent1_data(),
        "https://example.com",
        "Example Bio",
    )

    with patch.object(agent, "_collect_optional_pages", return_value=([], True)):
        result = agent._understand_prompt_and_verify(
            context,
            "Example Bio",
            "example.com",
        )

    assert result["is_verified"] is False
    assert result["intent_valid"] is True
    assert result["customer_relevant"] is False
    assert result["human_review_required"] is True
    assert "not evidence that the customer is irrelevant" in result["reason"]
    assert result["retrieved_pages"] == []


def test_optional_research_failure_is_logged_as_warning_and_returned():
    class BrokenResearch:
        @staticmethod
        def search(*_args, **_kwargs):
            raise TimeoutError("temporary research timeout")

    agent = GroundTruthAgent(
        api_key="mock-key",
        model="mock-model",
        research=BrokenResearch(),
    )

    pages, failed = agent._collect_optional_pages(
        "assay capabilities",
        "example.com",
        "Example",
        first_party=True,
    )

    assert pages == []
    assert failed is True
    assert any(
        "optional first-party research retrieval failed" in warning.casefold() for warning in agent._warnings
    )


def test_verified_relevance_without_exact_quote_requires_human_review():
    uncertain = {
        "main_topic": "VR training",
        "search_intent": "Find providers",
        "target_audience": "Medical device teams",
        "prompt_scope": "hybrid",
        "reason": "The model says this is relevant.",
        "is_verified": True,
        "intent_valid": True,
        "customer_relevant": True,
        "quote": "A model-invented quote that is not on the page.",
        "source_url": "https://example.com/page",
        "retrieved_pages": [
            {
                "url": "https://example.com/page",
                "text": "The retrieved page does not contain that claim.",
            }
        ],
    }

    validated = GroundTruthAgent._validate_understanding(uncertain)

    assert validated["intent_valid"] is True
    assert validated["customer_relevant"] is True
    assert validated["is_verified"] is True
    assert validated["human_review_required"] is True
    assert "Human review is required" in validated["reason"]


def test_collect_pages_resolves_google_grounding_redirect_before_domain_filter():
    redirect_url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/opaque"
    other_redirect_url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/other-opaque"

    class Research:
        def search(self, query: str, **kwargs) -> SearchResponse:
            return SearchResponse(
                results=[
                    SearchResult(url=redirect_url, title="Ethosh capability page"),
                    SearchResult(url=other_redirect_url, title="Other page"),
                ]
            )

        def extract(self, url: str, *, max_chars: int) -> ExtractedPage:
            if url == redirect_url:
                return ExtractedPage(
                    url="https://www.ethosh.com/virtual-reality-training",
                    text="Ethosh provides immersive virtual reality training for medical device teams.",
                )
            assert url == other_redirect_url
            return ExtractedPage(
                url="https://another-company.example/training",
                text="This is not an Ethosh page.",
            )

    agent = GroundTruthAgent(
        api_key="mock-key",
        model="mock-model",
        research=Research(),
    )

    pages, failed = agent._collect_pages(
        "VR training for medical device teams",
        "ethosh.com",
        "Ethosh",
        first_party=True,
    )

    assert failed is False
    assert len(pages) == 1
    assert pages[0]["url"] == "https://www.ethosh.com/virtual-reality-training"


def test_ground_truth_returns_structured_agent4_package():
    """If verified, Agent 3 returns GroundTruthGuide with candidate subtopics, evidence types, gaps, and limitations."""
    agent = GroundTruthAgent(api_key="mock-key", model="mock-model")

    mock_understanding = {
        "main_topic": "Immunology biomarker assay providers and platform capabilities",
        "search_intent": "Understand vendor capabilities, assay sensitivity, and multiplex options",
        "target_audience": "Biopharma translational researchers and CRO directors",
        "prompt_scope": "hybrid",
        "is_verified": True,
        "intent_valid": True,
        "customer_relevant": True,
        "reason": "Alamar Biosciences develops NULISA immunology assay panels and ARGO instruments.",
        "quote": "NULISAseq Immune 340 Panel provides deep profiling of immunology biomarkers.",
        "source_url": "https://alamarbio.com/immune-340",
        "retrieved_pages": [
            {
                "url": "https://alamarbio.com/immune-340",
                "text": "NULISAseq Immune 340 Panel provides deep profiling of immunology biomarkers.",
                "published": None,
            }
        ],
    }

    mock_guide = GroundTruthGuide(
        main_topic="Immunology biomarker assay providers and platform capabilities",
        search_intent="Understand vendor capabilities, assay sensitivity, and multiplex options",
        target_audience="Biopharma translational researchers and CRO directors",
        prompt_scope="hybrid",
        candidate_subtopics=[
            CandidateSubtopic(
                subtopic_id="subtopic_001",
                title="Ultra-Sensitive Multiplex Cytokine Profiling",
                description="How NULISA achieves attomolar sensitivity for low-abundance immune markers.",
                topic_angle="company_capability",
                relevance_to_main_topic="high",
                evidence_status="supported",
                evidence_ids=["ev_001"],
                suggested_questions=["What is the sensitivity limit of NULISA?"],
            ),
            CandidateSubtopic(
                subtopic_id="subtopic_002",
                title="Industry Landscape: Bead vs. Sequencing Immunoassays",
                description="Comparative overview of standard bead multiplexing vs. sequencing readouts.",
                topic_angle="industry_context",
                relevance_to_main_topic="high",
                evidence_status="supported",
                evidence_ids=["ev_002"],
                suggested_questions=["When should researchers choose NGS over qPCR readout?"],
            ),
        ],
        verified_facts=[
            VerifiedFact(
                fact_id="fact_001",
                statement="Alamar Biosciences provides NULISAseq Immune 340 panels.",
                category="company_capability",
                evidence_type="company_website",
                evidence_ids=["ev_001"],
                confidence="high",
            ),
            VerifiedFact(
                fact_id="fact_002",
                statement="Multiplex immunoassay workflows require validation of dynamic range in serum/plasma.",
                category="industry_context",
                evidence_type="reputable_external",
                evidence_ids=["ev_002"],
                confidence="high",
            ),
        ],
        evidence=[
            EvidenceItem(
                evidence_id="ev_001",
                url="https://alamarbio.com/immune-340",
                quote="NULISAseq Immune 340 Panel provides deep profiling of immunology biomarkers.",
                source_type="company_website",
                domain="alamarbio.com",
            ),
            EvidenceItem(
                evidence_id="ev_002",
                url="https://nature.com/articles/proteomics-review",
                quote="Multiplex protein profiling requires dynamic range coverage spanning picogram to milligram concentrations.",
                source_type="reputable_external",
                domain="nature.com",
            ),
        ],
        research_gaps=[
            ResearchGap(
                gap_id="gap_001",
                topic_or_question="CLIA/CAP accredited laboratory turnaround times",
                reason="Company website does not publish specific central laboratory turn-around-time SLAs.",
                suggested_search_direction="Check certified CRO partner pages",
            )
        ],
        limitations=[
            "Do not use superlative claims such as 'most sensitive' without verified primary certification.",
            "Research-use-only assay kits must not be described as clinical diagnostic services.",
        ],
        verification=PromptVerification(
            is_verified=True,
            intent_valid=True,
            customer_relevant=True,
            reason="Alamar Biosciences develops NULISA immunology assay panels.",
        ),
    )

    with (
        patch.object(agent, "_understand_prompt_and_verify", return_value=mock_understanding),
        patch.object(agent, "_research_topic", return_value=([], [], [])),
        patch.object(agent, "_discover_subtopics_and_evidence", return_value=mock_guide),
    ):
        agent1 = _mock_agent1_data()
        client = ClientInput(company_name="Alamar Biosciences", website="https://alamarbio.com")
        guide = agent.analyze(agent1, client)

        assert guide is not None
        assert isinstance(guide, GroundTruthGuide)
        assert guide.main_topic == "Immunology biomarker assay providers and platform capabilities"
        assert guide.prompt_scope == "hybrid"
        assert len(guide.candidate_subtopics) == 2
        assert guide.candidate_subtopics[0].topic_angle == "company_capability"
        assert guide.candidate_subtopics[1].topic_angle == "industry_context"

        # Check evidence distinction
        assert len(guide.verified_facts) == 2
        assert guide.verified_facts[0].evidence_type == "company_website"
        assert guide.verified_facts[1].evidence_type == "reputable_external"

        # Check evidence items
        assert len(guide.evidence) == 2
        assert guide.evidence[0].source_type == "company_website"
        assert guide.evidence[1].source_type == "reputable_external"

        # Check gaps & limitations
        assert len(guide.research_gaps) == 1
        assert len(guide.limitations) == 2


def test_pipeline_agent3_node_handles_null():
    """A negative verification outcome is preserved as a typed business result."""
    agent1_data = _mock_agent1_data()
    state = PipelineState(
        campaign_id=81,
        prompt_id="1409",
        website="https://unrelated.com",
        company_name="Unrelated Corp",
        prompt_analysis=PromptAnalysis.model_validate(agent1_data),
    )

    rejected_guide = GroundTruthGuide(
        verification=PromptVerification(
            is_verified=False,
            intent_valid=True,
            customer_relevant=False,
            reason="Customer relevance was not evidenced.",
        )
    )
    with patch.object(GroundTruthAgent, "analyze", return_value=rejected_guide):
        result = agent3_node(state)
        assert result.get("guide") is rejected_guide
        assert "errors" in result
        assert "verification did not pass" in result["errors"][0]
