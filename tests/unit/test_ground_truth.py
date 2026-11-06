"""Unit tests for Agent 3: Ground Truth & Research Discovery."""

from __future__ import annotations

import json
import logging
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from src.agents.aimpact_blog.ground_truth import (
    GroundTruthAgent,
    CheckedClaim,
    _EvidenceRegistry,
    _apply,
    _assemble,
    _deduplicate_claims,
    _is_superlative,
    _handoff_boundaries,
    _merge_pairs,
    _mismatch_reason,
    _relevant_snippet,
    _scope_note,
    build_prompt_context,
)
from src.models.ground_truth import (
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
from src.models.prompt_intelligence import PromptAnalysis
from src.orchestrator.pipeline import PipelineState, agent3_node
from src.tools.web_research import ExtractedPage, SearchResponse, SearchResult


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

    def complete(system: str, payload: dict, max_tokens: int, **_audit_fields: str) -> dict:
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
            "human_review_required": False,
            "reason": "The retrieved pages did not establish this customer's VR training capability.",
            "quote": "",
            "source_url": "",
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


def test_agent3_call_audit_logs_hash_claims_sources_and_actual_input_tokens(monkeypatch, caplog):
    monkeypatch.setenv("ACA_AGENT3_CALL_AUDIT", "1")
    response = SimpleNamespace(
        json=lambda: {
            "usage": {"input_tokens": 321, "output_tokens": 14},
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "name": "submit_agent3_result",
                    "input": {"result": "ok"},
                }
            ],
        },
        raise_for_status=lambda: None,
    )
    agent = GroundTruthAgent(api_key="mock-key", model="mock-model")
    payload = {
        "claim_id": "fact_007",
        "claim": "The company provides a biomarker assay.",
        "pages": [
            {"url": "HTTPS://WWW.Example.com:443/product/#details"},
            {"url": "https://www.example.com/product/"},
        ],
    }
    request = {}

    def fake_post(_url, **kwargs):
        request.update(kwargs)
        return response

    with patch("src.agents.aimpact_blog.ground_truth.httpx.post", side_effect=fake_post), caplog.at_level(
        logging.INFO,
        logger="src.agents.aimpact_blog.ground_truth",
    ):
        result = agent._complete(
            "Check the claim.",
            payload,
            max_tokens=500,
            stage="_read_pages",
            purpose="Check whether supplied pages support the claim.",
            schema={
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": False,
            },
        )

    audit_record = next(
        json.loads(record.getMessage().split("=", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("agent3_call_audit=")
    )
    assert result == {"result": "ok"}
    assert audit_record["stage"] == "_read_pages"
    assert audit_record["purpose"] == "Check whether supplied pages support the claim."
    assert len(audit_record["input_hash"]) == 64
    assert "fact_007" in audit_record["claim_ids"]
    assert len(audit_record["claim_ids"]) == 2
    assert audit_record["source_urls"] == ["https://www.example.com/product"]
    assert audit_record["input_tokens"] == 321
    assert "biomarker assay" not in json.dumps(audit_record)
    assert request["json"]["tool_choice"] == {
        "type": "tool",
        "name": "submit_agent3_result",
    }
    assert request["json"]["tools"][0]["input_schema"]["required"] == ["result"]
    assert len(request["json"]["messages"]) == 1


def test_agent3_call_audit_is_disabled_by_default(monkeypatch, caplog):
    monkeypatch.delenv("ACA_AGENT3_CALL_AUDIT", raising=False)
    response = SimpleNamespace(
        json=lambda: {
            "usage": {"input_tokens": 10, "output_tokens": 2},
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "name": "submit_agent3_result",
                    "input": {"result": "ok"},
                }
            ],
        },
        raise_for_status=lambda: None,
    )
    agent = GroundTruthAgent(api_key="mock-key", model="mock-model")

    with patch("src.agents.aimpact_blog.ground_truth.httpx.post", return_value=response), caplog.at_level(
        logging.INFO,
        logger="src.agents.aimpact_blog.ground_truth",
    ):
        agent._complete(
            "Check.",
            {"claim": "A test claim."},
            max_tokens=100,
            schema={
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": False,
            },
        )

    assert not any("agent3_call_audit=" in record.getMessage() for record in caplog.records)


def test_agent3_deduplicates_near_identical_claims_before_verification():
    claims = _deduplicate_claims(
        [
            {
                "claim": "Example Bio provides VR training for medical device teams.",
                "claim_type": "service",
                "relevance": "medium",
                "relevance_to_topic": "low",
                "checked_from": "prompt",
            },
            {
                "claim": "Example Bio provides VR training to medical device teams.",
                "claim_type": "service",
                "relevance": "high",
                "relevance_to_topic": "high",
                "checked_from": "content_opportunity",
            },
        ]
    )

    assert len(claims) == 1
    assert claims[0]["relevance"] == "high"
    assert claims[0]["relevance_to_topic"] == "high"


def test_agent3_batches_claim_verification_with_deduplicated_pages():
    page_quote = (
        "Example Bio provides immersive VR training for medical device teams "
        "through interactive clinical simulation."
    )
    page_text = ("General company overview without claim details. " * 90) + page_quote

    class Research:
        @staticmethod
        def search(*_args, **_kwargs):
            return SimpleNamespace(
                failed=False,
                results=[
                    SimpleNamespace(
                        url="https://example.com/training",
                        published="2026-11-02",
                    ),
                    SimpleNamespace(
                        url="https://example.com/training/",
                        published="2026-11-02",
                    ),
                ],
            )

        @staticmethod
        def extract(url, **_kwargs):
            return SimpleNamespace(url=url, text=page_text)

    agent = GroundTruthAgent(api_key="mock-key", model="mock-model", research=Research())
    claims = [
        {
            "topic": "Capability",
            "claim": f"Example Bio supports training option {index}.",
            "claim_type": "platform_capability",
            "relevance": "high",
            "relevance_to_topic": "high",
            "checked_from": "prompt",
        }
        for index in range(1, 5)
    ]
    calls = []

    def complete(_system, payload, max_tokens=None, **audit_fields):
        calls.append((payload, audit_fields))
        assert max_tokens is not None
        json.dumps(payload)
        assert payload["pages"][0]["published"] == "2026-11-02"
        return {
            "claims": [
                {
                    "claim_id": claim["claim_id"],
                    "pages": [
                        {
                            "page_id": claim["page_ids"][0],
                            "support": "full",
                            "quote": page_quote,
                            "narrow_statement": "",
                            "wording_limit": "",
                        }
                    ],
                }
                for claim in payload["claims"]
            ]
        }

    agent._complete = complete
    checked, pages = agent._check_questions_batch(claims, "Example Bio", "example.com")

    assert len(checked) == 4
    assert all(result.fact.status == "verified" for result in checked)
    assert len(calls) == 1
    assert len(calls[0][0]["claims"]) == 4
    assert len(calls[0][0]["pages"]) == 1
    assert len(calls[0][0]["pages"][0]["text"]) <= 2_200
    assert len(calls[0][0]["pages"][0]["text"]) < len(page_text) // 2
    assert page_quote in calls[0][0]["pages"][0]["text"]
    assert calls[0][0]["pages"][0]["text"].count("General company overview") <= 1
    assert calls[0][1]["stage"] == "_read_pages_batch"
    assert len(pages) == 1
    assert checked[0].fact.evidence[0].published == date(2026, 11, 2)


def test_agent3_reuses_cached_page_extraction_across_claim_searches():
    page_text = "Example Bio provides immersive VR training for medical device teams."
    calls = {"extract": 0}

    class Research:
        @staticmethod
        def search(*_args, **_kwargs):
            return SimpleNamespace(
                failed=False,
                results=[SimpleNamespace(url="https://example.com/training/", published=None)],
            )

        @staticmethod
        def extract(url, **_kwargs):
            calls["extract"] += 1
            return SimpleNamespace(url=url, text=page_text)

    agent = GroundTruthAgent(api_key="mock-key", model="mock-model", research=Research())

    first, first_failed = agent._collect_pages("VR training", "example.com", "Example Bio", True)
    second, second_failed = agent._collect_pages("medical device simulation", "example.com", "Example Bio", True)

    assert not first_failed and not second_failed
    assert calls["extract"] == 1
    assert first[0]["text"] == second[0]["text"] == page_text


def test_agent3_relevant_snippet_keeps_source_text_and_drops_irrelevant_page_text():
    relevant = "Ethosh offers VR clinical scenario simulations for medical device teams."
    page_text = ("Corporate overview with generic company history. " * 30) + relevant

    snippet = _relevant_snippet(page_text, "VR clinical scenario simulations medical device", 500)

    assert len(snippet) <= 500
    assert len(snippet) < len(page_text) // 5
    assert relevant in snippet
    assert snippet.count("Corporate overview") <= 1


def test_agent3_generates_prompt_and_opportunity_claims_in_one_llm_call():
    agent = GroundTruthAgent(api_key="mock-key", model="mock-model")
    context = build_prompt_context(
        _mock_agent1_data(),
        "https://example.com",
        "Example Bio",
    )
    opportunity = context.content_opportunities[0]
    calls = []

    def complete(_system, payload, max_tokens=None, **audit_fields):
        calls.append((payload, audit_fields))
        assert max_tokens == 3500
        return {
            "questions": [
                {
                    "axis": "capability",
                    "claim_type": "service",
                    "relevance_to_original_prompt": "high",
                    "relevance_to_topic": "high",
                    "topic": "VR training",
                    "claim": "Example Bio provides VR training for medical device companies.",
                }
            ],
            "opportunities": [
                {
                    "opportunity": opportunity,
                    "topic": "VR training",
                    "claim": "Example Bio provides VR training for medical device companies.",
                    "claim_type": "service",
                    "relevance_to_original_prompt": "high",
                    "relevance_to_topic": "high",
                }
            ],
        }

    agent._complete = complete
    items = agent._research_items(context)
    questions = agent._normalize_questions(items)
    opportunities = agent._normalize_opportunity_claims(context, items)

    assert len(calls) == 1
    assert calls[0][1]["stage"] == "_research_items"
    assert len(questions) == 1
    assert len(opportunities) == 1


def test_subtopic_discovery_sends_compact_verified_evidence_and_builds_facts_deterministically():
    agent = GroundTruthAgent(api_key="mock-key", model="mock-model")
    agent1 = {
        **_mock_agent1_data(),
        "prompt": "Find VR training providers for medical device teams.",
        "prompt_understanding": {
            "intent": "Find VR training providers",
            "topic": "VR training for medical device teams",
            "audience": "Medical device teams",
            "user_goal": "Find providers",
            "key_questions": [],
        },
        "content_opportunities": [],
    }
    context = build_prompt_context(agent1, "https://example.com", "Example Bio")
    quote = "Example Bio offers immersive VR clinical scenario simulations for device teams."
    page_text = quote + " " + ("Unrelated website detail without claims. " * 400)
    claim = "Example Bio offers immersive VR clinical scenario simulations for device teams."
    page_url = "https://example.com/vr-training"
    checked = _apply(
        "fact_001",
        "Capability",
        claim,
        [
            {
                "support": "full",
                "quote": quote,
                "url": page_url,
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
        [page_url],
    )
    understanding = {
        "main_topic": "VR training for medical device teams",
        "search_intent": "Find training providers",
        "target_audience": "Medical device teams",
        "prompt_scope": "hybrid",
        "is_verified": True,
        "intent_valid": True,
        "customer_relevant": True,
        "reason": "The official page supports the topic.",
        "quote": quote,
        "source_url": page_url,
        "retrieved_pages": [{"url": page_url, "text": page_text, "published": None}],
    }
    captured = {}

    def complete(_system, payload, max_tokens=None, **_audit_fields):
        captured.update(payload)
        return {
            "main_topic": "VR training for medical device teams",
            "search_intent": "Find training providers",
            "target_audience": "Medical device teams",
            "prompt_scope": "hybrid",
            "candidate_subtopics": [
                {
                    "title": "VR Clinical Scenario Simulations",
                    "description": "Training scenarios for device teams.",
                    "topic_angle": "company_capability",
                    "relevance_to_main_topic": "high",
                    "evidence_quotes": [quote],
                }
            ],
            "research_gaps": [],
            "limitations": [],
        }

    agent._complete = complete
    guide = agent._discover_subtopics_and_evidence(
        context,
        understanding,
        [{"url": page_url, "text": page_text, "published": None}],
        [(
            {
                "topic": "Capability",
                "claim": claim,
                "claim_type": "service",
                "relevance": "high",
                "relevance_to_topic": "high",
                "checked_from": "prompt",
            },
            checked,
        )],
        [],
        "Example Bio",
        "example.com",
    )

    assert "research_evidence" not in captured
    assert "known_facts" not in captured
    assert len(captured["verified_evidence"]) == 1
    assert captured["verified_evidence"][0]["source_url"] == page_url
    assert captured["verified_evidence"][0]["supporting_quote"] == quote
    assert len(json.dumps(captured, ensure_ascii=False)) < len(page_text) // 4
    assert json.dumps(captured, ensure_ascii=False).count("Unrelated website detail") == 0
    assert guide.verified_facts[0].fact_id == "fact_001"
    assert guide.verified_facts[0].statement == claim
    assert guide.verified_facts[0].status == "verified"
    assert guide.candidate_subtopics[0].verified_fact_ids == ["fact_001"]
    assert guide.candidate_subtopics[0].support_status == "fully_supported"


def test_ground_truth_serializes_duplicate_lists_once():
    gap = ResearchGap(
        gap_id="gap_001",
        topic_or_question="Training delivery",
        missing_claim="The company provides onsite training.",
        reason="The website does not specify delivery location.",
    )
    finding = {"claim": "Example Bio provides VR training."}
    evidence = {"evidence_id": "ev_001", "url": "https://example.com", "quote": "A verified quote."}
    subtopic = {"title": "VR training", "description": "Training approach"}
    guide = GroundTruthGuide(
        relevant_findings=[finding],
        evidence=[evidence],
        information_still_needed=[gap],
        research_gaps=[gap],
        candidate_subtopics=[subtopic],
        boundaries=["Do not infer outcomes."],
        limitations=["Do not infer outcomes."],
        agent4_handoff={
            "topic": "VR training",
            "search_intent": "Find providers",
            "target_audience": "Medical device teams",
            "relevant_findings": [finding],
            "evidence": [evidence],
            "information_still_needed": [gap],
            "candidate_subtopics": [subtopic],
            "boundaries": ["Do not infer outcomes."],
            "company": {"name": "Example Bio", "website": "https://example.com"},
        },
    )

    payload = guide.model_dump(mode="json")

    assert len(payload["research_gaps"]) == 1
    assert len(payload["evidence"]) == 1
    assert len(payload["relevant_findings"]) == 1
    assert "information_still_needed" not in payload
    assert "limitations" not in payload
    assert set(payload["agent4_handoff"]) == {"task", "company", "verification"}


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


def test_evidence_registry_only_accepts_official_site_quotes():
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
    assert same_quote_other_source is None
    assert fabricated is None
    assert [item.url for item in registry.items] == [
        "HTTPS://WWW.Example.com:443/product/#details",
    ]

    external_evidence = Evidence(
        url="https://example.org/review",
        quote=quote,
        source_tier="trusted_third_party",
    )
    assert registry.register_checked_evidence(external_evidence) is None


def test_evidence_registry_keeps_validated_checked_quote_and_original_source_url():
    from src.models.ground_truth import Evidence as GroundTruthEvidence

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
                "url": "https://www.example.com/service-limitations",
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

    assert supported.fact.status == "verified"
    assert supported.fact.evidence[0].source_tier == "first_party"
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

    with patch.object(agent, "_collect_optional_pages", return_value=([], True)) as collect:
        result = agent._understand_prompt_and_verify(
            context,
            "Example Bio",
            "example.com",
        )

    collect.assert_called_once_with(
        f"{context.original_prompt} {context.prompt_understanding.intent}".strip(),
        "example.com",
        "Example Bio",
        first_party=True,
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


def test_third_party_quote_cannot_verify_customer_claim():
    result = _apply(
        "fact_external",
        "Service",
        "Example Bio provides the service to customers",
        [
            {
                "support": "full",
                "quote": "Example Bio provides the service to customers.",
                "url": "https://example.org/review",
                "published": None,
                "client_host": "example.com",
                "narrow_statement": "Example Bio provides the service to customers",
                "wording_limit": "",
            }
        ],
        False,
        "service",
        "high",
        "high",
        "prompt",
        ["https://example.org/review"],
    )

    assert result.fact.status == "inconclusive"
    assert result.fact.evidence == []
    assert result.gap is not None


def test_check_question_does_not_run_third_party_fallback():
    agent = GroundTruthAgent(api_key="mock-key", model="mock-model")
    question = {
        "topic": "Capability",
        "claim": "Example Bio offers the capability",
        "claim_type": "platform_capability",
        "relevance": "high",
        "relevance_to_topic": "high",
        "checked_from": "prompt",
    }
    with patch.object(agent, "_collect_optional_pages", return_value=([], True)) as collect:
        result = agent._check_question(
            "fact_001",
            question,
            "Example Bio",
            "example.com",
        )

    collect.assert_called_once_with(
        question["claim"],
        "example.com",
        "Example Bio",
        first_party=True,
    )
    assert result.fact.status == "inconclusive"


def test_optional_external_page_request_is_not_sent_to_research():
    class Research:
        def search(self, *_args, **_kwargs):
            raise AssertionError("Third-party search must not be called.")

    agent = GroundTruthAgent(
        api_key="mock-key",
        model="mock-model",
        research=Research(),
    )

    pages, failed = agent._collect_optional_pages(
        "company service",
        "example.com",
        "Example Bio",
        first_party=False,
    )

    assert pages == []
    assert failed is False


def test_collect_pages_does_not_retrieve_third_party_search_results():
    redirect_url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/opaque"
    official_url = "https://www.ethosh.com/virtual-reality-training"

    class Research:
        def search(self, query: str, **kwargs) -> SearchResponse:
            return SearchResponse(
                results=[
                    SearchResult(url=redirect_url, title="Ethosh capability page"),
                    SearchResult(url=official_url, title="Official capability page"),
                ]
            )

        def extract(self, url: str, *, max_chars: int) -> ExtractedPage:
            assert url == official_url
            return ExtractedPage(
                url=official_url,
                text="Ethosh provides immersive virtual reality training for medical device teams.",
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
    assert pages[0]["url"] == official_url


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
