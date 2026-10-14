"""Unit tests for Agent 3: Ground Truth & Research Discovery."""
from __future__ import annotations

from unittest.mock import patch

from aca.agents.ground_truth import GroundTruthAgent
from aca.models.ground_truth import (
    CandidateSubtopic,
    ClientInput,
    EvidenceItem,
    GroundTruthGuide,
    PromptVerification,
    ResearchGap,
    VerifiedFact,
)
from aca.models.prompt_intelligence import PromptAnalysis
from aca.orchestrator.pipeline import PipelineState, agent3_node


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


def test_ground_truth_returns_null_when_irrelevant():
    """If customer is not relevant or intent is invalid, Agent 3 must return None (null)."""
    agent = GroundTruthAgent(api_key="mock-key", model="mock-model")

    mock_understanding = {
        "main_topic": "Footwear fashion",
        "is_verified": False,
        "intent_valid": False,
        "customer_relevant": False,
        "reason": "Customer is an apparel footwear brand with no relation to immunology biomarker assays.",
    }

    with patch.object(agent, "_understand_prompt_and_verify", return_value=mock_understanding):
        agent1 = _mock_agent1_data()
        client = ClientInput(company_name="Acme Footwear", website="https://acmefootwear.example.com")
        guide = agent.analyze(agent1, None, client)
        assert guide is None


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

    with patch.object(agent, "_understand_prompt_and_verify", return_value=mock_understanding), \
         patch.object(agent, "_research_topic", return_value=([], [], [])), \
         patch.object(agent, "_discover_subtopics_and_evidence", return_value=mock_guide):

        agent1 = _mock_agent1_data()
        client = ClientInput(company_name="Alamar Biosciences", website="https://alamarbio.com")
        guide = agent.analyze(agent1, None, client)

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
    """PipelineState agent3_node returns guide=None and error description when Agent 3 returns None."""
    agent1_data = _mock_agent1_data()
    state = PipelineState(
        campaign_id=81,
        prompt_id="1409",
        website="https://unrelated.com",
        company_name="Unrelated Corp",
        prompt_analysis=PromptAnalysis.model_validate(agent1_data),
    )

    with patch.object(GroundTruthAgent, "analyze", return_value=None):
        result = agent3_node(state)
        assert result.get("guide") is None
        assert "errors" in result
        assert "no blog content can be written" in result["errors"][0]
