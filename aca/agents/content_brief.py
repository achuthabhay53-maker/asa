"""Agent 4: Strategy and Blog Brief Builder for ACA.

Combines the outputs of:
- Agent 1 (Prompt & Answer Intelligence)
- Agent 2 (Source & Content Intelligence)
- Agent 3 (Ground Truth & Verification)

into a structured, evidence-backed content brief for Agent 5 (Writer).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv

from aca.models.content_brief import BriefSection, ContentBrief, WritingStyleGuide

load_dotenv()


def _extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^\s*```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as first_error:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(
                "Agent 4 did not return a JSON object. "
                f"Response begins: {text[:500]!r}"
            ) from first_error
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            snippet_start = max(0, exc.pos - 100)
            snippet_end = min(len(text), exc.pos + 100)
            raise ValueError(
                "Agent 4 returned malformed JSON "
                f"(line {exc.lineno}, column {exc.colno}): {exc.msg}. "
                f"Near error: {text[snippet_start:snippet_end]!r}"
            ) from exc
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from Claude.")
    return data


def _to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    return {}


class ContentBriefAgent:
    """Agent 4: Builds the strategy and content brief for the writer."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self.model = (
            model
            or os.getenv("ANTHROPIC_MODEL_NAME")
            or "claude-haiku-4-5-20251001"
        )

    def build_brief(
        self,
        agent1_data: Any,
        agent2_data: Any,
        agent3_data: Any,
    ) -> ContentBrief:
        """Create a structured brief from Agent 1, 2, and 3 outputs."""
        a1 = _to_dict(agent1_data)
        a2 = _to_dict(agent2_data)
        a3 = _to_dict(agent3_data)

        # 1. Extract Agent 1 Prompt Intelligence
        p_under = a1.get("prompt_understanding") or {}
        a_under = a1.get("answer_understanding") or {}
        customer_name = a1.get("customer_name") or a3.get("company", {}).get("name") or "the company"
        prompt_text = a1.get("prompt") or ""
        intent = p_under.get("intent") or a3.get("search_intent") or ""
        target_audience = p_under.get("audience") or a3.get("target_audience") or "Biopharma professionals"
        key_questions = p_under.get("key_questions") or []
        comparison_criteria = a_under.get("comparison_criteria") or []

        # 2. Extract Agent 2 Content Intelligence
        source_headings = []
        source_styles = []
        for analysis in a2.get("content_analyses") or []:
            ca = analysis.get("content_analysis") or {}
            c_struct = ca.get("content_structure") or {}
            headings = c_struct.get("headings") or []
            if headings:
                source_headings.extend(headings[:6])
            w_style = ca.get("writing_style") or {}
            if w_style:
                source_styles.append(w_style)

        # 3. Extract Agent 3 Ground Truth
        handoff = a3.get("agent4_handoff") or a3
        main_topic = handoff.get("topic") or a3.get("topic") or a3.get("main_topic") or prompt_text
        relevant_findings = handoff.get("relevant_findings") or a3.get("relevant_findings") or []
        evidence_items = handoff.get("evidence") or a3.get("evidence") or []
        gaps = handoff.get("information_still_needed") or handoff.get("research_gaps") or a3.get("research_gaps") or []
        boundaries = handoff.get("boundaries") or a3.get("boundaries") or []
        candidate_subtopics = handoff.get("candidate_subtopics") or a3.get("candidate_subtopics") or []
        related_topics = [
            str(value)
            for subtopic in candidate_subtopics
            for value in [subtopic.get("title"), subtopic.get("description")]
            if value
        ]
        question_based_queries = [
            str(question)
            for subtopic in candidate_subtopics
            for question in subtopic.get("suggested_questions") or []
            if question
        ]

        system_prompt = """You are Agent 4: Strategy and Blog Brief Builder in an automated content intelligence workflow.
Your responsibility is to combine the outputs of Agent 1 (Prompt Understanding), Agent 2 (Content Intelligence), and Agent 3 (Ground Truth and Verification) into a structured, evidence-backed brief for Agent 5 (Writer).

Do NOT write the complete blog article. You are creating the architectural brief that guides the writer.

Your Responsibilities:
1. Establish the article objective:
   - Make the original prompt's intent central.
   - Address the buyer's evaluation criteria (from Agent 1).
2. Develop the content structure:
   - Select 4 to 6 focused, high-value sections (H2s and optional H3s).
   - Use competitor topics and structure (from Agent 2) as inspiration, but tailor strictly to the customer's verified strengths.
3. Map verified evidence:
   - Explicitly assign relevant evidence IDs (e.g. "ev_001") and source URLs to the appropriate sections.
   - Every factual claim made in a section must be traceable to Agent 3's evidence.
4. Define each section:
   - Heading, section objective, key questions to answer, assigned evidence IDs, and specific guidelines.
5. Provide writing instructions:
   - Tone, audience level, paragraph structure, and formatting elements (e.g., comparison table, feature lists, FAQ).
   - Create SEO/AEO guidance from Agent 3's verified topic, candidate subtopics, and suggested questions. Choose one natural primary keyword, related secondary keywords/topics, and question-based queries. Do not invent search-volume or ranking metrics.
   - Provide 3 to 6 relevant FAQ questions based on search intent and the supplied question-based queries. Use questions answerable from the article's planned evidence-backed content.
6. Handle missing information & boundaries:
   - Enforce Agent 3's boundaries.
   - Explicitly list unsupported claims that Agent 5 must NOT make.

Return one JSON object matching this exact schema:
{
  "article_objective": "Clear statement of what the blog must accomplish and who it is for",
  "target_audience": "Specific audience description",
  "search_intent": "What the reader is trying to learn or evaluate",
  "primary_keyword": "The central natural-language search phrase derived from the prompt and Agent 3 topic",
  "secondary_keywords": ["Relevant phrase derived from Agent 3 findings"],
  "related_topics": ["Relevant related topic grounded in Agent 3 candidate subtopics"],
  "question_based_queries": ["Relevant question grounded in Agent 3 suggested questions"],
  "faq_questions": ["Relevant question answerable from evidence-backed article content"],
  "recommended_title": "Compelling, SEO-friendly working title",
  "alternative_titles": ["Alternative title 1", "Alternative title 2"],
  "sections": [
    {
      "heading": "Section Heading",
      "level": "H2",
      "section_objective": "What this section covers and why it matters to the reader",
      "topics_and_questions": ["Key question or topic 1", "Key question or topic 2"],
      "evidence_ids": ["ev_001"],
      "source_urls": ["https://..."],
      "guidelines_and_gaps": ["Specific constraint or note for this section"]
    }
  ],
  "writing_style": {
    "tone": "e.g. Authoritative, peer-to-peer scientific, objective",
    "audience_level": "e.g. Technical / translational scientist",
    "paragraph_style": "e.g. Concise paragraphs (3-4 sentences) with bulleted highlights",
    "formatting_elements": ["Feature comparison table", "Bulleted lists", "FAQ section"],
    "terminology": ["Specific trademarks or proper nouns"]
  },
  "unsupported_claims_to_avoid": [
    "Specific unsupported claim 1",
    "Specific unsupported claim 2"
  ]
}
JSON only."""

        payload = {
            "customer_name": customer_name,
            "original_prompt": prompt_text,
            "agent1_prompt_intelligence": {
                "intent": intent,
                "topic": p_under.get("topic") or main_topic,
                "target_audience": target_audience,
                "user_goal": p_under.get("user_goal"),
                "key_questions": key_questions,
                "comparison_criteria": comparison_criteria,
            },
            "agent2_content_intelligence": {
                "benchmark_headings": source_headings[:12],
                "observed_styles": source_styles[:2],
            },
            "agent3_ground_truth": {
                "main_topic": main_topic,
                "search_intent": a3.get("search_intent") or intent,
                "related_topics": related_topics,
                "question_based_queries": question_based_queries,
                "relevant_findings": relevant_findings,
                "evidence_items": [
                    {
                        "evidence_id": e.get("evidence_id"),
                        "url": e.get("url"),
                        "quote": e.get("quote", "")[:160],
                    }
                    for e in evidence_items
                ],
                "research_gaps": [
                    g.get("topic_or_question") or g.get("reason")
                    for g in gaps
                ],
                "candidate_subtopics": [
                    {
                        "title": s.get("title"),
                        "description": s.get("description"),
                        "evidence_ids": s.get("evidence_ids"),
                    }
                    for s in candidate_subtopics[:6]
                ],
                "boundaries": boundaries,
            },
        }

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        user_content = json.dumps(payload, ensure_ascii=False)
        request = {
            "model": self.model,
            "max_tokens": 6096,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": "{\n  \"article_objective\":"},
            ],
        }
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=request,
            timeout=120.0,
        )
        response.raise_for_status()
        response_data = response.json()
        blocks = response_data.get("content") or []
        raw = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        try:
            parsed = _extract_json('{\n  "article_objective":' + raw)
        except ValueError as first_error:
            retry_request = {
                **request,
                "max_tokens": 8192,
                "messages": [{
                    "role": "user",
                    "content": (
                        user_content
                        + "\n\nReturn the entire response as one syntactically valid JSON "
                        "object matching the schema. Do not add markdown or commentary."
                    ),
                }],
            }
            retry = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=retry_request,
                timeout=120.0,
            )
            retry.raise_for_status()
            retry_data = retry.json()
            retry_blocks = retry_data.get("content") or []
            retry_text = "\n".join(
                b.get("text", "") for b in retry_blocks if b.get("type") == "text"
            )
            try:
                parsed = _extract_json(retry_text)
            except ValueError as retry_error:
                stop_reason = retry_data.get("stop_reason")
                raise RuntimeError(
                    "Agent 4 returned invalid JSON after one retry. "
                    f"stop_reason={stop_reason!r}; first error: {first_error}; "
                    f"retry error: {retry_error}"
                ) from retry_error

        # Build BriefSection models
        sections = []
        for s in parsed.get("sections") or []:
            sections.append(
                BriefSection(
                    heading=str(s.get("heading") or ""),
                    level=str(s.get("level") or "H2"),
                    section_objective=str(s.get("section_objective") or ""),
                    topics_and_questions=[str(q) for q in (s.get("topics_and_questions") or [])],
                    evidence_ids=[str(e) for e in (s.get("evidence_ids") or [])],
                    source_urls=[str(u) for u in (s.get("source_urls") or [])],
                    guidelines_and_gaps=[str(g) for g in (s.get("guidelines_and_gaps") or [])],
                )
            )

        ws = parsed.get("writing_style") or {}
        style_guide = WritingStyleGuide(
            tone=str(ws.get("tone") or "Professional, authoritative, peer-to-peer scientific"),
            audience_level=str(ws.get("audience_level") or "Technical / scientific"),
            paragraph_style=str(ws.get("paragraph_style") or "Concise paragraphs (3-4 sentences) with bulleted lists"),
            formatting_elements=[str(f) for f in (ws.get("formatting_elements") or [])],
            terminology=[str(t) for t in (ws.get("terminology") or [])],
        )

        return ContentBrief(
            article_objective=str(parsed.get("article_objective") or ""),
            target_audience=str(parsed.get("target_audience") or target_audience),
            search_intent=str(parsed.get("search_intent") or intent),
            primary_keyword=str(parsed.get("primary_keyword") or main_topic),
            secondary_keywords=[str(value) for value in (parsed.get("secondary_keywords") or [])],
            related_topics=[str(value) for value in (parsed.get("related_topics") or related_topics)],
            question_based_queries=[
                str(value) for value in (parsed.get("question_based_queries") or question_based_queries)
            ],
            faq_questions=[str(value) for value in (parsed.get("faq_questions") or [])],
            recommended_title=str(parsed.get("recommended_title") or ""),
            alternative_titles=[str(t) for t in (parsed.get("alternative_titles") or [])],
            sections=sections,
            writing_style=style_guide,
            unsupported_claims_to_avoid=[str(c) for c in (parsed.get("unsupported_claims_to_avoid") or boundaries)],
        )
