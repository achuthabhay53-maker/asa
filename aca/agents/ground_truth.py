"""Agent 3: Ground truth for AImpact.

Starts from the original prompt and Agent 1's reading of it. The website
is supplied by the user. Prompt claims and each content opportunity are
searched, then marked as matching the original prompt, the topic being
searched, both, or neither.

The model's own knowledge is not evidence.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from typing import Any, NamedTuple
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from aca.models.ground_truth import (
    CheckedFrom,
    ClaimType,
    ClientInput,
    Confidence,
    DoNotWrite,
    Evidence,
    EvidenceBackedTopic,
    Fact,
    FactStatus,
    GroundTruthGuide,
    HeadingItem,
    LimitedStatement,
    PromptContext,
    PromptUnderstandingInput,
    PromptVerification,
    Relevance,
    ScopeNote,
    SourceTier,
    TopicDecision,
    TopicVerdict,
    UnresolvedClaim,
)
from aca.tools.web_research import WebResearch, get_web_research

load_dotenv()

# How many capability questions to test. Override with GroundTruthAgent(max_questions=...).
MAX_QUESTIONS = 6
PAGES_PER_PASS = 2
PAGE_CHARS = 12_000

_STANDING_BANS = [
    "industry leader",
    "most sensitive",
    "best",
    "largest",
    "leading",
    "#1",
]
_SUPERLATIVES = (
    "industry leader",
    "market leader",
    "most sensitive",
    "best",
    "largest",
    "leading",
    "number one",
    "#1",
    "world's first",
    "unmatched",
)
_CLAIM_TYPES = {
    "product",
    "platform_capability",
    "service",
    "certification",
    "geographic_service_availability",
    "other",
}
_RELEVANCE = {"high", "medium", "low"}
_LIMITATIONS = [
    "A lack of supporting evidence does not prove that a capability or certification does not exist.",
    "Research-use products and platform capabilities must not be presented as diagnostic services without supporting evidence.",
    "A quote that supports a narrower statement does not verify the broader claim.",
    "A customer or partner using the company's instrument is not evidence of the company's own locations or service turnaround.",
    "Sensitivity or multiplexing language is not evidence of comparative superiority or cost.",
]
_COMPARE = (
    "cost",
    "cheaper",
    "superior",
    "better than",
    "faster than",
    "compared with",
    "compared to",
    "differentiates",
    "differentiation",
    "advantage",
    "competitors",
)
_STOP = {
    "alamar", "biosciences", "that", "with", "from", "this", "have", "has", "been",
    "were", "will", "into", "about", "their", "which", "than", "such", "also",
}


class CheckedClaim(NamedTuple):
    fact: Fact
    verdict: TopicVerdict
    block: DoNotWrite | None
    extra_bans: list[str]
    gap: UnresolvedClaim | None
    statement: str | None
    wording_limit: str | None
    asked_claim: str


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _host(url: str) -> str:
    host = (urlparse(url if "://" in url else f"https://{url}").hostname or "").lower()
    return host.removeprefix("www.")


def _string_list(raw: Any) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for item in raw or []:
        text = _clean(str(item))
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            values.append(text)
    return values


def _extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^\s*```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Claude did not return a valid JSON object.")
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from Claude.")
    return data


def _quote_in_page(quote: str, page: str) -> bool:
    quote_n = _clean(quote)
    page_n = _clean(page)
    return len(quote_n) >= 30 and quote_n in page_n


def _is_superlative(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in _SUPERLATIVES)


def _parse_date(value: Any) -> date | None:
    if not value or not isinstance(value, str):
        return None
    raw = value.strip()[:10]
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


def _tier(url: str, client_host: str) -> SourceTier:
    return "first_party" if _host(url) == client_host else "other"


def build_prompt_context(agent1: Any, website: str, company_name: str | None = None) -> PromptContext:
    """Build Agent 3's core input from Agent 1. The website is the user's."""
    first = _dump(agent1)
    understanding = first.get("prompt_understanding") or {}
    name = (company_name or first.get("customer_name") or "").strip()
    site = website.strip()
    if not name or not site:
        raise ValueError("Agent 1 customer name and a user-supplied website are required.")
    return PromptContext(
        prompt_id=str(first.get("prompt_id") or "").strip(),
        original_prompt=str(first.get("prompt") or "").strip(),
        customer=ClientInput(company_name=name, website=site),
        prompt_understanding=PromptUnderstandingInput(
            intent=str(understanding.get("intent") or "").strip(),
            topic=str(understanding.get("topic") or "").strip(),
            audience=(str(understanding.get("audience")).strip() if understanding.get("audience") else None),
            user_goal=str(understanding.get("user_goal") or "").strip(),
            key_questions=_string_list(understanding.get("key_questions")),
        ),
        content_opportunities=_string_list(first.get("content_opportunities")),
        answer_topics=_string_list((first.get("answer_understanding") or {}).get("key_topics")),
        comparison_criteria=_string_list((first.get("answer_understanding") or {}).get("comparison_criteria")),
    )


class GroundTruthAgent:
    def __init__(
        self,
        model: str | None = None,
        max_questions: int | None = None,
        research: WebResearch | None = None,
        api_key: str | None = None,
    ) -> None:
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        if max_questions is None:
            max_questions = MAX_QUESTIONS
        if max_questions < 1:
            raise ValueError("max_questions must be at least 1.")
        self.max_questions = max_questions
        self.api_key = api_key
        self.model = model or os.getenv("ANTHROPIC_MODEL_NAME") or "claude-haiku-4-5-20251001"
        self.research = research or get_web_research()

    def analyze(
        self,
        agent1: Any,
        agent2: Any,
        client: ClientInput,
    ) -> GroundTruthGuide | None:
        context = build_prompt_context(agent1, client.website, client.company_name)
        client_name = context.customer.company_name
        client_host = _host(context.customer.website)
        if not client_host:
            raise ValueError("The website must include a host, for example https://alamarbio.com.")
        del agent2  # cited-page topics are not the customer's capabilities

        # Step 1: Verify the original prompt and its intent for the customer using web research.
        verification = self._verify_prompt_and_intent(context, client_name, client_host)
        if not verification.is_verified or not verification.intent_valid or not verification.customer_relevant:
            # If the intent is invalid or the customer is not relevant to the requested topic, return null and stop.
            print(f"Agent 3: Intent is invalid or customer '{client_name}' is not relevant to '{context.original_prompt}'. Reason: {verification.reason}. Returning null and stopping.")
            return None

        # Step 2: Research the web to identify the customer's actual capabilities.
        questions = self._questions(context)[: self.max_questions]
        pairs: list[tuple[dict[str, str], CheckedClaim]] = []
        for opportunity in self._opportunity_claims(context):
            if opportunity.get("relevance") == "low" and opportunity.get("relevance_to_topic") == "low":
                pairs.append((opportunity, _unresolved(
                    f"fact_{len(pairs) + 1:03d}",
                    opportunity.get("topic") or "Content opportunity",
                    opportunity["claim"],
                    opportunity.get("claim_type") or "other",
                    "low",
                    "low",
                    "content_opportunity",
                    "do_not_write",
                    "inconclusive",
                    "Outside the original prompt and the topic being searched.",
                    [],
                    [],
                    gap=False,
                    asked_claim=opportunity["claim"],
                )))
                continue
            questions.append(opportunity)
        banned = list(_STANDING_BANS)
        for index, question in enumerate(questions, start=1):
            checked = self._check_question(
                fact_id=f"fact_{index:03d}",
                question=question,
                client_name=client_name,
                client_host=client_host,
            )
            checked = _apply_target(question, checked)
            pairs.append((question, checked))
            for item in checked.extra_bans:
                if item not in banned:
                    banned.append(item)
        guide = _assemble(context, pairs, banned)
        guide = guide.model_copy(update={"verification": verification})

        # Step 3: Extract evidence-backed topics, potential headings, and related topics suitable for blog content.
        guidance = self._extract_blog_guidance(context, guide.can_write, guide.can_write_with_limits, client_name)
        eb_topics = [
            t if isinstance(t, EvidenceBackedTopic) else EvidenceBackedTopic.model_validate(t)
            for t in guidance.get("evidence_backed_topics", [])
        ]
        headings = [str(h) for h in guidance.get("potential_headings", [])]
        structured_headings = [
            h if isinstance(h, HeadingItem) else HeadingItem.model_validate(h)
            for h in guidance.get("structured_headings", [])
        ]
        related_topics = [str(r) for r in guidance.get("related_topics", [])]

        return guide.model_copy(update={
            "evidence_backed_topics": eb_topics,
            "potential_headings": headings,
            "structured_headings": structured_headings,
            "related_topics": related_topics,
        })

    def _verify_prompt_and_intent(
        self,
        context: PromptContext,
        client_name: str,
        client_host: str,
    ) -> PromptVerification:
        """Verify the original prompt and its intent for the customer using web research."""
        query_fp = f"{context.original_prompt} {context.prompt_understanding.intent}".strip()
        first_pages, _ = self._collect_pages(query_fp, client_host, client_name, first_party=True)
        retrieved_pages = list(first_pages)
        if len(retrieved_pages) < PAGES_PER_PASS:
            web_query = f"{client_name} {context.original_prompt}"
            web_pages, _ = self._collect_pages(web_query, client_host, client_name, first_party=False)
            retrieved_pages.extend(web_pages)

        if not retrieved_pages:
            try:
                root_page = self.research.extract(f"https://{client_host}")
                if root_page and root_page.text.strip():
                    retrieved_pages.append({"url": root_page.url, "text": root_page.text, "published": None})
            except Exception:
                pass

        packed = [
            {"url": page["url"], "text": page["text"][:PAGE_CHARS]}
            for page in retrieved_pages
        ]

        system = """You are Agent 3 verifying whether an original prompt and its intent are valid and relevant for a customer using web research.
Return one JSON object:
{
  "is_verified": true|false,
  "intent_valid": true|false,
  "customer_relevant": true|false,
  "reason": "Detailed explanation of why the customer is or is not relevant, and whether the intent is valid.",
  "quote": "Exact contiguous quote from retrieved text if available, or empty string",
  "source_url": "URL of the page containing the quote, or empty string"
}
Rules:
- Verify whether the customer actually offers, develops, or provides capabilities, products, platforms, or services relevant to the requested topic/prompt.
- If the customer operates in this space (e.g. provides assays, biomarker platforms, reagents, testing services, or technology), mark customer_relevant: true, intent_valid: true, and is_verified: true.
- If the customer is completely unrelated to the requested topic (e.g. asked about immunology biomarker assays for a shoe company, automotive dealer, or unrelated consumer software), mark customer_relevant: false, intent_valid: false, and is_verified: false.
- If the intent itself is invalid, nonsensical, or unaddressable for this customer, mark is_verified: false.
- Base your judgment strictly on the retrieved pages and customer facts. Do NOT hallucinate.
- quote must be an exact contiguous quote from the supplied page text (at least 30 characters) if one is found.
- JSON only."""

        payload = {
            "customer_name": client_name,
            "website": f"https://{client_host}",
            "original_prompt": context.original_prompt,
            "prompt_understanding": context.prompt_understanding.model_dump(),
            "retrieved_pages": packed,
        }

        try:
            data = self._complete(system, payload, max_tokens=1500)
        except Exception as exc:
            print(f"  Prompt verification LLM failed: {exc}")
            return PromptVerification(
                is_verified=False,
                intent_valid=False,
                customer_relevant=False,
                reason=f"Verification failed: {exc}",
            )

        is_verified = bool(data.get("is_verified", False))
        intent_valid = bool(data.get("intent_valid", False))
        customer_relevant = bool(data.get("customer_relevant", False))
        reason = _clean(str(data.get("reason") or ""))
        quote = _clean(str(data.get("quote") or ""))
        source_url = str(data.get("source_url") or "")

        evidence_list: list[Evidence] = []
        if quote and source_url:
            for page in retrieved_pages:
                if page["url"] == source_url and _quote_in_page(quote, page["text"]):
                    evidence_list.append(Evidence(
                        url=source_url,
                        quote=quote,
                        source_tier=_tier(source_url, client_host),
                        published=page.get("published"),
                    ))
                    break

        return PromptVerification(
            is_verified=is_verified and intent_valid and customer_relevant,
            intent_valid=intent_valid,
            customer_relevant=customer_relevant,
            reason=reason or ("Verified" if is_verified else "Customer is not relevant or intent is invalid."),
            evidence=evidence_list,
        )

    def _extract_blog_guidance(
        self,
        context: PromptContext,
        can_write: list[Fact],
        can_write_with_limits: list[LimitedStatement],
        client_name: str,
    ) -> dict[str, Any]:
        """Extract evidence-backed topics, potential headings, and related topics suitable for blog content."""
        facts_summary = [
            {
                "fact_id": f.fact_id,
                "claim": f.claim,
                "evidence_quotes": [e.quote for e in f.evidence],
                "urls": [e.url for e in f.evidence],
            }
            for f in can_write
        ]
        limits_summary = [
            {
                "fact_id": l.fact_id,
                "statement": l.statement,
                "wording_limit": l.wording_limit,
                "evidence_quotes": [e.quote for e in l.evidence],
                "urls": [e.url for e in l.evidence],
            }
            for l in can_write_with_limits
        ]

        system = """You are Agent 3 extracting evidence-backed topics, potential headings, and related topics suitable for blog content based on verified customer capabilities.
Return one JSON object:
{
  "evidence_backed_topics": [
    {
      "topic": "Concise topic title",
      "description": "How this relates to the customer's verified capabilities and the original prompt",
      "evidence_fact_ids": ["fact_001", ...],
      "supporting_quotes": ["Exact quote from evidence"],
      "source_urls": ["URL"]
    }
  ],
  "potential_headings": [
    "H2 Heading text",
    ...
  ],
  "structured_headings": [
    {"level": "H2", "text": "Heading text", "intent_hint": "Editorial intent"},
    {"level": "H3", "text": "Subheading text", "intent_hint": "Editorial intent"}
  ],
  "related_topics": [
    "Related topic or search query for blog content clustering",
    ...
  ]
}
Rules:
- Base topics strictly on the verified facts and statements provided. Do NOT assert unverified claims or use superlatives (e.g. best, #1, industry leader).
- Potential headings should form an outline for an authoritative, helpful blog post that addresses the original prompt while highlighting verified customer capabilities.
- Related topics should suggest 4-8 relevant semantic topics, comparisons, or search angles suitable for content expansion, FAQs, or pillar articles.
- JSON only."""

        payload = {
            "customer_name": client_name,
            "original_prompt": context.original_prompt,
            "prompt_understanding": context.prompt_understanding.model_dump(),
            "verified_facts": facts_summary,
            "limited_statements": limits_summary,
        }

        try:
            data = self._complete(system, payload, max_tokens=2500)
        except Exception as exc:
            print(f"  Blog guidance extraction failed: {exc}")
            return {}

        eb_topics: list[EvidenceBackedTopic] = []
        for item in data.get("evidence_backed_topics") or []:
            if not isinstance(item, dict):
                continue
            topic = _clean(str(item.get("topic") or ""))
            if not topic:
                continue
            eb_topics.append(EvidenceBackedTopic(
                topic=topic,
                description=_clean(str(item.get("description") or "")),
                evidence_fact_ids=_string_list(item.get("evidence_fact_ids")),
                supporting_quotes=_string_list(item.get("supporting_quotes")),
                source_urls=_string_list(item.get("source_urls")),
            ))

        headings = _string_list(data.get("potential_headings"))
        structured: list[HeadingItem] = []
        for item in data.get("structured_headings") or []:
            if not isinstance(item, dict):
                continue
            text = _clean(str(item.get("text") or ""))
            if not text:
                continue
            level = str(item.get("level") or "H2").strip().upper()
            if level not in {"H2", "H3", "H4"}:
                level = "H2"
            structured.append(HeadingItem(
                level=level,
                text=text,
                intent_hint=_clean(str(item.get("intent_hint") or "")),
            ))

        related = _string_list(data.get("related_topics"))

        return {
            "evidence_backed_topics": eb_topics,
            "potential_headings": headings,
            "structured_headings": structured,
            "related_topics": related,
        }

    def _questions(self, context: PromptContext) -> list[dict[str, str]]:
        system = """You write the claims Agent 3 will verify about one customer.
Return one JSON object:
{"questions":[{"axis":"relevance|capability|geography","claim_type":"product|platform_capability|service|certification|geographic_service_availability|other","relevance_to_original_prompt":"high|medium|low","relevance_to_topic":"high|medium|low","topic":"","claim":""}]}
The original prompt supplies the search intent. The AI answers supply topics and comparison criteria. Verify a customer claim only when it helps answer that intent or matches one of those topics or criteria.
Cover these axes when the prompt or the answers call for them:
- relevance: what this customer offers that matches the original prompt
- capability: a product, platform, or service this customer would need for that prompt
- geography: presence or service availability in the region the prompt is about
Rules:
- Claims are about the named customer only. Do not try to prove every capability mentioned about other companies.
- Separate products, platform capabilities, services, certifications, and geographic availability. Do not bundle them into one claim.
- A product or platform claim must not say the customer performs contract testing unless that is a separate service claim.
- content_opportunities are searched on their own. Do not turn them into these questions.
- relevance_to_original_prompt scores the claim against the original prompt.
- relevance_to_topic scores the claim against prompt_understanding.topic. If that topic adds a region or scope that is not in the original prompt, the addition is answer-derived. Do not treat it as part of the original prompt.
- Do not use superlatives such as best, leader, largest, or most sensitive.
- Keep the list short. JSON only."""
        data = self._complete(system, context.model_dump(), max_tokens=1500)
        questions: list[dict[str, str]] = []
        seen: set[str] = set()
        labels = {"relevance": "Relevance", "capability": "Capability", "geography": "Geography"}
        for item in data.get("questions") or []:
            if not isinstance(item, dict):
                continue
            topic = _clean(str(item.get("topic") or ""))
            claim = _clean(str(item.get("claim") or ""))
            axis = str(item.get("axis") or "capability").strip().lower()
            label = labels.get(axis, "Capability")
            if not topic.lower().startswith(label.lower()):
                topic = f"{label}: {topic}" if topic else label
            claim_type = str(item.get("claim_type") or "").strip().lower()
            if claim_type not in _CLAIM_TYPES:
                claim_type = "geographic_service_availability" if axis == "geography" else "other"
            relevance = str(item.get("relevance_to_original_prompt") or item.get("relevance") or "medium").strip().lower()
            if relevance not in _RELEVANCE:
                relevance = "medium"
            topic_relevance = str(item.get("relevance_to_topic") or "medium").strip().lower()
            if topic_relevance not in _RELEVANCE:
                topic_relevance = "medium"
            key = claim.lower()
            if not claim or key in seen or _is_superlative(claim):
                continue
            seen.add(key)
            questions.append({
                "topic": topic,
                "claim": claim,
                "claim_type": claim_type,
                "relevance": relevance,
                "relevance_to_topic": topic_relevance,
                "checked_from": "prompt",
            })
        if not questions:
            raise RuntimeError("Agent 3 did not produce any capability questions.")
        return questions

    def _check_question(
        self,
        fact_id: str,
        question: dict[str, str],
        client_name: str,
        client_host: str,
    ) -> CheckedClaim:
        topic = question["topic"]
        claim = question["claim"]
        first_pages, first_failed = self._collect_pages(claim, client_host, client_name, first_party=True)
        checked = self._read_pages(claim, client_name, first_pages) if first_pages else []
        usable = [item for item in checked if item["support"] != "none"]
        retrieved = [page["url"] for page in first_pages]
        search_failed = first_failed and not first_pages
        if not usable:
            web_pages, web_failed = self._collect_pages(claim, client_host, client_name, first_party=False)
            retrieved.extend(page["url"] for page in web_pages)
            if web_pages:
                checked.extend(self._read_pages(claim, client_name, web_pages))
                usable = [item for item in checked if item["support"] != "none"]
            search_failed = search_failed and web_failed and not web_pages and not first_pages
        for hit in usable:
            hit["client_host"] = client_host
        return _apply(
            fact_id,
            topic,
            claim,
            usable,
            search_failed,
            question["claim_type"],
            question["relevance"],
            str(question.get("relevance_to_topic") or "medium"),
            str(question.get("checked_from") or "prompt"),
            retrieved,
        )

    def _opportunity_claims(self, context: PromptContext) -> list[dict[str, str]]:
        """Turn every Agent 1 content opportunity into a claim that will be searched."""
        opportunities = context.content_opportunities
        if not opportunities:
            return []
        system = """Turn each content opportunity into one customer claim to search for, and score where it belongs.
Return one JSON object:
{"items":[{"opportunity":"","topic":"","claim":"","claim_type":"product|platform_capability|service|certification|geographic_service_availability|other","relevance_to_original_prompt":"high|medium|low","relevance_to_topic":"high|medium|low"}]}
The original prompt is what the user typed. prompt_understanding.topic is the topic they are searching for. Score those separately.
- high relevance_to_original_prompt: the opportunity helps answer the original prompt or its key questions.
- high relevance_to_topic: the opportunity is about that topic, even when the original prompt does not name it.
- low for both: speculation the prompt and the topic do not ask for, such as acquirers or partnerships.
Return one item for every opportunity. Copy the opportunity text exactly.
The claim is a concrete statement about the named customer that a page can support. No superlatives.
JSON only."""
        payload = {
            "original_prompt": context.original_prompt,
            "customer_name": context.customer.company_name,
            "prompt_understanding": context.prompt_understanding.model_dump(),
            "content_opportunities": opportunities,
        }
        try:
            data = self._complete(system, payload, max_tokens=2500)
        except Exception as exc:
            print(f"  Opportunity claims failed: {exc}")
            data = {}
        by_text: dict[str, dict[str, Any]] = {}
        for item in data.get("items") or []:
            if isinstance(item, dict):
                by_text[_clean(str(item.get("opportunity") or "")).lower()] = item
        claims: list[dict[str, str]] = []
        for opportunity in opportunities:
            item = by_text.get(opportunity.lower(), {})
            claim = _clean(str(item.get("claim") or "")) or opportunity
            if _is_superlative(claim):
                claim = opportunity
            topic = _clean(str(item.get("topic") or "")) or "Content opportunity"
            claim_type = str(item.get("claim_type") or "other").strip().lower()
            if claim_type not in _CLAIM_TYPES:
                claim_type = "other"
            relevance = str(item.get("relevance_to_original_prompt") or "medium").strip().lower()
            topic_relevance = str(item.get("relevance_to_topic") or "medium").strip().lower()
            if relevance not in _RELEVANCE:
                relevance = "medium"
            if topic_relevance not in _RELEVANCE:
                topic_relevance = "medium"
            claims.append({
                "topic": topic,
                "claim": claim,
                "claim_type": claim_type,
                "relevance": relevance,
                "relevance_to_topic": topic_relevance,
                "checked_from": "content_opportunity",
                "opportunity": opportunity,
            })
        return claims

    def _collect_pages(
        self,
        claim: str,
        client_host: str,
        client_name: str,
        first_party: bool,
    ) -> tuple[list[dict[str, Any]], bool]:
        query = claim if first_party else f"{client_name} {claim}"
        response = self.research.search(
            query,
            max_results=PAGES_PER_PASS,
            include_domains=[client_host] if first_party else None,
            exclude_domains=None if first_party else [client_host],
        )
        if response.failed:
            return [], True
        pages: list[dict[str, Any]] = []
        for result in response.results:
            if first_party and _host(result.url) != client_host:
                continue
            if not first_party and _host(result.url) == client_host:
                continue
            page = self.research.extract(result.url, max_chars=PAGE_CHARS)
            if page is None or not page.text.strip():
                continue
            pages.append({"url": page.url, "text": page.text, "published": _parse_date(result.published)})
            if len(pages) >= PAGES_PER_PASS:
                break
        return pages, False

    def _read_pages(self, claim: str, client_name: str, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        packed = [
            {"url": page["url"], "text": page["text"][:PAGE_CHARS]}
            for page in pages
        ]
        system = """You check one capability claim against supplied page text.
Return one JSON object:
{"pages":[{"url":"","support":"full|partial|contradicts|none","quote":"","narrow_statement":"","wording_limit":""}]}
Rules:
- Use only the supplied page text. Do not use outside knowledge.
- quote must be an exact contiguous span copied from that page, at least 30 characters.
- full means the quote supports the whole claim and does not require extra wording.
- partial means the quote supports a narrower fact. narrow_statement is that narrower fact, copied in meaning from the quote.
- Do not turn a product or kit into a testing service.
- Do not turn a customer or partner installation into the customer's own location or faster service.
- Do not add comparison, superiority, or cost unless the quote itself makes that comparison.
- contradicts means the quote says the claim is not true.
- none means the page does not settle the claim. Leave quote empty for none.
- wording_limit names the broader claim that must not be written.
- JSON only."""
        try:
            data = self._complete(
                system,
                {"client_name": client_name, "claim": claim, "pages": packed},
                max_tokens=1200,
            )
        except Exception as exc:
            print(f"  Evidence read failed: {exc}")
            return []
        by_url = {page["url"]: page for page in pages}
        checked: list[dict[str, Any]] = []
        for item in data.get("pages") or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            page = by_url.get(url)
            if page is None:
                continue
            support = str(item.get("support") or "none")
            quote = str(item.get("quote") or "")
            if support not in {"full", "partial", "contradicts"} or not _quote_in_page(quote, page["text"]):
                continue
            checked.append({
                "support": support,
                "quote": _clean(quote),
                "url": url,
                "published": page["published"],
                "narrow_statement": _clean(str(item.get("narrow_statement") or "")),
                "wording_limit": _clean(str(item.get("wording_limit") or "")),
            })
        return checked

    def _complete(self, system: str, payload: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            },
            timeout=90.0,
        )
        response.raise_for_status()
        blocks = response.json().get("content") or []
        text = "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        if not text.strip():
            raise RuntimeError("Claude returned no text for Agent 3.")
        return _extract_json(text)


def _apply(
    fact_id: str,
    topic: str,
    claim: str,
    hits: list[dict[str, Any]],
    search_failed: bool,
    claim_type: ClaimType,
    relevance: Relevance,
    topic_relevance: Relevance,
    checked_from: CheckedFrom,
    retrieved_urls: list[str] | None = None,
) -> CheckedClaim:
    """Apply the write rules in code. Quotes that are not in the page never arrive here."""

    def is_first(hit: dict[str, Any]) -> bool:
        host = hit.get("client_host")
        return bool(host) and _host(hit["url"]) == host

    full_first = [hit for hit in hits if hit["support"] == "full" and is_first(hit)]
    partial_first = [hit for hit in hits if hit["support"] == "partial" and is_first(hit)]
    full_other = [hit for hit in hits if hit["support"] == "full" and not is_first(hit)]
    partial_other = [hit for hit in hits if hit["support"] == "partial" and not is_first(hit)]
    contradictions = [hit for hit in hits if hit["support"] == "contradicts"]
    extra_bans: list[str] = []

    if _is_superlative(claim) and not full_first:
        return _unresolved(
            fact_id, topic, claim, claim_type, relevance, topic_relevance, checked_from,
            "do_not_write", "inconclusive",
            "Superlatives are not allowed without a first-party quote.",
            contradictions, extra_bans + [claim], gap=False, asked_claim=claim,
        )

    if full_first:
        hit = full_first[0]
        if _safe_narrow(claim, hit["quote"], claim):
            return _supported(
                fact_id, topic, claim, claim_type, relevance, topic_relevance, checked_from,
                "verified", "high", "write", hit, None, claim, None, claim,
            )
        full_first = []
        partial_first = [hit, *partial_first]
    if full_other:
        hit = full_other[0]
        if _safe_narrow(claim, hit["quote"], claim):
            return _supported(
                fact_id, topic, claim, claim_type, relevance, topic_relevance, checked_from,
                "verified", "medium", "write_limited", hit, None, claim,
                "Only a third-party page supports this.", claim,
            )
    if partial_first or partial_other or (full_other and not _safe_narrow(claim, full_other[0]["quote"], claim)):
        hit = (partial_first or partial_other or full_other)[0]
        narrow = _safe_narrow(hit["narrow_statement"], hit["quote"], claim)
        if narrow is None:
            narrow = _safe_narrow(hit["quote"], hit["quote"], claim)
        mismatch = _mismatch_reason(claim, hit["quote"])
        if narrow and narrow.lower() != claim.lower():
            return _supported(
                fact_id, topic, narrow, claim_type, relevance, topic_relevance, checked_from,
                "verified", "high" if is_first(hit) else "medium", "write_limited", hit, None, narrow,
                mismatch or "State only what the quote supports.", claim,
            )
        if mismatch:
            return _unresolved(
                fact_id, topic, claim, claim_type, relevance, topic_relevance, checked_from,
                "do_not_write", "inconclusive", mismatch, [hit], extra_bans, gap=False, asked_claim=claim,
            )
        return _unresolved(
            fact_id, topic, claim, claim_type, relevance, topic_relevance, checked_from,
            "research_more", "inconclusive",
            "The evidence reviewed does not establish this claim. Further research is required.",
            [], extra_bans, gap=True, asked_claim=claim,
        )
    if contradictions:
        extra_bans.append(claim)
        return _unresolved(
            fact_id, topic, claim, claim_type, relevance, topic_relevance, checked_from,
            "do_not_write", "contradicted",
            "A retrieved page says otherwise.",
            contradictions, extra_bans, gap=False, asked_claim=claim,
        )
    reason = "The evidence reviewed does not establish this claim. Further research is required."
    if search_failed:
        reason = "The search did not return a page."
    elif retrieved_urls:
        reason = f"A page was retrieved, but it did not contain a quote that establishes this claim. {retrieved_urls[0]}"
    return _unresolved(
        fact_id, topic, claim, claim_type, relevance, topic_relevance, checked_from,
        "research_more", "inconclusive", reason, [], extra_bans, gap=True, asked_claim=claim,
    )


def _evidence(hit: dict[str, Any]) -> Evidence:
    return Evidence(
        url=hit["url"],
        quote=hit["quote"],
        source_tier=_tier(hit["url"], str(hit.get("client_host") or "")),
        published=hit.get("published"),
    )


def _supported(
    fact_id: str,
    topic: str,
    claim: str,
    claim_type: ClaimType,
    relevance: Relevance,
    topic_relevance: Relevance,
    checked_from: CheckedFrom,
    status: FactStatus,
    confidence: Confidence,
    decision: TopicDecision,
    hit: dict[str, Any],
    reason: str | None,
    statement: str,
    wording_limit: str | None,
    asked_claim: str,
) -> CheckedClaim:
    fact = Fact(
        fact_id=fact_id,
        claim=claim,
        status=status,
        claim_type=claim_type,
        evidence=[_evidence(hit)],
        confidence=confidence,
        relevance_to_original_prompt=relevance,
        relevance_to_topic=topic_relevance,
        checked_from=checked_from,
        reason=None,
    )
    verdict = TopicVerdict(topic=topic, decision=decision, fact_ids=[fact_id], reason=None)
    blocked = None
    if asked_claim.lower() != claim.lower():
        blocked = DoNotWrite(
            claim_or_topic=asked_claim,
            reason=wording_limit or "The quote supports a narrower statement and does not establish this broader claim.",
        )
    return CheckedClaim(fact, verdict, blocked, [], None, statement, wording_limit, asked_claim)


def _unresolved(
    fact_id: str,
    topic: str,
    claim: str,
    claim_type: ClaimType,
    relevance: Relevance,
    topic_relevance: Relevance,
    checked_from: CheckedFrom,
    decision: TopicDecision,
    status: FactStatus,
    reason: str,
    hits: list[dict[str, Any]],
    extra_bans: list[str],
    gap: bool,
    asked_claim: str | None = None,
) -> CheckedClaim:
    asked = asked_claim or claim
    fact = Fact(
        fact_id=fact_id,
        claim=claim,
        status=status,
        claim_type=claim_type,
        evidence=[_evidence(hit) for hit in hits if hit.get("quote")],
        confidence="low",
        relevance_to_original_prompt=relevance,
        relevance_to_topic=topic_relevance,
        checked_from=checked_from,
        reason=reason,
    )
    verdict = TopicVerdict(topic=topic, decision=decision, fact_ids=[fact_id], reason=reason)
    research = UnresolvedClaim(claim=asked, reason=reason) if gap else None
    blocked = DoNotWrite(claim_or_topic=asked, reason=reason) if decision == "do_not_write" else None
    return CheckedClaim(
        fact, verdict, blocked, extra_bans, research, None, None, asked,
    )


def _target_label(prompt_relevance: str, topic_relevance: str) -> str:
    on_prompt = prompt_relevance in {"high", "medium"}
    on_topic = topic_relevance in {"high", "medium"}
    if on_prompt and on_topic:
        return "Matches the original prompt and the topic being searched."
    if on_prompt:
        return "Matches the original prompt. It is outside the topic being searched."
    if on_topic:
        return "Matches the topic being searched. It is outside the original prompt."
    return "Outside the original prompt and the topic being searched."


def _apply_target(question: dict[str, str], checked: CheckedClaim) -> CheckedClaim:
    """Keep an off-target claim out of the writing list without erasing its evidence."""
    prompt_relevance = str(question.get("relevance") or "medium")
    topic_relevance = str(question.get("relevance_to_topic") or "medium")
    if prompt_relevance in {"high", "medium"} or topic_relevance in {"high", "medium"}:
        return checked
    reason = _target_label(prompt_relevance, topic_relevance)
    fact = checked.fact.model_copy(update={"reason": reason})
    verdict = checked.verdict.model_copy(update={"decision": "do_not_write", "reason": reason})
    return CheckedClaim(
        fact, verdict, DoNotWrite(claim_or_topic=fact.claim, reason=reason), checked.extra_bans, None, None, None, checked.asked_claim,
    )


def _unique_blocks(items: list[DoNotWrite]) -> list[DoNotWrite]:
    kept: list[DoNotWrite] = []
    seen: set[str] = set()
    for item in items:
        key = item.claim_or_topic.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


def _unique_unresolved(items: list[UnresolvedClaim]) -> list[UnresolvedClaim]:
    kept: list[UnresolvedClaim] = []
    seen: set[str] = set()
    for item in items:
        key = item.claim.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


def _norm(text: str) -> str:
    words = [word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 2 and word not in _STOP]
    return " ".join(words)


def _same_claim(left: str, right: str) -> bool:
    a, b = _norm(left), _norm(right)
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return shorter in longer and len(shorter) / len(longer) >= 0.7


def _safe_narrow(statement: str, quote: str, claim: str) -> str | None:
    statement = _clean(statement)
    if not statement or _is_superlative(statement):
        return None
    if _mismatch_reason(statement, quote):
        return None
    if _mismatch_reason(claim, quote) and _same_claim(statement, claim):
        return None
    words = [word for word in re.findall(r"[a-z0-9]+", statement.lower()) if len(word) > 3 and word not in _STOP]
    if not words:
        return None
    quote_l = quote.lower()
    if sum(word in quote_l for word in words) / len(words) < 0.55:
        return None
    return statement


def _mismatch_reason(claim: str, quote: str) -> str | None:
    claim_l, quote_l = claim.lower(), quote.lower()
    if "service" in claim_l and "service provider for" in quote_l:
        return "The quote describes a partner or customer delivering services, not the customer providing that service."
    if any(marker in claim_l for marker in _COMPARE) and not any(marker in quote_l for marker in _COMPARE):
        return "The quote does not support a comparison, superiority claim, or cost claim."
    if any(marker in claim_l for marker in ("north america", "location", "turnaround", "faster")) and (
        "under 8 hours" in quote_l or "under 15 hours" in quote_l or "service provider for" in quote_l
    ):
        return "The quote describes instrument runtime or a customer installation, not the company's own locations or faster service."
    named = [item for item in ("clia", "gclp", "cap") if item in claim_l]
    if named and not any(item in quote_l for item in named):
        if "iso" in quote_l:
            return "The quote supports a different certification from the one claimed."
        return "The quote does not establish the named certification."
    return None


def _scope_note(context: PromptContext) -> ScopeNote:
    prompt = context.original_prompt
    topic = context.prompt_understanding.topic.strip() or None
    note = None
    if topic and "north america" in topic.lower() and "north america" not in prompt.lower():
        note = "North America is answer-derived from the stored AI answers. It is not part of the original prompt."
    elif topic and topic.lower() not in prompt.lower():
        note = "The topic adds wording that is not in the original prompt. The original prompt is unchanged."
    return ScopeNote(original_prompt=prompt, inferred_topic=topic, scope_note=note)


def _collapse(
    group: list[tuple[dict[str, str], CheckedClaim]],
) -> tuple[CheckedClaim, list[dict[str, str]]]:
    verified = [pair for pair in group if pair[1].fact.status == "verified"]
    _, checked = verified[0] if verified else group[0]
    evidence = list(checked.fact.evidence)
    seen = {(item.url, item.quote) for item in evidence}
    blocks = [checked.block] if checked.block else []
    gaps = [checked.gap] if checked.gap else []
    from_prompt = False
    opportunities: list[dict[str, str]] = []
    for question, other in group:
        from_prompt = from_prompt or question.get("checked_from") == "prompt" or other.fact.checked_from == "prompt"
        if question.get("checked_from") == "content_opportunity":
            opportunities.append(question)
        if other.fact.fact_id == checked.fact.fact_id:
            continue
        if other.block:
            blocks.append(other.block)
        if other.gap:
            gaps.append(other.gap)
        for item in other.fact.evidence:
            key = (item.url, item.quote)
            if key in seen:
                continue
            if checked.fact.status == "verified" and not _safe_narrow(checked.fact.claim, item.quote, checked.asked_claim):
                continue
            seen.add(key)
            evidence.append(item)
    fact = checked.fact.model_copy(update={
        "evidence": evidence,
        "checked_from": "prompt" if from_prompt else "content_opportunity",
    })
    block = next((item for item in blocks if item), None)
    gap = None if fact.status == "verified" else next((item for item in gaps if item), None)
    return checked._replace(fact=fact, block=block, gap=gap), opportunities


def _merge_pairs(
    pairs: list[tuple[dict[str, str], CheckedClaim]],
) -> list[tuple[CheckedClaim, list[dict[str, str]]]]:
    groups: list[list[tuple[dict[str, str], CheckedClaim]]] = []
    for pair in pairs:
        claim = pair[1].fact.claim
        asked = pair[1].asked_claim
        for group in groups:
            current = group[0][1]
            if _same_claim(claim, current.fact.claim) or _same_claim(asked, current.asked_claim):
                group.append(pair)
                break
        else:
            groups.append([pair])
    numbered: list[tuple[CheckedClaim, list[dict[str, str]]]] = []
    for index, group in enumerate(groups, start=1):
        checked, opportunities = _collapse(group)
        fact_id = f"fact_{index:03d}"
        fact = checked.fact.model_copy(update={"fact_id": fact_id})
        verdict = checked.verdict.model_copy(update={"fact_ids": [fact_id]})
        numbered.append((checked._replace(fact=fact, verdict=verdict), opportunities))
    return numbered


def _assemble(
    context: PromptContext,
    pairs: list[tuple[dict[str, str], CheckedClaim]],
    banned: list[str],
) -> GroundTruthGuide:
    prompt = context.original_prompt.lower()
    adjusted: list[tuple[dict[str, str], CheckedClaim]] = []
    for question, checked in pairs:
        fact = checked.fact
        if "north america" in fact.claim.lower() and "north america" not in prompt:
            fact = fact.model_copy(update={"relevance_to_original_prompt": "low"})
            checked = checked._replace(fact=fact)
        adjusted.append((question, checked))
    merged = _merge_pairs(adjusted)
    can_write: list[Fact] = []
    limited: list[LimitedStatement] = []
    blocked: list[DoNotWrite] = []
    for checked, _opportunity_questions in merged:
        fact = checked.fact
        on_target = fact.relevance_to_original_prompt in {"high", "medium"} or fact.relevance_to_topic in {"high", "medium"}
        narrowed = checked.asked_claim.lower() != fact.claim.lower()
        if fact.status == "verified" and on_target and checked.verdict.decision == "write" and not narrowed:
            can_write.append(fact)
            continue
        if fact.status == "verified" and on_target and checked.verdict.decision in {"write", "write_limited"}:
            limited.append(LimitedStatement(
                fact_id=fact.fact_id,
                statement=fact.claim,
                wording_limit=checked.wording_limit or (checked.block.reason if checked.block else "State only what the quote supports."),
                evidence=fact.evidence,
                confidence=fact.confidence,
                relevance_to_original_prompt=fact.relevance_to_original_prompt,
                relevance_to_topic=fact.relevance_to_topic,
            ))
            if narrowed:
                blocked.append(DoNotWrite(
                    claim_or_topic=checked.asked_claim,
                    reason=checked.wording_limit or "The quote supports a narrower statement.",
                ))
            continue
        reason = checked.verdict.reason or fact.reason or "The evidence reviewed does not establish this claim."
        blocked.append(DoNotWrite(claim_or_topic=checked.asked_claim, reason=reason))
    write_numbered = [
        fact.model_copy(update={"fact_id": f"fact_{index:03d}"})
        for index, fact in enumerate(can_write, start=1)
    ]
    limit_numbered = [
        item.model_copy(update={"fact_id": f"limit_{index:03d}"})
        for index, item in enumerate(limited, start=1)
    ]
    return GroundTruthGuide(
        scope=_scope_note(context),
        can_write=write_numbered,
        can_write_with_limits=limit_numbered,
        cannot_write=_unique_blocks(blocked),
        banned_claims=banned,
        limitations=list(_LIMITATIONS),
    )


