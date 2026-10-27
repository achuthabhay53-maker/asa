"""Agent 3: Ground truth for AImpact.

Starts from the original prompt and Agent 1's reading of it. The website
is supplied by the user. Prompt claims and each content opportunity are
searched, then marked as matching the original prompt, the topic being
searched, both, or neither.

The model's own knowledge is not evidence.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime
from typing import Any, NamedTuple
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from aca.evidence import canonicalize_evidence_url
from aca.models.ground_truth import (
    Agent4HandoffPackage,
    CandidateSubtopic,
    CheckedFrom,
    ClaimType,
    ClientInput,
    CompanyInfo,
    CompanyTalkingPoint,
    Confidence,
    DoNotWrite,
    Evidence,
    EvidenceBackedTopic,
    EvidenceItem,
    Fact,
    FactStatus,
    GroundTruthGuide,
    LimitedStatement,
    PromptContext,
    PromptUnderstandingInput,
    PromptVerification,
    Relevance,
    ResearchGap,
    ScopeNote,
    SourceTier,
    TopicDecision,
    TopicVerdict,
    UnresolvedClaim,
    VerificationHandoff,
    VerifiedFact,
)
from aca.tools.web_research import WebResearch, get_web_research

load_dotenv()

# How many capability questions to test. Override with GroundTruthAgent(max_questions=...).
MAX_QUESTIONS = 6
PAGES_PER_PASS = 2
PAGE_CHARS = 12_000
logger = logging.getLogger(__name__)

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
    "a",
    "an",
    "the",
    "that",
    "with",
    "from",
    "this",
    "have",
    "has",
    "been",
    "were",
    "will",
    "into",
    "about",
    "their",
    "which",
    "than",
    "such",
    "also",
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


class _EvidenceRegistry:
    def __init__(self, client_host: str) -> None:
        self.client_host = client_host
        self.items: list[EvidenceItem] = []
        self._ids_by_quote_and_url: dict[tuple[str, str], str] = {}

    def register_quote(
        self,
        quote: str,
        pages: list[dict[str, Any]],
        preferred_source_type: str | None = None,
    ) -> str | None:
        cleaned_quote = _clean(quote)
        if len(cleaned_quote) < 25:
            return None
        for page in pages:
            url = str(page.get("url") or "")
            canonical_url = canonicalize_evidence_url(url)
            if (
                not canonical_url
                or not _is_customer_url(url, self.client_host)
                or not _quote_in_page(cleaned_quote, str(page.get("text") or ""))
            ):
                continue
            key = (cleaned_quote.casefold(), canonical_url)
            if key in self._ids_by_quote_and_url:
                return self._ids_by_quote_and_url[key]
            evidence_id = f"ev_{len(self.items) + 1:03d}"
            self.items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    url=url,
                    quote=cleaned_quote,
                    source_type="company_website",
                    domain=_host(url),
                    page_title=page.get("title"),
                    published=page.get("published"),
                )
            )
            self._ids_by_quote_and_url[key] = evidence_id
            return evidence_id
        return None

    def register_checked_evidence(
        self,
        evidence: Evidence,
        preferred_source_type: str | None = None,
    ) -> str | None:
        """Register a quote already validated against its exact source page."""
        quote = _clean(evidence.quote)
        url = evidence.url.strip()
        canonical_url = canonicalize_evidence_url(url)
        if len(quote) < 25 or not canonical_url or not _is_customer_url(url, self.client_host):
            return None
        key = (quote.casefold(), canonical_url)
        existing = self._ids_by_quote_and_url.get(key)
        if existing:
            return existing
        evidence_id = f"ev_{len(self.items) + 1:03d}"
        self.items.append(
            EvidenceItem(
                evidence_id=evidence_id,
                url=url,
                quote=quote,
                source_type="company_website",
                domain=_host(url),
                published=evidence.published,
            )
        )
        self._ids_by_quote_and_url[key] = evidence_id
        return evidence_id


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _host(url: str) -> str:
    host = (urlparse(url if "://" in url else f"https://{url}").hostname or "").lower()
    return host.removeprefix("www.")


def _is_customer_url(url: str, client_host: str) -> bool:
    host = _host(url)
    return bool(host) and (host == client_host or host.endswith(f".{client_host}"))


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
    # Strip markdown fences
    text = re.sub(r"^\s*```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)

    # Direct parse
    try:
        data = json.loads(text, strict=False)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Slice from first { to last }
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            data = json.loads(candidate, strict=False)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        # Try json_repair on the sliced candidate
        try:
            import json_repair  # type: ignore[import]

            data = json_repair.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Last resort: json_repair on the full text
    try:
        import json_repair  # type: ignore[import]

        data = json_repair.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    preview = text[:300].replace("\n", " ")
    raise ValueError(f"Claude did not return a valid JSON object. Preview: {preview!r}")


def _quote_in_page(quote: str, page: str) -> bool:
    quote_n = _clean(quote)
    page_n = _clean(page)
    return len(quote_n) >= 30 and quote_n in page_n


def _is_superlative(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered) for term in _SUPERLATIVES)


def _parse_date(value: Any) -> date | None:
    if not value or not isinstance(value, str):
        return None
    raw = value.strip()[:10]
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


def _tier(url: str, client_host: str) -> SourceTier:
    return "first_party" if _is_customer_url(url, client_host) else "other"


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
        comparison_criteria=_string_list(
            (first.get("answer_understanding") or {}).get("comparison_criteria")
        ),
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
        self._warnings: list[str] = []

    def analyze(
        self,
        agent1: Any,
        client: ClientInput,
    ) -> GroundTruthGuide:
        self._warnings = []
        context = build_prompt_context(agent1, client.website, client.company_name)
        client_name = context.customer.company_name
        client_host = _host(context.customer.website)
        if not client_host:
            raise ValueError("The website must include a host, for example https://example.com.")

        # Step 1: Understand the original prompt & verify relevance
        understanding = self._understand_prompt_and_verify(context, client_name, client_host)
        understanding = self._validate_understanding(understanding)
        if not understanding["is_verified"]:
            return GroundTruthGuide(
                topic=understanding["main_topic"],
                main_topic=understanding["main_topic"],
                search_intent=understanding["search_intent"],
                target_audience=understanding["target_audience"],
                prompt_scope=understanding["prompt_scope"],
                verification=PromptVerification(
                    is_verified=False,
                    intent_valid=understanding["intent_valid"],
                    customer_relevant=understanding["customer_relevant"],
                    human_review_required=understanding["human_review_required"],
                    reason=understanding["reason"],
                ),
                boundaries=[
                    "The request did not pass prompt/customer relevance verification; "
                    "no content research or writing was performed."
                ]
                + list(dict.fromkeys(self._warnings)),
            )

        # Step 2: Research the specific topic (Company website vs. Reputable external sources)
        research_pages, checked_pairs, banned = self._research_topic(
            context, understanding, client_name, client_host
        )

        # Step 3: Discover candidate subtopics & organize evidence
        guide = self._discover_subtopics_and_evidence(
            context, understanding, research_pages, checked_pairs, banned, client_name, client_host
        )

        # Step 4: Return structured output ready for Agent 4
        return guide

    def _add_warning(self, warning: str) -> None:
        if warning not in self._warnings:
            self._warnings.append(warning)

    @staticmethod
    def _validate_understanding(data: dict[str, Any]) -> dict[str, Any]:
        """Require explicit verification flags and valid types from Agent 3's model."""
        if not isinstance(data, dict):
            raise ValueError("Agent 3 verification output must be a JSON object.")
        flags = ("is_verified", "intent_valid", "customer_relevant")
        for key in flags:
            if type(data.get(key)) is not bool:
                raise ValueError(f"Agent 3 verification output requires boolean field {key!r}.")
        expected_verified = data["intent_valid"] and data["customer_relevant"]
        if data["is_verified"] != expected_verified:
            raise ValueError(
                "Agent 3 verification output is inconsistent: is_verified must "
                "equal intent_valid and customer_relevant."
            )

        for key in ("main_topic", "search_intent", "target_audience", "reason"):
            if not isinstance(data.get(key), str):
                raise ValueError(f"Agent 3 verification output requires string field {key!r}.")
        if not data["main_topic"].strip():
            raise ValueError("Agent 3 verification output has an empty main_topic.")
        if not data["search_intent"].strip() or not data["target_audience"].strip():
            raise ValueError("Agent 3 verification output requires search_intent and target_audience.")
        if data.get("prompt_scope") not in {
            "broad_industry",
            "company_specific",
            "hybrid",
        }:
            raise ValueError("Agent 3 verification output has an invalid prompt_scope.")
        if not isinstance(data.get("quote"), str) or not isinstance(data.get("source_url"), str):
            raise ValueError("Agent 3 verification output requires string quote and source_url fields.")
        data["human_review_required"] = False
        if data["customer_relevant"]:
            source_url = canonicalize_evidence_url(data["source_url"])
            quote = data["quote"]
            pages = data.get("retrieved_pages")
            supported_by_retrieved_page = isinstance(pages, list) and any(
                isinstance(page, dict)
                and canonicalize_evidence_url(str(page.get("url") or "")) == source_url
                and _quote_in_page(quote, str(page.get("text") or ""))
                for page in pages
            )
            if not source_url or not supported_by_retrieved_page:
                data = data.copy()
                data["human_review_required"] = True
                data["reason"] = (
                    data["reason"].strip() + " Human review is required: customer relevance did not have "
                    "an exact matching quote from the retrieved source page."
                ).strip()
        return data

    def _understand_prompt_and_verify(
        self,
        context: PromptContext,
        client_name: str,
        client_host: str,
    ) -> dict[str, Any]:
        """Step 1: Understand prompt, infer main topic, intent, audience, scope, and verify relevance."""
        query_fp = f"{context.original_prompt} {context.prompt_understanding.intent}".strip()
        first_pages, search_failed = self._collect_optional_pages(
            query_fp, client_host, client_name, first_party=True
        )
        retrieved_pages = [
            page for page in first_pages if _is_customer_url(str(page.get("url") or ""), client_host)
        ]

        if not retrieved_pages:
            try:
                root_page = self.research.extract(f"https://{client_host}")
                if root_page and root_page.text.strip() and _is_customer_url(root_page.url, client_host):
                    retrieved_pages.append({"url": root_page.url, "text": root_page.text, "published": None})
            except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
                logger.warning(
                    "Agent 3 root-page fallback failed (%s).",
                    type(exc).__name__,
                )
                self._add_warning("Company root-page fallback failed during prompt verification.")

        if not retrieved_pages and search_failed:
            reason = (
                "Customer relevance could not be assessed because web search "
                "returned no usable pages and root-page retrieval failed. This is not evidence that the "
                "customer is irrelevant; rerun research or review manually."
            )
            self._add_warning(reason)
            return {
                "main_topic": context.prompt_understanding.topic or context.original_prompt,
                "search_intent": context.prompt_understanding.intent,
                "target_audience": context.prompt_understanding.audience or "Industry professionals",
                "prompt_scope": "hybrid",
                "is_verified": False,
                "intent_valid": True,
                "customer_relevant": False,
                "human_review_required": True,
                "reason": reason,
                "quote": "",
                "source_url": "",
                "retrieved_pages": [],
            }

        packed = [{"url": page["url"], "text": page["text"][:PAGE_CHARS]} for page in retrieved_pages]

        system = """You are Agent 3 in a content intelligence workflow.
Step 1: Understand the user prompt and verify customer relevance.
Return one JSON object:
{
  "main_topic": "The central subject inferred from the prompt",
  "search_intent": "What the intended reader wants to know or solve",
  "target_audience": "Who the blog is for (e.g. researchers, clinicians, buyers, lab directors)",
  "prompt_scope": "broad_industry | company_specific | hybrid",
  "is_verified": true|false,
  "intent_valid": true|false,
  "customer_relevant": true|false,
  "reason": "Detailed explanation of why the customer is or is not relevant, and whether the intent is valid.",
  "quote": "Exact contiguous quote from retrieved text if available, or empty string",
  "source_url": "URL of the page containing the quote, or empty string"
}
Rules:
- main_topic: Identify the central subject from the user's prompt (e.g., 'Immunology biomarker assay providers and platform capabilities').
- prompt_scope: Distinguish whether the original prompt describes a broad industry topic, a company-specific capability, or a hybrid. Preserve that distinction rather than forcing every topic to promote the company.
- Keep these decisions independent:
  * intent_valid is true when the user's question is meaningful and answerable, regardless of whether this customer offers what it asks for.
  * customer_relevant is true only when retrieved evidence supports that this customer offers, develops, or operates in the requested space. A valid question about a service the customer does not offer has intent_valid: true and customer_relevant: false.
  * is_verified is true only when both intent_valid and customer_relevant are true.
- Check functional capabilities, not just exact wording. Consider clear equivalents and alternate terms (for example, virtual reality / VR, immersive training, XR, and simulation for VR training), but do not infer a capability from a broad industry association alone.
- Do not mark the customer irrelevant only because the exact prompt wording is absent from its pages or search results. Use relevant capability pages and equivalent terminology; if evidence is insufficient, say that in reason rather than claiming the question itself is invalid.
- A valid prompt about an unrelated customer must have intent_valid: true, customer_relevant: false, and is_verified: false.
- Verify customer-specific relevance only from the customer's official website pages supplied below.
- Base your judgment strictly on those pages. Do NOT use third-party sources, outside knowledge, or assumptions.
- quote must be an exact contiguous quote from the supplied page text (at least 30 characters) if one is found.
- JSON only."""

        payload = {
            "customer_name": client_name,
            "website": f"https://{client_host}",
            "original_prompt": context.original_prompt,
            "prompt_understanding": context.prompt_understanding.model_dump(),
            "retrieved_pages": packed,
        }

        data = self._complete(system, payload, max_tokens=1500)
        data = dict(data)
        data["retrieved_pages"] = retrieved_pages
        return data

    def _research_topic(
        self,
        context: PromptContext,
        understanding: dict[str, Any],
        client_name: str,
        client_host: str,
    ) -> tuple[list[dict[str, Any]], list[tuple[dict[str, str], CheckedClaim]], list[str]]:
        """Step 2: Research customer-specific claims on the official website only."""
        all_pages: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        # Add any pages already collected in Step 1
        for p in understanding.get("retrieved_pages") or []:
            u = p.get("url")
            if u and _is_customer_url(u, client_host) and u not in seen_urls:
                seen_urls.add(u)
                all_pages.append(
                    {
                        "url": u,
                        "text": p.get("text", ""),
                        "published": p.get("published"),
                        "source_type": "company_website",
                    }
                )

        # 1. Company Website: Search products, assays, services, capabilities
        main_topic = understanding.get("main_topic") or context.original_prompt
        fp_query = f"{main_topic} products assays services technology".strip()
        fp_pages, _ = self._collect_optional_pages(fp_query, client_host, client_name, first_party=True)
        for p in fp_pages:
            u = p.get("url")
            if u and _is_customer_url(u, client_host) and u not in seen_urls:
                seen_urls.add(u)
                all_pages.append(
                    {
                        "url": u,
                        "text": p.get("text", ""),
                        "published": p.get("published"),
                        "source_type": "company_website",
                    }
                )

        # Check capability questions for exact factual grounding
        try:
            questions = self._questions(context)[: self.max_questions]
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Agent 3 capability-question generation failed (%s).",
                type(exc).__name__,
            )
            self._add_warning("Capability-question generation failed; those optional checks were skipped.")
            questions = []
        pairs: list[tuple[dict[str, str], CheckedClaim]] = []
        for opportunity in self._opportunity_claims(context):
            if opportunity.get("relevance") == "low" and opportunity.get("relevance_to_topic") == "low":
                pairs.append(
                    (
                        opportunity,
                        _unresolved(
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
                        ),
                    )
                )
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

        return all_pages, pairs, banned

    def _collect_optional_pages(
        self,
        claim: str,
        client_host: str,
        client_name: str,
        first_party: bool,
    ) -> tuple[list[dict[str, Any]], bool]:
        if not first_party:
            return [], False
        try:
            pages, failed = self._collect_pages(claim, client_host, client_name, first_party)
        except (httpx.HTTPError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning(
                "Agent 3 optional research retrieval failed (%s).",
                type(exc).__name__,
            )
            self._add_warning("Optional first-party research retrieval failed.")
            return [], True
        if failed:
            self._add_warning("Optional first-party search returned no usable results.")
        return [page for page in pages if _is_customer_url(str(page.get("url") or ""), client_host)], failed

    def _discover_subtopics_and_evidence(
        self,
        context: PromptContext,
        understanding: dict[str, Any],
        research_pages: list[dict[str, Any]],
        checked_pairs: list[tuple[dict[str, str], CheckedClaim]],
        banned: list[str],
        client_name: str,
        client_host: str,
    ) -> GroundTruthGuide:
        """Step 3: Discover candidate subtopics, verified facts, evidence, research gaps, limitations."""
        research_pages = [
            page for page in research_pages if _is_customer_url(str(page.get("url") or ""), client_host)
        ]
        checked_guide = _assemble(context, checked_pairs, banned)

        # Build prompt verification object
        v_quote = _clean(str(understanding.get("quote") or ""))
        v_url = str(understanding.get("source_url") or "")
        v_ev: list[Evidence] = []
        if v_quote and v_url:
            for p in research_pages:
                if canonicalize_evidence_url(p["url"]) == canonicalize_evidence_url(v_url) and _quote_in_page(
                    v_quote, p["text"]
                ):
                    v_ev.append(
                        Evidence(
                            url=v_url,
                            quote=v_quote,
                            source_tier=_tier(v_url, client_host),
                            published=p.get("published"),
                        )
                    )
                    break

        understanding = self._validate_understanding(understanding)
        verification = PromptVerification(
            is_verified=understanding["is_verified"],
            intent_valid=understanding["intent_valid"],
            customer_relevant=understanding["customer_relevant"],
            human_review_required=understanding["human_review_required"],
            reason=understanding["reason"],
            evidence=v_ev,
        )

        # Prepare evidence snippets from pages
        packed_evidence = [
            {
                "url": p["url"],
                "source_type": p.get("source_type", "company_website"),
                "text_snippet": p["text"][:PAGE_CHARS],
            }
            for p in research_pages
        ]

        # Prepare facts from checked claims
        known_facts = [
            {
                "statement": f.claim,
                "quote": f.evidence[0].quote if f.evidence else "",
                "source": f.evidence[0].url if f.evidence else "",
            }
            for f in checked_guide.can_write
        ] + [
            {
                "statement": limited_fact.statement,
                "limit": limited_fact.wording_limit,
                "quote": limited_fact.evidence[0].quote if limited_fact.evidence else "",
                "source": limited_fact.evidence[0].url if limited_fact.evidence else "",
            }
            for limited_fact in checked_guide.can_write_with_limits
        ]

        system = """You are Agent 3 in a content intelligence pipeline (Ground Truth & Research Discovery).
Your job is to discover what the company can write about and organize researched evidence so Agent 4 can decide the content strategy.

Return one JSON object with exactly these keys:
{
  "main_topic": "Central subject of the prompt",
  "search_intent": "What the intended reader wants to know",
  "target_audience": "Who the blog is for",
  "prompt_scope": "broad_industry | company_specific | hybrid",
  "candidate_subtopics": [
    {
      "title": "Subtopic Title",
      "description": "What this subtopic covers and why it is valuable to the reader",
      "topic_angle": "company_capability | industry_context | comparison_methodology | educational_guide",
      "relevance_to_main_topic": "high | medium | low",
      "evidence_status": "supported | needs_more_research",
      "research_needed": ["Specific claim that needs evidence before this subtopic can be written"],
      "suggested_questions": ["Key question 1", "Key question 2"],
      "evidence_quotes": ["Exact verbatim quote copied from the supplied research_evidence text"]
    }
  ],
  "verified_facts": [
    {
      "statement": "Concrete factual statement",
      "category": "company_capability | product_technology | regulatory_certification | service_scope",
      "evidence_type": "company_website",
      "confidence": "high | medium",
      "wording_boundary": "What the evidence does NOT support (or empty string if no boundary)",
      "evidence_quote": "Exact verbatim quote copied from the supplied research_evidence text"
    }
  ],
  "research_gaps": [
    {
      "topic_or_question": "Topic or question that is relevant to the prompt but lacks evidence",
      "missing_claim": "The specific assertion that could not be verified (e.g. 'Example Bio offers a 5-day service turnaround')",
      "reason": "Why the current evidence is inconclusive: what is present, what is absent",
      "suggested_search_direction": "Exact search query or page type that would resolve this gap",
      "affects_subtopic_ids": []
    }
  ],
  "limitations": [
    "A specific claim or wording that must be avoided or qualified"
  ]
}

Rules:
1. Prompt Focus:
   - The original_prompt is the anchor. Every candidate subtopic must directly help a reader evaluate the company for THAT prompt.
   - Topics about geography, regulatory compliance, dried blood spots, neurology applications, or technology comparisons are only valid if the original prompt specifically requests them. If not requested, tag them relevance_to_main_topic=low and include a note in description.
   - Do NOT expand the scope. Agent 4 decides whether to include optional topics.

2. Evidence Grounding for Facts:
   - Verify every customer-specific fact only from the supplied pages on the customer's official website.
   - Never use third-party sources or assumptions to fill an evidence gap.
   - The primary verification target is a claim specifically about the named company, its products, services, capabilities, processes, or results. Do not classify general industry context or educational explanations as company facts.
   - Do not create a research gap solely because a generic industry statement lacks evidence about the named company. Keep useful industry context separate from company-specific facts and gaps.
   - A verified_fact MUST include a non-empty evidence_quote that is verbatim from the supplied research_evidence.
   - If you cannot find an exact quote for a claim, do NOT include it as a verified_fact. Either omit it or add it as a research_gap.
   - wording_boundary must state what the quote does NOT support if the claim could be read more broadly than the quote.

3. Support Status Grading:
   - evidence_status='supported' requires at least one verbatim quote in evidence_quotes that directly supports the title and description.
   - If you can find a quote about a related product but not about the specific capability in the description, use evidence_status='needs_more_research'.
   - Do NOT mark a subtopic supported just because the company makes a related product.

4. Specific Research Gaps:
   - Every research_gap must identify: (a) the exact claim that is unverified in missing_claim, (b) what evidence exists and what is absent in reason, (c) a concrete search direction in suggested_search_direction.
   - Vague gaps like 'more research needed on platform capabilities' are not acceptable.

5. No Superlatives:
   - Never use superlatives like 'best', 'leading', '#1', 'world's first', or 'unmatched' unless verified by a first-party factual certification.

6. JSON only. No preamble, no explanation."""

        # Keep each page snippet small so the payload stays within Claude's
        # context window and leaves enough room for a full JSON response.
        _SNIPPET = 6_000
        trimmed_evidence = [
            {
                "url": p["url"],
                "source_type": p.get("source_type", "company_website"),
                "text_snippet": p["text_snippet"][:_SNIPPET]
                if isinstance(p.get("text_snippet"), str)
                else "",
            }
            for p in packed_evidence
        ]

        payload = {
            "customer_name": client_name,
            "website": f"https://{client_host}",
            "original_prompt": context.original_prompt,
            "understanding": {
                "main_topic": understanding.get("main_topic") or context.original_prompt,
                "search_intent": understanding.get("search_intent") or context.prompt_understanding.intent,
                "target_audience": understanding.get("target_audience")
                or context.prompt_understanding.audience,
                "prompt_scope": understanding.get("prompt_scope") or "hybrid",
            },
            "known_facts": known_facts,
            "research_evidence": trimmed_evidence,
        }

        try:
            # 8192 tokens gives enough room for a rich subtopic JSON response
            data = self._complete(system, payload, max_tokens=8192)
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Agent 3 subtopic discovery failed (%s); falling back to checked facts.",
                type(exc).__name__,
            )
            self._add_warning("Subtopic discovery failed; only independently checked claims are included.")
            data = {}

        # 1. Build EvidenceItem registry
        evidence_registry = _EvidenceRegistry(client_host)
        evidence_items = evidence_registry.items

        def add_evidence(quote: str, preferred_source_type: str | None = None) -> str | None:
            return evidence_registry.register_quote(
                quote,
                research_pages,
                preferred_source_type,
            )

        # Map checked-fact quote -> evidence_id so facts can be linked back
        checked_fact_ev_map: dict[str, list[str]] = {}  # fact_id -> [ev_id, ...]
        for f in checked_guide.can_write:
            ev_ids_for_fact: list[str] = []
            for ev in f.evidence:
                eid = evidence_registry.register_checked_evidence(
                    ev,
                    "company_website",
                )
                if eid:
                    ev_ids_for_fact.append(eid)
            checked_fact_ev_map[f.fact_id] = ev_ids_for_fact
        for limited_fact in checked_guide.can_write_with_limits:
            ev_ids_for_fact = []
            for ev in limited_fact.evidence:
                eid = evidence_registry.register_checked_evidence(
                    ev,
                    "company_website",
                )
                if eid:
                    ev_ids_for_fact.append(eid)
            checked_fact_ev_map[limited_fact.fact_id] = ev_ids_for_fact

        # 2. Parse candidate subtopics
        candidate_subtopics: list[CandidateSubtopic] = []
        for idx, item in enumerate(data.get("candidate_subtopics") or [], start=1):
            if not isinstance(item, dict):
                continue
            title = _clean(str(item.get("title") or ""))
            if not title:
                continue

            ev_ids: list[str] = []
            for q in item.get("evidence_quotes") or []:
                ev_id = add_evidence(str(q))
                if ev_id:
                    ev_ids.append(ev_id)

            # Fix 3: Stricter support_status — fully_supported requires BOTH
            # an evidence record AND a verified-fact cross-link.
            if ev_ids:
                support_status: str = "partially_supported"
            else:
                support_status = "needs_more_research"

            angle = str(item.get("topic_angle") or "company_capability").strip().lower()
            if angle not in {
                "company_capability",
                "industry_context",
                "comparison_methodology",
                "educational_guide",
            }:
                angle = "company_capability"

            relevance = str(item.get("relevance_to_main_topic") or "high").strip().lower()
            if relevance not in _RELEVANCE:
                relevance = "high"

            research_needed = _string_list(item.get("research_needed"))
            if support_status in {"needs_more_research", "partially_supported"} and not research_needed:
                research_needed = [f"Find primary evidence for: {title}"]

            candidate_subtopics.append(
                CandidateSubtopic(
                    subtopic_id=f"subtopic_{idx:03d}",
                    title=title,
                    description=_clean(str(item.get("description") or "")),
                    topic_angle=angle,
                    relevance=relevance,
                    support_status=support_status,
                    verified_fact_ids=[],
                    evidence_ids=ev_ids,
                    research_needed=research_needed,
                    suggested_questions=_string_list(item.get("suggested_questions")),
                )
            )

        # 3. Parse verified facts
        # Fix B: after building facts, link them to the evidence IDs we seeded from
        # checked facts (checked_fact_ev_map) so facts don't remain evidence-orphaned.
        verified_facts: list[VerifiedFact] = []
        for idx, item in enumerate(data.get("verified_facts") or [], start=1):
            if not isinstance(item, dict):
                continue
            statement = _clean(str(item.get("statement") or ""))
            if not statement:
                continue
            q = str(item.get("evidence_quote") or "")
            ev_id = add_evidence(q) if q else None

            ev_type = "company_website"
            boundary = _clean(str(item.get("wording_boundary") or ""))

            has_evidence = bool(ev_id and _safe_narrow(statement, q, statement))
            if not has_evidence:
                ev_id = None
            confidence: str
            if has_evidence:
                confidence = str(item.get("confidence") or "high").strip().lower()
                if confidence not in {"high", "medium", "low"}:
                    confidence = "high"
            else:
                confidence = "low"
                self._add_warning(
                    "An Agent 3 fact was not accepted as verified because its quote "
                    "did not both match a retrieved page and support the statement."
                )
                if not boundary:
                    boundary = "No verbatim evidence quote found; treat as unverified until sourced."

            fact_id = f"fact_model_{idx:03d}"
            verified_facts.append(
                VerifiedFact(
                    fact_id=fact_id,
                    statement=statement,
                    category=_clean(str(item.get("category") or "company_capability")),
                    evidence_type=ev_type,
                    evidence_ids=[ev_id] if ev_id else [],
                    confidence=confidence,
                    status="verified" if has_evidence else "inconclusive",
                    wording_boundary=boundary if boundary else None,
                )
            )

        # Add checked facts if LLM verified_facts was sparse
        if not verified_facts:
            for f in checked_guide.can_write:
                ev_ids_f = checked_fact_ev_map.get(f.fact_id, [])
                verified_facts.append(
                    VerifiedFact(
                        fact_id=f.fact_id,
                        statement=f.claim,
                        category="company_capability",
                        evidence_type="company_website",
                        evidence_ids=ev_ids_f,
                        confidence="high" if ev_ids_f else "low",
                        status="verified" if ev_ids_f else "inconclusive",
                    )
                )
            for limited_fact in checked_guide.can_write_with_limits:
                ev_ids_l = checked_fact_ev_map.get(limited_fact.fact_id, [])
                verified_facts.append(
                    VerifiedFact(
                        fact_id=limited_fact.fact_id,
                        statement=limited_fact.statement,
                        category="company_capability",
                        evidence_type="company_website",
                        evidence_ids=ev_ids_l,
                        confidence="medium" if ev_ids_l else "low",
                        status="verified" if ev_ids_l else "inconclusive",
                        wording_boundary=limited_fact.wording_limit,
                    )
                )

        # Fix B (post-fact): Re-link LLM-emitted facts to checked_fact_ev_map evidence.
        # The LLM may emit a fact whose statement matches a checked claim; re-use those
        # evidence IDs so the fact isn't left orphaned.
        checked_claim_norms = {
            _norm(f.claim): checked_fact_ev_map.get(f.fact_id, []) for f in checked_guide.can_write
        }
        checked_claim_norms.update(
            {
                _norm(limited_fact.statement): checked_fact_ev_map.get(limited_fact.fact_id, [])
                for limited_fact in checked_guide.can_write_with_limits
            }
        )
        for vf in verified_facts:
            if not vf.evidence_ids:
                match_ev = checked_claim_norms.get(_norm(vf.statement), [])
                if match_ev:
                    vf.evidence_ids = match_ev
                    vf.confidence = "medium"  # checked-claim source, not LLM-fresh

        # Fix 2 (post-fact): Link verified facts to candidate subtopics based on shared evidence IDs or topic matching
        ev_to_fact_ids: dict[str, list[str]] = {}
        for vf in verified_facts:
            for eid in vf.evidence_ids:
                ev_to_fact_ids.setdefault(eid, []).append(vf.fact_id)

        for sub in candidate_subtopics:
            linked_facts: set[str] = set()
            for eid in sub.evidence_ids:
                for fid in ev_to_fact_ids.get(eid, []):
                    linked_facts.add(fid)
            if not linked_facts:
                sub_words = set(_norm(sub.title).split()) | set(_norm(sub.description).split())
                for vf in verified_facts:
                    vf_words = set(_norm(vf.statement).split())
                    if len(sub_words & vf_words) >= 3:
                        linked_facts.add(vf.fact_id)
            sub.verified_fact_ids = sorted(list(linked_facts))
            if sub.verified_fact_ids and sub.evidence_ids:
                sub.support_status = "fully_supported"

        # 4. Parse research gaps
        # Build a lookup of subtopic titles so we can link gaps to affected subtopics
        subtopic_title_map = {sub.title.lower(): sub.subtopic_id for sub in candidate_subtopics}
        research_gaps: list[ResearchGap] = []
        for idx, item in enumerate(data.get("research_gaps") or [], start=1):
            if not isinstance(item, dict):
                continue
            t_or_q = _clean(str(item.get("topic_or_question") or ""))
            if not t_or_q:
                continue
            missing_claim = _clean(str(item.get("missing_claim") or ""))
            # Derive affects_subtopic_ids by matching gap topic against subtopic titles
            raw_affects = item.get("affects_subtopic_ids") or []
            affects: list[str] = [sid for sid in _string_list(raw_affects) if sid.startswith("subtopic_")]
            # Also auto-link if gap topic words appear in any subtopic title
            if not affects:
                t_words = set(t_or_q.lower().split())
                for title_lc, sid in subtopic_title_map.items():
                    if len(t_words & set(title_lc.split())) >= 2:
                        affects.append(sid)
            research_gaps.append(
                ResearchGap(
                    gap_id=f"gap_{idx:03d}",
                    topic_or_question=t_or_q,
                    missing_claim=missing_claim or t_or_q,
                    reason=_clean(str(item.get("reason") or "Evidence inconclusive")),
                    suggested_search_direction=_clean(str(item.get("suggested_search_direction") or ""))
                    or None,
                    affects_subtopic_ids=affects,
                )
            )

        seen_gap_claims = {_norm(gap.missing_claim or gap.topic_or_question) for gap in research_gaps}
        for gap in checked_guide.information_still_needed:
            key = _norm(gap.missing_claim or gap.topic_or_question)
            if key and key not in seen_gap_claims:
                research_gaps.append(gap.model_copy(update={"gap_id": f"gap_{len(research_gaps) + 1:03d}"}))
                seen_gap_claims.add(key)

        # 5. Parse limitations
        limitations: list[str] = _string_list(data.get("limitations"))
        for standing in _STANDING_BANS:
            limitations.append(
                f"Do not use superlative claims such as '{standing}' without verified primary certification."
            )
        for lim in _LIMITATIONS:
            if lim not in limitations:
                limitations.append(lim)
        for item in checked_guide.cannot_write:
            if item.reason and item.reason not in limitations:
                limitations.append(f"Avoid claiming '{item.claim_or_topic}': {item.reason}")
        limitations = _string_list(limitations)

        # Fix C: Legacy topics must use verified_fact_ids (fact_xxx), NOT evidence_ids (ev_xxx).
        # EvidenceBackedTopic.evidence_fact_ids is supposed to reference fact records.
        legacy_topics: list[EvidenceBackedTopic] = []
        for sub in candidate_subtopics:
            sub_ev = [ev for ev in evidence_items if ev.evidence_id in sub.evidence_ids]
            legacy_topics.append(
                EvidenceBackedTopic(
                    topic=sub.title,
                    description=sub.description,
                    evidence_fact_ids=sub.verified_fact_ids,  # fact_xxx only
                    supporting_quotes=[e.quote for e in sub_ev],
                    source_urls=[e.url for e in sub_ev],
                )
            )

        main_topic = _clean(
            str(data.get("main_topic") or understanding.get("main_topic") or context.original_prompt)
        )
        search_intent = _clean(
            str(
                data.get("search_intent")
                or understanding.get("search_intent")
                or context.prompt_understanding.intent
            )
        )
        target_audience = _clean(
            str(
                data.get("target_audience")
                or understanding.get("target_audience")
                or context.prompt_understanding.audience
                or "Industry professionals"
            )
        )
        prompt_scope = str(data.get("prompt_scope") or understanding.get("prompt_scope") or "hybrid")

        # ----------------------------------------------------------------
        # Build clean Agent 4 handoff — single canonical object, no dupes
        # ----------------------------------------------------------------
        # Editorial constraints: standing bans + per-claim limits
        # target_audience as list for Agent 4
        raw_audience = understanding.get("target_audience") or context.prompt_understanding.audience or ""
        if isinstance(raw_audience, list):
            audience_list: list[str] = [str(a) for a in raw_audience if a]
        elif raw_audience:
            audience_list = [s.strip() for s in str(raw_audience).split(",") if s.strip()]
        else:
            audience_list = ["Industry professionals"]

        # Fix D: Reconcile prohibited_claims vs. limited claims.
        # A claim already narrowly supported under can_write_with_limits must NOT appear
        # as a blanket prohibition. Only store the broader extension that is prohibited.
        limited_claim_norms: set[str] = {
            _norm(limited_fact.statement) for limited_fact in checked_guide.can_write_with_limits
        }
        raw_prohibited = [item.claim_or_topic for item in checked_guide.cannot_write] + banned
        deduped_prohibited: list[str] = []
        for claim in raw_prohibited:
            if _norm(claim) not in limited_claim_norms:
                deduped_prohibited.append(claim)

        # Fix E: Drop research gaps that are already resolved by verified facts.
        # A gap is resolved if its missing_claim text substantially overlaps with a
        # verified fact's statement that has at least one evidence_id.
        resolved_norms: set[str] = {_norm(vf.statement) for vf in verified_facts if vf.evidence_ids}

        def _gap_is_resolved(gap: ResearchGap) -> bool:
            mc_norm = _norm(gap.missing_claim)
            # Consider resolved if >=3 words match a verified fact norm
            mc_words = set(mc_norm.split())
            for fact_norm in resolved_norms:
                if len(mc_words & set(fact_norm.split())) >= 3:
                    return True
            return False

        research_gaps = [g for g in research_gaps if not _gap_is_resolved(g)]
        # Re-number gap IDs after filtering
        for i, g in enumerate(research_gaps, start=1):
            g.gap_id = f"gap_{i:03d}"

        # ----------------------------------------------------------------
        # Clean Agent 3 Research Result & Agent 4 Handoff (no duplicates)
        # ----------------------------------------------------------------
        brief_topic = main_topic or context.original_prompt
        brief_audience = target_audience or (
            ", ".join(audience_list) if audience_list else "Industry professionals"
        )
        brief_intent = search_intent or f"Understand capabilities, platforms, and services for {brief_topic}"

        # 1. Relevant findings (clean researched claims with source URLs and verification status)
        relevant_findings: list[CompanyTalkingPoint] = []
        seen_claims: set[str] = set()

        for vf in verified_facts:
            st = vf.statement.strip()
            if not st or st in seen_claims:
                continue
            seen_claims.add(st)
            urls = [ev.url for ev in evidence_items if ev.evidence_id in vf.evidence_ids and ev.url]
            status = "verified" if vf.evidence_ids and vf.status == "verified" else "needs_human_review"
            relevant_findings.append(
                CompanyTalkingPoint(
                    claim=st,
                    source_urls=urls,
                    verification_status=status,
                )
            )

        for cw in checked_guide.can_write_with_limits:
            st = cw.statement.strip()
            if not st or st in seen_claims:
                continue
            seen_claims.add(st)
            evidence_ids = checked_fact_ev_map.get(cw.fact_id, [])
            urls = [ev.url for ev in evidence_items if ev.evidence_id in evidence_ids and ev.url]
            relevant_findings.append(
                CompanyTalkingPoint(
                    claim=st,
                    source_urls=urls,
                    verification_status="needs_human_review",
                )
            )

        for cw in checked_guide.can_write:
            st = cw.statement.strip()
            if not st or st in seen_claims:
                continue
            seen_claims.add(st)
            evidence_ids = checked_fact_ev_map.get(cw.fact_id, [])
            relevant_findings.append(
                CompanyTalkingPoint(
                    claim=st,
                    source_urls=[
                        ev.url for ev in evidence_items if ev.evidence_id in evidence_ids and ev.url
                    ],
                    verification_status=("verified" if evidence_ids else "needs_human_review"),
                )
            )

        # 2. Boundaries (clean, practical rules)
        boundaries = _handoff_boundaries(
            client_name,
            deduped_prohibited,
            self._warnings,
        )

        # 3. Filtered candidate subtopics
        filtered_subtopics = [sub for sub in candidate_subtopics if sub.support_status != "unsupported"]

        # 4. Canonical research handoff for Agent 4
        agent4_handoff = Agent4HandoffPackage(
            task="Create evidence-backed content brief and strategy",
            topic=brief_topic,
            search_intent=brief_intent,
            target_audience=brief_audience,
            relevant_findings=relevant_findings,
            evidence=evidence_items,
            information_still_needed=research_gaps,
            candidate_subtopics=filtered_subtopics,
            boundaries=list(dict.fromkeys(boundaries)),
            company=CompanyInfo(
                name=client_name,
                website=f"https://{client_host}",
            ),
            verification=VerificationHandoff(
                is_verified=verification.is_verified,
                intent_valid=verification.intent_valid,
                customer_relevant=verification.customer_relevant,
                reason=verification.reason or "",
            ),
        )

        return GroundTruthGuide(
            topic=brief_topic,
            main_topic=brief_topic,
            search_intent=brief_intent,
            target_audience=brief_audience,
            prompt_scope=prompt_scope,
            relevant_findings=relevant_findings,
            evidence=evidence_items,
            information_still_needed=research_gaps,
            research_gaps=research_gaps,
            candidate_subtopics=candidate_subtopics,
            verified_facts=verified_facts,
            boundaries=boundaries,
            limitations=boundaries,
            verification=verification,
            agent4_handoff=agent4_handoff,
        )

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
            relevance = (
                str(item.get("relevance_to_original_prompt") or item.get("relevance") or "medium")
                .strip()
                .lower()
            )
            if relevance not in _RELEVANCE:
                relevance = "medium"
            topic_relevance = str(item.get("relevance_to_topic") or "medium").strip().lower()
            if topic_relevance not in _RELEVANCE:
                topic_relevance = "medium"
            key = claim.lower()
            if not claim or key in seen or _is_superlative(claim):
                continue
            seen.add(key)
            questions.append(
                {
                    "topic": topic,
                    "claim": claim,
                    "claim_type": claim_type,
                    "relevance": relevance,
                    "relevance_to_topic": topic_relevance,
                    "checked_from": "prompt",
                }
            )
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
        first_pages, first_failed = self._collect_optional_pages(
            claim, client_host, client_name, first_party=True
        )
        checked = self._read_pages(claim, client_name, first_pages) if first_pages else []
        usable = [item for item in checked if item["support"] != "none"]
        retrieved = [page["url"] for page in first_pages if _is_customer_url(page["url"], client_host)]
        search_failed = first_failed and not first_pages
        usable = [hit for hit in usable if _is_customer_url(hit["url"], client_host)]
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
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Agent 3 content-opportunity assessment failed (%s).",
                type(exc).__name__,
            )
            self._add_warning(
                "Content-opportunity assessment failed; original opportunities were treated as unverified."
            )
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
            claims.append(
                {
                    "topic": topic,
                    "claim": claim,
                    "claim_type": claim_type,
                    "relevance": relevance,
                    "relevance_to_topic": topic_relevance,
                    "checked_from": "content_opportunity",
                    "opportunity": opportunity,
                }
            )
        return claims

    def _collect_pages(
        self,
        claim: str,
        client_host: str,
        client_name: str,
        first_party: bool,
    ) -> tuple[list[dict[str, Any]], bool]:
        if not first_party:
            return [], False
        query = claim
        response = self.research.search(
            query,
            max_results=PAGES_PER_PASS,
            include_domains=[client_host],
        )
        if response.failed:
            return [], True
        pages: list[dict[str, Any]] = []
        for result in response.results:
            if not _is_customer_url(result.url, client_host):
                continue
            page = self.research.extract(result.url, max_chars=PAGE_CHARS)
            if page is None or not page.text.strip():
                continue
            if not _is_customer_url(page.url, client_host):
                continue
            pages.append(
                {
                    "url": page.url,
                    "text": page.text,
                    "published": _parse_date(result.published),
                    "source_type": "company_website",
                }
            )
            if len(pages) >= PAGES_PER_PASS:
                break
        return pages, False

    def _read_pages(self, claim: str, client_name: str, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        packed = [{"url": page["url"], "text": page["text"][:PAGE_CHARS]} for page in pages]
        system = """You check one capability claim against supplied page text.
Return one JSON object:
{"pages":[{"url":"","support":"full|partial|contradicts|none","quote":"","narrow_statement":"","wording_limit":""}]}
Rules:
- Use only the supplied page text. Do not use outside knowledge.
- These supplied pages are from the customer's official website; verify customer-specific claims only from them.
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
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Agent 3 evidence assessment failed (%s) for one claim.",
                type(exc).__name__,
            )
            self._add_warning(
                "One optional evidence assessment failed; the affected claim was left unresolved."
            )
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
            checked.append(
                {
                    "support": support,
                    "quote": _clean(quote),
                    "url": url,
                    "published": page["published"],
                    "narrow_statement": _clean(str(item.get("narrow_statement") or "")),
                    "wording_limit": _clean(str(item.get("wording_limit") or "")),
                }
            )
        return checked

    def _complete(self, system: str, payload: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        """Call the Anthropic API and return the parsed JSON dict.

        Uses an assistant prefill of ``{`` so Claude is forced to emit raw JSON
        from the very first token, which eliminates preamble text that causes
        JSON parsing to fail.
        """
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
                "messages": [
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    # Prefill: force Claude to start the reply with "{"
                    {"role": "assistant", "content": "{"},
                ],
            },
            timeout=120.0,
        )
        response.raise_for_status()
        blocks = response.json().get("content") or []
        raw = "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        # Prepend the prefill character we sent so the text is a complete object
        text = "{" + raw
        if not raw.strip():
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
        return bool(host) and _is_customer_url(hit["url"], host)

    official_hits = [hit for hit in hits if is_first(hit)]
    full_first = [hit for hit in official_hits if hit["support"] == "full"]
    partial_first = [hit for hit in official_hits if hit["support"] == "partial"]
    contradictions = [hit for hit in official_hits if hit["support"] == "contradicts"]
    extra_bans: list[str] = []

    supporting_hits = full_first + partial_first
    if contradictions and supporting_hits:
        reason = (
            "Conflicting retrieved evidence includes both support for and a direct "
            "contradiction of this claim. Further research and human review are required."
        )
        return _unresolved(
            fact_id,
            topic,
            claim,
            claim_type,
            relevance,
            topic_relevance,
            checked_from,
            "research_more",
            "inconclusive",
            reason,
            supporting_hits + contradictions,
            [claim],
            gap=True,
            asked_claim=claim,
        )

    if _is_superlative(claim) and not full_first:
        return _unresolved(
            fact_id,
            topic,
            claim,
            claim_type,
            relevance,
            topic_relevance,
            checked_from,
            "do_not_write",
            "inconclusive",
            "Superlatives are not allowed without an official website quote.",
            contradictions,
            extra_bans + [claim],
            gap=False,
            asked_claim=claim,
        )

    if full_first:
        hit = full_first[0]
        if _safe_narrow(claim, hit["quote"], claim):
            return _supported(
                fact_id,
                topic,
                claim,
                claim_type,
                relevance,
                topic_relevance,
                checked_from,
                "verified",
                "high",
                "write",
                hit,
                None,
                claim,
                None,
                claim,
            )
        full_first = []
        partial_first = [hit, *partial_first]
    if partial_first:
        hit = partial_first[0]
        narrow = _safe_narrow(hit["narrow_statement"], hit["quote"], claim)
        if narrow is None:
            narrow = _safe_narrow(hit["quote"], hit["quote"], claim)
        mismatch = _mismatch_reason(claim, hit["quote"])
        if narrow and narrow.lower() != claim.lower():
            return _supported(
                fact_id,
                topic,
                narrow,
                claim_type,
                relevance,
                topic_relevance,
                checked_from,
                "verified",
                "high",
                "write_limited",
                hit,
                None,
                narrow,
                mismatch or "State only what the quote supports.",
                claim,
            )
        if mismatch:
            return _unresolved(
                fact_id,
                topic,
                claim,
                claim_type,
                relevance,
                topic_relevance,
                checked_from,
                "do_not_write",
                "inconclusive",
                mismatch,
                [hit],
                extra_bans,
                gap=False,
                asked_claim=claim,
            )
        return _unresolved(
            fact_id,
            topic,
            claim,
            claim_type,
            relevance,
            topic_relevance,
            checked_from,
            "research_more",
            "inconclusive",
            "The evidence reviewed does not establish this claim. Further research is required.",
            [],
            extra_bans,
            gap=True,
            asked_claim=claim,
        )
    if contradictions:
        extra_bans.append(claim)
        return _unresolved(
            fact_id,
            topic,
            claim,
            claim_type,
            relevance,
            topic_relevance,
            checked_from,
            "do_not_write",
            "contradicted",
            "A retrieved page says otherwise.",
            contradictions,
            extra_bans,
            gap=False,
            asked_claim=claim,
        )
    reason = "The evidence reviewed does not establish this claim. Further research is required."
    if search_failed:
        reason = "The search did not return a page."
    elif retrieved_urls:
        reason = f"A page was retrieved, but it did not contain a quote that establishes this claim. {retrieved_urls[0]}"
    return _unresolved(
        fact_id,
        topic,
        claim,
        claim_type,
        relevance,
        topic_relevance,
        checked_from,
        "research_more",
        "inconclusive",
        reason,
        [],
        extra_bans,
        gap=True,
        asked_claim=claim,
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
            reason=wording_limit
            or "The quote supports a narrower statement and does not establish this broader claim.",
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
        fact,
        verdict,
        blocked,
        extra_bans,
        research,
        None,
        None,
        asked,
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
        fact,
        verdict,
        DoNotWrite(claim_or_topic=fact.claim, reason=reason),
        checked.extra_bans,
        None,
        None,
        None,
        checked.asked_claim,
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
    words = [
        word for word in re.findall(r"[a-z0-9]+", statement.lower()) if len(word) > 3 and word not in _STOP
    ]
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
    certifications = ("clia", "gclp", "cap", "iso")
    named = [item for item in certifications if re.search(rf"(?<!\w){item}(?!\w)", claim_l)]
    quoted = {item for item in certifications if re.search(rf"(?<!\w){item}(?!\w)", quote_l)}
    if named and not set(named) & quoted:
        if "iso" in quoted:
            return "The quote supports a different certification from the one claimed."
        return "The quote does not establish the named certification."
    return None


def _scope_note(context: PromptContext) -> ScopeNote:
    prompt = context.original_prompt
    topic = context.prompt_understanding.topic.strip() or None
    note = None
    if topic and topic.lower() not in prompt.lower():
        note = "The topic adds wording that is not in the original prompt. The original prompt is unchanged."
    return ScopeNote(original_prompt=prompt, inferred_topic=topic, scope_note=note)


def _handoff_boundaries(
    client_name: str,
    prohibited_claims: list[str],
    warnings: list[str],
) -> list[str]:
    boundaries = [
        "Use evidence to support company-specific claims.",
        f"Distinguish {client_name}'s capabilities from partner capabilities.",
        "Do not invent technical specifications, certifications, or performance metrics.",
        "Flag claims that need further research or human verification.",
    ]
    boundaries.extend(
        f"Do not make this unsupported or prohibited claim: {claim.strip()}"
        for claim in prohibited_claims
        if claim.strip()
    )
    boundaries.extend(warnings)
    return list(dict.fromkeys(boundaries))


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
    group_has_conflict = any(
        pair[1].fact.status == "contradicted"
        or "conflicting retrieved evidence" in (pair[1].fact.reason or "").casefold()
        for pair in group
    )
    opportunities: list[dict[str, str]] = []
    for question, other in group:
        from_prompt = (
            from_prompt or question.get("checked_from") == "prompt" or other.fact.checked_from == "prompt"
        )
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
            if (
                checked.fact.status == "verified"
                and not group_has_conflict
                and not _safe_narrow(checked.fact.claim, item.quote, checked.asked_claim)
            ):
                continue
            seen.add(key)
            evidence.append(item)

    conflicting_pairs = [
        pair
        for pair in group
        if pair[1].fact.status == "contradicted"
        or "conflicting retrieved evidence" in (pair[1].fact.reason or "").casefold()
    ]
    if conflicting_pairs:
        reason = (
            "Conflicting retrieved evidence includes both support for and a direct "
            "contradiction of this claim. Further research and human review are required."
        )
        fact = checked.fact.model_copy(
            update={
                "status": "inconclusive",
                "reason": reason,
                "evidence": evidence,
                "checked_from": "prompt" if from_prompt else "content_opportunity",
            }
        )
        verdict = checked.verdict.model_copy(update={"decision": "research_more", "reason": reason})
        gap = UnresolvedClaim(claim=checked.asked_claim, reason=reason)
        return checked._replace(fact=fact, verdict=verdict, block=None, gap=gap), opportunities

    fact = checked.fact.model_copy(
        update={
            "evidence": evidence,
            "checked_from": "prompt" if from_prompt else "content_opportunity",
        }
    )
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
    adjusted: list[tuple[dict[str, str], CheckedClaim]] = []
    for question, checked in pairs:
        fact = checked.fact
        adjusted.append((question, checked))
    merged = _merge_pairs(adjusted)
    can_write: list[Fact] = []
    limited: list[LimitedStatement] = []
    blocked: list[DoNotWrite] = []
    for checked, _opportunity_questions in merged:
        fact = checked.fact
        on_target = fact.relevance_to_original_prompt in {"high", "medium"} or fact.relevance_to_topic in {
            "high",
            "medium",
        }
        narrowed = checked.asked_claim.lower() != fact.claim.lower()
        if fact.status == "verified" and on_target and checked.verdict.decision == "write" and not narrowed:
            can_write.append(fact)
            continue
        if fact.status == "verified" and on_target and checked.verdict.decision in {"write", "write_limited"}:
            limited.append(
                LimitedStatement(
                    fact_id=fact.fact_id,
                    statement=fact.claim,
                    wording_limit=checked.wording_limit
                    or (checked.block.reason if checked.block else "State only what the quote supports."),
                    evidence=fact.evidence,
                    confidence=fact.confidence,
                    relevance_to_original_prompt=fact.relevance_to_original_prompt,
                    relevance_to_topic=fact.relevance_to_topic,
                )
            )
            if narrowed:
                blocked.append(
                    DoNotWrite(
                        claim_or_topic=checked.asked_claim,
                        reason=checked.wording_limit or "The quote supports a narrower statement.",
                    )
                )
            continue
        reason = (
            checked.verdict.reason or fact.reason or "The evidence reviewed does not establish this claim."
        )
        blocked.append(DoNotWrite(claim_or_topic=checked.asked_claim, reason=reason))
    gaps = _unique_unresolved([checked.gap for checked, _ in merged if checked.gap is not None])
    research_gaps = [
        ResearchGap(
            gap_id=f"gap_{index:03d}",
            topic_or_question=gap.claim,
            missing_claim=gap.claim,
            reason=gap.reason,
        )
        for index, gap in enumerate(gaps, start=1)
    ]
    return GroundTruthGuide(
        scope=_scope_note(context),
        can_write=can_write,
        can_write_with_limits=limited,
        cannot_write=_unique_blocks(blocked),
        information_still_needed=research_gaps,
        banned_claims=banned,
        limitations=list(_LIMITATIONS),
    )
