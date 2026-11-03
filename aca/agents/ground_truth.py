"""Agent 3: Ground truth for AImpact.

Starts from the original prompt and Agent 1's reading of it. The website
is supplied by the user. Prompt claims and each content opportunity are
searched, then marked as matching the original prompt, the topic being
searched, both, or neither.

The model's own knowledge is not evidence.
"""

from __future__ import annotations

import hashlib
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
from aca.observability.token_usage import record_claude_response
from aca.tools.web_research import WebResearch, get_web_research

load_dotenv()

# How many capability questions to test. Override with GroundTruthAgent(max_questions=...).
MAX_QUESTIONS = 6
PAGES_PER_PASS = 2
PAGE_CHARS = 12_000
CLAIMS_PER_VERIFICATION_BATCH = 4
PAGES_PER_VERIFICATION_BATCH = 6
CHARS_PER_VERIFICATION_BATCH = 14_000
RELEVANCE_SNIPPET_CHARS = 2_400
CLAIM_SNIPPET_CHARS = 2_200
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
_CALL_AUDIT_ENV = "ACA_AGENT3_CALL_AUDIT"
_AUDIT_CLAIM_ID_KEYS = {"claim_id", "claim_ids", "fact_id", "fact_ids"}
_AUDIT_CLAIM_TEXT_KEYS = {"claim", "statement", "missing_claim"}
_AUDIT_URL_KEYS = {"url", "urls", "source", "sources", "source_url", "source_urls"}
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
_SNIPPET_STOP_WORDS = {
    "about",
    "also",
    "and",
    "are",
    "company",
    "does",
    "for",
    "from",
    "has",
    "have",
    "into",
    "its",
    "offers",
    "that",
    "the",
    "their",
    "this",
    "through",
    "to",
    "was",
    "were",
    "with",
}


def _relevant_snippet(text: str, query: str, max_chars: int) -> str:
    """Select source-verbatim sentences with the strongest lexical match."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if sentence.strip()
    ]
    if not sentences:
        return ""
    terms = {
        term
        for term in _claim_key(query).split()
        if len(term) > 2 and term not in _SNIPPET_STOP_WORDS
    }
    if not terms:
        return text[:max_chars]

    ranked = []
    for index, sentence in enumerate(sentences):
        sentence_terms = set(_claim_key(sentence).split())
        score = len(terms & sentence_terms)
        if score:
            ranked.append((score, index))
    if not ranked:
        return text[:max_chars]

    selected: set[int] = set()
    used_chars = 0
    for _, index in sorted(ranked, key=lambda item: (-item[0], item[1])):
        additions = {index}
        if index > 0 and index - 1 not in selected:
            additions.add(index - 1)
        candidate = sorted(selected | additions)
        candidate_text = " [...] ".join(sentences[item] for item in candidate)
        if len(candidate_text) > max_chars:
            if not selected and len(sentences[index]) > max_chars:
                return sentences[index][:max_chars]
            continue
        selected.update(additions)
        used_chars = len(candidate_text)
        if used_chars >= max_chars:
            break
    if not selected:
        return text[:max_chars]
    return " [...] ".join(sentences[index] for index in sorted(selected))


def _audit_input_hash(system: str, payload: dict[str, Any], stage: str) -> str:
    serialized = json.dumps(
        {"stage": stage, "system": system, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _audit_claim_ids(payload: dict[str, Any]) -> list[str]:
    claim_ids: set[str] = set()

    def visit(value: Any, key: str = "") -> None:
        normalized_key = key.casefold()
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
        elif isinstance(value, list):
            for item in value:
                visit(item, key)
        elif isinstance(value, str) and value.strip():
            if normalized_key in _AUDIT_CLAIM_ID_KEYS:
                claim_ids.add(value.strip())
            elif normalized_key in _AUDIT_CLAIM_TEXT_KEYS or normalized_key == "content_opportunities":
                normalized_claim = " ".join(value.casefold().split())
                digest = hashlib.sha256(normalized_claim.encode("utf-8")).hexdigest()[:16]
                claim_ids.add(f"claim_{digest}")

    visit(payload)
    return sorted(claim_ids)


def _audit_source_urls(payload: dict[str, Any]) -> list[str]:
    source_urls: set[str] = set()

    def visit(value: Any, key: str = "") -> None:
        normalized_key = key.casefold()
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
        elif isinstance(value, list):
            for item in value:
                visit(item, key)
        elif isinstance(value, str) and normalized_key in _AUDIT_URL_KEYS:
            normalized_url = canonicalize_evidence_url(value)
            if normalized_url:
                source_urls.add(normalized_url)

    visit(payload)
    return sorted(source_urls)


def _claim_key(claim: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", claim.casefold()))


def _claims_are_duplicates(left: str, right: str) -> bool:
    left_key = _claim_key(left)
    right_key = _claim_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True

    left_words = set(left_key.split())
    right_words = set(right_key.split())
    if min(len(left_words), len(right_words)) < 5:
        return False
    overlap = len(left_words & right_words) / min(len(left_words), len(right_words))
    return overlap >= 0.85


def _deduplicate_claims(claims: list[dict[str, str]]) -> list[dict[str, str]]:
    relevance_rank = {"low": 0, "medium": 1, "high": 2}
    unique: list[dict[str, str]] = []
    for candidate in claims:
        claim = _clean(str(candidate.get("claim") or ""))
        if not claim:
            continue
        duplicate = next(
            (
                item
                for item in unique
                if (
                    _claim_key(item["claim"]) == _claim_key(claim)
                    or (
                        item.get("claim_type") == candidate.get("claim_type")
                        and _claims_are_duplicates(item["claim"], claim)
                    )
                )
            ),
            None,
        )
        if duplicate is None:
            unique.append({**candidate, "claim": claim})
            continue

        for field in ("relevance", "relevance_to_topic"):
            current = duplicate.get(field, "low")
            incoming = candidate.get(field, "low")
            if relevance_rank.get(incoming, 0) > relevance_rank.get(current, 0):
                duplicate[field] = incoming
        if duplicate.get("checked_from") == "content_opportunity" and candidate.get("checked_from") == "prompt":
            duplicate["checked_from"] = "prompt"
    return unique


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
        self._page_cache: dict[str, dict[str, Any]] = {}

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
                    cached_root = {
                        "url": root_page.url,
                        "text": root_page.text[:PAGE_CHARS],
                        "published": None,
                        "source_type": "company_website",
                    }
                    root_key = canonicalize_evidence_url(root_page.url)
                    if root_key:
                        self._page_cache[root_key] = cached_root
                    retrieved_pages.append(cached_root)
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

        relevance_query = " ".join(
            (
                context.original_prompt,
                context.prompt_understanding.intent,
                context.prompt_understanding.topic,
            )
        )
        packed = [
            {
                "url": page["url"],
                "text": _relevant_snippet(
                    page["text"],
                    relevance_query,
                    RELEVANCE_SNIPPET_CHARS,
                ),
            }
            for page in retrieved_pages
        ]

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

        data = self._complete(
            system,
            payload,
            max_tokens=1500,
            stage="_understand_prompt_and_verify",
            purpose="Assess prompt intent and company relevance using retrieved official-site pages.",
        )
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
            page_key = canonicalize_evidence_url(str(u or ""))
            if page_key and _is_customer_url(str(u), client_host) and page_key not in seen_urls:
                seen_urls.add(page_key)
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
            page_key = canonicalize_evidence_url(str(u or ""))
            if page_key and _is_customer_url(str(u), client_host) and page_key not in seen_urls:
                seen_urls.add(page_key)
                all_pages.append(
                    {
                        "url": u,
                        "text": p.get("text", ""),
                        "published": p.get("published"),
                        "source_type": "company_website",
                    }
                )

        # Generate the prompt questions and content-opportunity claims in one request.
        try:
            research_items = self._research_items(context)
            questions = self._normalize_questions(research_items)[: self.max_questions]
            opportunities = self._normalize_opportunity_claims(context, research_items)
            if not questions:
                raise RuntimeError("Agent 3 did not produce any capability questions.")
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            logger.warning("Agent 3 research-item generation failed (%s).", type(exc).__name__)
            self._add_warning("Research-item generation failed; prompt claim checks were skipped.")
            questions = []
            opportunities = self._normalize_opportunity_claims(context, {})
        pairs: list[tuple[dict[str, str], CheckedClaim]] = []
        claims_to_check = list(questions)
        for opportunity in opportunities:
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
            claims_to_check.append(opportunity)

        banned = list(_STANDING_BANS)
        claims_to_check = _deduplicate_claims(claims_to_check)
        checked_claims, claim_pages = self._check_questions_batch(
            claims_to_check,
            client_name,
            client_host,
            first_fact_index=len(pairs) + 1,
        )
        pairs.extend(
            (question, _apply_target(question, checked))
            for question, checked in zip(claims_to_check, checked_claims)
        )
        for checked in checked_claims:
            for item in checked.extra_bans:
                if item not in banned:
                    banned.append(item)

        for page in claim_pages:
            page_url = str(page.get("url") or "")
            page_key = canonicalize_evidence_url(page_url)
            if not page_key or page_key in seen_urls:
                continue
            seen_urls.add(page_key)
            all_pages.append(page)

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

        verified_evidence = [
            {
                "claim": fact.claim,
                "status": "verified",
                "source_url": fact.evidence[0].url,
                "supporting_quote": fact.evidence[0].quote,
                "limitations": "",
            }
            for fact in checked_guide.can_write
            if fact.evidence
        ] + [
            {
                "claim": fact.statement,
                "status": "verified_with_limits",
                "source_url": fact.evidence[0].url,
                "supporting_quote": fact.evidence[0].quote,
                "limitations": fact.wording_limit,
            }
            for fact in checked_guide.can_write_with_limits
            if fact.evidence
        ]
        unsupported_claims = [
            {"claim": item.claim_or_topic, "reason": item.reason}
            for item in checked_guide.cannot_write
        ] + [
            {
                "claim": item.missing_claim or item.topic_or_question,
                "reason": item.reason,
            }
            for item in checked_guide.information_still_needed
        ]

        system = """You are Agent 3. Organize only the supplied, already-verified evidence into useful blog subtopics and research gaps.
Return JSON with exactly: main_topic, search_intent, target_audience, prompt_scope, candidate_subtopics, research_gaps, limitations.
Each candidate_subtopics item has: title, description, topic_angle (company_capability|industry_context|comparison_methodology|educational_guide), relevance_to_main_topic (high|medium|low), research_needed (list), suggested_questions (list), evidence_quotes (list).
Each research_gaps item has: topic_or_question, missing_claim, reason, suggested_search_direction, affects_subtopic_ids (list).

Rules:
- Keep the original prompt as the scope anchor; do not introduce unrelated capabilities or topics.
- Use only verified_evidence for company-specific statements. Every evidence quote must exactly match a supplied supporting_quote; do not invent, paraphrase, or broaden claims.
- Evidence with status verified_with_limits can support only the narrow claim and must respect its limitations.
- Do not mark a subtopic supported unless its exact evidence quote supports the described company capability. Otherwise include a specific research need.
- Use unsupported_claims to describe relevant gaps, not to make claims about the company.
- Do not add superlatives, comparisons, outcomes, ROI, or certifications unless explicitly present in verified evidence.
- Return concise JSON only."""

        payload = {
            "customer_name": client_name,
            "original_prompt": context.original_prompt,
            "understanding": {
                "main_topic": understanding.get("main_topic") or context.original_prompt,
                "search_intent": understanding.get("search_intent") or context.prompt_understanding.intent,
                "target_audience": understanding.get("target_audience")
                or context.prompt_understanding.audience,
                "prompt_scope": understanding.get("prompt_scope") or "hybrid",
            },
            "verified_evidence": verified_evidence,
            "unsupported_claims": unsupported_claims,
        }

        try:
            # 8192 tokens gives enough room for a rich subtopic JSON response
            data = self._complete(
                system,
                payload,
                max_tokens=8192,
                stage="_discover_subtopics_and_evidence",
                purpose="Discover subtopics from compact verified evidence and unsupported claims.",
            )
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

        checked_fact_ev_map: dict[str, list[str]] = {}
        verified_quote_ids: dict[str, list[str]] = {}

        def remember_checked_evidence(fact_id: str, evidence: list[Evidence]) -> list[str]:
            evidence_ids: list[str] = []
            for item in evidence:
                evidence_id = evidence_registry.register_checked_evidence(item, "company_website")
                if not evidence_id:
                    continue
                evidence_ids.append(evidence_id)
                quote_key = _clean(item.quote).casefold()
                verified_quote_ids.setdefault(quote_key, [])
                if evidence_id not in verified_quote_ids[quote_key]:
                    verified_quote_ids[quote_key].append(evidence_id)
            checked_fact_ev_map[fact_id] = evidence_ids
            return evidence_ids

        for f in checked_guide.can_write:
            remember_checked_evidence(f.fact_id, f.evidence)
        for limited_fact in checked_guide.can_write_with_limits:
            remember_checked_evidence(limited_fact.fact_id, limited_fact.evidence)

        def add_evidence(quote: str) -> str | None:
            ids = verified_quote_ids.get(_clean(quote).casefold(), [])
            return ids[0] if ids else None

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

        # Facts are deterministic outputs of the claim verification rules, not model-generated.
        verified_facts: list[VerifiedFact] = []
        for f in checked_guide.can_write:
            evidence_ids = checked_fact_ev_map.get(f.fact_id, [])
            category = {
                "product": "product_technology",
                "certification": "regulatory_certification",
                "service": "service_scope",
                "geographic_service_availability": "service_scope",
            }.get(f.claim_type, "company_capability")
            verified_facts.append(
                VerifiedFact(
                    fact_id=f.fact_id,
                    statement=f.claim,
                    category=category,
                    evidence_type="company_website",
                    evidence_ids=evidence_ids,
                    confidence=f.confidence if evidence_ids else "low",
                    status="verified" if evidence_ids else "inconclusive",
                )
            )
        for limited_fact in checked_guide.can_write_with_limits:
            evidence_ids = checked_fact_ev_map.get(limited_fact.fact_id, [])
            verified_facts.append(
                VerifiedFact(
                    fact_id=limited_fact.fact_id,
                    statement=limited_fact.statement,
                    category="company_capability",
                    evidence_type="company_website",
                    evidence_ids=evidence_ids,
                    confidence=limited_fact.confidence if evidence_ids else "low",
                    status="verified" if evidence_ids else "inconclusive",
                    wording_boundary=limited_fact.wording_limit,
                )
            )

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
        unique_gaps: list[ResearchGap] = []
        seen_gap_keys: set[str] = set()
        for gap in research_gaps:
            gap_key = _claim_key(gap.missing_claim or gap.topic_or_question)
            if not gap_key or gap_key in seen_gap_keys:
                continue
            seen_gap_keys.add(gap_key)
            unique_gaps.append(gap)
        research_gaps = unique_gaps
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
            claim_key = _claim_key(st)
            if not claim_key or claim_key in seen_claims:
                continue
            seen_claims.add(claim_key)
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
            claim_key = _claim_key(st)
            if not claim_key or claim_key in seen_claims:
                continue
            seen_claims.add(claim_key)
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
            st = cw.claim.strip()
            claim_key = _claim_key(st)
            if not claim_key or claim_key in seen_claims:
                continue
            seen_claims.add(claim_key)
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
        filtered_subtopics = []
        seen_subtopics: set[str] = set()
        for subtopic in candidate_subtopics:
            subtopic_key = _claim_key(subtopic.title)
            if subtopic.support_status == "unsupported" or not subtopic_key or subtopic_key in seen_subtopics:
                continue
            seen_subtopics.add(subtopic_key)
            filtered_subtopics.append(subtopic)

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
        questions = self._normalize_questions(self._research_items(context))
        if not questions:
            raise RuntimeError("Agent 3 did not produce any capability questions.")
        return questions

    def _research_items(self, context: PromptContext) -> dict[str, Any]:
        system = """Create the customer claims Agent 3 should verify, including prompt-derived questions and claims for each supplied content opportunity.
Return one JSON object:
{"questions":[{"axis":"relevance|capability|geography","claim_type":"product|platform_capability|service|certification|geographic_service_availability|other","relevance_to_original_prompt":"high|medium|low","relevance_to_topic":"high|medium|low","topic":"","claim":""}],"opportunities":[{"opportunity":"exact supplied opportunity","topic":"","claim":"","claim_type":"product|platform_capability|service|certification|geographic_service_availability|other","relevance_to_original_prompt":"high|medium|low","relevance_to_topic":"high|medium|low"}]}
Prompt-derived questions should cover relevant capabilities, services, and geography without duplicating content opportunities. Claims must be about the named customer, be separately verifiable, and avoid superlatives. Keep the lists short, and copy every content opportunity exactly once with its concrete claim. Score relevance to the original prompt and inferred topic separately. JSON only."""
        return self._complete(
            system,
            context.model_dump(),
            max_tokens=3500,
            stage="_research_items",
            purpose="Generate prompt questions and opportunity claims together for verification.",
        )

    @staticmethod
    def _normalize_questions(data: dict[str, Any]) -> list[dict[str, str]]:
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

    def _check_questions_batch(
        self,
        questions: list[dict[str, str]],
        client_name: str,
        client_host: str,
        first_fact_index: int = 1,
    ) -> tuple[list[CheckedClaim], list[dict[str, Any]]]:
        """Search claims individually, then verify them together against deduplicated pages."""
        records: list[dict[str, Any]] = []
        all_pages: dict[str, dict[str, Any]] = {}
        for offset, question in enumerate(questions):
            fact_id = f"fact_{first_fact_index + offset:03d}"
            pages, failed = self._collect_optional_pages(
                question["claim"],
                client_host,
                client_name,
                first_party=True,
            )
            unique_claim_pages: dict[str, dict[str, Any]] = {}
            for page in pages:
                url = str(page.get("url") or "")
                page_key = canonicalize_evidence_url(url)
                if not page_key or not _is_customer_url(url, client_host):
                    continue
                normalized_page = {
                    **page,
                    "url": url,
                    "text": str(page.get("text") or "")[:PAGE_CHARS],
                }
                unique_claim_pages.setdefault(page_key, normalized_page)
                all_pages.setdefault(page_key, normalized_page)
            records.append(
                {
                    "fact_id": fact_id,
                    "question": question,
                    "pages": unique_claim_pages,
                    "search_failed": failed and not unique_claim_pages,
                    "retrieved": [page["url"] for page in unique_claim_pages.values()],
                }
            )

        batches: list[dict[str, Any]] = []
        for record in records:
            page_keys = set(record["pages"])
            if not page_keys:
                continue
            best_batch = None
            best_overlap = 0
            for batch in batches:
                combined_keys = batch["page_keys"] | page_keys
                combined_chars = sum(
                    min(len(all_pages[key]["text"]), CLAIM_SNIPPET_CHARS)
                    for key in combined_keys
                )
                if (
                    len(batch["claims"]) < CLAIMS_PER_VERIFICATION_BATCH
                    and len(combined_keys) <= PAGES_PER_VERIFICATION_BATCH
                    and combined_chars <= CHARS_PER_VERIFICATION_BATCH
                ):
                    overlap = len(batch["page_keys"] & page_keys)
                    if best_batch is None or overlap > best_overlap:
                        best_batch = batch
                        best_overlap = overlap
            if best_batch is None:
                batches.append({"claims": [record], "page_keys": page_keys})
            else:
                best_batch["claims"].append(record)
                best_batch["page_keys"].update(page_keys)

        checked_by_id: dict[str, list[dict[str, Any]]] = {}
        for batch in batches:
            if not batch["page_keys"]:
                continue
            page_keys = sorted(batch["page_keys"])
            page_ids = {key: f"page_{index:03d}" for index, key in enumerate(page_keys, start=1)}
            pages_payload = [
                {
                    "page_id": page_ids[key],
                    "url": all_pages[key]["url"],
                    "text": _relevant_snippet(
                        all_pages[key]["text"],
                        " ".join(
                            f"{record['question']['topic']} {record['question']['claim']}"
                            for record in batch["claims"]
                            if key in record["pages"]
                        ),
                        CLAIM_SNIPPET_CHARS,
                    ),
                    "published": (
                        all_pages[key]["published"].isoformat()
                        if isinstance(all_pages[key].get("published"), date)
                        else all_pages[key].get("published")
                    ),
                }
                for key in page_keys
            ]
            claims_payload = [
                {
                    "claim_id": record["fact_id"],
                    "claim": record["question"]["claim"],
                    "page_ids": [page_ids[key] for key in sorted(record["pages"])],
                }
                for record in batch["claims"]
            ]
            try:
                response = self._read_pages_batch(claims_payload, pages_payload, client_name)
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                logger.warning(
                    "Agent 3 batched evidence assessment failed (%s) for %d claims.",
                    type(exc).__name__,
                    len(batch["claims"]),
                )
                self._add_warning(
                    "A batched evidence assessment failed; affected claims were left unresolved."
                )
                response = {}

            for claim_id, items in response.items():
                checked_by_id[claim_id] = items

        checked_claims = []
        for record in records:
            question = record["question"]
            hits = checked_by_id.get(record["fact_id"], [])
            usable = [item for item in hits if item["support"] != "none"]
            for hit in usable:
                hit["client_host"] = client_host
            checked_claims.append(
                _apply(
                    record["fact_id"],
                    question["topic"],
                    question["claim"],
                    usable,
                    record["search_failed"],
                    question["claim_type"],
                    question["relevance"],
                    str(question.get("relevance_to_topic") or "medium"),
                    str(question.get("checked_from") or "prompt"),
                    record["retrieved"],
                )
            )
        return checked_claims, list(all_pages.values())

    def _read_pages_batch(
        self,
        claims: list[dict[str, Any]],
        pages: list[dict[str, Any]],
        client_name: str,
    ) -> dict[str, list[dict[str, Any]]]:
        system = """You assess multiple customer claims against shared official-site page excerpts.
Return JSON only:
{"claims":[{"claim_id":"exact supplied claim_id","pages":[{"page_id":"exact supplied page_id","support":"full|partial|contradicts|none","quote":"exact quote or empty for none","narrow_statement":"","wording_limit":""}]}]}
For each claim, assess only the pages whose IDs are listed for that claim. Use the same evidence rules for every item:
- Use only supplied page text; quote an exact contiguous span of at least 30 characters.
- full means the quote supports the whole claim; partial means it supports a narrower fact.
- Do not turn a product or platform into a testing service, or a customer installation into the company's own location or service speed.
- Do not add comparisons, superiority, or cost unless the quote states it.
- Use none and an empty quote when the page does not settle the claim.
- Return one result for every supplied claim and no other IDs."""
        try:
            data = self._complete(
                system,
                {
                    "client_name": client_name,
                    "claims": claims,
                    "pages": pages,
                },
                max_tokens=min(1200 * len(claims), 4096),
                stage="_read_pages_batch",
                purpose="Assess a small batch of claims against deduplicated official-site pages.",
            )
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Agent 3 batched evidence assessment failed (%s) for one batch.",
                type(exc).__name__,
            )
            self._add_warning(
                "One optional batched evidence assessment failed; affected claims were left unresolved."
            )
            return {}

        page_by_id = {page["page_id"]: page for page in pages}
        allowed_claim_ids = {claim["claim_id"] for claim in claims}
        results: dict[str, list[dict[str, Any]]] = {}
        for result in data.get("claims") or []:
            if not isinstance(result, dict):
                continue
            claim_id = str(result.get("claim_id") or "")
            if claim_id not in allowed_claim_ids:
                continue
            checked_pages = []
            allowed_page_ids = {
                page_id
                for claim in claims
                if claim["claim_id"] == claim_id
                for page_id in claim["page_ids"]
            }
            for item in result.get("pages") or []:
                if not isinstance(item, dict):
                    continue
                page_id = str(item.get("page_id") or "")
                page = page_by_id.get(page_id)
                support = str(item.get("support") or "none")
                quote = str(item.get("quote") or "")
                if (
                    page is None
                    or page_id not in allowed_page_ids
                    or support not in {"full", "partial", "contradicts"}
                    or not _quote_in_page(quote, page["text"])
                ):
                    continue
                checked_pages.append(
                    {
                        "support": support,
                        "quote": _clean(quote),
                        "url": page["url"],
                        "published": page.get("published"),
                        "narrow_statement": _clean(str(item.get("narrow_statement") or "")),
                        "wording_limit": _clean(str(item.get("wording_limit") or "")),
                    }
                )
            results[claim_id] = checked_pages
        return results

    def _opportunity_claims(self, context: PromptContext) -> list[dict[str, str]]:
        """Turn every Agent 1 content opportunity into a claim that will be searched."""
        opportunities = context.content_opportunities
        if not opportunities:
            return []
        try:
            data = self._research_items(context)
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Agent 3 research-item generation failed (%s).",
                type(exc).__name__,
            )
            self._add_warning(
                "Content-opportunity assessment failed; original opportunities were treated as unverified."
            )
            data = {}
        return self._normalize_opportunity_claims(context, data)

    @staticmethod
    def _normalize_opportunity_claims(
        context: PromptContext,
        data: dict[str, Any],
    ) -> list[dict[str, str]]:
        opportunities = context.content_opportunities
        if not opportunities:
            return []
        by_text: dict[str, dict[str, Any]] = {}
        for item in data.get("opportunities") or []:
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
            page_key = canonicalize_evidence_url(result.url)
            if page_key in self._page_cache:
                cached_page = self._page_cache[page_key]
                pages.append(
                    {
                        **cached_page,
                        "published": cached_page.get("published") or _parse_date(result.published),
                    }
                )
                if len(pages) >= PAGES_PER_PASS:
                    break
                continue
            page = self.research.extract(result.url, max_chars=PAGE_CHARS)
            if page is None or not page.text.strip():
                continue
            if not _is_customer_url(page.url, client_host):
                continue
            cached_page = {
                "url": page.url,
                "text": page.text[:PAGE_CHARS],
                "published": _parse_date(result.published),
                "source_type": "company_website",
            }
            self._page_cache[page_key] = cached_page
            pages.append(cached_page)
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
                stage="_read_pages",
                purpose="Assess whether supplied page excerpts support this specific claim.",
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

    def _complete(
        self,
        system: str,
        payload: dict[str, Any],
        max_tokens: int,
        *,
        stage: str = "unspecified",
        purpose: str = "unspecified",
    ) -> dict[str, Any]:
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
        usage = record_claude_response(response, self.model)
        if os.getenv(_CALL_AUDIT_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
            logger.info(
                "agent3_call_audit=%s",
                json.dumps(
                    {
                        "stage": stage,
                        "purpose": purpose,
                        "input_hash": _audit_input_hash(system, payload, stage),
                        "claim_ids": _audit_claim_ids(payload),
                        "source_urls": _audit_source_urls(payload),
                        "input_tokens": usage.input_tokens if usage.usage_available else None,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        response.raise_for_status()
        response_data = response.json()
        blocks = response_data.get("content") or []
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
