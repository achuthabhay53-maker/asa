"""AImpact Source Intelligence Analyzer.



Connects to AImpact PostgreSQL to extract AI prompts, model responses, and citations,

then analyzes each low-scoring prompt individually as well as producing an aggregated

cross-prompt intelligence report.



Report Structure:

    report

    ├── campaign_summary

    ├── low_scoring_summary

    ├── prompt_analysis

    │   ├── prompt_1

    │   ├── prompt_2

    │   └── ...

    └── cross_prompt_insights



Run directly from CLI:

    python .\\\aca\\\agents\\\analyzer.py

    python .\\\aca\\\agents\\\analyzer.py --campaign-id 81 --score-threshold 70 --output report.json

"""

from __future__ import annotations



import argparse

import anthropic

import json

import os

import re

import sys

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from datetime import datetime, timezone

from pathlib import Path

from typing import Any

from urllib.parse import urlsplit, urlunsplit



# Ensure project root is in sys.path when executed directly as a script

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))



try:

    import tldextract

except ImportError:

    tldextract = None



from aca.orchestrator.state import (

    ContentGap,

    WorkflowState,

)

from aca.tools.aimpact import (

    get_campaign,

    get_campaign_audit_dataset,

    get_project,

)



HAIKU_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
MAX_CONCURRENT = 12

TOP_DOMAIN_COUNT = 5

NON_SOURCE_DOMAINS = {
    "google.com",
    "www.google.com",
    "bing.com",
    "www.bing.com",
    "search.yahoo.com",
    "www.search.yahoo.com",
    "duckduckgo.com",
    "www.duckduckgo.com",
}
HAIKU_MAX_TOKENS = 3000

SOURCE_TYPE_RULES = [

    (

        "research_or_academic",

        (

            "pubmed.",

            "nature.com",

            "sciencedirect.com",

            "springer.com",

            "arxiv.org",

            "doi.org",

            "ncbi.nlm.nih.gov",

            "nih.gov",

            "cell.com",

            "biorxiv.org",

            "medrxiv.org",

            "wiley.com",

            "frontiersin.org",

            "thelancet.com",

            "academic.",

            "broadinstitute.org",

            ".edu",

        ),

    ),

    (

        "government_or_public_sector",

        (

            ".gov",

            ".gov.uk",

            ".gov.in",

            "europa.eu",

            "who.int",

            "cdc.gov",

            "fda.gov",

        ),

    ),

    (

        "commercial_or_vendor",

        (

            "sigmaaldrich.",

            "thermofisher.",

            "criver.",

            "alamarbio.",

            "evetechnologies.",

            "crownbio.",

            "quanterix.",

            "creative-proteomics.",

            "bio-rad.",

            "abcam.",

            "merck.",

            "cellsignal.",

            "rndsystems.",

            "illumina.",

            "olink.",

            "bio-techne.",

            "mesoscale.",

            "cellcarta.",

            "oncimmune.",

            "axispharm.",

            "biomere.",

            "immunologixlabs.",

            "biopharmiq.",

            "arborassays.",

            "precisionformedicine.",

            "biotech",

            "pharma",

            "labs.com",

            "/product",

            "/services",

            "/solutions",

            "corp.",

            "inc.",

        ),

    ),

    (

        "industry_analyst_or_market_report",

        (

            "marketsandmarkets.",

            "rootsanalysis.",

            "mordorintelligence.",

            "grandviewresearch.",

            "fortunebusinessinsights.",

            "verifiedmarketresearch.",

            "globenewswire.com",

            "prnewswire.com",

            "businesswire.com",

        ),

    ),

    (

        "community_or_forum",

        (

            "reddit.com",

            "quora.com",

            "stackoverflow.com",

            "stackexchange.com",

            "community.",

        ),

    ),

    (

        "video_or_social",

        (

            "youtube.com",

            "youtu.be",

            "linkedin.com",

            "facebook.com",

            "instagram.com",

            "tiktok.com",

            "twitter.com",

            "x.com",

        ),

    ),

    (

        "review_or_directory",

        (

            "g2.com",

            "capterra.com",

            "trustpilot.com",

            "yelp.com",

            "tripadvisor.com",

        ),

    ),

    (

        "news_or_publisher",

        (

            "reuters.com",

            "bbc.com",

            "cnn.com",

            "forbes.com",

            "techcrunch.com",

            "nytimes.com",

            "bloomberg.com",

            "wsj.com",

        ),

    ),

]



CONTENT_TYPE_RULES = [

    (

        "comparison_or_buyer_guide",

        (

            "compare",

            "comparison",

            "versus",

            " vs ",

            "best ",

            "top ",

            "alternatives",

        ),

    ),

    ("review", ("review", "pros and cons", "rating", "customer reviews")),

    (

        "research_or_study",

        (

            "research",

            "study",

            "methodology",

            "experiment",

            "dataset",

            "findings",

            "clinical",

            "trial",

        ),

    ),

    (

        "news_or_article",

        ("news", "announced", "reportedly", "journalist", "press release"),

    ),

    (

        "documentation_or_reference",

        (

            "documentation",

            "docs",

            "api reference",

            "manual",

            "specification",

            "guideline",

            "protocol",

        ),

    ),

    (

        "product_or_service_page",

        (

            "pricing",

            "features",

            "product",

            "plans",

            "request a demo",

            "platform",

            "assay",

        ),

    ),

    (

        "community_discussion",

        ("reddit", "forum", "thread", "discussion", "question"),

    ),

    (

        "educational_guide",

        (

            "how to",

            "guide",

            "tutorial",

            "what is",

            "explained",

            "definition",

            "overview",

        ),

    ),

]





def normalize_url(raw_url: str) -> str:

    raw_url = (raw_url or "").strip()

    if not raw_url:

        return ""

    if not urlsplit(raw_url).scheme:

        raw_url = "https://" + raw_url

    parts = urlsplit(raw_url)

    host = (parts.hostname or "").lower()

    if not host:

        return raw_url

    if host.startswith("www."):

        host = host[4:]

    tracking = {"gclid", "fbclid", "ref", "source"}

    query = "&".join(

        item

        for item in (parts.query.split("&") if parts.query else [])

        if not item.split("=", 1)[0].lower().startswith("utm_")

        and item.split("=", 1)[0].lower() not in tracking

    )

    path = re.sub(r"/{2,}", "/", parts.path or "/")

    if path != "/":

        path = path.rstrip("/")

    return urlunsplit((parts.scheme.lower() or "https", host, path, query, ""))





def is_grounding_redirect(url: str) -> bool:

    return "vertexaisearch.cloud.google.com" in url and "grounding-api-redirect" in url





def get_domain(url: str) -> str:

    host = (urlsplit(url).hostname or "").lower()

    if host.startswith("www."):

        host = host[4:]

    if not host:

        return "unknown"

    if tldextract:

        extracted = tldextract.extract(host)

        if extracted.domain and extracted.suffix:

            return f"{extracted.domain}.{extracted.suffix}"

    return host





def resolve_citation(raw_url: str, title: str = "") -> tuple[str, str]:

    """

    Returns (clean_url, domain).

    Resolves Google Search grounding redirects to their real domain

    using the title when available.

    """

    clean_url = normalize_url(raw_url)

    if is_grounding_redirect(clean_url):

        title_stripped = (title or "").strip().lower()

        if tldextract:

            extracted = tldextract.extract(title_stripped)

            if extracted.domain and extracted.suffix:

                domain = f"{extracted.domain}.{extracted.suffix}"

                resolved_url = (

                    title_stripped

                    if title_stripped.startswith("http")

                    else f"https://{title_stripped}"

                )

                return resolved_url, domain

    domain = get_domain(clean_url)

    return clean_url, domain





def classify_source(url: str, title: str = "", domain: str = "") -> str:

    text = f"{url} {domain} {title}".lower()

    for label, terms in SOURCE_TYPE_RULES:

        if any(term in text for term in terms):

            return label

    return "other_or_unclassified"





def classify_content_type(title: str, content: str, url: str) -> str:

    text = f"{title} {url} {(content or '')[:2500]}".lower()

    for label, terms in CONTENT_TYPE_RULES:

        if any(term in text for term in terms):

            return label

    return "general_article_or_page"





def content_signals(content: str, prompt: str = "") -> dict[str, Any]:

    text = re.sub(r"\s+", " ", content or "").strip()

    lower = text.lower()

    words = re.findall(r"\b[\w'-]+\b", text)

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    numbers = re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text)

    bullets = len(re.findall(r"(?:^|\s)(?:[-*•]|\d+[.)])\s+", content or ""))

    stopwords = {

        "what",

        "where",

        "when",

        "which",

        "with",

        "from",

        "that",

        "this",

        "does",

        "have",

        "about",

        "into",

    }

    terms = {

        w

        for w in re.findall(r"[a-zA-Z0-9]{3,}", (prompt or "").lower())

        if w not in stopwords

    }

    content_terms = set(re.findall(r"[a-zA-Z0-9]{3,}", lower))

    overlap = round(len(terms & content_terms) / max(1, len(terms)), 3)

    return {

        "content_available": bool(text),

        "word_count": len(words),

        "sentence_count": len(sentences),

        "number_or_statistic_count": len(numbers),

        "list_or_bullet_count": bullets,

        "has_comparison_language": any(

            x in lower

            for x in (

                "compared with",

                "versus",

                " vs ",

                "pros and cons",

                "alternatives",

            )

        ),

        "has_methodology_language": any(

            x in lower

            for x in (

                "methodology",

                "we measured",

                "sample size",

                "methods",

                "data were collected",

            )

        ),

        "has_firsthand_or_original_evidence_language": any(

            x in lower

            for x in (

                "our study",

                "we tested",

                "our survey",

                "we analyzed",

                "original data",

                "case study",

            )

        ),

        "has_freshness_language": bool(

            re.search(r"\b(202[4-9]|updated|last updated|published on)\b", lower)

        ),

        "has_definitional_language": any(

            x in lower

            for x in (

                " is defined as ",

                " refers to ",

                " means ",

                "what is ",

            )

        ),

        "query_term_overlap": overlap,

    }





def as_list(value: Any) -> list:

    return value if isinstance(value, list) else []





# ---------------------------------------------------------------------------

# Individual Prompt Analysis

# ---------------------------------------------------------------------------





def analyze_single_prompt(

    prompt_record: dict[str, Any],

    content_map: dict[str, str],

) -> dict[str, Any]:

    """

    Analyzes all AI responses and citations for a single prompt.

    """

    prompt_id = prompt_record["prompt_id"]

    prompt_text = prompt_record["prompt_text"]

    intent_category = prompt_record.get("intent_category") or ""

    responses = prompt_record["responses"]

    scores = prompt_record["scores"]



    avg_score = (

        round(sum(scores) / len(scores), 1) if scores else 0.0

    )



    scores_by_model: dict[str, list[float]] = defaultdict(list)

    for r in responses:

        m = r.get("model") or "unknown"

        s = r.get("score")

        if s is not None:

            try:

                scores_by_model[m].append(float(s))

            except (ValueError, TypeError):

                pass



    model_score_map = {

        m: round(sum(v) / len(v), 1)

        for m, v in scores_by_model.items()

        if v

    }



    models_seen = set()

    category_counts = Counter()

    format_counts = Counter()

    model_citations = Counter()

    domain_counts = Counter()

    domain_models: dict[str, set[str]] = defaultdict(set)

    url_counts = Counter()

    url_models: dict[str, set[str]] = defaultdict(set)

    url_details: dict[str, dict[str, Any]] = {}



    all_citations = []

    combined_content = []



    for resp in responses:

        model = str(resp.get("model") or "unknown")

        models_seen.add(model)

        citations = as_list(resp.get("citations")) or as_list(

            resp.get("sources")

        )



        for item in citations:

            item = {"url": item} if isinstance(item, str) else item

            if not isinstance(item, dict):

                continue

            raw_url = str(

                item.get("url")

                or item.get("source_url")

                or item.get("href")

                or ""

            )

            title = str(

                item.get("title")

                or item.get("page_title")

                or "")

            url, domain = resolve_citation(raw_url, title)

            if not url and not domain:

                continue

            if not url:

                url = f"https://{domain}"



            inline = str(

                item.get("content")

                or item.get("text")

                or item.get("excerpt")

                or ""

            )

            content = (

                inline

                or str(

                    content_map.get(raw_url)

                    or content_map.get(url)

                    or ""

                )

            )

            if content:

                combined_content.append(content)



            source_type = classify_source(url, title, domain)

            content_type = classify_content_type(title, content, url)



            category_counts[source_type] += 1

            format_counts[content_type] += 1

            model_citations[model] += 1

            domain_counts[domain] += 1

            domain_models[domain].add(model)

            url_counts[url] += 1

            url_models[url].add(model)



            if url not in url_details:

                url_details[url] = {

                    "url": url,

                    "domain": domain,

                    "title": title,

                    "source_type": source_type,

                    "content_type": content_type,

                    "citation_count": 0,

                    "models": set(),

                }

            url_details[url]["citation_count"] += 1

            url_details[url]["models"].add(model)

            all_citations.append(url)



    total_citations = len(all_citations)

    denominator = max(1, total_citations)



    top_domains = [

        {

            "domain": dom,

            "citation_count": cnt,

            "models_citing": sorted(domain_models[dom]),

            "model_count": len(domain_models[dom]),

        }

        for dom, cnt in domain_counts.most_common()
        if dom not in NON_SOURCE_DOMAINS

    ][:TOP_DOMAIN_COUNT]



    top_sources = [

        {

            "url": d["url"],

            "domain": d["domain"],

            "title": d["title"],

            "source_type": d["source_type"],

            "content_type": d["content_type"],

            "citation_count": d["citation_count"],

            "models": sorted(d["models"]),

            "model_count": len(d["models"]),

        }

        for d in sorted(

            url_details.values(),

            key=lambda x: (-x["citation_count"], -len(x["models"]), x["url"]),

        )[:15]

    ]



    # Sources cited across multiple AI models for this specific prompt

    multi_model_sources = [

        s

        for s in top_sources

        if s["model_count"] >= 2

    ]



    source_type_summary = [

        {

            "source_type": k,

            "citation_count": v,

            "share_percent": round(v / denominator * 100, 1),

        }

        for k, v in category_counts.most_common()

    ]



    content_type_summary = [

        {

            "content_type": k,

            "citation_count": v,

            "share_percent": round(v / denominator * 100, 1),

        }

        for k, v in format_counts.most_common()

    ]



    model_summary = [

        {

            "model": k,

            "citation_count": v,

            "share_percent": round(v / denominator * 100, 1),

        }

        for k, v in model_citations.most_common()

    ]



    # Determine dominant source ecosystem for this prompt

    academic_share = sum(

        v

        for k, v in category_counts.items()

        if k in ("research_or_academic", "government_or_public_sector")

    ) / denominator

    vendor_share = category_counts.get("commercial_or_vendor", 0) / denominator

    analyst_share = category_counts.get("industry_analyst_or_market_report", 0) / denominator

    review_share = category_counts.get("review_or_directory", 0) / denominator



    if academic_share >= 0.35:

        dominant_ecosystem = "academic_research"

    elif vendor_share >= 0.25:

        dominant_ecosystem = "commercial_vendor"

    elif analyst_share >= 0.15:

        dominant_ecosystem = "industry_market_intelligence"

    elif review_share >= 0.20:

        dominant_ecosystem = "review_directory"

    else:

        dominant_ecosystem = "mixed_or_general"



    # Aggregated text signals

    signals = content_signals(

        " ".join(combined_content[:10]), prompt=prompt_text

    )



    # Prompt-specific opportunities

    prompt_opportunities = []

    if source_type_summary:

        top_cat = source_type_summary[0]

        prompt_opportunities.append({

            "observation": (

                f"AI models predominantly cite '{top_cat['source_type']}' "

                f"({top_cat['share_percent']}%) for this question."

            ),

            "suggested_next_step": (

                f"Create authority content structured for the {top_cat['source_type']} ecosystem. "

                f"Top citing models: {', '.join(sorted(models_seen))}."

            ),

        })



    if multi_model_sources:

        lead_domain = multi_model_sources[0]["domain"]

        prompt_opportunities.append({

            "observation": (

                f"'{lead_domain}' achieved consensus citations across multiple AI models "

                f"({', '.join(multi_model_sources[0]['models'])})."

            ),

            "suggested_next_step": (

                f"Inspect {lead_domain}'s cited content structure on this topic to identify "

                "the factual coverage depth models expect."

            ),

        })



    return {

        "prompt_id": str(prompt_id),

        "prompt_text": prompt_text,

        "intent_category": intent_category,

        "average_score": avg_score,

        "scores_by_model": model_score_map,

        "response_count": len(responses),

        "citation_count": total_citations,

        "unique_domain_count": len(domain_counts),

        "unique_url_count": len(url_counts),

        "models_analyzed": sorted(models_seen),

        "model_breakdown": model_summary,

        "dominant_ecosystem": dominant_ecosystem,

        "top_domains": top_domains,

        "top_sources": top_sources,

        "multi_model_sources": multi_model_sources,

        "source_type_summary": source_type_summary,

        "content_type_summary": content_type_summary,

        "observable_signals": signals,

        "publishing_opportunities": prompt_opportunities,

    }





# ---------------------------------------------------------------------------

# Claude Haiku prompt-level analysis
# ---------------------------------------------------------------------------


def prepare_response_evidence(responses: list[dict[str, Any]], content_map: dict[str, str]) -> list[dict[str, Any]]:
    """Prepare bounded observable AImpact evidence for Claude Haiku."""
    evidence = []
    for response in responses:
        model = str(response.get("model") or "unknown")
        response_text = str(response.get("response") or response.get("answer") or response.get("content") or response.get("text") or "")
        item = {"model": model, "score": response.get("score"), "response": response_text[:6000], "citations": []}
        citations = as_list(response.get("citations")) or as_list(response.get("sources"))
        for citation in citations[:30]:
            citation = {"url": citation} if isinstance(citation, str) else citation
            if not isinstance(citation, dict):
                continue
            raw_url = str(citation.get("url") or citation.get("source_url") or citation.get("href") or "")
            title = str(citation.get("title") or citation.get("page_title") or "")
            url, domain = resolve_citation(raw_url, title)
            inline = str(citation.get("content") or citation.get("text") or citation.get("excerpt") or "")
            content = inline or str(content_map.get(raw_url) or content_map.get(url) or "")
            item["citations"].append({"url": url, "domain": domain, "title": title, "source_type": classify_source(url, title, domain), "content_type": classify_content_type(title, content, url), "excerpt": content[:1500]})
        evidence.append(item)
    return evidence


def _parse_haiku_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


def analyze_prompt_with_haiku(prompt_analysis: dict[str, Any]) -> dict[str, Any]:
    """Analyze ONE low-scoring prompt with Claude Haiku."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic(api_key=api_key)
    evidence = {
        "prompt": prompt_analysis["prompt_text"],
        "average_score": prompt_analysis["average_score"],
        "scores_by_model": prompt_analysis["scores_by_model"],
        "models_analyzed": prompt_analysis["models_analyzed"],
        "top_domains": prompt_analysis["top_domains"],
        "top_sources": prompt_analysis["top_sources"],
        "multi_model_sources": prompt_analysis["multi_model_sources"],
        "source_type_summary": prompt_analysis["source_type_summary"],
        "content_type_summary": prompt_analysis["content_type_summary"],
        "observable_signals": prompt_analysis["observable_signals"],
        "responses": prompt_analysis.get("responses", []),
    }
    analysis_prompt = f"""
You are the Source Intelligence analyst for an AI visibility platform.

Analyze ONE low-scoring AI search prompt.

Use only the supplied AImpact evidence. Do not claim to know hidden reasoning
inside ChatGPT, Claude, Gemini, GoogleAI, or any other model. Do not say a model
trusts, likes, or prefers a source unless the evidence directly demonstrates a
repeatable observable pattern.

Focus on what was cited, why those sources were likely useful based on observable
evidence, cross-model citation patterns, content characteristics, and realistic
publishing opportunities. Clearly distinguish observations from interpretations.
Do not invent facts about cited websites.

Return ONLY valid JSON with exactly these keys:
{{
  "source_selection_observations": [],
  "citation_patterns": [],
  "cross_model_observations": [],
  "content_characteristics": [],
  "likely_selection_factors": [],
  "publishing_opportunities": [],
  "priority_actions": []
}}

AIMPACT EVIDENCE:
{json.dumps(evidence, indent=2, ensure_ascii=False)}
"""
    response = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=HAIKU_MAX_TOKENS,

        messages=[{"role": "user", "content": analysis_prompt}],
    )
    text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()
    parsed = _parse_haiku_json(text)
    if parsed:
        return parsed
    return {
        "source_selection_observations": [],
        "citation_patterns": [],
        "cross_model_observations": [],
        "content_characteristics": [],
        "likely_selection_factors": [],
        "publishing_opportunities": [],
        "priority_actions": [],
        "raw_haiku_response": text,
    }


def attach_haiku_analyses(prompt_analyses: list[dict[str, Any]]) -> None:
    """Run Claude Haiku on every low-scoring prompt, up to 12 concurrently."""
    if not prompt_analyses:
        return
    print(f"\nRunning Claude Haiku analysis for {len(prompt_analyses)} low-scoring prompts (max concurrent: {MAX_CONCURRENT})...")
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {executor.submit(analyze_prompt_with_haiku, prompt): prompt for prompt in prompt_analyses}
        completed = 0
        for future in as_completed(futures):
            prompt = futures[future]
            completed += 1
            try:
                prompt["haiku_analysis"] = future.result()
                print(f"  [{completed}/{len(prompt_analyses)}] Prompt {prompt['prompt_id']} analyzed")
            except Exception as exc:
                prompt["haiku_analysis"] = {"error": str(exc)}
                print(f"  [{completed}/{len(prompt_analyses)}] Prompt {prompt['prompt_id']} FAILED: {exc}")


# Cross-Prompt Intelligence Synthesis

# ---------------------------------------------------------------------------





def analyze_cross_prompt_insights(

    prompt_analyses: list[dict[str, Any]],

    company_name: str | None = None,

    company_url: str | None = None,

) -> dict[str, Any]:

    """

    Synthesizes aggregated patterns across all analyzed low-scoring prompts.

    """

    total_prompts = len(prompt_analyses)

    if not total_prompts:

        return {}



    # 1. Common Domains across prompts

    domain_prompt_map: dict[str, set[str]] = defaultdict(set)

    domain_citations_total = Counter()

    domain_models_total: dict[str, set[str]] = defaultdict(set)



    for p in prompt_analyses:

        pid = p["prompt_id"]

        for dom in p["top_domains"]:

            d_name = dom["domain"]

            domain_prompt_map[d_name].add(pid)

            domain_citations_total[d_name] += dom["citation_count"]

            domain_models_total[d_name].update(dom.get("models_citing", []))



    common_domains = []

    for dom, p_set in sorted(

        domain_prompt_map.items(),

        key=lambda x: (-len(x[1]), -domain_citations_total[x[0]]),

    )[:20]:

        common_domains.append({

            "domain": dom,

            "total_citations": domain_citations_total[dom],

            "prompts_cited_in_count": len(p_set),

            "prompts_cited_in_percent": round(

                len(p_set) / total_prompts * 100, 1

            ),

            "models_citing": sorted(domain_models_total[dom]),

        })



    # 2. Source Ecosystem Breakdown across question types

    ecosystems: dict[str, list[dict[str, Any]]] = {

        "academic_research": [],

        "commercial_vendor": [],

        "industry_market_intelligence": [],

        "review_directory": [],

        "mixed_or_general": [],

    }



    for p in prompt_analyses:

        eco = p.get("dominant_ecosystem", "mixed_or_general")

        ecosystems.setdefault(eco, []).append({

            "prompt_id": p["prompt_id"],

            "prompt_text": p["prompt_text"],

            "average_score": p["average_score"],

            "top_domains": [d["domain"] for d in p["top_domains"][:5]],

        })



    # 3. Multi-Model Consensus Domains (domains cited by >=3 models in >=2 prompts)

    consensus_domains = [

        d

        for d in common_domains

        if len(d["models_citing"]) >= 3 and d["prompts_cited_in_count"] >= 2

    ]



    # 4. Brand Visibility Check across low-scoring queries

    customer_domain = get_domain(company_url) if company_url else "unknown"

    brand_citations = 0

    prompts_with_brand = []

    prompts_without_brand = []



    for p in prompt_analyses:

        brand_cited_here = any(

            d["domain"] == customer_domain

            for d in p["top_domains"]

        )

        if brand_cited_here:

            prompts_with_brand.append(p["prompt_id"])

            for d in p["top_domains"]:

                if d["domain"] == customer_domain:

                    brand_citations += d["citation_count"]

        else:

            prompts_without_brand.append({

                "prompt_id": p["prompt_id"],

                "prompt_text": p["prompt_text"],

                "average_score": p["average_score"],

            })



    brand_visibility = {

        "customer_domain": customer_domain,

        "customer_company": company_name or "Unknown",

        "total_brand_citations": brand_citations,

        "prompts_citing_brand_count": len(prompts_with_brand),

        "prompts_missing_brand_count": len(prompts_without_brand),

        "prompts_missing_brand": prompts_without_brand[:10],

    }



    # 5. Strategic Publishing Opportunities

    strategic_opportunities = []



    if common_domains:

        lead = common_domains[0]

        strategic_opportunities.append({

            "scope": "cross_prompt_authority",

            "observation": (

                f"'{lead['domain']}' is the universal authority across low-scoring queries, "

                f"appearing in {lead['prompts_cited_in_count']}/{total_prompts} prompts "

                f"({lead['prompts_cited_in_percent']}%) across models {', '.join(lead['models_citing'])}."

            ),

            "suggested_strategy": (

                f"Analyze the technical depth and citation structure of {lead['domain']} "

                "to benchmark your brand's technical specifications and documentation."

            ),

        })



    if ecosystems.get("academic_research"):

        strategic_opportunities.append({

            "scope": "ecosystem_academic",

            "observation": (

                f"{len(ecosystems['academic_research'])} queries fall into the Academic / Research ecosystem. "

                "AI models heavily weight PubMed, NIH, and peer-reviewed journals for these questions."

            ),

            "suggested_strategy": (

                "For research queries, publish peer-reviewed whitepapers, protocol validations, "

                "and open-access technical notes rather than promotional sales copy."

            ),

        })



    if ecosystems.get("commercial_vendor"):

        strategic_opportunities.append({

            "scope": "ecosystem_commercial",

            "observation": (

                f"{len(ecosystems['commercial_vendor'])} queries trigger Commercial / Vendor sources. "

                "AI models cite product specification pages, platform catalogs, and service overviews."

            ),

            "suggested_strategy": (

                "For commercial service queries, publish transparent capability matrices, "

                "instrument spec sheets, and sample-to-answer service workflows."

            ),

        })



    if brand_visibility["prompts_missing_brand_count"] > 0:

        strategic_opportunities.append({

            "scope": "brand_gap",

            "observation": (

                f"Your brand is completely unmentioned across {brand_visibility['prompts_missing_brand_count']} "

                "low-scoring queries where competitors and authority domains are cited."

            ),

            "suggested_strategy": (

                "Prioritize targeted content pieces specifically answering the unanswered queries "

                f"listed in 'prompts_missing_brand'."

            ),

        })



    return {

        "common_domains": common_domains,

        "source_ecosystem_breakdown": ecosystems,

        "multi_model_consensus_domains": consensus_domains,

        "brand_visibility": brand_visibility,

        "strategic_publishing_opportunities": strategic_opportunities,

    }





# ---------------------------------------------------------------------------

# Master Dataset Intelligence Orchestrator

# ---------------------------------------------------------------------------





def analyze_campaign_dataset(

    data: dict[str, Any],

    score_threshold: float = 70.0,

    company_url: str | None = None,

) -> dict[str, Any]:

    """

    Main analysis pipeline:

    1. Campaign summary

    2. Low-scoring summary

    3. Individual prompt analysis for every low-scoring prompt

    4. Cross-prompt insights synthesis

    """

    responses = as_list(data.get("responses"))

    content_map = (

        data.get("source_content")

        if isinstance(data.get("source_content"), dict)

        else {}

    )

    campaign_info = data.get("campaign") or {}

    project_info = data.get("project") or {}



    active_company_url = (

        company_url

        or project_info.get("company_url")

        or os.getenv("COMPANY_URL")

    )

    if not active_company_url and "alamar" in (project_info.get("company_name") or "").lower():

        active_company_url = "https://alamarbio.com"



    # Group responses by prompt_id

    prompts: dict[str, dict[str, Any]] = {}

    for r in responses:

        if not isinstance(r, dict):

            continue

        pid = str(r.get("prompt_id") or "")

        ptext = str(r.get("prompt") or r.get("prompt_text") or "").strip()

        if not pid or not ptext:

            continue



        if pid not in prompts:

            prompts[pid] = {

                "prompt_id": pid,

                "prompt_text": ptext,

                "intent_category": str(r.get("intent_category") or ""),

                "scores": [],

                "responses": [],

            }



        s = r.get("score")

        if s is not None:

            try:

                prompts[pid]["scores"].append(float(s))

            except (ValueError, TypeError):

                pass

        prompts[pid]["responses"].append(r)



    # Compute average score per prompt

    for p in prompts.values():

        scores = p["scores"]

        p["average_score"] = (

            round(sum(scores) / len(scores), 1) if scores else 0.0

        )



    # Filter to low-scoring prompts

    low_scoring_prompts = [

        p for p in prompts.values() if p["average_score"] < score_threshold

    ]

    low_scoring_prompts.sort(key=lambda x: x["average_score"])



    # Only prompts strictly below the threshold are analyzed.
    # There is intentionally no fallback to higher-scoring prompts.

    # 1. Individual Prompt Analysis

    prompt_analyses = [

        analyze_single_prompt(p, content_map)

        for p in low_scoring_prompts

    ]



    # 2. Claude Haiku analysis for EVERY low-scoring prompt
    attach_haiku_analyses(prompt_analyses)

    # 3. Cross-Prompt Insights

    cross_insights = analyze_cross_prompt_insights(

        prompt_analyses=prompt_analyses,

        company_name=project_info.get("company_name"),

        company_url=active_company_url,

    )



    # 3. Low-Scoring Summary Aggregation

    total_low_responses = sum(p["response_count"] for p in prompt_analyses)

    total_low_citations = sum(p["citation_count"] for p in prompt_analyses)

    all_low_scores = [

        p["average_score"]

        for p in prompt_analyses

        if p["average_score"] is not None

    ]

    avg_low_score = (

        round(sum(all_low_scores) / len(all_low_scores), 1)

        if all_low_scores

        else 0.0

    )



    low_model_counts = Counter()

    low_category_counts = Counter()

    low_format_counts = Counter()

    low_domains = Counter()



    for p in prompt_analyses:

        for m in p["model_breakdown"]:

            low_model_counts[m["model"]] += m["citation_count"]

        for c in p["source_type_summary"]:

            low_category_counts[c["source_type"]] += c["citation_count"]

        for f in p["content_type_summary"]:

            low_format_counts[f["content_type"]] += f["citation_count"]

        for d in p["top_domains"]:

            low_domains[d["domain"]] += d["citation_count"]



    denom = max(1, total_low_citations)

    low_summary = {

        "low_scoring_prompt_count": len(prompt_analyses),

        "low_scoring_response_count": total_low_responses,

        "low_scoring_citation_count": total_low_citations,

        "average_score": avg_low_score,

        "unique_cited_domain_count": len(low_domains),

        "model_breakdown": [

            {

                "model": k,

                "citation_count": v,

                "share_percent": round(v / denom * 100, 1),

            }

            for k, v in low_model_counts.most_common()

        ],

        "top_domains": [

            {"domain": k, "citation_count": v}

            for k, v in low_domains.most_common(15)

        ],

        "source_type_summary": [{

                "source_type": k,

                "citation_count": v,

                "share_percent": round(v / denom * 100, 1),

            }

            for k, v in low_category_counts.most_common()

        ],

        "content_type_summary": [

            {

                "content_type": k,

                "citation_count": v,

                "share_percent": round(v / denom * 100, 1),

            }

            for k, v in low_format_counts.most_common()

        ],

    }



    # 4. Campaign Summary

    campaign_summary = {

        "campaign_id": campaign_info.get("id"),

        "campaign_name": campaign_info.get("name"),

        "company_name": project_info.get("company_name"),

        "company_url": project_info.get("company_url"),

        "region": project_info.get("region"),

        "score_threshold": score_threshold,

        "total_prompts_audited": len(prompts),

        "total_responses_analyzed": len(responses),

        "total_citations_analyzed": sum(

            len(as_list(r.get("citations")) or as_list(r.get("sources")))

            for r in responses

            if isinstance(r, dict)

        ),

    }



    return {

        "report_version": "3.0",
        "analysis_engine": {
            "provider": "Anthropic",
            "model": HAIKU_MODEL,
            "max_concurrent": MAX_CONCURRENT,
        },

        "generated_at_utc": datetime.now(timezone.utc).isoformat(

            timespec="seconds"

        ),

        "scope_note": (

            "Source Intelligence analysis across individual low-scoring prompts "

            "and synthesized cross-prompt authority ecosystems. "

            "Visible citations only; does not infer hidden model reasoning."

        ),

        "campaign_summary": campaign_summary,

        "low_scoring_summary": low_summary,

        "prompt_analysis": prompt_analyses,

        "cross_prompt_insights": cross_insights,

    }





# ---------------------------------------------------------------------------

# LangGraph / Workflow Agent entrypoint

# ---------------------------------------------------------------------------





async def run(state: WorkflowState) -> WorkflowState:

    """

    Run Source Intelligence analysis for the campaign in WorkflowState.

    Connects to AImpact to retrieve campaign data, AI responses, and citations,

    analyzes each low-scoring prompt individually and cross-prompt,

    and stores the resulting ContentGap in state.

    """

    campaign_id = state.campaign_id

    if not campaign_id:

        raise ValueError("WorkflowState must contain a valid campaign_id.")



    dataset = get_campaign_audit_dataset(campaign_id=campaign_id)

    report = analyze_campaign_dataset(

        dataset,

        score_threshold=70.0,

        company_url=state.company_url,

    )



    cross = report.get("cross_prompt_insights", {})

    common_domains = [

        d["domain"] for d in cross.get("common_domains", [])[:10]

    ]



    strategic_opps = cross.get("strategic_publishing_opportunities", [])

    if strategic_opps:

        opportunity_text = strategic_opps[0].get("observation", "")

        angle_text = strategic_opps[0].get("suggested_strategy", "")

    else:

        opportunity_text = "Source intelligence analysis complete."

        angle_text = "Review top cited sources."



    prompt_id = (

        str(state.prompts[0].prompt_id)

        if state.prompts

        else f"campaign-{campaign_id}"

    )

    prompt_text = (

        state.prompts[0].prompt_text

        if state.prompts

        else f"Campaign {campaign_id} Analysis"

    )



    state.gap = ContentGap(

        prompt_id=prompt_id,

        prompt_text=prompt_text,

        candidate_themes=common_domains,

        themes=common_domains,

        target_angle=angle_text,

        opportunity=opportunity_text,

        hypothesis=(

            f"Common cited authority domains across low-scoring queries: "

            f"{', '.join(common_domains[:5])}."

        ),

        content_gap_status="established" if common_domains else "not_established",

    )



    return state





# ---------------------------------------------------------------------------

# CLI entrypoint

# ---------------------------------------------------------------------------





def main() -> None:

    parser = argparse.ArgumentParser(

        description="AImpact Source Intelligence Analyzer: prompt-level & cross-prompt intelligence."

    )

    parser.add_argument(

        "--campaign-id",

        type=int,

        default=None,

        help="AImpact Campaign ID to analyze (defaults to CAMPAIGN_ID env var, or 81)",

    )

    parser.add_argument(

        "--score-threshold",

        type=float,

        default=70.0,

        help="Score threshold for low-scoring prompts (default: 70.0)",

    )

    parser.add_argument(

        "--latest-only",

        action="store_true",

        help="Analyze only the latest audit per prompt/model combination",

    )

    parser.add_argument(

        "--company-url",

        default=None,

        help="Optional company website URL to assess brand visibility",

    )

    parser.add_argument(

        "--output",

        default="source_intelligence_report.json",

        help="Path for output report JSON (default: source_intelligence_report.json)",

    )

    args = parser.parse_args()



    campaign_id = (

        args.campaign_id

        or int(os.getenv("CAMPAIGN_ID", "81"))

    )



    print(f"Connecting to AImpact database for Campaign {campaign_id}...")

    dataset = get_campaign_audit_dataset(

        campaign_id=campaign_id,

        score_threshold=None,  # Load all so we can categorize by prompt score

        latest_only=args.latest_only,

    )



    campaign_name = dataset.get("campaign", {}).get("name", "Unknown")

    company_name = dataset.get("project", {}).get("company_name", "Unknown")

    print(f"Campaign: {campaign_name} | Company: {company_name}")

    print(f"Score threshold for low-scoring queries: < {args.score_threshold}")



    report = analyze_campaign_dataset(

        dataset,

        score_threshold=args.score_threshold,

        company_url=args.company_url,

    )

    output = Path(args.output)

    output.write_text(

        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"

    )



    print(f"\nReport written to: {output.resolve()}")



    camp_sum = report["campaign_summary"]

    low_sum = report["low_scoring_summary"]

    prompt_list = report["prompt_analysis"]

    cross = report["cross_prompt_insights"]



    print("\n================================================================================")

    print("CAMPAIGN & LOW-SCORING SUMMARY")

    print("================================================================================")

    print(f"Total Prompts Audited: {camp_sum['total_prompts_audited']}")

    print(f"Low-Scoring Prompts Analyzed: {low_sum['low_scoring_prompt_count']}")

    print(f"Average Score (Low-Scoring): {low_sum['average_score']}")

    print(f"Total Low-Scoring Citations: {low_sum['low_scoring_citation_count']}")

    print(f"Unique Domains Cited: {low_sum['unique_cited_domain_count']}")



    print("\nModel Citations Breakdown (Low-Scoring):")

    for m in low_sum["model_breakdown"]:

        print(f"  {m['model']}: {m['citation_count']} citations ({m['share_percent']}%)")



    print("\n================================================================================")

    print("INDIVIDUAL LOW-SCORING PROMPT ANALYSIS")

    print("================================================================================")

    for p in prompt_list:

        print(f"\n[Prompt {p['prompt_id']}] (Score: {p['average_score']}) | Ecosystem: {p['dominant_ecosystem']}")

        print(f'  Question: "{p["prompt_text"]}"')

        print(f"  Citations: {p['citation_count']} across {p['unique_domain_count']} domains ({len(p['models_analyzed'])} models)")

        top_3 = ", ".join(f"{d['domain']} ({d['citation_count']})" for d in p["top_domains"][:5])

        print(f"  Top Domains: {top_3 or 'None'}")

        if p["multi_model_sources"]:

            mm_lead = p["multi_model_sources"][0]

            print(f"  Consensus Source: {mm_lead['domain']} (cited by {', '.join(mm_lead['models'])})")

        if p["publishing_opportunities"]:

            print(f"  Opportunity: {p['publishing_opportunities'][0]['observation']}")



    print("\n================================================================================")

    print("CROSS-PROMPT INSIGHTS & SOURCE ECOSYSTEMS")

    print("================================================================================")

    print("\nTop Universal Authority Domains (across prompts):")

    for d in cross.get("common_domains", [])[:8]:

        print(

            f"  * {d['domain']}: cited in {d['prompts_cited_in_count']}/{low_sum['low_scoring_prompt_count']} prompts "

            f"({d['prompts_cited_in_percent']}%) | Total citations: {d['total_citations']}"

        )



    ecos = cross.get("source_ecosystem_breakdown", {})

    print("\nSource Ecosystems by Query Type:")

    for eco_name, prompts_in_eco in ecos.items():

        if prompts_in_eco:

            print(f"  * {eco_name.upper()} ({len(prompts_in_eco)} queries):")

            for p_sub in prompts_in_eco[:3]:

                print(f'      - "{p_sub["prompt_text"][:60]}..." (Top: {", ".join(p_sub["top_domains"])})')



    b_vis = cross.get("brand_visibility", {})

    if b_vis:

        print(f"\nBrand Visibility ({b_vis['customer_company']} / {b_vis['customer_domain']}):")

        print(f"  * Brand Cited In: {b_vis['prompts_citing_brand_count']} prompts ({b_vis['total_brand_citations']} citations)")

        print(f"  * Brand Completely Missing In: {b_vis['prompts_missing_brand_count']} low-scoring prompts")



    print("\nStrategic Opportunities:")

    for opp in cross.get("strategic_publishing_opportunities", []):

        print(f"  * [{opp['scope']}] {opp['observation']}")

        print(f"    Action: {opp['suggested_strategy']}")





if __name__ == "__main__":

    main()

