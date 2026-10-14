
"""Build Agent 5 input from Agent 1–4 outputs and run the Writer Agent.

Run from the project root:
    python .\scripts\test_writer_from_pipeline.py
"""

from __future__ import annotations


import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import os
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")



from aca.models.writer import (
    WriterEvidence,
    WriterInput,
    WriterSectionInput,
)
from aca.agents.writer import WriterAgent



# ---------------------------------------------------------------------
# Paths and JSON loading
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "agent5_output.json"


def load_json(candidates: list[str]) -> dict[str, Any]:
    """Load the first existing filename from the candidate list."""
    for filename in candidates:
        path = ROOT / filename
        if path.exists():
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data, dict):
                raise ValueError(f"{path.name} must contain a JSON object.")

            print(f"[OK] Loaded {path.name}")
            return data

    tried = ", ".join(candidates)
    raise FileNotFoundError(
        f"Could not find any of these files in {ROOT}:\n{tried}"
    )


def first_nonempty(*values: Any, default: str = "") -> str:
    """Return the first nonempty string."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def normalize_url(url: Any) -> str:
    """Normalize a URL for matching without changing the stored source URL."""
    if not isinstance(url, str) or not url.strip():
        return ""

    value = url.strip()
    parsed = urlparse(value if "://" in value else f"https://{value}")
    path = parsed.path.rstrip("/")
    return f"{parsed.netloc.lower()}{path}".lower()


# ---------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------

def map_status(status: Any) -> str:
    """Map Agent 3 statuses to the statuses accepted by WriterEvidence."""
    value = str(status or "").strip().lower()

    mapping = {
        "verified": "verified",
        "model_supported": "model_supported",
        "model-supported": "model_supported",
        "supported": "model_supported",
        "contradicted": "contradicted",
        "not_found": "not_found",
        "not found": "not_found",
        "inconclusive": "inconclusive",
        "needs_human_review": "inconclusive",
        "needs human review": "inconclusive",
        "unavailable": "unavailable",
    }

    # Unknown or missing statuses must not be promoted to verified.
    return mapping.get(value, "inconclusive")


def conservative_status(statuses: list[str]) -> str:
    """Choose a conservative status when multiple claims share evidence."""
    mapped = [map_status(status) for status in statuses if status]

    if not mapped:
        return "inconclusive"

    # Conflicting evidence must be reviewed rather than silently accepted.
    if "contradicted" in mapped:
        return "contradicted"

    if "inconclusive" in mapped:
        return "inconclusive"

    if "not_found" in mapped:
        return "not_found"

    if "unavailable" in mapped:
        return "unavailable"

    if "model_supported" in mapped:
        return "model_supported"

    if all(status == "verified" for status in mapped):
        return "verified"

    return "inconclusive"


# ---------------------------------------------------------------------
# Company information
# ---------------------------------------------------------------------

def find_company_url(agent1: dict, agent3: dict, agent2: dict) -> str | None:
    """Find a company URL from available agent output fields."""
    candidates: list[Any] = []

    for data in (agent1, agent3):
        handoff = data.get("agent4_handoff")
        if isinstance(handoff, dict):
            company = handoff.get("company")
            if isinstance(company, dict):
                candidates.extend(
                    [
                        company.get("website"),
                        company.get("url"),
                        company.get("website_url"),
                    ]
                )

        company = data.get("company")
        if isinstance(company, dict):
            candidates.extend(
                [
                    company.get("website"),
                    company.get("url"),
                    company.get("website_url"),
                ]
            )

    candidates.extend(
        [
            agent1.get("company_url"),
            agent1.get("website"),
            agent3.get("company_url"),
            agent3.get("website"),
        ]
    )

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    # Agent 2 may contain ranked source URLs. Use one only when its domain
    # matches the customer name, rather than choosing an arbitrary source.
    customer_name = first_nonempty(
        agent1.get("customer_name"),
        agent1.get("company_name"),
    )
    expected = re.sub(r"[^a-z0-9]", "", customer_name.lower())

    for source in as_list(agent2.get("ranked_sources")):
        if not isinstance(source, dict):
            continue

        url = first_nonempty(
            source.get("url"),
            source.get("final_url"),
            source.get("source_url"),
        )
        if not url:
            continue

        domain = re.sub(
            r"[^a-z0-9]",
            "",
            urlparse(url if "://" in url else f"https://{url}").netloc.lower(),
        )

        if expected and expected in domain:
            return url

    return None


# ---------------------------------------------------------------------
# Agent 3 evidence mapping
# ---------------------------------------------------------------------

def build_evidence(agent3: dict[str, Any]) -> list[WriterEvidence]:
    """Associate raw evidence with Agent 3 findings and verified facts.

    Agent 3 raw evidence records do not necessarily have their own claim or
    verification status. Statuses therefore come from linked findings/facts,
    never from the existence of a quote alone.
    """
    raw_evidence = as_list(agent3.get("evidence"))
    findings = as_list(agent3.get("relevant_findings"))
    facts = as_list(agent3.get("verified_facts"))

    evidence_by_id: dict[str, dict[str, Any]] = {}
    evidence_urls: dict[str, str] = {}

    for index, item in enumerate(raw_evidence, start=1):
        if not isinstance(item, dict):
            continue

        evidence_id = first_nonempty(
            item.get("evidence_id"),
            item.get("id"),
            default=f"ev_auto_{index:03d}",
        )

        evidence_by_id[evidence_id] = item
        evidence_urls[evidence_id] = first_nonempty(item.get("url"))

    # Each evidence ID gets claims and statuses from facts that explicitly
    # reference that ID.
    facts_by_evidence: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        if not isinstance(fact, dict):
            continue

        for evidence_id in as_list(fact.get("evidence_ids")):
            facts_by_evidence.setdefault(str(evidence_id), []).append(fact)

    # Findings often refer to source URLs rather than evidence IDs.
    findings_by_url: dict[str, list[dict[str, Any]]] = {}
    for finding in findings:
        if not isinstance(finding, dict):
            continue

        for url in as_list(finding.get("source_urls")):
            normalized = normalize_url(url)
            if normalized:
                findings_by_url.setdefault(normalized, []).append(finding)

    result: list[WriterEvidence] = []

    for evidence_id, item in evidence_by_id.items():
        url = first_nonempty(item.get("url")) or None
        quote = first_nonempty(item.get("quote")) or None
        context = first_nonempty(item.get("context"))
        page_title = first_nonempty(item.get("page_title"))

        linked_facts = facts_by_evidence.get(evidence_id, [])

        linked_findings: list[dict[str, Any]] = []
        normalized_url = normalize_url(url)
        if normalized_url:
            linked_findings = findings_by_url.get(normalized_url, [])

        claims: list[str] = []
        statuses: list[str] = []

        # Explicitly linked verified facts are the strongest structured
        # claim source. Keep their wording boundaries visible to the writer.
        for fact in linked_facts:
            statement = first_nonempty(fact.get("statement"))
            boundary = first_nonempty(fact.get("wording_boundary"))

            if statement:
                claim = statement
                if boundary:
                    claim += f" Wording boundary: {boundary}"
                claims.append(claim)

            if fact.get("status"):
                statuses.append(str(fact["status"]))

        # Use findings when they point to this same source URL.
        for finding in linked_findings:
            claim = first_nonempty(finding.get("claim"))
            if claim:
                claims.append(claim)

            if finding.get("verification_status"):
                statuses.append(str(finding["verification_status"]))

        # Deduplicate while preserving order.
        claims = list(dict.fromkeys(claims))

        if claims:
            claim_text = " | ".join(claims)
        else:
            # This is descriptive source context, not a verified claim.
            claim_text = first_nonempty(
                context,
                quote,
                page_title,
                default="Source evidence available; claim not yet linked to a verified finding.",
            )
            statuses.append("inconclusive")

        result.append(
            WriterEvidence(
                evidence_id=evidence_id,
                claim=claim_text,
                status=conservative_status(statuses),
                evidence_quote=quote,
                source_url=url,
            )
        )

    # Include verified facts whose evidence IDs do not exist in the raw
    # evidence array. They remain available to the writer, but a fact with
    # no linked evidence is downgraded to inconclusive.
    for fact in facts:
        if not isinstance(fact, dict):
            continue

        fact_id = first_nonempty(fact.get("fact_id"))
        statement = first_nonempty(fact.get("statement"))
        evidence_ids = [
            str(value) for value in as_list(fact.get("evidence_ids"))
        ]

        if not statement:
            continue

        missing_ids = [
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id not in evidence_by_id
        ]

        if not evidence_ids or missing_ids:
            fact_status = "inconclusive"
        else:
            fact_status = map_status(fact.get("status"))

        # Do not duplicate a fact already attached to an evidence record.
        already_attached = any(
            fact in facts_by_evidence.get(evidence_id, [])
            for evidence_id in evidence_ids
        )
        if already_attached:
            continue

        result.append(
            WriterEvidence(
                evidence_id=fact_id or f"fact_unlinked_{len(result) + 1}",
                claim=statement,
                status=fact_status,
                evidence_quote=None,
                source_url=None,
            )
        )

    return result


# ---------------------------------------------------------------------
# Agent 4 brief mapping
# ---------------------------------------------------------------------

def build_sections(agent4: dict[str, Any]) -> list[WriterSectionInput]:
    sections: list[WriterSectionInput] = []

    for section in as_list(agent4.get("sections")):
        if not isinstance(section, dict):
            continue

        level_value = section.get("level", "H2")

        if isinstance(level_value, int):
            level = level_value
        else:
            match = re.search(r"\d+", str(level_value))
            level = int(match.group()) if match else 2

        level = min(max(level, 1), 6)

        topics = as_list(section.get("topics_and_questions"))
        guidelines = as_list(section.get("guidelines_and_gaps"))

        sections.append(
            WriterSectionInput(
                heading=first_nonempty(
                    section.get("heading"),
                    default="Untitled section",
                ),
                level=level,
                objective=first_nonempty(section.get("section_objective")),
                topics=[
                    str(value).strip()
                    for value in topics
                    if value is not None and str(value).strip()
                ],
                questions_to_answer=[
                    str(value).strip()
                    for value in topics
                    if isinstance(value, str)
                    and "?" in value
                    and value.strip()
                ],
                evidence_ids=[
                    str(value)
                    for value in as_list(section.get("evidence_ids"))
                ],
                guidelines=[
                    str(value).strip()
                    for value in guidelines
                    if value is not None and str(value).strip()
                ],
            )
        )

    return sections


def build_writing_style(agent4: dict[str, Any]) -> dict[str, str]:
    style = agent4.get("writing_style", {})
    if not isinstance(style, dict):
        return {}

    converted: dict[str, str] = {}

    for key, value in style.items():
        if isinstance(value, list):
            converted[key] = ", ".join(str(item) for item in value)
        elif value is not None:
            converted[key] = str(value)

    return converted


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    agent1 = load_json(
        ["agent1_output.json", "agent1_output(1).json"]
    )
    agent2 = load_json(
        ["agent2_output.json", "agent2_output(1).json"]
    )
    agent3 = load_json(
        ["agent3_output.json", "agent3_output(1).json"]
    )
    agent4 = load_json(
        ["agent4_output.json", "agent4_output(1).json"]
    )

    company_name = first_nonempty(
        agent1.get("customer_name"),
        agent1.get("company_name"),
        agent3.get("customer_name"),
        agent3.get("company_name"),
    )

    if not company_name:
        handoff = agent3.get("agent4_handoff", {})
        if isinstance(handoff, dict):
            company = handoff.get("company", {})
            if isinstance(company, dict):
                company_name = first_nonempty(
                    company.get("name"),
                    company.get("company_name"),
                )

    if not company_name:
        raise ValueError(
            "Customer/company name was not found in Agent 1 or Agent 3 output."
        )

    title = first_nonempty(
        agent4.get("recommended_title"),
        default=f"{company_name}: Overview and Capabilities",
    )

    evidence = build_evidence(agent3)
    sections = build_sections(agent4)

    # Agent 3's research gaps and Agent 4's unsupported-claim list both
    # constrain what Agent 5 is allowed to write.
    unsupported_claims = [
        str(item).strip()
        for item in as_list(agent4.get("unsupported_claims_to_avoid"))
        if item is not None and str(item).strip()
    ]

    for key in ("information_still_needed", "research_gaps", "limitations"):
        values = agent3.get(key, [])
        if isinstance(values, list):
            for item in values:
                if isinstance(item, str) and item.strip():
                    unsupported_claims.append(
                        f"Do not fill this research gap with assumptions: {item.strip()}"
                    )
                elif isinstance(item, dict):
                    description = first_nonempty(
                        item.get("description"),
                        item.get("gap"),
                        item.get("question"),
                        item.get("claim"),
                    )
                    if description:
                        unsupported_claims.append(
                            "Do not fill this research gap with assumptions: "
                            + description
                        )

    unsupported_claims = list(dict.fromkeys(unsupported_claims))

    # Add source URLs from Agent 2 to the internal context indirectly through
    # evidence only when Agent 3 already provides the evidence record. Agent 2
    # source discovery alone does not make a claim verified.
    writer_input = WriterInput(
        company_name=company_name,
        company_url=find_company_url(agent1, agent3, agent2),
        title=title,
        meta_description="",
        article_objective=first_nonempty(
            agent4.get("article_objective"),
            default=f"Explain {company_name}'s relevant capabilities accurately.",
        ),
        target_audience=first_nonempty(
            agent4.get("target_audience"),
            default="Readers researching relevant providers and capabilities",
        ),
        search_intent=first_nonempty(agent4.get("search_intent")),
        sections=sections,
        writing_style=build_writing_style(agent4),
        unsupported_claims_to_avoid=unsupported_claims,
        evidence=evidence,
    )

    print("\n--- Agent 5 input summary ---")
    print(f"Company: {writer_input.company_name}")
    print(f"Title: {writer_input.title}")
    print(f"Sections: {len(writer_input.sections)}")
    print(f"Evidence records: {len(writer_input.evidence)}")
    print(f"Restrictions/research gaps: {len(writer_input.unsupported_claims_to_avoid)}")

    status_counts: dict[str, int] = {}
    for item in writer_input.evidence:
        status_counts[item.status] = status_counts.get(item.status, 0) + 1
    print(f"Evidence status counts: {status_counts}")

    print("\nRunning Agent 5...")
    result = WriterAgent().write(writer_input)

    # Save the structured JSON for Agent 6 and downstream processing.
    json_path = ROOT / "agent5_output.json"
    json_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # Save a clean Markdown file for reading and reviewing the blog.
    markdown_path = ROOT / "agent5_article.md"

    article_markdown = result.article_markdown.strip()

    # Ensure the Markdown file has a title, even if the model omitted the H1.
    if not article_markdown.startswith("# "):
        article_markdown = f"# {result.title.strip()}\n\n{article_markdown}"

    markdown_path.write_text(
        article_markdown + "\n",
        encoding="utf-8",
    )

    print(f"\n[OK] Agent 5 JSON saved to: {json_path}")
    print(f"[OK] Clean Markdown blog saved to: {markdown_path}")
    print(f"Article word count: {result.word_count}")
    print(f"Written sections: {len(result.sections)}")
    print(f"Claims used: {len(result.claims_used)}")
    print(f"Unresolved items: {len(result.unresolved_items)}")

    if result.unresolved_items:
        print("\nUnresolved items:")
        for item in result.unresolved_items:
            print(f" - {item}")



if __name__ == "__main__":
    main()
