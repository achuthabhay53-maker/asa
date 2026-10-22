"""Write the database rows for the campaign and prompt under test.

Reads AImpact in read-only mode. Defaults are campaign 81 and prompt 1409.

    python scripts/dump_prompt.py
    python scripts/dump_prompt.py --campaign 81 --prompt 1409 --output prompt_1409_db.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aca.tools.aimpact import _normalize_sources, get_campaign, get_connection, get_project


def _source_row(source) -> dict:
    return {
        "url": source.url,
        "title": source.title,
        "domain": source.domain,
    }


def load_prompt(campaign_id: int, prompt_id: int) -> dict:
    campaign = get_campaign(campaign_id)
    project = get_project(campaign["project_id"])
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    a.id AS audit_score_id,
                    a.model_name,
                    a.score,
                    a.region,
                    a.created_at,
                    a.response_text,
                    a.sources,
                    cp.prompt_text,
                    cp.intent_category
                FROM audit_scores a
                LEFT JOIN campaign_prompts cp ON cp.id = a.prompt_id
                WHERE a.campaign_id = %s
                  AND a.prompt_id = %s
                ORDER BY a.created_at DESC
                """,
                (campaign_id, prompt_id),
            )
            rows = cur.fetchall()

    if not rows:
        raise SystemExit(f"No audit rows for campaign {campaign_id}, prompt {prompt_id}.")

    latest_ids: set[int] = set()
    seen_models: set[str] = set()
    for row in rows:
        model = row["model_name"]
        if model and model not in seen_models:
            seen_models.add(model)
            latest_ids.add(row["audit_score_id"])

    answers = []
    for row in rows:
        sources = _normalize_sources(row["sources"], row["model_name"] or "")
        answers.append(
            {
                "audit_score_id": row["audit_score_id"],
                "model_name": row["model_name"],
                "score": row["score"],
                "region": row["region"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "is_latest_for_model": row["audit_score_id"] in latest_ids,
                "included_by_agent1": row["score"] is not None and row["score"] < 70 and row["audit_score_id"] in latest_ids,
                "response_text": row["response_text"] or "",
                "sources": [_source_row(source) for source in sources],
            }
        )

    latest = [answer for answer in answers if answer["is_latest_for_model"]]
    return {
        "campaign_id": campaign["id"],
        "campaign_name": campaign["name"],
        "project_id": project["id"],
        "company_name": project["company_name"],
        "company_url": project["company_url"],
        "region": project["region"],
        "prompt_id": prompt_id,
        "prompt_text": rows[0]["prompt_text"],
        "intent_category": rows[0]["intent_category"],
        "audit_row_count": len(answers),
        "latest_model_count": len(latest),
        "latest_citation_count": sum(len(answer["sources"]) for answer in latest),
        "answers": answers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump one AImpact campaign prompt for manual checking.")
    parser.add_argument("--campaign", type=int, default=81)
    parser.add_argument("--prompt", type=int, default=1409)
    parser.add_argument("--output", default="prompt_1409_db.json")
    args = parser.parse_args()

    payload = load_prompt(args.campaign, args.prompt)
    path = Path(args.output)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Campaign {payload['campaign_id']}: {payload['campaign_name']}")
    print(f"Company: {payload['company_name']}")
    print(f"Company URL: {payload['company_url'] or '(empty)'}")
    print(f"Prompt {payload['prompt_id']}: {payload['prompt_text']}")
    print(f"Audit rows: {payload['audit_row_count']}")
    print(f"Latest answers: {payload['latest_model_count']}")
    print(f"Citations on latest answers: {payload['latest_citation_count']}")
    print()
    for answer in payload["answers"]:
        if not answer["is_latest_for_model"]:
            continue
        print(
            f"- {answer['model_name']}: audit {answer['audit_score_id']}, "
            f"score {answer['score']}, {len(answer['sources'])} citations, "
            f"{len(answer['response_text'])} characters"
        )
    print(f"\nFull answers and sources saved to: {path.resolve()}")


if __name__ == "__main__":
    main()
