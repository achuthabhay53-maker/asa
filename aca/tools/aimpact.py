"""
Temporary development reader for AImpact PostgreSQL.

IMPORTANT:
This is a development adapter only.

Production ACA will use the AImpact REST API.

This module exists so the Source Analyzer can be developed
against real AImpact data before API credentials are available.
"""

import json
import os
from datetime import datetime
from typing import Any

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from aca.orchestrator.state import (
    CitedSource,
    ModelEvidence,
    PromptEvidence,
)


load_dotenv()


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set."
        )

    return database_url


def get_connection():
    return psycopg.connect(
        get_database_url(),
        connect_timeout=10,
        options="-c default_transaction_read_only=on",
        row_factory=dict_row,
    )


def _normalize_sources(
    sources: Any,
    model_name: str,
) -> list[CitedSource]:
    """
    Convert AImpact's sources JSON into CitedSource objects.

    AImpact source structures can evolve, so this function
    intentionally handles several common shapes.
    """

    if not sources:
        return []

    if isinstance(sources, str):
        try:
            sources = json.loads(sources)
        except json.JSONDecodeError:
            return [
                CitedSource(
                    model=model_name,
                    excerpt=sources,
                )
            ]

    if isinstance(sources, dict):
        sources = [sources]

    if not isinstance(sources, list):
        return []

    normalized: list[CitedSource] = []

    for source in sources:
        if isinstance(source, str):
            normalized.append(
                CitedSource(
                    model=model_name,
                    excerpt=source,
                )
            )
            continue

        if not isinstance(source, dict):
            continue

        url = (
            source.get("url")
            or source.get("link")
            or source.get("href")
        )

        domain = source.get("domain")
        title = source.get("title")

        competitor = (
            source.get("competitor")
            or source.get("company")
            or source.get("name")
        )

        excerpt = (
            source.get("excerpt")
            or source.get("snippet")
            or source.get("quote")
            or source.get("text")
            or source.get("title")
        )

        if not excerpt:
            excerpt = json.dumps(
                source,
                ensure_ascii=False,
            )

        normalized.append(
            CitedSource(
                model=model_name,
                url=url,
                domain=domain,
                competitor=competitor,
                title=str(title) if title else None,
                excerpt=str(excerpt),
            )
        )

    return normalized


def get_campaign(campaign_id: int) -> dict:
    query = """
        SELECT
            id,
            project_id,
            name
        FROM campaigns
        WHERE id = %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (campaign_id,),
            )

            row = cur.fetchone()

    if not row:
        raise ValueError(
            f"Campaign {campaign_id} was not found."
        )

    return row


def get_project(project_id: int) -> dict:
    query = """
        SELECT
            id,
            company_name,
            company_url,
            region,
            keywords
        FROM projects
        WHERE id = %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (project_id,),
            )

            row = cur.fetchone()

    if not row:
        raise ValueError(
            f"Project {project_id} was not found."
        )

    return row


def get_prompt_evidence(
    campaign_id: int,
    score_threshold: float = 70,
) -> list[PromptEvidence]:
    """
    Get the latest low-scoring audit for each
    prompt + model combination.

    Historical audits are ignored when a newer audit exists.
    """

    query = """
        SELECT
            a.id AS audit_score_id,
            a.project_id,
            a.campaign_id,
            a.prompt_id,
            a.score,
            a.model_name,
            a.region,
            a.response_text,
            a.sources,
            a.created_at,

            cp.prompt_text,
            cp.intent_category

        FROM audit_scores a

        LEFT JOIN campaign_prompts cp
            ON cp.id = a.prompt_id

        WHERE a.campaign_id = %s
          AND a.prompt_id IS NOT NULL

        ORDER BY
            a.created_at DESC;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (campaign_id,),
            )

            rows = cur.fetchall()

    # Keep latest audit for each prompt/model.
    latest: dict[tuple[int, str], dict] = {}

    for row in rows:
        prompt_id = row["prompt_id"]
        model_name = row["model_name"]

        if not model_name:
            continue

        key = (
            prompt_id,
            model_name,
        )

        existing = latest.get(key)

        if existing is None:
            latest[key] = row
            continue

        current_created = row["created_at"]
        existing_created = existing["created_at"]

        if current_created > existing_created:
            latest[key] = row

    # Keep only scored records below threshold.
    low_scores = [
        row
        for row in latest.values()
        if row["score"] is not None
        and row["score"] < score_threshold
    ]

    # Group by prompt.
    grouped: dict[int, PromptEvidence] = {}

    for row in low_scores:
        prompt_id = row["prompt_id"]
        prompt_text = row["prompt_text"]

        # Orphaned audit records cannot be analyzed because
        # the original prompt no longer exists.
        if not prompt_text:
            continue

        if prompt_id not in grouped:
            grouped[prompt_id] = PromptEvidence(
                prompt_id=str(prompt_id),
                prompt_text=prompt_text,
                per_model={},
            )

        model_name = row["model_name"]

        grouped[prompt_id].per_model[model_name] = (
            ModelEvidence(
                response_text=row["response_text"],
                sources=_normalize_sources(
                    row["sources"],
                    model_name,
                ),
            )
        )

    return list(grouped.values())


def get_source_analyzer_input(
    project_id: int,
    campaign_id: int,
    score_threshold: float = 70,
) -> tuple[dict, list[PromptEvidence]]:
    """
    Convenience function used by the development runner.
    """

    campaign = get_campaign(campaign_id)

    if campaign["project_id"] != project_id:
        raise ValueError(
            f"Campaign {campaign_id} does not belong "
            f"to project {project_id}."
        )

    project = get_project(project_id)

    prompts = get_prompt_evidence(
        campaign_id=campaign_id,
        score_threshold=score_threshold,
    )

    return project, prompts


def list_campaigns(project_id: int) -> list[dict]:
    """Campaigns that belong to one AImpact project."""
    query = """
        SELECT
            id,
            project_id,
            name
        FROM campaigns
        WHERE project_id = %s
        ORDER BY id;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (project_id,))
            return list(cur.fetchall())


def get_project_audit_dataset(
    project_id: int,
    score_threshold: float | None = None,
    latest_only: bool = True,
) -> dict[str, Any]:
    """
    Latest AI answers and citations for every campaign in one AImpact project.

    This is the input for Agent 1. ``score_threshold`` is applied later, per
    prompt, so a low-scoring prompt keeps every model's answer.
    """
    project = get_project(project_id)
    campaigns = list_campaigns(project_id)
    campaign_names = {row["id"]: row["name"] for row in campaigns}

    query = """
        SELECT
            a.id AS audit_score_id,
            a.project_id,
            a.campaign_id,
            a.prompt_id,
            a.score,
            a.model_name,
            a.region,
            a.response_text,
            a.sources,
            a.created_at,

            cp.prompt_text,
            cp.intent_category

        FROM audit_scores a

        LEFT JOIN campaign_prompts cp
            ON cp.id = a.prompt_id

        WHERE a.project_id = %s
          AND a.prompt_id IS NOT NULL

        ORDER BY
            a.created_at DESC;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (project_id,))
            rows = cur.fetchall()

    if latest_only:
        latest: dict[tuple[int, str], dict] = {}
        for row in rows:
            key = (row["prompt_id"], row["model_name"] or "")
            if key not in latest:
                latest[key] = row
        rows = list(latest.values())

    if score_threshold is not None:
        rows = [
            r for r in rows
            if r["score"] is not None and r["score"] < score_threshold
        ]

    responses = []
    for row in rows:
        model_name = str(row["model_name"] or "unknown")
        raw_sources = _normalize_sources(row["sources"], model_name)
        citations = []
        for s in raw_sources:
            citations.append({
                "url": s.url or "",
                "title": s.title or "",
                "domain": s.domain or "",
                "content": s.excerpt or "",
                "competitor": s.competitor,
            })

        timestamp = ""
        if row["created_at"]:
            if hasattr(row["created_at"], "isoformat"):
                timestamp = row["created_at"].isoformat()
            else:
                timestamp = str(row["created_at"])

        responses.append({
            "audit_score_id": row["audit_score_id"],
            "campaign_id": row["campaign_id"],
            "campaign_name": campaign_names.get(row["campaign_id"], ""),
            "prompt_id": str(row["prompt_id"]),
            "prompt": row["prompt_text"] or "",
            "intent_category": row["intent_category"] or "",
            "model": model_name,
            "score": float(row["score"]) if row["score"] is not None else None,
            "region": row["region"] or "",
            "timestamp": timestamp,
            "answer": row["response_text"] or "",
            "citations": citations,
        })

    return {
        "project": project,
        "campaigns": campaigns,
        "responses": responses,
        "source_content": {},
    }


def get_campaign_audit_dataset(
    campaign_id: int,
    score_threshold: float | None = None,
    latest_only: bool = False,
) -> dict[str, Any]:
    """
    Fetch all audit scores, prompts, AI model responses, and citations for a campaign
    directly from AImpact PostgreSQL.

    Returns:
        dict with 'campaign', 'project', 'responses', and 'source_content'.
    """
    campaign = get_campaign(campaign_id)
    project = get_project(campaign["project_id"])

    query = """
        SELECT
            a.id AS audit_score_id,
            a.project_id,
            a.campaign_id,
            a.prompt_id,
            a.score,
            a.model_name,
            a.region,
            a.response_text,
            a.sources,
            a.created_at,

            cp.prompt_text,
            cp.intent_category

        FROM audit_scores a

        LEFT JOIN campaign_prompts cp
            ON cp.id = a.prompt_id

        WHERE a.campaign_id = %s
          AND a.prompt_id IS NOT NULL

        ORDER BY
            a.created_at DESC;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (campaign_id,))
            rows = cur.fetchall()

    if latest_only:
        latest: dict[tuple[int, str], dict] = {}
        for row in rows:
            key = (row["prompt_id"], row["model_name"] or "")
            if key not in latest:
                latest[key] = row
        rows = list(latest.values())

    if score_threshold is not None:
        rows = [
            r for r in rows
            if r["score"] is not None and r["score"] < score_threshold
        ]

    responses = []
    for row in rows:
        model_name = str(row["model_name"] or "unknown")
        raw_sources = _normalize_sources(row["sources"], model_name)
        citations = []
        for s in raw_sources:
            citations.append({
                "url": s.url or "",
                "title": s.title or "",
                "domain": s.domain or "",
                "content": s.excerpt or "",
                "competitor": s.competitor,
            })

        timestamp = ""
        if row["created_at"]:
            if hasattr(row["created_at"], "isoformat"):
                timestamp = row["created_at"].isoformat()
            else:
                timestamp = str(row["created_at"])

        responses.append({
            "audit_score_id": row["audit_score_id"],
            "prompt_id": str(row["prompt_id"]),
            "prompt": row["prompt_text"] or "",
            "intent_category": row["intent_category"] or "",
            "model": model_name,
            "score": float(row["score"]) if row["score"] is not None else None,
            "region": row["region"] or "",
            "timestamp": timestamp,
            "answer": row["response_text"] or "",
            "citations": citations,
        })

    return {
        "campaign": campaign,
        "project": project,
        "responses": responses,
        "source_content": {},
    }