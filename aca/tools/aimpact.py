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