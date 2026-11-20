"""Persistence helpers for authenticated blog runs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import BlogRun, BlogVersion


def create_blog_run(
    db: Session,
    *,
    run_id: UUID,
    job_id: str,
    owner_id: UUID,
    workflow: str,
    request_data: dict[str, Any],
    title: str,
) -> BlogRun:
    run = BlogRun(
        id=run_id,
        job_id=job_id,
        owner_id=owner_id,
        workflow=workflow,
        status="queued",
        title=title[:500],
        request_data=request_data,
        progress_data={"message": "Workflow queued", "completed": []},
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_owned_run(db: Session, run_id: UUID, owner_id: UUID, workflow: str | None = None) -> BlogRun:
    query = select(BlogRun).where(BlogRun.id == run_id, BlogRun.owner_id == owner_id)
    if workflow is not None:
        query = query.where(BlogRun.workflow == workflow)
    run = db.scalar(query)
    if run is None:
        raise LookupError("Saved run was not found.")
    return run


def update_run_progress(
    db: Session,
    *,
    job_id: str,
    status: str | None = None,
    progress_data: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> BlogRun:
    run = db.scalar(select(BlogRun).where(BlogRun.job_id == job_id).with_for_update())
    if run is None:
        raise LookupError("Saved run was not found for this job.")
    if status is not None:
        run.status = status
    if progress_data is not None:
        run.progress_data = progress_data
    if error_message is not None:
        run.error_message = error_message[:2000]
    run.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(run)
    return run


def append_run_version(
    db: Session,
    *,
    run_id: UUID,
    owner_id: UUID,
    snapshot: dict[str, Any],
    source: str,
    title: str | None = None,
    status: str | None = None,
) -> BlogVersion:
    run = db.scalar(
        select(BlogRun).where(BlogRun.id == run_id, BlogRun.owner_id == owner_id).with_for_update()
    )
    if run is None:
        raise LookupError("Saved run was not found.")
    version_number = run.current_version + 1
    version = BlogVersion(
        id=uuid4(),
        run_id=run.id,
        version_number=version_number,
        source=source[:100],
        snapshot=snapshot,
    )
    run.current_version = version_number
    run.latest_snapshot = snapshot
    if title:
        run.title = title[:500]
    if status is not None:
        run.status = status
    run.updated_at = datetime.now(timezone.utc)
    db.add(version)
    db.commit()
    db.refresh(version)
    return version


def save_run_snapshot(
    db: Session,
    *,
    run_id: UUID,
    owner_id: UUID,
    snapshot: dict[str, Any],
    title: str | None = None,
    status: str | None = None,
) -> BlogRun:
    run = get_owned_run(db, run_id, owner_id)
    run.latest_snapshot = snapshot
    if title:
        run.title = title[:500]
    if status is not None:
        run.status = status
    run.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(run)
    return run
