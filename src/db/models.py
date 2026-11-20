"""SQLAlchemy models for users, sessions, and saved blog runs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.database import Base


JSON_DOCUMENT = JSON().with_variant(JSONB(), "postgresql")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class BlogRun(Base):
    __tablename__ = "blog_runs"
    __table_args__ = (
        CheckConstraint("workflow IN ('aimpact', 'general')", name="ck_blog_runs_workflow"),
        CheckConstraint(
            "status IN ('queued', 'running', 'complete', 'failed')",
            name="ck_blog_runs_status",
        ),
        Index("ix_blog_runs_owner_created", "owner_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    owner_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    workflow: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="queued")
    title: Mapped[str] = mapped_column(String(500), default="New blog run")
    request_data: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT)
    progress_data: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, default=dict)
    latest_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON_DOCUMENT, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    current_version: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class BlogVersion(Base):
    __tablename__ = "blog_versions"
    __table_args__ = (
        UniqueConstraint("run_id", "version_number", name="uq_blog_versions_run_number"),
        Index("ix_blog_versions_run_created", "run_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("blog_runs.id", ondelete="CASCADE"),
        index=True,
    )
    version_number: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(100))
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
