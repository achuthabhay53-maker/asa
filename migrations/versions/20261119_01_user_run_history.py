"""Create user, login session, and blog history tables.

Revision ID: 20261119_01
Revises:
Create Date: 2026-11-19
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20261119_01"
down_revision = None
branch_labels = None
depends_on = None

json_document = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "auth_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index("ix_auth_sessions_user_id", "auth_sessions", ["user_id"], unique=False)
    op.create_index("ix_auth_sessions_expires_at", "auth_sessions", ["expires_at"], unique=False)

    op.create_table(
        "blog_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.String(length=32), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("workflow", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("request_data", json_document, nullable=False),
        sa.Column("progress_data", json_document, nullable=False),
        sa.Column("latest_snapshot", json_document, nullable=True),
        sa.Column("error_message", sa.String(length=2000), nullable=True),
        sa.Column("current_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("workflow IN ('aimpact', 'general')", name="ck_blog_runs_workflow"),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'complete', 'failed')",
            name="ck_blog_runs_status",
        ),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_index("ix_blog_runs_job_id", "blog_runs", ["job_id"], unique=True)
    op.create_index("ix_blog_runs_owner_id", "blog_runs", ["owner_id"], unique=False)
    op.create_index("ix_blog_runs_owner_created", "blog_runs", ["owner_id", "created_at"], unique=False)

    op.create_table(
        "blog_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=100), nullable=False),
        sa.Column("snapshot", json_document, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["blog_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "version_number", name="uq_blog_versions_run_number"),
    )
    op.create_index("ix_blog_versions_run_id", "blog_versions", ["run_id"], unique=False)
    op.create_index("ix_blog_versions_run_created", "blog_versions", ["run_id", "created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_blog_versions_run_created", table_name="blog_versions")
    op.drop_index("ix_blog_versions_run_id", table_name="blog_versions")
    op.drop_table("blog_versions")
    op.drop_index("ix_blog_runs_owner_created", table_name="blog_runs")
    op.drop_index("ix_blog_runs_owner_id", table_name="blog_runs")
    op.drop_index("ix_blog_runs_job_id", table_name="blog_runs")
    op.drop_table("blog_runs")
    op.drop_index("ix_auth_sessions_expires_at", table_name="auth_sessions")
    op.drop_index("ix_auth_sessions_user_id", table_name="auth_sessions")
    op.drop_table("auth_sessions")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
