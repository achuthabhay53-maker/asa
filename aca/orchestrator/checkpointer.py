"""PostgreSQL checkpointer lifecycle for LangGraph workflows."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@contextmanager
def postgres_checkpointer() -> Iterator[PostgresSaver]:
    """Open and initialize the PostgreSQL checkpointer for one workflow operation."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured; PostgreSQL workflow checkpoints cannot be saved."
        )

    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        checkpointer.setup()
        yield checkpointer


def thread_config(run_id: str) -> dict[str, dict[str, str]]:
    """Return the LangGraph config that scopes checkpoints to one workflow run."""
    if not run_id.strip():
        raise ValueError("A non-empty run_id is required for checkpointed workflows.")
    return {"configurable": {"thread_id": run_id}}
