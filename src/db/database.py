"""SQLAlchemy session setup for ACA-owned data."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterator

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Base class for ACA database models."""


@lru_cache(maxsize=4)
def _session_factory(database_url: str) -> sessionmaker[Session]:
    engine = create_engine(database_url, pool_pre_ping=True)
    return sessionmaker(bind=engine, expire_on_commit=False)


def get_session_factory() -> sessionmaker[Session]:
    database_url = os.getenv("ACA_DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("ACA_DATABASE_URL is not configured.")
    return _session_factory(database_url)


def get_db() -> Iterator[Session]:
    try:
        factory = get_session_factory()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    with factory() as session:
        yield session
