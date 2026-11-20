"""Authentication dependencies for FastAPI routes."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.auth.security import hash_session_token
from src.db.database import get_db
from src.db.models import AuthSession, User

SESSION_COOKIE_NAME = "aca_session"


def session_cookie_secure() -> bool:
    return os.getenv("ACA_COOKIE_SECURE", "false").strip().casefold() in {
        "1",
        "true",
        "yes",
    }


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Sign in to continue.")

    token_hash = hash_session_token(token)
    auth_session = db.scalar(
        select(AuthSession).where(
            AuthSession.token_hash == token_hash,
            AuthSession.expires_at > datetime.now(timezone.utc),
        )
    )
    if auth_session is None:
        raise HTTPException(status_code=401, detail="Your session has expired. Sign in again.")

    user = db.get(User, auth_session.user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="The account for this session is not available.")
    return user
