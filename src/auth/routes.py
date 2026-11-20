"""Sign-up, sign-in, and sign-out routes."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.auth.dependencies import (
    SESSION_COOKIE_NAME,
    get_current_user,
    session_cookie_secure,
)
from src.auth.security import create_session_token, hash_password, hash_session_token, verify_password
from src.db.database import get_db
from src.db.models import AuthSession, User

SESSION_TTL = timedelta(days=14)
router = APIRouter(prefix="/api/auth", tags=["authentication"])
logger = logging.getLogger(__name__)


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Email must be text.")
        normalized = value.strip().casefold()
        if (
            normalized.count("@") != 1
            or any(character.isspace() for character in normalized)
            or "." not in normalized.rsplit("@", maxsplit=1)[-1]
        ):
            raise ValueError("Enter a valid email address.")
        return normalized


class SignUpCredentials(Credentials):
    password: str = Field(min_length=12, max_length=128)


def _public_user(user: User) -> dict[str, str]:
    return {"id": str(user.id), "email": user.email}


def _start_session(db: Session, user: User, response: Response) -> None:
    token = create_session_token()
    expires_at = datetime.now(timezone.utc) + SESSION_TTL
    db.add(
        AuthSession(
            user_id=user.id,
            token_hash=hash_session_token(token),
            expires_at=expires_at,
        )
    )
    db.commit()
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=int(SESSION_TTL.total_seconds()),
        expires=expires_at,
        httponly=True,
        secure=session_cookie_secure(),
        samesite="lax",
        path="/",
    )


@router.post("/signup", status_code=201)
def sign_up(
    credentials: SignUpCredentials,
    response: Response,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    user = User(
        email=credentials.email,
        password_hash=hash_password(credentials.password),
    )
    db.add(user)
    try:
        db.flush()
        _start_session(db, user, response)
    except IntegrityError as exc:
        db.rollback()
        existing_user = db.scalar(select(User.id).where(User.email == credentials.email))
        if existing_user is None:
            logger.exception("Could not create account for %s", credentials.email)
            raise HTTPException(status_code=503, detail="Could not create the account.") from exc
        raise HTTPException(status_code=409, detail="An account with this email already exists.") from exc
    return _public_user(user)


@router.post("/login")
def sign_in(
    credentials: Credentials,
    response: Response,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    user = db.scalar(select(User).where(User.email == credentials.email))
    if user is None or not verify_password(credentials.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    _start_session(db, user, response)
    return _public_user(user)


@router.post("/logout", status_code=204)
def sign_out(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> Response:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        auth_session = db.scalar(
            select(AuthSession).where(
                AuthSession.token_hash == hash_session_token(token),
            )
        )
        if auth_session is not None:
            db.delete(auth_session)
            db.commit()
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=session_cookie_secure(),
        samesite="lax",
    )
    response.status_code = 204
    return response


@router.get("/me")
def get_me(user: User = Depends(get_current_user)) -> dict[str, str]:
    return _public_user(user)
