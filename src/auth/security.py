"""Password and opaque-session-token helpers."""

from __future__ import annotations

import hashlib
import secrets

from pwdlib import PasswordHash


password_hasher = PasswordHash.recommended()
_DUMMY_PASSWORD_HASH = password_hasher.hash(secrets.token_urlsafe(24))


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    if password_hash is None:
        password_hasher.verify(password, _DUMMY_PASSWORD_HASH)
        return False
    return password_hasher.verify(password, password_hash)


def create_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
