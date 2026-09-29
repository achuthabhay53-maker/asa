"""FastAPI dependencies: DB session, tenant auth."""
from typing import Annotated
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from aca.db.session import SessionLocal
from aca.security.auth import verify_bearer_token


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_tenant(
    authorization: Annotated[str | None, Header()] = None,
    db: Session = Depends(get_db),
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    tenant = verify_bearer_token(db, token)
    if not tenant:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")
    return tenant
