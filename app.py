"""ASGI entry point for the ACA web application.

Run locally with: uvicorn app:app --reload
"""

from web.server import app

__all__ = ["app"]
