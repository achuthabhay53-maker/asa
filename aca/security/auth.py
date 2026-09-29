"""Verify ACA-tenant API tokens (issued from Content Studio)."""
# bcrypt-hashed, prefix-indexed for O(1) lookup, scoped.

def verify_bearer_token(db, token: str):
    # stub
    return None
