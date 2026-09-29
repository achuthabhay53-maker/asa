"""Publish history + measure lift."""
from fastapi import APIRouter, Depends
from aca.api.deps import get_current_tenant

router = APIRouter(tags=["publications"])


@router.get("/publications/ping")
def ping(tenant=Depends(get_current_tenant)):
    return {"ok": True, "tenant_id": tenant.id}
