"""Tenant self-service settings (WP link, Google Ads OAuth)."""
from fastapi import APIRouter, Depends
from aca.api.deps import get_current_tenant

router = APIRouter(tags=["tenants"])


@router.get("/tenants/ping")
def ping(tenant=Depends(get_current_tenant)):
    return {"ok": True, "tenant_id": tenant.id}
