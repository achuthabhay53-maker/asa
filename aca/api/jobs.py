"""Job lifecycle: start, poll status, list artifacts."""
from fastapi import APIRouter, Depends
from aca.api.deps import get_current_tenant

router = APIRouter(tags=["jobs"])


@router.get("/jobs/ping")
def ping(tenant=Depends(get_current_tenant)):
    return {"ok": True, "tenant_id": tenant.id}
