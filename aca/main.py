"""FastAPI entrypoint. Wires routers, middleware, lifespan, scheduler."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aca.api import jobs, drafts, publications, tenants
from aca.workers.scheduler import start_scheduler, stop_scheduler
from aca.observability.telemetry import configure_telemetry


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_telemetry()
    start_scheduler()
    try:
        yield
    finally:
        stop_scheduler()


app = FastAPI(title="AImpact Content Agent", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tightened by CloudFront in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router,         prefix="/aca")
app.include_router(drafts.router,       prefix="/aca")
app.include_router(publications.router, prefix="/aca")
app.include_router(tenants.router,      prefix="/aca")


@app.get("/aca/health")
def health():
    return {"status": "ok"}
