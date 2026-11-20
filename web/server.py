"""FastAPI backend and static web frontend for ACA."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.agents.aimpact_blog.source_intelligence import TOP_N_SOURCES
from src.agents.aimpact_blog.writer import WriterAgent
from src.agents.aimpact_blog.reviewer import ReviewerAgent
from src.auth.dependencies import get_current_user
from src.auth.routes import router as auth_router
from src.db.database import get_db, get_session_factory
from src.db.models import BlogRun, BlogVersion, User
from src.db.repository import (
    append_run_version,
    create_blog_run,
    get_owned_run,
    save_run_snapshot,
    update_run_progress,
)
from src.models.general_blog import GeneralBlogRequest, GeneralBlogResult
from src.models.writer import WriterOutput, WrittenSection
from src.observability.token_usage import merge_token_usage, token_usage_scope
from src.orchestrator.checkpointer import postgres_checkpointer, thread_config
from src.orchestrator.general_blog import (
    GeneralBlogProgress,
    GeneralBlogWorkflow,
    GeneralBlogWorkflowError,
)
from src.orchestrator.pipeline import PipelineState, _build_writer_input, build_pipeline
from src.tools.reference_fetcher import MAX_REFERENCE_BYTES, MAX_UPLOADED_PDFS
from src.ui.pdf_export import build_article_pdf
from src.ui.review_annotations import render_annotated_markdown
from src.utils.general_blog_human import result_after_draft_change, review_passed
from src.utils.output_storage import (
    AGENT1_DEBUG_OUTPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    clear_agent_outputs,
    save_agent_output,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "web" / "static"
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aca-workflow")
LOCK = threading.RLock()
JOBS: dict[str, dict[str, Any]] = {}
MAIN_STAGES = (
    ("prompt_analysis", "Agent 1 · Prompt Intelligence", 1),
    ("source_analysis", "Agent 2 · Source Intelligence", 2),
    ("guide", "Agent 3 · Ground Truth", 3),
    ("content_brief", "Agent 4 · Content Brief", 4),
    ("written_article", "Agent 5 · Blog Writer", 5),
    ("review_output", "Agent 6 · Reviewer", 6),
)
MAIN_STAGE_BY_NODE = {f"agent{index}": (key, label) for key, label, index in MAIN_STAGES}

app = FastAPI(title="AImpact Content Agent", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(auth_router)


class AImpactRunRequest(BaseModel):
    campaign_id: int = Field(ge=1)
    prompt_id: str = Field(min_length=1, max_length=200)
    website: str = Field(min_length=1, max_length=2000)
    company_name: str = Field(default="", max_length=200)
    top_n: int = Field(default=TOP_N_SOURCES, ge=1, le=20)
    target_word_count: int = Field(default=800, ge=500, le=5000)


class ArticleEdit(BaseModel):
    article_markdown: str = Field(min_length=1, max_length=500_000)


class RevisionRequest(BaseModel):
    instructions: str = Field(min_length=1, max_length=12_000)


class FindingDecisions(BaseModel):
    statuses: dict[str, str]


class ApprovalRequest(BaseModel):
    acknowledged: bool


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return jsonable_encoder(value)


def _fingerprint(value: Any) -> str:
    payload = json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _state_from_values(values: dict[str, Any]) -> PipelineState:
    try:
        return PipelineState.model_validate(values)
    except ValidationError as exc:
        raise HTTPException(status_code=500, detail="Saved workflow state is invalid.") from exc


def _load_main_state(run_id: str) -> PipelineState:
    try:
        with postgres_checkpointer() as checkpointer:
            snapshot = build_pipeline(checkpointer).get_state(thread_config(run_id))
    except Exception as exc:
        logger.exception("Could not load workflow run %s", run_id)
        raise HTTPException(status_code=503, detail="Could not load the saved workflow.") from exc
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="No saved workflow was found for this run ID.")
    return _state_from_values(snapshot.values)


def _save_main_state(run_id: str, values: dict[str, Any]) -> PipelineState:
    try:
        with postgres_checkpointer() as checkpointer:
            workflow = build_pipeline(checkpointer)
            snapshot = workflow.get_state(thread_config(run_id))
            if not snapshot.values:
                raise HTTPException(status_code=404, detail="No saved workflow was found.")
            workflow.update_state(thread_config(run_id), values)
            updated = workflow.get_state(thread_config(run_id))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Could not update workflow run %s", run_id)
        raise HTTPException(status_code=503, detail="Could not save the workflow update.") from exc
    return _state_from_values(updated.values)


def _main_run_payload(state: PipelineState) -> dict[str, Any]:
    article = state.written_article
    fingerprint = _fingerprint(article) if article else None
    review_current = bool(article and state.review_output and state.reviewed_draft_fingerprint == fingerprint)
    findings = state.review_output.findings if state.review_output and review_current else []
    decisions = state.finding_statuses or {}
    findings_with_ids = [
        {"id": _fingerprint(finding)[:12], "finding": _json_value(finding)} for finding in findings
    ]
    open_count = sum(1 for finding in findings if decisions.get(_fingerprint(finding)[:12], "Open") == "Open")
    approved = bool(
        state.draft_approval
        and article
        and state.draft_approval.get("version") == state.draft_version
        and state.draft_approval.get("fingerprint") == fingerprint
    )
    return {
        "run_id": state.run_id,
        "prompt_id": state.prompt_id,
        "campaign_id": state.campaign_id,
        "website": state.website,
        "target_word_count": state.target_word_count,
        "draft_version": state.draft_version,
        "draft_versions": state.draft_versions,
        "draft_approval": state.draft_approval,
        "approved": approved,
        "review_current": review_current,
        "open_findings": open_count,
        "findings": findings_with_ids,
        "finding_statuses": decisions,
        "agent_outputs": {
            key: _json_value(getattr(state, key))
            for key, _, _ in MAIN_STAGES
            if getattr(state, key) is not None
        },
        "article_html": render_annotated_markdown(
            article.article_markdown if article else "",
            findings,
        )[0],
        "writer_input": _json_value(_build_writer_input(state))
        if all((state.prompt_analysis, state.source_analysis, state.guide, state.content_brief))
        else None,
        "token_usage": [_json_value(item) for item in state.token_usage],
        "errors": state.errors,
    }


def _general_result_payload(result: GeneralBlogResult, run_id: str | None = None) -> dict[str, Any]:
    payload = result.model_dump(mode="json")
    payload["run_id"] = run_id or next(
        (record.run_id for record in result.token_usage if record.run_id),
        "",
    )
    findings = result.review.findings if result.review_current else []
    payload["article_html"] = render_annotated_markdown(
        result.draft.article_markdown,
        findings,
    )[0]
    return payload


def _run_uuid(run_id: str) -> UUID:
    try:
        return UUID(run_id)
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=404, detail="Saved run was not found.") from exc


def _require_run_owner(
    db: Session,
    user: User,
    run_id: str,
    workflow: str | None = None,
) -> BlogRun:
    try:
        return get_owned_run(db, _run_uuid(run_id), user.id, workflow)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Saved run was not found.") from exc


def _persist_run_version(
    run_id: str,
    owner_id: UUID,
    snapshot: dict[str, Any],
    source: str,
    *,
    title: str | None = None,
    status: str | None = None,
) -> None:
    with get_session_factory()() as db:
        append_run_version(
            db,
            run_id=_run_uuid(run_id),
            owner_id=owner_id,
            snapshot=snapshot,
            source=source,
            title=title,
            status=status,
        )


def _persist_run_snapshot(
    run_id: str,
    owner_id: UUID,
    snapshot: dict[str, Any],
    *,
    title: str | None = None,
    status: str | None = None,
) -> None:
    with get_session_factory()() as db:
        save_run_snapshot(
            db,
            run_id=_run_uuid(run_id),
            owner_id=owner_id,
            snapshot=snapshot,
            title=title,
            status=status,
        )


def _persist_job(job_id: str, job: dict[str, Any]) -> None:
    run_id = job.get("run_id")
    if not run_id:
        return
    progress_keys = ("message", "completed", "active_stage", "stage", "total")
    progress = {key: job[key] for key in progress_keys if key in job}
    try:
        with get_session_factory()() as db:
            update_run_progress(
                db,
                job_id=job_id,
                status=job.get("status"),
                progress_data=progress,
                error_message=job.get("message") if job.get("status") == "failed" else None,
            )
    except LookupError:
        logger.exception("Could not find saved run for job %s", job_id)
        raise


def _new_version(
    state: PipelineState,
    output: WriterOutput,
    source: str,
    instructions: str = "",
) -> PipelineState:
    version = int(state.draft_version or 1) + 1
    versions = list(state.draft_versions or [])
    versions.append(
        {
            "version": version,
            "source": source,
            "instructions": instructions,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "output": output.model_dump(mode="json"),
        }
    )
    updated = _save_main_state(
        state.run_id,
        {
            "written_article": output,
            "draft_version": version,
            "draft_versions": versions,
            "draft_approval": None,
            "proposed_revision": None,
            "review_output": None,
            "reviewed_draft_fingerprint": None,
            "finding_statuses": {},
        },
    )
    save_agent_output(5, output, DEFAULT_OUTPUT_DIR, run_id=state.run_id)
    return updated


def _edited_output(output: WriterOutput, article: str) -> WriterOutput:
    lines = article.splitlines()
    headings = [
        (index, line.strip()[3:].strip())
        for index, line in enumerate(lines)
        if re.match(r"^##\s+", line.strip())
    ]
    sections = []
    for pos, (start, heading) in enumerate(headings):
        end = headings[pos + 1][0] if pos + 1 < len(headings) else len(lines)
        sections.append(
            WrittenSection(
                heading=heading,
                content_markdown="\n".join(lines[start + 1 : end]).strip(),
            )
        )
    title_match = next(
        (line.strip()[2:].strip() for line in lines if re.match(r"^#\s+", line.strip())),
        output.title,
    )
    unresolved = list(output.unresolved_items)
    for note in (
        "Human-edited article; recheck claim-to-evidence support before approval.",
        "Review SEO title and meta description after human edits.",
    ):
        if note not in unresolved:
            unresolved.append(note)
    return output.model_copy(
        update={
            "title": title_match,
            "article_markdown": article,
            "word_count": len(re.findall(r"\b[\w’'-]+\b", article)),
            "sections": sections,
            "claims_used": [],
            "unresolved_items": unresolved,
        }
    )


def _update_job(job_id: str, **updates: Any) -> None:
    with LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(updates)
            job = dict(JOBS[job_id])
        else:
            return
    _persist_job(job_id, job)
    if job.get("status") in {"complete", "failed"}:
        with LOCK:
            JOBS.pop(job_id, None)


def _report_aimpact_stage_start(job_id: str, event: dict[str, Any]) -> None:
    if event.get("type") != "task":
        return
    payload = event.get("payload", {})
    stage = MAIN_STAGE_BY_NODE.get(payload.get("name"))
    if stage is None:
        return
    key, label = stage
    _update_job(
        job_id,
        active_stage=key,
        message=f"{label} is running",
    )


def _run_aimpact(
    job_id: str,
    run_id: str,
    owner_id: UUID,
    request: AImpactRunRequest,
) -> None:
    state = PipelineState(
        run_id=run_id,
        campaign_id=request.campaign_id,
        prompt_id=request.prompt_id.strip(),
        website=request.website.strip(),
        company_name=request.company_name.strip(),
        top_n=request.top_n,
        max_questions=3,
        target_word_count=request.target_word_count,
        review_requested=True,
    )
    try:
        _update_job(
            job_id,
            run_id=state.run_id,
            status="running",
            active_stage=None,
            message="Connecting to the workflow database",
        )
        clear_agent_outputs(DEFAULT_OUTPUT_DIR)
        with postgres_checkpointer() as checkpointer:
            workflow = build_pipeline(checkpointer)
            config = thread_config(state.run_id)
            for mode, event in workflow.stream(
                state,
                config=config,
                stream_mode=["debug", "values"],
            ):
                if mode == "debug":
                    _report_aimpact_stage_start(job_id, event)
                    continue
                if mode != "values":
                    continue

                current = _state_from_values(event)
                completed = []
                for key, _, index in MAIN_STAGES:
                    value = getattr(current, key)
                    if value is not None:
                        completed.append(key)
                        output = _json_value(value)
                        save_agent_output(index, output, DEFAULT_OUTPUT_DIR, run_id=state.run_id)
                        if index == 1:
                            save_agent_output(index, output, AGENT1_DEBUG_OUTPUT_DIR, run_id=state.run_id)
                _update_job(
                    job_id,
                    run_id=state.run_id,
                    completed=completed,
                    active_stage=None,
                    message=f"{len(completed)} of 6 agents complete",
                )
            snapshot = workflow.get_state(config)
        final_state = _state_from_values(snapshot.values)
        if final_state.written_article is not None:
            initial_output = final_state.written_article
            versions = [
                {
                    "version": 1,
                    "source": "Agent 5 initial draft",
                    "instructions": "",
                    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "output": initial_output.model_dump(mode="json"),
                }
            ]
            review_fingerprint = (
                _fingerprint(initial_output) if final_state.review_output is not None else None
            )
            with postgres_checkpointer() as checkpointer:
                workflow = build_pipeline(checkpointer)
                workflow.update_state(
                    thread_config(state.run_id),
                    {
                        "draft_version": 1,
                        "draft_versions": versions,
                        "draft_approval": None,
                        "proposed_revision": None,
                        "reviewed_draft_fingerprint": review_fingerprint,
                        "finding_statuses": {},
                    },
                )
                final_state = _state_from_values(workflow.get_state(thread_config(state.run_id)).values)
        final_status = "failed" if final_state.errors else "complete"
        result_payload = _main_run_payload(final_state)
        if final_state.written_article is not None:
            _persist_run_version(
                state.run_id,
                owner_id,
                result_payload,
                "Initial campaign draft",
                title=final_state.written_article.title,
                status=final_status,
            )
        _update_job(
            job_id,
            status=final_status,
            completed=[key for key, _, _ in MAIN_STAGES if getattr(final_state, key) is not None],
            active_stage=None,
            message="Workflow complete" if not final_state.errors else "Workflow stopped with errors",
            errors=final_state.errors,
        )
    except Exception as exc:
        logger.exception("AImpact workflow failed (job_id=%s)", job_id)
        failure: dict[str, Any] = {
            "status": "failed",
            "message": str(exc),
            "error_type": type(exc).__name__,
        }
        try:
            if state.run_id:
                partial_state = _load_main_state(state.run_id)
                partial_result = _main_run_payload(partial_state)
                if partial_state.written_article is not None:
                    _persist_run_version(
                        state.run_id,
                        owner_id,
                        partial_result,
                        "Partial campaign draft",
                        title=partial_state.written_article.title,
                        status="failed",
                    )
        except HTTPException as load_error:
            logger.warning("Could not load partial workflow state: %s", load_error.detail)
        _update_job(job_id, **failure)


def _run_general(
    job_id: str,
    run_id: str,
    owner_id: UUID,
    request: GeneralBlogRequest,
    pdfs: list[tuple[str, bytes]],
) -> None:
    def report(event: GeneralBlogProgress) -> None:
        _update_job(
            job_id,
            completed=event.completed_stages,
            total=event.total_stages,
            stage=event.label,
            message=event.message,
        )

    try:
        _update_job(job_id, status="running", message="Starting General Blog workflow")
        result = GeneralBlogWorkflow(progress_callback=report).run(
            request,
            uploaded_reference_pdfs=pdfs,
            run_id=run_id,
        )
        _persist_run_version(
            run_id,
            owner_id,
            result.model_dump(mode="json"),
            "Initial general blog",
            title=result.draft.title,
            status="complete",
        )
        _update_job(
            job_id,
            status="complete",
            message="Workflow complete",
        )
    except GeneralBlogWorkflowError as exc:
        message = str(exc.cause)
        _update_job(
            job_id,
            status="failed",
            stage=exc.label,
            message=message,
            error_type=type(exc.cause).__name__,
            token_usage=[_json_value(item) for item in exc.token_usage],
        )
    except Exception as exc:
        logger.exception("General Blog workflow failed (job_id=%s)", job_id)
        _update_job(job_id, status="failed", message=str(exc), error_type=type(exc).__name__)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/me/runs")
def list_my_runs(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    filters = BlogRun.owner_id == user.id
    total = db.scalar(select(func.count()).select_from(BlogRun).where(filters)) or 0
    runs = db.scalars(
        select(BlogRun)
        .where(filters)
        .order_by(BlogRun.created_at.desc(), BlogRun.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return {
        "items": [
            {
                "run_id": str(run.id),
                "job_id": run.job_id,
                "workflow": run.workflow,
                "title": run.title,
                "status": run.status,
                "version": run.current_version,
                "created_at": run.created_at,
                "updated_at": run.updated_at,
            }
            for run in runs
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/me/runs/{run_id}/versions")
def list_run_versions(
    run_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    run = _require_run_owner(db, user, run_id)
    versions = db.scalars(
        select(BlogVersion).where(BlogVersion.run_id == run.id).order_by(BlogVersion.version_number.desc())
    ).all()
    items = []
    for version in versions:
        snapshot = version.snapshot
        if run.workflow == "aimpact":
            output = snapshot.get("agent_outputs", {}).get("written_article", {})
        else:
            output = snapshot.get("draft", {})
        items.append(
            {
                "version": version.version_number,
                "source": version.source,
                "title": output.get("title", run.title),
                "article_markdown": output.get("article_markdown", ""),
                "created_at": version.created_at,
            }
        )
    return {"run_id": str(run.id), "items": items}


@app.post("/api/aimpact/runs", status_code=202)
def start_aimpact_run(
    request: AImpactRunRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    run_id = str(uuid4())
    job_id = uuid4().hex
    create_blog_run(
        db,
        run_id=UUID(run_id),
        job_id=job_id,
        owner_id=user.id,
        workflow="aimpact",
        request_data=request.model_dump(mode="json"),
        title=f"Campaign {request.campaign_id} · {request.prompt_id.strip()}",
    )
    with LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "kind": "aimpact",
            "run_id": run_id,
            "status": "queued",
            "message": "Workflow queued",
            "completed": [],
        }
    try:
        EXECUTOR.submit(_run_aimpact, job_id, run_id, user.id, request)
    except RuntimeError as exc:
        _update_job(job_id, status="failed", message="The workflow could not be queued.")
        logger.exception("Could not queue campaign workflow (job_id=%s)", job_id)
        raise HTTPException(status_code=503, detail="The workflow could not be queued.") from exc
    return {"job_id": job_id, "run_id": run_id}


@app.get("/api/jobs/{job_id}")
def get_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    run = db.scalar(select(BlogRun).where(BlogRun.job_id == job_id, BlogRun.owner_id == user.id))
    if run is None:
        raise HTTPException(status_code=404, detail="No workflow job was found.")
    with LOCK:
        active_in_process = job_id in JOBS
    if run.status in {"queued", "running"} and not active_in_process:
        run = update_run_progress(
            db,
            job_id=job_id,
            status="failed",
            progress_data={
                **run.progress_data,
                "message": "The server restarted before this workflow finished.",
            },
            error_message="The server restarted before this workflow finished.",
        )
    payload: dict[str, Any] = {
        "id": job_id,
        "kind": run.workflow,
        "run_id": str(run.id),
        "job_id": job_id,
        "status": run.status,
        **run.progress_data,
    }
    if run.error_message:
        payload["message"] = run.error_message
    if run.latest_snapshot is not None and run.workflow == "aimpact":
        payload["result"] = run.latest_snapshot
    return jsonable_encoder(payload)


@app.get("/api/aimpact/runs/{run_id}")
def get_aimpact_run(
    run_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _require_run_owner(db, user, run_id, "aimpact")
    return _main_run_payload(_load_main_state(run_id))


@app.post("/api/aimpact/runs/{run_id}/edit")
def edit_aimpact_article(
    run_id: str,
    edit: ArticleEdit,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    run = _require_run_owner(db, user, run_id, "aimpact")
    state = _load_main_state(run_id)
    if state.written_article is None:
        raise HTTPException(status_code=409, detail="This workflow has no article to edit.")
    updated = _new_version(state, _edited_output(state.written_article, edit.article_markdown), "Human edit")
    payload = _main_run_payload(updated)
    _persist_run_version(
        run_id,
        user.id,
        payload,
        "Human edit",
        title=updated.written_article.title if updated.written_article else run.title,
    )
    return payload


@app.post("/api/aimpact/runs/{run_id}/revise")
def revise_aimpact_article(
    run_id: str,
    body: RevisionRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    run = _require_run_owner(db, user, run_id, "aimpact")
    state = _load_main_state(run_id)
    if state.written_article is None:
        raise HTTPException(status_code=409, detail="This workflow has no article to revise.")
    writer_input = _build_writer_input(state)
    if writer_input is None:
        raise HTTPException(status_code=409, detail="Writer inputs are not available for this run.")
    try:
        with token_usage_scope(run_id, "agent5") as records:
            revised = WriterAgent().revise(writer_input, state.written_article, body.instructions.strip())
        updated = _new_version(state, revised, "Agent 5 revision", body.instructions.strip())
        updated = _save_main_state(
            run_id,
            {
                "token_usage": merge_token_usage(state.token_usage, records),
            },
        )
        payload = _main_run_payload(updated)
        _persist_run_version(
            run_id,
            user.id,
            payload,
            "Writer revision",
            title=updated.written_article.title if updated.written_article else run.title,
        )
    except Exception as exc:
        logger.exception("Writer revision failed for run %s", run_id)
        raise HTTPException(status_code=502, detail=f"Writer revision failed: {exc}") from exc
    return payload


@app.post("/api/aimpact/runs/{run_id}/review")
def review_aimpact_article(
    run_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    run = _require_run_owner(db, user, run_id, "aimpact")
    state = _load_main_state(run_id)
    if state.written_article is None:
        raise HTTPException(status_code=409, detail="This workflow has no article to review.")
    try:
        with postgres_checkpointer() as checkpointer:
            workflow = build_pipeline(checkpointer)
            config = thread_config(run_id)
            workflow.invoke({"review_requested": True, "review_output": None}, config=config)
            reviewed = _state_from_values(workflow.get_state(config).values)
        if reviewed.review_output is None:
            raise RuntimeError("Workflow ended without a Reviewer result.")
        reviewed = _save_main_state(
            run_id,
            {
                "reviewed_draft_fingerprint": _fingerprint(reviewed.written_article),
                "finding_statuses": {},
            },
        )
        save_agent_output(6, _json_value(reviewed.review_output), DEFAULT_OUTPUT_DIR, run_id=run_id)
        payload = _main_run_payload(reviewed)
        _persist_run_snapshot(run_id, user.id, payload, title=run.title)
    except Exception as exc:
        logger.exception("Review failed for run %s", run_id)
        raise HTTPException(status_code=502, detail=f"Review failed: {exc}") from exc
    return payload


@app.put("/api/aimpact/runs/{run_id}/findings")
def save_finding_decisions(
    run_id: str,
    body: FindingDecisions,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    run = _require_run_owner(db, user, run_id, "aimpact")
    state = _load_main_state(run_id)
    allowed = {"Open", "Resolved", "Dismissed"}
    if any(value not in allowed for value in body.statuses.values()):
        raise HTTPException(status_code=422, detail="Finding status must be Open, Resolved, or Dismissed.")
    if state.review_output is None:
        raise HTTPException(status_code=409, detail="Review the current article before saving decisions.")
    valid_ids = {_fingerprint(finding)[:12] for finding in state.review_output.findings}
    if set(body.statuses) - valid_ids:
        raise HTTPException(status_code=422, detail="Finding decisions contain an unknown finding ID.")
    values: dict[str, Any] = {"finding_statuses": body.statuses}
    if any(status == "Open" for status in body.statuses.values()):
        values["draft_approval"] = None
    updated = _save_main_state(run_id, values)
    payload = _main_run_payload(updated)
    _persist_run_snapshot(run_id, user.id, payload, title=run.title)
    return payload


@app.post("/api/aimpact/runs/{run_id}/approve")
def approve_aimpact_article(
    run_id: str,
    body: ApprovalRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    run = _require_run_owner(db, user, run_id, "aimpact")
    state = _load_main_state(run_id)
    if not body.acknowledged:
        raise HTTPException(status_code=422, detail="Confirm that you reviewed this version.")
    if state.written_article is None or state.review_output is None:
        raise HTTPException(status_code=409, detail="Review the article before approval.")
    fingerprint = _fingerprint(state.written_article)
    if state.reviewed_draft_fingerprint != fingerprint:
        raise HTTPException(status_code=409, detail="The current article version has not been reviewed.")
    open_findings = [
        finding
        for finding in state.review_output.findings
        if (state.finding_statuses or {}).get(_fingerprint(finding)[:12], "Open") == "Open"
    ]
    if open_findings:
        raise HTTPException(status_code=409, detail="Decide all review findings before approval.")
    record = {
        "version": state.draft_version,
        "fingerprint": fingerprint,
        "approved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    updated = _save_main_state(run_id, {"draft_approval": record})
    payload = _main_run_payload(updated)
    _persist_run_snapshot(run_id, user.id, payload, title=run.title)
    return payload


@app.get("/api/aimpact/runs/{run_id}/article.pdf")
def download_aimpact_pdf(
    run_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    _require_run_owner(db, user, run_id, "aimpact")
    state = _load_main_state(run_id)
    if state.written_article is None:
        raise HTTPException(status_code=404, detail="This workflow has no article.")
    pdf_bytes = build_article_pdf(state.written_article.article_markdown)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="article.pdf"'},
    )


@app.post("/api/general/runs", status_code=202)
async def start_general_run(
    request_json: str = Form(...),
    reference_pdfs: list[UploadFile] = File(default=[]),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    try:
        request = GeneralBlogRequest.model_validate_json(request_json)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    if len(reference_pdfs) > MAX_UPLOADED_PDFS:
        raise HTTPException(status_code=422, detail=f"Upload no more than {MAX_UPLOADED_PDFS} PDFs.")
    pdfs: list[tuple[str, bytes]] = []
    for item in reference_pdfs:
        contents = await item.read(MAX_REFERENCE_BYTES + 1)
        if len(contents) > MAX_REFERENCE_BYTES:
            raise HTTPException(status_code=413, detail=f"{item.filename} exceeds the 15 MB per-file limit.")
        pdfs.append((item.filename or "reference.pdf", contents))
    run_id = str(uuid4())
    job_id = uuid4().hex
    create_blog_run(
        db,
        run_id=UUID(run_id),
        job_id=job_id,
        owner_id=user.id,
        workflow="general",
        request_data=request.model_dump(mode="json"),
        title=request.prompt[:500],
    )
    with LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "kind": "general",
            "run_id": run_id,
            "status": "queued",
            "message": "Workflow queued",
            "completed": 0,
            "total": 7,
        }
    try:
        EXECUTOR.submit(_run_general, job_id, run_id, user.id, request, pdfs)
    except RuntimeError as exc:
        _update_job(job_id, status="failed", message="The workflow could not be queued.")
        logger.exception("Could not queue General Blog workflow (job_id=%s)", job_id)
        raise HTTPException(status_code=503, detail="The workflow could not be queued.") from exc
    return {"job_id": job_id, "run_id": run_id}


def _load_general_result(
    db: Session,
    user: User,
    run_id: str,
) -> tuple[BlogRun, GeneralBlogResult]:
    run = _require_run_owner(db, user, run_id, "general")
    snapshot = run.latest_snapshot
    if snapshot is None:
        latest = db.scalar(
            select(BlogVersion)
            .where(BlogVersion.run_id == run.id)
            .order_by(BlogVersion.version_number.desc())
            .limit(1)
        )
        snapshot = latest.snapshot if latest is not None else None
    if snapshot is None:
        raise HTTPException(
            status_code=409,
            detail=f"This General Blog run is {run.status} and has no saved article yet.",
        )
    return run, GeneralBlogResult.model_validate(snapshot)


def _general_result_with_history(
    db: Session,
    run: BlogRun,
    result: GeneralBlogResult,
) -> dict[str, Any]:
    versions = db.scalars(
        select(BlogVersion).where(BlogVersion.run_id == run.id).order_by(BlogVersion.version_number)
    ).all()
    payload = _general_result_payload(result, str(run.id))
    payload["draft_version"] = run.current_version
    payload["draft_versions"] = [
        {
            "version": version.version_number,
            "source": version.source,
            "created_at": version.created_at.isoformat(),
            "output": version.snapshot.get("draft", {}),
        }
        for version in versions
    ]
    return payload


@app.get("/api/general/runs/{run_id}")
def get_general_run(
    run_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    run, result = _load_general_result(db, user, run_id)
    return _general_result_with_history(db, run, result)


@app.post("/api/general/runs/{run_id}/edit")
def edit_general_article(
    run_id: str,
    edit: ArticleEdit,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    run, result = _load_general_result(db, user, run_id)
    updated = result_after_draft_change(result, _edited_output(result.draft, edit.article_markdown))
    _persist_run_version(
        run_id,
        user.id,
        updated.model_dump(mode="json"),
        "Human edit",
        title=updated.draft.title,
    )
    db.expire_all()
    run = _require_run_owner(db, user, run_id, "general")
    return _general_result_with_history(db, run, updated)


@app.post("/api/general/runs/{run_id}/revise")
def revise_general_article(
    run_id: str,
    body: RevisionRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    run, result = _load_general_result(db, user, run_id)
    try:
        usage_run_id = next((record.run_id for record in result.token_usage if record.run_id), run_id)
        with token_usage_scope(usage_run_id, "agent5_human_revision") as records:
            revised = WriterAgent().revise(result.writer_input, result.draft, body.instructions.strip())
        updated = result_after_draft_change(result, revised).model_copy(
            update={"token_usage": merge_token_usage(result.token_usage, records)}
        )
        _persist_run_version(
            run_id,
            user.id,
            updated.model_dump(mode="json"),
            "Writer revision",
            title=updated.draft.title,
        )
    except Exception as exc:
        logger.exception("General Blog revision failed for run %s", run_id)
        raise HTTPException(status_code=502, detail=f"Writer revision failed: {exc}") from exc
    db.expire_all()
    run = _require_run_owner(db, user, run_id, "general")
    return _general_result_with_history(db, run, updated)


@app.post("/api/general/runs/{run_id}/review")
def review_general_article(
    run_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    run, result = _load_general_result(db, user, run_id)
    try:
        usage_run_id = next((record.run_id for record in result.token_usage if record.run_id), run_id)
        with token_usage_scope(usage_run_id, "agent6_human_review") as records:
            review = ReviewerAgent().review(result.writer_input, result.draft)
        updated = result.model_copy(
            update={
                "review": review,
                "quality_passed": review_passed(review),
                "review_current": True,
                "review_cycles": result.review_cycles + 1,
                "token_usage": merge_token_usage(result.token_usage, records),
            }
        )
        _persist_run_snapshot(
            run_id,
            user.id,
            updated.model_dump(mode="json"),
            title=updated.draft.title,
        )
    except Exception as exc:
        logger.exception("General Blog review failed for run %s", run_id)
        raise HTTPException(status_code=502, detail=f"Review failed: {exc}") from exc
    db.expire_all()
    run = _require_run_owner(db, user, run_id, "general")
    return _general_result_with_history(db, run, updated)


@app.get("/api/general/runs/{run_id}/result.json")
def download_general_result(
    run_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    run, result = _load_general_result(db, user, run_id)
    return _general_result_with_history(db, run, result)
