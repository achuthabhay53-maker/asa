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
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from src.agents.aimpact_blog.source_intelligence import TOP_N_SOURCES
from src.agents.aimpact_blog.writer import WriterAgent
from src.agents.aimpact_blog.reviewer import ReviewerAgent
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

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "web" / "static"
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aca-workflow")
LOCK = threading.RLock()
JOBS: dict[str, dict[str, Any]] = {}
GENERAL_RESULTS: dict[str, GeneralBlogResult] = {}
MAIN_STAGES = (
    ("prompt_analysis", "Agent 1 · Prompt Intelligence", 1),
    ("source_analysis", "Agent 2 · Source Intelligence", 2),
    ("guide", "Agent 3 · Ground Truth", 3),
    ("content_brief", "Agent 4 · Content Brief", 4),
    ("written_article", "Agent 5 · Blog Writer", 5),
    ("review_output", "Agent 6 · Reviewer", 6),
)
MAIN_STAGE_BY_NODE = {
    f"agent{index}": (key, label)
    for key, label, index in MAIN_STAGES
}

app = FastAPI(title="AImpact Content Agent", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
    review_current = bool(
        article
        and state.review_output
        and state.reviewed_draft_fingerprint == fingerprint
    )
    findings = state.review_output.findings if state.review_output and review_current else []
    decisions = state.finding_statuses or {}
    findings_with_ids = [
        {"id": _fingerprint(finding)[:12], "finding": _json_value(finding)}
        for finding in findings
    ]
    open_count = sum(
        1 for finding in findings
        if decisions.get(_fingerprint(finding)[:12], "Open") == "Open"
    )
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


def _run_aimpact(job_id: str, request: AImpactRunRequest) -> None:
    state = PipelineState(
        campaign_id=request.campaign_id,
        prompt_id=request.prompt_id.strip(),
        website=request.website.strip(),
        company_name=request.company_name.strip(),
        top_n=request.top_n,
        max_questions=3,
        target_word_count=request.target_word_count,
        review_requested=True,
    )
    _update_job(
        job_id,
        run_id=state.run_id,
        status="running",
        active_stage=None,
        message="Connecting to the workflow database",
    )
    try:
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
                final_state = _state_from_values(
                    workflow.get_state(thread_config(state.run_id)).values
                )
        _update_job(
            job_id,
            status="failed" if final_state.errors else "complete",
            completed=[key for key, _, _ in MAIN_STAGES if getattr(final_state, key) is not None],
            active_stage=None,
            message="Workflow complete" if not final_state.errors else "Workflow stopped with errors",
            result=_main_run_payload(final_state),
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
                failure["result"] = _main_run_payload(partial_state)
        except HTTPException as load_error:
            logger.warning("Could not load partial workflow state: %s", load_error.detail)
        _update_job(job_id, **failure)


def _run_general(
    job_id: str,
    request: GeneralBlogRequest,
    pdfs: list[tuple[str, bytes]],
) -> None:
    run_id = job_id

    def report(event: GeneralBlogProgress) -> None:
        nonlocal run_id
        run_id = event.run_id
        _update_job(
            job_id,
            run_id=event.run_id,
            completed=event.completed_stages,
            total=event.total_stages,
            stage=event.label,
            message=event.message,
        )

    try:
        result = GeneralBlogWorkflow(progress_callback=report).run(request, uploaded_reference_pdfs=pdfs)
        with LOCK:
            GENERAL_RESULTS[run_id] = result
            GENERAL_RESULTS[job_id] = result
        _update_job(
            job_id,
            run_id=run_id,
            status="complete",
            message="Workflow complete",
            result=_general_result_payload(result, run_id),
        )
    except GeneralBlogWorkflowError as exc:
        _update_job(
            job_id,
            run_id=exc.run_id,
            status="failed",
            stage=exc.label,
            message=str(exc.cause),
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


@app.post("/api/aimpact/runs", status_code=202)
def start_aimpact_run(request: AImpactRunRequest) -> dict[str, str]:
    job_id = uuid4().hex
    with LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "kind": "aimpact",
            "status": "queued",
            "message": "Workflow queued",
            "completed": [],
        }
    EXECUTOR.submit(_run_aimpact, job_id, request)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No active job was found.")
        return jsonable_encoder(job)


@app.get("/api/aimpact/runs/{run_id}")
def get_aimpact_run(run_id: str) -> dict[str, Any]:
    return _main_run_payload(_load_main_state(run_id))


@app.post("/api/aimpact/runs/{run_id}/edit")
def edit_aimpact_article(run_id: str, edit: ArticleEdit) -> dict[str, Any]:
    state = _load_main_state(run_id)
    if state.written_article is None:
        raise HTTPException(status_code=409, detail="This workflow has no article to edit.")
    updated = _new_version(state, _edited_output(state.written_article, edit.article_markdown), "Human edit")
    return _main_run_payload(updated)


@app.post("/api/aimpact/runs/{run_id}/revise")
def revise_aimpact_article(run_id: str, body: RevisionRequest) -> dict[str, Any]:
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
    except Exception as exc:
        logger.exception("Writer revision failed for run %s", run_id)
        raise HTTPException(status_code=502, detail=f"Writer revision failed: {exc}") from exc
    return _main_run_payload(updated)


@app.post("/api/aimpact/runs/{run_id}/review")
def review_aimpact_article(run_id: str) -> dict[str, Any]:
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
    except Exception as exc:
        logger.exception("Review failed for run %s", run_id)
        raise HTTPException(status_code=502, detail=f"Review failed: {exc}") from exc
    return _main_run_payload(reviewed)


@app.put("/api/aimpact/runs/{run_id}/findings")
def save_finding_decisions(run_id: str, body: FindingDecisions) -> dict[str, Any]:
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
    return _main_run_payload(updated)


@app.post("/api/aimpact/runs/{run_id}/approve")
def approve_aimpact_article(run_id: str, body: ApprovalRequest) -> dict[str, Any]:
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
    return _main_run_payload(_save_main_state(run_id, {"draft_approval": record}))


@app.get("/api/aimpact/runs/{run_id}/article.pdf")
def download_aimpact_pdf(run_id: str) -> FileResponse:
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
    job_id = uuid4().hex
    with LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "kind": "general",
            "status": "queued",
            "message": "Workflow queued",
            "completed": 0,
            "total": 7,
        }
    EXECUTOR.submit(_run_general, job_id, request, pdfs)
    return {"job_id": job_id}


@app.get("/api/general/runs/{run_id}")
def get_general_run(run_id: str) -> dict[str, Any]:
    with LOCK:
        result = GENERAL_RESULTS.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="This General Blog run is not available in this server session.")
    return _general_result_payload(result, run_id)


@app.post("/api/general/runs/{run_id}/edit")
def edit_general_article(run_id: str, edit: ArticleEdit) -> dict[str, Any]:
    with LOCK:
        result = GENERAL_RESULTS.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="This General Blog run is not available.")
    updated = result_after_draft_change(result, _edited_output(result.draft, edit.article_markdown))
    with LOCK:
        GENERAL_RESULTS[run_id] = updated
    return _general_result_payload(updated, run_id)


@app.post("/api/general/runs/{run_id}/revise")
def revise_general_article(run_id: str, body: RevisionRequest) -> dict[str, Any]:
    with LOCK:
        result = GENERAL_RESULTS.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="This General Blog run is not available.")
    try:
        usage_run_id = next((record.run_id for record in result.token_usage if record.run_id), run_id)
        with token_usage_scope(usage_run_id, "agent5_human_revision") as records:
            revised = WriterAgent().revise(result.writer_input, result.draft, body.instructions.strip())
        updated = result_after_draft_change(result, revised).model_copy(
            update={"token_usage": merge_token_usage(result.token_usage, records)}
        )
    except Exception as exc:
        logger.exception("General Blog revision failed for run %s", run_id)
        raise HTTPException(status_code=502, detail=f"Writer revision failed: {exc}") from exc
    with LOCK:
        GENERAL_RESULTS[run_id] = updated
    return _general_result_payload(updated, run_id)


@app.post("/api/general/runs/{run_id}/review")
def review_general_article(run_id: str) -> dict[str, Any]:
    with LOCK:
        result = GENERAL_RESULTS.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="This General Blog run is not available.")
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
    except Exception as exc:
        logger.exception("General Blog review failed for run %s", run_id)
        raise HTTPException(status_code=502, detail=f"Review failed: {exc}") from exc
    with LOCK:
        GENERAL_RESULTS[run_id] = updated
    return _general_result_payload(updated, run_id)


@app.get("/api/general/runs/{run_id}/result.json")
def download_general_result(run_id: str) -> dict[str, Any]:
    return get_general_run(run_id)
