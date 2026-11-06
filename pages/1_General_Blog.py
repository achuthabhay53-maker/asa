"""Standalone General Blog page inside the existing Streamlit application."""

from __future__ import annotations

import json
import logging
from uuid import uuid4

import streamlit as st
from pydantic import ValidationError

from src.agents.aimpact_blog.reviewer import ReviewerAgent
from src.agents.aimpact_blog.writer import WriterAgent
from src.models.general_blog import GeneralBlogRequest, GeneralBlogResult
from src.models.token_usage import ClaudeTokenUsage
from src.observability.token_usage import token_usage_totals
from src.observability.token_usage import merge_token_usage, token_usage_scope
from src.utils.general_blog_human import (
    edited_writer_output,
    result_after_draft_change,
    review_passed,
)
from src.orchestrator.general_blog import (
    GeneralBlogProgress,
    GeneralBlogWorkflow,
    GeneralBlogWorkflowError,
)
from src.tools.reference_fetcher import MAX_REFERENCE_BYTES, MAX_UPLOADED_PDFS
from src.utils.output_storage import GENERAL_BLOG_DEBUG_OUTPUT_DIR
from src.ui.theme import (
    inject_theme,
    render_kpis,
    render_pipeline,
    render_sidebar_brand,
    render_token_usage_panel,
    render_topbar,
)

logger = logging.getLogger(__name__)

_GENERAL_BLOG_STEPS = (
    ("intake", "Intake", "Plan the topic and identify references"),
    ("company_research", "Company research", "Research the supplied company website"),
    ("reference_research", "Reference research", "Extract user-provided evidence"),
    ("brief", "Blog brief", "Build an evidence-mapped SEO brief"),
    ("writer", "Agent 5 · Writer", "Write the blog from the approved brief"),
    ("review", "Agent 6 · Reviewer", "Review accuracy, quality, and SEO"),
)
_WORKSPACE_VIEWS = ["Article", "Edit & revise", "Review", "Sources", "Agent outputs"]


def _render_pipeline(slot, statuses: dict[str, str], running: bool = False) -> None:
    pipeline_statuses = dict(statuses)
    if "revision" in pipeline_statuses:
        pipeline_statuses["writer"] = pipeline_statuses["revision"]
    slot.empty()
    with slot.container():
        render_pipeline(
            [
                (name, description, pipeline_statuses.get(key, "idle"))
                for key, name, description in _GENERAL_BLOG_STEPS
            ],
            running=running,
        )


def _render_token_usage(slot, records: list[ClaudeTokenUsage]) -> None:
    if not records:
        return
    input_tokens, output_tokens, missing_count = token_usage_totals(records)
    with slot.container():
        render_token_usage_panel(
            input_tokens,
            output_tokens,
            [
                {
                    "Call": index,
                    "Agent": record.agent_name,
                    "Model": record.model,
                    "Input": record.input_tokens if record.usage_available else "Unavailable",
                    "Output": record.output_tokens if record.usage_available else "Unavailable",
                }
                for index, record in enumerate(records, start=1)
            ],
            missing_count,
        )


def _display_agent_outputs(result: GeneralBlogResult) -> None:
    outputs = (
        ("Intake & research plan", result.plan),
        (
            "Company research",
            [
                source.model_dump(mode="json")
                for source in result.sources
                if source.source_type == "company_website"
            ],
        ),
        (
            "Reference research",
            [
                source.model_dump(mode="json")
                for source in result.sources
                if source.source_type == "user_reference"
            ],
        ),
        ("Blog brief", result.brief),
        ("Writer input", result.writer_input),
        ("Agent 5 · Blog writer", result.draft),
        ("Agent 6 · Reviewer", result.review),
    )
    for index, (label, output) in enumerate(outputs, start=1):
        data = output.model_dump(mode="json") if hasattr(output, "model_dump") else output
        encoded = json.dumps(data, ensure_ascii=False, indent=2)
        with st.expander(label):
            st.download_button(
                "Download JSON",
                data=encoded,
                file_name=f"general_blog_stage_{index}.json",
                mime="application/json",
                key=f"general_blog_output_{index}",
                width="content",
            )
            st.json(data, expanded=False)


def _render_result(result: GeneralBlogResult) -> None:
    input_words = result.writer_input.target_word_count
    words_hint = f"target {input_words:,}" if input_words else "writer-selected length"
    findings_count = len(result.review.findings) + len(result.review.unresolved_items)
    review_label = "Passed" if result.quality_passed else "Human review"
    review_tone = "ok" if result.quality_passed else "warn"
    render_kpis(
        [
            ("Words", f"{result.draft.word_count:,}", words_hint, "neutral"),
            ("Sources", str(len(result.sources)), "company and reference evidence", "neutral"),
            (
                "Review",
                review_label,
                f"{findings_count} finding(s)" if findings_count else "no findings",
                review_tone,
            ),
            ("Review cycles", str(result.review_cycles), "Agent 5 / Agent 6", "neutral"),
            ("Claude calls", str(len(result.token_usage)), "workflow total", "neutral"),
        ]
    )
    for pdf_status in result.uploaded_pdf_statuses:
        message = f"**{pdf_status.file_name}:** {pdf_status.message}"
        if pdf_status.status == "failed":
            st.error(message)
        elif pdf_status.status == "parsed_no_evidence":
            st.warning(message)
        else:
            st.success(message)

    active_view = st.session_state.get("general_blog_active_view", "Article")
    if active_view not in _WORKSPACE_VIEWS:
        active_view = "Article"
    st.segmented_control(
        "Workspace view",
        options=_WORKSPACE_VIEWS,
        key="general_blog_active_view",
        label_visibility="collapsed",
        selection_mode="single",
    )
    st.write("")

    if active_view == "Article":
        article_column, context_column = st.columns([2.7, 1], gap="large")
        with article_column:
            with st.container(border=True):
                st.markdown(f"## {result.draft.title}")
                st.markdown(result.draft.article_markdown)
                st.download_button(
                    "Download Markdown",
                    data=result.draft.article_markdown,
                    file_name="general_blog.md",
                    mime="text/markdown",
                    icon=":material/download:",
                    width="content",
                )
        with context_column:
            with st.container(border=True):
                st.subheader("Blog brief")
                st.write(result.brief.article_objective)
                st.caption(f"Audience: {result.brief.target_audience}")
                st.caption(f"Tone: {result.brief.tone}")
                st.caption(f"Primary keyword: {result.brief.primary_keyword}")
            with st.container(border=True):
                st.subheader("Review status")
                if not result.review_current:
                    st.warning("The draft has changed since its last review.")
                elif result.quality_passed:
                    st.success("The current draft passed review.")
                else:
                    st.warning("The draft needs human review before publication.")
                st.caption(f"Reviewed in {result.review_cycles} cycle(s).")
    elif active_view == "Edit & revise":
        st.subheader("Human final edit")
        st.caption(
            "Edit the article directly or ask Agent 5 for a targeted revision. "
            "Either action marks the existing review as stale."
        )
        version = st.session_state.general_blog_draft_version
        token = st.session_state.general_blog_draft_token
        edit_column, revision_column = st.columns([1.2, 1], gap="large")
        with edit_column:
            with st.form(f"general_blog_edit_{token}_{version}"):
                edited_article = st.text_area(
                    "Article Markdown",
                    value=result.draft.article_markdown,
                    height=560,
                    key=f"general_blog_article_{token}_{version}",
                )
                save_edits = st.form_submit_button(
                    "Save manual edits",
                    type="primary",
                    icon=":material/save:",
                    width="stretch",
                )
            if save_edits:
                if edited_article == result.draft.article_markdown:
                    st.info("No changes to save.")
                else:
                    updated_draft = edited_writer_output(result.draft, edited_article)
                    updated_result = result_after_draft_change(result, updated_draft)
                    st.session_state.general_blog_result = updated_result
                    st.session_state.general_blog_draft_version += 1
                    st.session_state.general_blog_human_action_error = None
                    st.rerun()

        with revision_column:
            st.markdown("#### Ask Agent 5 to revise")
            with st.form(f"general_blog_revision_{token}_{version}"):
                revision_instructions = st.text_area(
                    "Revision instructions",
                    height=180,
                    placeholder=(
                        "For example: make the introduction more concise and "
                        "clarify the limitations of the evidence."
                    ),
                    key=f"general_blog_instructions_{token}_{version}",
                )
                request_revision = st.form_submit_button(
                    "Apply writer revision",
                    icon=":material/auto_fix_high:",
                    width="stretch",
                )
            if request_revision:
                if not revision_instructions.strip():
                    st.error("Enter revision instructions before requesting a revision.")
                else:
                    try:
                        run_id = next(
                            (
                                record.run_id
                                for record in result.token_usage
                                if record.run_id
                            ),
                            "",
                        )
                        with token_usage_scope(run_id, "agent5_human_revision") as records:
                            with st.spinner("Agent 5 is revising the article…"):
                                revised_draft = WriterAgent().revise(
                                    result.writer_input,
                                    result.draft,
                                    revision_instructions,
                                )
                    except Exception as exc:
                        st.session_state.general_blog_human_action_error = (
                            f"Writer revision failed ({type(exc).__name__}): {exc}"
                        )
                    else:
                        updated_result = result_after_draft_change(result, revised_draft)
                        updated_result = updated_result.model_copy(
                            update={
                                "token_usage": merge_token_usage(
                                    result.token_usage,
                                    records,
                                )
                            }
                        )
                        st.session_state.general_blog_result = updated_result
                        st.session_state.general_blog_draft_version += 1
                        st.session_state.general_blog_human_action_error = None
                        st.rerun()
            if st.session_state.general_blog_human_action_error:
                st.error(st.session_state.general_blog_human_action_error)

        st.download_button(
            "Download current Markdown",
            data=result.draft.article_markdown,
            file_name="general_blog.md",
            mime="text/markdown",
            icon=":material/download:",
            width="content",
        )
    elif active_view == "Review":
        with st.container(border=True):
            st.subheader("Agent 6 · Reviewer")
            if not result.review_current:
                st.warning(
                    "The article was edited after the last review. Run Agent 6 again "
                    "to review the current version."
                )
            review_label = (
                "Review current version"
                if not result.review_current
                else "Review this version again"
            )
            if st.button(
                review_label,
                type="primary" if not result.review_current else "secondary",
                icon=":material/fact_check:",
                key=f"general_blog_review_{st.session_state.general_blog_draft_token}_{st.session_state.general_blog_draft_version}",
            ):
                try:
                    run_id = next(
                        (
                            record.run_id
                            for record in result.token_usage
                            if record.run_id
                        ),
                        "",
                    )
                    with token_usage_scope(run_id, "agent6_human_review") as records:
                        with st.spinner("Agent 6 is reviewing the current article…"):
                            current_review = ReviewerAgent().review(
                                result.writer_input,
                                result.draft,
                            )
                except Exception as exc:
                    st.session_state.general_blog_human_action_error = (
                        f"Reviewer failed ({type(exc).__name__}): {exc}"
                    )
                else:
                    st.session_state.general_blog_result = result.model_copy(
                        update={
                            "review": current_review,
                            "review_current": True,
                            "quality_passed": review_passed(current_review),
                            "token_usage": merge_token_usage(result.token_usage, records),
                        }
                    )
                    st.session_state.general_blog_human_action_error = None
                    st.rerun()
            if st.session_state.general_blog_human_action_error:
                st.error(st.session_state.general_blog_human_action_error)
            if result.review_current:
                st.write(result.review.summary)
            if result.review_current and result.review.findings:
                for finding in result.review.findings:
                    with st.container(border=True):
                        st.markdown(
                            f"**{finding.severity.upper()} · {finding.area}**"
                        )
                        st.write(finding.issue)
                        st.caption(finding.recommendation)
                        if finding.target_text:
                            st.caption(f"Target text: {finding.target_text}")
            elif result.review_current:
                st.success("No review findings were returned.")
            if result.review_current and result.review.unresolved_items:
                st.markdown("**Unresolved items**")
                for item in result.review.unresolved_items:
                    st.write(f"- {item}")
    elif active_view == "Sources":
        if result.uploaded_pdf_statuses:
            st.subheader("Uploaded PDF processing")
            for pdf_status in result.uploaded_pdf_statuses:
                details = (
                    f"{pdf_status.extracted_text_characters:,} text characters · "
                    f"{pdf_status.evidence_count} evidence item(s)"
                )
                with st.container(border=True):
                    if pdf_status.status == "failed":
                        st.error(f"**{pdf_status.file_name}** · {pdf_status.message}")
                    elif pdf_status.status == "parsed_no_evidence":
                        st.warning(f"**{pdf_status.file_name}** · {pdf_status.message}")
                        st.caption(details)
                    else:
                        st.success(f"**{pdf_status.file_name}** · {pdf_status.message}")
                        st.caption(details)
        if result.sources:
            st.subheader("Extracted evidence")
            for source in result.sources:
                source_kind = (
                    "Company website"
                    if source.source_type == "company_website"
                    else "Uploaded PDF"
                    if not source.source_url
                    else "User-provided reference"
                )
                title = source.source_title or source.source_url or source.evidence_id
                with st.container(border=True):
                    st.markdown(f"**{source.evidence_id} · {title}**")
                    st.caption(f"{source_kind} · relevance: {source.relevance}")
                    st.write(source.claim)
                    if source.evidence_quote:
                        st.markdown(f"> {source.evidence_quote}")
                    if source.status == "verified":
                        st.caption(
                            "Quote matched the source text; the claim is not "
                            "independently scientifically verified."
                        )
                    if source.source_url:
                        st.markdown(f"[Open source]({source.source_url})")
                    if source.publication_date:
                        st.caption(f"Publication date: {source.publication_date}")
        else:
            st.info("No source-backed claims were extracted.")
    else:
        _display_agent_outputs(result)

    if result.warnings:
        with st.expander("Workflow notes", icon=":material/info:"):
            for warning in result.warnings:
                st.warning(warning)

    st.download_button(
        "Download workflow result (JSON)",
        data=json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
        file_name="general_blog_result.json",
        mime="application/json",
        icon=":material/data_object:",
        width="content",
    )


st.set_page_config(
    page_title="General Blog | BlogGen",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()

for key, default in (
    ("general_blog_result", None),
    ("general_blog_error", None),
    ("general_blog_token_usage", []),
    ("general_blog_progress", {}),
    ("general_blog_active_view", "Article"),
    ("general_blog_draft_token", uuid4().hex),
    ("general_blog_draft_version", 0),
    ("general_blog_human_action_error", None),
    ("general_blog_run_id", None),
):
    st.session_state.setdefault(key, default)

with st.sidebar:
    render_sidebar_brand()
    with st.form("general_blog_request", border=False):
        st.markdown('<div class="acp-side-title">New general blog</div>', unsafe_allow_html=True)
        company_name = st.text_input("Company name", placeholder="Ethosh")
        company_website = st.text_input(
            "Company website",
            placeholder="https://ethosh.com",
        )
        prompt = st.text_area(
            "Blog topic or instructions",
            height=150,
            placeholder="How is VR transforming medical device training?",
        )
        target_audience = st.text_input(
            "Target audience (optional)",
            placeholder="Medical device training leaders",
        )
        tone = st.text_input(
            "Tone or style (optional)",
            placeholder="Authoritative and accessible",
        )
        with st.expander("Research references (optional)", icon=":material/library_books:"):
            reference_text = st.text_area(
                "Reference URLs",
                height=100,
                placeholder="One URL per line",
                help="Public HTML and PDF links are supported. URLs in your prompt are also collected.",
            )
            reference_pdfs = st.file_uploader(
                "Upload research PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                max_upload_size=MAX_REFERENCE_BYTES // (1024 * 1024),
                help=(
                    f"Upload up to {MAX_UPLOADED_PDFS} text-based PDFs "
                    "when a reference link cannot be fetched. Maximum 15 MB each; "
                    "scanned PDFs that require OCR are not supported."
                ),
            )
        target_word_count = st.number_input(
            "Target words",
            min_value=0,
            max_value=10_000,
            value=0,
            step=100,
            help="Leave at 0 to let the writer choose a suitable length.",
        )
        submitted = st.form_submit_button(
            "Research and generate",
            type="primary",
            width="stretch",
            icon=":material/play_arrow:",
        )

topbar_slot = st.empty()
pipeline_slot = st.empty()
run_slot = st.container()
token_slot = st.empty()

if submitted:
    st.session_state.general_blog_result = None
    st.session_state.general_blog_error = None
    st.session_state.general_blog_token_usage = []
    st.session_state.general_blog_draft_token = uuid4().hex
    st.session_state.general_blog_draft_version = 0
    st.session_state.general_blog_human_action_error = None
    st.session_state.general_blog_run_id = None
    statuses: dict[str, str] = {"intake": "running"}
    st.session_state.general_blog_progress = dict(statuses)

    reference_urls = [line.strip() for line in reference_text.splitlines() if line.strip()]
    uploaded_reference_pdfs = [
        (uploaded_file.name, uploaded_file.getvalue())
        for uploaded_file in reference_pdfs
    ]
    try:
        if len(uploaded_reference_pdfs) > MAX_UPLOADED_PDFS:
            raise ValueError(f"Upload no more than {MAX_UPLOADED_PDFS} PDFs per run.")
        request = GeneralBlogRequest(
            company_name=company_name,
            company_website=company_website,
            prompt=prompt,
            target_audience=target_audience or None,
            tone=tone or None,
            reference_urls=reference_urls,
            target_word_count=int(target_word_count) if target_word_count else None,
        )
    except ValidationError as exc:
        details = [
            f"{'.'.join(str(part) for part in error['loc']) or 'Input'}: {error['msg']}"
            for error in exc.errors()
        ]
        st.session_state.general_blog_error = {
            "run_id": None,
            "stage": "Input validation",
            "error_type": "ValidationError",
            "message": "\n".join(details),
        }
    except ValueError as exc:
        st.session_state.general_blog_error = {
            "run_id": None,
            "stage": "Input validation",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    else:
        _render_pipeline(pipeline_slot, statuses, running=True)
        with run_slot:
            with st.status(
                "Running the General Blog agents",
                state="running",
                expanded=True,
            ) as status:
                progress = st.progress(0, text="Starting General Blog workflow")
                stage_log = st.empty()

                def update_progress(event: GeneralBlogProgress) -> None:
                    st.session_state.general_blog_run_id = event.run_id
                    if event.stage != "complete":
                        statuses[event.stage] = (
                            "warn" if event.status == "skipped" else event.status
                        )
                        st.session_state.general_blog_progress = dict(statuses)
                        _render_pipeline(pipeline_slot, statuses, running=True)
                        progress.progress(
                            min(
                                99,
                                event.completed_stages * 100 // event.total_stages,
                            ),
                            text=f"{event.label}: {event.message}",
                        )
                        stage_log.write(
                            f"{'✓' if event.status in {'done', 'skipped'} else '✕' if event.status == 'error' else '⏳'} "
                            f"**{event.label}:** {event.message}"
                        )
                    else:
                        progress.progress(100, text="All General Blog agents complete")
                        status.update(
                            label="General Blog workflow complete",
                            state="complete",
                            expanded=False,
                        )
                    st.session_state.general_blog_token_usage = event.token_usage
                    _render_token_usage(token_slot, event.token_usage)

                try:
                    result = GeneralBlogWorkflow(
                        progress_callback=update_progress,
                    ).run(
                        request,
                        uploaded_reference_pdfs=uploaded_reference_pdfs,
                    )
                except GeneralBlogWorkflowError as exc:
                    st.session_state.general_blog_token_usage = exc.token_usage
                    st.session_state.general_blog_error = {
                        "run_id": exc.run_id,
                        "stage": exc.label,
                        "error_type": type(exc.cause).__name__,
                        "message": str(exc.cause),
                    }
                    status.update(
                        label=f"{exc.label} failed",
                        state="error",
                        expanded=True,
                    )
                except Exception as exc:
                    logger.exception("General Blog workflow initialization failed")
                    st.session_state.general_blog_error = {
                        "run_id": None,
                        "stage": "Workflow initialization",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                    status.update(
                        label="General Blog workflow failed",
                        state="error",
                        expanded=True,
                    )
                else:
                    st.session_state.general_blog_result = result
                    st.session_state.general_blog_token_usage = result.token_usage

result = st.session_state.general_blog_result
error = st.session_state.general_blog_error
run_id = next(
    (
        record.run_id
        for record in st.session_state.general_blog_token_usage
        if record.run_id
    ),
    error.get("run_id") if error else None,
)
if isinstance(result, GeneralBlogResult):
    header_title = result.draft.title
    header_subtitle = f"Standalone research workflow · Run {run_id}" if run_id else "Standalone research workflow"
    header_status = (
        ("Review passed", "ok")
        if result.quality_passed
        else ("Human review needed", "draft")
    )
elif error:
    header_title = "General Blog workspace"
    header_subtitle = f"{error['stage']} · Run {run_id}" if run_id else error["stage"]
    header_status = ("Failed", "error")
else:
    header_title = "General Blog workspace"
    header_subtitle = (
        f"Standalone research workflow · Run {run_id}"
        if run_id
        else "Create an evidence-grounded blog without AImpact context."
    )
    header_status = ("Ready", "idle")
with topbar_slot.container():
    render_topbar(header_title, header_subtitle, *header_status)

statuses = st.session_state.general_blog_progress or {}
_render_pipeline(pipeline_slot, statuses)
if run_id:
    st.caption(f"Run ID: `{run_id}`")
    st.caption(f"Debug JSON files: `{GENERAL_BLOG_DEBUG_OUTPUT_DIR / run_id}`")

if error:
    st.error(f"{error['stage']} failed. See the details below and application logs for diagnostics.")
    with st.expander("Error details"):
        st.code(f"{error['error_type']}: {error['message']}", language="text")

usage = (
    result.token_usage
    if isinstance(result, GeneralBlogResult)
    else st.session_state.general_blog_token_usage
)
_render_token_usage(token_slot, usage)

if isinstance(result, GeneralBlogResult):
    _render_result(result)
else:
    with st.container(border=True):
        st.subheader("Ready to create your general blog?")
        st.write(
            "Enter a company, website, and topic in the sidebar. The workflow "
            "researches the site and any references you provide, then writes "
            "with Agent 5 and reviews with Agent 6."
        )
        how_columns = st.columns(3)
        for column, heading, detail in zip(
            how_columns,
            ("1 · Research", "2 · Write", "3 · Review"),
            (
                "Gather evidence from the company website and optional research references.",
                "Build an SEO-focused brief and draft the article using the evidence.",
                "Check the draft and apply revisions when the reviewer identifies blocking issues.",
            ),
        ):
            with column:
                st.markdown(f"**{heading}**")
                st.caption(detail)
