"""AImpact Content Agent: Streamlit workspace for the LangGraph workflow.

Run with: streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from aca.article_sections import split_article_sections
from aca.agents.source_intelligence import TOP_N_SOURCES
from aca.agents.writer import WriterAgent
from aca.models.writer import WriterInput, WriterOutput, WrittenSection
from aca.output_storage import clear_agent_outputs, save_agent_output
from aca.review_annotations import anchorable_finding_indexes, render_annotated_markdown
from aca.orchestrator.checkpointer import postgres_checkpointer, thread_config
from aca.orchestrator.pipeline import PipelineState, _build_writer_input, build_pipeline
from aca.pdf_export import build_article_pdf
from ui_theme2 import (
    inject_theme,
    render_checklist,
    render_empty_state,
    render_kpis,
    render_meter,
    render_outline,
    render_pipeline,
    render_sidebar_brand,
    render_status_pill,
    render_topbar,
    render_versions,
    severity_badge,
)

st.set_page_config(
    page_title="AImpact Content Agent",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()


# ----------------------------- HELPERS ------------------------------
VERIFY_RE = re.compile(r"\[VERIFY:[^\]]*\]")
SEVERITY_ORDER = ["critical", "high", "medium", "low"]
VIEWS = ["Article", "Edit & revise", "Review", "Approve", "Agent outputs"]
QUICK_PROMPTS = [
    "Make the tone more conversational",
    "Tighten the introduction",
    "Add a clearer call to action",
    "Make the headings more specific",
    "Trim the article closer to the target length",
]
AGENT_STEPS = [
    ("agent1", "Agent 1", "Prompt", "run_error"),
    ("agent2", "Agent 2", "Sources", "run_error"),
    ("agent3", "Agent 3", "Ground truth", "run_error"),
    ("agent4", "Agent 4", "Content brief", "agent4_error"),
    ("agent5", "Agent 5", "Blog writer", "agent5_error"),
    ("agent6", "Agent 6", "Reviewer", "reviewer_error"),
]


def as_dict(value: Any) -> dict:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value if isinstance(value, dict) else {}


def download_json(label: str, obj: Any, filename: str, key: str):
    st.download_button(
        label,
        json.dumps(as_dict(obj), ensure_ascii=False, indent=2),
        file_name=filename,
        mime="application/json",
        key=key,
        icon=":material/download:",
    )


def article_word_count(article: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", article))


def output_with_edited_article(output: WriterOutput, article: str) -> WriterOutput:
    lines = article.splitlines()
    headings = [
        (index, line.strip()[3:].strip())
        for index, line in enumerate(lines)
        if re.match(r"^##\s+", line.strip())
    ]
    sections = []
    for position, (start, heading) in enumerate(headings):
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        content = "\n".join(lines[start + 1 : end]).strip()
        evidence_ids = [
            evidence_id
            for claim in output.claims_used
            if claim.claim.casefold() in content.casefold()
            for evidence_id in claim.evidence_ids
        ]
        sections.append(
            WrittenSection(
                heading=heading,
                content_markdown=content,
                evidence_ids_used=list(dict.fromkeys(evidence_ids)),
            )
        )

    h1 = next(
        (line.strip()[2:].strip() for line in lines if re.match(r"^#\s+", line.strip())),
        output.title,
    )
    unresolved = list(output.unresolved_items)
    review_note = "Human-edited article; recheck claim-to-evidence support before approval."
    if review_note not in unresolved:
        unresolved.append(review_note)
    seo_note = "Review SEO title and meta description after human edits."
    if seo_note not in unresolved:
        unresolved.append(seo_note)
    return output.model_copy(
        update={
            "title": h1,
            "article_markdown": article,
            "word_count": article_word_count(article),
            "sections": sections,
            "claims_used": [],
            "unresolved_items": unresolved,
        }
    )


def draft_fingerprint(output: Any) -> str:
    serialized = json.dumps(as_dict(output), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def finding_id_of(finding: dict) -> str:
    return hashlib.sha256(
        json.dumps(finding, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]


def persist_run_checkpoint(values: dict[str, Any]) -> None:
    run_id = st.session_state.workflow_run_id
    if not run_id:
        raise RuntimeError("Cannot persist human review state without a workflow run ID.")
    config = thread_config(str(run_id))
    with postgres_checkpointer() as checkpointer:
        workflow = build_pipeline(checkpointer)
        snapshot = workflow.get_state(config)
        if not snapshot.values:
            raise RuntimeError(f"No checkpointed workflow state exists for run {run_id}.")
        workflow.update_state(config, values)


def record_draft_version(output: WriterOutput, source: str, instructions: str = "") -> None:
    version = int(st.session_state.get("draft_version", 1)) + 1
    st.session_state.agent5 = output
    st.session_state.draft_version = version
    st.session_state.draft_approval = None
    st.session_state.proposed_revision = None
    st.session_state.agent6 = None
    st.session_state.reviewer_error = None
    st.session_state.reviewed_draft_fingerprint = None
    st.session_state.finding_statuses = {}
    st.session_state.writer_rewrite_error = None
    versions = list(st.session_state.get("draft_versions") or [])
    versions.append(
        {
            "version": version,
            "source": source,
            "instructions": instructions,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "output": output.model_dump(mode="json"),
        }
    )
    persist_run_checkpoint(
        {
            "written_article": output,
            "draft_version": version,
            "draft_versions": versions,
            "draft_approval": None,
            "proposed_revision": None,
            "review_output": None,
            "reviewed_draft_fingerprint": None,
            "finding_statuses": {},
        }
    )
    st.session_state.draft_versions = versions
    save_agent_output(5, output, OUTPUT_DIR, run_id=st.session_state.workflow_run_id)


@st.cache_data(show_spinner=False)
def cached_article_pdf(article: str) -> bytes:
    """Build the PDF once per article instead of on every Streamlit rerun."""
    return build_article_pdf(article)


def go_to(view: str) -> None:
    """Button callback: switch the active workspace view."""
    st.session_state.active_view = view


def apply_quick_prompt(pill_key: str, text_key: str) -> None:
    choice = st.session_state.get(pill_key)
    if choice:
        st.session_state[text_key] = choice


def pipeline(slot, running: bool = False) -> None:
    items = []
    for key, name, desc, err_key in AGENT_STEPS:
        value = st.session_state.get(key)
        if value is not None:
            verification = as_dict(value).get("verification") if key == "agent3" else None
            state = "warn" if verification and not verification.get("is_verified", False) else "done"
        elif st.session_state.get(err_key):
            state = "error"
        else:
            state = "idle"
        items.append((name, desc, state))
    with slot.container():
        render_pipeline(items, running=running)


# ------------------------------- STATE ------------------------------
KEYS = (
    "agent1",
    "agent2",
    "agent3",
    "agent4",
    "agent5",
    "agent6",
    "workflow_run_id",
    "run_error",
    "agent4_error",
    "agent5_error",
    "reviewer_error",
    "agent3_notice",
    "writer_input",
    "draft_versions",
    "draft_version",
    "draft_approval",
    "proposed_revision",
    "target_word_count",
    "writer_rewrite_error",
    "reviewed_draft_fingerprint",
    "finding_statuses",
)
OUTPUT_DIR = Path(__file__).resolve().parent
for key in KEYS:
    st.session_state.setdefault(key, None)
st.session_state.setdefault("active_view", "Article")


# ------------------------------ SIDEBAR -----------------------------
with st.sidebar:
    render_sidebar_brand()

    with st.form("run_workflow", border=False):
        st.markdown('<div class="acp-side-title">New run</div>', unsafe_allow_html=True)
        campaign_id = st.number_input("Campaign ID", min_value=1, value=81, step=1)
        prompt_id = st.text_input("Prompt ID", value="1409")
        company_link = st.text_input("Customer website", value="https://alamarbio.com")
        customer_name = st.text_input("Customer name", placeholder="Optional, inferred from Agent 1")
        sc1, sc2 = st.columns(2)
        top_n = sc1.number_input(
            "Sources", min_value=1, max_value=20, value=int(TOP_N_SOURCES), step=1,
            help="How many sources Agent 2 analyzes.",
        )
        target_word_count = sc2.number_input(
            "Target words",
            min_value=500,
            max_value=5000,
            value=800,
            step=100,
            help="Agent 5 reports the actual word count and still returns the blog if it differs.",
        )
        submitted = st.form_submit_button(
            "Run Agents 1–6", type="primary", width="stretch", icon=":material/play_arrow:"
        )

    with st.expander("Resume a saved run", icon=":material/history:"):
        with st.form("resume_workflow_form", border=False):
            resume_run_id = st.text_input("Workflow run ID")
            resume_submitted = st.form_submit_button("Load saved run", width="stretch")
        if resume_submitted:
            try:
                config = thread_config(resume_run_id.strip())
                with postgres_checkpointer() as checkpointer:
                    saved_workflow = build_pipeline(checkpointer)
                    snapshot = saved_workflow.get_state(config)
                if not snapshot.values:
                    raise ValueError(f"No saved workflow was found for run ID {resume_run_id!r}.")
                saved_state = PipelineState.model_validate(snapshot.values)
                st.session_state.workflow_run_id = saved_state.run_id
                st.session_state.target_word_count = saved_state.target_word_count
                st.session_state.agent1 = saved_state.prompt_analysis
                st.session_state.agent2 = saved_state.source_analysis
                st.session_state.agent3 = saved_state.guide
                st.session_state.agent4 = saved_state.content_brief
                st.session_state.agent5 = saved_state.written_article
                st.session_state.agent6 = saved_state.review_output
                st.session_state.draft_version = saved_state.draft_version
                st.session_state.draft_versions = saved_state.draft_versions
                st.session_state.draft_approval = saved_state.draft_approval
                st.session_state.proposed_revision = saved_state.proposed_revision
                st.session_state.reviewed_draft_fingerprint = saved_state.reviewed_draft_fingerprint
                st.session_state.finding_statuses = saved_state.finding_statuses
                st.session_state.writer_input = (
                    _build_writer_input(saved_state)
                    if all(
                        (
                            saved_state.prompt_analysis,
                            saved_state.source_analysis,
                            saved_state.guide,
                            saved_state.content_brief,
                        )
                    )
                    else None
                )
                st.session_state.run_error = None
                st.session_state.agent4_error = None
                st.session_state.agent5_error = None
                st.session_state.reviewer_error = None
                if saved_state.written_article and not st.session_state.draft_versions:
                    initial_output = saved_state.written_article
                    st.session_state.draft_version = 1
                    st.session_state.draft_versions = [
                        {
                            "version": 1,
                            "source": "Agent 5 initial draft",
                            "instructions": "",
                            "created_at": "Recovered from workflow checkpoint",
                            "output": initial_output.model_dump(mode="json"),
                        }
                    ]
                st.session_state.active_view = "Article"
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load workflow: {type(exc).__name__}: {exc}")

    run_id_slot = st.empty()


# --------------------------- MAIN SHELL -----------------------------
topbar_slot = st.empty()
pipeline_slot = st.empty()
run_slot = st.container()


# --------------------------- RUN WORKFLOW ----------------------------
if submitted:
    for key in KEYS:
        st.session_state[key] = None
    if not str(prompt_id).strip():
        st.session_state.run_error = "Please enter a Prompt ID."
    elif not str(company_link).strip():
        st.session_state.run_error = "Please enter a customer website."
    else:
        state = PipelineState(
            campaign_id=int(campaign_id),
            prompt_id=str(prompt_id).strip(),
            website=company_link.strip(),
            company_name=customer_name.strip(),
            top_n=int(top_n),
            max_questions=3,
            target_word_count=int(target_word_count),
            review_requested=True,
        )
        st.session_state.workflow_run_id = state.run_id
        st.session_state.target_word_count = int(target_word_count)
        try:
            clear_agent_outputs(OUTPUT_DIR)
            outputs = (
                ("prompt_analysis", "agent1", 1, "Agent 1 — Prompt Intelligence"),
                ("source_analysis", "agent2", 2, "Agent 2 — Source Intelligence"),
                ("guide", "agent3", 3, "Agent 3 — Ground Truth"),
                ("content_brief", "agent4", 4, "Agent 4 — Content Brief"),
                ("written_article", "agent5", 5, "Agent 5 — Blog Writer"),
                ("review_output", "agent6", 6, "Agent 6 — Reviewer"),
            )
            completed = set()
            pipeline(pipeline_slot, running=True)
            with run_slot:
                with st.status("Running the LangGraph workflow", state="running", expanded=True) as status:
                    progress = st.progress(0, text="Starting workflow…")
                    try:
                        with postgres_checkpointer() as checkpointer:
                            workflow = build_pipeline(checkpointer)
                            config = thread_config(state.run_id)
                            for snapshot in workflow.stream(state, config=config, stream_mode="values"):
                                state = PipelineState.model_validate(snapshot)
                                for state_field, session_key, output_number, label in outputs:
                                    result = getattr(state, state_field)
                                    if result is not None and session_key not in completed:
                                        st.session_state[session_key] = result
                                        save_agent_output(
                                            output_number,
                                            as_dict(result),
                                            OUTPUT_DIR,
                                            run_id=state.run_id,
                                        )
                                        completed.add(session_key)
                                        st.write(f"✓ {label} complete")
                                        pipeline(pipeline_slot, running=True)
                                progress.progress(
                                    min(100, len(completed) * 100 // len(outputs)),
                                    text=f"{len(completed)} of {len(outputs)} agents complete",
                                )

                        if state.errors:
                            message = " ".join(state.errors)
                            if (
                                state.guide is not None
                                and state.guide.verification is not None
                                and not state.guide.verification.is_verified
                            ):
                                st.session_state.agent3_notice = message
                                status.update(
                                    label="Stopped by prompt/customer verification",
                                    state="complete",
                                    expanded=True,
                                )
                            elif state.guide is None or state.guide.verification is None:
                                st.session_state.run_error = message
                                status.update(label="LangGraph workflow stopped", state="error", expanded=True)
                            elif state.content_brief is None or not state.content_brief.sections:
                                st.session_state.agent4_error = message
                                status.update(label="LangGraph workflow stopped", state="error", expanded=True)
                            elif state.written_article is None:
                                st.session_state.agent5_error = message
                                status.update(label="LangGraph workflow stopped", state="error", expanded=True)
                            else:
                                st.session_state.reviewer_error = message
                                status.update(label="LangGraph workflow stopped", state="error", expanded=True)
                        elif len(completed) == len(outputs):
                            progress.progress(100, text="All agents complete")
                            status.update(label="LangGraph workflow complete", state="complete", expanded=False)
                        else:
                            raise RuntimeError(
                                "Workflow ended before all agents completed without reporting an error."
                            )
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        if state.prompt_analysis is None or state.source_analysis is None or state.guide is None:
                            st.session_state.run_error = message
                        elif state.content_brief is None:
                            st.session_state.agent4_error = message
                        elif state.written_article is not None:
                            st.session_state.reviewer_error = message
                        else:
                            st.session_state.agent5_error = message
                        status.update(label="LangGraph workflow failed", state="error", expanded=True)
                        traceback.print_exc()
            if state.written_article is not None:
                st.session_state.writer_input = _build_writer_input(state)
                st.session_state.draft_version = 1
                st.session_state.draft_approval = None
                st.session_state.proposed_revision = None
                st.session_state.draft_versions = [
                    {
                        "version": 1,
                        "source": "Agent 5 initial draft",
                        "instructions": "",
                        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "output": state.written_article.model_dump(mode="json"),
                    }
                ]
                persist_run_checkpoint(
                    {
                        "draft_version": 1,
                        "draft_versions": st.session_state.draft_versions,
                        "draft_approval": None,
                        "proposed_revision": None,
                        "reviewed_draft_fingerprint": (
                            draft_fingerprint(state.written_article)
                            if state.review_output is not None
                            else None
                        ),
                        "finding_statuses": {},
                    }
                )
                if state.review_output is not None:
                    st.session_state.reviewed_draft_fingerprint = draft_fingerprint(state.written_article)
                st.session_state.active_view = "Article"
        except Exception as exc:
            st.session_state.run_error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()


# ------------------------- DERIVED PAGE STATE ------------------------
a1, a2, a3, a4, a5, a6 = (st.session_state[k] for k in ("agent1", "agent2", "agent3", "agent4", "agent5", "agent6"))
has_any = any(x is not None for x in (a1, a2, a3, a4, a5, a6))

if a5 is not None:
    writer_output = as_dict(a5)
    typed_output = WriterOutput.model_validate(writer_output)
    fingerprint = draft_fingerprint(typed_output)
    article = str(writer_output.get("article_markdown") or "").strip()
    title = str(writer_output.get("title") or "Generated blog")
    draft_version = int(st.session_state.draft_version or 1)
    draft_token = str(st.session_state.workflow_run_id or "draft")
    target_words = int(st.session_state.target_word_count or target_word_count)
    words = article_word_count(article)
    sections = split_article_sections(article) if article else None
    h2_headings = [m.strip() for m in re.findall(r"^##\s+(.+)$", article, flags=re.MULTILINE)]
    faq_count = len((sections.faqs if sections else None) or [])
    verify_count = len(VERIFY_RE.findall(article))

    review_is_current = a6 is not None and st.session_state.reviewed_draft_fingerprint == fingerprint
    current_review = as_dict(a6) if review_is_current else {}
    findings = current_review.get("findings") or []
    saved_statuses = dict(st.session_state.finding_statuses or {})
    status_of: dict[str, str] = {}
    for raw in findings:
        fid = finding_id_of(as_dict(raw))
        status_of[fid] = st.session_state.get(
            f"finding_{draft_token}_{draft_version}_{fid}", saved_statuses.get(fid, "Open")
        )
    open_count = sum(1 for s in status_of.values() if s == "Open")
    all_findings_resolved = open_count == 0

    approval = st.session_state.draft_approval or {}
    if approval.get("version") == draft_version and review_is_current and not all_findings_resolved:
        st.session_state.draft_approval = None
        approval = {}
    is_approved = approval.get("version") == draft_version and approval.get("fingerprint") == fingerprint
    export_status = "approved" if is_approved else "draft"


# ------------------------------ TOP BAR ------------------------------
pipeline(pipeline_slot)
run_id = str(st.session_state.workflow_run_id or "")
if a5 is not None:
    topbar_title = title
    topbar_sub = f"Run {run_id}" if run_id else "Saved draft"
    if is_approved:
        topbar_status = (f"Version {draft_version} approved", "ok")
    else:
        topbar_status = (f"Version {draft_version} draft", "draft")
else:
    topbar_title = "Content workspace"
    topbar_sub = f"Run {run_id}" if run_id else "Set up a run in the sidebar to begin."
    topbar_status = ("Ready", "idle") if not has_any else ("In progress", "draft")
with topbar_slot.container():
    render_topbar(topbar_title, topbar_sub, *topbar_status)

if run_id:
    with run_id_slot.container():
        st.caption("Current run ID")
        st.code(run_id, language=None)


# ---------------------------- STATUS MESSAGES ------------------------
for err_key, label in [
    ("run_error", "Agents 1–3"),
    ("agent4_error", "Agent 4"),
    ("agent5_error", "Agent 5"),
    ("reviewer_error", "Agent 6"),
]:
    if st.session_state.get(err_key):
        st.error(f"{label}: {st.session_state[err_key]}")
if st.session_state.agent3_notice:
    st.warning(st.session_state.agent3_notice)


# ------------------------------- VIEWS -------------------------------
def view_article() -> None:
    main, rail = st.columns([2.7, 1], gap="large")
    review_output = as_dict(a6) if a6 else {}
    all_findings = review_output.get("findings") or []
    anchorable = anchorable_finding_indexes(article, all_findings)

    with main:
        with st.container(border=True, key="blog_header"):
            st.markdown(
                f'<h1 class="blog-title">{html.escape(title)}</h1>',
                unsafe_allow_html=True,
            )
            word_color = "green" if abs(words - target_words) <= target_words * 0.10 else "orange"
            chips = [f":{word_color}-badge[{words:,} of {target_words:,} words]"]
            if h2_headings:
                chips.append(f":gray-badge[{len(h2_headings)} sections]")
            if faq_count:
                chips.append(f":gray-badge[{faq_count} FAQs]")
            if verify_count:
                chips.append(f":orange-badge[⚠ {verify_count} claims need a source]")
            else:
                chips.append(":green-badge[✓ no open source flags]")
            st.markdown(" ".join(chips))

        if st.session_state.reviewer_error:
            st.warning(f"Agent 6 review failed: {st.session_state.reviewer_error}")
        elif review_output:
            severity_counts: dict[str, int] = {}
            for finding in all_findings:
                sev = str(as_dict(finding).get("severity") or "medium").lower()
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            counts_text = ", ".join(
                f"{severity_counts[s]} {s}" for s in SEVERITY_ORDER if s in severity_counts
            )
            summary = f"Agent 6 found {len(all_findings)} issue(s)"
            if counts_text:
                summary += f" ({counts_text})"
            st.caption(summary + ". Hover over highlighted text to see the finding and recommendation.")
            unanchored = len(all_findings) - len(anchorable)
            if unanchored:
                st.caption(
                    f"{unanchored} finding(s) could not be matched to one exact passage. "
                    "Find them in the Review view."
                )

        def render_reviewed_markdown(markdown: str) -> None:
            if markdown.strip():
                rendered, _ = render_annotated_markdown(
                    markdown, all_findings, allowed_finding_indexes=anchorable
                )
                st.html(rendered)

        if not article:
            st.warning("Agent 5 returned an empty article_markdown field.")
        else:
            with st.container(border=True, key="blog_card"):
                if sections.introduction:
                    render_reviewed_markdown(sections.introduction)
                if sections.main_content:
                    render_reviewed_markdown(sections.main_content)
                if sections.conclusion:
                    st.subheader("Conclusion")
                    render_reviewed_markdown(sections.conclusion)
                if sections.faqs or sections.faq_fallback:
                    st.subheader("Frequently Asked Questions")
                    if sections.faq_intro:
                        st.markdown(sections.faq_intro)
                    if sections.faqs:
                        for question, answer in sections.faqs:
                            with st.expander(question):
                                render_reviewed_markdown(answer)
                    else:
                        render_reviewed_markdown(sections.faq_fallback)

    with rail:
        with st.container(key="card_next"):
            st.markdown('<div class="acp-card-title">Next step</div>', unsafe_allow_html=True)
            if is_approved:
                st.markdown('<p class="acp-card-text">This version is approved. Export it below.</p>', unsafe_allow_html=True)
            elif not review_is_current:
                st.markdown(
                    '<p class="acp-card-text">This version has not been reviewed yet.</p>',
                    unsafe_allow_html=True,
                )
                st.button("Go to review", on_click=go_to, args=("Review",), type="primary",
                          width="stretch", key="next_review")
            elif open_count:
                st.markdown(
                    f'<p class="acp-card-text">{open_count} review finding(s) still need a decision.</p>',
                    unsafe_allow_html=True,
                )
                st.button("Resolve findings", on_click=go_to, args=("Review",), type="primary",
                          width="stretch", key="next_resolve")
            else:
                st.markdown(
                    '<p class="acp-card-text">Review is complete. You can approve this version.</p>',
                    unsafe_allow_html=True,
                )
                st.button("Go to approval", on_click=go_to, args=("Approve",), type="primary",
                          width="stretch", key="next_approve")

        with st.container(key="card_export"):
            st.markdown(f'<div class="acp-card-title">Export {export_status} version</div>', unsafe_allow_html=True)
            if article:
                st.download_button(
                    "Download PDF",
                    data=cached_article_pdf(article),
                    file_name=f"agent5_v{draft_version}_{export_status}_{prompt_id}.pdf",
                    mime="application/pdf",
                    key="download_agent5_pdf",
                    type="primary",
                    icon=":material/picture_as_pdf:",
                    width="stretch",
                )
                st.download_button(
                    "Download Markdown",
                    article + "\n",
                    file_name=f"agent5_v{draft_version}_{export_status}_{prompt_id}.md",
                    mime="text/markdown",
                    key="download_agent5_blog",
                    icon=":material/download:",
                    width="stretch",
                )
                with st.popover("Copy Markdown", icon=":material/content_copy:", width="stretch"):
                    st.caption("Use the copy icon at the top right of the box.")
                    st.code(article, language="markdown", wrap_lines=True)

        with st.container(key="card_outline"):
            st.markdown('<div class="acp-card-title">Outline</div>', unsafe_allow_html=True)
            render_outline(h2_headings)

        seo_title = str(writer_output.get("seo_title") or "").strip()
        meta_description = str(writer_output.get("meta_description") or "").strip()
        if seo_title or meta_description:
            with st.container(key="card_seo"):
                st.markdown('<div class="acp-card-title">SEO metadata</div>', unsafe_allow_html=True)
                for label, value, limit in (
                    ("SEO title", seo_title, 60),
                    ("Meta description", meta_description, 160),
                ):
                    if value:
                        st.caption(label)
                        st.code(value, language=None, wrap_lines=True)
                        render_meter(label, len(value), limit)


def view_edit() -> None:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Edit the article")
        st.caption("Markdown. Saving creates a new version and clears the approval and the current review.")
        with st.form(f"draft_editor_form_{draft_token}_v{draft_version}"):
            edited_article = st.text_area(
                "Article Markdown",
                value=article,
                height=560,
                key=f"draft_editor_{draft_token}_v{draft_version}",
                label_visibility="collapsed",
            )
            save_manual_edits = st.form_submit_button(
                "Save my edits", type="primary", icon=":material/save:"
            )
        st.caption(f"{words:,} words in this version")
        if save_manual_edits:
            if edited_article == article:
                st.info("No changes to save.")
            else:
                updated = output_with_edited_article(typed_output, edited_article)
                record_draft_version(updated, "Human edit")
                st.rerun()

    with right:
        st.subheader("Ask Agent 5 to revise")
        writer_input_value = st.session_state.writer_input
        if writer_input_value is None:
            st.info(
                "Revisions need the writer inputs from a full run. Start a new run, "
                "or load a saved run that includes all agent outputs."
            )
        else:
            text_key = f"revision_instructions_{draft_token}_v{draft_version}"
            pill_key = f"quick_prompt_{draft_token}_v{draft_version}"
            st.pills(
                "Quick prompts",
                options=QUICK_PROMPTS,
                selection_mode="single",
                key=pill_key,
                on_change=apply_quick_prompt,
                args=(pill_key, text_key),
            )
            with st.form(f"writer_revision_form_{draft_token}_v{draft_version}"):
                revision_instructions = st.text_area(
                    "Instructions",
                    height=130,
                    placeholder="Describe the wording, tone, style, or content change you want. The writer keeps to the supplied evidence.",
                    key=text_key,
                )
                request_revision = st.form_submit_button(
                    "Apply writer revision", icon=":material/auto_fix_high:", width="stretch"
                )
            if request_revision:
                if not revision_instructions.strip():
                    st.error("Enter instructions before requesting a revision.")
                else:
                    try:
                        writer_input_model = (
                            writer_input_value
                            if isinstance(writer_input_value, WriterInput)
                            else WriterInput.model_validate(writer_input_value)
                        )
                        with st.spinner("Agent 5 is revising the draft…"):
                            proposal = WriterAgent().revise(
                                writer_input_model, typed_output, revision_instructions
                            )
                        record_draft_version(proposal, "Agent 5 revision", revision_instructions.strip())
                        st.session_state.writer_rewrite_error = None
                    except Exception as exc:
                        st.session_state.writer_rewrite_error = f"{type(exc).__name__}: {exc}"
                        st.error(f"Writer revision failed: {st.session_state.writer_rewrite_error}")
                    else:
                        st.rerun()
        if st.session_state.writer_rewrite_error:
            st.error(f"Writer revision: {st.session_state.writer_rewrite_error}")

        st.subheader("Version history")
        versions = st.session_state.draft_versions or []
        if versions:
            render_versions(versions, draft_version)
            restorable = [item for item in versions if item["version"] != draft_version]
            if restorable:
                with st.form(f"restore_draft_form_{draft_token}_v{draft_version}"):
                    selected_version = st.selectbox(
                        "Restore an earlier version as a new version",
                        options=[item["version"] for item in restorable],
                    )
                    restore_version = st.form_submit_button("Restore selected version", width="stretch")
                if restore_version:
                    selected = next(item for item in restorable if item["version"] == selected_version)
                    restored = WriterOutput.model_validate(selected["output"])
                    record_draft_version(restored, f"Restored version {selected_version}")
                    st.rerun()
        else:
            st.caption("No versions yet.")


def run_review() -> None:
    try:
        config = thread_config(str(st.session_state.workflow_run_id))
        with st.status("Reviewing the current draft", state="running") as review_status:
            with postgres_checkpointer() as checkpointer:
                workflow = build_pipeline(checkpointer)
                workflow.invoke({"review_requested": True, "review_output": None}, config=config)
                reviewed_snapshot = workflow.get_state(config)
            if not reviewed_snapshot.values.get("review_output"):
                raise RuntimeError("Workflow ended without a Reviewer result.")
            reviewed_output = reviewed_snapshot.values["review_output"]
            review_status.update(label="Review completed", state="complete", expanded=False)
        st.session_state.agent6 = reviewed_output
        st.session_state.reviewer_error = None
        st.session_state.reviewed_draft_fingerprint = fingerprint
        st.session_state.finding_statuses = {}
        persist_run_checkpoint(
            {"reviewed_draft_fingerprint": fingerprint, "finding_statuses": {}}
        )
        save_agent_output(6, as_dict(reviewed_output), OUTPUT_DIR, run_id=st.session_state.workflow_run_id)
        st.rerun()
    except Exception as exc:
        st.session_state.reviewer_error = f"{type(exc).__name__}: {exc}"
        st.error(f"Review failed: {st.session_state.reviewer_error}")


def view_review() -> None:
    if not review_is_current:
        with st.container(key="card_review_start"):
            st.markdown('<div class="acp-card-title">Review this version</div>', unsafe_allow_html=True)
            st.markdown(
                '<p class="acp-card-text">Agent 6 checks the draft against the brief and evidence and '
                "lists issues with recommendations. Any edit creates a new version that needs a new review.</p>",
                unsafe_allow_html=True,
            )
            if st.button(
                "Review this version",
                key=f"review_draft_{draft_token}_v{draft_version}",
                type="primary",
                icon=":material/fact_check:",
            ):
                run_review()
        return

    unresolved_items = current_review.get("unresolved_items") or []
    if not findings:
        st.success("Agent 6 found no issues in this draft.")
    else:
        resolved = len(findings) - open_count
        h1, h2 = st.columns([3, 1])
        with h1:
            st.subheader("Resolve review findings")
            st.progress(resolved / len(findings), text=f"{resolved} of {len(findings)} findings decided")
        with h2:
            st.metric("Still open", open_count)

        entries = []
        for raw_finding in findings:
            finding = as_dict(raw_finding)
            entries.append((finding_id_of(finding), finding))
        entries.sort(
            key=lambda item: SEVERITY_ORDER.index(str(item[1].get("severity") or "medium").lower())
            if str(item[1].get("severity") or "medium").lower() in SEVERITY_ORDER
            else len(SEVERITY_ORDER)
        )

        with st.form(f"finding_status_form_{draft_token}_v{draft_version}", border=False):
            for finding_id, finding in entries:
                status_key = f"finding_{draft_token}_{draft_version}_{finding_id}"
                if status_key not in st.session_state:
                    st.session_state[status_key] = saved_statuses.get(finding_id, "Open")
                with st.container(key=f"finding_card_{finding_id}"):
                    st.markdown(
                        f'<div class="acp-finding-head">{severity_badge(str(finding.get("severity") or ""))}'
                        f'<span class="acp-finding-area">{html.escape(str(finding.get("area") or "Content"))}</span>'
                        f'<br>{html.escape(str(finding.get("issue") or ""))}</div>',
                        unsafe_allow_html=True,
                    )
                    st.write(finding.get("recommendation", ""))
                    chosen_status = st.radio(
                        "Decision",
                        options=["Open", "Resolved", "Dismissed"],
                        horizontal=True,
                        key=status_key,
                    )
                    if chosen_status == "Dismissed":
                        st.text_input(
                            "Reason for dismissal (optional)",
                            key=f"dismiss_reason_{draft_token}_{draft_version}_{finding_id}",
                        )
            save_findings = st.form_submit_button(
                "Save finding decisions", type="primary", icon=":material/save:"
            )
        if save_findings:
            new_statuses = {fid: st.session_state[f"finding_{draft_token}_{draft_version}_{fid}"] for fid, _ in entries}
            st.session_state.finding_statuses = new_statuses
            persist_run_checkpoint({"finding_statuses": new_statuses})
            if any(s == "Open" for s in new_statuses.values()):
                st.session_state.draft_approval = None
                persist_run_checkpoint({"draft_approval": None})
            st.rerun()

    if unresolved_items:
        with st.expander("Evidence or brief items to verify before approval"):
            for item in unresolved_items:
                st.markdown(f"- {item}")


def view_approve() -> None:
    unresolved_items = current_review.get("unresolved_items") or []
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Approval checklist")
        with st.container(key="card_checklist"):
            render_checklist(
                [
                    (
                        "This version has been reviewed",
                        "ok" if review_is_current else "todo",
                        f"Version {draft_version} was reviewed by Agent 6."
                        if review_is_current
                        else "Run the review on this exact version first.",
                    ),
                    (
                        "Review findings are decided",
                        "ok" if review_is_current and all_findings_resolved else "todo",
                        "Every finding is resolved or dismissed."
                        if review_is_current and all_findings_resolved
                        else f"{open_count} finding(s) still open." if review_is_current else "Waiting for the review.",
                    ),
                    (
                        "Source flags",
                        "ok" if not verify_count else "warn",
                        "No [VERIFY] markers in the article."
                        if not verify_count
                        else f"{verify_count} [VERIFY] marker(s). Resolve them or acknowledge below.",
                    ),
                    (
                        "Items to verify",
                        "ok" if not unresolved_items else "warn",
                        "Nothing flagged by the reviewer."
                        if not unresolved_items
                        else f"{len(unresolved_items)} evidence or brief item(s) to check.",
                    ),
                ]
            )
        if unresolved_items:
            with st.expander("Evidence or brief items to verify before approval"):
                for item in unresolved_items:
                    st.markdown(f"- {item}")

    with right:
        st.subheader("Approve")
        with st.container(key="card_approve"):
            if is_approved:
                render_status_pill(f"Version {draft_version} approved", "ok")
                st.markdown(
                    f'<p class="acp-card-text">Approved on {html.escape(str(approval.get("approved_at", "")))}. '
                    "Export the final files from the Article view.</p>",
                    unsafe_allow_html=True,
                )
                st.button("Open article", on_click=go_to, args=("Article",), width="stretch", key="approved_open")
            elif review_is_current and all_findings_resolved:
                with st.form(f"approve_draft_form_{draft_token}_v{draft_version}", border=False):
                    acknowledgement = st.checkbox(
                        "I reviewed this exact version and accept responsibility for any remaining "
                        "evidence flags or unresolved items."
                    )
                    approve_draft = st.form_submit_button(
                        "Approve final draft", type="primary", width="stretch", icon=":material/verified:"
                    )
                if approve_draft:
                    if not acknowledgement:
                        st.error("Confirm that you reviewed this version before approving.")
                    else:
                        approval_record = {
                            "version": draft_version,
                            "fingerprint": fingerprint,
                            "approved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                        }
                        persist_run_checkpoint({"draft_approval": approval_record})
                        st.session_state.draft_approval = approval_record
                        st.rerun()
            else:
                st.markdown(
                    '<p class="acp-card-text">Approval unlocks once this version is reviewed and every finding is decided.</p>',
                    unsafe_allow_html=True,
                )
                st.button("Go to review", on_click=go_to, args=("Review",), type="primary",
                          width="stretch", key="approve_goto_review")


def view_outputs() -> None:
    st.subheader("Agent outputs")
    st.caption("Raw output from each agent. Each run also saves these as agent1_output.json through agent6_output.json.")
    labels = [f"{name} · {desc}" for _, name, desc, _ in AGENT_STEPS]
    for tab, (key, name, desc, _) in zip(st.tabs(labels), AGENT_STEPS):
        with tab:
            value = st.session_state.get(key)
            if value is None:
                st.info(f"{name} has not produced output for this run.")
            else:
                download_json("Download JSON", value, f"{key}_output.json", f"dl_{key}")
                st.json(as_dict(value), expanded=1)


# ------------------------------ RENDER -------------------------------
if not has_any and not any(
    st.session_state.get(k) for k in ("run_error", "agent4_error", "agent5_error", "reviewer_error", "agent3_notice")
):
    render_empty_state()
elif a5 is None:
    if st.session_state.agent5_error:
        st.warning("The brief may be ready, but Agent 5 failed. See the error above.")
    else:
        st.info("Agent 5 has not completed.")
    if has_any:
        view_outputs()
else:
    review_label = "No review yet"
    review_tone = "neutral"
    review_hint = "Run Agent 6 on this version"
    if review_is_current:
        if findings:
            review_label = f"{len(findings) - open_count}/{len(findings)}"
            review_hint = "findings decided"
            review_tone = "ok" if all_findings_resolved else "warn"
        else:
            review_label, review_hint, review_tone = "Clean", "no issues found", "ok"
    on_target = abs(words - target_words) <= target_words * 0.10
    render_kpis(
        [
            ("Words", f"{words:,}", f"target {target_words:,}", "ok" if on_target else "warn"),
            ("Sections", str(len(h2_headings)), f"{faq_count} FAQs", "neutral"),
            (
                "Source flags",
                str(verify_count),
                "claims need a source" if verify_count else "none open",
                "warn" if verify_count else "ok",
            ),
            ("Review", review_label, review_hint, review_tone),
            ("Version", f"v{draft_version}", "approved" if is_approved else "draft", "ok" if is_approved else "neutral"),
        ]
    )

    if st.session_state.active_view not in VIEWS:
        st.session_state.active_view = "Article"
    st.segmented_control(
        "Workspace view",
        options=VIEWS,
        key="active_view",
        label_visibility="collapsed",
        selection_mode="single",
    )
    st.write("")
    active = st.session_state.active_view or "Article"
    if active == "Article":
        view_article()
    elif active == "Edit & revise":
        view_edit()
    elif active == "Review":
        view_review()
    elif active == "Approve":
        view_approve()
    else:
        view_outputs()