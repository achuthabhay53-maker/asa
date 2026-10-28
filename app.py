"""AImpact Content Agent: Streamlit UI for the LangGraph workflow.

Run with: streamlit run app.py
"""

from __future__ import annotations

import html
import hashlib
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
from ui_theme import inject_theme, render_header, render_pipeline, render_status_pill


# ------------------------------- UI ---------------------------------
st.set_page_config(page_title="AImpact Content Agent", page_icon="✦", layout="wide")
inject_theme()


# ----------------------------- HELPERS ------------------------------
VERIFY_RE = re.compile(r"\[VERIFY:[^\]]*\]")


def as_dict(value: Any) -> dict:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value if isinstance(value, dict) else {}


def show_json(label: str, obj: Any):
    with st.expander(label):
        st.json(as_dict(obj))


def download_json(label: str, obj: Any, filename: str, key: str):
    st.download_button(
        label,
        json.dumps(as_dict(obj), ensure_ascii=False, indent=2),
        file_name=filename,
        mime="application/json",
        key=key,
        width="stretch",
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
    save_agent_output(
        5,
        output,
        OUTPUT_DIR,
        run_id=st.session_state.workflow_run_id,
    )


@st.cache_data(show_spinner=False)
def cached_article_pdf(article: str) -> bytes:
    """Build the PDF once per article instead of on every Streamlit rerun."""
    return build_article_pdf(article)


def pipeline():
    steps = [
        ("agent1", "Agent 1", "Prompt", "run_error"),
        ("agent2", "Agent 2", "Sources", "run_error"),
        ("agent3", "Agent 3", "Ground truth", "run_error"),
        ("agent4", "Agent 4", "Content brief", "agent4_error"),
        ("agent5", "Agent 5", "Blog writer", "agent5_error"),
        ("agent6", "Agent 6", "Reviewer", "reviewer_error"),
    ]
    items = []
    for key, name, desc, err_key in steps:
        value = st.session_state.get(key)
        if value is not None:
            verification = as_dict(value).get("verification") if key == "agent3" else None
            state = "warn" if verification and not verification.get("is_verified", False) else "done"
        elif st.session_state.get(err_key):
            state = "error"
        else:
            state = "idle"
        items.append((name, desc, state))
    render_pipeline(items)


def seo_metadata_panel(seo_title: str, meta_description: str):
    """Copy-ready SEO fields with character counts."""
    fields = (
        ("SEO title", seo_title, 60),
        ("Meta description", meta_description, 160),
    )
    if not any(value for _, value, _ in fields):
        return
    with st.expander("SEO metadata", icon=":material/search:"):
        for label, value, limit in fields:
            if not value:
                continue
            st.markdown(f"**{label}**")
            st.code(value, language=None, wrap_lines=True)  # st.code has a built-in copy button
            count = len(value)
            if count > limit:
                st.caption(f":red[{count} characters, over the recommended {limit}]")
            else:
                st.caption(f"{count} of {limit} recommended characters")


def render_final_output(
    article: str,
    title: str,
    seo_title: str,
    meta_description: str,
    draft_version: int,
    is_approved: bool,
    prompt_id: str,
    target_word_count: int,
) -> None:
    st.subheader("Final blog")
    _, page, _ = st.columns([1, 6, 1])
    with page:
        if not article:
            st.warning("Agent 5 returned an empty article_markdown field.")
        else:
            sections = split_article_sections(article)
            words = article_word_count(article)
            h2_count = len(re.findall(r"^##\s", article, flags=re.MULTILINE))
            faq_count = len(sections.faqs or [])
            verify_count = len(VERIFY_RE.findall(article))

            with st.container(border=True, key="blog_header"):
                st.markdown(
                    f'<h1 class="blog-title">{html.escape(title)}</h1>',
                    unsafe_allow_html=True,
                )
                word_color = "green" if abs(words - target_word_count) <= target_word_count * 0.10 else "orange"
                chips = [f":{word_color}-badge[{words:,} of {target_word_count:,} words]"]
                if h2_count:
                    chips.append(f":gray-badge[{h2_count} sections]")
                if faq_count:
                    chips.append(f":gray-badge[{faq_count} FAQs]")
                if verify_count:
                    chips.append(f":orange-badge[⚠ {verify_count} claims need a source]")
                else:
                    chips.append(":green-badge[✓ no open source flags]")
                st.markdown(" ".join(chips))

                export_status = "approved" if is_approved else "draft"
                b1, b2, b3, _spacer = st.columns([1.6, 1.9, 1.45, 2.1])
                with b1:
                    st.download_button(
                        f"Download {export_status} PDF",
                        data=cached_article_pdf(article),
                        file_name=f"agent5_v{draft_version}_{export_status}_{prompt_id}.pdf",
                        mime="application/pdf",
                        key="download_agent5_pdf",
                        type="primary",
                        icon=":material/download:",
                        width="stretch",
                    )
                with b2:
                    st.download_button(
                        f"Download {export_status} Markdown",
                        article + "\n",
                        file_name=f"agent5_v{draft_version}_{export_status}_{prompt_id}.md",
                        mime="text/markdown",
                        key="download_agent5_blog",
                        icon=":material/download:",
                        width="stretch",
                    )
                with b3:
                    with st.popover("Copy Markdown", icon=":material/content_copy:", width="stretch"):
                        st.caption("Use the copy icon at the top right of the box.")
                        st.code(article, language="markdown", wrap_lines=True)

            seo_metadata_panel(seo_title, meta_description)
            review_output = as_dict(st.session_state.agent6) if st.session_state.agent6 else {}
            findings = review_output.get("findings") or []
            anchorable = anchorable_finding_indexes(article, findings)
            if st.session_state.reviewer_error:
                st.warning(f"Agent 6 review failed: {st.session_state.reviewer_error}")
            elif review_output:
                severity_counts: dict[str, int] = {}
                for finding in findings:
                    severity = str(as_dict(finding).get("severity") or "medium").lower()
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                counts_text = ", ".join(
                    f"{count} {severity}"
                    for severity, count in sorted(
                        severity_counts.items(),
                        key=lambda item: ["critical", "high", "medium", "low"].index(item[0])
                        if item[0] in {"critical", "high", "medium", "low"}
                        else 4,
                    )
                )
                review_summary = f"Agent 6 · {len(findings)} finding(s)"
                if counts_text:
                    review_summary += f" ({counts_text})"
                st.caption(review_summary + ". Hover over highlighted text for the finding and recommendation.")
                unanchored_count = len(findings) - len(anchorable)
                if unanchored_count:
                    st.caption(
                        f"{unanchored_count} finding(s) could not be matched to one unique exact passage; "
                        "they remain available in review decisions below."
                    )
            else:
                st.info("Reviewer findings are not available for this draft yet.")

            def render_reviewed_markdown(markdown: str) -> None:
                if markdown.strip():
                    rendered, _ = render_annotated_markdown(
                        markdown,
                        findings,
                        allowed_finding_indexes=anchorable,
                    )
                    st.html(rendered)

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


# ------------------------------ INPUT UI ----------------------------
render_header()
pipeline()

with st.form("run_workflow", border=True):
    st.subheader("Run a test case")
    c1, c2, c3 = st.columns([1, 1, 2])
    campaign_id = c1.number_input("Campaign ID", min_value=1, value=81, step=1)
    prompt_id = c2.text_input("Prompt ID", value="1409")
    company_link = c3.text_input("Customer website", value="https://alamarbio.com")
    c4, c5 = st.columns([2, 1])
    customer_name = c4.text_input("Customer name", placeholder="Optional if inferred from Agent 1")
    top_n = c5.number_input("Sources to analyze", min_value=1, max_value=20, value=int(TOP_N_SOURCES), step=1)
    target_word_count = st.number_input(
        "Blog target word count",
        min_value=500,
        max_value=5000,
        value=800,
        step=100,
        help="Adjust the target as needed. Agent 5 reports the actual word count but will still return the blog if it differs.",
    )
    submitted = st.form_submit_button("Run Agents 1–6", type="primary", width="stretch")

with st.expander("Resume a saved workflow"):
    with st.form("resume_workflow_form"):
        resume_run_id = st.text_input("Workflow run ID")
        resume_submitted = st.form_submit_button("Load saved run")
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
            st.success(f"Loaded workflow {saved_state.run_id}.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load workflow: {type(exc).__name__}: {exc}")


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
        except Exception as exc:
            st.session_state.run_error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()


# ---------------------------- STATUS & RESULTS -----------------------
for key, label in [
    ("run_error", "Agents 1–3"),
    ("agent4_error", "Agent 4"),
    ("agent5_error", "Agent 5"),
    ("reviewer_error", "Agent 6"),
]:
    if st.session_state.get(key):
        st.error(f"{label}: {st.session_state[key]}")
if st.session_state.agent3_notice:
    st.warning(st.session_state.agent3_notice)
pipeline()

a1, a2, a3 = st.session_state.agent1, st.session_state.agent2, st.session_state.agent3
a4, a5, a6 = (
    st.session_state.agent4,
    st.session_state.agent5,
    st.session_state.agent6,
)
if all(x is None for x in (a1, a2, a3, a4, a5, a6)):
    st.info("Enter a Campaign ID, Prompt ID, and customer website, then click **Run Agents 1–6**.")

if any(x is not None for x in (a1, a2, a3, a4, a5, a6)):
    st.subheader("")

    with st.container():
        if a5 is None:
            if st.session_state.agent5_error:
                st.warning("The brief may be ready, but Agent 5 failed. See the error above.")
            else:
                st.info("Agent 5 has not completed.")
        else:
            writer_output = as_dict(a5)
            article = str(writer_output.get("article_markdown") or "").strip()
            title = str(writer_output.get("title") or "Generated blog")
            seo_title = str(writer_output.get("seo_title") or "").strip()
            meta_description = str(writer_output.get("meta_description") or "").strip()
            draft_version = int(st.session_state.draft_version or 1)
            draft_token = str(st.session_state.workflow_run_id or "draft")
            typed_output = WriterOutput.model_validate(writer_output)

            approval = st.session_state.draft_approval or {}
            is_approved = (
                approval.get("version") == draft_version
                and approval.get("fingerprint") == draft_fingerprint(typed_output)
            )
            render_final_output(
                article,
                title,
                seo_title,
                meta_description,
                draft_version,
                is_approved,
                str(prompt_id),
                int(st.session_state.target_word_count or target_word_count),
            )

            st.subheader("Human review and editing")
            render_status_pill(draft_version, is_approved)
            st.caption("Saved workflow run ID")
            st.code(str(st.session_state.workflow_run_id), language=None)

            with st.form(f"draft_editor_form_{draft_token}_v{draft_version}"):
                edited_article = st.text_area(
                    "Edit the article Markdown",
                    value=article,
                    height=420,
                    key=f"draft_editor_{draft_token}_v{draft_version}",
                    help="Save your edits to create a new version. Any prior approval and Agent 6 review will be cleared.",
                )
                save_manual_edits = st.form_submit_button("Save my edits", type="primary")
            if save_manual_edits:
                updated = output_with_edited_article(typed_output, edited_article)
                if edited_article == article:
                    st.info("No changes to save.")
                else:
                    record_draft_version(updated, "Human edit")
                    st.rerun()

            writer_input_value = st.session_state.writer_input
            if writer_input_value is not None:
                with st.form(f"writer_revision_form_{draft_token}_v{draft_version}"):
                    revision_instructions = st.text_area(
                        "Ask Agent 5 to revise this draft",
                        height=110,
                        placeholder="Describe the wording, tone, style, or specific content change you want. The writer will keep to the supplied evidence.",
                        key=f"revision_instructions_{draft_token}_v{draft_version}",
                    )
                    request_revision = st.form_submit_button("Apply Writer revision")
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
                                    writer_input_model,
                                    typed_output,
                                    revision_instructions,
                                )
                            record_draft_version(
                                proposal,
                                "Agent 5 revision",
                                revision_instructions.strip(),
                            )
                            st.session_state.writer_rewrite_error = None
                        except Exception as exc:
                            st.session_state.writer_rewrite_error = f"{type(exc).__name__}: {exc}"
                            st.error(f"Writer revision failed: {st.session_state.writer_rewrite_error}")
                        else:
                            st.rerun()

            if st.session_state.writer_rewrite_error:
                st.error(f"Writer revision: {st.session_state.writer_rewrite_error}")

            versions = st.session_state.draft_versions or []
            if versions:
                with st.expander("Version history"):
                    for version_record in reversed(versions):
                        st.markdown(
                            f"**Version {version_record['version']} · {version_record['source']}** "
                            f"· {version_record['created_at']}"
                        )
                        if version_record.get("instructions"):
                            st.caption(version_record["instructions"])
                    restorable = [item for item in versions if item["version"] != draft_version]
                    if restorable:
                        with st.form(f"restore_draft_form_{draft_token}_v{draft_version}"):
                            selected_version = st.selectbox(
                                "Restore an earlier version as a new version",
                                options=[item["version"] for item in restorable],
                            )
                            restore_version = st.form_submit_button("Restore selected version")
                        if restore_version:
                            selected = next(item for item in restorable if item["version"] == selected_version)
                            restored = WriterOutput.model_validate(selected["output"])
                            record_draft_version(restored, f"Restored version {selected_version}")
                            st.rerun()

            review_is_current = (
                st.session_state.agent6 is not None
                and st.session_state.reviewed_draft_fingerprint == draft_fingerprint(typed_output)
            )
            current_review = as_dict(st.session_state.agent6) if review_is_current else {}
            findings = current_review.get("findings") or []
            statuses = dict(st.session_state.finding_statuses or {})
            all_findings_resolved = True
            if review_is_current and findings:
                st.markdown("#### Resolve review findings")
                with st.form(f"finding_status_form_{draft_token}_v{draft_version}"):
                    for raw_finding in findings:
                        finding = as_dict(raw_finding)
                        finding_id = hashlib.sha256(
                            json.dumps(finding, ensure_ascii=False, sort_keys=True).encode("utf-8")
                        ).hexdigest()[:12]
                        status_key = f"finding_{draft_token}_{draft_version}_{finding_id}"
                        if status_key not in st.session_state:
                            st.session_state[status_key] = statuses.get(finding_id, "Open")
                        with st.expander(
                            f"{finding.get('severity', 'Review')} · {finding.get('area', 'Content')}: "
                            f"{finding.get('issue', '')}",
                            expanded=True,
                        ):
                            st.write(finding.get("recommendation", ""))
                            chosen_status = st.radio(
                                "Finding status",
                                options=["Open", "Resolved", "Dismissed"],
                                horizontal=True,
                                key=status_key,
                            )
                            statuses[finding_id] = chosen_status
                            if chosen_status == "Dismissed":
                                st.text_input(
                                    "Reason for dismissal (optional)",
                                    key=f"dismiss_reason_{draft_token}_{draft_version}_{finding_id}",
                                )
                            if chosen_status == "Open":
                                all_findings_resolved = False
                    save_findings = st.form_submit_button("Save finding decisions")
                if save_findings:
                    st.session_state.finding_statuses = statuses
                    persist_run_checkpoint({"finding_statuses": statuses})
                    if not all_findings_resolved:
                        st.session_state.draft_approval = None
                        persist_run_checkpoint({"draft_approval": None})
                    st.rerun()
                current_approval = st.session_state.draft_approval or {}
                if (
                    current_approval.get("version") == draft_version
                    and not all_findings_resolved
                ):
                    st.session_state.draft_approval = None
                    st.rerun()
            elif review_is_current:
                st.success("Agent 6 found no issues in this draft.")
            else:
                if st.button(
                    "Review this version",
                    key=f"review_draft_{draft_token}_v{draft_version}",
                    type="primary",
                ):
                    try:
                        config = thread_config(str(st.session_state.workflow_run_id))
                        with st.status("Reviewing the current draft", state="running") as review_status:
                            with postgres_checkpointer() as checkpointer:
                                workflow = build_pipeline(checkpointer)
                                workflow.invoke(
                                    {"review_requested": True, "review_output": None},
                                    config=config,
                                )
                                reviewed_snapshot = workflow.get_state(config)
                            if not reviewed_snapshot.values.get("review_output"):
                                raise RuntimeError("Workflow ended without a Reviewer result.")
                            reviewed_output = reviewed_snapshot.values["review_output"]
                            review_status.update(
                                label="Review completed",
                                state="complete",
                                expanded=False,
                            )
                        st.session_state.agent6 = reviewed_output
                        st.session_state.reviewer_error = None
                        st.session_state.reviewed_draft_fingerprint = draft_fingerprint(typed_output)
                        st.session_state.finding_statuses = {}
                        persist_run_checkpoint(
                            {
                                "reviewed_draft_fingerprint": st.session_state.reviewed_draft_fingerprint,
                                "finding_statuses": {},
                            }
                        )
                        save_agent_output(
                            6,
                            as_dict(reviewed_output),
                            OUTPUT_DIR,
                            run_id=st.session_state.workflow_run_id,
                        )
                        st.rerun()
                    except Exception as exc:
                        st.session_state.reviewer_error = f"{type(exc).__name__}: {exc}"
                        st.error(f"Review failed: {st.session_state.reviewer_error}")

            unresolved_items = current_review.get("unresolved_items") or []
            if unresolved_items:
                with st.expander("Evidence or brief items to verify before approval"):
                    for item in unresolved_items:
                        st.markdown(f"- {item}")
            has_verify_flags = bool(VERIFY_RE.search(article))
            if has_verify_flags:
                st.warning("This draft contains [VERIFY] markers. Resolve them or explicitly acknowledge them before approval.")
            if review_is_current and all_findings_resolved:
                with st.form(f"approve_draft_form_{draft_token}_v{draft_version}"):
                    acknowledgement = st.checkbox(
                        "I reviewed this exact version and accept responsibility for any remaining evidence flags or unresolved items."
                    )
                    approve_draft = st.form_submit_button("Approve final draft", type="primary")
                if approve_draft:
                    if not acknowledgement:
                        st.error("Confirm that you reviewed this version before approving.")
                    else:
                        approval_record = {
                            "version": draft_version,
                            "fingerprint": draft_fingerprint(typed_output),
                            "approved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                        }
                        persist_run_checkpoint({"draft_approval": approval_record})
                        st.session_state.draft_approval = approval_record
                        st.rerun()
