"""AImpact Content Agent: Streamlit UI for the LangGraph workflow.

Run with: streamlit run app.py
"""

from __future__ import annotations

import html
import json
import re
import traceback
from pathlib import Path
from typing import Any

import streamlit as st
from aca.article_sections import split_article_sections
from aca.agents.source_intelligence import TOP_N_SOURCES
from aca.output_storage import clear_agent_outputs, save_agent_output
from aca.orchestrator.pipeline import PipelineState, build_pipeline
from aca.pdf_export import build_article_pdf


# ------------------------------- UI ---------------------------------
st.set_page_config(page_title="AImpact Content Agent", page_icon="✦", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top:1.25rem; padding-bottom:3rem; max-width:1450px;}
h1 {font-size:1.85rem; font-weight:750;}
[data-testid="stMetric"] {background:rgba(128,128,128,.07); padding:12px; border-radius:12px;}

/* ---- Blog view ---- */
h1.blog-title {
    font-size:2.05rem; line-height:1.25; font-weight:750;
    margin:0 0 .6rem 0; padding:0; letter-spacing:-.01em;
}
/* hide the hover "link" icon Streamlit adds next to headings */
.st-key-blog_header [data-testid="stHeaderActionElements"],
.st-key-blog_header_reviewed [data-testid="stHeaderActionElements"],
.st-key-blog_card [data-testid="stHeaderActionElements"],
.st-key-blog_card_reviewed [data-testid="stHeaderActionElements"] {display:none;}

/* comfortable reading typography inside the article card */
.st-key-blog_card,
.st-key-blog_card_reviewed {padding:.5rem .75rem;}
.st-key-blog_card p,
.st-key-blog_card li,
.st-key-blog_card_reviewed p,
.st-key-blog_card_reviewed li {font-size:1.05rem; line-height:1.8;}
.st-key-blog_card h2,
.st-key-blog_card_reviewed h2 {
    font-size:1.55rem; font-weight:700; line-height:1.3;
    margin-top:1.8rem; padding-bottom:.4rem;
    border-bottom:1px solid rgba(128,128,128,.25);
}
.st-key-blog_card h3,
.st-key-blog_card_reviewed h3 {font-size:1.2rem; font-weight:650; margin-top:1.2rem;}
.st-key-blog_card table,
.st-key-blog_card_reviewed table {width:100%; font-size:.95rem;}
.st-key-blog_card th,
.st-key-blog_card_reviewed th {background:rgba(128,128,128,.12);}
</style>
""",
    unsafe_allow_html=True,
)


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


@st.cache_data(show_spinner=False)
def cached_article_pdf(article: str) -> bytes:
    """Build the PDF once per article instead of on every Streamlit rerun."""
    return build_article_pdf(article)


def style_verify_markers(text: str) -> str:
    """Show '[VERIFY: ...]' placeholders as small orange badges in the preview.

    Only the on-screen preview changes; the PDF/Markdown downloads keep the raw text.
    """
    return VERIFY_RE.sub(":orange-badge[⚠ needs source]", text)


def pipeline():
    st.subheader("Workflow pipeline")
    steps = [
        ("agent1", "Agent 1", "Prompt", "run_error"),
        ("agent2", "Agent 2", "Sources", "run_error"),
        ("agent3", "Agent 3", "Ground truth", "run_error"),
        ("agent4", "Agent 4", "Content brief", "agent4_error"),
        ("agent5", "Agent 5", "Blog writer", "agent5_error"),
        ("agent6", "Agent 6", "Reviewer", "reviewer_error"),
    ]
    cols = st.columns(6)
    for col, (key, name, desc, err_key) in zip(cols, steps):
        with col:
            if st.session_state.get(key) is not None:
                value = st.session_state.get(key)
                verification = as_dict(value).get("verification") if key == "agent3" else None
                if verification and not verification.get("is_verified", False):
                    st.warning(f"⚠ {name}")
                else:
                    st.success(f"✓ {name}")
            elif st.session_state.get(err_key):
                st.error(f"✕ {name}")
            else:
                st.info(f"○ {name}")
            st.caption(desc)


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
)
OUTPUT_DIR = Path(__file__).resolve().parent
for key in KEYS:
    st.session_state.setdefault(key, None)


# ------------------------------ INPUT UI ----------------------------
st.title("✦ AImpact Content Agent")
st.caption("Prompt intelligence → source research → ground truth → content brief → blog → optional review")
st.caption("Each valid run refreshes agent1_output.json through agent5_output.json; Agent 6 runs on request.")
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
    submitted = st.form_submit_button("Run Agents 1–5", type="primary", width="stretch")


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
        )
        st.session_state.workflow_run_id = state.run_id
        try:
            clear_agent_outputs(OUTPUT_DIR)
            outputs = (
                ("prompt_analysis", "agent1", 1, "Agent 1 — Prompt Intelligence"),
                ("source_analysis", "agent2", 2, "Agent 2 — Source Intelligence"),
                ("guide", "agent3", 3, "Agent 3 — Ground Truth"),
                ("content_brief", "agent4", 4, "Agent 4 — Content Brief"),
                ("written_article", "agent5", 5, "Agent 5 — Blog Writer"),
            )
            completed = set()
            with st.status("Running the LangGraph workflow", state="running", expanded=True) as status:
                progress = st.progress(0, text="Starting workflow…")
                try:
                    for snapshot in build_pipeline().stream(state, stream_mode="values"):
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
                            min(100, len(completed) * 20),
                            text=f"{len(completed)} of 5 agents complete",
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
                        else:
                            st.session_state.agent5_error = message
                            status.update(label="LangGraph workflow stopped", state="error", expanded=True)
                    elif len(completed) == 5:
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
                    else:
                        st.session_state.agent5_error = message
                    status.update(label="LangGraph workflow failed", state="error", expanded=True)
                    traceback.print_exc()
        except Exception as exc:
            st.session_state.run_error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()


# ---------------------------- STATUS & RESULTS -----------------------
for key, label in [("run_error", "Agents 1–3"), ("agent4_error", "Agent 4"), ("agent5_error", "Agent 5")]:
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
    st.info("Enter a Campaign ID, Prompt ID, and customer website, then click **Run Agents 1–5**.")

if any(x is not None for x in (a1, a2, a3, a4, a5, a6)):
    writer_tab, reviewer_tab = st.tabs(["Agent 5 - Blog Writer", "Agent 6 - Review"])

    with writer_tab:
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

            # Center the blog in a reading-width column instead of stretching across the page.
            _, page, _ = st.columns([1, 6, 1])
            with page:
                if not article:
                    st.warning("Agent 5 returned an empty article_markdown field.")
                else:
                    sections = split_article_sections(article)
                    words = len(article.split())
                    target = int(target_word_count)
                    h2_count = len(re.findall(r"^##\s", article, flags=re.MULTILINE))
                    faq_count = len(sections.faqs or [])
                    verify_count = len(VERIFY_RE.findall(article))

                    # ---- Header card: title, stats, actions ----
                    with st.container(border=True, key="blog_header"):
                        st.markdown(
                            f'<h1 class="blog-title">{html.escape(title)}</h1>',
                            unsafe_allow_html=True,
                        )

                        word_color = "green" if abs(words - target) <= target * 0.10 else "orange"
                        chips = [f":{word_color}-badge[{words:,} of {target:,} words]"]
                        if h2_count:
                            chips.append(f":gray-badge[{h2_count} sections]")
                        if faq_count:
                            chips.append(f":gray-badge[{faq_count} FAQs]")
                        if verify_count:
                            chips.append(f":orange-badge[⚠ {verify_count} claims need a source]")
                        else:
                            chips.append(":green-badge[✓ no open source flags]")
                        st.markdown(" ".join(chips))

                        b1, b2, b3, _spacer = st.columns([1.25, 1.35, 1.45, 2.5])
                        with b1:
                            st.download_button(
                                "Download PDF",
                                data=cached_article_pdf(article),
                                file_name=f"agent5_article_{prompt_id}.pdf",
                                mime="application/pdf",
                                key="download_agent5_pdf",
                                type="primary",
                                icon=":material/download:",
                                width="stretch",
                            )
                        with b2:
                            st.download_button(
                                "Download Markdown",
                                article + "\n",
                                file_name=f"agent5_article_{prompt_id}.md",
                                mime="text/markdown",
                                key="download_agent5_blog",
                                icon=":material/download:",
                                width="stretch",
                            )
                        with b3:
                            with st.popover(
                                "Copy Markdown",
                                icon=":material/content_copy:",
                                width="stretch",
                            ):
                                st.caption("Use the copy icon at the top right of the box.")
                                st.code(article, language="markdown", wrap_lines=True)

                    seo_metadata_panel(seo_title, meta_description)

                    # ---- Article body ----
                    with st.container(border=True, key="blog_card"):
                        if sections.introduction:
                            st.markdown(style_verify_markers(sections.introduction))
                        if sections.main_content:
                            st.markdown(style_verify_markers(sections.main_content))
                        if sections.conclusion:
                            st.subheader("Conclusion")
                            st.markdown(style_verify_markers(sections.conclusion))
                        if sections.faqs or sections.faq_fallback:
                            st.subheader("Frequently Asked Questions")
                            if sections.faq_intro:
                                st.markdown(sections.faq_intro)
                            if sections.faqs:
                                for question, answer in sections.faqs:
                                    with st.expander(question):
                                        st.markdown(style_verify_markers(answer))
                            else:
                                st.markdown(style_verify_markers(sections.faq_fallback))

    with reviewer_tab:
        if a5 is None or a4 is None or a3 is None or a2 is None or a1 is None:
            st.info("Complete Agents 1-5 before requesting the optional review.")
        else:
            if st.session_state.reviewer_error:
                st.error(f"Reviewer: {st.session_state.reviewer_error}")
            if a6 is None and st.button(
                "Run Agent 6 review",
                type="primary",
                key="run_agent6_review",
            ):
                handoff = as_dict(a3).get("agent4_handoff") or {}
                handoff_company = handoff.get("company") or {}
                review_state = PipelineState(
                    run_id=str(st.session_state.workflow_run_id),
                    campaign_id=int(as_dict(a2).get("campaign_id") or campaign_id),
                    prompt_id=str(as_dict(a1).get("prompt_id") or prompt_id),
                    website=str(handoff_company.get("website") or company_link).strip(),
                    company_name=str(
                        handoff_company.get("name") or as_dict(a1).get("customer_name") or customer_name
                    ),
                    prompt_analysis=a1,
                    source_analysis=a2,
                    guide=a3,
                    content_brief=a4,
                    written_article=a5,
                    review_requested=True,
                )
                try:
                    with st.status("Running optional Reviewer Agent", state="running") as review_status:
                        reviewed_state = PipelineState.model_validate(build_pipeline().invoke(review_state))
                        review_status.update(label="Reviewer completed", state="complete", expanded=False)
                    if reviewed_state.review_output is None:
                        raise RuntimeError("Workflow ended without a Reviewer result.")
                    st.session_state.agent6 = reviewed_state.review_output
                    st.session_state.reviewer_error = None
                    save_agent_output(
                        6,
                        as_dict(reviewed_state.review_output),
                        OUTPUT_DIR,
                        run_id=reviewed_state.run_id,
                    )
                    st.rerun()
                except Exception as exc:
                    st.session_state.reviewer_error = f"{type(exc).__name__}: {exc}"
                    st.error(f"Reviewer could not complete: {st.session_state.reviewer_error}")

            if st.session_state.agent6 is not None:
                review_output = as_dict(st.session_state.agent6)
                st.caption("Agent 6 reports issues only; it does not edit the Agent 5 draft.")
                if review_output.get("summary"):
                    st.info(str(review_output["summary"]))
                findings = review_output.get("findings") or []
                if findings:
                    with st.expander("Review findings", icon=":material/rate_review:"):
                        for raw_finding in findings:
                            finding = as_dict(raw_finding)
                            st.markdown(
                                f"**{finding.get('severity', 'Review')} · "
                                f"{finding.get('area', 'Content')}** — {finding.get('issue', '')}"
                            )
                            if finding.get("recommendation"):
                                st.write(finding["recommendation"])
                else:
                    st.success("No material issues were reported. Human review is still required.")
                unresolved = review_output.get("unresolved_items") or []
                if unresolved:
                    st.subheader("Unresolved items")
                    for item in unresolved:
                        st.markdown(f"- {item}")