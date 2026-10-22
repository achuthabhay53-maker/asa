"""AImpact Content Agent: Streamlit UI for the LangGraph workflow.

Run with: streamlit run app.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import streamlit as st
from aca.article_sections import split_article_sections
from aca.agents.source_intelligence import TOP_N_SOURCES
from aca.output_storage import clear_agent_outputs, save_agent_output
from aca.orchestrator.pipeline import PipelineState, build_pipeline
from aca.review_diff import render_markdown_diff


# ------------------------------- UI ---------------------------------
st.set_page_config(page_title="AImpact Content Agent", page_icon="✦", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top:1.25rem; padding-bottom:3rem; max-width:1450px;}
h1 {font-size:1.85rem; font-weight:750;}
[data-testid="stMetric"] {background:rgba(128,128,128,.07); padding:12px; border-radius:12px;}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------- HELPERS ------------------------------
def as_dict(value: Any) -> dict:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value if isinstance(value, dict) else {}


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [
            x.strip() if isinstance(x, str) else str(x)
            for x in value
            if x is not None and str(x).strip() and not isinstance(x, (dict, list, tuple))
        ]
    return []


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(x for x in items if x and x.strip()))


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


def bullets(items: list[str], empty="Nothing available."):
    if items:
        for item in items:
            st.markdown(f"- {item}")
    else:
        st.caption(empty)


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
    tabs = st.tabs(
        [
            "Agent 1 · Prompt",
            "Agent 2 · Sources",
            "Agent 3 · Ground truth",
            "Agent 4 · Brief",
            "Agent 5 · Draft blog",
            "Agent 6 · Review",
        ]
    )

    with tabs[0]:
        if a1 is None:
            st.info("Agent 1 has not completed.")
        else:
            d = as_dict(a1)
            p = d.get("prompt_understanding") or {}
            ans = d.get("answer_understanding") or {}
            vis = d.get("customer_visibility") or {}
            x, y, z = st.columns(3)
            x.metric("Intent", str(p.get("intent") or "—"))
            y.metric("Topic", str(p.get("topic") or "—"))
            z.metric("Customer visibility", str(vis.get("status") or "unknown").replace("_", " ").title())
            left, right = st.columns(2)
            with left:
                st.subheader("Prompt")
                st.write(p.get("user_goal") or "—")
                st.caption(p.get("audience") or "Audience not stated")
                bullets(string_list(p.get("key_questions")))
            with right:
                st.subheader("AI answers")
                st.write(ans.get("summary") or "—")
                for label, key in [
                    ("Topics", "key_topics"),
                    ("Concepts", "key_concepts"),
                    ("Comparison criteria", "comparison_criteria"),
                ]:
                    st.markdown(f"**{label}**")
                    bullets(string_list(ans.get(key)))
            companies = d.get("companies_mentioned") or []
            if companies:
                st.dataframe(
                    [{"Company": c.get("name", ""), "Context": c.get("context", "")} for c in companies],
                    hide_index=True,
                    width="stretch",
                )
            st.markdown("**Content opportunities**")
            bullets(string_list(d.get("content_opportunities")))
            show_json("Agent 1 JSON", a1)
            download_json("Download Agent 1 JSON", a1, f"agent1_{prompt_id}.json", "dl_a1")

    with tabs[1]:
        if a2 is None:
            st.info("Agent 2 has not completed.")
        else:
            d = as_dict(a2)
            summary = d.get("citation_summary") or {}
            x, y, z = st.columns(3)
            x.metric("Citation occurrences", summary.get("total_citation_occurrences", 0))
            y.metric("Unique sources", summary.get("unique_sources", 0))
            z.metric("Pages analyzed", len(d.get("content_analyses") or []))
            rows = []
            analyzed = {s.get("source_id") for s in d.get("content_analyses") or []}
            for source in d.get("ranked_sources") or []:
                rows.append(
                    {
                        "Rank": source.get("rank"),
                        "Source": source.get("title") or source.get("url"),
                        "Citations": source.get("citation_count", 0),
                        "URL": source.get("url"),
                        "Status": "Analyzed" if source.get("source_id") in analyzed else "Not analyzed",
                    }
                )
            if rows:
                st.dataframe(rows, hide_index=True, width="stretch")
            with st.expander("Source page analysis"):
                for item in d.get("content_analyses") or []:
                    ca = item.get("content_analysis") or {}
                    st.markdown(
                        f"**{item.get('source_id', '')} · "
                        f"{item.get('title') or item.get('url') or 'Source'}**"
                    )
                    st.caption(item.get("url") or "")
                    st.write(ca.get("summary") or "—")
                    bullets(string_list(ca.get("main_topics")))
                    for finding in ca.get("findings") or []:
                        finding_data = as_dict(finding)
                        st.markdown(f"**Finding:** {finding_data.get('finding', '')}")
                        if finding_data.get("quote_found_on_page"):
                            st.markdown(f"> {finding_data.get('evidence_quote', '')}")
                            st.caption(
                                "The quote matches fetched page text; check the source "
                                "to assess whether it supports the finding."
                            )
                        else:
                            st.warning("No exact supporting quote was confirmed in fetched page text.")
                            if finding_data.get("evidence_quote"):
                                st.caption(
                                    f"Unmatched quote returned by analysis: "
                                    f"{finding_data['evidence_quote']}"
                                )
            show_json("Agent 2 JSON", a2)
            download_json("Download Agent 2 JSON", a2, f"agent2_{prompt_id}.json", "dl_a2")

    with tabs[2]:
        if a3 is None:
            st.info("Agent 3 has not completed.")
        else:
            d = as_dict(a3)
            ver = d.get("verification") or {}
            if ver.get("is_verified") and ver.get("human_review_required"):
                st.warning(
                    "Prompt/customer relevance passed, but supporting evidence "
                    "needs human review. " + str(ver.get("reason") or "")
                )
            elif ver.get("is_verified"):
                st.success("Prompt and customer verified. " + str(ver.get("reason") or ""))
            else:
                st.error("Verification failed: " + str(ver.get("reason") or "No reason supplied."))
            x, y, z = st.columns(3)
            x.metric("Main topic", str(d.get("main_topic") or d.get("topic") or "—"))
            y.metric("Search intent", str(d.get("search_intent") or "—"))
            z.metric("Target audience", str(d.get("target_audience") or "—"))
            subs = d.get("candidate_subtopics") or []
            st.subheader(f"Candidate subtopics ({len(subs)})")
            for sub in subs:
                with st.container(border=True):
                    st.markdown(f"**{sub.get('title') or sub.get('subtopic_id') or 'Subtopic'}**")
                    st.caption(
                        f"Evidence status: {sub.get('evidence_status', sub.get('support_status', 'unknown'))}"
                    )
                    st.write(sub.get("description") or "")
                    bullets(string_list(sub.get("suggested_questions")))
                    subtopic_evidence_ids = string_list(sub.get("evidence_ids"))
                    if subtopic_evidence_ids:
                        st.caption("Evidence IDs: " + ", ".join(subtopic_evidence_ids))
            facts = d.get("verified_facts") or []
            st.subheader(f"Verified facts ({len(facts)})")
            evidence_by_id = {
                str(item.get("evidence_id")): item
                for item in d.get("evidence") or []
                if item.get("evidence_id")
            }
            if facts:
                st.dataframe(
                    [
                        {
                            "ID": f.get("fact_id"),
                            "Statement": f.get("statement"),
                            "Status": f.get("status"),
                            "Evidence IDs": ", ".join(string_list(f.get("evidence_ids"))),
                            "Confidence": f.get("confidence"),
                            "Boundary": f.get("wording_boundary"),
                        }
                        for f in facts
                    ],
                    hide_index=True,
                    width="stretch",
                )
                for fact in facts:
                    fact_evidence = [
                        evidence_by_id[evidence_id]
                        for evidence_id in string_list(fact.get("evidence_ids"))
                        if evidence_id in evidence_by_id
                    ]
                    with st.expander(f"Evidence for {fact.get('fact_id', 'fact')}"):
                        st.write(fact.get("statement") or "")
                        if fact_evidence:
                            for item in fact_evidence:
                                st.markdown(f"**{item.get('evidence_id')}** · {item.get('source_type', '')}")
                                st.markdown(f"> {item.get('quote', '')}")
                                if item.get("url"):
                                    st.markdown(f"[Open source]({item['url']})")
                        else:
                            st.warning("No evidence reference is attached to this finding.")
            with st.expander(f"Evidence quotes ({len(d.get('evidence') or [])})"):
                for ev in d.get("evidence") or []:
                    st.markdown(f"**{ev.get('evidence_id', '')} · {ev.get('source_type', '')}**")
                    st.markdown(f"> {ev.get('quote', '')}")
                    if ev.get("url"):
                        st.caption(ev["url"])
            show_json("Agent 3 JSON", a3)
            download_json("Download Agent 3 JSON", a3, f"agent3_{prompt_id}.json", "dl_a3")

    with tabs[3]:
        if a4 is None:
            st.info("Agent 4 has not completed.")
        else:
            d = as_dict(a4)
            st.header("Content brief")
            st.markdown("**Proposed title**")
            st.write(d.get("title") or d.get("recommended_title") or "Title not available")
            st.markdown("**Objective**")
            st.write(d.get("article_objective") or d.get("objective") or "—")
            st.markdown("**Target audience**")
            st.write(d.get("target_audience") or "—")
            sections = d.get("sections") or d.get("outline") or d.get("brief_sections") or []
            st.subheader(f"Article outline ({len(sections)} sections)")
            for i, raw in enumerate(sections, 1):
                s = as_dict(raw)
                with st.container(border=True):
                    st.markdown(
                        f"**{i}. {s.get('heading') or s.get('title') or s.get('name') or 'Section'}**"
                    )
                    st.write(
                        s.get("section_objective")
                        or s.get("objective")
                        or s.get("description")
                        or s.get("purpose")
                        or ""
                    )
                    bullets(
                        unique(
                            string_list(s.get("topics_and_questions"))
                            + string_list(s.get("topics"))
                            + string_list(s.get("key_points"))
                        )
                    )
                    evidence_ids = string_list(s.get("evidence_ids"))
                    source_urls = string_list(s.get("source_urls"))
                    if evidence_ids:
                        st.caption("Evidence IDs: " + ", ".join(evidence_ids))
                    if source_urls:
                        st.markdown("**Sources**")
                    for url in source_urls:
                        st.markdown(f"- [Source]({url})")
                    guidelines = s.get("guidelines_and_gaps") or s.get("guidelines") or []
                    if guidelines:
                        st.markdown("**Guidelines and evidence limits**")
                        bullets(string_list(guidelines))
            show_json("Agent 4 JSON", a4)
            download_json("Download Agent 4 JSON", a4, f"agent4_brief_{prompt_id}.json", "dl_a4")

    with tabs[4]:
        if a5 is None:
            if st.session_state.agent5_error:
                st.warning("The brief may be ready, but Agent 5 failed. See the error above.")
            else:
                st.info("Agent 5 has not completed.")
        else:
            d = as_dict(a5)
            article = str(d.get("article_markdown") or "").strip()
            title = str(d.get("title") or "Final Blog")
            if d.get("seo_title"):
                st.caption(f"SEO title ({len(d['seo_title'])}/60): {d['seo_title']}")
            if d.get("meta_description"):
                st.caption(
                    f"Meta description ({len(d['meta_description'])}/155): " f"{d['meta_description']}"
                )
            if not article:
                st.warning("Agent 5 returned an empty article_markdown field.")
            else:
                if not article.startswith("# "):
                    article = f"# {title}\\n\\n{article}"
                claims, unresolved = d.get("claims_used") or [], d.get("unresolved_items") or []
                x, y, z, w = st.columns(4)
                x.metric("Word count", d.get("word_count", 0))
                y.metric("Written sections", len(d.get("sections") or []))
                z.metric("Tracked claims", len(claims))
                w.metric("Unresolved items", len(unresolved))
                btabs = st.tabs(["Read blog", "Markdown source", "Evidence & review"])
                with btabs[0]:
                    sections = split_article_sections(article)
                    st.markdown(f"### {title}")
                    st.divider()
                    with st.container(border=True):
                        if sections.introduction:
                            st.subheader("Introduction")
                            st.markdown(sections.introduction)
                        if sections.main_content:
                            st.markdown(sections.main_content)
                        if sections.conclusion:
                            st.subheader("Conclusion")
                            st.markdown(sections.conclusion)
                        if sections.faqs or sections.faq_fallback:
                            st.subheader("Frequently Asked Questions")
                            if sections.faq_intro:
                                st.markdown(sections.faq_intro)
                            if sections.faqs:
                                for question, answer in sections.faqs:
                                    with st.expander(question):
                                        st.markdown(answer)
                            else:
                                st.markdown(sections.faq_fallback)
                with btabs[1]:
                    st.code(article, language="markdown")
                    st.download_button(
                        "Download blog (.md)",
                        article + "\\n",
                        file_name=f"agent5_article_{prompt_id}.md",
                        mime="text/markdown",
                        type="primary",
                        width="stretch",
                    )
                with btabs[2]:
                    st.subheader("Claims used")
                    if claims:
                        for claim in claims:
                            c = as_dict(claim)
                            with st.container(border=True):
                                st.write(c.get("claim", ""))
                                st.caption("Claim scope: " + str(c.get("claim_scope", "company_specific")))
                                ids = string_list(c.get("evidence_ids"))
                                if ids:
                                    st.caption("Evidence IDs: " + ", ".join(ids))
                                supporting_evidence = c.get("supporting_evidence") or []
                                st.caption(
                                    "Citation mapping: " + str(c.get("citation_status", "not_checked"))
                                )
                                st.caption(
                                    "Evidence assessment (AI, not independently verified): "
                                    + str(c.get("support_status", "not_assessed"))
                                )
                                for note in string_list(c.get("validation_notes")):
                                    st.warning(note)
                                if (
                                    not supporting_evidence
                                    and c.get("claim_scope", "company_specific") == "company_specific"
                                ):
                                    st.warning(
                                        "No supplied evidence is attached to this claim; "
                                        "review it before publication."
                                    )
                                for reference in supporting_evidence:
                                    evidence = as_dict(reference)
                                    st.markdown(
                                        f"**{evidence.get('evidence_id')} · "
                                        f"{evidence.get('status', 'unknown')}**"
                                    )
                                    if evidence.get("wording_boundary"):
                                        st.warning("Wording boundary: " + str(evidence["wording_boundary"]))
                                    st.markdown(
                                        f"> {evidence.get('evidence_quote') or 'No quote available.'}"
                                    )
                                    if evidence.get("source_url"):
                                        st.markdown(f"[Open source]({evidence['source_url']})")
                    else:
                        st.caption("No structured claims returned.")
                    st.subheader("Unresolved items")
                    bullets(string_list(unresolved), "No unresolved items reported.")
                    st.subheader("Brief coverage")
                    bullets(string_list(d.get("brief_coverage")), "No coverage report returned.")
                show_json("Agent 5 JSON", a5)
                download_json("Download Agent 5 JSON", a5, f"agent5_output_{prompt_id}.json", "dl_a5")

    with tabs[5]:
        if a5 is None or a4 is None or a3 is None or a2 is None or a1 is None:
            st.info("Complete Agents 1–5 before requesting the optional review.")
        else:
            st.write(
                "Review formatting, headings, content, keywords, SEO, and evidence-grounded "
                "claims. The Reviewer uses an additional model call; it runs only when requested."
            )
            if st.session_state.reviewer_error:
                st.error(f"Reviewer: {st.session_state.reviewer_error}")
            if a6 is None and st.button(
                "Review and revise blog",
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
                    with st.status(
                        "Running optional Reviewer Agent",
                        state="running",
                        expanded=True,
                    ) as review_status:
                        reviewed_state = PipelineState.model_validate(build_pipeline().invoke(review_state))
                        review_status.update(
                            label="Reviewer completed",
                            state="complete",
                            expanded=False,
                        )
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

            a6 = st.session_state.agent6
            if a6 is not None:
                review_data = as_dict(a6)
                revised = review_data.get("revised_blog") or {}
                revised_article = str(revised.get("article_markdown") or "").strip()
                st.success(str(review_data.get("summary") or "Review completed."))
                review_tabs = st.tabs(["Revised blog with changes", "Final Markdown", "Review findings"])
                with review_tabs[0]:
                    st.caption("Green text marks words added or changed by Agent 6.")
                    if a5.article_markdown and revised_article:
                        st.markdown(
                            render_markdown_diff(a5.article_markdown, revised_article),
                            unsafe_allow_html=True,
                        )
                with review_tabs[1]:
                    st.code(revised_article, language="markdown")
                    st.download_button(
                        "Download revised blog (.md)",
                        revised_article + "\n",
                        file_name=f"reviewed_article_{prompt_id}.md",
                        mime="text/markdown",
                        key="download_reviewed_blog",
                        width="stretch",
                    )
                with review_tabs[2]:
                    st.subheader("Review findings")
                    for finding in review_data.get("findings") or []:
                        finding_data = as_dict(finding)
                        st.markdown(
                            f"**{finding_data.get('severity', 'review')} · "
                            f"{finding_data.get('area', 'content')}** — "
                            f"{finding_data.get('issue', '')}"
                        )
                        st.write(finding_data.get("recommendation", ""))
                    st.subheader("Claims requiring human verification")
                    flagged = [
                        as_dict(claim)
                        for claim in revised.get("claims_used") or []
                        if as_dict(claim).get("claim_scope", "company_specific") == "company_specific"
                        and as_dict(claim).get("support_status")
                        in {
                            "partially_supported",
                            "unsupported",
                            "contradicted",
                            "inconclusive",
                            "not_assessed",
                        }
                    ]
                    if flagged:
                        for claim in flagged:
                            st.warning(f"{claim.get('support_status')}: {claim.get('claim')}")
                    else:
                        st.success("No company-specific claims were flagged by the reviewer.")
                    st.subheader("Unresolved items")
                    bullets(string_list(revised.get("unresolved_items")), "No unresolved items.")
                show_json("Agent 6 JSON", a6)
                download_json(
                    "Download Agent 6 JSON",
                    a6,
                    f"agent6_review_{prompt_id}.json",
                    "dl_agent6",
                )

if a3 is not None and a4 is not None and a5 is not None:
    st.info(
        "Draft generated by Agents 1–5. Human review of claims, citations, and "
        "unresolved items is required before publication."
    )
elif a3 is not None and a5 is None:
    st.info("Research results are available. Check the Agent 4/5 status and errors above.")
