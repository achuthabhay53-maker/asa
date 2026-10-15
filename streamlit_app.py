"""AImpact Content Agent: interactive five-agent Streamlit workflow.
Run from D:\\ACA2\\ASA with: streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import re
import time
import traceback
from typing import Any, Callable

import streamlit as st
from aca.agents.ground_truth import GroundTruthAgent
from aca.agents.prompt_intelligence import run_agent1
from aca.agents.source_intelligence import TOP_N_SOURCES, run_agent2
from aca.agents.content_brief import ContentBriefAgent
from aca.agents.writer import WriterAgent
from aca.models.ground_truth import ClientInput
from aca.models.writer import WriterInput, WriterSectionInput, WriterEvidence


# ------------------------------- UI ---------------------------------
st.set_page_config(page_title="AImpact Content Agent", page_icon="✦", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1.25rem; padding-bottom:3rem; max-width:1450px;}
h1 {font-size:1.85rem; font-weight:750;}
[data-testid="stMetric"] {background:rgba(128,128,128,.07); padding:12px; border-radius:12px;}
</style>
""", unsafe_allow_html=True)


# ----------------------------- HELPERS ------------------------------
def as_dict(value: Any) -> dict:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value if isinstance(value, dict) else {}


def get_value(obj: Any, *names: str, default=None):
    data = as_dict(obj)
    for name in names:
        if data.get(name) is not None:
            return data[name]
        if hasattr(obj, name) and getattr(obj, name) is not None:
            return getattr(obj, name)
    return default


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [
            x.strip() if isinstance(x, str) else str(x)
            for x in value if x is not None and str(x).strip()
            and not isinstance(x, (dict, list, tuple))
        ]
    return []


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(x for x in items if x and x.strip()))


def show_json(label: str, obj: Any):
    with st.expander(label):
        st.json(as_dict(obj))


def download_json(label: str, obj: Any, filename: str, key: str):
    st.download_button(
        label, json.dumps(as_dict(obj), ensure_ascii=False, indent=2),
        file_name=filename, mime="application/json", key=key, width="stretch"
    )


def bullets(items: list[str], empty="Nothing available."):
    if items:
        for item in items:
            st.markdown(f"- {item}")
    else:
        st.caption(empty)


def run_step(number: int, name: str, description: str, callback: Callable[[], Any]):
    """Progress indicates step status; actual internal model progress isn't exposed."""
    start = time.perf_counter()
    with st.status(f"Step {number}/5 · {name}", state="running", expanded=True) as status:
        st.write(description)
        bar = st.progress(5, text="Starting…")
        message = st.empty()
        try:
            bar.progress(35, text="Working…")
            result = callback()
            elapsed = time.perf_counter() - start
            bar.progress(100, text="Completed")
            message.success(f"Completed in {elapsed:.1f} seconds.")
            status.update(label=f"Step {number}/5 · {name} complete", state="complete", expanded=False)
            return result
        except Exception as exc:
            message.error(f"{type(exc).__name__}: {exc}")
            status.update(label=f"Step {number}/5 · {name} failed", state="error", expanded=True)
            raise


def pipeline():
    st.subheader("Workflow pipeline")
    steps = [
        ("agent1", "Agent 1", "Prompt", "run_error"),
        ("agent2", "Agent 2", "Sources", "run_error"),
        ("agent3", "Agent 3", "Ground truth", "run_error"),
        ("agent4", "Agent 4", "Content brief", "agent4_error"),
        ("agent5", "Agent 5", "Blog writer", "agent5_error"),
    ]
    cols = st.columns(5)
    for col, (key, name, desc, err_key) in zip(cols, steps):
        with col:
            if st.session_state.get(key) is not None:
                st.success(f"✓ {name}")
            elif st.session_state.get(err_key):
                st.error(f"✕ {name}")
            else:
                st.info(f"○ {name}")
            st.caption(desc)


# ---------------------- AGENT 3 -> WRITER EVIDENCE ------------------
def build_writer_evidence(guide: Any) -> list[WriterEvidence]:
    data = as_dict(guide)
    facts = data.get("verified_facts") or []
    evidence = data.get("evidence") or []
    by_id: dict[str, list[dict]] = {}
    for fact in facts:
        for evidence_id in fact.get("evidence_ids") or []:
            by_id.setdefault(str(evidence_id), []).append(fact)

    output = []
    for ev in evidence:
        eid = str(ev.get("evidence_id") or "").strip()
        if not eid:
            continue
        related = by_id.get(eid, [])
        if related:
            fact = related[0]
            claim = str(fact.get("statement") or ev.get("quote") or "")
            status = str(fact.get("status") or "inconclusive").lower()
            if status not in {"verified", "contradicted", "not_found", "inconclusive"}:
                status = "inconclusive"
        else:
            claim, status = str(ev.get("quote") or ""), "inconclusive"
        output.append(WriterEvidence(
            evidence_id=eid, claim=claim, status=status,
            evidence_quote=ev.get("quote"), source_url=ev.get("url")
        ))
    return output


def build_writer_input(agent1: Any, agent3: Any, brief: Any) -> WriterInput:
    b, a1, a3 = as_dict(brief), as_dict(agent1), as_dict(agent3)
    understanding = a1.get("prompt_understanding") or {}
    company = a3.get("company") or {}
    company_name = (
        get_value(brief, "company_name", "customer_name")
        or company.get("name") or a1.get("customer_name") or "Customer"
    )
    company_url = get_value(brief, "company_url", "website") or company.get("website")
    raw_sections = b.get("sections") or b.get("outline") or b.get("brief_sections") or []
    sections = []
    for raw in raw_sections:
        s = as_dict(raw)
        heading = str(s.get("heading") or s.get("title") or s.get("name") or "").strip()
        if not heading:
            continue
        level = s.get("level", 2)
        if isinstance(level, str):
            m = re.search(r"\\d+", level)
            level = int(m.group()) if m else 2
        try:
            level = max(1, min(6, int(level or 2)))
        except (ValueError, TypeError):
            level = 2
        sections.append(WriterSectionInput(
            heading=heading, level=level,
            objective=str(s.get("objective") or s.get("description") or s.get("purpose") or ""),
            topics=unique(string_list(s.get("topics")) + string_list(s.get("key_points")) + string_list(s.get("subtopics"))),
            questions_to_answer=unique(string_list(s.get("questions_to_answer")) + string_list(s.get("questions")) + string_list(s.get("suggested_questions"))),
            evidence_ids=unique(string_list(s.get("evidence_ids")) + string_list(s.get("verified_fact_ids"))),
            guidelines=unique(string_list(s.get("guidelines")) + string_list(s.get("instructions")) + string_list(s.get("constraints"))),
        ))
    style = b.get("writing_style") or b.get("writing_style_guide") or b.get("style_guide") or {}
    if not isinstance(style, dict):
        style = as_dict(style)
    faqs = []
    for faq in b.get("faqs") or b.get("frequently_asked_questions") or []:
        if isinstance(faq, dict):
            q, a = faq.get("question") or faq.get("q"), faq.get("answer") or faq.get("a")
            if q:
                faqs.append({"question": str(q), "answer": str(a or "")})
    target_count = b.get("target_word_count") or b.get("recommended_word_count") or b.get("word_count_target")
    try:
        target_count = int(target_count) if target_count else None
    except (ValueError, TypeError):
        target_count = None
    return WriterInput(
        company_name=str(company_name),
        company_url=str(company_url) if company_url else None,
        title=str(b.get("title") or b.get("recommended_title") or b.get("article_title") or understanding.get("topic") or a1.get("prompt") or "Research article"),
        meta_description=str(b.get("meta_description") or b.get("seo_meta_description") or ""),
        article_objective=str(b.get("article_objective") or b.get("objective") or b.get("content_goal") or "Answer the search intent with an evidence-based, useful article."),
        target_audience=str(b.get("target_audience") or understanding.get("audience") or a3.get("target_audience") or "Relevant industry readers"),
        search_intent=str(b.get("search_intent") or a3.get("search_intent") or understanding.get("intent") or ""),
        sections=sections, faqs=faqs,
        internal_links=unique(string_list(b.get("internal_links"))),
        writing_style={str(k): str(v) for k, v in style.items() if v is not None},
        unsupported_claims_to_avoid=unique(
            string_list(b.get("unsupported_claims_to_avoid") or b.get("claims_to_avoid") or b.get("prohibited_claims"))
            + string_list(b.get("research_gaps"))
        ),
        evidence=build_writer_evidence(agent3),
        target_word_count=target_count,
    )


# ------------------------------- STATE ------------------------------
KEYS = ("agent1", "agent2", "agent3", "agent4", "agent5",
        "run_error", "agent4_error", "agent5_error")
for key in KEYS:
    st.session_state.setdefault(key, None)


# ------------------------------ INPUT UI ----------------------------
st.title("✦ AImpact Content Agent")
st.caption("Prompt intelligence → source research → ground truth → content brief → final blog")
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
    submitted = st.form_submit_button("Run Agents 1–5", type="primary", width="stretch")


# ---------------------------- RUN AGENTS 1-3 -------------------------
if submitted:
    for key in KEYS:
        st.session_state[key] = None
    if not str(prompt_id).strip():
        st.session_state.run_error = "Please enter a Prompt ID."
    elif not str(company_link).strip():
        st.session_state.run_error = "Please enter a customer website."
    else:
        try:
            a1 = run_step(1, "Agent 1 — Prompt Intelligence",
                "Understanding the prompt, audience, stored AI answers, and customer visibility.",
                lambda: run_agent1(int(campaign_id), str(prompt_id).strip(), customer_name.strip() or None))
            st.session_state.agent1 = a1

            a2 = run_step(2, "Agent 2 — Source Intelligence",
                "Analyzing citations, competitor pages, source quality, and content structure.",
                lambda: run_agent2(int(campaign_id), str(prompt_id).strip(), top_n=int(top_n)))
            st.session_state.agent2 = a2

            effective_name = customer_name.strip() or get_value(a1, "customer_name", default="")
            a3 = run_step(3, "Agent 3 — Ground Truth",
                "Verifying prompt/customer relevance and collecting evidence and research gaps.",
                lambda: GroundTruthAgent(max_questions=3).analyze(
                    a1, a2, ClientInput(company_name=effective_name, website=company_link.strip())))
            if a3 is None:
                raise RuntimeError("Agent 3 returned no result; Agents 4 and 5 were stopped.")
            st.session_state.agent3 = a3
            verification = as_dict(a3).get("verification") or {}
            if not verification.get("is_verified", False):
                st.session_state.run_error = (
                    "Agent 3 verification failed. Agents 4 and 5 were not run. "
                    + str(verification.get("reason") or "")
                )
        except Exception as exc:
            st.session_state.run_error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()


# ---------------------------- RUN AGENTS 4-5 -------------------------
if (submitted and st.session_state.agent1 is not None
        and st.session_state.agent2 is not None and st.session_state.agent3 is not None
        and st.session_state.run_error is None):
    try:
        brief = run_step(4, "Agent 4 — Content Brief",
            "Preparing article direction, title, outline, and writing constraints.",
            lambda: ContentBriefAgent().build_brief(
                st.session_state.agent1, st.session_state.agent2, st.session_state.agent3))
        st.session_state.agent4 = brief
        writer_input = build_writer_input(st.session_state.agent1, st.session_state.agent3, brief)
        if not writer_input.sections:
            raise ValueError("Agent 4 returned no usable sections. Check the ContentBrief model section fields.")
        blog = run_step(5, "Agent 5 — Blog Writer",
            "Generating the article from the brief and Agent 3 evidence.",
            lambda: WriterAgent().write(writer_input))
        st.session_state.agent5 = blog
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if st.session_state.agent4 is None:
            st.session_state.agent4_error = message
        else:
            st.session_state.agent5_error = message
        traceback.print_exc()


# ---------------------------- STATUS & RESULTS -----------------------
for key, label in [("run_error", "Agents 1–3"), ("agent4_error", "Agent 4"), ("agent5_error", "Agent 5")]:
    if st.session_state.get(key):
        st.error(f"{label}: {st.session_state[key]}")
pipeline()

a1, a2, a3 = st.session_state.agent1, st.session_state.agent2, st.session_state.agent3
a4, a5 = st.session_state.agent4, st.session_state.agent5
if all(x is None for x in (a1, a2, a3, a4, a5)):
    st.info("Enter a Campaign ID, Prompt ID, and customer website, then click **Run Agents 1–5**.")

if any(x is not None for x in (a1, a2, a3, a4, a5)):
    tabs = st.tabs(["Agent 1 · Prompt", "Agent 2 · Sources", "Agent 3 · Ground truth", "Agent 4 · Brief", "Agent 5 · Final blog"])

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
                for label, key in [("Topics", "key_topics"), ("Concepts", "key_concepts"), ("Comparison criteria", "comparison_criteria")]:
                    st.markdown(f"**{label}**")
                    bullets(string_list(ans.get(key)))
            companies = d.get("companies_mentioned") or []
            if companies:
                st.dataframe([{"Company": c.get("name", ""), "Context": c.get("context", "")} for c in companies], hide_index=True, width="stretch")
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
                rows.append({"Rank": source.get("rank"), "Source": source.get("title") or source.get("url"),
                             "Citations": source.get("citation_count", 0), "URL": source.get("url"),
                             "Status": "Analyzed" if source.get("source_id") in analyzed else "Not analyzed"})
            if rows:
                st.dataframe(rows, hide_index=True, width="stretch")
            with st.expander("Source page analysis"):
                for item in d.get("content_analyses") or []:
                    ca = item.get("content_analysis") or {}
                    st.markdown(f"**{item.get('title') or item.get('url') or item.get('source_id')}**")
                    st.caption(item.get("url") or "")
                    st.write(ca.get("summary") or "—")
                    bullets(string_list(ca.get("main_topics")))
            show_json("Agent 2 JSON", a2)
            download_json("Download Agent 2 JSON", a2, f"agent2_{prompt_id}.json", "dl_a2")

    with tabs[2]:
        if a3 is None:
            st.info("Agent 3 has not completed.")
        else:
            d = as_dict(a3)
            ver = d.get("verification") or {}
            if ver.get("is_verified"):
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
                    st.caption(f"Evidence status: {sub.get('evidence_status', sub.get('support_status', 'unknown'))}")
                    st.write(sub.get("description") or "")
                    bullets(string_list(sub.get("suggested_questions")))
            facts = d.get("verified_facts") or []
            st.subheader(f"Verified facts ({len(facts)})")
            if facts:
                st.dataframe([{"ID": f.get("fact_id"), "Statement": f.get("statement"), "Status": f.get("status"),
                               "Confidence": f.get("confidence"), "Boundary": f.get("wording_boundary")} for f in facts],
                             hide_index=True, width="stretch")
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
                    st.markdown(f"**{i}. {s.get('heading') or s.get('title') or s.get('name') or 'Section'}**")
                    st.write(s.get("objective") or s.get("description") or s.get("purpose") or "")
                    bullets(unique(string_list(s.get("topics")) + string_list(s.get("key_points"))))
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
                    st.markdown(f"### {title}")
                    st.divider()
                    with st.container(border=True):
                        st.markdown(article)
                with btabs[1]:
                    st.code(article, language="markdown")
                    st.download_button("Download blog (.md)", article + "\\n", file_name=f"agent5_article_{prompt_id}.md",
                                       mime="text/markdown", type="primary", width="stretch")
                with btabs[2]:
                    st.subheader("Claims used")
                    if claims:
                        for claim in claims:
                            c = as_dict(claim)
                            with st.container(border=True):
                                st.write(c.get("claim", ""))
                                ids = string_list(c.get("evidence_ids"))
                                if ids:
                                    st.caption("Evidence IDs: " + ", ".join(ids))
                                for url in string_list(c.get("source_urls")):
                                    st.markdown(f"- [{url}]({url})")
                    else:
                        st.caption("No structured claims returned.")
                    st.subheader("Unresolved items")
                    bullets(string_list(unresolved), "No unresolved items reported.")
                    st.subheader("Brief coverage")
                    bullets(string_list(d.get("brief_coverage")), "No coverage report returned.")
                show_json("Agent 5 JSON", a5)
                download_json("Download Agent 5 JSON", a5, f"agent5_output_{prompt_id}.json", "dl_a5")

if a3 is not None and a4 is not None and a5 is not None:
    st.success("Workflow complete: all five agents completed successfully.")
elif a3 is not None and a5 is None:
    st.info("Research results are available. Check the Agent 4/5 status and errors above.")
