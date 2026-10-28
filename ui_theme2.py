"""UI theme and components for the AImpact Content Agent (Streamlit).

Presentation only. No workflow logic lives here.
"""

from __future__ import annotations

import html

import streamlit as st

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,400&display=swap');

:root {
    --acp-bg: #F4F6FB;
    --acp-surface: #FFFFFF;
    --acp-ink: #141B34;
    --acp-muted: #5B6480;
    --acp-line: #DDE2EE;
    --acp-primary: #2B3FD6;
    --acp-primary-dark: #1F2EA8;
    --acp-primary-soft: #E8EBFC;
    --acp-ok: #12805C;
    --acp-ok-soft: #E1F4EC;
    --acp-warn: #B86E00;
    --acp-warn-soft: #FFF1D6;
    --acp-err: #C0362C;
    --acp-err-soft: #FCE6E3;
    --acp-serif: 'Newsreader', Georgia, 'Times New Roman', serif;
    --acp-sans: 'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif;
}

/* ---------- Base ---------- */
.stApp, [data-testid="stAppViewContainer"] {background: var(--acp-bg);}
[data-testid="stHeader"] {background: transparent;}
html, body, .stApp, .stMarkdown, label, button, input, textarea {font-family: var(--acp-sans);}
.block-container {padding-top: 2rem; padding-bottom: 5rem; max-width: 1360px;}
[data-testid="stHeading"] h2, [data-testid="stHeading"] h3, [data-testid="stHeading"] h4 {
    font-family: var(--acp-sans); color: var(--acp-ink); letter-spacing: -.01em; font-weight: 750;
}
[data-testid="stHeading"] h3 {font-size: 1.2rem;}
[data-testid="stCaptionContainer"] {color: var(--acp-muted);}
[data-testid="stHeaderActionElements"] {display: none;}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {background: var(--acp-surface); border-right: 1px solid var(--acp-line);}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding: 1.4rem 1.1rem 2rem;}
.acp-brand {display: flex; align-items: center; gap: 12px; padding: 0 0 1.1rem; margin-bottom: 1rem; border-bottom: 1px solid var(--acp-line);}
.acp-mark {
    width: 38px; height: 38px; border-radius: 11px; background: var(--acp-primary);
    color: #fff; display: grid; place-items: center; font-size: 1.1rem; font-weight: 800; flex: none;
}
.acp-brand-name {font-size: 1.05rem; font-weight: 800; letter-spacing: -.01em; color: var(--acp-ink); line-height: 1.15;}
.acp-brand-sub {font-size: .78rem; color: var(--acp-muted); margin-top: 2px;}
.acp-side-title {font-size: .95rem; font-weight: 750; color: var(--acp-ink); margin: 0 0 .6rem;}

/* ---------- Top bar ---------- */
.acp-topbar {display: flex; justify-content: space-between; align-items: flex-end; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.2rem;}
.acp-page-title {font-size: 1.75rem; font-weight: 800; letter-spacing: -.02em; color: var(--acp-ink); line-height: 1.2; margin: 0; max-width: 60ch;}
.acp-page-sub {color: var(--acp-muted); font-size: .92rem; margin-top: .3rem;}

/* ---------- Pills ---------- */
.acp-pill {
    display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px;
    border-radius: 999px; font-weight: 700; font-size: .85rem; white-space: nowrap;
}
.acp-pill::before {content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor;}
.acp-pill.is-ok {background: var(--acp-ok-soft); color: var(--acp-ok);}
.acp-pill.is-draft {background: var(--acp-warn-soft); color: var(--acp-warn);}
.acp-pill.is-idle {background: var(--acp-primary-soft); color: var(--acp-primary);}
.acp-pill.is-error {background: var(--acp-err-soft); color: var(--acp-err);}

/* ---------- Pipeline stepper ---------- */
.acp-pipeline {background: var(--acp-surface); border: 1px solid var(--acp-line); border-radius: 16px; padding: 1.25rem 1.5rem 1.1rem; margin: 0 0 1.1rem;}
.acp-track {display: grid; grid-template-columns: repeat(6, 1fr); list-style: none; margin: 0; padding: 0;}
.acp-step {position: relative; padding-right: 14px;}
.acp-step::before {content: ""; position: absolute; top: 17px; left: 42px; right: 8px; height: 2px; background: var(--acp-line); border-radius: 2px;}
.acp-step:last-child::before {display: none;}
.acp-step.is-done::before {background: var(--acp-primary);}
.acp-node {
    position: relative; z-index: 1; width: 36px; height: 36px; border-radius: 50%;
    display: grid; place-items: center; font-weight: 800; font-size: .95rem;
    background: var(--acp-surface); border: 2px solid var(--acp-line); color: var(--acp-muted);
}
.acp-step.is-done .acp-node {background: var(--acp-primary); border-color: var(--acp-primary); color: #fff;}
.acp-step.is-warn .acp-node {background: var(--acp-warn-soft); border-color: var(--acp-warn); color: var(--acp-warn);}
.acp-step.is-error .acp-node {background: var(--acp-err-soft); border-color: var(--acp-err); color: var(--acp-err);}
.acp-step.is-next .acp-node {border-color: var(--acp-primary); color: var(--acp-primary); background: var(--acp-primary-soft);}
.acp-step.is-running .acp-node {border-color: var(--acp-primary-soft); border-top-color: var(--acp-primary); color: transparent; animation: acp-spin .9s linear infinite;}
@keyframes acp-spin {to {transform: rotate(360deg);}}
@media (prefers-reduced-motion: reduce) {.acp-step.is-running .acp-node {animation: none; color: var(--acp-primary);}}
.acp-name {margin-top: .6rem; font-weight: 700; font-size: .92rem; color: var(--acp-ink);}
.acp-desc {font-size: .82rem; color: var(--acp-muted); margin-top: 1px;}
.acp-state {font-size: .74rem; font-weight: 600; margin-top: 4px; color: var(--acp-muted);}
.acp-step.is-done .acp-state, .acp-step.is-running .acp-state {color: var(--acp-primary);}
.acp-step.is-warn .acp-state {color: var(--acp-warn);}
.acp-step.is-error .acp-state {color: var(--acp-err);}
@media (max-width: 820px) {
    .acp-track {grid-template-columns: repeat(2, 1fr); row-gap: 1.1rem;}
    .acp-step::before {display: none;}
}

/* ---------- KPI cards ---------- */
.acp-kpis {display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 0 0 1.4rem;}
.acp-kpi {background: var(--acp-surface); border: 1px solid var(--acp-line); border-radius: 14px; padding: .95rem 1.1rem;}
.acp-kpi-label {font-size: .8rem; color: var(--acp-muted); font-weight: 600;}
.acp-kpi-value {font-size: 1.65rem; font-weight: 800; letter-spacing: -.02em; margin-top: .15rem; color: var(--acp-ink);}
.acp-kpi-value.is-ok {color: var(--acp-ok);}
.acp-kpi-value.is-warn {color: var(--acp-warn);}
.acp-kpi-value.is-err {color: var(--acp-err);}
.acp-kpi-hint {font-size: .78rem; color: var(--acp-muted); margin-top: 2px;}
@media (max-width: 1000px) {.acp-kpis {grid-template-columns: repeat(2, 1fr);}}

/* ---------- Empty state ---------- */
.acp-empty {background: var(--acp-surface); border: 1px solid var(--acp-line); border-radius: 18px; padding: 2.4rem 2.2rem;}
.acp-empty h2 {font-size: 1.6rem; font-weight: 800; letter-spacing: -.02em; margin: 0 0 .5rem; color: var(--acp-ink);}
.acp-empty p {color: var(--acp-muted); max-width: 60ch; line-height: 1.6; margin: 0;}
.acp-how {display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 1.6rem;}
.acp-how-card {border: 1px solid var(--acp-line); border-radius: 14px; padding: 1.1rem 1.2rem; background: var(--acp-bg);}
.acp-how-num {width: 28px; height: 28px; border-radius: 50%; background: var(--acp-primary); color: #fff; display: grid; place-items: center; font-weight: 800; font-size: .85rem;}
.acp-how-card h4 {margin: .7rem 0 .3rem; font-size: 1rem; font-weight: 750; color: var(--acp-ink);}
.acp-how-card p {font-size: .88rem;}
@media (max-width: 900px) {.acp-how {grid-template-columns: 1fr;}}

/* ---------- Cards (keyed containers) ---------- */
[class*="st-key-card"], [class*="st-key-finding"] {
    background: var(--acp-surface); border: 1px solid var(--acp-line); border-radius: 14px; padding: 1.05rem 1.2rem;
}
.acp-card-title {font-size: .95rem; font-weight: 750; color: var(--acp-ink); margin: 0 0 .5rem;}
.acp-card-text {font-size: .9rem; color: var(--acp-muted); line-height: 1.55; margin: 0 0 .75rem;}

/* ---------- Outline, meter, timeline, checklist, severity ---------- */
.acp-outline {margin: 0; padding-left: 1.15rem; color: var(--acp-ink); font-size: .88rem; line-height: 1.75;}
.acp-meter {height: 6px; background: var(--acp-line); border-radius: 999px; overflow: hidden; margin: .35rem 0 .2rem;}
.acp-meter > span {display: block; height: 100%; background: var(--acp-primary); border-radius: 999px;}
.acp-meter.is-over > span {background: var(--acp-err);}
.acp-meter-label {font-size: .78rem; color: var(--acp-muted);}
.acp-meter-label.is-over {color: var(--acp-err); font-weight: 600;}
.acp-timeline {list-style: none; margin: 0; padding: 0;}
.acp-timeline li {position: relative; padding: 0 0 1rem 1.5rem; border-left: 2px solid var(--acp-line); margin-left: 6px;}
.acp-timeline li:last-child {padding-bottom: 0; border-left-color: transparent;}
.acp-timeline li::before {content: ""; position: absolute; left: -7px; top: 3px; width: 12px; height: 12px; border-radius: 50%; background: var(--acp-surface); border: 2px solid var(--acp-line);}
.acp-timeline li.is-current::before {background: var(--acp-primary); border-color: var(--acp-primary);}
.acp-tl-title {font-weight: 700; font-size: .9rem; color: var(--acp-ink);}
.acp-tl-meta {font-size: .78rem; color: var(--acp-muted);}
.acp-tl-note {font-size: .82rem; color: var(--acp-ink); margin-top: 3px; line-height: 1.45;}
.acp-check {display: flex; gap: 12px; align-items: flex-start; padding: .8rem 0; border-bottom: 1px solid var(--acp-line);}
.acp-check:last-child {border-bottom: none;}
.acp-check-icon {flex: none; width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center; font-size: .8rem; font-weight: 800; border: 2px solid var(--acp-line); color: var(--acp-muted);}
.acp-check.is-ok .acp-check-icon {background: var(--acp-ok); border-color: var(--acp-ok); color: #fff;}
.acp-check.is-warn .acp-check-icon {background: var(--acp-warn-soft); border-color: var(--acp-warn); color: var(--acp-warn);}
.acp-check-label {font-weight: 700; font-size: .93rem; color: var(--acp-ink);}
.acp-check-detail {font-size: .84rem; color: var(--acp-muted); margin-top: 2px;}
.acp-sev {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .74rem; font-weight: 700; margin-right: 8px;}
.acp-sev.is-critical {background: var(--acp-err); color: #fff;}
.acp-sev.is-high {background: var(--acp-err-soft); color: var(--acp-err);}
.acp-sev.is-medium {background: var(--acp-warn-soft); color: var(--acp-warn);}
.acp-sev.is-low {background: var(--acp-primary-soft); color: var(--acp-primary);}
.acp-finding-head {font-weight: 700; color: var(--acp-ink); line-height: 1.45; margin-bottom: .35rem;}
.acp-finding-area {font-size: .8rem; color: var(--acp-muted); font-weight: 600; margin-left: 2px;}

/* ---------- Forms, inputs, buttons ---------- */
[data-testid="stForm"] {border: 1px solid var(--acp-line); border-radius: 14px; padding: 1.1rem 1.2rem; background: var(--acp-surface);}
[data-testid="stSidebar"] [data-testid="stForm"] {border: none; padding: 0; background: transparent;}
[data-testid="stTextInput"] label p, [data-testid="stNumberInput"] label p,
[data-testid="stTextArea"] label p, [data-testid="stSelectbox"] label p,
[data-testid="stRadio"] > label p {font-weight: 650; font-size: .85rem; color: var(--acp-ink);}
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {
    border-radius: 10px !important; border-color: var(--acp-line) !important; background: #fff !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within {
    border-color: var(--acp-primary) !important; box-shadow: 0 0 0 3px var(--acp-primary-soft);
}
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
    border-radius: 10px; font-weight: 650; min-height: 2.7rem; border: 1px solid var(--acp-line);
    transition: background-color .15s, border-color .15s;
}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    border-color: var(--acp-primary); color: var(--acp-primary);
}
button[kind="primary"], button[kind="primaryFormSubmit"] {background: var(--acp-primary); border-color: var(--acp-primary); color: #fff;}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {background: var(--acp-primary-dark); border-color: var(--acp-primary-dark); color: #fff;}
button:focus-visible {outline: 3px solid var(--acp-primary-soft); outline-offset: 2px;}
[data-testid="stExpander"] details {border: 1px solid var(--acp-line); border-radius: 12px; background: var(--acp-surface);}
[data-testid="stExpander"] summary {font-weight: 650;}
[data-testid="stAlert"] {border-radius: 12px;}
[data-testid="stCode"], .stCodeBlock {border-radius: 10px;}

/* ---------- Blog reading view ---------- */
h1.blog-title {
    font-family: var(--acp-serif); font-size: 2.4rem; line-height: 1.15; font-weight: 650;
    margin: 0 0 .8rem; padding: 0; letter-spacing: -.015em; color: var(--acp-ink);
}
.st-key-blog_header, .st-key-blog_card {background: var(--acp-surface); border-radius: 16px;}
.st-key-blog_header {padding: .6rem .9rem;}
.st-key-blog_card {padding: 1.6rem clamp(1rem, 4vw, 3.2rem) 2rem;}
.st-key-blog_card p, .st-key-blog_card li {font-family: var(--acp-serif); font-size: 1.16rem; line-height: 1.85; color: #202842;}
.st-key-blog_card h2 {
    font-family: var(--acp-serif); font-size: 1.7rem; font-weight: 650; line-height: 1.25;
    margin-top: 2.2rem; padding-bottom: .45rem; border-bottom: 1px solid var(--acp-line);
}
.st-key-blog_card h3 {font-family: var(--acp-serif); font-size: 1.3rem; font-weight: 650; margin-top: 1.4rem;}
.st-key-blog_card blockquote {margin: 1.2rem 0; padding: .2rem 1.1rem; border-left: 3px solid var(--acp-primary-soft); color: var(--acp-muted); font-style: italic;}
.st-key-blog_card table {width: 100%; font-family: var(--acp-sans); font-size: .95rem; border-collapse: collapse;}
.st-key-blog_card th {background: var(--acp-primary-soft); text-align: left;}
.st-key-blog_card th, .st-key-blog_card td {padding: .55rem .75rem; border: 1px solid var(--acp-line);}
.st-key-blog_card a {color: var(--acp-primary);}
"""


def inject_theme() -> None:
    """Inject fonts and CSS. Call once, right after st.set_page_config()."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


# ------------------------------ Components ------------------------------
def render_sidebar_brand() -> None:
    st.html(
        """
<div class="acp-brand">
  <div class="acp-mark">✦</div>
  <div>
    <div class="acp-brand-name">AImpact Content Agent</div>
    <div class="acp-brand-sub">Blog workflow for customers</div>
  </div>
</div>
"""
    )


def render_topbar(title: str, subtitle: str, status_label: str, status_tone: str) -> None:
    st.html(
        f"""
<div class="acp-topbar">
  <div>
    <h1 class="acp-page-title">{html.escape(title)}</h1>
    <div class="acp-page-sub">{html.escape(subtitle)}</div>
  </div>
  <span class="acp-pill is-{status_tone}">{html.escape(status_label)}</span>
</div>
"""
    )


_STATE_LABEL = {
    "done": "Complete",
    "warn": "Needs attention",
    "error": "Failed",
    "idle": "Waiting",
    "next": "Up next",
    "running": "Running",
}
_STATE_ICON = {"done": "✓", "warn": "!", "error": "✕"}


def render_pipeline(steps: list[tuple[str, str, str]], running: bool = False) -> None:
    """steps: [(name, description, state)] with state done | warn | error | idle."""
    next_index = next((i for i, (_, _, s) in enumerate(steps) if s == "idle"), None)
    any_done = any(s == "done" for _, _, s in steps)
    items = []
    for index, (name, desc, state) in enumerate(steps):
        shown = state
        if index == next_index:
            if running:
                shown = "running"
            elif any_done:
                shown = "next"
        icon = _STATE_ICON.get(shown, str(index + 1))
        items.append(
            f'<li class="acp-step is-{shown}">'
            f'<div class="acp-node">{icon}</div>'
            f'<div class="acp-name">{html.escape(name)}</div>'
            f'<div class="acp-desc">{html.escape(desc)}</div>'
            f'<div class="acp-state">{_STATE_LABEL[shown]}</div>'
            "</li>"
        )
    st.html('<div class="acp-pipeline"><ol class="acp-track" role="list">' + "".join(items) + "</ol></div>")


def render_kpis(items: list[tuple[str, str, str, str]]) -> None:
    """items: [(label, value, hint, tone)] with tone neutral | ok | warn | err."""
    cards = "".join(
        f'<div class="acp-kpi"><div class="acp-kpi-label">{html.escape(label)}</div>'
        f'<div class="acp-kpi-value is-{tone}">{html.escape(value)}</div>'
        f'<div class="acp-kpi-hint">{html.escape(hint)}</div></div>'
        for label, value, hint, tone in items
    )
    st.html(f'<div class="acp-kpis">{cards}</div>')


def render_empty_state() -> None:
    st.html(
        """
<div class="acp-empty">
  <h2>Turn a customer prompt into a reviewed blog post</h2>
  <p>Six agents research the sources, check the facts, plan the brief, write the article and review it.
  You stay in control: edit, ask the writer for changes, resolve findings and approve the final version.</p>
  <div class="acp-how">
    <div class="acp-how-card"><div class="acp-how-num">1</div><h4>Set up a run</h4>
      <p>Enter the campaign, prompt ID and customer website in the sidebar, then choose a target length.</p></div>
    <div class="acp-how-card"><div class="acp-how-num">2</div><h4>Watch the agents work</h4>
      <p>The tracker above updates as each agent finishes, from prompt analysis through the review.</p></div>
    <div class="acp-how-card"><div class="acp-how-num">3</div><h4>Edit and approve</h4>
      <p>Read the draft, fix what the reviewer flagged, then approve a version to export it as PDF or Markdown.</p></div>
  </div>
</div>
"""
    )


def render_status_pill(label: str, tone: str) -> None:
    st.html(f'<span class="acp-pill is-{tone}">{html.escape(label)}</span>')


def render_meter(label: str, count: int, limit: int) -> None:
    over = count > limit
    pct = min(100, int(count / limit * 100)) if limit else 0
    text = f"{count} characters, over the recommended {limit}" if over else f"{count} of {limit} recommended characters"
    st.html(
        f'<div class="acp-meter{" is-over" if over else ""}"><span style="width:{pct}%"></span></div>'
        f'<div class="acp-meter-label{" is-over" if over else ""}">{html.escape(text)}</div>'
    )


def render_outline(headings: list[str]) -> None:
    if not headings:
        st.caption("No sections found.")
        return
    st.html('<ol class="acp-outline">' + "".join(f"<li>{html.escape(h)}</li>" for h in headings) + "</ol>")


def render_versions(versions: list[dict], current: int) -> None:
    rows = []
    for record in reversed(versions):
        is_current = record["version"] == current
        note = record.get("instructions") or ""
        rows.append(
            f'<li class="{"is-current" if is_current else ""}">'
            f'<div class="acp-tl-title">Version {record["version"]}{" (current)" if is_current else ""}</div>'
            f'<div class="acp-tl-meta">{html.escape(str(record["source"]))} · {html.escape(str(record["created_at"]))}</div>'
            + (f'<div class="acp-tl-note">{html.escape(note)}</div>' if note else "")
            + "</li>"
        )
    st.html('<ul class="acp-timeline">' + "".join(rows) + "</ul>")


def render_checklist(rows: list[tuple[str, str, str]]) -> None:
    """rows: [(label, state, detail)] with state ok | warn | todo."""
    out = []
    for label, state, detail in rows:
        icon = {"ok": "✓", "warn": "!", "todo": ""}[state]
        out.append(
            f'<div class="acp-check is-{state}"><div class="acp-check-icon">{icon}</div>'
            f'<div><div class="acp-check-label">{html.escape(label)}</div>'
            f'<div class="acp-check-detail">{html.escape(detail)}</div></div></div>'
        )
    st.html("".join(out))


def severity_badge(severity: str) -> str:
    sev = (severity or "medium").lower()
    if sev not in {"critical", "high", "medium", "low"}:
        sev = "medium"
    return f'<span class="acp-sev is-{sev}">{html.escape(sev.capitalize())}</span>'