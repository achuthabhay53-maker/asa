"""UI theme for the AImpact Content Agent (Streamlit).

Drop this file next to app.py. It only changes presentation; no workflow logic.
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
.block-container {padding-top: 1.5rem; padding-bottom: 4rem; max-width: 1320px;}

h2, h3, [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {
    font-family: var(--acp-sans); color: var(--acp-ink);
    letter-spacing: -.01em; font-weight: 750;
}
[data-testid="stHeading"] h3 {font-size: 1.25rem;}
[data-testid="stCaptionContainer"] {color: var(--acp-muted);}

/* ---------- Brand bar ---------- */
.acp-brand {display: flex; align-items: center; gap: 14px; padding: .25rem 0 1.1rem;}
.acp-mark {
    width: 42px; height: 42px; border-radius: 12px; background: var(--acp-primary);
    color: #fff; display: grid; place-items: center; font-size: 1.25rem; font-weight: 800;
}
.acp-brand-name {font-size: 1.5rem; font-weight: 800; letter-spacing: -.02em; color: var(--acp-ink); line-height: 1.1;}
.acp-brand-sub {font-size: .92rem; color: var(--acp-muted); margin-top: 2px;}

/* ---------- Pipeline stepper (the signature element) ---------- */
.acp-pipeline {
    background: var(--acp-surface); border: 1px solid var(--acp-line);
    border-radius: 16px; padding: 1.25rem 1.5rem 1.1rem; margin: 0 0 1.25rem;
}
.acp-track {display: grid; grid-template-columns: repeat(6, 1fr); list-style: none; margin: 0; padding: 0;}
.acp-step {position: relative; padding-right: 14px;}
.acp-step::before {
    content: ""; position: absolute; top: 17px; left: 42px; right: 8px;
    height: 2px; background: var(--acp-line); border-radius: 2px;
}
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
.acp-name {margin-top: .6rem; font-weight: 700; font-size: .92rem; color: var(--acp-ink);}
.acp-desc {font-size: .82rem; color: var(--acp-muted); margin-top: 1px;}
.acp-state {font-size: .74rem; font-weight: 600; margin-top: 4px; color: var(--acp-muted);}
.acp-step.is-done .acp-state {color: var(--acp-primary);}
.acp-step.is-warn .acp-state {color: var(--acp-warn);}
.acp-step.is-error .acp-state {color: var(--acp-err);}

@media (max-width: 820px) {
    .acp-track {grid-template-columns: repeat(2, 1fr); row-gap: 1.1rem;}
    .acp-step::before {display: none;}
}

/* ---------- Forms, inputs, buttons ---------- */
[data-testid="stForm"] {
    background: var(--acp-surface); border: 1px solid var(--acp-line);
    border-radius: 16px; padding: 1.4rem 1.5rem;
}
[data-testid="stTextInput"] label p, [data-testid="stNumberInput"] label p,
[data-testid="stTextArea"] label p, [data-testid="stSelectbox"] label p,
[data-testid="stRadio"] > label p {font-weight: 650; font-size: .86rem; color: var(--acp-ink);}
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
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {border-color: var(--acp-primary); color: var(--acp-primary);}
button[kind="primary"], button[kind="primaryFormSubmit"] {
    background: var(--acp-primary); border-color: var(--acp-primary); color: #fff;
}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {
    background: var(--acp-primary-dark); border-color: var(--acp-primary-dark); color: #fff;
}
button:focus-visible {outline: 3px solid var(--acp-primary-soft); outline-offset: 2px;}

/* ---------- Expanders, alerts, metrics, code ---------- */
[data-testid="stExpander"] details {
    border: 1px solid var(--acp-line); border-radius: 12px; background: var(--acp-surface);
}
[data-testid="stExpander"] summary {font-weight: 650;}
[data-testid="stAlert"] {border-radius: 12px;}
[data-testid="stMetric"] {background: var(--acp-surface); border: 1px solid var(--acp-line); padding: 12px 14px; border-radius: 12px;}
[data-testid="stCode"], .stCodeBlock {border-radius: 10px;}

/* ---------- Draft status pill ---------- */
.acp-pill {
    display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px;
    border-radius: 999px; font-weight: 700; font-size: .88rem; margin: .25rem 0 .75rem;
}
.acp-pill::before {content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor;}
.acp-pill.is-ok {background: var(--acp-ok-soft); color: var(--acp-ok);}
.acp-pill.is-draft {background: var(--acp-warn-soft); color: var(--acp-warn);}

/* ---------- Blog view (reading mode) ---------- */
h1.blog-title {
    font-family: var(--acp-serif); font-size: 2.4rem; line-height: 1.15; font-weight: 650;
    margin: 0 0 .8rem; padding: 0; letter-spacing: -.015em; color: var(--acp-ink);
}
.st-key-blog_header, .st-key-blog_card {
    background: var(--acp-surface); border-radius: 16px;
}
.st-key-blog_header [data-testid="stHeaderActionElements"],
.st-key-blog_card [data-testid="stHeaderActionElements"] {display: none;}
.st-key-blog_header {padding: .6rem .9rem;}
.st-key-blog_card {padding: 1.6rem clamp(1rem, 4vw, 3.2rem) 2rem;}
.st-key-blog_card p, .st-key-blog_card li {
    font-family: var(--acp-serif); font-size: 1.16rem; line-height: 1.85; color: #202842;
}
.st-key-blog_card h2 {
    font-family: var(--acp-serif); font-size: 1.7rem; font-weight: 650; line-height: 1.25;
    margin-top: 2.2rem; padding-bottom: .45rem; border-bottom: 1px solid var(--acp-line);
}
.st-key-blog_card h3 {font-family: var(--acp-serif); font-size: 1.3rem; font-weight: 650; margin-top: 1.4rem;}
.st-key-blog_card blockquote {
    margin: 1.2rem 0; padding: .2rem 1.1rem; border-left: 3px solid var(--acp-primary-soft);
    color: var(--acp-muted); font-style: italic;
}
.st-key-blog_card table {width: 100%; font-family: var(--acp-sans); font-size: .95rem; border-collapse: collapse;}
.st-key-blog_card th {background: var(--acp-primary-soft); text-align: left;}
.st-key-blog_card th, .st-key-blog_card td {padding: .55rem .75rem; border: 1px solid var(--acp-line);}
.st-key-blog_card a {color: var(--acp-primary);}
"""


def inject_theme() -> None:
    """Inject fonts and CSS. Call once, right after st.set_page_config()."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def render_header() -> None:
    st.html(
        """
<div class="acp-brand">
  <div class="acp-mark">✦</div>
  <div>
    <div class="acp-brand-name">AImpact Content Agent</div>
    <div class="acp-brand-sub">From a customer prompt to a sourced, reviewed, publish-ready blog.</div>
  </div>
</div>
"""
    )


_STATE_LABEL = {
    "done": "Complete",
    "warn": "Needs attention",
    "error": "Failed",
    "idle": "Waiting",
    "next": "Up next",
}
_STATE_ICON = {"done": "✓", "warn": "!", "error": "✕"}


def render_pipeline(steps: list[tuple[str, str, str]]) -> None:
    """steps: [(name, description, state)] where state is done | warn | error | idle."""
    next_index = next((i for i, (_, _, state) in enumerate(steps) if state == "idle"), None)
    items = []
    for index, (name, desc, state) in enumerate(steps):
        shown = "next" if index == next_index and any(s[2] == "done" for s in steps) else state
        icon = _STATE_ICON.get(state, str(index + 1))
        css_class = f"is-{shown}"
        items.append(
            f'<li class="acp-step {css_class}">'
            f'<div class="acp-node">{icon}</div>'
            f'<div class="acp-name">{html.escape(name)}</div>'
            f'<div class="acp-desc">{html.escape(desc)}</div>'
            f'<div class="acp-state">{_STATE_LABEL[shown]}</div>'
            "</li>"
        )
    st.html(
        '<div class="acp-pipeline"><ol class="acp-track" role="list">' + "".join(items) + "</ol></div>"
    )


def render_status_pill(version: int, approved: bool) -> None:
    if approved:
        st.html(f'<span class="acp-pill is-ok">Version {version} approved</span>')
    else:
        st.html(f'<span class="acp-pill is-draft">Version {version} is a draft, not approved</span>')