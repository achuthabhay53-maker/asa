# AImpact Content Agent (ACA)

The current interactive application is a Streamlit workflow that reads AImpact prompt evidence, researches sources and company facts, creates a content brief, generates a blog draft, and can optionally review and revise that draft. Human approval is still required before publication; WordPress publishing and post-publication re-audits are not implemented.

**Current flow:** Agent 1 → Agent 2 → Agent 3 → Agent 4 → Agent 5 → optional Agent 6 (Reviewer). Agent 6 runs only when requested from the Streamlit interface and makes one model call. It returns the full revised article with a 16,384-token output limit. Human review is required before publishing.

The repository also contains a separate FastAPI entrypoint and deployment configuration. The Streamlit workflow doesn't use them, and they don't currently start (see [Known gaps](#known-gaps)).

A Mermaid diagram of the current flow is in [`current-agent-flow.md`](current-agent-flow.md).

---

## How the workflow runs

The pipeline is a LangGraph `StateGraph` built by `build_pipeline()` in
`aca/orchestrator/pipeline.py`. All agents share one `PipelineState`.

```
Agent 1 → Agent 2 → Agent 3 ─(verified?)→ Agent 4 ─(brief has sections?)→ Agent 5 ─(review requested?)→ Agent 6
                         └─ no → END             └─ no → END                       └─ no → END
```

The entry router starts at the first agent whose output is missing. Because of this, Agent 6 can run on its own once Agents 1–5 have produced their outputs.

| # | Agent | Code | Output model | What it does |
|---|---|---|---|---|
| 1 | Prompt Intelligence | `aca/agents/prompt_intelligence.py` | `PromptAnalysis` | Interprets the prompt and stored AI answers; works out customer visibility and competitor mentions |
| 2 | Source Intelligence | `aca/agents/source_intelligence.py` | `SourceIntelligence` | Ranks cited URLs, fetches and parses those pages, and analyses their structure and style |
| 3 | Ground Truth & Research | `aca/agents/ground_truth.py` | `GroundTruthGuide` + `Agent4HandoffPackage` | Researches the customer website with Exa, checks quotes against page text, sorts claims into verified, limited or unresolved, and builds the Agent 3 → 4 handoff |
| 4 | Content Brief | `aca/agents/content_brief.py` | `ContentBrief` | Turns the outputs of Agents 1–3 into a section plan and writing guidance |
| 5 | Writer | `aca/agents/writer.py` | `WriterOutput` | Streams the article and adds SEO metadata, a FAQ, tables and claim/citation tracking |
| 6 | Reviewer (optional) | `aca/agents/reviewer.py` | `ReviewOutput` | Makes one Claude call that returns the full revised article and its findings |

The output models live in `aca/models/`, one file per agent. All agents call Claude through the `anthropic` SDK.

---

## Repo layout

```
aca/
├── agents/            The six agent classes (table above)
├── models/            Pydantic input/output schemas, one file per agent
├── orchestrator/
│   ├── pipeline.py    LangGraph graph, nodes, routing, PipelineState
│   └── state.py       Older shared models (still imported by tools/aimpact.py and tests)
├── tools/
│   ├── aimpact.py         Direct psycopg queries against the AImpact PostgreSQL DB
│   ├── web_research.py    Exa search (HTTP API) + page extraction, used by Agent 3
│   └── source_fetcher.py  Tavily fetcher + site URL discovery (used only by a manual test)
├── evidence.py        Evidence URL canonicalisation shared by Agents 3, 5 and 6
├── article_sections.py, review_diff.py, pdf_export.py, output_storage.py   UI/output helpers
├── config.py          pydantic-settings (used only by the FastAPI entrypoint)
├── main.py            FastAPI entrypoint (not working, see Known gaps)
├── workers/           APScheduler setup; feedback job is a stub
└── observability/     Telemetry stub
app.py                 Streamlit UI (main way to run the workflow)
scripts/               Run single agents or the workflow, evaluator, debugging helpers
tests/unit/            Offline unit tests (synthetic fixtures in tests/agent_fixtures.py)
current-agent-flow.md  Mermaid diagram of the current flow
```

---

## Local dev

```bash
cp .env.example .env
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Environment variables used by the Streamlit workflow:

| Variable | Used by |
|---|---|
| `DATABASE_URL` | `aca/tools/aimpact.py`: reads prompts, stored AI answers and citations from AImpact PostgreSQL |
| `ANTHROPIC_API_KEY` | All agents and the evaluator |
| `ANTHROPIC_MODEL_NAME` | Agents 1–4 and the evaluator (defaults to `claude-haiku-4-5-20251001` when unset). Agents 5 and 6 don't read it; they default to `claude-haiku-4-5` in code |
| `EXA_API_KEY` | Agent 3 web research |
| `TAVILY_API_KEY` | Only `aca/tools/source_fetcher.py` (manual test) |

`ANTHROPIC_MODEL_NAME` and `TAVILY_API_KEY` are not yet listed in `.env.example`, so add them to `.env` yourself. The `AIMPACT_*`, `ANTHROPIC_MODEL_HAIKU`/`_SONNET`, AWS, Langfuse and cost-cap variables in `.env.example` are read only by `aca/config.py` (FastAPI path). The Streamlit workflow doesn't use them.

The Streamlit form takes a campaign ID, a prompt ID, the customer website, an optional customer name, the number of sources to analyse (1–20) and a blog target word count. `Run Agents 1–5` runs the pipeline. Agent 6 runs only when you request it from its tab.

### Command-line scripts

| Script | Purpose |
|---|---|
| `scripts/run_workflow.py --campaign-id 81 --prompt-id 1409 --website https://example.com` | Runs the pipeline from the CLI; saves `agent1_output.json`–`agent3_output.json` and `workflow_output.json` |
| `scripts/run_agent3.py --website … --agent1 … --agent2 …` | Runs Agent 3 on saved Agent 1/2 JSON |
| `scripts/run_agent4.py` | Runs Agent 4 on saved Agent 1–3 JSON |
| `scripts/run_agent5.py <input.json>` | Runs Agent 5 on a `WriterInput` JSON (see `scripts/agent5_input.example.json`) |
| `scripts/run_agent6.py` | Runs the Reviewer on saved Agent 1–5 JSON (details below) |
| `scripts/evaluate_agent_outputs.py` | DeepEval G-Eval scoring of saved outputs (details below) |

Run tests (offline; no model calls):

```bash
pytest tests/unit
```

## Evaluate real agent outputs

The offline pytest tests use synthetic fixtures to check schemas and workflow
handoffs. For quality evaluation, `scripts/evaluate_agent_outputs.py` instead
loads actual saved Agent 1–5 JSON outputs, validates their schemas, and scores
them with Claude using DeepEval G-Eval. It also runs a Faithfulness check on
Agent 5's tracked claims against Agent 3's evidence quotes. It does not rerun
the agents. By default, it reads `agent1_output.json` through
`agent5_output.json` from the repository root and writes
`agent_quality_report.json`.

The Streamlit app (`streamlit run app.py`) refreshes those five agent output
files for every valid run. It clears the previous run's files before starting,
then saves each agent's JSON as that agent completes. If a run stops early,
only the completed agents' files remain, so the evaluator will report missing
outputs instead of combining results from different runs. Each saved output
also carries a shared `_run_id`, which the evaluator checks for consistency.

The Streamlit form defaults Agent 5's target to 800 words; adjust the
`Blog target word count` field from 500 to 5,000 as needed. The target guides
the draft but is not a pass/fail limit: the app still displays the article and
reports its actual word count if Claude writes more or less. Agent 5 streams
Claude responses to support long requests and the app passes no more than five
FAQ questions from Agent 4 into the Writer.
Agent 5 returns a separate SEO title (up to 60 characters) and meta
description (up to 155 characters), alongside the H1 and article Markdown.
If Claude returns invalid metadata, Agent 5 creates a short fallback from the
article title instead of failing the blog run.
Agent 5 also includes at least one useful Markdown table, based only on
information in the brief and supplied evidence. Unsupported details are
marked as unspecified rather than guessed.
The `Read blog` tab offers a formatted PDF download with styled headings,
readable tables, FAQ question headings, clickable web links, and page numbers.
The Markdown source download remains available separately.
Agent 3 researches the customer website and external sources independently;
Agent 2's content structure and style analysis is used by Agent 4, not Agent 3.
Agent 3's evidence checks focus on claims about the customer, its products,
services, capabilities, processes, and results. Generic industry context and
educational explanations are kept distinct from company facts; a missing
company-specific quote should not by itself create a research gap for generic
context. Registered evidence quotes must still match retrieved page text.

The Writer receives Agent 3's evidence wording boundaries. Citation metadata
is checked against the evidence IDs supplied to the Writer, and source URLs in
claim metadata are derived from those evidence records. `citation_status`
reports whether a mapping is linked, missing, or mismatched; a linked citation
is not proof that the source semantically supports the claim. Claim support is
reported separately as supported, partially supported, unsupported,
contradicted, inconclusive, or not assessed; this is an AI assessment, not
independent verification. Citation mapping errors do not by themselves make a
claim unsupported or contradicted. Mismatches and unsupported company-specific
claims and numerical statements are recorded in `unresolved_items`, and
FAQ answers outside the 40–50 word target are also flagged. These checks are
review aids, not proof of factual accuracy; human review is required before
publication.

After Agents 1–5 finish, open `Agent 6 · Review` and select
`Review and revise blog` to run the optional Reviewer. It uses one Claude call,
checks the draft and approved evidence, then returns the complete revised
Markdown article, findings, and claims needing verification. Its output limit is
16,384 tokens. The reviewed
article view highlights added or changed words in green; unchanged text keeps
its normal styling. The original draft remains available in `Agent 5 · Draft
blog`. Agent 6 output is saved separately as `agent6_output.json`; it does
not replace the Agent 5 draft or automatically publish anything.

To run Agent 6 separately from saved Agent outputs, use the standalone script.
It validates Agent 1-5 JSON files and calls only the Reviewer; it does not rerun
the LangGraph pipeline. The files must be from the same run. If the customer
website is not present in Agent 3 output, pass it explicitly:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_agent6.py --website https://example.com
```

By default, the script reads `agent1_output.json` through `agent5_output.json`
from the repository root and writes `agent6_output.json` and
`agent6_reviewed.md`. To use a saved Agent 5 result elsewhere, pass its path
with `--agent5`; also pass matching Agent 1-4 paths if they are not in the
repository root. This command makes one Claude API call. The local loader tests
do not call a model:

```powershell
.\.venv\Scripts\python.exe -m pytest .\tests\unit\test_run_agent6.py -q
```

Install the optional evaluator into a separate environment:

```powershell
python -m venv .venv-llm-eval
.\.venv-llm-eval\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-llm-eval\Scripts\python.exe -m pip install -e ".[llm-eval]"
```

The evaluator reads `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL_NAME` from `.env`.
Run it after the five real Agent output files have been produced:

```powershell
.\.venv-llm-eval\Scripts\python.exe scripts\evaluate_agent_outputs.py
```

To score another set of outputs, supply all five files explicitly:

```powershell
.\.venv-llm-eval\Scripts\python.exe scripts\evaluate_agent_outputs.py `
  --agent1 path\to\agent1_output.json `
  --agent2 path\to\agent2_output.json `
  --agent3 path\to\agent3_output.json `
  --agent4 path\to\agent4_output.json `
  --agent5 path\to\agent5_output.json `
  --report path\to\agent_quality_report.json
```

The command makes Claude judge calls and may incur API costs. It reports a
0–1 score and reason for each agent, plus Agent 5 Faithfulness; scores below
the default `0.7` threshold are marked `REVIEW` and make the command exit with
status 1. Change the judge with `--judge-model` or adjust the threshold with
`--threshold 0.6`. Treat these model-judged scores as review signals rather
than deterministic proof of factual correctness. No judge evaluations are run
by ordinary pytest.

---

## Known gaps

The following are not implemented yet, or don't match the original design:

- **FastAPI / App Runner path doesn't work.** `aca/main.py` imports `aca.api`, which doesn't exist. As a result, `scripts/dev_run.sh` and `apprunner.yaml` (`uvicorn aca.main:app`) fail at startup.
- **No ORM database or migrations.** There is no `aca/db/` and no Alembic setup. `aca/models/` contains Pydantic schemas, not SQLAlchemy tables.
- **Not built yet:** WordPress publishing, post-publication re-audits, the API-token/KMS security layer, OpenTelemetry/Langfuse tracing (`observability/telemetry.py` is a stub), and the scheduled feedback job (`workers/feedback.py` is a stub; the scheduler job is commented out).
- **Agent 2's output reaches Agent 3 but isn't used there.** Agent 4 uses it.
- **Two sets of state models.** `aca/orchestrator/state.py` is from an earlier design, but `tools/aimpact.py` still imports it.

---

## The 30-second pitch (product goal)

This is the target product vision. Steps such as WordPress publishing and score-lift tracking are not implemented yet (see [Known gaps](#known-gaps)).

1. AImpact tells you which prompts you lose on **and which URLs the models cite** when they answer those prompts.
2. ACA reads both. It figures out what the cited sources cover that you don't, forms a content-gap hypothesis with an editorial angle, drafts a blog against it, human approves, publishes to WordPress.
3. Every hop is persisted and traced, so jobs are resumable and every draft is auditable end-to-end.
4. Success = measurable AImpact score lift on covered prompts within 30 days. Cost target ≤ $2 per published blog.

---

