# AImpact Content Agent (ACA)

The interactive application uses an HTML, CSS, and JavaScript frontend served by FastAPI. Python runs the existing research, writing, review, and workflow code behind the API. Agent 6 reports issues for a human to assess; it does not revise the campaign draft. Human approval is still required before publication; WordPress publishing and post-publication re-audits are not implemented.

**Current flow:** Agent 1 → Agent 2 → Agent 3 → Agent 4 → Agent 5 → Agent 6 (Reviewer) → human editing and approval. Reviewer findings with an exact, unique article excerpt appear as severity-colored inline highlights with hover explanations; findings without an unambiguous text anchor remain in the review-decision controls. The human can edit directly or request an evidence-constrained Writer revision. Revisions apply as a new draft version without a separate proposal/diff screen; a changed version requires a fresh review and approval.

### General Blog Generation

The application also includes **General Blog Generation** as a separate
workflow in the sidebar. This workflow does not use AImpact data or Agents
1–4. New intake, research, and brief agents use the company website and
user-provided reference URLs (including URLs detected in the prompt). Public
HTML pages and text-based PDFs are supported. Users can also upload up to five
text-based research PDFs (15 MB each) when a reference link cannot be fetched;
uploaded documents are parsed in memory and are not saved to disk. Scanned PDFs
that require OCR are not supported. Blocked or inaccessible sources are
reported in the result rather than treated as evidence. Exact source quotes are
matched against their source text before they are passed to the existing Agent 5
Writer and Agent 6 Reviewer. Blocking review findings trigger a bounded revision
loop; unresolved blocking findings are surfaced for human review.

The General Blog workflow uses the existing `ANTHROPIC_API_KEY` and requires
`EXA_API_KEY` for company-site research.
Users can sign up, sign in, and sign out. PostgreSQL stores account records,
hashed login-session tokens, user-owned blog runs, and immutable blog versions.
The Saved Runs page loads the signed-in user's history from the backend.

`ACA_DATABASE_URL` is the database URL for accounts and blog history.
`LANGGRAPH_CHECKPOINT_DATABASE_URL` is the PostgreSQL URL for LangGraph
checkpoints. These URLs can point to separate databases or to separate
databases on the same PostgreSQL service. `DATABASE_URL` remains the URL for
the temporary, read-only AImpact development reader.

The LangGraph checkpointer stores campaign workflow state. Each campaign run
also has an owner record in the ACA database. Draft versions, review findings,
finding decisions, and approvals remain in the workflow checkpoint. Human
edits and Writer revisions create new saved blog versions. A changed draft
still requires a new review and approval.

Runs created before user accounts were added do not have owner records. The
app does not assign those runs to a user automatically. General Blog results
from an earlier server process were held in memory and cannot be recovered
after that process stopped.

Claude input and output tokens are recorded for each API call, tagged with the workflow run ID and agent, and saved in the run checkpoint. The web UI shows the run outputs and workflow status.

---

## Documentation map

The four foundational diagrams live in [`docs/images/`](docs/images). Each has an extensive companion document:

| # | Diagram | Companion doc |
|---|---|---|
| 1 | Architecture — components and boundaries | [`docs/01-architecture.md`](docs/01-architecture.md) |
| 2 | Runtime coordination — the 20-hop workflow | [`docs/02-workflow.md`](docs/02-workflow.md) |
| 3 | System architecture — infra, network, data, SaaS | [`docs/03-system-architecture.md`](docs/03-system-architecture.md) |
| 4 | Code architecture — repo layout, module dependencies | [`docs/04-code-architecture.md`](docs/04-code-architecture.md) |

Supporting docs:

- [`docs/05-agents.md`](docs/05-agents.md) — deep dive on the content agents.
- [`docs/06-data-model.md`](docs/06-data-model.md) — schema + retention + ER notes.
- [`docs/07-api-reference.md`](docs/07-api-reference.md) — every ACA endpoint with request/response shapes.
- [`docs/08-integration-aimpact.md`](docs/08-integration-aimpact.md) — exact AImpact endpoints ACA calls, plus the small additive changes AImpact must ship.
- [`docs/09-deployment.md`](docs/09-deployment.md) — App Runner, Neon, CloudFront, env vars, rollout.
- [`docs/10-security.md`](docs/10-security.md) — auth, secrets, multi-tenant, blast-radius rules.

---

## Repo layout

The FastAPI entry point is `app.py`. Static HTML, CSS, and JavaScript are in
`web/static/`. Application logic and presentation helpers are grouped by responsibility:

```
src/
├── agents/
│   ├── aimpact_blog/ ACA Agents 1–6
│   └── general_blog/ Independent General Blog agents
├── models/          Typed agent inputs and outputs
├── observability/   Claude token usage and telemetry
├── orchestrator/    ACA and General Blog workflow coordinators
├── tools/           AImpact, web research, and reference retrieval
├── ui/              Article rendering helpers, PDF, and review logic
├── utils/           Shared helpers, evidence URLs, and writer lint
└── workers/         Background worker scaffolding
web/
└── static/           HTML, CSS, and JavaScript frontend
```

See [`docs/04-code-architecture.md`](docs/04-code-architecture.md) for module-by-module explanation and the dependency graph.

---

## Local dev

```bash
cp .env.example .env
# Set ACA_DATABASE_URL, LANGGRAPH_CHECKPOINT_DATABASE_URL, and the required API keys.
# Set ACA_COOKIE_SECURE=true when the app is served over HTTPS.
# Set DATABASE_URL only when the temporary AImpact PostgreSQL reader is needed.
python -m venv .venv
# Windows: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m alembic upgrade head
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` in a browser. The FastAPI API documentation is
available at `http://127.0.0.1:8000/docs`.

Run tests:

```bash
pytest
```

## Evaluate real agent outputs

The offline pytest tests use synthetic fixtures to check schemas and workflow
handoffs. For quality evaluation, `scripts/evaluate_agent_outputs.py` instead
loads actual saved Agent 1–5 JSON outputs, validates their schemas, and scores
them with Claude using DeepEval G-Eval. It also runs a Faithfulness check on
Agent 5's tracked claims against Agent 3's evidence quotes. It does not rerun
the agents. By default, it reads `outputs/agent1_output.json` through
`outputs/agent5_output.json` and writes `outputs/agent_quality_report.json`.

The campaign workflow refreshes those five agent output
files in the `outputs` folder for every valid run. It clears the previous run's
files before starting, then saves each agent's JSON as that agent completes. If a run stops early,
only the completed agents' files remain, so the evaluator will report missing
outputs instead of combining results from different runs. Each saved output
also carries a shared `_run_id`, which the evaluator checks for consistency.
Agent 1's latest structured output is also saved separately to
`outputs2/agent1/agent1_output.json` for debugging; it includes the same
`_run_id` and is not cleared when a new run starts.
General Blog stage diagnostics are saved per run under
`outputs2/general_blog/<run_id>/`. Each stage JSON includes its inputs,
validated output (or error), token usage, and duration. Repeated writer and
reviewer stages use numbered filenames.

The campaign form defaults Agent 5's target to 800 words; adjust the
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
The Article view offers a formatted PDF download with styled headings,
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

After Agents 1–5 finish, open the Review view and select
`Run Agent 6 review` to run the optional Reviewer. It uses one Claude call,
checks the draft and approved evidence, and returns findings and claims that
need verification. The reviewer does not rewrite the article. The Article view
highlights exact matching passages when possible. Use Edit & revise to change
the draft or ask Agent 5 for a revision. A changed draft needs a new review.
Agent 6 output is saved separately as `outputs/agent6_output.json`; it does
not replace the Agent 5 draft or automatically publish anything.

The app requires sign-in for workflow and history APIs. Set
`ACA_COOKIE_SECURE=true` when serving it over HTTPS. Before public launch, add
rate limits, email verification, password reset, and a production security
review. Keep the app on localhost until those controls are in place.

To run Agent 6 separately from saved Agent outputs, use the standalone script.
It validates Agent 1-5 JSON files and calls only the Reviewer; it does not rerun
the LangGraph pipeline. The files must be from the same run. If the customer
website is not present in Agent 3 output, pass it explicitly:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_agent6.py --website https://example.com
```

By default, the script reads `outputs/agent1_output.json` through
`outputs/agent5_output.json` and writes `outputs/agent6_output.json`. To use a
saved Agent 5 result elsewhere, pass its path with `--agent5`; also pass
matching Agent 1-4 paths if they are not in the `outputs` folder. This command
makes one Claude API call. The local loader tests do not call a model:

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

## The 30-second pitch

1. AImpact tells you which prompts you lose on **and which URLs the models cite** when they answer those prompts.
2. ACA reads both. It figures out what the cited sources cover that you don't, forms a content-gap hypothesis with an editorial angle, drafts a blog against it, human approves, publishes to WordPress.
3. Every hop is persisted and traced, so jobs are resumable and every draft is auditable end-to-end.
4. Success = measurable AImpact score lift on covered prompts within 30 days. Cost target ≤ $2 per published blog.

---

## Author

Aditya Pasare · 2026-09-29
