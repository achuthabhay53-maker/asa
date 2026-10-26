# AImpact Content Agent (ACA)

The current interactive application is a Streamlit workflow that reads AImpact prompt evidence, researches sources and company facts, creates a content brief, generates a blog draft, and can optionally review and revise that draft. Human approval is still required before publication; WordPress publishing and post-publication re-audits are not implemented.

**Current flow:** Agent 1 → Agent 2 → Agent 3 → Agent 4 → Agent 5 → optional Agent 6 (Reviewer). Agent 6 runs only when requested from the Streamlit interface and makes one model call. It returns the full revised article with a 16,384-token output limit. Human review is required before publishing.

The repository also contains a separate FastAPI entrypoint and deployment configuration; those are not used by the Streamlit workflow.

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

```
aca/
├── main.py                 FastAPI entrypoint · CORS · routers · lifespan
├── config.py               pydantic-settings, one place to read env
├── api/                    HTTP surface
├── orchestrator/           LangGraph state machine + async runner
├── agents/                 Prompt · Sources · Ground truth · Brief · Writer · Reviewer
├── tools/                  AImpact REST, Claude, WordPress, Langfuse
├── models/                 SQLAlchemy ORM — 7 tables
├── db/                     Engine + Alembic migrations
├── workers/                APScheduler nightly picker + day-7/30 feedback
├── security/               API-token verify + KMS envelope encryption
└── observability/          OpenTelemetry setup
```

See [`docs/04-code-architecture.md`](docs/04-code-architecture.md) for module-by-module explanation and the dependency graph.

---

## Local dev

```bash
cp .env.example .env
# fill in DATABASE_URL, AIMPACT_API_TOKEN, ANTHROPIC_API_KEY at minimum
python -m venv .venv && source .venv/Scripts/activate   # or .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
scripts/dev_run.sh
```

The service listens on `http://localhost:8002`. `GET /aca/health` should return `{"status":"ok"}`.

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

## The 30-second pitch

1. AImpact tells you which prompts you lose on **and which URLs the models cite** when they answer those prompts.
2. ACA reads both. It figures out what the cited sources cover that you don't, forms a content-gap hypothesis with an editorial angle, drafts a blog against it, human approves, publishes to WordPress.
3. Every hop is persisted and traced, so jobs are resumable and every draft is auditable end-to-end.
4. Success = measurable AImpact score lift on covered prompts within 30 days. Cost target ≤ $2 per published blog.

---

## Author

Aditya Pasare · 2026-09-29
