# AImpact Content Agent (ACA)

Multi-agent workflow that closes GEO (Generative Engine Optimization) gaps discovered by [AImpact](https://aimpact.example.com). ACA reads AImpact's low-scoring prompts **and the sources that ChatGPT / Claude / Gemini / Google AI cite in their answers**, identifies the content gap, drafts remediation content, puts it through an editorial review, waits for a human approval, and publishes to WordPress. It then re-audits the covered prompts at day 7 and day 30 to attribute lift.

**Status:** v0.2 — source-driven flow. Human approval before publish is mandatory. Target tenant: AImpact itself (dogfood).

**Change from v0.1:** the separate Keyword agent (Google Ads Keyword Planner) has been dropped. The Source Analyzer now derives the editorial angle straight from the citations, so the flow is **Analyzer → Brief → Writer → Reviewer → Publisher** (five agents, not six).

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

- [`docs/05-agents.md`](docs/05-agents.md) — deep dive on each of the five agents.
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
├── agents/                 Analyzer · Brief · Writer · Reviewer · Publisher (5)
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

---

## The 30-second pitch

1. AImpact tells you which prompts you lose on **and which URLs the models cite** when they answer those prompts.
2. ACA reads both. It figures out what the cited sources cover that you don't, forms a content-gap hypothesis with an editorial angle, drafts a blog against it, human approves, publishes to WordPress.
3. Every hop is persisted and traced, so jobs are resumable and every draft is auditable end-to-end.
4. Success = measurable AImpact score lift on covered prompts within 30 days. Cost target ≤ $2 per published blog.

---

## Author

Aditya Pasare · 2026-09-29
