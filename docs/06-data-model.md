# 06 · Data Model

Seven tables, one schema, all prefixed `aca_`. Lives in its own Postgres database, separate from AImpact's.

Design tenets:
- **Everything is tenant-scoped.** Every table (except `aca_tenants` itself) has `tenant_id` and a partial index on it.
- **Immutable outputs, mutable status.** Agent outputs (gap, brief, draft, publication) are append-only. Only `aca_jobs.status` and `aca_publications.lift_*` are mutable after creation.
- **Foreign keys to AImpact are logical.** We store `aimpact_user_id`, `campaign_id`, `prompt_id` as plain integers/strings — no cross-database FK.
- **JSONB for open-ended fields.** Anything the LLM emits as a nested structure lives in a JSONB column.

**v0.2 change:** the `aca_keywords` table is gone. The gap (`aca_gaps`) now carries the sources and editorial angle directly.

---

## Entity relationship (text)

```
aca_tenants (1) ─┬─ (N) aca_jobs (1) ─┬─ (N) aca_gaps
                 │                    ├─ (1) aca_briefs
                 │                    ├─ (N) aca_drafts   (versions)
                 │                    ├─ (1) aca_publications
                 │                    └─ (N) aca_agent_traces
                 └─ (1) tenant secrets  (KMS-encrypted WP creds inline)
```

---

## `aca_tenants`

| Column | Type | Notes |
|---|---|---|
| `id` | `SERIAL PK` | |
| `aimpact_user_id` | `BIGINT UNIQUE NOT NULL` | logical FK to AImpact `users.id` |
| `api_token_hash` | `TEXT NOT NULL` | bcrypt hash of the `at_...` token |
| `api_token_prefix` | `TEXT NOT NULL` | first 8 chars for prefix-index lookup |
| `daily_usd_cap` | `NUMERIC(8,2)` | overrides `ACA_DAILY_USD_CAP_PER_TENANT` |
| `wp_credentials_ct` | `BYTEA` | KMS envelope-encrypted: `{site_url, username, app_password}` |
| `wp_credentials_valid` | `BOOLEAN DEFAULT TRUE` | flipped false on 401 from WP |
| `brand_summary` | `TEXT` | short brand description used in Analyzer prompt |
| `style_guide_md` | `TEXT` | prompt-cached in Sonnet calls |
| `taxonomy_json` | `JSONB` | site categories, tags, url patterns |
| `default_category_id` | `INT` | WordPress category to file posts under |
| `enabled` | `BOOLEAN DEFAULT FALSE` | admin toggle |
| `created_at` | `TIMESTAMPTZ DEFAULT now()` | |
| `updated_at` | `TIMESTAMPTZ` | |

Indexes: `UNIQUE (aimpact_user_id)`, `UNIQUE (api_token_prefix)`.

---

## `aca_jobs`

| Column | Type | Notes |
|---|---|---|
| `id` | `UUID PK` | client-safe |
| `tenant_id` | `INT FK aca_tenants.id ON DELETE CASCADE` | |
| `campaign_id` | `INT` | logical FK to AImpact `campaigns.id` |
| `status` | `TEXT` | `pending` `running` `awaiting_human` `needs_human` `completed` `failed` `throttled` `timed_out` |
| `current_agent` | `TEXT` | analyzer/brief/writer/reviewer/publisher, or null |
| `revise_count` | `INT DEFAULT 0` | |
| `retry_count` | `INT DEFAULT 0` | |
| `error` | `TEXT` | populated on `failed` |
| `started_at` | `TIMESTAMPTZ` | |
| `ended_at` | `TIMESTAMPTZ` | |
| `created_at` | `TIMESTAMPTZ DEFAULT now()` | |

Indexes: `(tenant_id, status, created_at DESC)`, partial `(status)` where status IN pending/running.

**State machine:**
```
pending ─► running ─┬─► completed
                    ├─► awaiting_human ─► running ─► completed
                    ├─► needs_human ─► (manual)
                    ├─► failed
                    ├─► throttled ─► (auto resume when quota clears)
                    └─► timed_out ─► (resume) ─► running
```

---

## `aca_gaps`

Source Analyzer output. One row per prompt covered by a job.

| Column | Type | Notes |
|---|---|---|
| `id` | `SERIAL PK` | |
| `job_id` | `UUID FK aca_jobs.id ON DELETE CASCADE` | |
| `tenant_id` | `INT FK` | denormalized |
| `prompt_id` | `TEXT` | AImpact `campaign_prompts.id` |
| `prompt_text` | `TEXT` | | 
| `baseline_score` | `NUMERIC(5,2)` | at time of analysis |
| `cited_sources` | `JSONB` | `[{model, url, domain, competitor, excerpt}, ...]` |
| `themes` | `JSONB` | `[str, ...]` — recurring topics competitors cover |
| `candidate_themes` | `JSONB` | Stage 1 themes before page verification |
| `theme_evidence` | `JSONB` | Verified themes with source ID, fetched URL, and exact evidence quote |
| `theme_assessments` | `JSONB` | Per-theme verification status, sources, models, competitors, and opportunity |
| `candidate_target_angle` | `TEXT` | Stage 1 angle suggestion, not treated as verified |
| `target_angle` | `TEXT` | one-line editorial angle |
| `hypothesis` | `TEXT` | natural-language gap statement |
| `target_site_coverage` | `TEXT` | `unknown` until a target-site inventory is compared |
| `content_gap_status` | `TEXT` | `not_established` until first-party coverage is assessed |
| `opportunity` | `TEXT` | Investigation prompt; not a claim that a gap exists |
| `verification_status` | `TEXT` | verified / partially_verified / unverified / unavailable |
| `verification_summary` | `TEXT` | concise summary of page-verification coverage |
| `created_at` | `TIMESTAMPTZ DEFAULT now()` | |

Index: `(job_id)`.

---

## `aca_briefs`

| Column | Type | Notes |
|---|---|---|
| `id` | `SERIAL PK` | |
| `job_id` | `UUID FK aca_jobs.id ON DELETE CASCADE` | |
| `tenant_id` | `INT FK` | |
| `title` | `TEXT` | |
| `slug` | `TEXT` | |
| `meta` | `TEXT` | |
| `outline` | `JSONB` | `[{h2, intent_hint, format?}, ...]` |
| `faqs` | `JSONB` | `[{q, a}, ...]` |
| `internal_links` | `JSONB` | `[url, ...]` |
| `created_at` | `TIMESTAMPTZ DEFAULT now()` | |

Index: `(job_id)` unique.

---

## `aca_drafts`

Writer + Reviewer output. Versioned per revision.

| Column | Type | Notes |
|---|---|---|
| `id` | `SERIAL PK` | |
| `brief_id` | `INT FK aca_briefs.id ON DELETE CASCADE` | |
| `tenant_id` | `INT FK` | |
| `version` | `INT NOT NULL` | 1, 2 (revise), … |
| `source` | `TEXT` | `writer` or `human_edit` |
| `markdown` | `TEXT` | |
| `word_count` | `INT` | |
| `reviewer_verdict` | `TEXT` | `approve` `revise` `reject` `null (not yet reviewed)` |
| `reviewer_notes` | `JSONB` | |
| `brand_voice_score` | `NUMERIC(4,3)` | |
| `grounding_score` | `NUMERIC(4,3)` | |
| `keyword_density` | `NUMERIC(5,4)` | measured against the target angle keyword |
| `created_at` | `TIMESTAMPTZ DEFAULT now()` | |

Indexes: `(brief_id, version DESC)`, `(tenant_id, created_at DESC)`.

---

## `aca_publications`

Publisher output + feedback measurements.

| Column | Type | Notes |
|---|---|---|
| `id` | `SERIAL PK` | |
| `draft_id` | `INT FK aca_drafts.id ON DELETE RESTRICT` | |
| `tenant_id` | `INT FK` | |
| `wp_post_id` | `INT` | as returned by WordPress |
| `permalink` | `TEXT` | |
| `scheduled_at` | `TIMESTAMPTZ NOT NULL` | |
| `published_at` | `TIMESTAMPTZ` | after the WP schedule fires |
| `baseline_score` | `NUMERIC(5,2)` | aggregated over covered prompts |
| `lift_7d` | `NUMERIC(6,2)` | |
| `lift_30d` | `NUMERIC(6,2)` | |
| `covered_prompt_ids` | `JSONB` | for re-audit scoping |
| `created_at` | `TIMESTAMPTZ DEFAULT now()` | |

Indexes: `(tenant_id, published_at DESC)`, partial `(scheduled_at) WHERE published_at IS NULL`.

---

## `aca_agent_traces`

Per-hop trace. Mirrored to Langfuse + OTel; kept in Postgres for joins.

| Column | Type | Notes |
|---|---|---|
| `id` | `BIGSERIAL PK` | |
| `job_id` | `UUID FK aca_jobs.id ON DELETE CASCADE` | |
| `tenant_id` | `INT FK` | |
| `agent` | `TEXT` | analyzer / brief / writer / reviewer / publisher |
| `input_ref` | `TEXT` | e.g., `gap:123` |
| `output_ref` | `TEXT` | |
| `tokens_in` | `INT` | |
| `tokens_out` | `INT` | |
| `cost_usd` | `NUMERIC(10,6)` | |
| `latency_ms` | `INT` | |
| `verdict` | `TEXT` | for reviewer |
| `error` | `TEXT` | on failure |
| `warnings` | `JSONB` | |
| `created_at` | `TIMESTAMPTZ DEFAULT now()` | |

Indexes: `(job_id, created_at)`, `(tenant_id, created_at DESC)`, `(agent, created_at DESC)`.

**Retention:** 180 days. Aggregate to daily rollups after.

---

## Cross-cutting notes

**Timezones.** Everything is UTC (`TIMESTAMPTZ`). Presentation converts to the tenant's timezone.

**JSONB validation.** SQLAlchemy hybrid properties return pydantic models. Writes always go through pydantic, so malformed structures never persist.

**Row-level security.** Not using Postgres RLS in v0.1. `api/deps.get_current_tenant()` returns a tenant; every query is scoped via `.filter(Model.tenant_id == tenant.id)`. An `import-linter` rule flags queries that don't include that filter.

**Migrations.** Alembic. Auto-generated migrations reviewed by a second engineer before merge.

---

## Sample analytical queries

**Cost by agent this month:**
```sql
SELECT agent, SUM(cost_usd) AS spend, SUM(tokens_out) AS tokens
FROM aca_agent_traces
WHERE created_at >= date_trunc('month', now())
GROUP BY 1
ORDER BY spend DESC;
```

**Lift attribution by editorial angle:**
```sql
SELECT g.target_angle,
       AVG(p.lift_30d) AS avg_lift_30d,
       COUNT(*)        AS n
FROM aca_publications p
JOIN aca_drafts d ON d.id = p.draft_id
JOIN aca_briefs b ON b.id = d.brief_id
JOIN aca_gaps   g ON g.job_id = b.job_id
WHERE p.lift_30d IS NOT NULL
GROUP BY 1
HAVING COUNT(*) >= 3
ORDER BY avg_lift_30d DESC;
```

**Most-cited competitor domains, ranked by how often we build content against them:**
```sql
SELECT src->>'domain' AS domain,
       COUNT(*) AS times_cited
FROM   aca_gaps g,
       jsonb_array_elements(g.cited_sources) AS src
GROUP  BY 1
ORDER  BY 2 DESC
LIMIT  20;
```

**Human bottleneck:**
```sql
SELECT tenant_id, COUNT(*) AS stuck
FROM aca_jobs
WHERE status IN ('awaiting_human','needs_human')
  AND created_at < now() - interval '48 hours'
GROUP BY 1
ORDER BY stuck DESC;
```
