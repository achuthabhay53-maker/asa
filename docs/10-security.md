# 10 · Security

## Threat model in one line

ACA runs LLM-generated content against a real WordPress site using a real Google Ads customer's OAuth token. The two loud failure modes are **credential leakage** (someone steals a tenant's WP or Ads creds from us) and **content escape** (LLM-hallucinated content gets published without human review). Everything below is designed against those.

---

## Auth

### Users (browser)
- Same httpOnly JWT cookie as AImpact. ACA does not issue its own.
- `SameSite=Lax`, `Secure` (prod), path=`/`, exp = 14d, rotating on activity.
- Cookies decoded in `security/auth.py::verify_session_cookie`. The user resolves to an `aimpact_user_id`, which resolves to an ACA `Tenant` (or `403`).

### Server-to-server (bots, scripts)
- `Authorization: Bearer at_...`.
- Token format: `at_` + 32 URL-safe random chars.
- Stored: bcrypt hash + 8-char plaintext prefix (indexed, unique).
- Scopes: `read`, `write`. A `read`-only token can list jobs but cannot approve drafts.
- Rotation: tokens are revocable individually. New token = new row; old token continues working until explicitly revoked.

### Auth precedence
Cookie wins. Bearer used only if no cookie. Both present + valid = cookie's user identity is used; the bearer is ignored for identity but still verified so brute-forcing via header is not a bypass.

---

## Secrets

### At-rest encryption
- **WordPress application passwords** and **Google Ads refresh tokens** are envelope-encrypted with AWS KMS (`alias/aca-secrets`) before insertion into `aca_tenants`. The DEK is per-tenant, wrapped with the CMK.
- `security/secrets.encrypt()` / `.decrypt()` are the only entrypoints. Application code never sees ciphertext; it either passes plaintext in (encrypt path) or receives plaintext out (decrypt path, only in the immediate scope of a WP or Ads call).

### Never-logged rules
- Custom log formatter in `observability/telemetry.py` filters any field named `password`, `secret`, `token`, `app_password`, `refresh_token`, `access_token`, `authorization`.
- Traces to Langfuse strip system-prompt sections that include the brand style guide only when it contains a `[[SECRET]]` marker — style guides are otherwise considered non-sensitive.

### Runtime secrets
- Anthropic, Google Ads developer token, Langfuse keys → AWS Secrets Manager, injected as env vars at App Runner start.
- Local dev uses `.env` and never checks it in.

---

## Multi-tenant isolation

Enforced at four layers, so any single mistake still fails safe.

1. **Auth resolves to a tenant.** Every request has exactly one `Tenant` object in scope. No global "impersonation" helper.
2. **Every ORM read is filtered.** `Model.tenant_id == tenant.id` is required in every query. An `import-linter` rule scans the codebase to ensure `select(Model).where(...)` calls that touch tenant-scoped tables include a `tenant_id` predicate.
3. **Row-level `tenant_id`.** Denormalized onto every table so a missing filter cannot silently succeed with wrong data — it returns rows for other tenants, which unit tests catch via a "cross-tenant leak" harness.
4. **Credentials never cross tenants.** WP secrets are loaded from `Tenant.wp_credentials_ct` at call time, decrypted for the length of the request, and dropped. No process-wide cache keyed by anything but `tenant_id`.

---

## Content-escape prevention

- **Mandatory human gate.** No draft reaches WordPress without a `POST /aca/drafts/{id}/approve`. This is a product requirement in v0.1 and it is not overridable via config.
- **Publisher requires an `aca_publications` row that already carries `wp_post_id=null` and `published_at=null`.** Trying to re-publish an already-published draft is a `409`.
- **`status=future` only.** `Publisher.publish()` refuses to set `status=publish`. Even if the tenant asked for immediate publish, we schedule 30 seconds out to preserve a rollback window.
- **Content Studio has a "Cancel scheduled post" button** that hits WP `DELETE /wp/v2/posts/{id}?force=true` and marks the ACA publication as `canceled`.
- **Reviewer verdict `reject` short-circuits Writer**, so we don't burn tokens producing worse revisions of a fundamentally wrong draft.

---

## Input validation

- Every request body is a pydantic model. Unknown fields are rejected (extra=`forbid`) on user-facing endpoints; extra fields are ignored on outbound HTTP (extra=`ignore`) to survive vendor changes.
- Schedule times are validated against `now() + 60s` at minimum.
- Slugs are re-generated server-side even if the human edits them — never trust a caller-supplied slug.

---

## Rate limiting

- Per-tenant: 60 req/min against `/aca/*`.
- Per-IP anonymous: 20 req/min against public routes (currently only `/aca/health`).
- Job starts: `ACA_MAX_JOBS_PER_DAY` per tenant, default 20.
- LLM cost: `ACA_DAILY_USD_CAP_PER_TENANT`, default $25/day. Enforced *before* the LLM call by summing `aca_agent_traces.cost_usd` for the current UTC day.

Backpressure vs. AImpact: before enqueuing a job, ACA calls `GET /me/audit-budget`. If AImpact says the tenant's budget wouldn't cover the two feedback re-audits over the next 30 days, the job is queued (`status=pending`) and only picked up when budget clears.

---

## Egress hygiene

- Every outbound call goes through the tool client in `tools/`. Nothing in `agents/` or `orchestrator/` opens sockets directly.
- Tool clients set `timeout` on every request. Default 30 s.
- Retry policy: `tenacity`, expo backoff, max 3 attempts on 5xx and 429.
- URL allowlists — the WordPress client refuses to POST to a URL whose host doesn't match the tenant's stored `wp_site_url`. Prevents credential exfiltration via a swapped URL.

---

## Prompt-injection posture

The Analyzer reads competitor citation snippets from AImpact. Those snippets come from third-party AI outputs and can contain injected instructions. Mitigations:

- **Snippets are wrapped in a fixed delimiter block** in the prompt (`<<<COMPETITOR_SNIPPET_START>>> … <<<COMPETITOR_SNIPPET_END>>>`) with an explicit instruction to treat the contents as **data**, not instructions.
- **Structured output only.** Analyzer's output schema is a pydantic model, so free-form model text that doesn't fit the schema is rejected.
- **No tool access from Analyzer.** It cannot call the AImpact API, cannot spawn shell commands, cannot fetch URLs. Its only side effect is returning a JSON blob.
- **Reviewer runs the same rubric regardless of what's in the draft.** If Writer somehow produces an "ignore previous instructions" section, Reviewer flags it (rubric explicitly checks for prompt-leakage markers).

We do not consider ACA fully hardened against sophisticated prompt injection; we consider it low-risk because the outputs pass a human before any external action.

---

## Audit trail

- **`aca_agent_traces`** — every LLM hop, every tool call. 180-day retention.
- **`aca_drafts`** — every version, including human edits with `source="human_edit"`.
- **`aca_publications`** — schedule time, publish time, `wp_post_id`, `covered_prompt_ids`.
- **CloudTrail** — every KMS decrypt.
- **AImpact `usage_events`** — cross-system record of `content_published`, etc.

An incident review can reconstruct exactly: which prompts triggered a job, which sources the Analyzer read, which editorial angle it picked, which model produced which draft version, which human approved it, when it went live, and which prompts it was supposed to move.

---

## Incident response cheatsheet

**Suspected WordPress credential leak:**
1. Rotate the tenant's WP application password (user-controlled).
2. `DELETE /aca/tenants/me/wordpress` to zero our ciphertext.
3. Grep `aca_agent_traces` for the tenant + agent=`publisher` in the incident window.
4. Grep CloudTrail for KMS decrypt calls on the tenant's DEK.

**Suspected prompt injection:**
1. Pull `aca_agent_traces` for the job.
2. Pull the Langfuse trace — full LLM I/O.
3. Diff against the rubric — Reviewer should have caught it. If not, tighten the rubric.

**Sudden cost spike:**
1. CloudWatch cost dashboard, group by tenant.
2. `aca_agent_traces` by agent for the offending tenant.
3. If a specific brief is looping the Reviewer forever (rare, but possible), kill the job and flag Reviewer's decision boundary as too strict.

---

## Known limits (v0.1)

- No SOC 2. Design informed by SOC 2 conventions (least privilege, encryption in transit/at rest, audit trail) but no attestation.
- No VPC private endpoints for Neon / Anthropic / Google. All egress is public HTTPS.
- No formal per-tenant DPA in v0.1. Content Studio's T&C is the placeholder until legal produces one.
