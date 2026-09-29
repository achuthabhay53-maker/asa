# 09 · Deployment

Target: AWS App Runner (source-code deploy, no Docker), Neon Postgres, AWS Amplify for the frontend (unchanged, ACA piggybacks). Same cloud footprint as AImpact so we don't add ops surface.

The `apprunner.yaml` at the repo root tells App Runner to build a Python 3.11 runtime, `pip install -r requirements.txt`, and run `uvicorn aca.main:app`. No Dockerfile, no ECR.

---

## Prerequisites

- AWS account with permissions to create App Runner services, IAM roles, KMS keys, Secrets Manager entries.
- Neon project with one database dedicated to ACA (`aca_db`).
- Anthropic API key (prod key, not shared with dev).
- Google Ads developer token approved by Google (2–4 week lead time).
- CloudFront distribution routing `/api/aca/*` to the new App Runner service.
- Access to the AImpact backend to add the three additive endpoints described in [08-integration-aimpact.md](08-integration-aimpact.md).

---

## First-time setup (one hour)

### 1. Neon
```bash
# In the Neon console: create a new database in the existing project
#   name: aca_prod
# Copy the pooled connection string.
```

### 2. AWS resources
```bash
# KMS key
aws kms create-key --description "ACA envelope encryption"
aws kms create-alias --alias-name alias/aca-secrets --target-key-id <keyId>

# Secrets Manager
aws secretsmanager create-secret --name aca/prod/anthropic --secret-string '{"api_key":"sk-ant-..."}'

# IAM role for App Runner
# Trust: build.apprunner.amazonaws.com + tasks.apprunner.amazonaws.com
# Policies: SecretsManagerReadWrite (scoped to aca/prod/*), kms:Encrypt/Decrypt on the ACA key,
#           logs:CreateLogStream, logs:PutLogEvents.
```

### 3. App Runner service
```bash
# From the AWS console: create service from Source code (GitHub).
# Repository: this repo · Branch: main · Config: use apprunner.yaml
# Port 8002 (already set in apprunner.yaml).
# Environment variables: everything in .env.example, values from Secrets Manager where sensitive.
# Auto-deploy on push to main.
```

### 4. CloudFront path routing
Add a behavior:
- Path pattern: `/api/aca/*`
- Origin: the App Runner service's default URL
- Origin path: (empty; ACA's `StripApiPrefixMiddleware` handles it)
- Allowed methods: GET, HEAD, OPTIONS, PUT, POST, PATCH, DELETE
- Cache policy: `Managed-CachingDisabled` (API responses)
- Origin request policy: `Managed-AllViewer`

### 5. First migration
```bash
DATABASE_URL="postgresql+psycopg://…/aca_prod" alembic upgrade head
```

### 6. Bootstrap the AImpact tenant
```sql
INSERT INTO aca_tenants (aimpact_user_id, api_token_hash, api_token_prefix, brand_summary, enabled)
VALUES (:aimpact_user_id, :bcrypt_hash, :first_8_chars, '...', true);
```
Store the plaintext `at_...` token in Secrets Manager or your password vault. It's never displayed again.

---

## Continuous deployment

`.github/workflows/deploy.yml`:

```yaml
name: deploy
on:
  push:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: mypy aca/
      - run: pytest -q

  deploy:
    needs: test
    runs-on: ubuntu-latest
    permissions: { id-token: write, contents: read }
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::<acct>:role/github-oidc-deploy
          aws-region: ap-south-1
      - name: Trigger App Runner deploy
        # App Runner is configured for auto-deploy on main; this is a manual belt-and-braces trigger.
        run: aws apprunner start-deployment --service-arn ${{ secrets.APPRUNNER_ARN }}
      - name: Migrate
        run: |
          pip install alembic psycopg[binary] SQLAlchemy pydantic-settings
          DATABASE_URL=${{ secrets.DATABASE_URL }} alembic upgrade head
```

Because App Runner uses source-code deploys, GitHub Actions doesn't build or push an image — App Runner pulls the source directly on the auto-deploy trigger. CI is only for tests + the DB migration.

Blue-green: App Runner does an automatic rolling deploy. If the health check fails the new revision, traffic stays on the old one.

---

## Environment variables

Full list in [`.env.example`](../.env.example). Grouped:

**Runtime**
- `ACA_ENV`, `ACA_PORT`, `ACA_LOG_LEVEL`

**DB**
- `DATABASE_URL`

**AImpact client**
- `AIMPACT_BASE_URL`
- `AIMPACT_API_TOKEN` (Secrets Manager)
- `AIMPACT_TIMEOUT_S`

**LLM (Claude only)**
- `ANTHROPIC_API_KEY` (Secrets Manager)
- `ANTHROPIC_MODEL_HAIKU`, `ANTHROPIC_MODEL_SONNET`

**AWS**
- `AWS_REGION`, `AWS_KMS_KEY_ID`

**Observability**
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
- `OTEL_EXPORTER_OTLP_ENDPOINT`

**Guard rails**
- `ACA_DAILY_USD_CAP_PER_TENANT`
- `ACA_JOB_TIMEOUT_S`
- `ACA_MAX_REVISE_LOOPS`
- `ACA_MAX_JOBS_PER_DAY`

---

## Health, readiness, and probes

- `GET /aca/health` — liveness (always 200 unless the process is unresponsive).
- `GET /aca/health/ready` — readiness (checks DB connection, Anthropic ping, AImpact ping). Used by CloudFront's origin health.

App Runner defaults: healthy after 3 consecutive 200s on `/aca/health`, unhealthy after 5 consecutive non-200s.

---

## Observability wiring

```
ACA container ─► OTLP HTTP ─► OTel collector (sidecar or shared)
                                  ├─► CloudWatch Logs   (structured JSON)
                                  ├─► CloudWatch Metrics
                                  └─► Langfuse           (LLM spans only)
```

Alarms:
- `ACA-CostBreach` — daily USD spend per tenant > 80% of cap.
- `ACA-JobFailureRate` — > 5% failed jobs in the trailing hour.
- `ACA-P95Latency` — job wall-clock p95 > 20 min.
- `ACA-HumanBacklog` — jobs in `awaiting_human` > 48 h.

Dashboards (CloudWatch + Langfuse):
- **Cost by agent** — daily spend per model per agent.
- **Lift by editorial angle** — 30-day lift rolled up by `aca_gaps.target_angle` cluster.
- **Lift by cited-source domain** — which competitor domains, when we build against them, actually move the needle.
- **Approval rate** — % of drafts approved without revision.
- **Time-to-review** — hours from `awaiting_human` to `approve`.

---

## Rollback

- **Code** — App Runner's "Rollback to previous version" button.
- **DB migration** — Alembic downgrade. Migrations are reviewed to be reversible where possible; irreversible ones (dropping columns) require a two-step deploy (deploy code that stops writing → deploy migration).
- **WordPress publish** — Content Studio's "Cancel scheduled post" button calls `DELETE /wp/v2/posts/{id}?force=true`.
- **Runaway spend** — flip `ACA_ENABLED_FOR_TENANT_IDS` to exclude the offender; jobs already running finish; new ones refused.

---

## Cost model (v0.2)

Per active tenant per month, assuming 20 published blogs:

| Line | Estimate |
|---|---|
| App Runner (shared, 1 instance) | $30 |
| Neon compute (shared) | $10 |
| Neon storage (~200 MB) | $1 |
| Anthropic (20 jobs × ~$0.12) | $2.40 |
| Langfuse | $5 (Team plan share) |
| SES + S3 + KMS + Secrets | $2 |
| **Total** | **~$50/tenant/month at PoC scale** |

Scales sub-linearly with tenant count until we cross ~20 concurrent jobs (App Runner autoscale kicks in).

---

## Runbooks (kept short; expand as we hit issues)

- **A tenant reports "my post didn't publish"** → Content Studio → jobs → filter by `status=failed_publish` → check the last `aca_agent_traces.error` on the publisher hop → most common cause is WP password rotated.
- **Costs spiking** → CloudWatch cost dashboard → group by tenant/agent → if Writer's tokens_out is anomalously high, check for a huge brief that overflowed length limits.
- **Claude 429s** → retry handles transient; if sustained, check the tenant's daily cap and request an Anthropic quota bump if legitimate.
- **AImpact endpoint returns 500** → ACA marks the job `failed` with `error="aimpact.5xx"`; retry manually after AImpact recovers.
