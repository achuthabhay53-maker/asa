# 07 · ACA API Reference

Base URL (prod): `https://app.example.com/api/aca`
Base URL (dev):  `http://localhost:8002/aca`

**Auth on every endpoint** (except `/aca/health`):
- **Cookie** — the AImpact httpOnly JWT (same-origin browser calls from `/content`).
- **Bearer** — `Authorization: Bearer at_...` (server-to-server, scoped API token).

Errors follow FastAPI's default:
```json
{ "detail": "human-readable message" }
```

---

## `GET /aca/health`
Unauthenticated liveness check.
**200** `{"status":"ok"}`

---

## Jobs

### `POST /aca/jobs`
Start a workflow for a campaign.

**Body**
```json
{
  "campaign_id": 4123,
  "threshold": 60,               // optional, default from tenant settings
  "prompt_ids": ["cp_1","cp_2"]  // optional, restricts scope
}
```

**200**
```json
{
  "id": "9c0e...uuid",
  "status": "pending",
  "created_at": "2026-09-29T04:12:01Z"
}
```

**403** if the tenant's daily USD cap is already exhausted.

### `GET /aca/jobs/{id}`
Poll job status.

**200**
```json
{
  "id": "9c0e...",
  "campaign_id": 4123,
  "status": "running",
  "current_agent": "writer",
  "revise_count": 0,
  "created_at": "...",
  "started_at": "...",
  "ended_at": null
}
```

### `GET /aca/jobs/{id}/artifacts`
Return the full trail — gap, brief, current draft, verdict.

**200**
```json
{
  "job": {...},
  "gap": {
    "prompt_id": "cp_9182",
    "cited_sources": [...],
    "themes": [...],
    "target_angle": "...",
    "hypothesis": "..."
  },
  "brief": {...},
  "drafts": [
    {"version":1, "reviewer_verdict":"revise", ...},
    {"version":2, "reviewer_verdict":"approve", ...}
  ],
  "publication": null
}
```

### `POST /aca/jobs/{id}/resume`
Resume a `timed_out` or `failed` job from its last checkpoint.

**200** returns the fresh job row.

### `GET /aca/jobs`
List the current tenant's jobs.

**Query params:** `status`, `campaign_id`, `limit` (default 25), `cursor`.

**200**
```json
{
  "items": [ {...}, {...} ],
  "next_cursor": "..."
}
```

---

## Drafts

### `POST /aca/drafts/{id}/edit`
Save inline human edits. Creates a new draft version with `source="human_edit"`.

**Body**
```json
{ "markdown": "...", "note": "tightened intro" }
```

**200** returns the new draft version row.

### `POST /aca/drafts/{id}/approve`
Approve and schedule publication.

**Body**
```json
{ "schedule_at": "2026-10-05T09:00:00+05:30" }
```

**200**
```json
{
  "publication_id": 887,
  "wp_post_id": 42193,
  "permalink": "https://tenant.example.com/blog/hipaa-compliant-patient-intake",
  "scheduled_at": "2026-10-05T03:30:00Z"
}
```

**409** if `schedule_at` is in the past.

### `POST /aca/drafts/{id}/reject`
Kill this line of work. Job moves to `status=completed` with no publication.

**Body**
```json
{ "reason": "off-topic; will pick a different angle next run" }
```

---

## Publications

### `GET /aca/publications`
List the tenant's publications.

**Query params:** `campaign_id`, `since`, `until`, `has_lift` (bool).

**200**
```json
{
  "items": [
    {
      "id": 887,
      "wp_post_id": 42193,
      "permalink": "...",
      "scheduled_at": "...",
      "published_at": "...",
      "baseline_score": 42.0,
      "lift_7d": 5.5,
      "lift_30d": 11.2,
      "covered_prompt_ids": ["cp_1","cp_2"]
    }
  ]
}
```

### `POST /aca/publications/{id}/measure`
Trigger an on-demand re-audit (e.g., for a manual comparison).

**200**
```json
{ "aimpact_audit_job_id": "aj_xyz", "eta_seconds": 90 }
```

---

## Tenant settings

### `GET /aca/tenants/me/settings`
```json
{
  "wp_linked": true,
  "wp_site_url": "https://tenant.example.com",
  "wp_credentials_valid": true,
  "daily_usd_cap": 25.0,
  "brand_summary": "...",
  "style_guide_uploaded_at": "2026-08-11T00:00:00Z"
}
```

### `POST /aca/tenants/me/wordpress`
Link a WordPress site with an Application Password. Password is KMS-encrypted before insert; the plaintext is not stored, not logged, not echoed back.

**Body**
```json
{
  "site_url": "https://tenant.example.com",
  "username": "aca-bot",
  "app_password": "xxxx xxxx xxxx xxxx xxxx xxxx"
}
```

**200** `{ "wp_linked": true }`
**400** if the credentials fail a live `GET /wp/v2/users/me` probe.

### `DELETE /aca/tenants/me/wordpress`
Unlink and zero the ciphertext.

### `POST /aca/tenants/me/brand`
Update `brand_summary`, upload `style_guide.md`, edit `taxonomy_json`.

**Body**
```json
{
  "brand_summary": "...",
  "style_guide_md": "# ...",
  "taxonomy_json": {"categories":[...], "url_pattern":"..."}
}
```

---

## Admin (super-viewer only)

Available when the caller's cookie session resolves to an AImpact user with `is_super_viewer=true` (mirrors AImpact's model). All read-only.

- `GET /aca/admin/tenants`
- `GET /aca/admin/tenants/{id}`
- `GET /aca/admin/jobs?status=failed&since=...`
- `GET /aca/admin/costs?group_by=tenant|day|agent`

---

## Webhook receiver

### `POST /aca/webhooks/aimpact/audit-completed`
Optional inbound webhook so ACA doesn't have to poll `GET /audit-jobs/{id}` for the feedback loop. HMAC-signed by AImpact using `AIMPACT_WEBHOOK_SECRET`.

**Body**
```json
{
  "aimpact_audit_job_id": "aj_xyz",
  "campaign_id": 4123,
  "completed_at": "2026-10-12T04:00:00Z"
}
```

**200** always (idempotent — dedup by `aimpact_audit_job_id`).

---

## Error taxonomy

| Status | When |
|---|---|
| 400 | Malformed body or invalid enum value |
| 401 | No auth header/cookie, or invalid token |
| 402 | Daily USD cap exceeded — retry tomorrow or raise the cap |
| 403 | Tenant not enabled for ACA (`aca_tenants.enabled=false`) |
| 404 | Resource does not exist for this tenant |
| 409 | State conflict (e.g., approving a rejected draft) |
| 429 | Rate limit — per-tenant (bucket: 60 req/min) |
| 500 | Uncaught error — logs the traceback, response body has an `error_id` for support |

---

## Rate limits

- Per-tenant: 60 req/min per token.
- Per-IP anonymous: 20 req/min.
- Job starts: 20/day/tenant by default (config `ACA_MAX_JOBS_PER_DAY`).

---

## OpenAPI

FastAPI auto-emits an OpenAPI schema at `GET /aca/openapi.json` and a Swagger UI at `GET /aca/docs` (gated to the `dev` env by default).
