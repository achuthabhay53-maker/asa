# 08 · Integration with AImpact

This doc is for the AImpact side of the contract. What ACA reads, what ACA triggers, and the small additive changes AImpact must ship so ACA can plug in cleanly.

Design tenets:
- **ACA is a normal API client.** Same auth surface AImpact already exposes to any external caller.
- **AImpact is authoritative.** Its Postgres, its quota, its billing. ACA never bypasses them.
- **Additions are small and backward compatible.** No route renames, no schema breaks. Everything new is a fresh endpoint or a fresh `usage_event` type.

---

## Endpoints ACA calls (existing)

All requests carry `Authorization: Bearer at_...`. All responses are JSON.

| Method | Path | ACA uses it for |
|---|---|---|
| `GET` | `/me` | Verify token, discover `user_id` and plan tier. |
| `GET` | `/projects` | Enumerate projects for a tenant. |
| `GET` | `/campaigns?project_id=…` | List campaigns per project. |
| `GET` | `/campaigns/{id}/audit-results?threshold=&include=sources` | The primary input: low-scoring prompts with competitor citations. |
| `GET` | `/campaigns/{id}/market-share` | Optional context for Analyzer prompting. |
| `GET` | `/campaigns/{id}/competitor-visibility` | Same. |
| `GET` | `/audit-scores/trend?campaign_id=…&from=&to=` | Baseline computation for lift attribution. |
| `POST` | `/audit-campaign/start` | Trigger re-audit at day 7 and day 30. |
| `GET` | `/audit-jobs/{id}` | Poll async audit job until `status=completed`. |
| `GET` | `/me/audit-budget` | Backpressure — do not enqueue a job that will run the tenant out of audit budget. |

If any of these change shape, the contract test in `tests/integration/test_aimpact_contract.py` fails before ACA hits prod.

---

## Additive endpoints AImpact must ship

Three of them. All are thin wrappers over existing data or existing capabilities.

### 1. `GET /campaigns/{id}/content-gaps`
**Purpose:** Return prompts ranked by `(impact × opportunity)` so ACA doesn't have to re-implement AImpact's ranking.

**Query params**
- `threshold` — max avg score to consider (default 60).
- `limit` — top N (default 20).

**Response**
```json
{
  "campaign_id": 4123,
  "generated_at": "...",
  "items": [
    {
      "prompt_id": "cp_9182",
      "text": "...",
      "avg_score": 42,
      "per_model": {"chatgpt":38,"claude":50,"gemini":41,"google_ai":39},
      "impact": 0.82,       // 0..1, based on search volume / prior traffic
      "opportunity": 0.71,  // 0..1, easier to move if competitor citation is weak
      "sources": [ {"model":"chatgpt","competitor":"acme.io","excerpt":"..."} ]
    }
  ]
}
```

**Implementation sketch** (in `backend/server.py`):
```python
@router.get("/campaigns/{campaign_id}/content-gaps")
def content_gaps(campaign_id: int, threshold: int = 60, limit: int = 20,
                 user = Depends(get_current_user), db = Depends(get_db)):
    _require_role(db, user, campaign_id, {"owner","editor","viewer"})
    prompts = db.query(CampaignPrompt).filter(...).all()
    scored = [_rank(p) for p in prompts if p.latest_score < threshold]
    scored.sort(key=lambda x: -(x["impact"]*x["opportunity"]))
    return {"items": scored[:limit], ...}
```

### 2. `POST /webhooks/audit-completed`  →  ACA
**Purpose:** Push signal instead of pulling. ACA needs to know when a re-audit for the feedback loop finishes so it can attribute lift promptly.

**Direction:** AImpact **calls** ACA's `POST /aca/webhooks/aimpact/audit-completed` endpoint. HMAC-signed with `AIMPACT_WEBHOOK_SECRET` in the `X-AImpact-Signature` header.

**Body**
```json
{
  "aimpact_audit_job_id": "aj_xyz",
  "campaign_id": 4123,
  "completed_at": "..."
}
```

Retries: exponential backoff, 5 attempts, dead-letter to CloudWatch after that.
Optional — ACA will fall back to polling `GET /audit-jobs/{id}` if the webhook doesn't arrive within 5 minutes.

### 3. New `usage_event` types
`usage_events` is an append-only table already in AImpact. Three new type strings:

- `content_brief_created` — one per successful `Brief` agent output.
- `content_published` — one per successful WordPress publish.
- `content_lift_measured` — one per completed 30-day feedback loop.

Payload per event:
```json
{
  "type": "content_published",
  "user_id": 123,
  "created_at": "...",
  "data": {
    "campaign_id": 4123,
    "aca_job_id": "9c0e...",
    "aca_publication_id": 887,
    "wp_post_id": 42193,
    "covered_prompt_ids": ["cp_1","cp_2"]
  }
}
```

No schema change — the existing table already stores `data JSONB`.

---

## What ACA does NOT touch in AImpact

- **`AuditScore`, `CampaignPrompt`, `Campaign`, `Project`, `User`** — never written from ACA.
- **JWT session cookie** — never issued from ACA; only *received* from browsers via CloudFront.
- **Razorpay / billing** — off-limits. Content pack pricing (if we launch one) is a new plan tier in AImpact's existing Plan table.
- **AImpact's Postgres directly** — no dblink, no shared connection pool.

---

## Handling AImpact evolution

Two forces to watch:

1. **AImpact adds a new field to `audit-results`.**
   Safe. Pydantic on the ACA side ignores unknown fields by default.

2. **AImpact renames a field or changes types.**
   Breaking. Contract test fails on the next CI run against staging. Bump the ACA client's version, patch the mapping, deploy.

Cadence: AImpact ships changes ~2x/week. ACA runs its contract test suite in staging nightly at 03:00 IST. Failures page the ACA on-call before customers see impact.

---

## Sequencing for the AImpact team

If AImpact ships changes on a normal 2-week cycle:

| Sprint | AImpact task | Unblocks |
|---|---|---|
| S1 | `GET /campaigns/{id}/content-gaps` | ACA Analyzer's real inputs |
| S1 | New `usage_event` types (documentation + validation) | ACA analytics |
| S2 | Outbound webhook `audit-completed` + `AIMPACT_WEBHOOK_SECRET` | ACA feedback loop without polling |
| S3 | (optional) `GET /me/audit-budget` endpoint | ACA backpressure |
| S4 | Content pack plan tier in the Plan table (only if we ship a paid add-on) | Billing |

Everything above is opt-in from AImpact's perspective — nothing in the existing product breaks if none of it ships. ACA just runs in a slightly less efficient mode (polling instead of webhook, no backpressure).
