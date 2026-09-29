# 05 · Agents — Deep Dive

Five agents, each doing one thing well. This doc pairs each agent with its prompt strategy, IO contract, cost target, failure modes, and how to test it.

Design tenets (apply to every agent):
- **Typed IO** — pydantic on both ends. If the LLM returns something that doesn't parse, the orchestrator retries once with a "your last output failed validation because …" corrective prompt, then fails the hop.
- **No shared state** — an agent never reads or writes `WorkflowState` fields belonging to another agent. It receives `state` and returns a new `state` with only its own field mutated.
- **Prompt cache first** — brand-voice rubric, site style guide, and per-tenant taxonomy are always in the cached prefix.
- **Fail fast, resume cleanly** — every agent's output is idempotent; running it twice on the same input produces the same output (or a strictly-equivalent one).

---

## 1 · Source Analyzer

**Model:** Claude Haiku 4.5.
**Job:** Read the low-scoring prompt, model responses, and citations; then, within this same agent, fetch the cited public pages and verify candidate claims/themes against their actual text. Emit a `ContentGap` that distinguishes candidate themes from quote-verified themes. The target site's content coverage remains unknown unless a site inventory is supplied.

**Input:** `state.prompts: list[PromptWithSources]` (from `tools/aimpact.get_audit_results(include=sources)`).

**Output:** `state.gap: ContentGap` for the current single-prompt workflow.

### Two-stage source analysis

1. **Analyze AImpact evidence.** Produce candidate source contributions, one or more factual claims per source to check, candidate themes, and a candidate angle.
2. **Verify cited pages.** The Analyzer calls `tools/source_fetcher.py` for bounded public HTTP(S) page fetches. It follows only a limited number of redirects, checks each destination resolves publicly, limits size and concurrency, and extracts readable HTML/plain-text content. The verifier treats page text as untrusted input. Stage 1 maps source-specific claims to candidate themes; a mapped theme becomes verified only when a claim check has a supported verdict and its evidence quote is found in that same source's fetched page text.

`ContentGap.themes` contains only verified themes; `candidate_themes` retains Stage 1 suggestions. `theme_evidence` maps each verified theme to a fetched URL and quote. Each source reports its original URL, resolved domain/URL when fetched, candidate and verified competitor labels, verification status, and claim evidence. Unavailable pages and unsupported claims stay explicitly unverified. The final angle is assembled only from verified themes; `candidate_target_angle` retains the Stage 1 suggestion. A verified citation proves what that page contains, not that the target company's site is missing the topic.

**Prompt shape** (batched, ~5 prompts per call, CACHED prefix in **bold**):

```
System (CACHED):
You are a GEO content strategist for {tenant.brand}.
Brand summary: {tenant.brand_summary}
Our site taxonomy (URLs, categories, pillar pages): {tenant.taxonomy_json}
Return a strict JSON array matching this schema: {CONTENT_GAP_SCHEMA}.

User (per batch):
For each user query below, read the four model answers and the URLs each model cited.
Identify:
  1. Candidate THEMES present in the supplied answers/citations
  2. One to three factual claims per source to check against its page
  3. Competitor identities only when the supplied evidence supports them
  4. A candidate editorial angle, without asserting target-site content is missing
  5. An observed AI visibility hypothesis, not a claim about why the model selected a source

Prompts:
1. {prompt.text}
   ChatGPT answered: {answer_chatgpt} — cited: {sources_chatgpt}
   Claude answered:  {answer_claude}  — cited: {sources_claude}
   Gemini answered:  {answer_gemini}  — cited: {sources_gemini}
   Google AI:        {answer_google}  — cited: {sources_google}

2. ...
```

**Output schema:**
```json
{
  "prompt_id": "cp_9182",
  "prompt_text": "...",
  "cited_sources": [
    {"model":"chatgpt", "url":"...", "domain":"...", "competitor":"...", "excerpt":"...",
      "candidate_competitor":"Acme", "contribution":"Supports the answer's comparison of BAA scope.",
      "verification_status":"verified", "verified_url":"https://acme.io/hipaa-checklist",
      "verification_evidence":["Exact text from the fetched page that supports the claim."]}
  ],
  "candidate_themes": ["BAA scope", "audit-log requirements"],
  "themes": ["BAA scope"],
  "theme_evidence": [{"theme":"BAA scope", "source_id":"chatgpt:0", "source_url":"https://acme.io/hipaa-checklist", "evidence_quote":"Exact text from the fetched page."}],
  "theme_assessments": [{"theme":"BAA scope", "status":"verified", "candidate_source_ids":["chatgpt:0"], "candidate_cited_by":["chatgpt"], "candidate_competitors":["Acme"], "source_ids":["chatgpt:0"], "cited_by":["chatgpt"], "competitors":["Acme"], "target_site_coverage":"unknown", "content_gap_status":"not_established", "opportunity":"Investigate whether the target company already has content explaining BAA scope."}],
  "verification_status": "partially_verified",
  "verification_summary": "Verified 1 of 2 candidate themes against fetched page text. Target-site coverage remains unknown.",
  "candidate_target_angle": "Stage 1 editorial angle suggestion.",
  "target_angle": "Explore a practical guide to BAA scope and audit-log checks for HIPAA intake; target-site coverage has not been assessed.",
  "target_site_coverage":"unknown",
  "content_gap_status":"not_established",
  "opportunity":"Investigate whether the target company already covers BAA scope. Compare the site inventory before establishing a content gap.",
  "hypothesis": "Fetched cited pages support BAA scope and audit-log themes. The target company's content inventory was not supplied, so a gap on its site is not established."
}
```

Each analyzed source includes individual `verified_claims`, a `verification_status`, and `competitor_verification_status`. Theme assessments identify supporting sources/models and exact evidence quotes. `target_site_coverage` remains `unknown` and `content_gap_status` remains `not_established` until a first-party site inventory is supplied and compared. The `opportunity` asks whether relevant target-site content already exists; it does not assert a gap.

**Cost target:** ~$0.004 per 5-prompt batch (Haiku). ~30 batches for 150 prompts = ~$0.12 per campaign.

**Failure modes:**
- Empty citations (prompt scored low because we don't appear at all, not because a competitor does) → themes derived from the model answers alone; hypothesis notes the absence.
- Contradictory citations across models → the Analyzer picks the most-cited domain across the 4 models as the primary comparator and flags the rest.
- JSON parse failure → retry once with a corrective prompt, then log the error to `aca_agent_traces` and fail the job.

**How to test:** unit test with 3 fixture prompts + recorded Haiku output. Assert `target_angle` is present and `themes` has at least 2 entries.

---

## 2 · Brief

**Model:** Claude Sonnet.
**Job:** Turn the gap and editorial angle into a structured content brief the Writer can expand mechanically.

**Input:** `{prompt, gap, brand_voice_ref, tenant.taxonomy_json}`.

**Output:** `state.brief: Brief`.

**Prompt shape** (cached prefix in **bold**):

```
System (CACHED):
You are a senior content strategist writing for {tenant.brand}.
Brand voice guide: {tenant.style_guide_md}
Taxonomy: {tenant.taxonomy_json}
Output schema: {BRIEF_JSON_SCHEMA}

User (per-job):
Target user query: {prompt}
Editorial angle: {gap.target_angle}
Themes competitors cover: {gap.themes}
Cited competitor sources (for reference, NOT to copy): {gap.cited_sources}

Produce a brief that would out-cover the cited sources on the target angle.
Meta description: 130–155 chars. Slug: kebab-case, ≤ 60 chars.
4–6 FAQs. 2–5 internal-link candidates from the taxonomy.
```

**Cost target:** ~$0.030 per brief (Sonnet), ~80% prompt-cache hit after the first job of the day for a tenant.

**Style rules encoded in the schema:**
- Title: 45–65 chars.
- Slug: kebab-case, ≤ 60 chars.
- Meta: 130–155 chars.
- Outline: 4–8 H2s, each with a 1-line intent hint.
- FAQs: 4–6 Q/A pairs.
- Internal links: 2–5 candidates from `tenant.taxonomy_json`.

**Failure modes:**
- Missing FAQs or < 4 H2s → orchestrator retries once with a corrective prompt.
- Meta > 155 chars → auto-truncate at the last full word before 155.

---

## 3 · Writer

**Model:** Claude Sonnet.
**Job:** Expand the brief into a Markdown draft.

**Input:** `state.brief` (+ `reviewer_notes` and `gap.cited_sources` on revise loops).

**Output:** `state.draft: Draft`.

**Structure enforced by the prompt:**
- Intro paragraph: 60–100 words stating the question and previewing the answer.
- Body: one H2 per `brief.outline` entry, 150–300 words each.
- Practical section: bullet list or table where `outline[i].intent == "table"` or `"list"`.
- FAQ block: `## FAQs` heading + Q/A pairs.
- Trailing `<script type="application/ld+json">` block with `schema.org` `FAQPage` data generated from `brief.faqs`.
- Target length: 1,200–2,000 words.

**Revise-loop behavior:**
- Reviewer's `notes[]` are passed as a second user message: "Address these issues in v2 …".
- Version increments; the previous version is retained in `aca_drafts`.

**Cost target:** ~$0.055 (v1), ~$0.045 (v2 revise).

**Failure modes:**
- > 2,500 words → truncate at the last complete H2 section; add a warning to `aca_agent_traces.warnings`.
- Malformed JSON-LD → strip and regenerate from `brief.faqs` deterministically (server-side, no LLM).

---

## 4 · Reviewer

**Model:** Claude Sonnet with a rubric prompt.
**Job:** Score the draft against the brand voice, factual grounding (against the Analyzer's cited sources + tenant source docs), and structure.

**Input:** `{state.draft.markdown, state.brief, state.gap.cited_sources, brand_voice_rubric, tenant_sources}`.

**Output:** `state.verdict: Verdict`:
```json
{
  "decision": "approve | revise | reject",
  "notes": [
    {"quote":"exact span from draft", "issue":"unsupported|off-voice|density|structure",
     "suggestion":"..."}
  ],
  "brand_voice_score": 0.0..1.0,
  "grounding_score": 0.0..1.0,
  "keyword_density": 0.0..1.0
}
```

**Rubric axes:**
| Axis | Signal | Threshold |
|---|---|---|
| Factuality | Every non-obvious claim traces to `gap.cited_sources`, a tenant source doc, or well-known reference material | Flag if unsupported |
| Brand voice | Style-feature similarity vs. style guide | < 0.7 = revise |
| Density | Target-angle keyword occurrences per 100 words | Outside 0.5%–2% = revise |
| Structure | Intro, H2s, FAQs, JSON-LD present | Any missing = revise |

**Decisions:**
- All axes pass → `approve`.
- One or two axis failures with clear notes → `revise`.
- Systemic failure (grounding_score < 0.4) → `reject` (skip Writer loop, escalate to human).

**Cost target:** ~$0.035 per pass (Sonnet). Rubric is prompt-cached.

**Failure modes:**
- Reviewer approves everything → weekly sampling: 10% of approved drafts re-scored by a stricter human-tuned prompt. Rubric adjusted.
- Reviewer rejects everything → same sampling picks up over-strictness.

---

## 5 · Publisher

**Model:** none. Deterministic client.
**Job:** Push the approved draft to WordPress as a scheduled post.

**Input:** approved draft + `schedule_at` + tenant WP credentials.

**Output:** `aca_publications` row.

**Steps:**
1. Decrypt WP application password via `security/secrets.decrypt(tenant.wp_credentials_ct)`.
2. Convert Markdown to HTML (`mistune` with a custom renderer that inlines `<figure>` for images and preserves the trailing `<script type="application/ld+json">`).
3. Upload featured image if present in the brief metadata (`POST /wp-json/wp/v2/media`).
4. Create post:
   ```http
   POST /wp-json/wp/v2/posts
   {
     "title": brief.title,
     "slug":  brief.slug,
     "content": html,
     "excerpt": brief.meta,
     "status": "future",
     "date_gmt": schedule_at_utc,
     "categories": [tenant.default_category_id],
     "meta": {"aca_job_id": job.id}
   }
   ```
5. Persist returned `id`, `link`, `date` to `aca_publications`.

**Failure modes:**
- WP 401 → mark `tenant.wp_credentials_valid=false` and surface to Content Studio.
- WP 4xx on content → job goes `status=failed_publish`. Human retries.
- WP 5xx → retry 3× with exponential backoff.
- Network partition after WP 201 but before our DB write → the `aca_job_id` meta on the WP post lets us reconcile on retry (idempotency).

---

## Adding a new agent

1. New file `aca/agents/<name>.py` with the `run(state) -> state` signature.
2. Extend `WorkflowState` in `orchestrator/state.py` with the new field.
3. Add a node + edge in `orchestrator/graph.py`.
4. Add a table (if the agent produces persistent output) in `aca/models/`.
5. Unit test in `tests/unit/test_<name>.py` with a recorded LLM fixture.
6. Update this doc.

---

## Prompt-cache hygiene

Cached prefixes (per tenant, refreshed daily):
- Brand voice guide (Markdown, ~2–8 KB).
- Site taxonomy JSON (~1–4 KB).
- Rubric text (Reviewer only, ~2 KB).
- Analyzer schema + output examples (~1 KB).

Target cache reuse: ≥ 70% of Sonnet input tokens for the day.
