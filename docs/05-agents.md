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
**Job:** Read the low-scoring prompt, the four model responses, and the URLs each response cites. Group the citations by theme. Identify what those sources cover that our own site does not. Emit a `ContentGap` with a one-line editorial angle.

**Input:** `state.prompts: list[PromptWithSources]` (from `tools/aimpact.get_audit_results(include=sources)`).

**Output:** `state.gaps: list[ContentGap]`.

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
  1. THEMES the cited sources cover (concrete concepts, not just topics)
  2. What of those themes we do NOT already have a page for (compare with our taxonomy)
  3. A one-line EDITORIAL ANGLE for a blog that would win this query back
  4. A natural-language HYPOTHESIS explaining why we're losing today

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
    {"model":"chatgpt", "url":"...", "domain":"...", "competitor":"...", "excerpt":"..."}
  ],
  "themes": ["BAA scope", "audit-log requirements", "e-signature + PHI retention"],
  "target_angle": "Show what an auditor actually checks, using our product as the worked example.",
  "hypothesis": "Every cited source is a compliance checklist page. We have no walk-through of a real HIPAA-compliant intake form flow."
}
```

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
