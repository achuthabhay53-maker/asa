const stageNames = {
  aimpact: ["Prompt intelligence", "Source intelligence", "Ground truth", "Content brief", "Blog writer", "Reviewer"],
  general: ["Input & research plan", "Company research", "Reference research", "Blog brief", "Blog writer", "Quality review"],
};
const stageKeys = {
  aimpact: ["prompt_analysis", "source_analysis", "guide", "content_brief", "written_article", "review_output"],
  general: ["intake", "company_research", "reference_research", "brief", "writer", "review"],
};
const tabs = {
  aimpact: ["Article", "Edit & revise", "Review", "Approve", "Agent outputs"],
  general: ["Article", "Edit & revise", "Review", "Sources", "Agent outputs"],
};
const state = { workflow: "aimpact", run: null, activeTab: "Article", pollTimer: null, working: false };
const $ = (selector) => document.querySelector(selector);
const workspace = $("#workspace");

function escapeHtml(value = "") {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function showNotice(message, error = false) {
  const notice = $("#notice");
  notice.textContent = message;
  notice.classList.toggle("error", error);
  notice.hidden = !message;
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const type = response.headers.get("content-type") || "";
  const data = type.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const message = typeof data === "object" ? (data.detail || data.message || response.statusText) : data;
    throw new Error(Array.isArray(message) ? message.map((item) => item.msg || JSON.stringify(item)).join("; ") : message);
  }
  return data;
}

function renderStages(completed = []) {
  const names = stageNames[state.workflow];
  const keys = stageKeys[state.workflow];
  $("#stage-list").innerHTML = names.map((name, index) => {
    const done = Array.isArray(completed) ? completed.includes(keys[index]) : Number(completed) > index;
    return `<li class="stage-item ${done ? "done" : ""}"><span class="stage-node">${done ? "✓" : index + 1}</span><div class="stage-name">${escapeHtml(name)}</div></li>`;
  }).join("");
}

function setWorkflow(workflow) {
  state.working = false;
  clearTimeout(state.pollTimer);
  state.workflow = workflow;
  state.run = null;
  state.activeTab = "Article";
  document.querySelectorAll(".workflow-link").forEach((button) => {
    button.classList.toggle("active", button.dataset.workflow === workflow);
  });
  const isGeneral = workflow === "general";
  $("#page-title").textContent = isGeneral ? "General blog" : "Campaign blog";
  $("#page-subtitle").textContent = isGeneral
    ? "Bring your topic and keywords. Build a clear, evidence-led article."
    : "Turn a campaign prompt into a sourced, reviewed article.";
  $("#form-title").textContent = isGeneral ? "Build a general blog" : "Build a campaign blog";
  $("#aimpact-form").hidden = isGeneral;
  $("#general-form").hidden = !isGeneral;
  $("#run-state").className = "state-chip idle";
  $("#run-state").innerHTML = "<i></i><span>Ready</span>";
  $("#result-title").textContent = "Your next story starts here";
  $("#result-content").hidden = false;
  $("#result-tabs").hidden = true;
  $("#result-view").hidden = true;
  $("#result-content").innerHTML = '<div class="empty-art"><span>✦</span><i></i><b></b></div><h3>Ready when you are</h3><p>Complete the form to start a new workflow. Your progress and results will appear here.</p>';
  renderStages([]);
  showNotice("");
}

function parseMarkdown(markdown) {
  let html = escapeHtml(markdown);
  html = html.replace(/^###### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^##### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^#### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/^# (.+)$/gm, "<h1>$1</h1>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/^\s*[-*] (.+)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>\n?)+/g, (items) => `<ul>${items}</ul>`);
  html = html.split(/\n{2,}/).map((chunk) => {
    const trimmed = chunk.trim();
    if (!trimmed || /^<(h[1-6]|ul|ol|blockquote|table|li)/.test(trimmed)) return trimmed;
    return `<p>${trimmed.replace(/\n/g, "<br>")}</p>`;
  }).join("\n");
  return html.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}

function currentOutput() {
  if (!state.run) return null;
  return state.workflow === "aimpact"
    ? state.run.agent_outputs?.written_article
    : state.run.draft;
}

function currentReview() {
  if (!state.run) return null;
  return state.workflow === "aimpact"
    ? state.run.agent_outputs?.review_output
    : state.run.review;
}

function articleMarkdown() {
  const output = currentOutput();
  return state.workflow === "aimpact" ? output?.article_markdown : output?.article_markdown;
}

function downloadText(name, text, mime = "text/plain") {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([text], { type: mime }));
  link.download = name;
  link.click();
  URL.revokeObjectURL(link.href);
}

function downloadJson(name, value) {
  downloadText(name, JSON.stringify(value, null, 2), "application/json");
}

function switchTab(tab) {
  state.activeTab = tab;
  renderRun();
}

function renderArticle() {
  const output = currentOutput();
  if (!output) return '<div class="empty-inline">The writer has not completed yet.</div>';
  const markdown = output.article_markdown || "";
  const brief = state.workflow === "aimpact" ? state.run.agent_outputs?.content_brief : state.run.brief;
  const approved = state.workflow === "aimpact" && state.run.approved;
  const target = state.workflow === "aimpact" ? state.run.target_word_count : state.run.writer_input?.target_word_count;
  const primaryKeyword = state.workflow === "aimpact"
    ? (brief?.primary_keyword || brief?.seo_keyword || "")
    : (brief?.primary_keyword || "");
  const seoTitle = output.seo_title || "";
  const metaDescription = output.meta_description || "";
  const usage = state.run.token_usage || [];
  const headings = (markdown.match(/^##\s+.+$/gm) || []).map((heading) => heading.replace(/^##\s+/, ""));
  const articleHtml = state.run.article_html;
  return `<div class="article-layout">
    <article class="article-content">${highlightFlags(articleHtml || parseMarkdown(markdown))}</article>
    <aside class="side-panel">
      <h3>Article details</h3>
      <p><strong>Reading time</strong><br>${Math.max(1, Math.round((output.word_count || 0) / 220))} min</p>
      <p><strong>Words</strong><br>${Number(output.word_count || 0).toLocaleString()}${target ? ` / ${Number(target).toLocaleString()} target` : ""}</p>
      ${primaryKeyword ? `<p><strong>Primary keyword</strong><br>${escapeHtml(primaryKeyword)}</p>` : ""}
      ${seoTitle ? `<p><strong>SEO title</strong><br>${escapeHtml(seoTitle)}</p>` : ""}
      ${metaDescription ? `<p><strong>Meta description</strong><br>${escapeHtml(metaDescription)}</p>` : ""}
      <p><strong>Review</strong><br>${state.workflow === "general" ? (state.run.review_current ? (state.run.quality_passed ? "Passed" : "Needs human review") : "Review is out of date") : (state.run.review_current ? "Current" : "Not reviewed")}</p>
      <p><strong>Approval</strong><br>${approved ? "Approved" : "Draft"}</p>
      ${headings.length ? `<details><summary>Article outline</summary><ul>${headings.map((heading) => `<li>${escapeHtml(heading)}</li>`).join("")}</ul></details>` : ""}
      ${usage.length ? `<details><summary>Claude usage · ${usage.length} call(s)</summary>${usage.map((call) => `<p>${escapeHtml(call.agent_name || "Agent")}<br>${call.input_tokens ?? "?"} input · ${call.output_tokens ?? "?"} output</p>`).join("")}</details>` : ""}
      <button class="button button-secondary full-width" data-action="copy-md">Copy Markdown</button>
      <button class="button button-secondary full-width" data-action="download-md">Download Markdown</button>
      <button class="button button-secondary full-width" data-action="download-result">Download result JSON</button>
      ${state.workflow === "aimpact" ? `<a class="button button-secondary full-width" href="/api/aimpact/runs/${encodeURIComponent(state.run.run_id)}/article.pdf">Download PDF</a>` : ""}
    </aside>
  </div>`;
}

function renderEdit() {
  const output = currentOutput();
  if (!output) return '<div class="empty-inline">There is no article to edit yet.</div>';
  return `<h3 class="view-title">Edit or revise this article</h3>
    <p class="side-panel">Saving an edit or revision makes the current review out of date. Review the new version before approval.</p>
    <div class="panel-grid">
      <section><label class="eyebrow" for="article-editor">ARTICLE MARKDOWN</label>
        <textarea id="article-editor" class="editor">${escapeHtml(output.article_markdown || "")}</textarea>
        <button class="button button-primary" data-action="save-edit">Save edits as a new version</button>
      </section>
      <section><label class="eyebrow" for="revision-instructions">WRITER REVISION</label>
        <textarea id="revision-instructions" class="editor" style="min-height:120px" placeholder="Describe the changes you want."></textarea>
        <button class="button button-secondary" data-action="revise">Ask Agent 5 to revise</button>
        ${state.workflow === "aimpact" ? renderVersions() : ""}
      </section>
    </div>`;
}

function renderVersions() {
  const versions = state.run?.draft_versions || [];
  if (!versions.length) return "";
  return `<div class="soft-panel" style="margin-top:16px"><h3>Version history</h3>${versions.map((version) => `<div class="decision-card"><span>v${version.version} · ${escapeHtml(version.source || "Version")}</span>${version.version !== state.run.draft_version ? `<button class="button button-secondary" data-action="restore-version" data-version="${version.version}">Restore</button>` : '<span class="optional">Current</span>'}</div>`).join("")}</div>`;
}

function renderReview() {
  const review = currentReview();
  const isCurrent = state.workflow === "aimpact" ? state.run.review_current : state.run.review_current;
  if (!isCurrent) {
    return `<div class="soft-panel"><h3>Review the current version</h3><p>The draft has not been reviewed, or it changed after the last review.</p><button class="button button-primary" data-action="review">Run Agent 6 review</button></div>`;
  }
  const findings = review?.findings || [];
  const statuses = state.run.finding_statuses || {};
  const unresolved = review?.unresolved_items || [];
  const cards = state.workflow === "aimpact" ? (state.run.findings || []).map(({ id, finding }) => {
    const severity = (finding.severity || "medium").toLowerCase();
    const status = statuses[id] || "Open";
    return `<article class="finding-card">
      <div class="finding-head"><span class="severity ${escapeHtml(severity)}">${escapeHtml(severity)}</span><span class="optional">${escapeHtml(finding.area || "Content")}</span></div>
      <h3>${escapeHtml(finding.issue || "")}</h3><p>${escapeHtml(finding.recommendation || "")}</p>
      ${finding.target_text ? `<blockquote>${escapeHtml(finding.target_text)}</blockquote>` : ""}
      <div class="decision-card"><label for="decision-${id}">Decision</label><select id="decision-${id}" data-finding-id="${id}">${["Open","Resolved","Dismissed"].map((item) => `<option ${item === status ? "selected" : ""}>${item}</option>`).join("")}</select></div>
    </article>`;
  }).join("") : findings.map((finding) => `<article class="finding-card"><div class="finding-head"><span class="severity ${(finding.severity || "medium").toLowerCase()}">${escapeHtml(finding.severity || "medium")}</span><span class="optional">${escapeHtml(finding.area || "")}</span></div><h3>${escapeHtml(finding.issue || "")}</h3><p>${escapeHtml(finding.recommendation || "")}</p>${finding.target_text ? `<blockquote>${escapeHtml(finding.target_text)}</blockquote>` : ""}</article>`).join("");
  return `<div class="result-heading"><div><h3 class="view-title">Agent 6 · Review</h3><p>${escapeHtml(review?.summary || "Review complete.")}</p></div></div>
    ${cards || '<div class="empty-inline">No review findings were returned.</div>'}
    ${unresolved.length ? `<div class="soft-panel"><h3>Items to check before approval</h3><ul>${unresolved.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : ""}
    ${state.workflow === "aimpact" && state.run.findings?.length ? '<button class="button button-primary" data-action="save-decisions">Save review decisions</button>' : ""}`;
}

function renderApprove() {
  const output = currentOutput();
  const review = currentReview();
  if (!output) return '<div class="empty-inline">Create an article before approval.</div>';
  if (state.workflow === "general") {
    return `<div class="soft-panel"><h3>General Blog review status</h3><p>${state.run.review_current ? (state.run.quality_passed ? "This draft passed review." : "This draft needs human review.") : "The current draft needs a fresh review."}</p><p>Review and edit controls are available in the Article, Edit, and Review views.</p></div>`;
  }
  const openCount = state.run.open_findings || 0;
  const ready = state.run.review_current && openCount === 0;
  return `<div class="soft-panel"><h3>Approval checklist</h3>
    <p>${state.run.review_current ? "✓ The current version has been reviewed." : "○ Review this exact version first."}</p>
    <p>${state.run.review_current && openCount === 0 ? "✓ All findings have a decision." : `○ ${openCount} finding(s) are open.`}</p>
    <p>${(output.article_markdown.match(/\[VERIFY:/g) || []).length ? "⚠ This draft has source flags. Check them before approval." : "✓ No [VERIFY] markers found."}</p>
    ${state.run.approved ? `<p class="approved-message">Approved for export.</p>` : ready ? `<label class="decision-card"><input id="approval-ack" type="checkbox"> I reviewed this version and accept responsibility for remaining evidence flags.</label><button class="button button-primary" data-action="approve">Approve final draft</button>` : '<button class="button button-secondary" data-action="review">Review this version</button>'}
  </div>`;
}

function renderSources() {
  if (state.workflow === "aimpact") {
    const sources = state.run.agent_outputs?.source_analysis;
    const guide = state.run.agent_outputs?.guide;
    return `<div class="panel-grid">${[["Source research", sources],["Ground truth", guide]].map(([label, item]) => `<section class="soft-panel"><h3>${label}</h3><pre class="json-output">${item ? escapeHtml(JSON.stringify(item, null, 2)) : "No output yet."}</pre></section>`).join("")}</div>`;
  }
  const sources = state.run.sources || [];
  const pdfs = state.run.uploaded_pdf_statuses || [];
  const warnings = state.run.warnings || [];
  return `${warnings.map((warning) => `<article class="source-card"><p>${escapeHtml(warning)}</p></article>`).join("")}
    ${pdfs.map((pdf) => `<article class="source-card"><h3>${escapeHtml(pdf.file_name)} · ${escapeHtml(pdf.status)}</h3><p>${escapeHtml(pdf.message)}</p><small>${Number(pdf.extracted_text_characters || 0).toLocaleString()} text characters · ${pdf.evidence_count || 0} evidence items</small></article>`).join("")}
    ${sources.map((source) => `<article class="source-card"><h3>${escapeHtml(source.evidence_id)} · ${escapeHtml(source.source_title || source.source_url || "Reference")}</h3><p>${escapeHtml(source.claim)}</p>${source.evidence_quote ? `<blockquote>${escapeHtml(source.evidence_quote)}</blockquote>` : ""}${source.source_url ? `<a href="${escapeHtml(source.source_url)}" target="_blank" rel="noopener">Open source ↗</a>` : ""}</article>`).join("") || '<div class="empty-inline">No source-backed claims were extracted.</div>'}`;
}

function renderOutputs() {
  const outputs = state.workflow === "aimpact" ? state.run.agent_outputs : {
    plan: state.run.plan, brief: state.run.brief, writer_input: state.run.writer_input,
    writer: state.run.draft, reviewer: state.run.review,
  };
  if (!outputs) return '<div class="empty-inline">Agent outputs are not available.</div>';
  return Object.entries(outputs).map(([name, value]) => `<details class="output-card"><summary>${escapeHtml(name.replaceAll("_", " "))}</summary><button class="button button-secondary" data-action="download-output" data-name="${escapeHtml(name)}">Download JSON</button><pre class="json-output">${escapeHtml(JSON.stringify(value, null, 2))}</pre></details>`).join("");
}

function renderRun() {
  if (!state.run) return;
  const output = currentOutput();
  $("#result-content").hidden = true;
  $("#result-tabs").hidden = false;
  $("#result-view").hidden = false;
  const resultTabs = state.workflow === "general" ? tabs.general : tabs.aimpact;
  $("#result-title").textContent = output?.title || "Workflow results";
  $("#result-tabs").innerHTML = resultTabs.map((tab) => `<button class="tab-button ${state.activeTab === tab ? "active" : ""}" data-tab="${tab}">${tab}</button>`).join("");
  let view = "";
  if (state.activeTab === "Article") view = renderArticle();
  else if (state.activeTab === "Edit & revise") view = renderEdit();
  else if (state.activeTab === "Review") view = renderReview();
  else if (state.activeTab === "Approve") view = renderApprove();
  else if (state.activeTab === "Sources") view = renderSources();
  else view = renderOutputs();
  $("#result-view").innerHTML = view;
}

function applyRun(run, workflow = state.workflow) {
  state.workflow = workflow;
  state.run = run;
  const id = run.run_id || "";
  $("#result-title").textContent = currentOutput()?.title || (id ? `Run ${id.slice(0, 8)}` : "Workflow results");
  const chip = $("#run-state");
  const approved = workflow === "aimpact" && run.approved;
  chip.className = `state-chip ${approved ? "complete" : "complete"}`;
  chip.innerHTML = `<i></i><span>${approved ? "Approved" : "Complete"}</span>`;
  const completed = workflow === "aimpact"
    ? stageKeys.aimpact.filter((key) => run.agent_outputs?.[key])
    : stageKeys.general;
  renderStages(completed);
  state.activeTab = "Article";
  renderRun();
  showNotice(id ? `Run ID: ${id}` : "Workflow complete.");
}

async function pollJob(jobId, workflow) {
  state.working = true;
  const chip = $("#run-state");
  chip.className = "state-chip running";
  chip.innerHTML = "<i></i><span>In progress</span>";
  $("#result-title").textContent = "Your workflow is in progress";
  $("#result-content").hidden = false;
  $("#result-tabs").hidden = true;
  $("#result-view").hidden = true;
  $("#result-content").innerHTML = '<div class="empty-art"><span>✦</span><i></i><b></b></div><h3>Good work takes research</h3><p id="progress-message">Preparing your workflow…</p>';
  const timer = async () => {
    if (!state.working) return;
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
      $("#progress-message").textContent = job.message || "Working…";
      renderStages(job.completed || []);
      if (job.status === "complete") {
        state.working = false;
        if (workflow === "aimpact") applyRun(job.result, workflow);
        else {
          const actualId = job.run_id || jobId;
          const result = await api(`/api/general/runs/${encodeURIComponent(actualId)}`);
          applyRun(result, workflow);
        }
        return;
      }
      if (job.status === "failed") {
        state.working = false;
        chip.className = "state-chip failed";
        chip.innerHTML = "<i></i><span>Needs attention</span>";
        if (job.result) applyRun(job.result, workflow);
        showNotice(`${job.stage ? `${job.stage}: ` : ""}${job.error_type ? `${job.error_type}: ` : ""}${job.message || "Workflow failed."}`, true);
        return;
      }
      state.pollTimer = window.setTimeout(timer, 1200);
    } catch (error) {
      state.working = false;
      showNotice(error.message, true);
    }
  };
  await timer();
}

async function submitAimpact(event) {
  event.preventDefault();
  showNotice("");
  const data = Object.fromEntries(new FormData(event.currentTarget).entries());
  data.campaign_id = Number(data.campaign_id);
  data.top_n = Number(data.top_n);
  data.target_word_count = Number(data.target_word_count);
  try {
    const result = await api("/api/aimpact/runs", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
    });
    state.run = null;
    await pollJob(result.job_id, "aimpact");
  } catch (error) {
    showNotice(error.message, true);
  }
}

async function submitGeneral(event) {
  event.preventDefault();
  showNotice("");
  const form = event.currentTarget;
  const values = new FormData(form);
  const keywords = values.get("target_keywords").split(/[\n,]+/).map((item) => item.trim()).filter(Boolean);
  const urls = values.get("reference_urls").split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
  const request = {
    company_name: values.get("company_name"),
    company_website: values.get("company_website"),
    prompt: values.get("prompt"),
    target_audience: values.get("target_audience") || null,
    tone: values.get("tone") || null,
    target_keywords: keywords,
    reference_urls: urls,
    target_word_count: values.get("target_word_count") ? Number(values.get("target_word_count")) : null,
  };
  const body = new FormData();
  body.append("request_json", JSON.stringify(request));
  for (const file of values.getAll("reference_pdfs")) if (file.size) body.append("reference_pdfs", file);
  try {
    const result = await api("/api/general/runs", { method: "POST", body });
    state.run = null;
    await pollJob(result.job_id, "general");
  } catch (error) {
    showNotice(error.message, true);
  }
}

async function useAction(action, target) {
  if (!state.run) return;
  const id = encodeURIComponent(state.run.run_id);
  const prefix = state.workflow === "aimpact" ? `/api/aimpact/runs/${id}` : `/api/general/runs/${id}`;
  let response;
  if (action === "copy-md") { await navigator.clipboard.writeText(articleMarkdown() || ""); return showNotice("Markdown copied to clipboard."); }
  if (action === "download-md") return downloadText(`${state.workflow}_${state.run.run_id}.md`, articleMarkdown(), "text/markdown");
  if (action === "download-result") return downloadJson(`${state.workflow}_${state.run.run_id}_result.json`, state.run);
  if (action === "download-output") {
    const output = state.workflow === "aimpact" ? state.run.agent_outputs?.[target] : state.run[target];
    return downloadJson(`${target}.json`, output);
  }
  if (action === "save-edit") {
    response = await api(`${prefix}/edit`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ article_markdown: $("#article-editor").value }),
    });
  } else if (action === "revise") {
    response = await api(`${prefix}/revise`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instructions: $("#revision-instructions").value }),
    });
  } else if (action === "review") {
    response = await api(`${prefix}/review`, { method: "POST" });
  } else if (action === "save-decisions") {
    const statuses = Object.fromEntries([...document.querySelectorAll("[data-finding-id]")].map((element) => [element.dataset.findingId, element.value]));
    response = await api(`${prefix}/findings`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ statuses }),
    });
  } else if (action === "approve") {
    response = await api(`${prefix}/approve`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ acknowledged: $("#approval-ack")?.checked || false }),
    });
  } else if (action === "restore-version") {
    const version = (state.run.draft_versions || []).find((item) => Number(item.version) === Number(target));
    if (!version) throw new Error("The selected version is not available.");
    response = await api(`${prefix}/edit`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ article_markdown: version.output.article_markdown }),
    });
  }
  if (response) {
    state.run = response;
    if (action === "save-edit" || action === "revise") state.activeTab = "Article";
    renderRun();
    showNotice(action === "approve" ? "This draft is approved." : "Your changes are saved.");
  }
}

$("#aimpact-form").addEventListener("submit", submitAimpact);
$("#general-form").addEventListener("submit", submitGeneral);
document.querySelectorAll(".workflow-link").forEach((button) => button.addEventListener("click", () => setWorkflow(button.dataset.workflow)));
$("#new-run").addEventListener("click", () => setWorkflow(state.workflow));
$("#resume-button").addEventListener("click", async () => {
  const runId = $("#resume-id").value.trim();
  if (!runId) return showNotice("Enter a workflow run ID.", true);
  try {
    const run = await api(`/api/aimpact/runs/${encodeURIComponent(runId)}`);
    applyRun(run, "aimpact");
    showNotice(`Loaded saved run ${runId}.`);
  } catch (error) { showNotice(error.message, true); }
});
workspace.addEventListener("click", async (event) => {
  const tab = event.target.closest("[data-tab]");
  if (tab) return switchTab(tab.dataset.tab);
  const button = event.target.closest("[data-action]");
  if (!button) return;
  try { await useAction(button.dataset.action, button.dataset.version || button.dataset.name); }
  catch (error) { showNotice(error.message, true); }
});
setWorkflow("aimpact");
renderStages([]);

function highlightFlags(html) {
  return html.replace(/\[VERIFY:([^\]]*)\]/g, '<mark class="flag">[VERIFY:$1]</mark>');
}
function showView(view) {
  const studio = view === "app";
  $("#landing").hidden = studio;
  $("#app").hidden = !studio;
  document.title = studio ? "BlogGen | Studio" : "BlogGen | Evidence-led content studio";
  history.replaceState(null, "", studio ? "#studio" : location.pathname);
  window.scrollTo(0, 0);
}
document.addEventListener("click", (event) => {
  const go = event.target.closest("[data-go]");
  if (!go) return;
  event.preventDefault();
  showView(go.dataset.go);
});
if ("IntersectionObserver" in window) {
  const io = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.12 });
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
} else document.querySelectorAll(".reveal").forEach((el) => el.classList.add("in"));
showView(location.hash === "#studio" ? "app" : "landing");