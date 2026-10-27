const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const version = () => document.querySelector('input[name="version"]:checked')?.value || "baseline";
const color = (score) => (score >= 80 ? "var(--good)" : score >= 60 ? "var(--warn)" : "var(--bad)");
const riskColor = (r) => color(100 * (1 - r));
const pct = (r) => `${Math.round(r * 100)}%`;

// Browser storage is only a convenience (theme, history). The app works without it.
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* ignore */ } },
};

// ---------- theme (dark / light) ----------
$("#theme-toggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("aices-theme", next); } catch (e) { /* ignore */ }
});

// ---------- navigation ----------
function showTab(name) {
  $$(".navtab").forEach((x) => x.classList.toggle("active", x.dataset.tab === name));
  $$(".panel").forEach((p) => p.classList.toggle("hidden", p.id !== `tab-${name}`));
  if (name === "history") renderHistory();
}
$$(".navtab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
$$(".subtab").forEach((t) => t.addEventListener("click", () => {
  $$(".subtab").forEach((x) => x.classList.toggle("active", x === t));
  $$(".subpanel").forEach((p) => p.classList.toggle("hidden", p.id !== `sub-${t.dataset.sub}`));
}));

// ---------- status + pipeline cards ----------
const MODULES = [
  { key: "bias", tag: "M1", name: "Bias", what: "All bias types in one module: identity (stereotypes about gender, religion, race), framing (loaded / opinionated words) and one-sided arguments", data: "Jigsaw Unintended Bias · BABE · IBM Claim Stance", model: "Identity: TF-IDF + LR + group lexicon · Framing: TF-IDF + LR · One-sided: pretrained NLI (zero-shot)", xai: "LIME / word weights · PRO / CON per sentence", keys: ["identity_bias", "framing_bias", "stance"] },
  { key: "hallucination", tag: "M2", name: "Hallucination", what: "Answer not supported by the facts", data: "HaluEval (QA)", model: "3 overlap features + Logistic Regression", xai: "Feature contributions" },
  { key: "privacy", tag: "M3", name: "Privacy leakage", what: "Names, emails, phones, addresses, IDs", data: "ai4privacy PII", model: "Word features + Logistic Regression", xai: "Highlighted spans" },
  { key: "deepfake", tag: "M4", name: "AI-generated image", what: "How suspicious an image is of being AI-generated", data: "Pretrained on real vs AI images", model: "Pretrained Swin Transformer (Organika/sdxl-detector)", xai: "Occlusion map" },
];

fetch("/api/status").then((r) => r.json()).then((s) => {
  const base = s.models.baseline;
  const ready = Object.values(base).filter(Boolean).length;
  $("#status").textContent = `Models ready: ${ready}/${Object.keys(base).length}` + (s.lime_installed ? " · LIME on" : "");
  const ok = (m) => (m.keys || [m.key]).every((k) => base[k]);
  $("#module-cards").innerHTML = MODULES.map((m) => `
    <div class="card mcard">
      <div class="row between"><span class="kicker">${m.tag}</span>
        <span class="status-dot ${ok(m) ? "on" : "off"}">${ok(m) ? "trained" : "not trained"}</span></div>
      <h4>${m.name}</h4><div class="small">${m.what}</div>
      <dl><dt>Dataset</dt><dd>${m.data}</dd><dt>Model</dt><dd>${m.model}</dd><dt>Explain</dt><dd>${m.xai}</dd></dl>
    </div>`).join("");
}).catch(() => { $("#status").textContent = "Could not connect to the server"; });

// ---------- benchmarks ----------
const fmt = (v) => (v == null ? `<span class="pending">pending</span>` : (v * 100).toFixed(1) + "%");
fetch("/api/metrics").then((r) => r.json()).then((d) => {
  const w = d.weights || {};
  $("#eq-weights").textContent = "Weights: " + Object.entries(w).map(([k, v]) => `${k} ${v}`).join(" · ");
  $("#bench-body").innerHTML = Object.values(d.metrics || {}).map((m) => `
    <tr><td>${esc(m.name)}</td><td>${esc(m.dataset)}</td><td>${esc(m.model)}</td>
    <td class="num">${fmt(m.accuracy)}</td><td class="num">${fmt(m.f1)}</td><td class="num">${fmt(m.auc)}</td></tr>`).join("");
});

// ---------- sample ----------
const SAMPLE = {
  question: "Should we use nuclear energy?",
  answer: "Nuclear energy is the cleanest and most reliable source of power. It produces almost no carbon emissions. " +
    "Anyone who opposes it is ignorant and reckless. The Eiffel Tower was built in 1925 by Gustave Eiffel. " +
    "For details contact Rahul Sharma at rahul.sharma@gmail.com or 9876543210.",
  reference: "Nuclear power produces very low greenhouse gas emissions but creates radioactive waste and has high construction costs. " +
    "The Eiffel Tower was completed in 1889 and designed by the company of Gustave Eiffel.",
};
$("#sample").addEventListener("click", () => {
  $("#question").value = SAMPLE.question; $("#answer").value = SAMPLE.answer; $("#reference").value = SAMPLE.reference;
});

// ---------- render helpers ----------
const na = (m) => `<div class="na">Not available: ${esc(m.reason)}</div>`;
const bar = (risk) => `<div class="bar"><i style="width:${pct(risk)};background:${riskColor(risk)}"></i></div>`;
const chips = (words) => !words || !words.length ? "" :
  `<div class="chips">${words.map((w) => `<span class="chip ${w.weight > 0 ? "pos" : "neg"}">${esc(w.feature)} ${w.weight > 0 ? "+" : ""}${w.weight}</span>`).join("")}</div>`;
const scorePill = (score) => `<span class="pill" style="background:${color(score)}">${score}/100</span>`;
const head = (tag, title, m) => `<div class="mod-head"><h3><small>${tag}</small>${title}</h3>${m.available ? scorePill(m.score) : ""}</div>`;

function renderSub(title, part) {
  if (!part.available) return `<div class="sub"><b>${title}</b>${na(part)}</div>`;
  let body = `${bar(part.risk)}<div class="small">risk ${pct(part.risk)} · ${esc(part.model)}</div>`;
  if (part.flagged_sentences) {
    body += part.flagged_sentences.length
      ? part.flagged_sentences.map((f) => `<div class="quote">${esc(f.sentence)} <span class="small">(${pct(f.probability)})</span></div>`).join("")
      : `<div class="small">No sentence was flagged.</div>`;
    if (part.explanation && part.explanation.words.length)
      body += `<div class="small" style="margin-top:6px">Why (${esc(part.explanation.method)}), most suspicious sentence:</div>${chips(part.explanation.words)}`;
  }
  if (part.sentences && part.formula) {
    body += `<div class="small">PRO ${part.pro} · CON ${part.con} · neutral ${part.neutral} → leaning <b>${esc(part.leaning)}</b> · ${esc(part.formula)}</div>`;
    body += part.sentences.map((s) => `<div class="quote"><span class="stance-${s.stance}">${s.stance}</span> ${esc(s.sentence)}</div>`).join("");
  }
  return `<div class="sub"><b>${title}</b>${body}</div>`;
}

function renderBias(m) {
  if (!m.available) return `<div class="card mod">${head("M1", "Bias", m)}${na(m)}</div>`;
  const lv = (l) => l === "High" ? "var(--bad)" : l === "Medium" ? "var(--warn)" : "var(--good)";
  const summary = m.found.length ? `Bias found: <b>${m.found.join(", ")}</b>` : "No clear bias found";
  const types = m.types.map((t) => t.available
    ? `<div class="btype"><div class="row between"><span><b>${t.type}</b> <span class="small">${esc(t.what)}</span></span>
         <span class="pill" style="background:${lv(t.level)}">${t.level} · ${pct(t.risk)}</span></div>${bar(t.risk)}</div>`
    : `<div class="btype"><b>${t.type}</b> <span class="small">not available: ${esc(t.reason)}</span></div>`).join("");
  const flagged = m.sentences.filter((r) => r.tags.length);
  const rows = flagged.length
    ? flagged.map((r) => `<div class="quote">${esc(r.sentence)}<div>${r.tags.map((t) => `<span class="chip pos">${t.type}</span> <span class="small">${esc(t.detail)}</span>`).join("<br>")}</div></div>`).join("")
    : `<div class="small">No sentence was flagged.</div>`;
  const ex = [m.parts.identity, m.parts.framing].filter((p) => p.available && p.explanation && p.explanation.words && p.explanation.words.length && p.flagged_sentences.length);
  const why = ex.map((p) => `<div class="small" style="margin-top:6px">Why (${esc(p.explanation.method)}): ${esc(p.most_suspicious)}</div>${chips(p.explanation.words)}`).join("");
  const side = m.parts.one_sided.available ? `<div class="small">Stance: PRO ${m.parts.one_sided.pro} · CON ${m.parts.one_sided.con} · neutral ${m.parts.one_sided.neutral} → leaning <b>${esc(m.parts.one_sided.leaning)}</b></div>` : "";
  return `<div class="card mod">${head("M1", "Bias", m)}
    <div class="small" style="margin-bottom:8px">${summary} · score = 100 × (1 − max(1.0 × identity, 0.8 × framing, 0.6 × one-sided))</div>
    ${types}${side}
    <div class="small" style="margin-top:10px"><b>Biased sentences</b></div>${rows}${why}</div>`;
}

function renderHalluc(m) {
  if (!m.available) return `<div class="card mod">${head("M2", "Hallucination", m)}${na(m)}</div>`;
  const rows = m.sentences.map((s) => s.skipped
    ? `<div class="quote" style="opacity:.6">${esc(s.sentence)} <span class="small">(not fact-checked: ${esc(s.reason)})</span></div>`
    : `<div class="quote">${esc(s.sentence)} <span class="small">(${pct(s.probability)} hallucinated)</span>
      ${s.reasons ? `<div class="chips">${s.reasons.filter((r) => r.weight > 0).map((r) => `<span class="chip pos">${esc(r.reason)}</span>`).join("")}</div>` : ""}</div>`).join("");
  return `<div class="card mod">${head("M2", "Hallucination", m)}
    <div class="small">${esc(m.model)} · evidence: ${esc(m.evidence_source)} · only sentences with a number or a name are fact-checked</div>${bar(m.risk)}${rows}</div>`;
}

function highlight(text, spans) {
  let out = "", i = 0;
  for (const s of spans) {
    if (s.start < i) continue;
    out += esc(text.slice(i, s.start)) + `<mark class="pii" title="${esc(s.source)} · ${s.confidence}">${esc(text.slice(s.start, s.end))}<sup>${esc(s.label)}</sup></mark>`;
    i = s.end;
  }
  return out + esc(text.slice(i));
}

function renderPrivacy(m, text) {
  if (!m.available) return `<div class="card mod">${head("M3", "Privacy", m)}${na(m)}</div>`;
  const counts = Object.entries(m.counts).map(([k, v]) => `<span class="chip pos">${esc(k)} × ${v}</span>`).join("");
  return `<div class="card mod">${head("M3", "Privacy", m)}
    <div class="small">${esc(m.model)} + regex backup (Aadhaar / PAN / Indian mobile)</div>
    <div class="chips">${counts || '<span class="small">No PII found.</span>'}</div>
    <div class="textbox">${highlight(text, m.spans)}</div></div>`;
}

function renderTextResult(r, text, multi) {
  const o = r.overall;
  const ring = o.score == null ? "" : `<div class="ring" style="--p:${o.score};--c:${color(o.score)}"><span>${Math.round(o.score)}</span></div>`;
  const minis = ["bias", "hallucination", "privacy"].map((k) => {
    const m = r.modules[k];
    return `<span class="mini">${k}<b style="color:${m.available ? color(m.score) : "var(--faint)"}">${m.available ? Math.round(m.score) : "–"}</b></span>`;
  }).join("");
  return `<div class="result">${multi ? `<div class="vtitle">${esc(r.version)}</div>` : ""}
    <div class="card overall">${ring}<div>
      <div class="kicker">Integrity Score · weighted harmonic mean</div>
      <div class="verdict">${esc(o.verdict)}</div>
      <div class="mini-scores">${minis}</div></div></div>
    ${renderBias(r.modules.bias)}${renderHalluc(r.modules.hallucination)}${renderPrivacy(r.modules.privacy, text)}</div>`;
}

async function post(url, opts, btn, target) {
  btn.disabled = true;
  target.innerHTML = `<div class="card small loading">Evaluating…</div>`;
  try {
    const res = await fetch(url, opts);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  } catch (e) {
    target.innerHTML = `<div class="card error">${esc(e.message)}</div>`;
    return null;
  } finally {
    btn.disabled = false;
  }
}

// ---------- text evaluate ----------
$("#text-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const answer = $("#answer").value;
  const body = { question: $("#question").value, answer, reference: $("#reference").value, version: version() };
  const target = $("#text-results");
  const data = await post("/api/analyze-text", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
    e.submitter || $("#text-form button"), target);
  if (!data) return;
  const multi = data.results.length > 1;
  target.innerHTML = `<div class="columns ${multi ? "two" : ""}">${data.results.map((r) => renderTextResult(r, answer.trim(), multi)).join("")}</div>`;
  saveHistory(body, data.results[0].overall);
  target.scrollIntoView({ behavior: "smooth", block: "start" });
});

// ---------- history (stored in this browser only) ----------
function saveHistory(input, overall) {
  const list = store.get("aices-history", []);
  list.unshift({ ...input, score: overall.score, verdict: overall.verdict, at: new Date().toISOString() });
  store.set("aices-history", list.slice(0, 20));
}

function renderHistory() {
  const list = store.get("aices-history", []);
  $("#history-list").innerHTML = list.length ? list.map((h, i) => `
    <div class="card hitem" data-i="${i}" role="button" tabindex="0">
      <div class="hscore" style="color:${h.score == null ? "var(--faint)" : color(h.score)}">${h.score == null ? "–" : Math.round(h.score)}</div>
      <div style="min-width:0"><div class="kicker">${esc(h.verdict)}</div><div class="htext">${esc(h.answer)}</div></div>
      <div class="kicker">${new Date(h.at).toLocaleString()}</div>
    </div>`).join("") : `<div class="card small">No evaluations yet. Go to Analyze Content and click Evaluate.</div>`;
  $$(".hitem").forEach((el) => {
    const open = () => {
      const h = list[+el.dataset.i];
      $("#question").value = h.question || ""; $("#answer").value = h.answer || ""; $("#reference").value = h.reference || "";
      showTab("analyze");
    };
    el.addEventListener("click", open);
    el.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
  });
}
$("#clear-history").addEventListener("click", () => { store.set("aices-history", []); renderHistory(); });

// ---------- image ----------
$("#image").addEventListener("change", () => {
  const f = $("#image").files[0];
  if (!f) return;
  $("#preview").src = URL.createObjectURL(f);
  $("#preview").classList.remove("hidden");
});

$("#image-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = $("#image").files[0];
  const target = $("#image-results");
  if (!f) { target.innerHTML = `<div class="card error">Please choose an image first</div>`; return; }
  const fd = new FormData();
  fd.append("image", f);
  fd.append("version", version());
  const data = await post("/api/analyze-image", { method: "POST", body: fd }, e.submitter || $("#image-form button"), target);
  if (!data) return;
  const src = $("#preview").src;
  const multi = data.results.length > 1;
  target.innerHTML = `<div class="columns ${multi ? "two" : ""}">${data.results.map((r) => {
    const d = r.deepfake;
    const title = multi ? `<div class="vtitle">${esc(r.version)}</div>` : "";
    if (!d.available) return `<div class="result">${title}<div class="card mod">${head("M4", "AI-generated image check", d)}${na(d)}</div></div>`;
    const c = d.level === "High" ? "var(--bad)" : d.level === "Medium" ? "var(--warn)" : "var(--good)";
    return `<div class="result">${title}<div class="card mod">
      <div class="mod-head"><h3><small>M4</small>AI-generated image check</h3><span class="pill" style="background:${c}">${d.suspicion}% suspicious · ${d.level}</span></div>
      <div class="small"><b>${d.label}</b> · ${esc(d.model)} · P(AI) = ${d.prob_fake} · Low &lt; 30% · Medium 30–70% · High ≥ 70%</div>
      <div class="imgpair"><img src="${src}" alt="uploaded">
      ${d.gradcam_png ? `<div><img src="data:image/png;base64,${d.gradcam_png}" alt="Occlusion map"><div class="small">Occlusion map: red areas = regions that drive the decision</div></div>` : ""}</div>
    </div></div>`;
  }).join("")}</div>`;
});
