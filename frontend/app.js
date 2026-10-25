/* app.js - UI logic. All API text goes in via textContent (never innerHTML) for XSS safety.
   Privacy (rule 8): nothing is saved in localStorage/cookies; the URL hash holds the query and
   filters (NOT the ZIP, age, sex or conditions) so Back keeps them. */
"use strict";

const STATUS_LABELS = {
  RECRUITING: "Recruiting", NOT_YET_RECRUITING: "Not yet recruiting",
  ACTIVE_NOT_RECRUITING: "Active, not recruiting", ENROLLING_BY_INVITATION: "Enrolling by invitation",
  COMPLETED: "Completed", TERMINATED: "Terminated", WITHDRAWN: "Withdrawn", SUSPENDED: "Suspended",
  UNKNOWN: "Status unknown"
};
const PHASE_LABELS = { EARLY_PHASE1: "Early Phase 1", PHASE1: "Phase 1", PHASE2: "Phase 2", PHASE3: "Phase 3", PHASE4: "Phase 4", NA: "Not applicable" };
const DOCTOR_QUESTIONS = [
  "Is this study a good fit to talk about for my type and stage of cancer?",
  "What would happen to me during the study, and how often would I need to visit?",
  "What are the possible benefits and risks compared with my other options?",
  "Will I have to pay for any part of the study, such as tests, travel, or medicines?",
  "How would joining affect my current treatment and my regular doctors?",
  "Who can I contact with questions, and can I leave the study at any time?"
];
const ICON_CHECK = "✓", ICON_WARN = "⚠", ICON_X = "✕", ICON_DOT = "•";

const $ = id => document.getElementById(id);
let cancerTypes = [];
let lastFocus = null;
let lastCancer = "";

/* Only ever link to a real ClinicalTrials.gov study page; rebuild from a validated NCT id, never trust data URLs. */
function safeStudyUrl(url, nct) {
  const base = "https://clinicaltrials.gov/study/";
  if (typeof url === "string" && url.startsWith(base) && /^NCT\d{8}$/.test(url.slice(base.length))) return url;
  return /^NCT\d{8}$/.test(nct || "") ? base + nct : null;
}

function el(tag, opts, ...kids) {
  const n = document.createElement(tag);
  if (opts) Object.entries(opts).forEach(([k, v]) => {
    if (k === "text") n.textContent = v; else if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v);
  });
  kids.forEach(k => k && n.appendChild(k));
  return n;
}
function safeLink(o, attrs) {
  const href = safeStudyUrl(o.url, o.nct_id);
  if (!href) return null;
  return el("a", Object.assign({ href, target: "_blank", rel: "noopener", text: "View on ClinicalTrials.gov (opens new tab)" }, attrs));
}
const say = msg => { $("status").textContent = msg; };
const clear = n => { while (n.firstChild) n.removeChild(n.firstChild); };

function phaseWords(p) {
  if (!p) return "Not stated";
  return String(p).split(/[|\/,]/).map(s => PHASE_LABELS[s.trim()] || s.trim())
    .join(" / ").replace(/Phase (\d) \/ Phase (\d)/, "Phase $1/$2");
}
const statusWords = s => STATUS_LABELS[s] || String(s || "Unknown");
function statusBadge(s) {
  const known = ["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"].includes(s);
  return el("span", { class: "badge s-" + (known ? s : "other"), text: statusWords(s) });
}
function typeLabel(id) { const t = cancerTypes.find(x => x.id === id); return t ? t.label : id; }

/* ---------- form state / hash ---------- */
function readForm() {
  return { q: $("q").value.trim(), zip: $("zip").value.trim(), radius_mi: $("radius").value,
           status: $("status-filter").value, age: $("age").value.trim(),
           sex: $("sex").value, conditions: $("conditions").value.trim() };
}
/* The "About you" details, or null when left blank. Used only for the current request, never stored. */
function profileOf(p) {
  return (p.age || p.sex || p.conditions) ? { age: p.age, sex: p.sex, conditions: p.conditions } : null;
}
const hasProfile = () => !!profileOf(readForm());

function writeHash(p, cancer) {
  const h = new URLSearchParams();
  if (p.q) h.set("q", p.q);
  if (cancer) h.set("cancer", cancer);
  h.set("radius", p.radius_mi); h.set("status", p.status);
  history.replaceState(null, "", "#" + h.toString());   // ZIP, age, sex and conditions deliberately excluded
}
function restoreFromHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.has("q")) $("q").value = h.get("q");
  if (h.has("radius")) $("radius").value = h.get("radius");
  if (h.has("status")) $("status-filter").value = h.get("status");
  return h.get("q") || h.get("cancer") ? { cancer: h.get("cancer") || "" } : null;
}

/* ---------- search ---------- */
async function runSearch(cancer) {
  const p = readForm();
  const zipErr = $("zip-err");
  if (p.zip && !/^\d{5}$/.test(p.zip)) {
    zipErr.hidden = false; $("zip").setAttribute("aria-invalid", "true"); $("zip").focus();
    say("Please enter a 5-digit ZIP code."); return;
  }
  zipErr.hidden = true; $("zip").removeAttribute("aria-invalid");
  if (!p.q && !cancer) { say("Type a cancer type or pick one below."); $("q").focus(); return; }
  lastCancer = cancer || "";
  const prof = profileOf(p);
  const params = { q: p.q || typeLabel(cancer), cancer: cancer || "", zip: p.zip, radius_mi: p.zip ? p.radius_mi : "",
                   status: p.status, age: p.age, sex: p.sex, conditions: p.conditions,
                   sort: prof && $("sort").value === "match" ? "match" : "" };
  $("sort-field").hidden = !prof;
  writeHash(p, cancer);
  $("chips").querySelectorAll(".chip").forEach(c => c.setAttribute("aria-pressed", String(c.dataset.id === cancer)));
  say("Loading results...");
  try {
    renderResults(await fetchSearch(params));
  } catch (e) {
    $("results-area").hidden = false; $("offtopic").hidden = true; clear($("results"));
    $("results").appendChild(el("li", { class: "error", role: "alert", text: e.message }));
    say("Error: " + e.message);
  }
}

function renderResults(data) {
  $("results-area").hidden = false;
  const list = $("results"); clear(list);
  const off = !!data.off_topic;
  $("offtopic").hidden = !off;
  if (off) {
    $("sort-field").hidden = true;
    const box = $("offtopic-chips"); clear(box);
    (data.suggested_types || []).forEach(t => box.appendChild(chipButton(t, true)));
    say("We only cover cancer trials. Pick a cancer type to continue.");
  } else if (!data.results.length) {
    list.appendChild(el("li", { text: "No trials matched. Try a wider distance or the status \"Any\"." }));
    say("No results found.");
  } else {
    data.results.forEach(r => list.appendChild(card(r)));
    say(data.results.length + " trial" + (data.results.length === 1 ? "" : "s") + " found.");
  }
  $("results-h").focus();
}

function card(r) {
  const li = el("li", { class: "card" });
  li.appendChild(el("h3", { text: r.title }));
  li.appendChild(el("div", { class: "meta" },
    statusBadge(r.status), el("span", { text: r.nct_id }), el("span", { text: phaseWords(r.phase) })));
  const tags = el("div");
  (r.cancer_tags || []).slice(0, 3).forEach(t => tags.appendChild(el("span", { class: "tag", text: typeLabel(t) })));
  li.appendChild(tags);
  if (r.match && hasProfile()) li.appendChild(matchBlock(r.match));
  const info = el("div", { class: "meta" });
  if (r.nearest_site) info.appendChild(el("span", { text: "Nearest site: " + r.nearest_site.city + ", " + r.nearest_site.state +
    " (" + Math.round(r.nearest_site.distance_mi) + " miles)" }));
  info.appendChild(el("span", { text: "Sponsor: " + r.sponsor }));
  info.appendChild(el("span", { text: "Last updated: " + r.last_updated }));
  li.appendChild(info);
  li.appendChild(el("div", { class: "actions" },
    safeLink(r, { class: "btn" }),
    copyButton(r.citation),
    el("button", { type: "button", class: "btn primary", text: "Explain this study",
      "aria-label": "Explain this study: " + r.nct_id, onclick: ev => openSummary(r.nct_id, ev.currentTarget) })));
  return li;
}

/* Match badge: icon + words (never color alone). Wording is a reading aid, not a verdict (rule 7). */
const MATCH_UI = {
  "Possible match": { cls: "m-possible", icon: ICON_CHECK },
  "Check with study team": { cls: "m-check", icon: ICON_WARN },
  "Likely not a match": { cls: "m-not", icon: ICON_X }
};
function matchBlock(m) {
  const ui = MATCH_UI[m.summary_label] || MATCH_UI["Check with study team"];
  const wrap = el("div", { class: "match-block" });
  wrap.appendChild(el("span", { class: "mbadge " + ui.cls },
    el("span", { "aria-hidden": "true", text: ui.icon }), el("span", { text: m.summary_label })));
  const names = l => (l || []).map(x => x.term).filter((t, i, a) => a.indexOf(t) === i).join(", ");
  const bits = [];
  if (m.age === "outside") bits.push("Your age is outside this study's age range.");
  if (m.sex === "no") bits.push("This study is for a different sex than you chose.");
  if ((m.mentioned_in_requirements || []).length) bits.push("Requirements mention: " + names(m.mentioned_in_requirements) + ".");
  if ((m.possible_conflicts || []).length) bits.push("A requirement may conflict with: " + names(m.possible_conflicts) + ".");
  if ((m.mentioned_in_exclusions || []).length) bits.push("Reasons not to join mention: " + names(m.mentioned_in_exclusions) + ".");
  bits.push("The study team decides who can join.");
  wrap.appendChild(el("p", { class: "match-note", text: bits.join(" ") }));
  return wrap;
}

function copyButton(text) {
  const b = el("button", { type: "button", class: "btn", text: "Copy citation" });
  b.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(text); b.textContent = "Copied"; say("Citation copied."); }
    catch (e) { b.textContent = "Copy failed"; say("Could not copy; select the text manually."); }
    setTimeout(() => { b.textContent = "Copy citation"; }, 2000);
  });
  return b;
}

function chipButton(t, fromOffTopic) {
  const b = el("button", { type: "button", class: "chip", text: t.label, "data-id": t.id, "aria-pressed": "false" });
  b.addEventListener("click", () => { if (fromOffTopic) $("q").value = t.label; runSearch(t.id); });
  return b;
}

/* ---------- summary panel ---------- */
async function openSummary(nct, trigger) {
  lastFocus = trigger;
  $("search-view").hidden = true; $("summary-view").hidden = false;
  const body = $("summary-body"); clear(body);
  body.appendChild(el("h2", { id: "summary-h", tabindex: "-1", text: "Loading summary..." }));
  say("Loading summary..."); $("summary-h").focus(); window.scrollTo(0, 0);
  const p = readForm();
  const [study, summary] = await Promise.allSettled([
    fetchStudy(nct, Object.assign({ zip: p.zip }, profileOf(p) || {})), fetchSummary(nct)]);
  clear(body);
  const sv = study.status === "fulfilled" ? study.value : null;
  const mv = summary.status === "fulfilled" ? summary.value : null;
  body.appendChild(el("h2", { id: "summary-h", tabindex: "-1", text: sv ? sv.title : nct }));
  body.appendChild(el("p", { class: "disclaimer", text: "For education, not medical advice." }));
  if (sv && sv.overview) body.appendChild(glance(sv.overview, hasProfile() ? sv.match : null));
  if (mv) renderSummary(body, mv); else body.appendChild(errBox(summary.reason));
  if (sv) body.appendChild(criteriaPanel(sv.criteria, mv ? mv.who_can_join : null, hasProfile()));
  if (mv) body.appendChild(originalText(mv.original));
  if (sv) body.appendChild(keyFacts(sv)); else body.appendChild(errBox(study.reason));
  body.appendChild(doctorQuestions());
  const acts = el("div", { class: "actions no-print" });
  if (sv) acts.appendChild(copyButton(sv.citation));
  acts.appendChild(el("button", { type: "button", class: "btn", text: "Print", onclick: () => window.print() }));
  body.appendChild(acts);
  say("Summary ready."); $("summary-h").focus();
}
const errBox = e => el("p", { class: "error", role: "alert", text: e.message });

function keyFacts(s) {
  const ages = s.min_age ? (s.max_age ? s.min_age + " to " + s.max_age : s.min_age + " and older") : "Not stated";
  const rows = [
    ["ClinicalTrials.gov ID", s.nct_id], ["Status", statusWords(s.status)], ["Phase", phaseWords(s.phase)],
    ["Conditions", (s.conditions || []).join(", ")], ["Ages", ages], ["Sex", s.sex === "ALL" ? "All" : s.sex],
    ["Sponsor", s.sponsor], ["Start date", s.start_date], ["Last updated", s.last_updated],
    ["Enrollment", s.enrollment != null ? String(s.enrollment) : "Not stated"],
    ["Sites", (s.sites || []).slice(0, 5).map(x => x.facility + ", " + x.city + ", " + x.state).join("; ")]
  ];
  const dl = el("dl", { class: "facts" });
  rows.forEach(([k, v]) => { if (v) { dl.appendChild(el("dt", { text: k })); dl.appendChild(el("dd", { text: v })); } });
  dl.appendChild(el("dt", { text: "Official page" }));
  dl.appendChild(el("dd", null, safeLink(s, {}) || el("span", { text: "Not available" })));
  return el("section", { class: "panel" }, el("h3", { text: "Key facts" }), dl);
}

function bullets(items) { const ul = el("ul"); (items || []).forEach(i => ul.appendChild(el("li", { text: i }))); return ul; }

/* "Study at a glance": the sentence and chips come from stored API fields only (rule 2), never the model. */
function glance(ov, m) {
  const chips = el("ul", { class: "fact-chips" });
  (ov.facts || []).forEach(f => chips.appendChild(el("li", { text: f.label + ": " + f.value })));
  const sec = el("section", { class: "panel glance", "aria-labelledby": "glance-h" },
    el("h3", { id: "glance-h", text: "Study at a glance" }), el("p", { class: "sentence", text: ov.sentence }), chips);
  if (m) {
    sec.appendChild(el("h4", { text: "How this study lines up with what you entered" }));
    sec.appendChild(matchBlock(m));
  }
  return sec;
}

function renderSummary(body, m) {
  const ok = !m.fallback_used;
  const status = ok ? el("span", { class: "check", text: ICON_CHECK + " Checked against the original" })
                    : el("span", { class: "notice", text: "Some parts are shown in the original wording (with plain definitions) because our checker couldn't confirm the simplified version." });
  const rl = m.reading_level ? el("p", { class: "muted", text: "Reading level: grade " + m.reading_level.before + " → grade " + m.reading_level.after }) : null;
  body.appendChild(el("section", { class: "panel" },
    el("h3", { text: "What this study is testing" }), status, rl, el("p", { text: m.what_is_tested })));
}

function originalText(o) {
  o = o || {};
  return el("details", { class: "panel" }, el("summary", { text: "Original text from ClinicalTrials.gov" }),
    el("h4", { text: "Summary" }), el("pre", { class: "orig", text: o.summary || "" }),
    el("h4", { text: "Eligibility" }), el("pre", { class: "orig", text: o.eligibility || "" }));
}

/* "What they're looking for" / "Who can't join". The plain-language bullets (already faithfulness-checked) are used
   only when they line up one-to-one with the original criteria; otherwise we show the original wording with plain
   definitions added. Highlights use an icon AND words, never color alone. */
function critList(items, plain, showFlags) {
  const useModel = Array.isArray(plain) && plain.length === items.length;
  const ul = el("ul", { class: "crit-list" });
  items.forEach((c, i) => {
    const flag = showFlags ? c.flag : null;
    const li = el("li", { class: flag ? "c-" + flag : "" });
    li.appendChild(el("span", { class: "ico", "aria-hidden": "true", text: flag === "match" ? ICON_CHECK : flag ? ICON_WARN : ICON_DOT }));
    const txt = el("span", null, el("span", { text: useModel ? plain[i] : c.text }));
    if (flag) {
      const lead = flag === "match" ? "Mentions something you entered: " : "Check this one. It mentions something you entered: ";
      txt.appendChild(el("span", { class: "why", text: lead + (c.terms || []).join(", ") }));
    }
    li.appendChild(txt);
    ul.appendChild(li);
  });
  if (!items.length) ul.appendChild(el("li", { text: "None listed." }));
  return ul;
}
function criteriaPanel(criteria, who, showFlags) {
  const c = criteria || { include: [], exclude: [] };
  const w = who || {};
  return el("section", { class: "panel", "aria-labelledby": "looking-h" },
    el("h3", { id: "looking-h", text: "What they're looking for" }),
    critList(c.include, w.include, showFlags),
    el("h3", { text: "Who can't join" }),
    critList(c.exclude, w.exclude, showFlags),
    showFlags ? el("p", { class: "muted", text: "Highlights only show where a rule mentions something you entered. The study team decides who can join." }) : null);
}

function doctorQuestions() {
  return el("section", { class: "panel" }, el("h3", { text: "Questions to ask your doctor" }), bullets(DOCTOR_QUESTIONS));
}

function backToResults() {
  $("summary-view").hidden = true; $("search-view").hidden = false;
  say("Back to results.");
  if (lastFocus && document.contains(lastFocus)) lastFocus.focus(); else $("results-h").focus();
}

/* ---------- init ---------- */
async function init() {
  $("search-form").addEventListener("submit", ev => { ev.preventDefault(); runSearch(""); });
  $("sort").addEventListener("change", () => runSearch(lastCancer));
  $("back").addEventListener("click", backToResults);
  try {
    cancerTypes = await fetchCancerTypes();
    cancerTypes.forEach(t => $("chips").appendChild(chipButton(t, false)));
  } catch (e) { say("Could not load cancer types: " + e.message); }
  const restored = restoreFromHash();
  if (restored) runSearch(restored.cancer);
}
init();
