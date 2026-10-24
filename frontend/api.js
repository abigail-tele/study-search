/* api.js - data access. Flip USE_MOCK to false to call the real /api/... endpoints.
   WHY one flag: the UI never knows where data came from, so mock and live behave the same. */
"use strict";
const USE_MOCK = true;
const MOCK_BASE = "mock/";

async function getJSON(url) {
  let resp;
  try { resp = await fetch(url, { headers: { Accept: "application/json" } }); }
  catch (e) { throw new Error("Could not reach the server. Please check your connection and try again."); }
  let data = null;
  try { data = await resp.json(); } catch (e) { /* non-JSON body */ }
  if (!resp.ok || (data && !Array.isArray(data) && data.error)) {
    throw new Error((data && data.error) || "Something went wrong (error " + resp.status + ").");
  }
  return data;
}

let cancerTypesCache = null;

async function fetchCancerTypes() {
  if (!cancerTypesCache) {
    cancerTypesCache = await getJSON(USE_MOCK ? MOCK_BASE + "cancer-types.json" : "/api/cancer-types");
  }
  return cancerTypesCache;
}

/* Mock-only: broad phrases the real backend treats as on-topic (umbrella terms). */
const UMBRELLA_PHRASES = ["blood cancer", "any cancer", "all cancers"];

/* Mock-only: does the query contain a lay term (or label) of a known type? */
async function mockMatches(q, cancer) {
  if (cancer) return true;
  const text = q.toLowerCase();
  if (UMBRELLA_PHRASES.some(p => text.includes(p))) return true;
  const types = await fetchCancerTypes();
  return types.some(t => t.lay_terms.concat([t.label]).some(term => text.includes(term.toLowerCase())));
}

/* params: {q, cancer, zip, radius_mi, status, age}; empty values are omitted. */
async function fetchSearch(params) {
  if (USE_MOCK) {
    const hit = await mockMatches(params.q || "", params.cancer);
    return getJSON(MOCK_BASE + (hit ? "search.json" : "search-offtopic.json"));
  }
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => { if (v !== "" && v != null) qs.set(k, v); });
  return getJSON("/api/search?" + qs.toString());
}

async function fetchStudy(nct) {
  return getJSON(USE_MOCK ? MOCK_BASE + "study-NCT05000001.json" : "/api/study/" + encodeURIComponent(nct));
}

async function fetchSummary(nct) {
  /* Mock: NCT05000002 demos the fallback_used=true look; every other id gets the normal mock. */
  if (USE_MOCK) return getJSON(MOCK_BASE + (nct === "NCT05000002" ? "summary-NCT05000002-fallback.json" : "summary-NCT05000001.json"));
  return getJSON("/api/summary/" + encodeURIComponent(nct));
}
