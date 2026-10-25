"""'Study at a glance': one plain sentence plus fact chips, built ONLY from stored API fields (rule 2).

Why a template and not the model: status, phase, ages, sponsor and sites must never be rewritten by
a model, so this is fixed wording with the stored values dropped in. Nothing here is medical advice.
"""
from __future__ import annotations

import re

from backend import config, geo

_STATUS_WORDS = {
    "RECRUITING": "now recruiting", "NOT_YET_RECRUITING": "not yet recruiting",
    "ACTIVE_NOT_RECRUITING": "active but no longer recruiting",
    "ENROLLING_BY_INVITATION": "enrolling by invitation only",
}
_PHASE_WORDS = {"EARLY_PHASE1": "Early Phase 1", "PHASE1": "Phase 1", "PHASE2": "Phase 2",
                "PHASE3": "Phase 3", "PHASE4": "Phase 4"}
_SEX_WORDS = {"FEMALE": "women", "MALE": "men"}


_TYPE_WORDS = {"DRUG": "drug", "BIOLOGICAL": "biological", "DEVICE": "device", "PROCEDURE": "procedure",
               "RADIATION": "radiation", "BEHAVIORAL": "behavioral", "GENETIC": "genetic",
               "DIETARY_SUPPLEMENT": "dietary supplement", "COMBINATION_PRODUCT": "combination product",
               "DIAGNOSTIC_TEST": "diagnostic test"}   # OTHER / unknown: no label
_TYPE_ORDER = ["DRUG", "BIOLOGICAL", "RADIATION", "PROCEDURE", "DEVICE", "COMBINATION_PRODUCT", "GENETIC",
               "DIETARY_SUPPLEMENT", "BEHAVIORAL", "DIAGNOSTIC_TEST"]


def intervention_phrase(items: list[dict] | None) -> str | None:
    """'pembrolizumab (drug), radiation (radiation) and 2 more'; None when the trial lists none.

    Why: shows what is being tested from the stored API names (rule 2), most treatment-like types
    first so 'Biospecimen Collection' does not crowd out the actual drug. Names are never reworded.
    """
    seen, uniq = set(), []
    for i in items or []:
        name = (i.get("name") or "").strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            uniq.append({"type": i.get("type"), "name": name})
    if not uniq:
        return None
    rank = {t: n for n, t in enumerate(_TYPE_ORDER)}
    uniq.sort(key=lambda i: rank.get(i["type"], len(rank)))   # stable: API order kept within a type
    limit = config.OVERVIEW_MAX_INTERVENTIONS
    shown = [f"{i['name']} ({_TYPE_WORDS[i['type']]})" if i["type"] in _TYPE_WORDS else i["name"]
             for i in uniq[:limit]]
    more = len(uniq) - limit
    if more > 0:
        shown.append(f"{more} more")
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


def phase_words(phase: str | None) -> str | None:
    """'PHASE1|PHASE2' -> 'Phase 1/2'; 'NA' or empty -> None (we say nothing rather than guess)."""
    parts = [p for p in (phase or "").split("|") if p in _PHASE_WORDS]
    if not parts:
        return None
    words = [_PHASE_WORDS[p] for p in parts]
    return re.sub(r"Phase (\d)/Phase (\d)", r"Phase \1/\2", "/".join(words))


def _short_age(age: str) -> str:
    """'18 Years' -> '18'; '6 Months' -> '6 months'."""
    m = re.match(r"^(\d+)\s+Years?$", age)
    return m.group(1) if m else age.lower()


def age_phrase(min_age: str | None, max_age: str | None) -> str:
    if min_age and max_age:
        return f"ages {_short_age(min_age)}–{_short_age(max_age)}"
    if min_age:
        return f"ages {_short_age(min_age)} and older"
    if max_age:
        return f"ages up to {_short_age(max_age)}"
    return "all ages"


def study_overview(study: dict, zip_code: str | None = None) -> dict:
    """study = the get_study() dict. Returns {sentence, facts[{label,value}]}. zip_code (optional) adds the nearest site."""
    phase = phase_words(study.get("phase"))
    status = _STATUS_WORDS.get(study.get("status"), (study.get("status") or "").replace("_", " ").lower())
    who = _SEX_WORDS.get(study.get("sex"), "people")
    ages = age_phrase(study.get("min_age"), study.get("max_age"))
    conds = (study.get("conditions") or [])[:2]
    if len(conds) == 2 and conds[0].lower() in conds[1].lower():   # "Breast Cancer" + "HER2-Positive Breast Cancer": say the specific one
        conds = conds[1:]
    head = f"A {phase} study" if phase else "A study"
    s = head + (f", {status}," if status else "") + f" for {who} {ages}"
    s += f" with {' and '.join(conds)}." if conds else "."
    tested = intervention_phrase(study.get("interventions"))
    if tested:
        s += f" It is testing {tested}."
    sites = study.get("sites") or []
    near = geo.nearest_site(sites, zip_code) if zip_code else None
    tail = []
    if study.get("sponsor"):
        tail.append(f"Run by {study['sponsor']}")
    if sites:
        n = len(sites)
        tail.append(f"{'at' if tail else 'Held at'} {n} site{'' if n == 1 else 's'}")
    sentence = s
    if tail:
        sentence += " " + " ".join(tail)
        if near:
            sentence += f" (nearest: {near['city']}, {near['state']} — {round(near['distance_mi'])} mi)"
        sentence += "."
    facts = [{"label": "Status", "value": status.capitalize()}] if status else []
    if phase:
        facts.append({"label": "Phase", "value": phase})
    facts.append({"label": "Ages", "value": ages})
    facts.append({"label": "Who", "value": "All sexes" if who == "people" else who.capitalize() + " only"})
    if tested:
        facts.append({"label": "Being tested", "value": tested})
    if study.get("enrollment"):
        facts.append({"label": "Planned size", "value": f"{study['enrollment']} people"})
    if sites:
        facts.append({"label": "Sites", "value": str(len(sites))})
    if near:
        facts.append({"label": "Nearest site",
                      "value": f"{near['city']}, {near['state']} — {round(near['distance_mi'])} mi"})
    if study.get("sponsor"):
        facts.append({"label": "Sponsor", "value": study["sponsor"]})
    return {"sentence": sentence, "facts": facts}
