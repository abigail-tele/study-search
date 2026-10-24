import json
from pathlib import Path

import httpx
import pytest

from backend import config, db
from pipeline import indexer

FIX = Path(__file__).parent / "fixtures"
PAGE1 = json.loads((FIX / "ctgov_page1.json").read_text())
PAGE2 = json.loads((FIX / "ctgov_page2.json").read_text())


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(indexer.time, "sleep", sleeps.append)
    return sleeps


def make_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def paged_handler(requests):
    def handler(req):
        requests.append(req)
        if req.url.params.get("pageToken") == "TOK1":
            return httpx.Response(200, json=PAGE2)
        return httpx.Response(200, json=PAGE1)
    return handler


def test_parse_study():
    rec, locs = indexer.parse_study(PAGE1["studies"][1])
    assert rec["nct_id"] == "NCT00000002"
    assert rec["min_age_years"] == 0.5 and rec["max_age_years"] == 17.0
    assert rec["phases"] == ["PHASE2"] and rec["sponsor"] == "Test Univ"
    assert rec["enrollment"] == 120 and rec["healthy_volunteers"] == "False"
    assert locs[0]["lat"] == 44.02 and locs[1]["lat"] is None


def test_parse_missing_modules():
    rec, locs = indexer.parse_study({"protocolSection": {"identificationModule": {"nctId": "NCT9"}}})
    assert rec["nct_id"] == "NCT9" and rec["max_age_years"] is None and locs == []


def test_paging_follows_token(no_sleep):
    reqs = []
    out = list(indexer.fetch_studies(make_client(paged_handler(reqs)), "breast cancer"))
    assert len(out) == 4 and len(reqs) == 2
    assert reqs[1].url.params["pageToken"] == "TOK1"
    assert no_sleep == [config.CTGOV_SLEEP_SECONDS]
    assert reqs[0].url.params["filter.overallStatus"] == ",".join(config.INDEX_STATUSES)


def test_limit_pages():
    reqs = []
    out = list(indexer.fetch_studies(make_client(paged_handler(reqs)), "x", limit_pages=1))
    assert len(out) == 3 and len(reqs) == 1


def test_incremental_param():
    reqs = []
    list(indexer.fetch_studies(make_client(paged_handler(reqs)), "x", since="2026-10-01", limit_pages=1))
    assert reqs[0].url.params["filter.advanced"] == "AREA[LastUpdatePostDate]RANGE[2026-10-01,MAX]"
    assert "filter.advanced" not in indexer.build_params("x", None, None)


def test_retry_backoff(no_sleep):
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429)
        if len(calls) == 2:
            raise httpx.ReadTimeout("slow", request=req)
        if len(calls) == 3:
            return httpx.Response(503)
        return httpx.Response(200, json=PAGE2)

    data = indexer.get_with_retry(make_client(handler), {})
    assert data == PAGE2 and len(calls) == 4
    b = config.CTGOV_BACKOFF_BASE_SECONDS
    assert no_sleep == [b, 2 * b, 4 * b]


def test_retry_gives_up(no_sleep):
    with pytest.raises(RuntimeError):
        indexer.get_with_retry(make_client(lambda r: httpx.Response(500)), {})
    assert len(no_sleep) == config.CTGOV_MAX_RETRIES + 1


def test_4xx_not_retried(no_sleep):
    with pytest.raises(httpx.HTTPStatusError):
        indexer.get_with_retry(make_client(lambda r: httpx.Response(400)), {})
    assert no_sleep == []


def test_dedupe_across_types(tmp_path):
    types = [{"id": "breast", "ctgov_query": "breast", "lay_terms": ["breast cancer"]},
             {"id": "lung", "ctgov_query": "lung", "lay_terms": ["breast cancer"]}]
    path = tmp_path / "t.db"
    res = indexer.run(types, db_path=path, client=make_client(paged_handler([])))
    assert res["breast"]["new"] == 4 and res["lung"]["new"] == 0 and res["lung"]["updated"] == 4
    conn = db.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM trial_cancer_types").fetchone()[0] == 8
    assert conn.execute("SELECT COUNT(*) FROM fts").fetchone()[0] == 4
    # locations replaced, not duplicated, on the second type
    assert conn.execute("SELECT COUNT(*) FROM locations WHERE nct_id='NCT00000001'").fetchone()[0] == 2
    assert db.get_meta(conn, "last_index_run") is not None


def test_limit_pages_does_not_advance_watermark(tmp_path):
    path = tmp_path / "t.db"
    indexer.run([{"id": "b", "ctgov_query": "b"}], limit_pages=1, db_path=path,
                client=make_client(paged_handler([])))
    assert db.get_meta(db.connect(path), "last_index_run") is None


# --- tag verification / oncology filter / retag / incremental close ---------

REAL_TYPES = json.loads(config.CANCER_TYPES_PATH.read_text(encoding="utf-8"))
BY_ID = {t["id"]: t for t in REAL_TYPES}


def T(conditions, title="Study", keywords=None):
    return {"conditions": conditions, "keywords": keywords or [], "brief_title": title,
            "official_title": None}


def tags_of(trial, hint=None):
    return indexer.assign_tags(trial, REAL_TYPES)


def test_sclc_not_matched_by_non_small_cell():
    nsclc = T(["Non-Small Cell Lung Cancer"])
    assert not indexer.verify_type_match(nsclc, BY_ID["lung_sclc"])
    assert indexer.verify_type_match(nsclc, BY_ID["lung_nsclc"])
    assert indexer.verify_type_match(T(["Extensive Stage Small Cell Lung Cancer"]), BY_ID["lung_sclc"])
    # NSCLC returned for the SCLC query must not fall back to an SCLC tag either
    assert tags_of(T(["NSCLC", "lung carcinoma"]), ["lung_sclc"])[0] == ["lung_nsclc"]
    assert tags_of(T(["Non-small cell carcinoma"]), ["lung_sclc"])[0] == ["lung_nsclc"]


def test_hodgkin_not_matched_by_non_hodgkin():
    nhl = T(["Non-Hodgkin Lymphoma"])
    assert not indexer.verify_type_match(nhl, BY_ID["lymphoma_hodgkin"])
    assert indexer.verify_type_match(nhl, BY_ID["lymphoma_nhl"])
    assert indexer.verify_type_match(T(["Classical Hodgkin Lymphoma"]), BY_ID["lymphoma_hodgkin"])


def test_cervical_cancer_vs_cervical_spine():
    assert indexer.verify_type_match(T(["Cervical Cancer"]), BY_ID["cervical"])
    assert not indexer.verify_type_match(T(["Cervical Spine Fusion"]), BY_ID["cervical"])
    hn = T(["Head and Neck Squamous Cell Carcinoma"], "Cervical lymph node metastases")
    assert tags_of(hn, ["cervical"]) == ([], "no_type_match") or "cervical" not in tags_of(hn)[0]
    assert "head_and_neck" in tags_of(hn)[0]


def test_non_oncology_dropped():
    burns = T(["Burns", "Sepsis"], "Fluids in Pediatric ALL burns", ["pediatric ALL"])
    assert tags_of(burns, ["leukemia_all"]) == ([], "non_oncology")
    assert tags_of(T(["All Conditions"], "A healthy volunteer study"), ["breast"])[1] == "non_oncology"
    assert tags_of(T(["Parkinson Disease"]), ["brain"])[1] == "non_oncology"
    assert not indexer.is_oncology(T(["Uterine Fibroids"]))


def test_all_only_counts_next_to_leukemia():
    assert tags_of(T(["Acute Lymphoblastic Leukemia"]))[0] == ["leukemia_all"]
    assert indexer.is_oncology(T(["ALL"], "Study of AML patients"))
    assert not indexer.is_oncology(T(["aml"], "all"))  # lowercase acronyms do not count


def test_supportive_care_with_cancer_kept():
    trial = T(["Breast Cancer", "Hot Flashes"], "Acupuncture for hot flashes")
    assert tags_of(trial) == (["breast"], None)


def test_fallback_removed_oncology_without_type_is_dropped():
    trial = T(["Malignant Neoplasm"], "Mystery tumor drug")
    assert tags_of(trial) == ([], "no_type_match")


def test_b_t_cell_lymphoma_is_nhl_not_hodgkin():
    for cond in ("B-Cell Lymphoma", "Peripheral T-cell Lymphoma", "NK/T-cell lymphoma", "Lymphoma"):
        assert tags_of(T([cond])) == (["lymphoma_nhl"], None), cond
    assert tags_of(T(["Hodgkin Lymphoma"]))[0] == ["lymphoma_hodgkin"]


def test_hodgkin_possessive():
    assert tags_of(T(["Hodgkin's Disease"]))[0] == ["lymphoma_hodgkin"]
    assert tags_of(T(["Hodgkin’s Lymphoma"]))[0] == ["lymphoma_hodgkin"]


def test_mds_is_aml_not_nhl():
    tags = tags_of(T(["Myelodysplastic Syndromes", "Acute Myeloid Leukemia"]))[0]
    assert "leukemia_aml" in tags and "lymphoma_nhl" not in tags
    assert tags_of(T(["Myelodysplastic Syndrome"]))[0] == ["leukemia_aml"]


def test_tumor_necrosis_factor_not_oncology():
    ibd = T(["Crohn Disease"], "Anti-tumor necrosis factor therapy in IBD")
    assert tags_of(ibd) == ([], "non_oncology")
    assert indexer.is_oncology(T(["Brain Tumor"]))


def test_ependymoma_kept_and_pituitary_adenoma_not_brain():
    assert tags_of(T(["Ependymoma"])) == (["brain"], None)
    assert "brain" not in tags_of(T(["Pituitary Adenoma"], "Cancer center study"))[0]


def _study(nct, conds, status="RECRUITING", title="T"):
    return {"protocolSection": {
        "identificationModule": {"nctId": nct, "briefTitle": title},
        "statusModule": {"overallStatus": status},
        "conditionsModule": {"conditions": conds}}}


def test_index_type_drops_non_oncology_and_unverified(tmp_path):
    studies = [_study("NCT1", ["Breast Cancer"]), _study("NCT2", ["Burns"]),
               _study("NCT3", ["Lung Cancer, Non-Small Cell"])]
    client = make_client(lambda r: httpx.Response(200, json={"studies": studies}))
    path = tmp_path / "t.db"
    res = indexer.run([BY_ID["lung_sclc"]], db_path=path, client=client, all_types=REAL_TYPES)
    conn = db.connect(path)
    got = {r[0]: r[1] for r in conn.execute("SELECT nct_id, cancer_type_id FROM trial_cancer_types")}
    assert got == {"NCT1": "breast", "NCT3": "lung_nsclc"}  # NCT2 dropped; NCT3 not SCLC
    assert res["lung_sclc"]["dropped"] == 1


def test_retag_fixture_db(tmp_path):
    path = tmp_path / "t.db"
    conn = db.connect(path)
    db.init_schema(conn)
    rows = [("NCT1", ["Non-Small Cell Lung Cancer"], "lung_sclc"),   # wrongly tagged SCLC
            ("NCT2", ["Burns"], "lung_sclc"),                       # not cancer
            ("NCT3", ["Breast Cancer"], "breast")]
    for nct, conds, tag in rows:
        db.upsert_trial(conn, {"nct_id": nct, "brief_title": "x", "conditions": conds, "status": "RECRUITING"})
        db.add_cancer_tags(conn, nct, [tag])
        db.replace_locations(conn, nct, [{"city": "A"}])
        conn.execute("INSERT INTO summaries (nct_id) VALUES (?)", (nct,))
    conn.commit()
    conn.close()
    csv_path = tmp_path / "dropped.csv"
    out = indexer.retag(path, REAL_TYPES, csv_path)
    assert out["trials_before"] == 3 and out["trials_after"] == 2 and out["dropped"] == 1
    assert out["before"]["lung_sclc"] == 2 and "lung_sclc" not in out["after"]
    conn = db.connect(path)
    tags = {r[0]: r[1] for r in conn.execute("SELECT nct_id, cancer_type_id FROM trial_cancer_types")}
    assert tags == {"NCT1": "lung_nsclc", "NCT3": "breast"}
    for table in ("locations", "fts", "summaries"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE nct_id='NCT2'").fetchone()[0] == 0
    assert "NCT2,non_oncology" in csv_path.read_text(encoding="utf-8")


def test_incremental_run_has_no_status_filter_and_deletes_closed(tmp_path):
    assert "filter.overallStatus" not in indexer.build_params("x", "2026-10-01", None)
    path = tmp_path / "t.db"
    open_page = {"studies": [_study("NCT1", ["Breast Cancer"]), _study("NCT2", ["Breast Cancer"])]}
    types = [BY_ID["breast"]]
    indexer.run(types, db_path=path, client=make_client(lambda r: httpx.Response(200, json=open_page)),
                all_types=REAL_TYPES)
    closed = {"studies": [_study("NCT1", ["Breast Cancer"], status="COMPLETED"),
                          _study("NCT2", ["Breast Cancer"], status="ACTIVE_NOT_RECRUITING")]}
    seen = []

    def handler(req):
        seen.append(req.url.params)
        return httpx.Response(200, json=closed)

    res = indexer.run(types, since="2026-10-01", db_path=path, client=make_client(handler),
                      all_types=REAL_TYPES)
    assert "filter.overallStatus" not in seen[0] and res["breast"]["removed"] == 1
    conn = db.connect(path)
    assert [r[0] for r in conn.execute("SELECT nct_id FROM trials")] == ["NCT2"]
    assert conn.execute("SELECT status FROM trials").fetchone()[0] == "ACTIVE_NOT_RECRUITING"
    assert conn.execute("SELECT COUNT(*) FROM trial_cancer_types WHERE nct_id='NCT1'").fetchone()[0] == 0


def test_cml_not_aml_and_cll_variant():
    assert tags_of(T(["Chronic Myeloid Leukemia"]))[0] == ["leukemia_cml"]
    assert tags_of(T(["Chronic Myelogenous Leukemia, BCR-ABL Positive"]))[0] == ["leukemia_cml"]
    assert tags_of(T(["Chronic Lymphoid Leukemia"]))[0] == ["leukemia_cll"]
    both = tags_of(T(["Chronic Myeloid Leukemia", "Acute Myeloid Leukemia"]))[0]
    assert set(both) == {"leukemia_cml", "leukemia_aml"}


def test_organ_word_variant_wordings():
    for cond, tid in [("Cancer, Breast", "breast"), ("Cancer of the Breast", "breast"),
                      ("Malignant Neoplasm of Breast", "breast"), ("Cancer, Rectal", "colorectal"),
                      ("Adenocarcinoma of the Colon", "colorectal"), ("Head/Neck Neoplasm", "head_and_neck"),
                      ("Laryngeal Tumors", "head_and_neck"), ("Breast Neoplasm Female", "breast")]:
        assert tid in tags_of(T([cond]))[0], cond


def test_organ_word_alone_is_not_enough():
    assert tags_of(T(["Biliary Drainage"], "Stent trial for cancer patients"))[0] == []
    hpb = tags_of(T(["Hepato-Pancreato-Biliary Cancer"]))[0]
    assert "biliary" not in hpb
    assert "biliary" in tags_of(T(["Biliary Tract Cancer"]))[0]
    assert "biliary" not in tags_of(T(["Pancreatobiliary Neoplasm"]))[0]
    assert "oral" not in str(tags_of(T(["Solid Tumors"], "Oral chemotherapy for cancer")))


def test_non_colorectal_and_hyphen_hodgkin():
    assert "colorectal" not in tags_of(T(["Advanced Non-Colorectal Gastrointestinal Cancer"]))[0]
    assert "lymphoma_nhl" not in tags_of(T(["Hodgkin-lymphoma"]))[0]
    assert "lymphoma_nhl" not in tags_of(T(["Hodgkin's lymphoma"]))[0]
