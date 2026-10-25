# Data files

All data here comes from public-domain U.S. government sources or is synthetic
project data. The repository's MIT license applies to the original selection,
cleaning, alignment, and annotations. It does not relicense the underlying
source text, which remains public domain. See [`../NOTICE`](../NOTICE) for full
attribution.

| File(s) | Source | Status / license | Notes |
|---|---|---|---|
| `pdq_pairs.jsonl`, `pdq_pairs.predup.jsonl`, `pdq_pairs_sent.jsonl`, `pdq_pairs_sent.predup.jsonl`, `pdq_pairs_clean.jsonl`, `pdq_pairs_clean_para.jsonl`, `pdq_pairs_combined.jsonl`, `pdq_val_fixed.jsonl` | NCI PDQ | Public domain (U.S. government work); credit: National Cancer Institute | Professional-to-patient text pairs adapted from NCI PDQ (cleaned, split, aligned, filtered). |
| `pdq_pages.json` | NCI PDQ | Public domain (U.S. government work); credit: National Cancer Institute | Page list and fetch URLs only; no page HTML is stored. |
| `pdq_stats.json`, `pdq_clean_stats.json`, `pdq_borderline_samples.json` | Derived from NCI PDQ | Public-domain source; project annotations MIT | Counts and samples from the pair-building and cleaning steps. |
| `cancer_types.json`, `glossary.json`, `umbrella_terms.json` | NCI Dictionary of Cancer Terms / NCI Thesaurus; project mapping | Public domain (U.S. government work); credit: National Cancer Institute | Glossary definitions are plain-language paraphrases. Human proofreading is still pending. |
| `trial_eval_set.jsonl`, `retag_dropped.csv` | ClinicalTrials.gov | Public domain (U.S. government work) | Trial text and titles. Not affiliated with or endorsed by NLM/NIH. |
| `zip_centroids.csv` | U.S. Census Bureau 2024 Gazetteer ZCTA file | Public domain (U.S. government work) | ZIP-code centroid coordinates. |
| `test_queries.json` | Synthetic project data | MIT | No user queries are stored. |

Generated files (`trials.db`, `faiss.index`, `faiss_rowmap.json`, `raw/`) are
git-ignored and are not part of the published data set.
