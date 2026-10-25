"""Central settings for the whole project.

Why one file: thresholds and paths change while we tune the app, and judges
should be able to see every "magic number" in one place. No other module
should hard-code a path, threshold, or k value.
"""
import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"  # cached downloads (API pages, PDQ HTML); git-ignored
DEFAULT_DB_PATH = DATA_DIR / "trials.db"      # full offline database (indexer/pipeline write here)
SHIP_DB_PATH = DATA_DIR / "trials_ship.db"    # slim read-only copy built by pipeline/make_ship_db.py for deployment
DB_PATH = Path(os.environ.get("STUDY_SEARCH_DB", str(DEFAULT_DB_PATH)))  # deployment points this at the ship DB
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
FAISS_ROWMAP_PATH = DATA_DIR / "faiss_rowmap.json"
CANCER_TYPES_PATH = DATA_DIR / "cancer_types.json"
UMBRELLA_TERMS_PATH = DATA_DIR / "umbrella_terms.json"  # "lung cancer" -> [type ids]; detection must never match bare "all"
RETAG_DROPPED_PATH = DATA_DIR / "retag_dropped.csv"  # trials removed by --retag, for human spot-check
ZIP_CENTROIDS_PATH = DATA_DIR / "zip_centroids.csv"
TEST_QUERIES_PATH = DATA_DIR / "test_queries.json"
PDQ_PAIRS_PATH = DATA_DIR / "pdq_pairs.jsonl"
PDQ_STATS_PATH = DATA_DIR / "pdq_stats.json"
PDQ_SENT_PAIRS_PATH = DATA_DIR / "pdq_pairs_sent.jsonl"   # sentence-level pairs (extra training data)
PDQ_COMBINED_PAIRS_PATH = DATA_DIR / "pdq_pairs_combined.jsonl"  # paragraph + sentence pairs, deduped together
PDQ_VAL_FIXED_PATH = DATA_DIR / "pdq_val_fixed.jsonl"     # frozen val set used to compare checkpoints
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
MODELS_DIR = ROOT_DIR / "models"  # downloaded once by pipeline/fetch_models.py; git-ignored
EMBED_MODEL_DIR = MODELS_DIR / "all-MiniLM-L6-v2"  # the ONLY place the live app loads MiniLM from (rule 4)
FRONTEND_DIR = ROOT_DIR / "frontend"
MOCK_DIR = FRONTEND_DIR / "mock"

# --- ClinicalTrials.gov v2 API (offline indexer only; never at request time) -
CTGOV_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
CTGOV_PAGE_SIZE = 1000          # API maximum per page
CTGOV_SLEEP_SECONDS = 1.0       # pause between pages to be polite
CTGOV_MAX_RETRIES = 5           # retries with exponential backoff
CTGOV_TIMEOUT_SECONDS = 60.0
STUDY_URL_TEMPLATE = "https://clinicaltrials.gov/study/{nct}"
CTGOV_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 CancerTrialsFinder-indexer"
)
CTGOV_BACKOFF_BASE_SECONDS = 2.0  # wait = base * 2**attempt
# Statuses worth showing to a patient: trials that are open or about to open,
# or running but closed to new people (still useful to read about). Completed,
# terminated and withdrawn trials can't be joined, so we skip them.
INDEX_STATUSES = (
    "RECRUITING",
    "NOT_YET_RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "ENROLLING_BY_INVITATION",
)
CTGOV_FIELDS = (
    "NCTId,BriefTitle,OfficialTitle,OverallStatus,StartDate,LastUpdatePostDate,"
    "Phase,EnrollmentCount,Condition,Keyword,LeadSponsorName,BriefSummary,"
    "DetailedDescription,EligibilityCriteria,HealthyVolunteers,Sex,MinimumAge,"
    "MaximumAge,LocationFacility,LocationCity,LocationState,LocationZip,"
    "LocationCountry,LocationStatus,LocationGeoPoint,InterventionType,InterventionName"
)
CTGOV_BACKFILL_BATCH = 200       # NCT ids per filter.ids request (URL stays short)
INTERVENTION_FIELDS = "NCTId,InterventionType,InterventionName"
OVERVIEW_MAX_INTERVENTIONS = 3  # names shown in the overview sentence before "and N more"
# Rule 1 (cancer only) post-filter. A trial is oncology if its conditions,
# keywords or titles match this (case-insensitive) OR the acronym list
# (case-sensitive, so "all" / "aml" in ordinary prose never counts). The word
# "ALL" is deliberately NOT an acronym here: it only counts next to
# "leukemia"/"lymphoblastic", and those words already match the first regex.
ONCOLOGY_REGEX = (
    r"cancer|carcinom|neoplas|tumou?r(?! necrosis factor)|malignan|leuka?emi|lymphom|myelom|sarcom|"
    r"melanom|gliom|ependymom|paget|blastom|metasta|mesotheliom|seminom|oncolog|chemotherap|"
    r"radiotherap|mastectomy|precancer|intraepithelial neoplasia"
)
ONCOLOGY_ACRONYMS_REGEX = (
    r"\b(?:NSCLC|SCLC|AML|CLL|CML|HCC|RCC|CRPC|TNBC|DLBCL|HNSCC|GIST|NMIBC|MIBC)\b"
)
# Used with each cancer type's `organ_words` (data/cancer_types.json) to catch
# variant wordings: "breast cancer", "cancer of the breast", "Cancer, Breast",
# "malignant neoplasm of breast". The organ word alone never counts.
CANCER_NOUN_REGEX = (
    r"(?:cancers?|carcinomas?|adenocarcinomas?|neoplasms?|neoplasias?|tumou?rs?|"
    r"malignan\w*|sarcomas?|oncolog\w*)"
)
ZIP_GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2024_Gazetteer/2024_Gaz_zcta_national.zip"
)
EARTH_RADIUS_MILES = 3958.8

# --- Search ----------------------------------------------------------------
FTS_TOP_K = 50                  # keyword (FTS5) candidates
FAISS_TOP_K = 50                # meaning (embedding) candidates
RRF_K = 60                      # reciprocal rank fusion constant
RESULT_LIMIT = 10               # cards shown to the user
MAX_CANCER_TAGS_ON_CARD = 3
DEFAULT_STATUS = "RECRUITING"
RADIUS_OPTIONS_MI = (25, 50, 100, 250)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TYPE_DETECT_FUZZY_THRESHOLD = 85      # rapid/fuzzy string match score (0-100)
TYPE_DETECT_EMBED_THRESHOLD = 0.55    # cosine similarity for embedding fallback
# The embedding fallback only runs if the query also contains one of these cancer cues. Why: MiniLM scores
# "stomach ache" 0.63 against "stomach cancer" (same organ), so similarity alone cannot tell a symptom from a cancer.
TYPE_DETECT_CUE_REGEX = (
    r"cancer|tumou?r|carcinom|neoplas|malignan|leuka?emi|lymphom|myelom|sarcom|melanom|gliom|blastom|"
    r"metasta|oncolog|chemo|radiation|lump|growth|stage \d|\bstage (?:i|ii|iii|iv)\b|"
    r"\b(?:egfr|her2|kras|braf|alk|brca\d?|psa|nsclc|sclc)\b"
)

# --- PDQ training data -----------------------------------------------------
PDQ_ALIGN_SIM_THRESHOLD = 0.60  # min cosine similarity to keep a paragraph pair
PDQ_MAX_LEN_RATIO = 1.5         # drop pair if plain text > 1.5x professional text
PDQ_MIN_WORDS = 8               # drop tiny fragments
PDQ_SPLIT = (0.8, 0.1, 0.1)     # train/val/test, split by page (no leakage)
PDQ_SLEEP_SECONDS = 1.0
PDQ_MAX_RETRIES = 4             # retries with exponential backoff per page
PDQ_TIMEOUT_SECONDS = 60.0
PDQ_RAW_DIR = RAW_DIR / "pdq"   # cached PDQ HTML (git-ignored)
PDQ_PAGES_PATH = DATA_DIR / "pdq_pages.json"  # which cancer types/pages to scrape
PDQ_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
PDQ_HEADING_MATCH_THRESHOLD = 0.70  # min fuzzy ratio (0-1) to pair HP and patient headings
PDQ_MAX_SOURCE_WORDS = 300      # drop professional paragraphs too long for the 512-token input
PDQ_FALLBACK_SIM_THRESHOLD = 0.70  # stricter bar for page-wide matches (no heading evidence)
PDQ_MAX_WINDOW = 3              # join up to 3 CONSECUTIVE professional paragraphs per source
PDQ_WINDOW_MIN_GAIN = 0.02      # a joined window must beat its single paragraphs by this much
PDQ_BORDERLINE_PATH = DATA_DIR / "pdq_borderline_samples.json"  # pairs below the default threshold, for human review
PDQ_BORDERLINE_COUNT = 3        # samples written per similarity band
PDQ_BORDERLINE_BANDS = ((0.60, 0.65), (0.65, 0.70))  # bands a human reviews when choosing the threshold
PDQ_NEAR_DUP_RATIO = 0.85       # val/test target this similar (difflib) to a train target is dropped (leakage)
PDQ_XREF_PATTERNS = (           # target sentences that point elsewhere ("see the ... section") add no teachable content
    r"\blearn about\b", r"\bsee (?:the|our|this|these)\b", r"\bfor more information\b",
    r"\bbooklets?\b", r"\bvisit\b", r"\bsee [^.]{0,80}\bsections?\b", r"\bcall(?:ing)? (?:the )?cancer information\b",
    r"^\s*(?:learn|read|find out) more\b", r"\bto learn more\b", r"\bfind out more\b", r"\bvisit the [^.]{0,60}\b(?:page|website|site)\b",
)
PDQ_MAX_ALIGN_SIM = 0.97        # near-identical pair = copied boilerplate, teaches nothing
PDQ_SKIP_HEADINGS = (           # sections with no teachable content (boilerplate/admin)
    "about this pdq summary", "latest updates", "references", "to learn more",
    "current clinical trials", "key points", "about pdq", "purpose of this summary",
    "reviewers and updates", "clinical trial information", "permission to use this summary",
    "disclaimer", "contact us", "levels of evidence", "changes to this summary",
)
PDQ_SENT_SIM_THRESHOLD = 0.70   # min cosine similarity to keep a SENTENCE-level pair
PDQ_SENT_MIN_WORDS = 6          # sentences are short; lower floor than PDQ_MIN_WORDS
PDQ_GENERIC_TARGET_PATTERNS = (  # generic PDQ intro/definition sentences; removed from targets unless the source has them
    r"\bis a disease in which\b", r"\bare diseases? in which\b", r"\bthere are different types of\b",
    r"\bare different types of\b", r"\bis a type of cancer that (?:forms|starts|begins)\b",
    r"\bis a cancer that (?:forms|starts|begins)\b", r"\bsigns? (?:and symptoms )?of [^.]{0,60}\binclude\b",
    r"\b(?:is|are) (?:a|an) (?:kind|type) of (?:cancer|tumor)\b", r"\bcancer is (?:a )?disease\b",
)
PDQ_CLEAN_SUPPORT_SIM = 0.45     # target sentence needs this MiniLM cosine to a source window (patient text uses different words, so no word-overlap test)
PDQ_CLEAN_WORD_FRACTION = 0.0   # word-overlap test is off when cleaning targets (calibrated: 0.7 would drop 2/3 of sentences)
PDQ_SENT_MAX_WINDOW = 3        # professional side may be 1-3 consecutive sentences
PDQ_COMBINED_MIN_PARA_SIM = 0.70  # paragraph pairs at least this similar are what the combined run trains on
PDQ_SENT_PATIENT_WINDOW = 2     # patient side: single sentence or 2-sentence window
RANDOM_SEED = 42

# --- Evaluation (pipeline/evaluate.py) ---------------------------------------
EVAL_RETRIEVAL_CSV = ROOT_DIR / "results" / "eval_retrieval.csv"
EVAL_JSON = ROOT_DIR / "results" / "eval_summary.json"
EVAL_DOC_PATH = ROOT_DIR / "docs" / "EVAL.md"
BATCH_SUMMARY_STATS_PATH = DATA_DIR / "batch_summary_stats.json"  # written by pipeline/batch_summarize.py
EVAL_TOP_K = 10                 # precision / recall depth
EVAL_HIT_K = 5                  # hit@k depth
EVAL_LATENCY_REPEATS = 5        # timed runs per query per path (after one warm-up pass)
EVAL_LATENCY_ZIP = "10001"      # ZIP used for the "with ZIP filter" latency rows
EVAL_LATENCY_RADIUS_MI = 100

# --- Model -----------------------------------------------------------------
BASE_MODEL_NAME = "google/flan-t5-base"
TASK_PREFIX = "simplify: "
MAX_INPUT_TOKENS = 512
MAX_OUTPUT_TOKENS = 256
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 3

# --- Training (pipeline/train.py) --------------------------------------------
TRAIN_MIN_SIMILARITY = 0.60     # only pairs at least this well aligned are used for train/val
TRAIN_LEARNING_RATE = 3e-4      # standard for fine-tuning T5-family models with AdamW
TRAIN_BATCH_SIZE = 8            # examples per forward pass (lower if the GPU runs out of memory)
TRAIN_GRAD_ACCUM = 2            # batches added up before each weight update (effective batch = 16)
TRAIN_EPOCHS = 5                # max passes over the training pairs
TRAIN_EARLY_STOP_PATIENCE = 2   # stop after this many epochs with no val-loss improvement
TRAIN_WARMUP_RATIO = 0.06       # fraction of steps spent ramping the learning rate up
TRAIN_WEIGHT_DECAY = 0.01       # mild regularisation against memorising the pairs
TRAIN_MAX_GRAD_NORM = 1.0       # clip gradients so one odd batch cannot wreck the weights
TRAIN_CHECKPOINT_NAME = "flan-t5-pdq"  # folder under CHECKPOINT_DIR for the best model
TRAIN_BEST_CHECKPOINT_NAME = "flan-t5-pdq-best"  # winner of the experiments in docs/EVAL.md

# --- Summarizer (backend/summarize.py) ---------------------------------------
SUMMARY_MODEL_DIR = CHECKPOINT_DIR / TRAIN_BEST_CHECKPOINT_NAME      # preferred checkpoint
SUMMARY_FALLBACK_MODEL_DIR = CHECKPOINT_DIR / TRAIN_CHECKPOINT_NAME  # used until the "best" one exists
DRUG_LEXICON_PATH = DATA_DIR / "drug_lexicon.txt"   # drug names the suffix rules miss (faithfulness_check)
GLOSSARY_PATH = DATA_DIR / "glossary.json"   # NCI Dictionary of Cancer Terms paraphrases (fallback wording)
SUMMARY_BATCH_SIZE = 8              # chunks sent through the model at once (CPU-friendly)
SUMMARY_FALLBACK_TOKENS_PER_WORD = 1.4  # token estimate when no tokenizer is loaded (tests only)
SUMMARY_MAX_CRITERIA_PER_SIDE = 25  # criteria beyond this are shown in original (glossary-swapped) wording
SUMMARY_MIN_CRITERION_WORDS = 3     # shorter fragments ("None") are not worth a model call
# --- Batch summarizer (pipeline/batch_summarize.py) ---------------------------
BATCH_GEN_PIECES = 128              # pieces per GPU generate() call (sorted by length so padding stays small)
BATCH_DTYPE = "bfloat16"            # GPU precision for the offline run (the live CPU path stays float32)
BATCH_TRIALS_PER_COMMIT = 100       # trials generated, then committed to SQLite together (also the resume granularity)
BATCH_STATUS_ORDER = ("RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION")  # generation order; unlisted statuses go last
BATCH_DB_WAIT_SECONDS = 600         # wait this long for a locked trials.db (the API and eval scripts read it too)
BATCH_STATS_PATH = DATA_DIR / "batch_summary_stats.json"   # per-check pass/fallback counts from the check stage
BATCH_GUARD_SAMPLE_TRIALS = 1500    # trials used to measure the guard-only pass rate (it embeds sentences, so it is sampled)
READING_LEVEL_DECIMALS = 1
SUMMARY_MIN_KEPT_FRACTION = 0.5     # if unsupported/boilerplate sentences are dropped, keep the rest only when at least this share of the output survives

# --- Semantic support check (backend/support.py, pipeline/baselines.py) -------
SUPPORT_SIM_THRESHOLD = 0.60        # min MiniLM cosine between an output sentence and a source sentence/window
SUPPORT_WORD_FRACTION = 0.70        # min share of an output sentence's content words found in source or glossary
SUPPORT_MIN_CONTENT_WORDS = 2       # shorter sentences are judged on similarity alone
SUPPORT_WINDOW = 2                  # source windows of 1..N consecutive sentences
BOILERPLATE_OPENERS = (             # generic-claim / textbook-definition phrases: an output sentence containing one is
    "there are different types of", "are different types of",   # DROPPED unless the exact phrase is also in the source
    "is a disease in which", "is a type of cancer that forms", "is a cancer that forms",
    "more likely to", "less likely to", "is a type of", "is a kind of", "are a group of",
    "is used to treat", "may cause", "can cause", "most common", "is called", "forms in",
    "signs of", "symptoms of",
)
RISK_WORDS = (                      # words that turn an eligibility rule into a medical claim (rule 7); a sentence that
    "likely", "risk", "cause", "common", "survival", "cure", "better", "worse", "effective", "safe",
)                                   # uses one the source does not use is dropped
DEFINITION_RE = r"^\s*[\w\s\-\(\)]{1,60}\b(?:is|are)\s+(?:a|an|the)\s+\w+(?:\s+\w+){0,3}\s+(?:that|which|who)\b"
DEFINITION_VERBS = ("may include", "can include", "includes")  # 'X may include Y' is also a definition-style claim
DEFINITION_ABSENT_FRACTION = 0.50   # drop a definitional sentence when MORE than this share of its predicate's content words are absent from the source
# direction / negation check (support.direction_check): comparator phrases that sit next to a number
BOUND_LOWER_PREFIX = ("greater than or equal to", "at least", "no less than", "not less than", "minimum of", "minimum",
                      "more than", "greater than", "older than", "over", "above", "≥", ">=", ">")
BOUND_UPPER_PREFIX = ("less than or equal to", "at most", "no more than", "not more than", "maximum of", "maximum",
                      "up to", "less than", "fewer than", "younger than", "under", "below", "≤", "<=", "<")
BOUND_LOWER_SUFFIX = ("or older", "and older", "or more", "or above", "and above", "or greater", "or higher", "or longer")
BOUND_UPPER_SUFFIX = ("or younger", "or less", "or below", "or under", "or fewer", "or lower", "or shorter")
NEGATION_CUES = ("no", "not", "without", "never", "none", "cannot", "excluded", "excluding", "exclude", "excludes",
                 "ineligible", "free of", "absence of", "lack of")
NEGATION_NEUTRAL_PHRASES = ("no more than", "not more than", "no less than", "not less than", "no longer than",
                            "no later than", "not limited to", "not only", "not otherwise specified")  # 'not' here does not negate the rule
DIRECTION_MIN_SHARED_WORDS = 2      # a source clause "says the same thing" when it shares this many content words (or all of a shorter output clause)
DIRECTION_MIN_SHARED_FRACTION = 0.5 # ...and at least this share of the output clause's content words
DIRECTION_MIN_SOURCE_FRACTION = 0.4 # ...and at least this share of the SOURCE clause's content words (a long clause with one 'not' somewhere is not 'the same statement')
TAUTOLOGY_CONTENT_WORDS = 3        # "X is a type of X": the first N content words of the subject must reappear in the predicate
RESULTS_DIR = ROOT_DIR / "results"
TRIAL_EVAL_PATH = DATA_DIR / "trial_eval_set.jsonl"
TRIAL_EVAL_COUNT = 40               # real trials in the trial-domain eval set
TRIAL_EVAL_CRITERIA = 5             # eligibility criteria per trial
TRIAL_EVAL_MIN_TYPES = 10           # trials must span at least this many cancer types
TRIAL_EVAL_MAX_SUMMARY_WORDS = 200  # brief summaries are cut at a sentence boundary near this length
FEWSHOT_K = 3                       # shots for the few-shot baseline
FEWSHOT_MAX_SOURCE_WORDS = 60       # shots are drawn from short train pairs

# --- UI text ---------------------------------------------------------------
DISCLAIMER = "For education, not medical advice."


# --- "About you" matching (backend/match.py) ---------------------------------
MATCH_SIM_THRESHOLD = 0.50      # min MiniLM cosine between a user's term and a criterion (starting guess; tune on real trials)
MATCH_MAX_TERMS = 10            # most condition terms we look at
MATCH_MAX_TERM_CHARS = 60       # longer terms are cut to this
MATCH_MAX_CONDITIONS_CHARS = 600  # API limit for the free-text box
MATCH_MAX_PER_TERM = 3          # most criteria listed per term per side
MATCH_CRITERION_MAX_CHARS = 400  # longer criteria are phrase-matched only (a paragraph's vector says little)
MATCH_CACHE_SIZE = 256          # trials whose criterion vectors we keep in memory (public text only, never user text)

# --- In-domain ClinicalTrials.gov training pairs (pipeline/ctgov_pairs.py) -----
CTGOV_PAIRS_PATH = DATA_DIR / "ctgov_pairs.jsonl"
CTGOV_PAIRS_STATS_PATH = DATA_DIR / "ctgov_pairs_stats.json"
CTGOV_PAIR_SIM = 0.70           # min MiniLM cosine between a brief-summary piece and its detailed-description window
CTGOV_MIN_FK_DROP = 2.0         # target must read at least this many grade levels easier than the source
CTGOV_MAX_LEN_RATIO = 1.5       # target words <= ratio * source words
CTGOV_MAX_WINDOW = 3            # detailed-description side: 1-3 consecutive sentences
CTGOV_TARGET_WINDOW = 2         # brief-summary side: 1 sentence or a 2-sentence window
CTGOV_MIN_TARGET_WORDS = 8
CTGOV_MAX_SOURCE_WORDS = 300
CTGOV_VAL_FRACTION = 0.10       # split by NCT id, so no trial is in both
