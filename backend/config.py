"""Central settings for the whole project.

Why one file: thresholds and paths change while we tune the app, and judges
should be able to see every "magic number" in one place. No other module
should hard-code a path, threshold, or k value.
"""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"  # cached downloads (API pages, PDQ HTML); git-ignored
DB_PATH = DATA_DIR / "trials.db"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
FAISS_ROWMAP_PATH = DATA_DIR / "faiss_rowmap.json"
CANCER_TYPES_PATH = DATA_DIR / "cancer_types.json"
UMBRELLA_TERMS_PATH = DATA_DIR / "umbrella_terms.json"  # "lung cancer" -> [type ids]; detection must never match bare "all"
RETAG_DROPPED_PATH = DATA_DIR / "retag_dropped.csv"  # trials removed by --retag, for human spot-check
ZIP_CENTROIDS_PATH = DATA_DIR / "zip_centroids.csv"
TEST_QUERIES_PATH = DATA_DIR / "test_queries.json"
PDQ_PAIRS_PATH = DATA_DIR / "pdq_pairs.jsonl"
PDQ_STATS_PATH = DATA_DIR / "pdq_stats.json"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
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
    "LocationCountry,LocationStatus,LocationGeoPoint"
)
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
RANDOM_SEED = 42

# --- Model -----------------------------------------------------------------
BASE_MODEL_NAME = "google/flan-t5-base"
TASK_PREFIX = "simplify: "
MAX_INPUT_TOKENS = 512
MAX_OUTPUT_TOKENS = 256
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 3

# --- UI text ---------------------------------------------------------------
DISCLAIMER = "For education, not medical advice."
