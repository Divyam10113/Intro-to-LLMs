"""Central configuration for the QUAL-SG implementation.

Everything that a reader needs to reproduce a run lives here: paths, model
names, API endpoints, re-ranking weights and evaluation thresholds.  Values can
be overridden with environment variables (see ``.env.example``).

No secret is ever printed by this module.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent          # .../Paper implementation
PKG_DIR = Path(__file__).resolve().parent                  # .../Paper implementation/qual_sg

#: Optional per-run output folder.  When set, every output, intermediate file,
#: log and cache of the run is written under it instead of the shared default
#: locations, so runs never overwrite each other.
_RUN_DIR_ENV = os.environ.get("QUAL_SG_RUN_DIR", "").strip()
RUN_DIR = (Path(_RUN_DIR_ENV) if Path(_RUN_DIR_ENV).is_absolute() else BASE_DIR / _RUN_DIR_ENV) if _RUN_DIR_ENV else None
OUT_DIR = RUN_DIR or BASE_DIR
INTERMEDIATES_DIR = (RUN_DIR / "intermediates") if RUN_DIR else PKG_DIR / "intermediates"
LOG_DIR = INTERMEDIATES_DIR / "logs"
CACHE_DIR = INTERMEDIATES_DIR / "cache"

TOPIC_CSV = Path(os.environ.get("QUAL_SG_TOPIC_CSV", BASE_DIR / "example" / "topic.csv"))

# Canonical outputs (kept at the same paths the original pipeline used, so that
# downstream tooling keeps working).
CANONICAL_REF_SET = OUT_DIR / "final_ref_set.json"        # step2 output
CANONICAL_RANKED_CSV = Path(
    os.environ.get("QUAL_SG_RANKED_CSV", OUT_DIR / "paper_after_reranking.csv")
)  # step3 output
CANONICAL_OUTLINE = OUT_DIR / "survey_outline.json"       # step4 output
CANONICAL_CONTENT = OUT_DIR / "survey_content.json"       # step5 output
CANONICAL_SECTIONS_DIR = OUT_DIR / "sections"             # step5 output
CANONICAL_EVALUATION = OUT_DIR / "evaluation_report.json"  # step6 output
CANONICAL_TEX = OUT_DIR / "literature_review.tex"         # step5 human-readable
CANONICAL_BIBLIOGRAPHY = OUT_DIR / "bibliography.json"    # selected refs with full metadata

# Intermediate artefacts (never overwritten by hand; good for auditing).
STEP1_CANDIDATES = INTERMEDIATES_DIR / "step1_candidates.json"
STEP2_COCITATION = INTERMEDIATES_DIR / "step2_cocitation_stats.json"
STEP3_SCORES = INTERMEDIATES_DIR / "step3_relevance_scores.jsonl"
STEP3_EMBEDDINGS = INTERMEDIATES_DIR / "step3_embeddings.npz"
STEP3_RANKINGS = INTERMEDIATES_DIR / "step3_rankings.json"
STEP4_RAW = INTERMEDIATES_DIR / "step4_outline_raw.txt"
STEP6_REPORT = INTERMEDIATES_DIR / "step6_metrics.json"

# Optional gold-standard survey (for the evaluation stage).  The repository only
# ships a gold standard for the example topic; evaluation degrades gracefully.
GOLD_JSON = Path(
    os.environ.get(
        "QUAL_SG_GOLD_JSON",
        BASE_DIR / "example" / "Gold_Standard_Deep Learning Applications in Computer Vision.json",
    )
)

# --------------------------------------------------------------------------- #
# LLM provider
# --------------------------------------------------------------------------- #
# We use Groq (https://console.groq.com) because it serves the strongest
# *open-weights* models behind a free, OpenAI-compatible endpoint.
LLM_BASE_URL = os.environ.get("QUAL_SG_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY_ENV = "GROQ_API_KEY"

# Model used for each stage.  ``openai/gpt-oss-120b`` is the strongest open-weight
# model on Groq (120B MoE); ``llama-3.3-70b-versatile`` is the fallback.
MODEL_JUDGE = os.environ.get("QUAL_SG_JUDGE_MODEL", "openai/gpt-oss-120b")
MODEL_OUTLINE = os.environ.get("QUAL_SG_OUTLINE_MODEL", "openai/gpt-oss-120b")
MODEL_WRITER = os.environ.get("QUAL_SG_WRITER_MODEL", "openai/gpt-oss-120b")
MODEL_EVAL = os.environ.get("QUAL_SG_EVAL_MODEL", "openai/gpt-oss-120b")
MODEL_FALLBACKS = [
    m.strip()
    for m in os.environ.get(
        "QUAL_SG_MODEL_FALLBACKS", "llama-3.3-70b-versatile,openai/gpt-oss-20b"
    ).split(",")
    if m.strip()
]

#: "openai" (any OpenAI-compatible endpoint, Groq by default) or "file": every
#: call is written to ``LLM_EXCHANGE_DIR/pending`` and answered out of band (here,
#: by Claude models in the operator's session); answers are read back from
#: ``LLM_EXCHANGE_DIR/answers`` and cached like API responses.
LLM_BACKEND = os.environ.get("QUAL_SG_BACKEND", "openai").strip().lower()
LLM_EXCHANGE_DIR = INTERMEDIATES_DIR / "llm_exchange"

LLM_TEMPERATURE = float(os.environ.get("QUAL_SG_TEMPERATURE", "0"))
LLM_MAX_RETRIES = int(os.environ.get("QUAL_SG_MAX_RETRIES", "8"))
LLM_REQUEST_TIMEOUT = int(os.environ.get("QUAL_SG_REQUEST_TIMEOUT", "120"))
# Free-tier Groq quotas are per-minute; a small concurrency + spacing is safest.
LLM_CONCURRENCY = int(os.environ.get("QUAL_SG_CONCURRENCY", "3"))
LLM_MIN_INTERVAL = float(os.environ.get("QUAL_SG_MIN_INTERVAL", "2.0"))

# --------------------------------------------------------------------------- #
# Literature APIs
# --------------------------------------------------------------------------- #
OPENALEX_BASE = "https://api.openalex.org"
OPENALEX_MAILTO = os.environ.get("QUAL_SG_MAILTO", "surveygen.pipeline@gmail.com")
S2_BASE = "https://api.semanticscholar.org/graph/v1"
# Semantic Scholar often rate-limits anonymous traffic; OpenAlex is the primary
# source and S2 is only attempted first when this flag is set.
USE_SEMANTIC_SCHOLAR = os.environ.get("QUAL_SG_USE_S2", "0") == "1"
S2_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
#: arXiv is a free, key-less source that is very strong for CS/ML topics; it is
#: merged with OpenAlex before the Eq.(1) embedding filter.
USE_ARXIV = os.environ.get("QUAL_SG_USE_ARXIV", "1") == "1"

HTTP_MIN_INTERVAL = float(os.environ.get("QUAL_SG_HTTP_MIN_INTERVAL", "0.7"))
HTTP_MAX_RETRIES = int(os.environ.get("QUAL_SG_HTTP_MAX_RETRIES", "6"))
HTTP_TIMEOUT = int(os.environ.get("QUAL_SG_HTTP_TIMEOUT", "40"))
#: Every successful API response is cached on disk, so reruns reuse it instead of
#: spending the provider's daily budget again.
HTTP_CACHE = os.environ.get("QUAL_SG_HTTP_CACHE", "1") == "1"
HTTP_CACHE_DIR = CACHE_DIR / "http"
#: Step 2 stops the run when fewer than this share of retrieved candidates end up
#: with a reference list (OpenAlex plus fallbacks): an empty citation graph would
#: silently disable co-citation and snowball expansion (what happened in run v4).
MIN_REFERENCE_COVERAGE = float(os.environ.get("QUAL_SG_MIN_REF_COVERAGE", "0.5"))
OPENCITATIONS_BASE = "https://opencitations.net/index/api/v2"

# --------------------------------------------------------------------------- #
# Retrieval / re-ranking hyper-parameters
# --------------------------------------------------------------------------- #
CANDIDATE_POOL = int(os.environ.get("QUAL_SG_CANDIDATE_POOL", "300"))   # n in the paper
#: Raw hits pulled before the Eq.(1) embedding filter trims down to CANDIDATE_POOL.
#: Must be >= CANDIDATE_POOL; a smaller value disables the filter.
RETRIEVAL_FETCH = int(os.environ.get("QUAL_SG_RETRIEVAL_FETCH", "600"))
CO_CITATION_MIN = int(os.environ.get("QUAL_SG_COCITATION_MIN", "2"))    # cited by >= 2 of D
CO_CITATION_TOPN = int(os.environ.get("QUAL_SG_COCITATION_TOPN", "50"))
N_BINS = 4                                                              # quartile bins Q1..Q4

# --- v2 improvements (generic; see IMPLEMENTATION_NOTES.md §7) -------------- #
#: Upper bound of the publication window.  "now" (default) uses the current year,
#: so a survey can cite work published after the topic's nominal year; "topic"
#: restores the original behaviour.  Recency matters for a survey, and the
#: topic-year cut was silently truncating the candidate pool.
YEAR_MODE = os.environ.get("QUAL_SG_YEAR_MODE", "now")
#: Number of LLM-generated search queries used to widen recall (0 = keywords only).
QUERY_EXPANSION_N = int(os.environ.get("QUAL_SG_QUERY_EXPANSION", "6"))
#: Backward-citation expansion: pull the reference lists of the N most
#: topically-similar retrieved papers (bibliography "snowballing").
EXPAND_FROM_TOP = int(os.environ.get("QUAL_SG_EXPAND_FROM_TOP", "30"))
EXPANSION_PER_SEED = int(os.environ.get("QUAL_SG_EXPANSION_PER_SEED", "12"))
#: Cap on how many snowballed references are fetched (keeps API cost bounded).
SNOWBALL_MAX = int(os.environ.get("QUAL_SG_SNOWBALL_MAX", "400"))
#: OpenAlex is the richest source but the strictest about volume; the query bank
#: is truncated to this many queries for OpenAlex and sent in full to Crossref
#: and arXiv, which are far more permissive.
OPENALEX_QUERY_LIMIT = int(os.environ.get("QUAL_SG_OA_QUERY_LIMIT", "4"))
ARXIV_QUERY_LIMIT = int(os.environ.get("QUAL_SG_ARXIV_QUERY_LIMIT", "8"))
CROSSREF_PER_QUERY = int(os.environ.get("QUAL_SG_CROSSREF_PER_QUERY", "50"))
EUROPEPMC_PER_QUERY = int(os.environ.get("QUAL_SG_EPMC_PER_QUERY", "50"))
#: Optional anchor paper (DOI or OpenAlex id).  When set, its abstract widens the
#: query and its bibliography is added to the candidate pool.  This makes any
#: user-supplied paper a retrieval seed, without special-casing it.
ANCHOR_DOI = os.environ.get("QUAL_SG_ANCHOR_DOI", "").strip()
#: Anchor paper as a PDF path (alternative to ANCHOR_DOI).  Its title/abstract
#: sharpen the topic query and its parsed bibliography joins the candidate pool.
ANCHOR_PDF = os.environ.get("QUAL_SG_ANCHOR_PDF", "").strip()
ANCHOR_BIBLIOGRAPHY = os.environ.get("QUAL_SG_ANCHOR_BIB", "1") == "1"
#: Safety cap on how many of the anchor's bibliography entries enter the pool.
ANCHOR_MAX_REFS = int(os.environ.get("QUAL_SG_ANCHOR_MAX_REFS", "120"))
#: How strongly papers cited by the anchor are prioritised during selection:
#: "off" = ignore provenance, "boost" = exempt from the relevance gate + MMR prior
#: (default), "strict" = fill with the anchor's bibliography first, then MMR.
ANCHOR_REF_PRIORITY = os.environ.get("QUAL_SG_ANCHOR_PRIORITY", "boost")
ANCHOR_PRIOR_SCORE = float(os.environ.get("QUAL_SG_ANCHOR_PRIOR_SCORE", "0.85"))
#: Impact term: normalise citations by paper age (paper Appendix C fn.15).
CITE_AGE_NORMALIZE = os.environ.get("QUAL_SG_CITE_AGE_NORMALIZE", "1") == "1"
#: Final selection: "mmr" (relevance-first, diversity-aware; default) or the
#: paper's plain "mean_rank".
SELECTION = os.environ.get("QUAL_SG_SELECTION", "mmr")
#: Minimum topical-relevance score kept (EXCLUSIVE of lower scores).
RELEVANCE_GATE = int(os.environ.get("QUAL_SG_RELEVANCE_GATE", "3"))
#: Hard gate (default): only S_t >= RELEVANCE_GATE papers are selectable - no
#: relaxation when the pool is small and no anchor-provenance exemption.  Set to 0
#: to restore the paper's softer behaviour (relax the gate, let anchor citations in).
RELEVANCE_GATE_HARD = os.environ.get("QUAL_SG_RELEVANCE_GATE_HARD", "1") == "1"
#: Selected references must carry an abstract (no title-only entries).
REQUIRE_ABSTRACT = os.environ.get("QUAL_SG_REQUIRE_ABSTRACT", "1") == "1"
#: "Fully evaluated" pool size: only this many top candidates (by Eq. 1 similarity)
#: are sent to the expensive LLM judge.  Selection then happens inside that
#: shortlist.  0 = judge the whole pool.  If fewer than K papers clear the gate the
#: shortlist is extended automatically (logged).
EVAL_POOL_TARGET = int(os.environ.get("QUAL_SG_EVAL_POOL_TARGET", "0"))
MMR_LAMBDA = float(os.environ.get("QUAL_SG_MMR_LAMBDA", "0.72"))

# Composite weights.  Values are taken verbatim from the authors' released
# ``paper_reranking.py`` and correspond to alpha, beta, gamma of Eq. (3) plus the
# topical-relevance and diversity terms.
W_CITED = 0.165
W_AUTHOR = 0.067
W_VENUE = 0.10
W_RELEVANCE = 0.33
W_DIVERSITY = 0.33

# Tie-break order for the average-rank aggregation (paper §2.4 final paragraph).
RANK_AGGREGATION = os.environ.get("QUAL_SG_RANK_AGG", "mean_rank")  # mean_rank | composite

# --------------------------------------------------------------------------- #
# Embeddings (content diversity + evaluation)
# --------------------------------------------------------------------------- #
#: Paper (Appendix C fn.13): "bge-large-en-v1.5 ... to compute semantic similarity
#: throughout the pipeline".  The 1.3 GB large checkpoint could not be downloaded
#: reliably here, so the *same family* small variant is vendored under
#: ``qual_sg/models/bge-small-en-v1.5`` (133 MB, 384-dim) and used by default.
#: Override with QUAL_SG_EMBED_MODEL (e.g. "BAAI/bge-large-en-v1.5"); the step
#: modules fall back to TF-IDF if the transformer cannot be loaded.
EMBED_MODEL_DIR = PKG_DIR / "models" / "bge-small-en-v1.5"
EMBED_MODEL_NAME = os.environ.get(
    "QUAL_SG_EMBED_MODEL",
    str(EMBED_MODEL_DIR) if (EMBED_MODEL_DIR / "config.json").exists() else "BAAI/bge-small-en-v1.5",
)
EMBED_MODEL_FALLBACKS = [
    m.strip()
    for m in os.environ.get("QUAL_SG_EMBED_FALLBACKS", "sentence-transformers/all-MiniLM-L6-v2").split(",")
    if m.strip()
]

# --------------------------------------------------------------------------- #
# Evaluation thresholds (authors' defaults)
# --------------------------------------------------------------------------- #
TITLE_TEXTUAL_THR = float(os.environ.get("QUAL_SG_TITLE_THR", "0.95"))
SECTION_SEM_THR = float(os.environ.get("QUAL_SG_SECTION_THR", "0.80"))

# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
OUTLINE_MAX_REFS = int(os.environ.get("QUAL_SG_OUTLINE_REFS", "40"))
#: Max references handed to the writers; 0 = use every selected reference (no cap).
GENERATION_MAX_REFS = int(os.environ.get("QUAL_SG_GENERATION_REFS", "0"))
MIN_SUBSECTION_WORDS = int(os.environ.get("QUAL_SG_MIN_WORDS", "300"))
#: Length budget of the full survey; the outline is sized to it and each
#: subsection gets a word range derived from it (approved changes 5 and 6).
SURVEY_TARGET_WORDS = int(os.environ.get("QUAL_SG_SURVEY_WORDS", "4800"))
SURVEY_SECTIONS = os.environ.get("QUAL_SG_SURVEY_SECTIONS", "4-6")
SURVEY_SUBSECTIONS = os.environ.get("QUAL_SG_SURVEY_SUBSECTIONS", "12-15")
#: References given to each subsection, chosen by embedding similarity (change 4).
REFS_PER_SUBSECTION = int(os.environ.get("QUAL_SG_REFS_PER_SUBSECTION", "10"))
#: Abstract cap in the 2-page review prompt (change 8: full abstracts, safety cap).
REVIEW_ABSTRACT_CHARS = int(os.environ.get("QUAL_SG_REVIEW_ABSTRACT_CHARS", "2500"))
#: Claim check + one revision pass on every generated text (change 11).
CLAIM_CHECK = os.environ.get("QUAL_SG_CLAIM_CHECK", "1") == "1"


def load_env() -> None:
    """Load ``.env`` files (repo root, BASE_DIR, PKG_DIR) into ``os.environ``.

    Implemented without third-party dependencies so the pipeline runs on a bare
    Python install.  Existing environment variables are *not* overwritten.
    """
    for folder in (BASE_DIR.parent, BASE_DIR, PKG_DIR):
        env_path = folder / ".env"
        if not env_path.exists():
            continue
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def api_key() -> str:
    """Return the configured LLM API key (empty string when missing)."""
    load_env()
    return os.environ.get(LLM_API_KEY_ENV, "").strip()


def openalex_api_key() -> str:
    """Return the OpenAlex API key (empty string when missing).

    OpenAlex has required a key since February 2026: without one the daily budget
    is $0.10, which a single run can exhaust (the cause of the 429s in run v4).
    """
    load_env()
    return os.environ.get("OPENALEX_API_KEY", "").strip()


def model_for(stage: str) -> str:
    return {
        "judge": MODEL_JUDGE,
        "outline": MODEL_OUTLINE,
        "writer": MODEL_WRITER,
        "eval": MODEL_EVAL,
    }[stage]


def ensure_dirs() -> None:
    for d in (INTERMEDIATES_DIR, LOG_DIR, CACHE_DIR, CANONICAL_SECTIONS_DIR, OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
