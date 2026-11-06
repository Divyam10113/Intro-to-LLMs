"""Step 3 — quality-aware re-ranking of the candidate reference set (paper §2.4).

QUAL-SG / SurveyGen (arXiv 2508.17647) ranks the ~300 candidates retrieved in
steps 1-2 along three orthogonal quality axes and keeps the best ``K = topic.max_refs``:

* topical relevance ``S_t``     — an LLM (5-point rubric judge) scores each paper 1..5;
* academic impact  ``S_a``     — Eq. (3): ``alpha*C + beta*A + gamma*V`` where ``C``,
  ``A`` and ``V`` are the *quartile-bin* encodings (Q1 best -> 4, Q4 -> 1, then
  min-max to [0, 1]) of ``cited_by_count``, the mean first/last-author h-index and
  the venue (journal) h-index;
* content diversity ``S_d``    — Eq. (4): the mean cosine *distance* of a paper's
  embedding to the embeddings of every other candidate,
  ``S_d(i) = 1/(n-1) * sum_{j != i} (1 - cos(e_i, e_j))``.

Final ordering (paper §2.4, last paragraph) is *rank aggregation*: each paper is
ranked independently on the three axes and the winner is the paper with the best
mean rank; ties are broken by the weighted composite score.  The composite score
mirrors the authors' released ``paper_reranking.py`` (dynamic weights, divided by
the sum of the weights of the components that are actually available).

Inputs
------
``config.CANONICAL_REF_SET``  (step-2 JSON list of paper records)

Outputs
-------
``config.CANONICAL_RANKED_CSV``  final ranked table (exact legacy column order)
``config.STEP3_RANKINGS``        audit: weights, method, counts, ordered selection
``config.STEP3_SCORES``          JSONL audit of every relevance judgement (resume file)
``config.STEP3_EMBEDDINGS``      ``npz(dois, embeddings)`` when the transformer ran
                                 (otherwise the TF-IDF fallback is used and this file
                                 is left untouched — the fallback is recorded in
                                 ``STEP3_RANKINGS["diversity"]`` and in the log)

Run with::

    python3 -m qual_sg.step3_rerank     # from "Paper implementation/"

The stage is deterministic, resumable and idempotent: papers already present in
``STEP3_SCORES`` are not re-judged (whatever backend produced them).
"""
from __future__ import annotations

import itertools
import numbers
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import common, config

logger = common.get_logger("step3_rerank")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BIN_LABELS: Tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")

#: Extra (non-binning) numeric columns that must exist even when the source JSON
#: was produced by a variant of step 2.
NUMERIC_COLUMNS: Tuple[str, ...] = (
    "co_cited_count",
    "cited_by_count",
    "first_author_hindex",
    "last_author_hindex",
    "journal_hindex",
)

#: Exact legacy column order of ``paper_after_reranking.csv``.
CSV_COLUMNS: Tuple[str, ...] = (
    "title",
    "doi",
    "abstract",
    "origin",
    "cited_by_anchor",
    "co_cited_count",
    "cited_by_count",
    "pub_year",
    "citations_per_year",
    "cited_by_bin",
    "first_author_hindex",
    "last_author_hindex",
    "author_hindex_avg",
    "author_h_bin",
    "journal_hindex",
    "journal_h_bin",
    "relevance_llm_1to5",
    "relevance_bin",
    "avg_cos_sim_to_others",
    "avg_cos_dist_to_others",
    "rank_relevance",
    "rank_impact",
    "rank_diversity",
    "avg_rank",
    "impact_score",
    "diversity_score",
    "composite_score",
    "tag",
)

#: Weights of the composite score (Eq. (3) alpha/beta/gamma + relevance + diversity).
WEIGHT_VECTOR: Tuple[float, ...] = (
    config.W_CITED,
    config.W_AUTHOR,
    config.W_VENUE,
    config.W_RELEVANCE,
    config.W_DIVERSITY,
)


# --------------------------------------------------------------------------- #
# Loading / binning helpers
# --------------------------------------------------------------------------- #
def load_records(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Build the working frame, coercing numeric columns and filling gaps with NaN."""
    df = pd.DataFrame(list(records))
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("title", "doi", "abstract", "origin"):
        if col not in df.columns:
            df[col] = np.nan
    return df


def percentile_bin_q1_best(
    series: pd.Series,
    n_bins: int = config.N_BINS,
    labels: Sequence[str] = BIN_LABELS,
) -> pd.Series:
    """Quantile bins with ``Q1`` = best (highest) value, ``Q4`` = worst.

    Percentile ranks (``ascending=False``) are cut into ``n_bins`` equal-width
    intervals labelled ``Q1..Q4`` (paper §2.4 impact quartiles).
    """
    pct = series.rank(pct=True, ascending=False)
    return pd.cut(
        pct,
        bins=np.linspace(0, 1, n_bins + 1),
        labels=list(labels),
        include_lowest=True,
        duplicates="drop",
    )


def map_bin_q1_best(s: pd.Series) -> pd.Series:
    """Q1 -> 4 ... Q4 -> 1 (higher = better), NaN preserved."""
    return s.map({"Q1": 4, "Q2": 3, "Q3": 2, "Q4": 1}).astype(float)


def minmax01(s: pd.Series) -> pd.Series:
    """Min-max normalisation to [0, 1]; a constant (or empty) series maps to 0.5."""
    s = s.astype(float)
    if s.dropna().nunique() <= 1:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def safe_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def author_hindex_avg_row(row: pd.Series) -> float:
    """NaN-safe mean of the first- and last-author h-indices (paper §2.4)."""
    vals = [v for v in (row.get("first_author_hindex"), row.get("last_author_hindex")) if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan


def add_impact_components(work: pd.DataFrame) -> pd.DataFrame:
    """Add ``author_hindex_avg``, the three quartile bins and Eq.-(3) ``impact_score``."""
    work = work.copy()
    work["author_hindex_avg"] = work.apply(author_hindex_avg_row, axis=1)

    # Citations are age-normalised (paper Appendix C, fn. 15): raw counts make every
    # slot go to decade-old mega-cited papers ("Attention Is All You Need": 78k) and
    # starve recent, on-topic work.  citations_per_year restores the balance.
    if "publication_date" in work.columns:
        work["pub_year"] = pd.to_datetime(work["publication_date"], errors="coerce", format="mixed").dt.year
    else:
        work["pub_year"] = np.nan
    age = (pd.Timestamp.utcnow().year - work["pub_year"]).clip(lower=0) + 1
    work["citations_per_year"] = work["cited_by_count"] / age
    cited_series = work["citations_per_year"] if config.CITE_AGE_NORMALIZE else work["cited_by_count"]

    work["cited_by_bin"] = percentile_bin_q1_best(cited_series)
    work["author_h_bin"] = percentile_bin_q1_best(work["author_hindex_avg"])
    work["journal_h_bin"] = percentile_bin_q1_best(work["journal_hindex"])

    # Eq. (3): S_a = (alpha*C + beta*A + gamma*V) / (alpha + beta + gamma), with all
    # three components min-max normalised to [0, 1] (higher = better).
    cited = minmax01(map_bin_q1_best(work["cited_by_bin"]))
    author = minmax01(map_bin_q1_best(work["author_h_bin"]))
    venue = minmax01(map_bin_q1_best(work["journal_h_bin"]))
    w_c, w_a, w_v = config.W_CITED, config.W_AUTHOR, config.W_VENUE
    work["impact_score"] = (w_c * cited + w_a * author + w_v * venue) / (w_c + w_a + w_v)
    return work


# --------------------------------------------------------------------------- #
# Topical relevance (LLM judge)
# --------------------------------------------------------------------------- #
def build_relevance_prompt(topic: str, title: Any, abstract: Any) -> str:
    """The 5-point rubric prompt, byte-for-byte identical to ``paper_reranking.py``."""
    return f"""
You are an academic expert helping to write a survey on the topic: "{topic}".
You will be provided with the title and abstract of a research paper. Your task is to rating the paper's relevance to the survey topic based on the following criteria:

Score-1 (poor): The paper is unrelated to the topic or only mentions it in passing, with no meaningful contribution.
Score-2 (low): The paper is loosely connected or provides general background, but not focused on the topic.
Score-3 (moderate): The paper discusses a specific sub-aspect of the topic; somewhat useful, but not central.
Score-4 (high): The paper substantially addresses key elements of the topic; would likely be cited in the survey.
Score-5 (very high): The paper is entirely focused on the topic, offering essential insights; likely foundational to the survey.

Your output should only include a single score from 1 to 5. Do not provide any explanation or additional text.

Title: {title}
Abstract: {abstract}

Topical relevance score:
""".strip()


def load_resume_scores(path: Path) -> Dict[str, Any]:
    """Preload ``STEP3_SCORES`` so an interrupted run resumes without re-judging."""
    resume: Dict[str, Any] = {}
    for row in common.read_jsonl(path):
        if not isinstance(row, dict):
            continue
        key = str(row.get("key") or common.norm_doi(row.get("doi")))
        if key:
            resume[key] = row
    return resume


def record_key(item: Dict[str, Any]) -> str:
    """Resume key: the normalised DOI (falls back to a title hash when DOI-less)."""
    doi = common.norm_doi(item.get("doi"))
    if doi:
        return doi
    return "title:" + common.sha256_hex(safe_text(item.get("title")))[:16]


def _coerce_score(value: Any) -> Optional[int]:
    """Normalise whatever a backend returned (int, float or raw text) to 1..5."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, numbers.Integral):
        ival = int(value)
        return ival if 1 <= ival <= 5 else None
    if isinstance(value, numbers.Real):
        fval = float(value)
        return int(fval) if 1 <= fval <= 5 and fval.is_integer() else None
    return common.parse_first_int(value, 1, 5) if isinstance(value, str) else None


def score_relevance(
    client: Any,
    topic: common.Topic,
    items: List[Dict[str, Any]],
    *,
    audit_path: Path,
) -> Dict[int, Optional[int]]:
    """Judge topical relevance ``S_t`` for ``items`` (keyed by DataFrame label).

    Results are appended to the JSONL audit file as they arrive, under a lock, so
    a crash keeps everything judged so far.
    """
    results: Dict[int, Optional[int]] = {}
    if not items:
        return results

    append_lock = threading.Lock()
    n_total = len(items)
    counter = itertools.count(1)

    def persist(item: Dict[str, Any], score: Optional[int], res: Any, attempts: int) -> None:
        usage = getattr(res, "usage", None) or {}
        record = {
            "key": item["key"],
            "doi": common.norm_doi(item.get("doi")),
            "title": safe_text(item.get("title")),
            "score": score,
            "attempts": attempts,
            "model": getattr(res, "model", ""),
            "cached": bool(getattr(res, "cached", False)),
            "tokens": int(usage.get("total_tokens", 0) or 0),
            "ts": time.time(),
        }
        with append_lock:  # keep JSONL appends atomic across worker threads
            common.append_jsonl(audit_path, record)

    def prompt_fn(item: Dict[str, Any]) -> str:
        # Scope text: the anchor paper's title + abstract when one is configured
        # (a precise, human-written statement of the survey's scope), else the
        # topic title as in the authors' released code.
        return build_relevance_prompt(judge_topic_text(topic), item.get("title", ""), item.get("abstract", ""))

    # -- fast path: the in-session harness backend exposes ``score_relevance`` -- #
    if hasattr(client, "score_relevance"):
        try:
            raw = client.score_relevance(items, prompt_fn)
            if isinstance(raw, (list, tuple)) and len(raw) == len(items):
                for item, value in zip(items, raw):
                    score = _coerce_score(value)
                    results[item["idx"]] = score
                    persist(item, score, None, 1)
                    logger.info(
                        "[%d/%d] score=%s cached=%s tokens=0",
                        next(counter), n_total, score, False,
                    )
                return results
            logger.warning("harness score_relevance returned an unusable payload; falling back to chat()")
        except Exception as exc:  # noqa: BLE001 - any backend hiccup -> generic path
            logger.warning("harness score_relevance failed (%s); falling back to chat()", exc)

    # -- generic path: one chat() call per paper, retried once when unparsable -- #
    def score_one(item: Dict[str, Any]) -> Optional[int]:
        prompt = prompt_fn(item)
        score: Optional[int] = None
        attempts = 0
        res: Any = None
        for attempt in (1, 2):
            attempts = attempt
            try:
                # The retry bypasses the disk cache so a cached unparsable answer
                # does not make the second attempt a guaranteed repeat.
                res = client.chat(
                    [{"role": "user", "content": prompt}],
                    tag="relevance",
                    use_cache=(attempt == 1),
                )
            except common.PendingLLM:
                raise  # never record a waiting prompt as a failed (None) score
            except Exception as exc:  # noqa: BLE001
                logger.warning("relevance call failed (attempt %d): %s", attempt, exc)
                continue
            score = common.parse_first_int(res.text, 1, 5)
            if score is not None:
                break
            logger.warning("unparsable relevance answer (attempt %d): %r", attempt, res.text[:80])
        persist(item, score, res, attempts)
        usage = getattr(res, "usage", None) or {}
        logger.info(
            "[%d/%d] score=%s cached=%s tokens=%s",
            next(counter), n_total, score,
            bool(getattr(res, "cached", False)), int(usage.get("total_tokens", 0) or 0),
        )
        return score

    for item, score in zip(items, client.map_many(items, score_one, desc="relevance")):
        results[item["idx"]] = score
    return results


# --------------------------------------------------------------------------- #
# Content diversity (Eq. 4)
# --------------------------------------------------------------------------- #
def _cosine_distance_profile(embeddings: np.ndarray) -> np.ndarray:
    """Mean cosine similarity of every row to every *other* row (NaN when n < 2)."""
    n = embeddings.shape[0]
    if n < 2:
        return np.full(n, np.nan)
    emb = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = emb / norms
    sims = unit @ unit.T
    np.fill_diagonal(sims, np.nan)
    return np.nanmean(sims, axis=1)


def compute_diversity(work: pd.DataFrame) -> Tuple[pd.Series, Optional[np.ndarray], str]:
    """Return ``(avg_cos_sim_to_others, embeddings|None, method)`` for the work subset.

    Sentence-Transformers embeddings are preferred (normalised, so the dot product
    *is* the cosine); any failure falls back to a TF-IDF + cosine-similarity
    approximation, and the fallback is logged and recorded in the rankings audit.
    """
    texts = [
        f"{safe_text(t)}. {safe_text(a)}".strip()
        for t, a in zip(work["title"], work["abstract"])
    ]
    n = len(texts)
    if n == 0:
        return pd.Series(dtype=float, index=work.index), None, "none"

    embeddings: Optional[np.ndarray] = None
    method = "tfidf"
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        model = SentenceTransformer(config.EMBED_MODEL_NAME)
        embeddings = np.asarray(
            model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        method = f"sentence-transformers:{config.EMBED_MODEL_NAME}"
        logger.info("Diversity embeddings: %s (dim=%d)", method, embeddings.shape[1])
    except Exception as exc:  # noqa: BLE001 - missing torch/model, no network, OOM...
        logger.warning(
            "sentence-transformers unavailable (%s); falling back to TF-IDF cosine similarity",
            exc,
        )
        embeddings = None

    if embeddings is None:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415
        from sklearn.metrics.pairwise import cosine_similarity  # noqa: PLC0415

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        pairwise = cosine_similarity(matrix)
        avg_sims = _cosine_distance_profile(np.asarray(pairwise, dtype=np.float64))
        method = "tfidf"
        embeddings = None
    else:
        avg_sims = _cosine_distance_profile(embeddings)

    series = pd.Series(avg_sims, index=work.index, dtype=float)
    return series, embeddings, method


# --------------------------------------------------------------------------- #
# Rank aggregation + composite score
# --------------------------------------------------------------------------- #
#: Anchor descriptor written by step 1 (title/abstract of the user-supplied paper).
ANCHOR_JSON = config.INTERMEDIATES_DIR / "anchor.json"


def judge_topic_text(topic: common.Topic) -> str:
    """Text the relevance judge scores against.

    The survey's scope is stated by the topic (title + keywords) *and* sharpened by
    the anchor paper's title + abstract when one is configured: the topic says what
    the survey is about, the anchor says at what depth and along which method line.
    """
    parts = [topic.title]
    if topic.keywords:
        parts.append("Keywords: " + topic.keywords.replace(";", ", "))
    try:
        if ANCHOR_JSON.exists():
            anchor = common.read_json(ANCHOR_JSON)
            title = str(anchor.get("title") or "").strip()
            abstract = str(anchor.get("abstract") or "").strip()
            if title:
                parts.append(f"Anchor paper: {title}")
            if abstract:
                parts.append(abstract)
    except Exception as exc:  # noqa: BLE001 - never block scoring on the anchor file
        LOGGER.warning("could not read anchor scope (%s); using the topic only", exc)
    return "\n".join(p for p in parts if p).strip()


def select_references(
    work: pd.DataFrame,
    k: int,
    embeddings: Optional[np.ndarray],
    logger: Any,
) -> List[Any]:
    """Relevance-gated MMR selection.

    Plain mean-rank selection can admit off-topic papers that happen to be highly
    cited and maximally *dissimilar* (diversity is a distance, so it rewards
    irrelevance).  Instead we (1) gate on the topical-relevance score, relaxing the
    gate only if too few papers survive, then (2) greedily maximise
    ``lambda*relevance - (1-lambda)*max_similarity_to_selected`` over the gated
    pool (Maximal Marginal Relevance), which keeps the set both on-topic and
    non-redundant.
    """
    rel = work["relevance_llm_1to5"].astype(float)
    gate = config.RELEVANCE_GATE

    # Provenance: papers the user-supplied anchor cites are part of its citation
    # context, so they are exempt from the relevance gate and receive a prior.
    prior = pd.Series(False, index=work.index)
    if config.ANCHOR_REF_PRIORITY != "off":
        if "cited_by_anchor" in work.columns:
            prior |= work["cited_by_anchor"].fillna(False).astype(bool)
        if "origin" in work.columns:
            prior |= work["origin"].eq("anchor_ref")

    has_abstract = work["abstract"].apply(lambda x: isinstance(x, str) and x.strip() != "")
    if config.REQUIRE_ABSTRACT:
        eligible = has_abstract & (rel >= gate)
    else:
        eligible = (rel >= gate) | prior

    if config.RELEVANCE_GATE_HARD:
        pool = work[eligible]
        used_gate = gate
        if len(pool) < k:
            logger.warning(
                "strict gate S_t >= %d yields only %d candidates (< K=%d); selecting all of them "
                "rather than admitting lower-scored papers", gate, len(pool), k,
            )
    else:
        pool = work[eligible]
        used_gate = gate
        if len(pool) < k:
            for g in range(gate - 1, 0, -1):
                pool = work[((has_abstract & (rel >= g)) if config.REQUIRE_ABSTRACT else ((rel >= g) | prior))]
                used_gate = g
                if len(pool) >= k:
                    break
        if len(pool) < k:
            pool = work
            used_gate = 1
    logger.info(
        "selection: %s | anchor_priority=%s | gate>=%d keeps %d/%d candidates (%d cited by anchor), selecting %d",
        config.SELECTION, config.ANCHOR_REF_PRIORITY, used_gate, len(pool), len(work),
        int(prior.reindex(pool.index).sum()), k,
    )

    rel_pool = pool["relevance_llm_1to5"].astype(float).fillna(0.0)
    rel_norm = minmax01(rel_pool).fillna(0.0)
    prior_pool = prior.reindex(pool.index).fillna(False).to_numpy()
    if config.ANCHOR_REF_PRIORITY != "off" and prior_pool.any():
        rel_norm = pd.Series(np.maximum(rel_norm.to_numpy(dtype=float), np.where(prior_pool, config.ANCHOR_PRIOR_SCORE, 0.0)), index=pool.index)

    if config.SELECTION != "mmr":
        ordered = pool.assign(_rel=rel_norm).sort_values(
            ["avg_rank", "composite_score"], ascending=[True, False], kind="mergesort"
        )
        return list(ordered.head(k).index)

    # similarity matrix over the gated pool (fallback: static average similarity)
    if embeddings is not None and len(embeddings) == len(work):
        pos = {idx: i for i, idx in enumerate(work.index)}
        rows = [pos[i] for i in pool.index]
        emb = embeddings[rows]
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
        sim = emb @ emb.T
    else:
        s = pool["avg_cos_sim_to_others"].astype(float).fillna(0.0).to_numpy()
        sim = np.tile(s, (len(pool), 1))

    lam = config.MMR_LAMBDA
    n = len(pool)
    pool_idx = list(pool.index)
    rel_arr = rel_norm.to_numpy(dtype=float)
    chosen: List[int] = []
    remaining = set(range(n))

    # "strict" fills the quota with the anchor's bibliography first (ordered by the
    # paper's mean rank), then completes with MMR.
    if config.ANCHOR_REF_PRIORITY == "strict" and prior_pool.any():
        anchor_pos = sorted((i for i in range(n) if prior_pool[i]),
                            key=lambda i: (pool["avg_rank"].iloc[i] if pd.notna(pool["avg_rank"].iloc[i]) else 1e9,
                                           -(pool["composite_score"].iloc[i] or 0.0)))
        for i in anchor_pos[:k]:
            chosen.append(i)
            remaining.discard(i)

    while len(chosen) < min(k, n) and remaining:
        best_j, best_val = None, -np.inf
        for j in remaining:
            redundancy = max((sim[j, c] for c in chosen), default=0.0)
            val = lam * rel_arr[j] - (1 - lam) * redundancy
            if val > best_val:
                best_j, best_val = j, val
        chosen.append(best_j)
        remaining.discard(best_j)
    return [pool_idx[j] for j in chosen]


def add_ranks_and_composite(work: pd.DataFrame) -> pd.DataFrame:
    """Add §2.4 rank aggregation columns and the legacy weighted composite score."""
    work = work.copy()
    work["avg_cos_dist_to_others"] = 1.0 - work["avg_cos_sim_to_others"]  # Eq. (4) distance

    work["rank_relevance"] = work["relevance_llm_1to5"].rank(ascending=False, method="average")
    work["rank_impact"] = work["impact_score"].rank(ascending=False, method="average")
    work["rank_diversity"] = work["avg_cos_dist_to_others"].rank(ascending=False, method="average")

    # Mean rank across the three axes (paper §2.4 final ordering); NaN-safe mean of
    # whichever ranks are available for a paper.
    ranks = work[["rank_relevance", "rank_impact", "rank_diversity"]]
    work["avg_rank"] = ranks.mean(axis=1, skipna=True)

    # Normalised components of the legacy composite ("higher = better").
    cited_score = minmax01(map_bin_q1_best(work["cited_by_bin"]))
    auth_score = minmax01(map_bin_q1_best(work["author_h_bin"]))
    jour_score = minmax01(map_bin_q1_best(work["journal_h_bin"]))
    rel_score = minmax01(work["relevance_llm_1to5"])
    uniq_score = 1.0 - minmax01(work["avg_cos_sim_to_others"])  # lower similarity = better

    weights = np.asarray(WEIGHT_VECTOR, dtype=float)
    parts = np.column_stack([
        cited_score.to_numpy(dtype=float),
        auth_score.to_numpy(dtype=float),
        jour_score.to_numpy(dtype=float),
        rel_score.to_numpy(dtype=float),
        uniq_score.to_numpy(dtype=float),
    ])

    def row_weighted_score(vals: np.ndarray) -> float:
        mask = ~np.isnan(vals)
        if not mask.any():
            return np.nan
        wsum = weights[mask].sum()
        if wsum <= 0:
            return np.nan
        return float((vals[mask] * weights[mask]).sum() / wsum)

    work["composite_score"] = [row_weighted_score(v) for v in parts]
    work["diversity_score"] = uniq_score  # normalised S_d used by the composite
    work["relevance_bin"] = percentile_bin_q1_best(work["relevance_llm_1to5"])
    return work


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def main() -> None:
    config.ensure_dirs()
    topic = common.load_topic()
    logger.info("Step 3 re-ranking | topic=%r | K=%d", topic.title, topic.max_refs)

    records = common.read_json(config.CANONICAL_REF_SET)
    if isinstance(records, dict):  # tolerate {"papers": [...]} style payloads
        records = records.get("papers") or records.get("records") or []
    df = load_records(records)
    total = len(df)

    # --- 1) keep only records that can be judged / embedded ------------------- #
    # Anchor-bibliography entries are exempt from the abstract requirement: they
    # were resolved from a human bibliography (Crossref frequently has no abstract
    # for conference papers) and are judged on their title, flagged title_only.
    has_abstract = df["abstract"].apply(lambda x: isinstance(x, str) and x.strip() != "")
    is_anchor_ref = df["origin"].eq("anchor_ref") if "origin" in df.columns else pd.Series(False, index=df.index)
    keep = has_abstract | is_anchor_ref
    dropped = int((~keep).sum())
    work = df[keep].copy()
    logger.info("Loaded %d records; kept %d with abstracts (dropped %d)", total, len(work), dropped)

    if len(work) == 0:
        logger.warning("No records with abstracts — writing an empty ranking.")
        write_outputs(df, work, topic, diversity_method="none", diversity_source="no rows",
                      summary=None, selected=[])
        return

    # --- 2) academic impact (Eq. 3) ------------------------------------------- #
    work = add_impact_components(work)

    # --- 2b) evaluation shortlist -------------------------------------------- #
    # Only the N most similar candidates (Eq. 1) are "fully evaluated"; selection
    # happens inside that shortlist.  If fewer than K papers clear the gate the
    # shortlist is widened to the whole pool (logged), so K is still reachable.
    work_full = work
    shortlisted = False
    if config.EVAL_POOL_TARGET > 0 and len(work) > config.EVAL_POOL_TARGET:
        sim = pd.to_numeric(work["retrieval_similarity"], errors="coerce") if "retrieval_similarity" in work.columns else pd.Series(np.nan, index=work.index)
        order = pd.DataFrame({"s": sim.fillna(-1.0), "c": pd.to_numeric(work["citations_per_year"], errors="coerce").fillna(-1.0)})
        ordered = order.sort_values(["s", "c"], ascending=False, kind="mergesort").index
        work = work.loc[ordered[: config.EVAL_POOL_TARGET]]
        shortlisted = True
        logger.info("evaluation shortlist: top %d of %d candidates by Eq.(1) similarity (EVAL_POOL_TARGET=%d)",
                    len(work), len(work_full), config.EVAL_POOL_TARGET)

    # --- 3) topical relevance (LLM judge, resumable) --------------------------- #
    llm_clients: List[Any] = []

    def judge_set(wk: pd.DataFrame) -> Dict[int, Optional[int]]:
        """Judge every row of ``wk`` that is not already in the resume file."""
        resume = load_resume_scores(config.STEP3_SCORES)
        pending: List[Dict[str, Any]] = []
        out: Dict[int, Optional[int]] = {}
        for idx, row in wk.iterrows():
            item = {"idx": idx, "doi": row.get("doi"), "title": row.get("title"), "abstract": row.get("abstract")}
            item["key"] = record_key(item)
            if item["key"] in resume:
                out[idx] = _coerce_score(resume[item["key"]].get("score"))
            else:
                pending.append(item)
        if pending:
            logger.info("Judging relevance for %d papers (%d resumed from cache)", len(pending), len(wk) - len(pending))
            client = common.make_llm_client("judge")
            llm_clients.append(client)
            out.update(score_relevance(client, topic, pending, audit_path=config.STEP3_SCORES))
        else:
            logger.info("All %d papers already scored — skipping the LLM stage.", len(wk))
        return out

    scores = judge_set(work)
    work["relevance_llm_1to5"] = pd.Series(
        {idx: (float(s) if s is not None else np.nan) for idx, s in scores.items()}, dtype=float
    ).reindex(work.index)

    if shortlisted:
        passing = int((pd.to_numeric(work["relevance_llm_1to5"], errors="coerce") >= config.RELEVANCE_GATE).sum())
        if passing < topic.max_refs:
            logger.warning(
                "shortlist yields only %d papers with S_t >= %d (< K=%d); widening to the full %d-candidate pool",
                passing, config.RELEVANCE_GATE, topic.max_refs, len(work_full),
            )
            work = work_full
            scores = judge_set(work)
            work["relevance_llm_1to5"] = pd.Series(
                {idx: (float(s) if s is not None else np.nan) for idx, s in scores.items()}, dtype=float
            ).reindex(work.index)

    # --- 4) content diversity (Eq. 4) ----------------------------------------- #
    diversity, embeddings, diversity_method = compute_diversity(work)
    work["avg_cos_sim_to_others"] = diversity

    if embeddings is not None:
        dois = [common.norm_doi(d) for d in work["doi"]]
        np.savez(
            config.STEP3_EMBEDDINGS,
            dois=np.asarray(dois, dtype=object),
            embeddings=embeddings.astype(np.float32),
        )
        logger.info("Saved embeddings -> %s", config.STEP3_EMBEDDINGS)

    # --- 5) rank aggregation + composite score -------------------------------- #
    work = add_ranks_and_composite(work)

    # --- 6) selection: relevance gate + maximal marginal relevance ------------- #
    selected_idx = select_references(work, topic.max_refs, embeddings, logger)

    # --- 7) write artefacts ---------------------------------------------------- #
    write_outputs(
        df, work, topic,
        diversity_method=diversity_method,
        diversity_source=(
            f"cosine similarity over {diversity_method} embeddings"
            if diversity_method.startswith("sentence-transformers")
            else "cosine similarity over TF-IDF vectors (sentence-transformers fallback)"
        ),
        summary=(llm_clients[-1].summary() if llm_clients else None),
        selected=selected_idx,
    )

    n_selected = len(selected_idx)
    logger.info(
        "Step 3 done | scored=%d failed=%d selected=%d/%d | csv=%s",
        int(work["relevance_llm_1to5"].notna().sum()),
        int(work["relevance_llm_1to5"].isna().sum()),
        n_selected, topic.max_refs, config.CANONICAL_RANKED_CSV,
    )


def write_outputs(
    df: pd.DataFrame,
    work: pd.DataFrame,
    topic: common.Topic,
    *,
    diversity_method: str,
    diversity_source: str,
    summary: Optional[Dict[str, Any]],
    selected: Sequence[Any],
) -> None:
    """Write the ranked CSV and the step-3 audit JSON."""
    out = df.copy()
    for col in CSV_COLUMNS:
        if col in work.columns:
            # Recomputed/derived column: lift it onto the full index (rows without
            # an abstract simply stay NaN/empty).
            out[col] = work[col].reindex(df.index)
        elif col not in out.columns:
            out[col] = np.nan

    out["tag"] = pd.Series("", index=out.index, dtype=object)
    out.loc[list(selected), "tag"] = "selected"

    # Deterministic ordering: mean rank ascending (unranked rows last), composite
    # score as the tie-break.
    out = out.sort_values(
        ["avg_rank", "composite_score"], ascending=[True, False], kind="mergesort"
    )
    out = out[list(CSV_COLUMNS)]

    # NaN/NA -> empty cells (exact legacy export behaviour).
    printable = out.astype(object).where(pd.notna(out), "")
    config.CANONICAL_RANKED_CSV.parent.mkdir(parents=True, exist_ok=True)
    printable.to_csv(config.CANONICAL_RANKED_CSV, index=False)
    logger.info("Wrote %d rows -> %s", len(printable), config.CANONICAL_RANKED_CSV)

    ordered = out.loc[list(selected)]
    selected_rows = [
        {
            "doi": common.norm_doi(row.get("doi")),
            "title": safe_text(row.get("title")),
            "avg_rank": _jsonable(row.get("avg_rank")),
            "composite_score": _jsonable(row.get("composite_score")),
            "relevance_llm_1to5": _jsonable(row.get("relevance_llm_1to5")),
        }
        for _, row in ordered.iterrows()
    ]

    rankings = {
        "step": 3,
        "stage": "quality_aware_reranking",
        "paper_section": "2.4",
        "topic": {
            "title": topic.title,
            "year": topic.year,
            "keywords": topic.keywords,
            "max_refs": topic.max_refs,
        },
        "K": topic.max_refs,
        "weights": {
            "cited": config.W_CITED,
            "author": config.W_AUTHOR,
            "venue": config.W_VENUE,
            "relevance": config.W_RELEVANCE,
            "diversity": config.W_DIVERSITY,
        },
        "equations": {
            "impact": "S_a = (alpha*C + beta*A + gamma*V) / (alpha+beta+gamma) over Q1..Q4 bins",
            "diversity": "S_d(i) = mean_{j!=i} (1 - cos(e_i, e_j))",
        },
        "rank_aggregation": {
            "method": config.RANK_AGGREGATION,
            "detail": "mean of the independent ranks on S_t, S_a and S_d (NaN-safe)",
            "tie_break": "composite_score descending",
        },
        "selection_criterion": (
            f"{config.SELECTION} over a relevance-gated pool (gate S_t >= {config.RELEVANCE_GATE}, "
            f"anchor provenance priority '{config.ANCHOR_REF_PRIORITY}', MMR lambda={config.MMR_LAMBDA}); "
            f"K={topic.max_refs}"
        ),
        "counts": {
            "records_total": int(len(df)),
            "records_with_abstract": int(len(work)),
            "records_dropped_no_abstract": int(len(df) - len(work)),
            "scored_relevance": int(work["relevance_llm_1to5"].notna().sum()) if len(work) else 0,
            "relevance_failed": int(work["relevance_llm_1to5"].isna().sum()) if len(work) else 0,
            "selected": len(selected_rows),
        },
        "diversity": {
            "method": diversity_method,
            "model": config.EMBED_MODEL_NAME if diversity_method.startswith("sentence-transformers") else "tfidf",
            "source": diversity_source,
        },
        "relevance_backend": summary or {"note": "no LLM calls were needed (cache hit for every paper)"},
        "paths": {
            "input": str(config.CANONICAL_REF_SET),
            "csv": str(config.CANONICAL_RANKED_CSV),
            "scores": str(config.STEP3_SCORES),
            "embeddings": str(config.STEP3_EMBEDDINGS),
        },
        "selected": selected_rows,
    }
    common.write_json(config.STEP3_RANKINGS, rankings)
    logger.info("Wrote ranking audit -> %s", config.STEP3_RANKINGS)


def _jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain JSON types, NaN -> None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


if __name__ == "__main__":
    main()
