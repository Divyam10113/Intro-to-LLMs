"""Step 6 — Evaluation of the generated survey (paper §3.2).

The paper evaluates a generated survey along three axes:

1. **Citation quality** — precision / recall / F1 between the reference set that the
   pipeline *selected* (step 3) and the citations of the gold-standard survey.
   A selected title counts as a true positive when its lexical similarity to some
   gold reference title is ``>= config.TITLE_TEXTUAL_THR`` (greedy one-to-one).
2. **Content quality** — ROUGE-1 / ROUGE-2 / ROUGE-Lsum against the gold full text
   plus the cosine similarity of whole-document embeddings.
3. **Structural consistency** — the gold and generated section trees are embedded
   and matched greedily (all pairs sorted by cosine, one-to-one, accept while
   ``sim >= config.SECTION_SEM_THR``); the score is
   ``matched / (n_generated + n_gold)``.  An LLM additionally rates the alignment
   of the two outlines on the paper's 5-point rubric.

Two deviations from the authors' ``evaluation.py`` are intentional and documented:

* every gold-dependent metric is skipped (never fatal) when the gold file is
  missing, because the repository only ships a gold standard for the *example*
  topic;
* the shipped gold standard is for a different topic than the configured survey,
  so the report always carries ``gold_topic_mismatch``, a ``warnings`` list and a
  set of *intrinsic* metrics that stay meaningful without any gold standard
  (section counts, citation coverage, intra-document semantic coherence, length).

Nothing here performs a network/LLM call at import time; the optional LLM score
runs only when a gold standard *and* an API key are present.
"""
from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import common, config

LOGGER = common.get_logger("step6_evaluate")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# Sections such as funding/acknowledgements are not part of the scientific
# content and are filtered out of the gold outline before comparing structures.
NON_CONTENT_SEC_PATTERNS: List[str] = [
    r"\bfunding\b",
    r"\backnowledg(e)?ments?\b",
    r"\bauthor contributions?\b",
    r"\bcompeting interests?\b",
    r"\bconflicts? of interest\b",
    r"\bsupplementary( materials?| information)?\b",
    r"\bappendix\b",
    r"\bdata availability\b",
    r"\bethics\b",
]

# A gold standard is considered "for another topic" when fewer than this share of
# the (content-word) tokens of its title also occur in the configured topic.
GOLD_TOPIC_OVERLAP_THR = 0.34

# Step 5 writes citations as ``ref [12]`` (usually wrapped in ``\cite{...}``).
_CITE_RE = re.compile(r"ref\s*\[\s*([0-9][0-9\s,\u2013\-]*)\s*\]", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\u2019\-]*")

_STOPWORDS = {
    "a", "an", "and", "the", "of", "for", "in", "on", "to", "with", "without",
    "from", "by", "at", "as", "is", "are", "be", "brief", "review", "survey",
    "study", "paper", "using", "based", "via", "towards", "toward", "new",
}

# ``normalize_text`` is shared with the rest of the package (same semantics as the
# evaluated reference implementation).
normalize_text = common.normalize_text


# --------------------------------------------------------------------------- #
# Small text helpers
# --------------------------------------------------------------------------- #
def textual_similarity(a: str, b: str) -> float:
    """Lexical similarity in ``[0, 1]`` (difflib ratio over normalized text)."""
    return difflib.SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _content_tokens(text: str) -> Set[str]:
    return {w for w in _WORD_RE.findall(normalize_text(text)) if w not in _STOPWORDS and len(w) > 1}


def token_overlap(a: str, b: str) -> float:
    """Share of ``a``'s content tokens that appear (substring-wise) in ``b``."""
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta or not tb:
        return 0.0
    hits = sum(1 for w in ta if any(w in x or x in w for x in tb))
    return hits / len(ta)


# --------------------------------------------------------------------------- #
# Gold-standard extraction
# --------------------------------------------------------------------------- #
def is_non_content_section(title: str) -> bool:
    t = normalize_text(title)
    return any(re.search(pat, t) for pat in NON_CONTENT_SEC_PATTERNS)


def extract_gold_reference_titles(gold: Dict[str, Any]) -> List[str]:
    """All ``sections[].paragraphs[].citations[].title`` values (deduplicated)."""
    titles: List[str] = []
    for sec in gold.get("sections", []) or []:
        for para in sec.get("paragraphs", []) or []:
            for cit in para.get("citations", []) or []:
                t = cit.get("title")
                if isinstance(t, str) and t.strip():
                    titles.append(t.strip())
    seen: Set[str] = set()
    uniq: List[str] = []
    for t in titles:
        key = normalize_text(t)
        if key and key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq


def extract_gold_section_titles_filtered(gold: Dict[str, Any]) -> List[str]:
    """Gold section titles with the non-content sections removed."""
    titles: List[str] = []
    for sec in gold.get("sections", []) or []:
        t = sec.get("title")
        if isinstance(t, str) and t.strip() and not is_non_content_section(t):
            titles.append(t.strip())
    return titles


def extract_gold_fulltext(gold: Dict[str, Any]) -> str:
    """Concatenated ``sections[].paragraphs[].text`` of the gold survey."""
    texts: List[str] = []
    for sec in gold.get("sections", []) or []:
        for para in sec.get("paragraphs", []) or []:
            tx = para.get("text")
            if isinstance(tx, str) and tx.strip():
                texts.append(tx.strip())
    return "\n".join(texts).strip()


def gold_title_hint(gold: Dict[str, Any]) -> str:
    """Best-effort title of the gold survey (metadata title, else top-level)."""
    meta = gold.get("metadata") or {}
    if isinstance(meta, dict):
        for key in ("title", "name"):
            val = meta.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    for key in ("title", "name"):
        val = gold.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


# --------------------------------------------------------------------------- #
# Generated-survey extraction
# --------------------------------------------------------------------------- #
def load_selected_titles(csv_path: Path) -> List[str]:
    """Titles of the papers tagged ``selected`` in the step-3 ranked CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "tag" in df.columns:
        df = df[df["tag"] == "selected"]
    col = "title" if "title" in df.columns else df.columns[0]
    return [t.strip() for t in df[col].tolist() if isinstance(t, str) and t.strip()]


def load_outline(path: Path) -> Dict[str, Any]:
    """Read ``survey_outline.json`` and return title + section/subsection titles.

    Accepts both ``subsections`` shapes used by the generator (dict of
    ``title -> description`` and list of titles/objects).
    """
    outline = common.read_json(path)
    sections: List[str] = []
    subsections: List[str] = []
    for sec in (outline.get("outline") or []):
        t = sec.get("section_title")
        if isinstance(t, str) and t.strip():
            sections.append(t.strip())
        subs = sec.get("subsections") or {}
        names: List[str] = []
        if isinstance(subs, dict):
            names = [k for k in subs.keys() if isinstance(k, str)]
        elif isinstance(subs, list):
            for item in subs:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict):
                    names.append(
                        item.get("subsection_title") or item.get("title") or ""
                    )
        subsections.extend(n.strip() for n in names if isinstance(n, str) and n.strip())
    return {
        "title": str(outline.get("title") or "").strip(),
        "sections": sections,
        "subsections": subsections,
    }


def _normalize_bundle_rows(rows: Iterable[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(row)
        elif isinstance(row, str) and row.strip():
            out.append({"content": row.strip()})
    return out


def load_generated_content(path: Path, fallback_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load the generated subsections as a list of ``{title, content}`` rows.

    Primary source is ``survey_content.json`` (a bundle with a ``subsections``
    list); when it is absent we fall back to the per-section JSON files written
    by step 5 under ``config.CANONICAL_SECTIONS_DIR``.
    """
    if path.exists():
        data = common.read_json(path)
        if isinstance(data, dict):
            subs = data.get("subsections")
            if isinstance(subs, dict):  # {title: description | {"content": ...}}
                return _normalize_bundle_rows(
                    ({**v, "subsection_title": v.get("subsection_title") or k}
                     if isinstance(v, dict) else {"subsection_title": k, "content": v})
                    for k, v in subs.items()
                )
            if isinstance(subs, list):
                return _normalize_bundle_rows(subs)
            for key in ("sections", "content"):
                node = data.get(key)
                if isinstance(node, list):
                    return _normalize_bundle_rows(node)
        if isinstance(data, list):
            return _normalize_bundle_rows(data)

    if fallback_dir and Path(fallback_dir).is_dir():
        rows: List[Dict[str, Any]] = []
        for fp in sorted(Path(fallback_dir).glob("*.json")):
            try:
                item = common.read_json(fp)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Skipping unreadable section file %s (%s)", fp.name, exc)
                continue
            if isinstance(item, dict):
                item.setdefault("subsection_title", item.get("title") or fp.stem)
                rows.append(item)
        return rows
    return []


def content_fulltext(rows: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for row in rows:
        for key in ("content", "text", "generated_content"):
            val = row.get(key) if isinstance(row, dict) else None
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
                break
    return "\n\n".join(parts).strip()


def bundle_reference_ids(rows: Sequence[Dict[str, Any]]) -> Set[str]:
    """Distinct reference identifiers recorded inside the generated bundles."""
    ids: Set[str] = set()
    for row in rows:
        refs = row.get("references") if isinstance(row, dict) else None
        if isinstance(refs, dict):
            refs = list(refs.values())
        if isinstance(refs, list):
            for ref in refs:
                if isinstance(ref, str) and ref.strip():
                    ids.add(normalize_text(ref))
                elif isinstance(ref, dict):
                    key = ref.get("ref_id") or ref.get("doi") or ref.get("title")
                    if isinstance(key, str) and key.strip():
                        ids.add(normalize_text(key))
    return ids


# --------------------------------------------------------------------------- #
# Citation markers
# --------------------------------------------------------------------------- #
def parse_cited_indices(text: str) -> Tuple[Set[int], int]:
    """Parse ``ref [n]`` / ``\\cite{ref [n]}`` markers in the generated text.

    Returns the set of distinct reference numbers plus the total number of
    markers seen (a range such as ``ref [3-5]`` counts as one marker but three
    distinct references).
    """
    distinct: Set[int] = set()
    markers = 0
    for m in _CITE_RE.finditer(text or ""):
        markers += 1
        body = m.group(1)
        for part in re.split(r"[,;]", body):
            part = part.strip()
            rng = re.match(r"^(\d+)\s*[\u2013\-]\s*(\d+)$", part)
            if rng:
                lo, hi = int(rng.group(1)), int(rng.group(2))
                if lo > hi:
                    lo, hi = hi, lo
                if 0 < hi - lo <= 100:
                    distinct.update(range(lo, hi + 1))
                else:
                    distinct.update({lo, hi})
            else:
                distinct.update(int(x) for x in re.findall(r"\d+", part))
    return distinct, markers


# --------------------------------------------------------------------------- #
# Embeddings (with a TF-IDF fallback so evaluation never hard-fails)
# --------------------------------------------------------------------------- #
class Embedder:
    """Cosine-ready embeddings from SentenceTransformers, or TF-IDF as fallback."""

    def __init__(self, model_name: str = config.EMBED_MODEL_NAME) -> None:
        self.backend = "tfidf"
        self.model_name = model_name
        self._model: Any = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self.backend = "sentence-transformers"
            LOGGER.info("Embedding backend: sentence-transformers (%s)", model_name)
        except Exception as exc:  # noqa: BLE001 — offline machines must still work
            LOGGER.warning(
                "SentenceTransformer(%s) unavailable (%s); using TF-IDF cosine.",
                model_name, exc,
            )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """L2-normalized embedding matrix (cosine similarity == dot product)."""
        items = [t if isinstance(t, str) and t.strip() else " " for t in texts]
        if not items:
            return np.zeros((0, 0), dtype=float)
        if self._model is not None:
            embs = self._model.encode(
                items, convert_to_tensor=True, normalize_embeddings=True
            ).cpu().numpy()
            return np.asarray(embs, dtype=float)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        return np.asarray(normalize(vec.fit_transform(items)).toarray(), dtype=float)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    return a @ b.T


def mean_pairwise_cosine(embs: np.ndarray) -> float:
    """Mean of all off-diagonal pairwise cosines (intra-document coherence)."""
    n = embs.shape[0]
    if n < 2:
        return 0.0
    sims = cosine_matrix(embs, embs)
    total = float(sims.sum()) - float(np.trace(sims))
    return total / (n * (n - 1))


# --------------------------------------------------------------------------- #
# Metric 1 — citation quality (PRF)
# --------------------------------------------------------------------------- #
def compute_reference_prf(
    gold_titles: Sequence[str], selected_titles: Sequence[str], thr: float = config.TITLE_TEXTUAL_THR
) -> Dict[str, Any]:
    """Greedy one-to-one lexical matching between selected and gold titles.

    All ``(selected, gold)`` pairs with ``similarity >= thr`` are sorted by
    similarity and consumed greedily, so a title can be matched only once.
    """
    pairs = [
        (difflib.SequenceMatcher(None, normalize_text(s), normalize_text(g)).ratio(), i, j)
        for i, s in enumerate(selected_titles)
        for j, g in enumerate(gold_titles)
    ]
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))

    used_sel: Set[int] = set()
    used_gold: Set[int] = set()
    matches: List[Dict[str, Any]] = []
    for sim, i, j in pairs:
        if sim < thr:
            break
        if i in used_sel or j in used_gold:
            continue
        used_sel.add(i)
        used_gold.add(j)
        matches.append(
            {
                "selected_idx": i,
                "gold_idx": j,
                "selected_title": selected_titles[i],
                "gold_title": gold_titles[j],
                "similarity": float(sim),
            }
        )

    tp = len(matches)
    n_sel, n_gold = len(selected_titles), len(gold_titles)
    precision = tp / n_sel if n_sel else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "threshold": thr,
        "selected_count": n_sel,
        "gold_ref_count": n_gold,
        "matched_count": tp,
        "matched_selected_count": len(used_sel),
        "matched_gold_count": len(used_gold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }


# --------------------------------------------------------------------------- #
# Metric 3a — structural consistency (semantic)
# --------------------------------------------------------------------------- #
def greedy_semantic_matching(
    a_titles: Sequence[str], b_titles: Sequence[str], embedder: Embedder, thr: float
) -> Tuple[int, List[Tuple[int, int, float]]]:
    """One-to-one greedy matching over all pairs above ``thr``."""
    if not a_titles or not b_titles:
        return 0, []
    sims = cosine_matrix(embedder.encode(a_titles), embedder.encode(b_titles))
    flat = sorted(
        ((float(sims[i, j]), i, j) for i in range(len(a_titles)) for j in range(len(b_titles))),
        key=lambda x: (-x[0], x[1], x[2]),
    )
    used_a: Set[int] = set()
    used_b: Set[int] = set()
    pairs: List[Tuple[int, int, float]] = []
    for sim, i, j in flat:
        if sim < thr:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, sim))
    return len(pairs), pairs


def structural_consistency(
    gen_sections: Sequence[str],
    gold_sections: Sequence[str],
    embedder: Embedder,
    thr: float = config.SECTION_SEM_THR,
) -> Dict[str, Any]:
    matched, pairs = greedy_semantic_matching(gen_sections, gold_sections, embedder, thr)
    denom = len(gen_sections) + len(gold_sections)
    return {
        "threshold": thr,
        "generated_sections": len(gen_sections),
        "gold_sections": len(gold_sections),
        "matched_sections": matched,
        "consistency": matched / denom if denom else 0.0,
        "pairs": [
            {
                "gen_idx": gi,
                "gold_idx": gj,
                "gen_title": gen_sections[gi],
                "gold_title": gold_sections[gj],
                "similarity": sim,
            }
            for gi, gj, sim in pairs
        ],
    }


# --------------------------------------------------------------------------- #
# Metric 2 — content quality (ROUGE + document-level semantic similarity)
# --------------------------------------------------------------------------- #
def compute_rouge(gen_text: str, gold_text: str) -> Optional[Dict[str, Any]]:
    try:
        from rouge_score import rouge_scorer
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("rouge_score unavailable (%s); ROUGE metrics skipped.", exc)
        return None
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    res = scorer.score(gold_text, gen_text)
    return {
        name: {
            "p": res[name].precision,
            "r": res[name].recall,
            "f": res[name].fmeasure,
        }
        for name in ("rouge1", "rouge2", "rougeLsum")
    }


def compute_content_quality(gen_text: str, gold_text: str, embedder: Embedder) -> Dict[str, Any]:
    rouge = compute_rouge(gen_text, gold_text)
    embs = embedder.encode([gen_text, gold_text])
    sem_sim = float(cosine_matrix(embs[:1], embs[1:2])[0, 0]) if embs.shape[0] >= 2 else 0.0
    return {"rouge": rouge, "semantic_similarity": sem_sim}


# --------------------------------------------------------------------------- #
# Metric 3b — structural consistency (LLM 5-point rubric)
# --------------------------------------------------------------------------- #
def build_llm_structure_prompt(
    topic: str, gen_outline_titles: Sequence[str], gold_outline_titles: Sequence[str]
) -> str:
    """Rubric prompt from the authors' implementation (score 1–5, output only)."""
    def stringify(titles: Sequence[str]) -> str:
        return "\n".join(f"- {t}" for t in titles) if titles else "(empty outline)"

    return f"""
You are an academic expert tasked with evaluating a draft outline for a literature survey on the topic of "{topic}".

You will be provided with two outlines:
1.A draft outline  written as a preliminary version.
--{stringify(gen_outline_titles)}

2.A gold-standard outline created  by domain experts.
--{stringify(gold_outline_titles)}

Your task is to rate the structural consistency between the draft outline and the gold-standard outline, following the criteria below:
Score 1 (Very Poor): The draft outline is largely inconsistent with the gold-standard; major sections are missing or irrelevant.
Score 2 (Poor): The draft outline only partially overlaps with the gold-standard; some key themes are omitted or misplaced.
Score 3 (Moderate): The draft includes some of the core topics but lacks structural alignment or completeness.
Score 4 (Good): The draft mostly aligns with the gold-standard; minor deviations in structure or scope.
Score 5 (Excellent): The draft closely mirrors the gold-standard in both structure and content coverage.

Your output should only include a single score from 1 to 5. Do not provide any explanation or additional text.
""".strip()


def llm_structure_score(prompt: str) -> Dict[str, Any]:
    """Call the eval-stage LLM for the 1–5 rubric score; never raises."""
    info: Dict[str, Any] = {"score": None, "model": None, "cached": None, "error": None}
    if not config.api_key():
        info["error"] = f"no API key ({config.LLM_API_KEY_ENV}); LLM structure score skipped"
        return info
    try:
        client = common.make_llm_client("eval")
        res = client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            tag="step6-structure-score",
        )
        info["model"] = res.model
        info["cached"] = res.cached
        info["score"] = common.parse_first_int(res.text, 1, 5)
        if info["score"] is None:
            info["error"] = f"unparsable score: {res.text[:120]!r}"
    except Exception as exc:  # noqa: BLE001 — evaluation must not fail the pipeline
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


# --------------------------------------------------------------------------- #
# Intrinsic metrics (always computable, gold optional)
# --------------------------------------------------------------------------- #
def intrinsic_metrics(
    outline: Dict[str, Any],
    content_rows: Sequence[Dict[str, Any]],
    gen_text: str,
    selected_titles: Sequence[str],
    embedder: Embedder,
) -> Dict[str, Any]:
    contents: List[str] = []
    for row in content_rows:
        for field in ("content", "text", "generated_content"):
            val = row.get(field) if isinstance(row, dict) else None
            if isinstance(val, str) and val.strip():
                contents.append(val.strip())
                break

    cited, markers = parse_cited_indices(gen_text)
    n_subsections = max(len(content_rows), len(outline.get("subsections") or []))
    available = len(selected_titles) or len(bundle_reference_ids(content_rows))
    word_count = count_words(gen_text)

    coherence = 0.0
    if len(contents) >= 2:
        coherence = mean_pairwise_cosine(embedder.encode(contents))

    return {
        "n_generated_sections": len(outline.get("sections") or []),
        "n_generated_subsections": n_subsections,
        "n_generated_subsections_with_content": len(contents),
        "n_selected_refs": len(selected_titles),
        "distinct_refs_available": available,
        "n_distinct_refs_cited": len(cited),
        "n_citation_markers": markers,
        "max_ref_index_cited": max(cited) if cited else None,
        "out_of_range_ref_indices": sorted(i for i in cited if i < 1 or i > available) if available else [],
        "citation_coverage": (len(cited) / available) if available else 0.0,
        "mean_section_semantic_coherence": coherence,
        "generated_word_count": word_count,
        "mean_words_per_subsection": (word_count / len(contents)) if contents else 0.0,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _say(line: str = "") -> None:
    """Print to stdout and mirror the line into the stage log."""
    print(line)
    if line.strip():
        LOGGER.info(line.strip())


def print_report(report: Dict[str, Any]) -> None:
    intr = report["intrinsic"]
    _say("\n================== Citation quality (title textual PRF) ================")
    prf = report.get("reference_prf")
    if prf:
        _say(f"threshold >= {prf['threshold']:.2f} | gold refs: {prf['gold_ref_count']} | "
             f"selected: {prf['selected_count']} | matched: {prf['matched_count']}")
        _say(f"Precision: {prf['precision']:.4f} | Recall: {prf['recall']:.4f} | F1: {prf['f1']:.4f}")
    else:
        _say("skipped (no usable gold standard)")

    _say("\n================== Structural consistency (semantic) ==================")
    struct = report.get("structural_consistency_semantic")
    if struct:
        _say(f"gold sections (filtered): {struct['gold_sections']} | generated titles: "
             f"{struct['generated_sections']} | matched: {struct['matched_sections']}")
        _say(f"Consistency = {struct['matched_sections']} / ({struct['generated_sections']} + "
             f"{struct['gold_sections']}) = {struct['consistency']:.4f}")
        for p in (struct.get("pairs") or [])[:10]:
            _say(f"  - GEN[{p['gen_idx']}] '{p['gen_title']}' <-> GOLD[{p['gold_idx']}] "
                 f"'{p['gold_title']}' | sim={p['similarity']:.3f}")
    else:
        _say("skipped (no usable gold standard)")

    llm = report.get("structural_consistency_llm") or {}
    _say("\n================== Structural consistency (LLM 1-5) ===================")
    if llm.get("score") is not None:
        _say(f"LLM structural consistency score: {llm['score']}/5 (model={llm.get('model')})")
    else:
        _say(f"LLM score unavailable: {llm.get('error')}")

    _say("\n================== Content quality ===================================")
    quality = report.get("content_quality")
    if quality:
        rouge = quality.get("rouge")
        if rouge:
            for name in ("rouge1", "rouge2", "rougeLsum"):
                _say(f"{name:<9} P/R/F: {rouge[name]['p']:.4f} / {rouge[name]['r']:.4f} / "
                     f"{rouge[name]['f']:.4f}")
        else:
            _say("ROUGE skipped (rouge_score unavailable)")
        _say(f"Semantic similarity (full document): {quality['semantic_similarity']:.4f}")
    else:
        _say("skipped (no usable gold standard)")

    _say("\n================== Intrinsic metrics (gold-free) =====================")
    _say(f"Generated sections: {intr['n_generated_sections']} | subsections: "
         f"{intr['n_generated_subsections']} "
         f"(with content: {intr['n_generated_subsections_with_content']})")
    _say(f"Selected refs: {intr['n_selected_refs']} | distinct refs cited: "
         f"{intr['n_distinct_refs_cited']} ({intr['n_citation_markers']} markers) | "
         f"coverage: {intr['citation_coverage']:.4f}")
    _say(f"Section semantic coherence (mean pairwise cosine): "
         f"{intr['mean_section_semantic_coherence']:.4f}")
    _say(f"Words: {intr['generated_word_count']} | mean per subsection: "
         f"{intr['mean_words_per_subsection']:.1f}")

    meta = report.get("meta", {})
    _say("\n================== Evaluation metadata ================================")
    _say(f"Embedding backend: {meta.get('embedding_backend')} | gold present: "
         f"{meta.get('gold_present')} | gold topic mismatch: {meta.get('gold_topic_mismatch')}")
    for w in report.get("warnings") or []:
        _say(f"WARNING: {w}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _try_load(label: str, loader: Any, warnings: List[str], *args: Any, **kwargs: Any) -> Any:
    try:
        return loader(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — missing step outputs degrade, never crash
        msg = f"{label} unavailable: {type(exc).__name__}: {exc}"
        LOGGER.warning(msg)
        warnings.append(msg)
        return None


def main(run_llm: bool = True) -> Dict[str, Any]:
    """Compute every metric, write ``config.CANONICAL_EVALUATION`` + ``STEP6_REPORT``."""
    config.ensure_dirs()
    warnings: List[str] = []

    # --- topic ------------------------------------------------------------- #
    topic = _try_load("topic CSV", common.load_topic, warnings)
    topic_title = topic.title if topic else ""
    topic_keywords = topic.keywords if topic else ""

    # --- generated artefacts ------------------------------------------------ #
    outline = _try_load("generated outline", load_outline, warnings, config.CANONICAL_OUTLINE) or {}
    if not topic_title:
        topic_title = str(outline.get("title") or "").strip()
    content_rows = (
        _try_load("generated content", load_generated_content, warnings,
                  config.CANONICAL_CONTENT, config.CANONICAL_SECTIONS_DIR) or []
    )
    gen_text = content_fulltext(content_rows)
    selected_titles = (
        _try_load("ranked CSV", load_selected_titles, warnings, config.CANONICAL_RANKED_CSV) or []
    )
    if not gen_text:
        warnings.append("generated survey text is empty; content metrics are degenerate")

    if not topic_title:
        warnings.append("topic title unresolved (topic CSV and outline both empty)")

    # --- gold standard (optional) ------------------------------------------- #
    gold: Optional[Dict[str, Any]] = None
    gold_present = config.GOLD_JSON.exists()
    if gold_present:
        gold = _try_load("gold standard", common.read_json, warnings, config.GOLD_JSON)
        gold_present = gold is not None
    else:
        msg = f"gold standard not found at {config.GOLD_JSON}; gold-dependent metrics skipped"
        LOGGER.warning(msg)
        warnings.append(msg)

    gold_ref_titles: List[str] = []
    gold_sec_titles: List[str] = []
    gold_text = ""
    gold_title = ""
    mismatch = False
    gold_overlap = 0.0
    if gold is not None:
        gold_ref_titles = extract_gold_reference_titles(gold)
        gold_sec_titles = extract_gold_section_titles_filtered(gold)
        gold_text = extract_gold_fulltext(gold)
        gold_title = gold_title_hint(gold)
        gold_overlap = token_overlap(gold_title, f"{topic_title} {topic_keywords}")
        mismatch = bool(gold_title) and gold_overlap < GOLD_TOPIC_OVERLAP_THR
        if mismatch:
            msg = (
                f"gold standard is for a different topic ('{gold_title}', overlap "
                f"{gold_overlap:.2f} < {GOLD_TOPIC_OVERLAP_THR}); gold-dependent scores are "
                "not comparable to the configured survey"
            )
            LOGGER.warning(msg)
            warnings.append(msg)

    has_comparable_gold = bool(gold is not None and gold_ref_titles and gold_text)

    # --- embeddings --------------------------------------------------------- #
    embedder = Embedder(config.EMBED_MODEL_NAME)
    if embedder.backend != "sentence-transformers":
        warnings.append(
            f"sentence-transformers unavailable; embedding_backend='{embedder.backend}'"
        )

    # --- metrics ------------------------------------------------------------ #
    prf = (
        compute_reference_prf(gold_ref_titles, selected_titles)
        if (gold is not None and gold_ref_titles) else None
    )
    struct = (
        structural_consistency(outline.get("sections", []) + outline.get("subsections", []),
                               gold_sec_titles, embedder)
        if (gold is not None and gold_sec_titles and (outline.get("sections") or outline.get("subsections")))
        else None
    )
    quality = compute_content_quality(gen_text, gold_text, embedder) if has_comparable_gold else None
    if quality is None and gold is not None and not has_comparable_gold:
        warnings.append("gold standard has no reference titles/full text; content+PRF metrics skipped")

    llm_info: Dict[str, Any] = {"score": None, "model": None, "cached": None, "error": None}
    if not gold_present:
        llm_info["error"] = "no gold standard"
    elif not run_llm:
        llm_info["error"] = "disabled by --no-llm"
    elif not gold_sec_titles:
        llm_info["error"] = "gold outline has no content sections"
    else:
        llm_info = llm_structure_score(
            build_llm_structure_prompt(
                topic_title,
                list(outline.get("sections", [])) + list(outline.get("subsections", [])),
                gold_sec_titles,
            )
        )
        if llm_info.get("error"):
            warnings.append(f"LLM structure score unavailable: {llm_info['error']}")

    intrinsic = intrinsic_metrics(outline, content_rows, gen_text, selected_titles, embedder)

    report: Dict[str, Any] = {
        "meta": {
            "topic": topic_title,
            "gold_path": str(config.GOLD_JSON),
            "gold_present": gold_present,
            "gold_title": gold_title,
            "gold_topic_overlap": gold_overlap,
            "gold_topic_mismatch": mismatch,
            "embedding_backend": embedder.backend,
            "embedding_model": embedder.model_name if embedder.backend == "sentence-transformers" else "tfidf",
            "thresholds": {
                "title_textual": config.TITLE_TEXTUAL_THR,
                "section_semantic": config.SECTION_SEM_THR,
            },
        },
        "gold_topic_mismatch": mismatch,
        "warnings": warnings,
        "reference_prf": prf,
        "structural_consistency_semantic": struct,
        "structural_consistency_llm": llm_info,
        "structural_consistency_llm_score": llm_info.get("score"),
        "content_quality": quality,
        "intrinsic": intrinsic,
        "inputs": {
            "ranked_csv": str(config.CANONICAL_RANKED_CSV),
            "outline": str(config.CANONICAL_OUTLINE),
            "content": str(config.CANONICAL_CONTENT),
            "n_content_rows": len(content_rows),
            "gold_refs": len(gold_ref_titles),
            "gold_sections": len(gold_sec_titles),
            "gold_words": count_words(gold_text),
            "generated_words": intrinsic["generated_word_count"],
        },
    }

    print_report(report)

    common.write_json(config.CANONICAL_EVALUATION, report)
    summary = {
        "intrinsic": intrinsic,
        "reference_prf": ({k: v for k, v in prf.items() if k != "matches"} if prf else None),
        "structural_consistency": (
            {k: v for k, v in struct.items() if k != "pairs"} if struct else None
        ),
        "structural_consistency_llm_score": llm_info.get("score"),
        "content_quality": quality,
        "gold_topic_mismatch": mismatch,
        "embedding_backend": embedder.backend,
        "warnings": warnings,
    }
    common.write_json(config.STEP6_REPORT, summary)

    LOGGER.info("Evaluation written to %s and %s", config.CANONICAL_EVALUATION, config.STEP6_REPORT)
    _say(f"\nEvaluation report -> {config.CANONICAL_EVALUATION}")
    _say(f"Metrics summary   -> {config.STEP6_REPORT}")
    return report


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 6: evaluate the generated survey.")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM structural rubric call")
    return parser.parse_args()


if __name__ == "__main__":
    main(run_llm=not _cli().no_llm)
