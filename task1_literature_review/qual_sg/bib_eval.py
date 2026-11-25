"""Bibliography-overlap evaluation.

Measures how much of a *given* publication's reference list our pipeline
rediscovers.  This is the generic counterpart of the paper's citation-quality
metric: instead of requiring the SurveyGen dataset's human survey, any paper the
user supplies (PDF/DOI/text) can serve as the reference standard.

Matching is title-based and robust to author lists, venues and formatting: a
candidate title matches a bibliography entry when the longest common substring
covers at least ``thr`` of the (normalised) title.

CLI
---
    python3 -m qual_sg.bib_eval --pdf "../2403.12570v1.pdf"          # pdf -> refs
    python3 -m qual_sg.bib_eval --pdf X.pdf --selected               # our top-K
    python3 -m qual_sg.bib_eval --pdf X.pdf --pool                   # whole D_ex

No LLM or network access is used.
"""
from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from . import common, config

LOGGER = common.get_logger("bib_eval")
MATCH_THR = 0.80


# --------------------------------------------------------------------------- #
# Bibliography extraction and matching
# --------------------------------------------------------------------------- #
def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).replace("  ", " ").strip()


def guess_title_from_reference(ref_text: str, max_words: int = 22) -> str:
    """Heuristically pull a title out of a raw bibliography entry."""
    segments = [s.strip() for s in re.split(r"\.\s+", ref_text) if s.strip()]
    candidates = []
    for s in segments:
        low = s.lower()
        if low.startswith(("in ", "arxiv", "proceedings", "ieee", "acm", "journal")):
            continue
        words = s.split()
        if len(words) < 4:
            continue
        if s.count(",") / max(1, len(words)) > 0.35:   # author lists are comma-dense
            continue
        candidates.append(s)
    if not candidates:
        candidates = segments
    if not candidates:
        return ""
    best = max(candidates, key=lambda s: len(s.split()))
    return " ".join(best.split()[:max_words])


def parse_pdf_references(pdf_path: Path, page_hint: Optional[str] = "References") -> List[Dict[str, Any]]:
    """Extract ``[n] ...`` bibliography entries from a PDF (pypdf)."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = [(i, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    start = next((i for i, t in pages if page_hint and page_hint in t), 0) if page_hint else 0
    text = "\n".join(t for i, t in pages if i >= start)
    parts = re.split(r"\[(\d+)\]\s*", text)
    entries: List[Dict[str, Any]] = []
    for i in range(1, len(parts) - 1, 2):
        num, body = int(parts[i]), " ".join(parts[i + 1].split())
        if body:
            entries.append({"num": num, "text": body})
    # keep strictly increasing sequence [1..N] and stop at the first big regression
    seen: Dict[int, Dict[str, Any]] = {}
    for e in entries:
        seen.setdefault(e["num"], e)
    ordered = [seen[n] for n in sorted(seen) if 1 <= n <= 200]
    return ordered


def fuzzy_match(title: str, ref_texts: Sequence[str], thr: float = MATCH_THR) -> Tuple[Optional[int], float]:
    """Return ``(index, score)`` of the best bibliography match for ``title``."""
    t = normalize(title)
    if not t:
        return None, 0.0
    best_i, best_s = None, 0.0
    for i, ref in enumerate(ref_texts):
        r = normalize(ref)
        if not r:
            continue
        m = SequenceMatcher(None, t, r).find_longest_match(0, len(t), 0, len(r))
        score = m.size / len(t)
        if score > best_s:
            best_i, best_s = i, score
    return (best_i if best_s >= thr else None), round(float(best_s), 3)


def accept_match(title: str, ref_text: str, min_words: int = 4,
                 title_ratio: float = 0.75, ref_ratio: float = 0.15) -> Tuple[bool, float]:
    """Guard against false-positive bibliography resolutions.

    A candidate is accepted only when the longest common substring covers
    ``title_ratio`` of the candidate title **and** ``ref_ratio`` of the reference
    string, and the title has at least ``min_words`` words.  Without the second and
    third conditions short generic titles swallow author names (e.g. the reference
    for *Deep Semi-Supervised Anomaly Detection* resolving to a paper called
    "Uller", taken from "M ¨uller") or partial phrases ("Personalized federated
    learning strategies" for *Grace: A Generalized and Personalized Federated
    Learning*).
    """
    t, r = normalize(title), normalize(ref_text)
    if len(t.split()) < min_words or not t or not r:
        return False, 0.0
    m = SequenceMatcher(None, t, r).find_longest_match(0, len(t), 0, len(r))
    tr = m.size / len(t)
    rr = m.size / len(r)
    return (tr >= title_ratio and rr >= ref_ratio), round(float(min(tr, rr * 4)), 3)


def evaluate(titles: Sequence[str], ref_texts: Sequence[str], thr: float = MATCH_THR) -> Dict[str, Any]:
    matched, details = set(), []
    for t in titles:
        i, s = fuzzy_match(t, ref_texts, thr)
        details.append({"title": t, "matched_ref_index": i, "score": s})
        if i is not None:
            matched.add(i)
    tp = len(matched)
    n_sel, n_gold = len(titles), len(ref_texts)
    prec = tp / n_sel if n_sel else 0.0
    rec = tp / n_gold if n_gold else 0.0
    return {
        "n_gold_refs": n_gold,
        "n_candidates": n_sel,
        "matched": tp,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
        "details": details,
    }


# --------------------------------------------------------------------------- #
# Convenience loaders
# --------------------------------------------------------------------------- #
def selected_titles(csv_path: Optional[Path] = None) -> List[str]:
    df = pd.read_csv(csv_path or config.CANONICAL_RANKED_CSV)
    df = df[df.get("tag") == "selected"]
    return [str(t).strip() for t in df["title"] if isinstance(t, str) and t.strip()]


def pool_titles() -> List[str]:
    records = common.read_json(config.CANONICAL_REF_SET)
    return [str(r.get("title", "")).strip() for r in records if isinstance(r.get("abstract"), str) and r["abstract"].strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Bibliography overlap of the pipeline output vs a given paper")
    ap.add_argument("--pdf", required=True, help="path to the reference paper (PDF)")
    ap.add_argument("--selected", action="store_true", help="evaluate the top-K selection")
    ap.add_argument("--pool", action="store_true", help="evaluate the whole candidate pool")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    args = ap.parse_args()

    refs = parse_pdf_references(Path(args.pdf))
    ref_texts = [r["text"] for r in refs]
    LOGGER.info("parsed %d bibliography entries from %s", len(ref_texts), args.pdf)

    result: Dict[str, Any] = {"pdf": str(args.pdf), "n_gold_refs": len(ref_texts)}
    if args.selected or not args.pool:
        r = evaluate(selected_titles(), ref_texts)
        result["selected"] = r
        LOGGER.info("SELECTED: recall=%.3f precision=%.3f f1=%.3f matched=%d/%d",
                    r["recall"], r["precision"], r["f1"], r["matched"], r["n_gold_refs"])
    if args.pool:
        r = evaluate(pool_titles(), ref_texts)
        result["pool"] = r
        LOGGER.info("POOL: recall=%.3f precision=%.3f f1=%.3f matched=%d/%d",
                    r["recall"], r["precision"], r["f1"], r["matched"], r["n_gold_refs"])
    if args.out:
        common.write_json(Path(args.out), result)
    else:
        print(json.dumps({k: v for k, v in result.items() if k != "selected"}, indent=2, ensure_ascii=False)[:1500])
        if "selected" in result:
            print("\nmatched gold refs:")
            for d in result["selected"]["details"]:
                if d["matched_ref_index"] is not None:
                    print(f"  [{refs[d['matched_ref_index']]['num']}] {d['title'][:70]} (score {d['score']})")
            print("\nMISSED gold refs (first 25):")
            matched = {d["matched_ref_index"] for d in result["selected"]["details"] if d["matched_ref_index"] is not None}
            for i, r in enumerate(refs):
                if i not in matched:
                    print(f"  [{r['num']}] {r['text'][:110]}")


if __name__ == "__main__":
    main()
