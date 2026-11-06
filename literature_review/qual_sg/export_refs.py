"""Export the pipeline's own reference set.

``final_ref_set.json`` is the *candidate pool* (all the papers the pipeline
retrieved, enriched and considered).  This module exports the much smaller
artefact that users actually want to consume: the **K references the pipeline
selected**, in citation order, with the provenance and scores behind each choice.

    python3 -m qual_sg.export_refs            # -> pipeline_ref.json
    python3 -m qual_sg.export_refs --out x.json

The numbering matches the numbering used by the generated review
(``literature_review_2page.json`` / ``.md``), so ``ref [7]`` in the prose is
``refNo 7`` here.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import common, config

ANCHOR_JSON = config.INTERMEDIATES_DIR / "anchor.json"


def _year(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    return int(text[:4]) if text[:4].isdigit() else None


def build(out_csv: Optional[Path] = None, anchor_priority: Optional[str] = None, sort_by: str = "relevance") -> Dict[str, Any]:
    out_csv = Path(out_csv or config.CANONICAL_RANKED_CSV)
    df = pd.read_csv(out_csv)
    selected = df[df["tag"] == "selected"].copy()

    pool: Dict[str, Dict[str, Any]] = {}
    if config.CANONICAL_REF_SET.exists():
        for rec in common.read_json(config.CANONICAL_REF_SET):
            pool[common.norm_doi(rec.get("doi")) or common.normalize_text(rec.get("title", ""))] = rec

    refs: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(selected.iterrows(), start=1):
        key = common.norm_doi(row.get("doi")) or common.normalize_text(row.get("title", ""))
        rec = pool.get(key, {})
        refs.append(
            {
                "refNo": i,
                "refNoInReview": i,
                "title": str(row.get("title", "")).strip(),
                "doi": common.norm_doi(row.get("doi")),
                "year": _year(row.get("pub_year")) or _year(rec.get("publication_date")),
                "venue": rec.get("journal_name") or None,
                "citations": None if pd.isna(row.get("cited_by_count")) else int(row["cited_by_count"]),
                "citations_per_year": None if pd.isna(row.get("citations_per_year")) else round(float(row["citations_per_year"]), 2),
                "relevance_llm_1to5": None if pd.isna(row.get("relevance_llm_1to5")) else int(row["relevance_llm_1to5"]),
                "cited_by_anchor": bool(row.get("cited_by_anchor")) if not pd.isna(row.get("cited_by_anchor")) else False,
                "anchor_ref_num": int(rec["anchor_ref_num"]) if rec.get("anchor_ref_num") else None,
                "origin": row.get("origin"),
                "avg_rank": None if pd.isna(row.get("avg_rank")) else round(float(row["avg_rank"]), 2),
                "composite_score": None if pd.isna(row.get("composite_score")) else round(float(row["composite_score"]), 4),
                "abstract": ("" if pd.isna(row.get("abstract")) else str(row.get("abstract", "")).strip()),
            }
        )

    # --- ordering ------------------------------------------------------------- #
    # Default: highest topical-relevance first (S_t descending, ties broken by
    # composite score then citations/year).  `refNo` is renumbered in the new
    # order; `refNoInReview` keeps the number used by the generated review.
    order_mode = "relevance_llm_1to5 desc (tie-break composite_score, citations_per_year)"
    if sort_by == "rank":
        order_mode = "pipeline avg_rank asc (review citation order)"
    else:
        refs.sort(key=lambda r: (
            -(r["relevance_llm_1to5"] if r["relevance_llm_1to5"] is not None else -1),
            -(r["composite_score"] if r["composite_score"] is not None else -1.0),
            -(r["citations_per_year"] if r["citations_per_year"] is not None else -1.0),
            r["refNoInReview"],
        ))
        for new_no, r in enumerate(refs, start=1):
            r["refNo"] = new_no

    anchor = common.read_json(ANCHOR_JSON) if ANCHOR_JSON.exists() else None
    overlap = None
    # use the most recently written overlap measurement for the current selection
    candidates = sorted(config.INTERMEDIATES_DIR.glob("bib_overlap*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        s = (common.read_json(p) or {}).get("selected")
        if s:
            overlap = {"gold": p.name, "n_gold_refs": s["n_gold_refs"], "matched": s["matched"],
                       "recall": s["recall"], "precision": s["precision"], "f1": s["f1"]}
            break

    topic = common.load_topic()
    return {
        "name": "pipeline_ref.json - references selected by the implemented QUAL-SG pipeline",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "topic": {"title": topic.title, "year": topic.year, "keywords": topic.keywords, "max_refs": topic.max_refs},
        "pipeline": {
            "stages": ["step1_retrieve", "step2_quality", "step3_rerank"],
            "source": "QUAL-SG (arXiv:2508.17647) reimplementation, quality-aware retrieval + re-ranking",
            "candidate_pool_size": len(pd.read_csv(out_csv)),
            "K": len(refs),
            "selection": config.SELECTION,
            "anchor_priority": anchor_priority or config.ANCHOR_REF_PRIORITY,
            "reference_order": order_mode,
            "relevance_gate": config.RELEVANCE_GATE,
            "mmr_lambda": config.MMR_LAMBDA,
            "weights": {"cited": config.W_CITED, "author": config.W_AUTHOR, "venue": config.W_VENUE,
                        "relevance": config.W_RELEVANCE, "diversity": config.W_DIVERSITY},
            "cite_age_normalized": config.CITE_AGE_NORMALIZE,
            "embedding_model": config.EMBED_MODEL_NAME,
            "judge": "LLM-as-judge, 5-point topical-relevance rubric",
            "anchor": ({"spec": anchor.get("anchor_spec"), "title": anchor.get("title")} if anchor else None),
            "reranked_csv": str(out_csv.name),
            "candidate_pool": config.CANONICAL_REF_SET.name,
        },
        "overlap_vs_anchor_bibliography": overlap,
        "references": refs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export the pipeline's selected reference set")
    ap.add_argument("--csv", default=None, help="ranked CSV to read (default: the canonical one)")
    ap.add_argument("--out", default=None, help="output path (default: pipeline_ref.json)")
    ap.add_argument("--anchor-priority", default=None, choices=["off", "boost", "strict"],
                    help="anchor mode that produced the ranked CSV (recorded in the output)")
    ap.add_argument("--sort", default="relevance", choices=["relevance", "rank"],
                    help="order of the references array: relevance (default) or pipeline rank")
    args = ap.parse_args()
    config.ensure_dirs()
    obj = build(Path(args.csv) if args.csv else None, args.anchor_priority, args.sort)
    out = Path(args.out) if args.out else (config.OUT_DIR / "pipeline_ref.json")
    common.write_json(out, obj)
    print(f"wrote {out} | {len(obj['references'])} references | overlap={obj['overlap_vs_anchor_bibliography']}")


if __name__ == "__main__":
    main()
