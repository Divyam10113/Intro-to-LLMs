"""Compare the three 2-page literature reviews (QUAL-SG vs baselines).

Produces ``comparison_report.json`` plus a printable table covering:

* length / structure compliance (words, sections, citation markers, coverage);
* topic fidelity (cosine similarity between the review text and the topic query,
  using the same embedding model as the pipeline);
* reference-set quality (mean LLM relevance judge score, share of references with
  score >= 3, median citation count, mean venue h-index);
* pairwise overlap of the three reference sets.

Run: ``python3 -m qual_sg.compare``  (no LLM calls; local embeddings only).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import common, config

SOURCES = ["qual_sg", "naive_rag", "fully_llm"]


def review_path(source: str) -> Path:
    base = config.BASE_DIR if source == "qual_sg" else (config.BASE_DIR / "baselines" / source)
    return base / "literature_review_2page.json"


def review_text(bundle: Dict[str, Any]) -> str:
    parts = [str(bundle.get("title", "")), str(bundle.get("abstract", ""))]
    parts += [f"{s.get('heading','')}. {s.get('content','')}" for s in bundle.get("sections", [])]
    parts.append(str(bundle.get("takeaway", "")))
    return "\n".join(parts)


def reference_quality(refs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not refs:
        return {"n": 0}
    scores = {}
    for row in common.read_jsonl(config.STEP3_SCORES):
        key = str(row.get("key") or common.norm_doi(row.get("doi")))
        if key:
            scores[key] = row.get("score")
    pool = {}
    if config.CANONICAL_REF_SET.exists():
        for rec in common.read_json(config.CANONICAL_REF_SET):
            pool[common.norm_doi(rec.get("doi"))] = rec
    rel, cited, jh = [], [], []
    for r in refs:
        doi = common.norm_doi(r.get("doi"))
        s = scores.get(doi)
        if s is not None:
            rel.append(float(s))
        rec = pool.get(doi, {})
        if isinstance(rec.get("cited_by_count"), (int, float)):
            cited.append(float(rec["cited_by_count"]))
        if isinstance(rec.get("journal_hindex"), (int, float)):
            jh.append(float(rec["journal_hindex"]))
    return {
        "n": len(refs),
        "mean_relevance": round(float(np.mean(rel)), 3) if rel else None,
        "share_relevance_ge3": round(float(np.mean([x >= 3 for x in rel])), 3) if rel else None,
        "median_citations": float(np.median(cited)) if cited else None,
        "mean_venue_hindex": round(float(np.mean(jh)), 2) if jh else None,
    }


def main() -> None:
    config.ensure_dirs()
    topic = common.load_topic()
    bundles, texts, refsets = {}, {}, {}
    for src in SOURCES:
        p = review_path(src)
        if not p.exists():
            print(f"missing {p}")
            continue
        b = json.loads(p.read_text(encoding="utf-8"))
        bundles[src] = b
        texts[src] = review_text(b)
        refsets[src] = [common.norm_doi(r.get("doi")) or common.normalize_text(r.get("title")) for r in b.get("references", [])]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.EMBED_MODEL_NAME)
    topic_vec = model.encode([topic.query], normalize_embeddings=True)[0]

    report: Dict[str, Any] = {"topic": topic.title, "sources": {}}
    for src, b in bundles.items():
        text = texts[src]
        refs = b.get("references", [])
        markers = [int(m) for m in re.findall(r"ref \[(\d+)\]", text)]
        distinct = sorted(set(markers))
        n_avail = len(refs)
        vec = model.encode([text], normalize_embeddings=True)[0]
        entry = {
            "title": b.get("title"),
            "words": int(sum(len(str(s.get("content", "")).split()) for s in b.get("sections", []))),
            "n_sections": len(b.get("sections", [])),
            "n_refs_available": n_avail,
            "citation_markers": len(markers),
            "distinct_refs_cited": len(distinct),
            "out_of_range_citations": sorted({m for m in markers if m < 1 or m > n_avail}),
            "citation_coverage": round(len(distinct) / n_avail, 3) if n_avail else None,
            "topic_similarity": round(float(np.dot(vec, topic_vec)), 4),
            "reference_quality": reference_quality(refs),
        }
        report["sources"][src] = entry

    # pairwise reference-set overlap (Jaccard)
    report["reference_set_overlap"] = {}
    for i, a in enumerate(SOURCES):
        for b_ in SOURCES[i + 1:]:
            if a in refsets and b_ in refsets and refsets[a] and refsets[b_]:
                sa, sb = set(refsets[a]), set(refsets[b_])
                inter = len(sa & sb)
                report["reference_set_overlap"][f"{a}__{b_}"] = {
                    "shared": inter, "jaccard": round(inter / len(sa | sb), 3),
                    "share_of_%s" % a: round(inter / len(sa), 3), "share_of_%s" % b_: round(inter / len(sb), 3),
                }

    common.write_json(config.BASE_DIR / "comparison_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
