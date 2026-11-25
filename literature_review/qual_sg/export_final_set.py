"""Write ``final_ref_set.json``: the one consumable output of a run, with a fixed format.

A JSON array with one object per candidate paper, every object with exactly
these fields (no others, none empty):

    doi              str   verified DOI (resolves to this paper; arXiv DOIs are 10.48550/arxiv.*)
    title            str
    abstract         str
    year             int   publication year
    venue            str   journal / proceedings, "arXiv" for preprints
    citation_count   int   OpenAlex cited_by_count (Semantic Scholar when OpenAlex has none)
    keywords         [str] OpenAlex keywords
    topics           [str] OpenAlex topics
    relevance_score  int   LLM-judge topical relevance, 1-5
    composite_score  float step-3 weighted composite (impact, relevance, diversity), 0-1
    origin           [str] how the paper entered the pool, primary route first:
                           "search", "<anchor>_reference" (anchor's bibliography),
                           "co_citation", "snowball", "cites_<anchor>" (who cites the anchor);
                           a later tag records a second route ("cites_<anchor>",
                           "<anchor>_reference"); "anchor" marks the anchor paper itself
    selected         bool  true for the papers the review cites (its reference list)

Order: selected papers first (in the review's numbering), then by relevance and
composite score.  Candidates that cannot satisfy the format (no abstract, DOI
that does not resolve to the paper, no relevance score) and duplicates are left
out and listed, with the reason, in ``intermediates/final_ref_set_excluded.json``.

    python3 -m qual_sg.export_final_set
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from . import common, config

LOGGER = common.get_logger("export_final_set")
EXCLUDED = config.INTERMEDIATES_DIR / "final_ref_set_excluded.json"
LABEL = os.environ.get("QUAL_SG_ANCHOR_LABEL", "anchor").strip().lower() or "anchor"
FIELDS = list(config.FINAL_REF_SET_FIELDS)


def _tkey(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _origin(rec: Dict[str, Any]) -> List[str]:
    o, ro = rec.get("origin"), rec.get("retrieval_origin")
    if rec.get("is_anchor"):
        tags = ["anchor"]
    elif o == "co_cited":
        tags = ["co_citation"]
    elif o == "snowball":
        tags = ["snowball"]
    elif o == "cites_anchor":
        tags = [f"cites_{LABEL}"]
    elif ro == "anchor_ref":
        tags = [f"{LABEL}_reference"]
    else:
        tags = ["search"]
    if rec.get("cited_by_anchor") and f"{LABEL}_reference" not in tags and "anchor" not in tags:
        tags.append(f"{LABEL}_reference")
    if rec.get("cites_anchor") and f"cites_{LABEL}" not in tags:
        tags.append(f"cites_{LABEL}")
    return tags


def main() -> None:
    config.ensure_dirs()
    if config.CANONICAL_REF_SET.resolve() == config.FINAL_REF_SET.resolve():
        raise SystemExit(
            f"the candidate pool is {config.FINAL_REF_SET} (pre-v6 layout); writing the export there would "
            "overwrite it. Rename the pool to candidate_pool.json or set QUAL_SG_POOL_JSON.")
    pool: List[Dict[str, Any]] = common.read_json(config.CANONICAL_REF_SET)
    csv = pd.read_csv(config.CANONICAL_RANKED_CSV)
    scores: Dict[str, Dict[str, Any]] = {}
    for _, row in csv.iterrows():
        key = common.norm_doi(row.get("doi")) or _tkey(row.get("title"))
        scores[key] = {"rel": _num(row.get("relevance_llm_1to5")), "comp": _num(row.get("composite_score")),
                       "selected": str(row.get("tag")) == "selected"}

    review_no: Dict[str, int] = {}
    ref_json = config.FIXED_SELECTION or (config.OUT_DIR / "pipeline_ref.json")
    if ref_json.exists():
        for r in common.read_json(ref_json)["references"]:
            n = int(r.get("refNoInReview") or r["refNo"])
            review_no[common.norm_doi(r.get("doi"))] = n
            review_no[_tkey(r.get("title"))] = n

    out: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}
    for rec in pool:
        doi = common.norm_doi(rec.get("doi"))
        sc = scores.get(doi) or scores.get(_tkey(rec.get("title"))) or {}
        cc = rec.get("cited_by_count")
        if cc in (None, ""):
            cc = rec.get("citation_count")
        row = {
            "doi": doi,
            "title": re.sub(r"\s+", " ", str(rec.get("title") or "")).strip(),
            "abstract": re.sub(r"\s+", " ", str(rec.get("abstract") or "")).strip(),
            "year": int(rec["year"]) if _num(rec.get("year")) else None,
            "venue": (str(rec.get("venue") or "").strip() or None),
            "citation_count": int(_num(cc)) if _num(cc) is not None else None,
            "keywords": [k for k in (rec.get("keywords") or []) if k],
            "topics": [t for t in (rec.get("topics") or []) if t],
            "relevance_score": int(sc["rel"]) if sc.get("rel") is not None else None,
            "composite_score": round(sc["comp"], 4) if sc.get("comp") is not None else None,
            "origin": _origin(rec),
            "selected": bool(sc.get("selected")),
        }
        reasons = [f for f in FIELDS if row[f] in (None, "", [])]
        if rec.get("doi_verified") is False:
            reasons.append("doi_not_verified")
        if rec.get("is_retracted"):
            reasons.append("retracted")
        if reasons:
            excluded.append({"title": row["title"], "doi": doi or None, "selected": row["selected"],
                             "missing_or_invalid": reasons})
            continue
        # duplicates (same DOI or same title): keep the selected / better-scored record, merge origins
        dkey = [k for k in (doi, _tkey(row["title"])) if k and k in seen]
        if dkey:
            prev = out[seen[dkey[0]]]
            # keep the selected / better-scored record; on a tie, the published version (non-arXiv DOI)
            rank = lambda r: (r["selected"], r["relevance_score"], not r["doi"].startswith("10.48550/"))  # noqa: E731
            keep, drop = (row, prev) if rank(row) > rank(prev) else (prev, row)
            keep["origin"] += [t for t in drop["origin"] if t not in keep["origin"]]
            out[seen[dkey[0]]] = keep
            excluded.append({"title": drop["title"], "doi": drop["doi"], "selected": drop["selected"],
                             "missing_or_invalid": [f"duplicate_of:{keep['doi']}"]})
            seen[doi] = seen[_tkey(row["title"])] = seen[dkey[0]]
            continue
        seen[doi] = seen[_tkey(row["title"])] = len(out)
        out.append(row)

    out.sort(key=lambda r: (not r["selected"],
                            review_no.get(r["doi"], review_no.get(_tkey(r["title"]), 10**6)) if r["selected"] else 0,
                            -(r["relevance_score"] or 0), -(r["composite_score"] or 0.0)))
    out = [{f: r[f] for f in FIELDS} for r in out]

    n_sel = sum(r["selected"] for r in out)
    expected = len({v for v in review_no.values()}) if review_no else None
    lost_selected = [e for e in excluded if e["selected"] and not any(
        m.startswith("duplicate_of") for m in e["missing_or_invalid"])]
    if lost_selected:
        raise SystemExit(f"selected papers would be left out of final_ref_set.json: {lost_selected}")
    with open(config.FINAL_REF_SET, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    common.write_json(EXCLUDED, {"n_excluded": len(excluded), "excluded": excluded})
    summary = {"papers": len(out), "selected": n_sel, "review_references": expected,
               "excluded": len(excluded), "out": str(config.FINAL_REF_SET)}
    LOGGER.info("final_ref_set: %s", summary)
    print(summary)


if __name__ == "__main__":
    main()
