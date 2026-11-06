"""Merge a user-supplied paper's bibliography into the candidate pool.

Runs without OpenAlex: each bibliography entry is resolved through **Crossref**
(free, rate-friendly) and appended to ``final_ref_set.json`` as an
``origin="anchor_ref"`` record.  Entries Crossref returns without an abstract are
kept anyway — step 3 judges them on title alone (flagged ``title_only``), which is
enough for dataset/benchmark papers that a survey must cite.

This is the source-independent half of anchor support; it can run while OpenAlex
is throttled and it never touches any paper-specific logic.

    python3 -m qual_sg.anchor_merge --pdf "../2403.12570v1.pdf"
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

from . import anchor as anchor_mod
from . import common, config, crossref

LOGGER = common.get_logger("anchor_merge")


def flag_cited_by_anchor(pool: List[Dict[str, Any]], refs: List[Dict[str, Any]]) -> int:
    """Mark pool records that appear in the anchor's bibliography.

    A record may have entered the pool by ordinary retrieval and still be cited by
    the anchor.  ``cited_by_anchor`` records that provenance, which step 3 uses as
    a prior (papers the anchor cites are part of its citation context).
    """
    from . import bib_eval

    ref_texts = [r["text"] for r in refs]
    if not ref_texts:
        return 0
    flagged = 0
    for rec in pool:
        best_num, best_score = None, 0.0
        for ref in refs:
            ok, score = bib_eval.accept_match(rec.get("title", ""), ref["text"])
            if ok and score > best_score:
                best_num, best_score = ref.get("num"), score
        if best_num is not None:
            rec["cited_by_anchor"] = True
            rec["anchor_ref_num"] = best_num
            flagged += 1
        else:
            rec.pop("cited_by_anchor", None)
    return flagged


def flag_only(refs: List[Dict[str, Any]]) -> Dict[str, Any]:
    pool: List[Dict[str, Any]] = common.read_json(config.CANONICAL_REF_SET)
    flagged = flag_cited_by_anchor(pool, refs)
    common.write_json(config.CANONICAL_REF_SET, pool)
    stats = {"bibliography_entries": len(refs), "flagged_cited_by_anchor": flagged, "pool_total": len(pool)}
    LOGGER.info("anchor_merge --flag-only: %s", stats)
    return stats


def merge(anchor: Dict[str, Any], client: common.HttpClient) -> Dict[str, Any]:
    pool: List[Dict[str, Any]] = common.read_json(config.CANONICAL_REF_SET)

    # Drop earlier anchor-bibliography records that no longer pass the strict
    # title/reference match (false positives such as "Uller"); they are re-resolved
    # below with the stricter guard.
    from . import bib_eval
    kept, dropped_bad = [], 0
    for rec in pool:
        if rec.get("origin") == "anchor_ref" and rec.get("anchor_ref_text"):
            ok, _ = bib_eval.accept_match(rec.get("title", ""), rec["anchor_ref_text"])
            if not ok:
                dropped_bad += 1
                LOGGER.warning("dropping bad anchor resolution: %r (ref: %s)", rec.get("title"), rec["anchor_ref_text"][:60])
                continue
        kept.append(rec)
    pool = kept

    known_doi = {common.norm_doi(r.get("doi")) for r in pool if r.get("doi")}
    known_title = {common.normalize_text(r.get("title", "")) for r in pool}

    refs = anchor.get("bibliography_raw") or []
    added, unresolved, duplicates, title_only = 0, 0, 0, 0
    for ref in refs:
        rec = anchor_mod.resolve_reference(ref["text"], client)
        if not rec:
            unresolved += 1
            LOGGER.info("anchor_merge [%s] unresolved", ref.get("num"))
            continue
        doi, title = common.norm_doi(rec.get("doi")), common.normalize_text(rec.get("title", ""))
        if (doi and doi in known_doi) or (title and title in known_title):
            duplicates += 1
            continue
        rec["origin"] = "anchor_ref"
        rec["co_cited_count"] = 0
        rec["anchor_ref_num"] = ref.get("num")
        rec["anchor_ref_text"] = ref["text"]
        rec["title_only"] = not bool(rec.get("abstract"))
        if rec["title_only"]:
            title_only += 1
        # Fields step 3 expects (unknown for Crossref-only records -> None/NaN).
        rec.setdefault("topics", [])
        rec.setdefault("keywords", [])
        rec.setdefault("referenced_works", [])
        rec.setdefault("referenced_works_count", 0)
        rec.setdefault("cited_by_count", rec.get("citation_count"))
        rec.setdefault("first_author_id", None)
        rec.setdefault("last_author_id", None)
        rec.setdefault("journal_id", None)
        rec["first_author_hindex"] = rec.get("first_author_hindex")
        rec["last_author_hindex"] = rec.get("last_author_hindex")
        rec["journal_hindex"] = rec.get("journal_hindex")
        pool.append(rec)
        added += 1
        if doi:
            known_doi.add(doi)
        known_title.add(title)
        time.sleep(0.15)

    flagged = flag_cited_by_anchor(pool, refs)
    common.write_json(config.CANONICAL_REF_SET, pool)
    stats = {
        "bibliography_entries": len(refs),
        "added": added,
        "duplicates_already_in_pool": duplicates,
        "unresolved": unresolved,
        "dropped_bad_resolutions": dropped_bad,
        "title_only": title_only,
        "flagged_cited_by_anchor": flagged,
        "pool_total": len(pool),
    }
    LOGGER.info("anchor_merge: %s", stats)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge a paper's bibliography into the candidate pool")
    ap.add_argument("--pdf", default=config.ANCHOR_PDF or None, help="path to the anchor paper PDF")
    ap.add_argument("--doi", default=config.ANCHOR_DOI or None, help="anchor DOI (Crossref metadata only)")
    ap.add_argument("--flag-only", action="store_true", help="only (re)derive cited_by_anchor flags")
    args = ap.parse_args()
    config.ensure_dirs()
    config.ANCHOR_PDF = args.pdf or config.ANCHOR_PDF
    if not config.ANCHOR_PDF:
        raise SystemExit("provide --pdf (or set QUAL_SG_ANCHOR_PDF)")
    client = common.HttpClient(logger=LOGGER)
    a = anchor_mod.load_anchor(client)
    if not a:
        raise SystemExit("could not load the anchor")
    stats = flag_only(a["bibliography_raw"]) if args.flag_only else merge(a, client)
    print(stats)


if __name__ == "__main__":
    main()
