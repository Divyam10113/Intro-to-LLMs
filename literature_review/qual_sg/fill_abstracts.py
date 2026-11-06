"""Fill missing abstracts (and missing citation counts) for the candidate pool.

A reference set is only usable if every selected paper has an abstract: without one
the topical-relevance judge and the embedding model have nothing to score, so such
records must either be enriched or excluded.

Sources are tried in order, cheapest/most complete first:

1. **OpenAlex** by DOI, then by OpenAlex id (best coverage; returns the inverted
   abstract index plus `cited_by_count`).
2. **arXiv** by title (preprints that OpenAlex has not indexed).
3. **Semantic Scholar** by DOI (may answer 429; skipped automatically when blocked).
4. **Crossref** by title (JATS abstract when the publisher deposits one).
5. **DataCite** by title (preprint/dataset descriptions).

    python3 -m qual_sg.fill_abstracts            # update final_ref_set.json in place
    python3 -m qual_sg.fill_abstracts --check    # report only
"""
from __future__ import annotations

import argparse
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from . import anchor as anchor_mod
from . import bib_eval, common, config, crossref, datacite, europepmc

LOGGER = common.get_logger("fill_abstracts")


def _has_abstract(rec: Dict[str, Any]) -> bool:
    return isinstance(rec.get("abstract"), str) and rec["abstract"].strip() != ""


def _from_openalex(rec: Dict[str, Any], client: common.HttpClient) -> Tuple[Optional[str], Optional[int]]:
    item = None
    doi = common.norm_doi(rec.get("doi"))
    if doi:
        item = client.get_json(f"{config.OPENALEX_BASE}/works/https://doi.org/{doi}",
                               {"mailto": config.OPENALEX_MAILTO})
    if not (isinstance(item, dict) and item.get("id")) and rec.get("openalex_id"):
        item = client.get_json(f"{config.OPENALEX_BASE}/works/{rec['openalex_id']}",
                               {"mailto": config.OPENALEX_MAILTO})
    if not (isinstance(item, dict) and item.get("id")):
        return None, None
    abstract = common.invert_abstract(item.get("abstract_inverted_index")) or None
    cited = item.get("cited_by_count")
    if not rec.get("openalex_id"):
        rec["openalex_id"] = common.openalex_short_id(item.get("id"))
    if not rec.get("publication_date"):
        rec["publication_date"] = item.get("publication_date") or ""
    return abstract, cited


def _from_arxiv(rec: Dict[str, Any], client: common.HttpClient) -> Optional[str]:
    return (anchor_mod._resolve_arxiv_title(rec.get("title", ""), client) or {}).get("abstract") or None


def _from_s2(rec: Dict[str, Any], client: common.HttpClient) -> Optional[str]:
    doi = common.norm_doi(rec.get("doi"))
    if not doi:
        return None
    data = client.get_json(f"{config.S2_BASE}/paper/DOI:{doi}",
                           {"fields": "abstract,title,citationCount,externalIds"})
    if isinstance(data, dict) and data.get("abstract"):
        if data.get("citationCount") is not None and rec.get("cited_by_count") in (None, ""):
            rec["cited_by_count"] = data["citationCount"]
        return data["abstract"].strip()
    return None


def _from_epmc(rec: Dict[str, Any], client: common.HttpClient) -> Optional[str]:
    """Europe PMC by DOI, then by title (best coverage for medical literature)."""
    got = europepmc.fetch_by_doi(rec.get("doi", ""), client)
    if not got or not got.get("abstract"):
        got = europepmc.fetch_by_title(rec.get("title", ""), client)
    if not got:
        return None
    if got.get("citation_count") is not None and rec.get("cited_by_count") in (None, ""):
        rec["cited_by_count"] = got["citation_count"]
    if not rec.get("pub_year") and got.get("year"):
        rec["pub_year"] = got["year"]
    return got.get("abstract") or None


def _from_crossref(rec: Dict[str, Any], client: common.HttpClient) -> Optional[str]:
    text = rec.get("title", "")
    if len(text.split()) < 3:
        return None
    got = crossref.resolve_reference(text, client)
    return (got or {}).get("abstract") or None


def _from_datacite(rec: Dict[str, Any], client: common.HttpClient) -> Optional[str]:
    got = datacite.resolve_reference(rec.get("title", ""), client)
    return (got or {}).get("abstract") or None


def main() -> None:
    ap = argparse.ArgumentParser(description="Fill missing abstracts in the candidate pool")
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    args = ap.parse_args()
    config.ensure_dirs()

    pool: List[Dict[str, Any]] = common.read_json(config.CANONICAL_REF_SET)
    targets = [r for r in pool if not _has_abstract(r)]
    LOGGER.info("pool=%d | missing abstract=%d", len(pool), len(targets))
    if args.check:
        for r in targets:
            print(f"  [{r.get('origin')}] {str(r.get('title'))[:80]} | doi={r.get('doi')}")
        return

    client = common.HttpClient(logger=LOGGER)
    source_counts: Dict[str, int] = {}
    still_missing: List[Dict[str, Any]] = []
    for i, rec in enumerate(targets, start=1):
        abstract, cited = None, None
        for name, fn in (
            ("openalex", lambda: _from_openalex(rec, client)),
            ("europepmc", lambda: (_from_epmc(rec, client), None)),
            ("arxiv", lambda: (_from_arxiv(rec, client), None)),
            ("semantic_scholar", lambda: (_from_s2(rec, client), None)),
            ("crossref", lambda: (_from_crossref(rec, client), None)),
            ("datacite", lambda: (_from_datacite(rec, client), None)),
        ):
            try:
                abstract, cited = fn()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("%s failed for %r: %s", name, rec.get("title", "")[:50], exc)
                abstract, cited = None, None
            if abstract:
                rec["abstract"] = re.sub(r"\s+", " ", abstract).strip()
                rec["abstract_source"] = name
                source_counts[name] = source_counts.get(name, 0) + 1
                break
        if isinstance(cited, int) and rec.get("cited_by_count") in (None, ""):
            rec["cited_by_count"] = cited
        if not _has_abstract(rec):
            still_missing.append(rec)
            rec["abstract_missing"] = True
        else:
            rec.pop("abstract_missing", None)
        if i % 5 == 0 or i == len(targets):
            LOGGER.info("abstract fill: %d/%d (filled=%d, still missing=%d)",
                        i, len(targets), sum(source_counts.values()), len(still_missing))
        time.sleep(0.2)

    common.write_json(config.CANONICAL_REF_SET, pool)
    filled = sum(source_counts.values())
    print(f"filled {filled}/{len(targets)} missing abstracts | sources={source_counts}")
    print(f"still missing: {len(still_missing)} -> flagged abstract_missing=true (excluded from selection)")
    for r in still_missing:
        print(f"  - {str(r.get('title'))[:70]} | doi={r.get('doi')}")


if __name__ == "__main__":
    main()
