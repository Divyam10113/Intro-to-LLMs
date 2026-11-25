"""Forward-citation stage: add the papers that cite the anchor (MVFA) to the pool.

Step 2 expands the pool *backwards* (co-citation and snowballing over reference
lists) and the anchor stage adds the anchor's own bibliography.  Neither finds the
work published *after* the anchor that builds on it, which is what a research-gap
analysis (Task 3) needs most.  This stage asks "who cites the anchor?":

1. OpenAlex ``/works?filter=cites:<anchor work id>`` (all pages, cursor paging);
2. Semantic Scholar ``/paper/<id>/citations`` as a second source (best effort;
   it often lists arXiv preprints that OpenAlex has not linked yet).

Every citing work is either

* already in the pool -> flagged ``cites_anchor: true`` (its origin is kept), or
* new -> appended with ``origin: "cites_anchor"`` and the same field schema as
  co-cited records, including first/last-author and venue h-indices so step 3's
  impact score can be computed.

The anchor is given by ``QUAL_SG_ANCHOR_WORKS`` (comma-separated DOIs, arXiv ids
or OpenAlex ids), else ``QUAL_SG_ANCHOR_DOI``; with neither the stage is skipped.
``QUAL_SG_ANCHOR_LABEL`` (e.g. ``mvfa``) names the tag in the export's ``origin``
(``cites_mvfa``).  Records published before ``QUAL_SG_EARLIEST_YEAR`` are dropped
(none can cite a 2024 paper anyway).  Re-running is idempotent.

    python3 -m qual_sg.cites_anchor
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from . import common, config
from . import step2_quality as s2q

LOGGER = common.get_logger("cites_anchor")
STATS = config.INTERMEDIATES_DIR / "cites_anchor_stats.json"
EARLIEST_YEAR = int(os.environ.get("QUAL_SG_EARLIEST_YEAR", "2000"))
#: falls back to QUAL_SG_ANCHOR_DOI, the anchor already used by the anchor stage
ANCHOR_WORKS = [s.strip() for s in (os.environ.get("QUAL_SG_ANCHOR_WORKS", "").strip() or config.ANCHOR_DOI).split(",")
                if s.strip()]


ANCHOR_KEYS: set = set()  # normalised titles and DOIs of the anchor itself (never flagged as citing it)


def _resolve_anchor_ids(client: common.HttpClient) -> Tuple[List[str], List[Dict[str, Any]]]:
    """OpenAlex work ids (and S2 lookup ids) for the anchor specs."""
    oa_ids: List[str] = []
    s2_ids: List[Dict[str, Any]] = []
    for spec in ANCHOR_WORKS:
        if re.fullmatch(r"W\d+", spec):
            path = spec
        else:
            doi = common.norm_doi(spec)
            if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", doi):
                doi = "10.48550/arxiv." + re.sub(r"v\d+$", "", doi)
            path = "https://doi.org/" + doi
            if doi.startswith("10.48550/arxiv."):
                s2_ids.append({"spec": spec, "s2": "arXiv:" + doi.split("arxiv.", 1)[1]})
            else:
                s2_ids.append({"spec": spec, "s2": "DOI:" + doi})
        item = client.get_json(f"{config.OPENALEX_BASE}/works/{path}", {"mailto": config.OPENALEX_MAILTO})
        wid = common.openalex_short_id((item or {}).get("id"))
        if not re.fullmatch(r"W\d+", spec):
            ANCHOR_KEYS.add(common.norm_doi(path))
        if item:
            ANCHOR_KEYS.update({common.normalize_text(item.get("title")), common.norm_doi(item.get("doi"))})
        if wid and wid not in oa_ids:
            oa_ids.append(wid)
        LOGGER.info("anchor spec %s -> OpenAlex %s", spec, wid or "not found")
    return oa_ids, s2_ids


def _openalex_citing(client: common.HttpClient, work_ids: List[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    cursor = "*"
    while cursor:
        data = client.get_json(
            f"{config.OPENALEX_BASE}/works",
            {"filter": f"cites:{'|'.join(work_ids)},is_paratext:false", "per-page": 200,
             "cursor": cursor, "mailto": config.OPENALEX_MAILTO},
        )
        if not isinstance(data, dict):
            LOGGER.warning("OpenAlex citing query failed; keeping %d works fetched so far", len(items))
            break
        batch = [r for r in data.get("results") or [] if isinstance(r, dict)]
        items.extend(batch)
        cursor = (data.get("meta") or {}).get("next_cursor") if batch else None
    return items


def _s2_citing(client: common.HttpClient, s2_ids: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for spec in s2_ids:
        offset = 0
        while offset is not None:
            data = client.get_json(
                f"{config.S2_BASE}/paper/{spec['s2']}/citations",
                {"fields": "title,externalIds,abstract,year,publicationDate,venue,citationCount",
                 "limit": 1000, "offset": offset},
            )
            if not isinstance(data, dict):
                LOGGER.warning("Semantic Scholar citations unavailable for %s", spec["s2"])
                break
            for row in data.get("data") or []:
                paper = (row or {}).get("citingPaper") or {}
                if paper.get("paperId") and paper.get("title"):
                    out[paper["paperId"]] = paper
            offset = data.get("next")
    return list(out.values())


def _s2_doi(paper: Dict[str, Any]) -> str:
    ext = paper.get("externalIds") or {}
    if ext.get("DOI"):
        return common.norm_doi(ext["DOI"])
    if ext.get("ArXiv"):
        return "10.48550/arxiv." + str(ext["ArXiv"]).lower()
    return ""


def _year_ok(date_or_year: Any) -> bool:
    m = re.match(r"(\d{4})", str(date_or_year or ""))
    return bool(m) and int(m.group(1)) >= EARLIEST_YEAR


def main() -> None:
    config.ensure_dirs()
    if not ANCHOR_WORKS:
        # no anchor paper in this run: nothing cites "it", so the stage is a no-op (not a failure)
        LOGGER.warning("no anchor set (QUAL_SG_ANCHOR_WORKS / QUAL_SG_ANCHOR_DOI): skipping who-cites-the-anchor")
        common.write_json(STATS, {"skipped": "no anchor paper configured"})
        return
    client = common.HttpClient(logger=LOGGER)
    pool: List[Dict[str, Any]] = common.read_json(config.CANONICAL_REF_SET)
    by_doi = {common.norm_doi(r.get("doi")): r for r in pool if common.norm_doi(r.get("doi"))}
    by_oa = {r.get("openalex_id"): r for r in pool if r.get("openalex_id")}
    by_title = {common.normalize_text(r.get("title")): r for r in pool if r.get("title")}

    anchor_ids, s2_ids = _resolve_anchor_ids(client)
    if not anchor_ids:
        raise SystemExit("could not resolve the anchor paper in OpenAlex")
    ANCHOR_KEYS.discard("")
    for r in pool:  # the anchor's own pool record never "cites the anchor"
        if (common.norm_doi(r.get("doi")) in ANCHOR_KEYS or common.normalize_text(r.get("title")) in ANCHOR_KEYS
                or r.get("openalex_id") in anchor_ids):
            r.pop("cites_anchor", None)
            r["is_anchor"] = True

    def find(doi: str, oa: str, title: str) -> Optional[Dict[str, Any]]:
        return (by_doi.get(doi) if doi else None) or (by_oa.get(oa) if oa else None) \
            or by_title.get(common.normalize_text(title))

    stats = {"anchor_openalex_ids": anchor_ids, "openalex_citing": 0, "s2_citing": 0,
             "already_in_pool": 0, "added_openalex": 0, "added_s2_only": 0,
             "skipped_anchor_itself": 0, "skipped_year": 0, "skipped_no_doi": 0}

    # -- 1. OpenAlex -------------------------------------------------------- #
    oa_items = _openalex_citing(client, anchor_ids)
    stats["openalex_citing"] = len(oa_items)
    new: List[Dict[str, Any]] = []
    for item in oa_items:
        wid = common.openalex_short_id(item.get("id"))
        doi = common.norm_doi(item.get("doi"))
        if wid in anchor_ids or (doi and doi in ANCHOR_KEYS):
            stats["skipped_anchor_itself"] += 1
            continue
        if not _year_ok(item.get("publication_date")):
            stats["skipped_year"] += 1
            continue
        hit = find(doi, wid, item.get("title") or "")
        if hit is not None:
            if not hit.get("cites_anchor"):
                stats["already_in_pool"] += 1
            hit["cites_anchor"] = True
            continue
        if not doi:
            stats["skipped_no_doi"] += 1
            continue
        rec = s2q._co_cited_record(item, {})
        rec.update({"origin": "cites_anchor", "co_cited_count": 0, "cites_anchor": True,
                    "retrieval_origin": "cites_anchor", "cites_anchor_source": "openalex"})
        new.append(rec)
        by_doi[doi] = rec
        by_oa[wid] = rec
        by_title[common.normalize_text(rec["title"])] = rec
    stats["added_openalex"] = len(new)

    # -- 2. Semantic Scholar (second source, best effort) ------------------- #
    s2_items = _s2_citing(client, s2_ids)
    stats["s2_citing"] = len(s2_items)
    s2_only: List[Dict[str, Any]] = []
    for paper in s2_items:
        doi = _s2_doi(paper)
        title = (paper.get("title") or "").strip()
        if common.normalize_text(title) in ANCHOR_KEYS or (doi and doi in ANCHOR_KEYS):
            continue
        if not _year_ok(paper.get("publicationDate") or paper.get("year")):
            stats["skipped_year"] += 1
            continue
        hit = find(doi, "", title)
        if hit is not None and hit.get("is_anchor"):
            continue
        if hit is not None:
            if not hit.get("cites_anchor"):
                stats["already_in_pool"] += 1
            hit["cites_anchor"] = True
            continue
        if not doi:
            stats["skipped_no_doi"] += 1
            continue
        # Prefer the OpenAlex record when OpenAlex knows the DOI but had not linked the citation.
        item = client.get_json(f"{config.OPENALEX_BASE}/works/https://doi.org/{requests.utils.quote(doi, safe='/')}",
                               {"mailto": config.OPENALEX_MAILTO})
        if isinstance(item, dict) and item.get("id"):
            if common.openalex_short_id(item.get("id")) in anchor_ids:
                continue
            rec = s2q._co_cited_record(item, {})
            rec["doi"] = rec["doi"] or doi
            rec["metadata_source"] = "openalex"
        else:
            rec = {
                "doi": doi, "title": title, "abstract": (paper.get("abstract") or "").strip(),
                "source": "semantic_scholar", "openalex_id": "", "semantic_scholar_id": paper.get("paperId", ""),
                "publication_date": paper.get("publicationDate") or (str(paper["year"]) if paper.get("year") else ""),
                "citation_count": paper.get("citationCount"), "topics": [], "keywords": [],
                "referenced_works": [], "referenced_works_count": 0, "cited_by_count": paper.get("citationCount"),
                "first_author_id": None, "last_author_id": None, "journal_id": None,
                "journal_name": paper.get("venue") or None, "metadata_source": "semantic_scholar",
                "references_source": "none",
            }
        if not rec.get("abstract") and paper.get("abstract"):
            rec["abstract"] = paper["abstract"].strip()
            rec["abstract_source"] = "semantic_scholar"
        rec.update({"origin": "cites_anchor", "co_cited_count": 0, "cites_anchor": True,
                    "retrieval_origin": "cites_anchor", "cites_anchor_source": "semantic_scholar"})
        s2_only.append(rec)
        by_doi[doi] = rec
        by_title[common.normalize_text(title)] = rec
    stats["added_s2_only"] = len(s2_only)
    new.extend(s2_only)

    # -- 3. h-indices for the impact score ---------------------------------- #
    author_ids = sorted({a for r in new for a in (r.get("first_author_id"), r.get("last_author_id")) if a})
    journal_ids = sorted({r["journal_id"] for r in new if r.get("journal_id")})
    a_h = s2q._fetch_hindex(client, "authors", author_ids) if author_ids else {}
    j_h = s2q._fetch_hindex(client, "sources", journal_ids) if journal_ids else {}
    for r in new:
        r["first_author_hindex"] = a_h.get(r.get("first_author_id")) if r.get("first_author_id") else None
        r["last_author_hindex"] = a_h.get(r.get("last_author_id")) if r.get("last_author_id") else None
        r["journal_hindex"] = j_h.get(r.get("journal_id")) if r.get("journal_id") else None

    pool.extend(new)
    stats["cites_anchor_total_in_pool"] = sum(1 for r in pool if r.get("cites_anchor"))
    stats["pool_total"] = len(pool)
    stats["new_without_abstract"] = sum(1 for r in new if not (r.get("abstract") or "").strip())
    stats["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    common.write_json(config.CANONICAL_REF_SET, pool)
    common.write_json(STATS, stats)
    LOGGER.info("cites_anchor: %s", stats)
    print(stats)


if __name__ == "__main__":
    main()
