"""Step 2 — quality-signal augmentation and co-citation expansion.

Paper context (SurveyGen, arXiv:2508.17647, §2.3 bibliographic statistics and
§2.4 "expanded candidate set" ``D_ex``): the candidate pool ``D`` produced by
step 1 only carries ``{doi, title, abstract, ...}``, which is not enough to
compute the quality signals of Eq. (3):

* ``S_c``  citation impact        -> ``cited_by_count``            (Eq. 3, term 1)
* ``S_a``  author reputation      -> ``first/last_author_hindex``  (Eq. 3, term 2)
* ``S_v``  venue reputation       -> ``journal_hindex``            (Eq. 3, term 3)

so this stage re-queries OpenAlex, keyed by DOI in batches of 25, and merges the
missing metadata (inverted abstract, topics, keywords, referenced works, author
and journal ids, journal name).  It then performs **co-citation expansion**: every
OpenAlex work that is cited by at least ``config.CO_CITATION_MIN`` papers of ``D``
is a strong topical signal, so the top ``config.CO_CITATION_TOPN`` such works are
fetched by id and appended as ``origin="co_cited"`` records, yielding ``D_ex``.

Author / venue h-indices are looked up once for the union of all ids (batches of
50) and attached to every record, so both subsets are scored uniformly.

All identifiers (``openalex_id``, ``referenced_works``, author and journal ids)
are stored in canonical short form (``W123`` / ``A123`` / ``S123``) via
``common.openalex_short_id``: OpenAlex returns full URLs, and normalising keeps
the co-citation counter, the id-based filters and the h-index join consistent.

Outputs
-------
``config.CANONICAL_REF_SET`` (``final_ref_set.json``) : ``D_ex``, the combined,
enriched list consumed by step 3.
``config.STEP2_COCITATION``  : stats (input/enriched counts, co-cited ids with
counts, h-index lookups, elapsed seconds).

Resumable: if the output already exists and ``QUAL_SG_FORCE`` is unset the stage
is skipped.  Run with ``python3 -m qual_sg.step2_quality``.
"""

from __future__ import annotations

import os
import time
from collections import Counter, OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import requests

from . import common, config

LOGGER = common.get_logger("step2_quality")

WORK_BATCH = 25      # DOIs per ``doi:`` filter (paper's authors used 10; 25 is safe).
ID_BATCH = 50        # OpenAlex OR-filter and per_page ceiling.
PER_PAGE = 50
TOP_LOGGED = 20      # Co-cited ids kept in the stats file for auditing.


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _chunks(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _force() -> bool:
    """``QUAL_SG_FORCE=1`` re-runs stages whose output already exists."""
    return os.environ.get("QUAL_SG_FORCE", "").strip() not in ("", "0", "false", "False")


def _openalex_url(path: str, filter_expr: str, per_page: int = PER_PAGE) -> str:
    """Build an OpenAlex URL with the filter in the query string.

    The filter is embedded verbatim (rather than passed through ``params``) so the
    ``|`` separators and ``https://doi.org/`` prefixes survive untouched.
    """
    mailto = requests.utils.quote(config.OPENALEX_MAILTO, safe="")
    return (
        f"{config.OPENALEX_BASE}/{path}"
        f"?mailto={mailto}&per-page={per_page}&filter={filter_expr}"
    )


def _first_last_author(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(first_author_id, last_author_id)`` (short form) of a work."""
    first: Optional[str] = None
    last: Optional[str] = None
    for authorship in item.get("authorships") or []:
        if not isinstance(authorship, dict):
            continue
        author_id = common.openalex_short_id((authorship.get("author") or {}).get("id"))
        if not author_id:
            continue
        position = authorship.get("author_position")
        if position == "first":
            first = author_id
        elif position == "last":
            last = author_id
    return first, last


def _display_names(key: str, item: Dict[str, Any]) -> List[str]:
    """``topics``/``keywords`` -> list of ``display_name`` strings."""
    return [
        entry.get("display_name", "")
        for entry in (item.get(key) or [])
        if isinstance(entry, dict)
    ]


def _referenced_works(item: Dict[str, Any]) -> List[str]:
    return [
        rid
        for rid in (common.openalex_short_id(r) for r in (item.get("referenced_works") or []))
        if rid
    ]


def _journal(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """``(journal_id, journal_name)`` from ``primary_location.source``."""
    source = ((item.get("primary_location") or {}).get("source") or {})
    return common.openalex_short_id(source.get("id")) or None, source.get("display_name") or None


# --------------------------------------------------------------------------- #
# Works enrichment
# --------------------------------------------------------------------------- #
def _fetch_works_by_doi(
    client: common.HttpClient, dois: Sequence[str]
) -> List[Dict[str, Any]]:
    """Fetch the OpenAlex works behind ``dois`` in a single ``doi:`` OR filter."""
    values = "|".join(
        "https://doi.org/" + requests.utils.quote(common.norm_doi(d)) for d in dois
    )
    data = client.get_json(_openalex_url("works", f"doi:{values}"))
    if not isinstance(data, dict):
        return []
    return [item for item in (data.get("results") or []) if isinstance(item, dict)]


def _fetch_works_by_id(
    client: common.HttpClient, ids: Sequence[str]
) -> List[Dict[str, Any]]:
    """Fetch OpenAlex works by their short OpenAlex id (``ids.openalex:``)."""
    data = client.get_json(_openalex_url("works", "ids.openalex:" + "|".join(ids)))
    if not isinstance(data, dict):
        return []
    return [item for item in (data.get("results") or []) if isinstance(item, dict)]


# --------------------------------------------------------------------------- #
# Reference-list fallbacks (when OpenAlex has no reference list for a paper)
# --------------------------------------------------------------------------- #
def _crossref_reference_dois(client: common.HttpClient, doi: str) -> List[str]:
    """DOIs of the references Crossref holds for ``doi`` (publisher-deposited)."""
    data = client.get_json(
        "https://api.crossref.org/works/" + requests.utils.quote(doi, safe="/"),
        params={"mailto": config.OPENALEX_MAILTO},
    )
    refs = ((data or {}).get("message") or {}).get("reference") or []
    return [d for d in (common.norm_doi(r.get("DOI")) for r in refs if isinstance(r, dict)) if d]


def _opencitations_references(client: common.HttpClient, doi: str) -> Tuple[List[str], List[str]]:
    """``(dois, openalex_ids)`` of the works ``doi`` cites, from OpenCitations."""
    data = client.get_json(f"{config.OPENCITATIONS_BASE}/references/doi:{doi}")
    dois, ids = [], []
    for row in data if isinstance(data, list) else []:
        for token in str(row.get("cited", "")).split():
            if token.startswith("doi:"):
                dois.append(common.norm_doi(token[4:]))
            elif token.startswith("openalex:"):
                ids.append(common.openalex_short_id(token[9:]))
    return [d for d in dois if d], [i for i in ids if i]


def _dois_to_openalex_ids(client: common.HttpClient, dois: Sequence[str]) -> Dict[str, str]:
    """Map DOIs to OpenAlex work ids (batched list calls)."""
    out: Dict[str, str] = {}
    for batch in _chunks(sorted(set(dois)), WORK_BATCH):
        for item in _fetch_works_by_doi(client, batch):
            doi, wid = common.norm_doi(item.get("doi")), common.openalex_short_id(item.get("id"))
            if doi and wid:
                out[doi] = wid
    return out


def fill_missing_references(client: common.HttpClient, records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Give records without an OpenAlex reference list one from Crossref or OpenCitations.

    The cited works are mapped back to OpenAlex ids so co-citation counting and
    snowballing work unchanged.  ``references_source`` records where each list came from.
    """
    stats = {"missing_before": 0, "crossref": 0, "opencitations": 0, "still_missing": 0}
    wanted: Dict[int, List[str]] = {}
    direct_ids: Dict[int, List[str]] = {}
    for i, rec in enumerate(records):
        if rec.get("referenced_works"):
            rec["references_source"] = "openalex"
            continue
        stats["missing_before"] += 1
        doi = common.norm_doi(rec.get("doi"))
        if not doi or doi.startswith("10.48550/"):  # arXiv DOIs carry no deposited references
            rec["references_source"] = "none"
            continue
        dois = _crossref_reference_dois(client, doi)
        source = "crossref"
        oc_ids: List[str] = []
        if not dois:
            dois, oc_ids = _opencitations_references(client, doi)
            source = "opencitations"
        if dois or oc_ids:
            wanted[i] = dois
            direct_ids[i] = oc_ids
            rec["references_source"] = source
        else:
            rec["references_source"] = "none"
    mapping = _dois_to_openalex_ids(client, [d for ds in wanted.values() for d in ds])
    for i, dois in wanted.items():
        ids = list(dict.fromkeys(direct_ids.get(i, []) + [mapping[d] for d in dois if d in mapping]))
        records[i]["referenced_works"] = ids
        records[i]["referenced_works_count"] = len(ids)
        if ids:
            stats[records[i]["references_source"]] += 1
        else:
            records[i]["references_source"] = "none"
    stats["still_missing"] = sum(1 for r in records if not r.get("referenced_works"))
    return stats


def _merge_work(record: Dict[str, Any], item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge OpenAlex metadata into a step-1 record in place.

    ``item`` is ``None`` when the DOI was not found in OpenAlex: the record is
    still returned (with empty enrichment and ``origin="retrieved"``) so the
    downstream stage can decide what to do with it.
    """
    record.setdefault("retrieval_origin", record.get("origin") or "search")  # keeps "anchor_ref" provenance
    record["origin"] = "retrieved"
    record["co_cited_count"] = 0
    record["metadata_source"] = "openalex" if isinstance(item, dict) else (record.get("source") or "retrieval")

    if not isinstance(item, dict):
        # No OpenAlex match: keep the retrieval metadata, fill the schema holes.
        record.setdefault("topics", [])
        record.setdefault("keywords", [])
        record.setdefault("referenced_works", [])
        record["referenced_works_count"] = len(record.get("referenced_works") or [])
        record.setdefault("first_author_id", None)
        record.setdefault("last_author_id", None)
        record.setdefault("journal_id", None)
        record.setdefault("journal_name", None)
        # Unknown citation counts stay unknown (NaN in step 3) rather than being
        # coerced to 0: an arXiv-only paper that OpenAlex has not indexed should
        # not be ranked as if it were uncited.
        citation_count = record.get("citation_count")
        record["cited_by_count"] = int(citation_count) if citation_count is not None else None
        for field in ("first_author_hindex", "last_author_hindex", "journal_hindex"):
            record.setdefault(field, None)
        return record

    merged_abstract = common.invert_abstract(item.get("abstract_inverted_index"))
    record["abstract"] = merged_abstract or record.get("abstract", "")
    record["title"] = record.get("title") or (item.get("title") or "")
    record["topics"] = _display_names("topics", item)
    record["keywords"] = _display_names("keywords", item)
    record["referenced_works"] = _referenced_works(item)
    record["referenced_works_count"] = len(record["referenced_works"])
    record["cited_by_count"] = int(item.get("cited_by_count") or 0)
    record["first_author_id"], record["last_author_id"] = _first_last_author(item)
    journal_id, journal_name = _journal(item)
    record["journal_id"] = journal_id
    record["journal_name"] = journal_name
    record["openalex_id"] = common.openalex_short_id(item.get("id")) or record.get("openalex_id", "")
    record["publication_date"] = record.get("publication_date") or item.get("publication_date") or ""
    return record


def _co_cited_record(item: Dict[str, Any], counts: Dict[str, int]) -> Dict[str, Any]:
    """Build a ``co_cited`` record with the same field schema as retrieved ones."""
    work_id = common.openalex_short_id(item.get("id"))
    journal_id, journal_name = _journal(item)
    references = _referenced_works(item)
    first, last = _first_last_author(item)
    cited_by = int(item.get("cited_by_count") or 0)
    return {
        "doi": common.norm_doi(item.get("doi")),
        "title": (item.get("title") or "").strip(),
        "abstract": common.invert_abstract(item.get("abstract_inverted_index")),
        "source": "openalex",
        "openalex_id": work_id,
        "semantic_scholar_id": "",
        "publication_date": item.get("publication_date") or "",
        "citation_count": cited_by,
        "topics": _display_names("topics", item),
        "keywords": _display_names("keywords", item),
        "referenced_works": references,
        "referenced_works_count": len(references),
        "cited_by_count": cited_by,
        "first_author_id": first,
        "last_author_id": last,
        "journal_id": journal_id,
        "journal_name": journal_name,
        "origin": "co_cited",
        "co_cited_count": counts.get(work_id, 0),
        "metadata_source": "openalex",
        "references_source": "openalex" if references else "none",
    }


# --------------------------------------------------------------------------- #
# h-index lookups
# --------------------------------------------------------------------------- #
def _fetch_hindex(
    client: common.HttpClient, path: str, ids: Sequence[str]
) -> Dict[str, Optional[int]]:
    """Map short OpenAlex id -> ``summary_stats.h_index`` for authors or sources."""
    resolved: Dict[str, Optional[int]] = {}
    batches = list(_chunks(ids, ID_BATCH))
    for index, batch in enumerate(batches, start=1):
        data = client.get_json(_openalex_url(path, "ids.openalex:" + "|".join(batch)))
        results = (data or {}).get("results") or []
        for entry in results:
            if isinstance(entry, dict):
                stats = entry.get("summary_stats") or {}
                resolved[common.openalex_short_id(entry.get("id"))] = stats.get("h_index")
        LOGGER.info(
            "%s h-index: batch %d/%d (%d resolved)", path, index, len(batches), len(resolved)
        )
    return resolved


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    config.ensure_dirs()

    if config.CANONICAL_REF_SET.exists() and not _force():
        LOGGER.info(
            "%s already exists (set QUAL_SG_FORCE=1 to redo) — skipping step 2",
            config.CANONICAL_REF_SET,
        )
        return

    started = time.time()
    candidates: List[Dict[str, Any]] = common.read_json(config.STEP1_CANDIDATES)
    LOGGER.info("Loaded %d candidates from %s", len(candidates), config.STEP1_CANDIDATES)
    client = common.HttpClient(logger=LOGGER)

    # -- 1. enrich every candidate that has a DOI, in batches of 25 --------- #
    doied = [
        common.norm_doi(c.get("doi"))
        for c in candidates
        if common.norm_doi(c.get("doi"))
    ]
    batches = list(_chunks(doied, WORK_BATCH))
    item_by_doi: Dict[str, Dict[str, Any]] = {}
    for index, batch in enumerate(batches, start=1):
        for item in _fetch_works_by_doi(client, batch):
            item_by_doi[common.norm_doi(item.get("doi"))] = item
        LOGGER.info(
            "OpenAlex enrichment: batch %d/%d (%d/%d DOIs matched, %d works)",
            index, len(batches),
            sum(1 for doi in doied if doi in item_by_doi), len(doied),
            len(item_by_doi),
        )

    enriched: List[Dict[str, Any]] = []
    matched = 0
    for record in candidates:
        doi = common.norm_doi(record.get("doi"))
        item = item_by_doi.get(doi) if doi else None
        if item is not None:
            matched += 1
        enriched.append(_merge_work(record, item))

    LOGGER.info("Enriched %d/%d candidates via OpenAlex", matched, len(candidates))

    # -- 1b. reference-list fallbacks + coverage check ---------------------- #
    ref_stats = fill_missing_references(client, enriched)
    coverage = sum(1 for r in enriched if r.get("referenced_works")) / max(1, len(enriched))
    ref_stats["coverage"] = round(coverage, 3)
    LOGGER.info("Reference lists: %s", ref_stats)
    if coverage < config.MIN_REFERENCE_COVERAGE:
        common.write_json(config.STEP2_COCITATION, {"aborted": True, "reference_lists": ref_stats,
                                                    "enriched_matched": matched, "input_candidates": len(candidates)})
        raise SystemExit(
            f"step 2 stopped: only {coverage:.0%} of candidates have a reference list "
            f"(< {config.MIN_REFERENCE_COVERAGE:.0%}); co-citation and snowball expansion would be empty. "
            "Check the OpenAlex key/budget (python3 -m qual_sg.preflight) and re-run."
        )

    # -- 2. co-citation expansion ------------------------------------------ #
    counter: Counter[str] = Counter()
    for record in enriched:
        for ref in record.get("referenced_works") or []:
            counter[ref] += 1

    eligible = [
        (ref, count) for ref, count in counter.most_common() if count >= config.CO_CITATION_MIN
    ][: config.CO_CITATION_TOPN]
    counts = dict(eligible)
    LOGGER.info(
        "%d referenced works cited by >=%d candidates; fetching top %d",
        len([1 for _, c in counter.most_common() if c >= config.CO_CITATION_MIN]),
        config.CO_CITATION_MIN, len(eligible),
    )

    known_dois = {common.norm_doi(r.get("doi")) for r in enriched} - {""}
    known_ids = {r.get("openalex_id") for r in enriched if r.get("openalex_id")}
    co_cited: List[Dict[str, Any]] = []
    with_abstract = 0
    without_abstract = 0
    duplicates_skipped = 0
    id_batches = list(_chunks([ref for ref, _ in eligible], ID_BATCH))
    for index, batch in enumerate(id_batches, start=1):
        for item in _fetch_works_by_id(client, batch):
            record = _co_cited_record(item, counts)
            key_doi = common.norm_doi(record.get("doi"))
            if (key_doi and key_doi in known_dois) or record["openalex_id"] in known_ids:
                duplicates_skipped += 1
                continue
            if record["abstract"]:
                with_abstract += 1
            else:
                without_abstract += 1
                LOGGER.warning("Co-cited %s has no abstract", record["openalex_id"])
            if key_doi:
                known_dois.add(key_doi)
            known_ids.add(record["openalex_id"])
            co_cited.append(record)
        LOGGER.info(
            "Co-cited fetch: batch %d/%d (%d kept so far)",
            index, len(id_batches), len(co_cited),
        )

    # -- 2b. backward-citation ("snowball") expansion ----------------------- #
    # A human writing about a topic cites the seminal methods and benchmark
    # datasets that the *most relevant* papers build on.  Those are often cited
    # by only one pool paper (so co-citation misses them) or sit deep in the
    # relevance ranking.  We therefore pull the reference lists of the top-N most
    # topically similar papers and add the cited works with abstracts.
    seeds = sorted(
        [r for r in enriched if r.get("abstract")],
        key=lambda r: (r.get("retrieval_similarity") is None, -(r.get("retrieval_similarity") or 0.0)),
    )[: config.EXPAND_FROM_TOP]
    # Rank every reference of the seeds by how many seeds cite it (approved
    # change F2).  Reference lists are not returned in citation order, so taking
    # the first N per seed picked effectively arbitrary works; a work cited by
    # many seeds is a foundation or a key baseline of the topic.
    seed_ids: "OrderedDict[str, int]" = OrderedDict()
    for seed in seeds:
        for ref in dict.fromkeys(seed.get("referenced_works") or []):
            if ref in known_ids or ref in counts:
                continue
            seed_ids[ref] = seed_ids.get(ref, 0) + 1
    ranked_seed_refs = sorted(seed_ids.items(), key=lambda kv: -kv[1])  # stable: ties keep first-seen order
    snowball_ids = [ref for ref, _ in ranked_seed_refs][: config.SNOWBALL_MAX]
    LOGGER.info(
        "Snowball expansion: %d seeds -> %d unseen cited works, ranked by seed count "
        "(top counts %s; fetching %d)",
        len(seeds), len(seed_ids), [c for _, c in ranked_seed_refs[:10]], len(snowball_ids),
    )

    snowball: List[Dict[str, Any]] = []
    snowball_with_abstract = 0
    for index, batch in enumerate(_chunks(snowball_ids, ID_BATCH), start=1):
        for item in _fetch_works_by_id(client, batch):
            record = _co_cited_record(item, {})
            record["origin"] = "snowball"
            record["co_cited_count"] = 0
            record["snowball_seed_count"] = seed_ids.get(record["openalex_id"], 0)
            key_doi = common.norm_doi(record.get("doi"))
            if (key_doi and key_doi in known_dois) or record["openalex_id"] in known_ids:
                continue
            if not record["abstract"]:
                continue
            snowball_with_abstract += 1
            if key_doi:
                known_dois.add(key_doi)
            known_ids.add(record["openalex_id"])
            snowball.append(record)
        LOGGER.info("Snowball fetch: batch %d/%d (%d kept)", index, len(list(_chunks(snowball_ids, ID_BATCH))), len(snowball))

    # -- 3. author / venue h-index for the union of all records ------------- #
    all_records = enriched + co_cited + snowball
    author_ids = sorted(
        {
            common.openalex_short_id(r.get(field))
            for r in all_records
            for field in ("first_author_id", "last_author_id")
        }
        - {""}
    )
    journal_ids = sorted(
        {common.openalex_short_id(r.get("journal_id")) for r in all_records} - {""}
    )
    LOGGER.info(
        "h-index lookups: %d authors, %d venues", len(author_ids), len(journal_ids)
    )
    author_hindex = _fetch_hindex(client, "authors", author_ids) if author_ids else {}
    journal_hindex = _fetch_hindex(client, "sources", journal_ids) if journal_ids else {}

    for record in all_records:
        record["first_author_hindex"] = author_hindex.get(
            common.openalex_short_id(record.get("first_author_id"))
        )
        record["last_author_hindex"] = author_hindex.get(
            common.openalex_short_id(record.get("last_author_id"))
        )
        record["journal_hindex"] = journal_hindex.get(
            common.openalex_short_id(record.get("journal_id"))
        )

    # -- 4. persist --------------------------------------------------------- #
    common.write_json(config.CANONICAL_REF_SET, all_records)

    stats = {
        "input_candidates": len(candidates),
        "enriched_matched": matched,
        "enriched_unmatched": len(candidates) - matched,
        "reference_lists": ref_stats,
        "co_citation_min": config.CO_CITATION_MIN,
        "co_citation_topn": config.CO_CITATION_TOPN,
        "co_cited_candidates_found": len(
            [1 for _, count in counter.most_common() if count >= config.CO_CITATION_MIN]
        ),
        "co_cited_requested": len(eligible),
        "co_cited_kept": len(co_cited),
        "co_cited_with_abstract": with_abstract,
        "co_cited_without_abstract": without_abstract,
        "co_cited_duplicates_skipped": duplicates_skipped,
        "snowball": {
            "selection": "ranked by number of seeds citing the work (change F2)",
            "seeds_used": len(seeds),
            "unseen_cited_works": len(seed_ids),
            "seed_count_histogram": dict(sorted(Counter(seed_ids.values()).items())),
            "fetched": len(snowball_ids),
            "kept_with_abstract": len(snowball),
        },
        "hindex_lookups": {
            "authors_requested": len(author_ids),
            "authors_resolved": len(author_hindex),
            "journals_requested": len(journal_ids),
            "journals_resolved": len(journal_hindex),
        },
        "top_co_cited": [
            {"openalex_id": ref, "count": count} for ref, count in eligible[:TOP_LOGGED]
        ],
        "total_records": len(all_records),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    common.write_json(config.STEP2_COCITATION, stats)

    LOGGER.info(
        "Step 2 done: %d retrieved + %d co-cited + %d snowballed = %d records in %.1fs -> %s",
        len(enriched), len(co_cited), len(snowball), len(all_records),
        stats["elapsed_seconds"], config.CANONICAL_REF_SET,
    )


if __name__ == "__main__":
    main()
