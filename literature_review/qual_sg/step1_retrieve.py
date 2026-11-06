"""Step 1 — literature retrieval: topic -> candidate pool ``D``.

Paper context (SurveyGen, arXiv:2508.17647, §2.4 and Eqs. (1)-(2)): the survey
topic is turned into a *candidate set* ``D`` of at most ``n`` papers
(``n = config.CANDIDATE_POOL``, 300 by default) that the quality-aware
re-ranker later scores.  Two bibliographic sources are queried:

1. **Semantic Scholar Graph API** (``/paper/search``) — only attempted when
   ``config.USE_SEMANTIC_SCHOLAR`` is set.  Anonymous traffic is rate limited, so
   any 429 / transport error simply falls through to OpenAlex (never aborts).
2. **OpenAlex ``/works``** — the primary source: a free-text ``search`` over the
   topic keywords (authors' query form), bounded to papers published between
   2000 and the topic year, restricted to records with a DOI, ranked by OpenAlex
   relevance to the topic query (the free stand-in for Semantic Scholar's
   relevance search used by the paper), 100 records per page.

Every record is normalised to the same flat schema so that step 2 can enrich it
by DOI:

    {doi, title, abstract, source, openalex_id, semantic_scholar_id,
     publication_date, citation_count}

Records lacking a DOI or an abstract are dropped (Eq. (1) operates on citable,
readable references).  The pool is then de-duplicated by bare DOI (falling back
to the normalised title) keeping the first occurrence, which makes the S2 ->
OpenAlex merge idempotent.

Outputs
-------
``config.STEP1_CANDIDATES``      : JSON list of normalised candidate records.
``intermediates/step1_retrieval_log.json`` : provenance (sources attempted, raw
and usable counts, duplicates, elapsed seconds).

Run with ``python3 -m qual_sg.step1_retrieve`` from ``Paper implementation/``.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree

import numpy as np

from . import common, config, crossref, europepmc
from . import anchor as anchor_mod

LOGGER = common.get_logger("step1_retrieve")

#: Retrieval audit trail, written next to the candidate list.
STEP1_LOG = config.STEP1_CANDIDATES.with_name("step1_retrieval_log.json")
#: Anchor descriptor (title/abstract/id), consumed by step 3's relevance judge.
ANCHOR_JSON = config.INTERMEDIATES_DIR / "anchor.json"

S2_FIELDS = (
    "corpusId,externalIds,title,abstract,citationCount,referenceCount,"
    "influentialCitationCount,publicationDate,journal,authors,references"
)
S2_PAGE_LIMIT = 100          # Semantic Scholar caps ``limit`` at 100.
OPENALEX_PER_PAGE = 100      # OpenAlex caps ``per_page`` at 200; 100 is polite.
OPENALEX_MAX_PAGE = 100      # OpenAlex refuses ``page * per_page`` > 10 000.
EARLIEST_YEAR = int(os.environ.get("QUAL_SG_EARLIEST_YEAR", "2000"))  # lower publication bound


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def year_end(topic: common.Topic) -> int:
    """Upper publication-year bound.

    ``QUAL_SG_YEAR_MODE=now`` (default) uses the current year so a survey can
    cite work published after the topic's nominal year; ``topic`` restores the
    original topic-year cut.  Recency matters for a survey and the topic-year cut
    silently truncated the candidate pool.
    """
    if config.YEAR_MODE == "topic":
        return topic.year or datetime.now().year
    return max(datetime.now().year, topic.year or 0)


def openalex_search_query(topic: common.Topic) -> str:
    """Free-text query for OpenAlex: the topic keywords, space-joined.

    This mirrors the authors' released ``paper_retrieval.py``, which searches
    Semantic Scholar with the *keywords* string (not the title).  Adding the
    title to the query over-constrains OpenAlex's AND-style ``search`` and shrinks
    the usable pool (409 vs 620 hits for this topic).
    """
    keywords = [
        kw.strip()
        for kw in topic.keywords.replace(";", ",").split(",")
        if kw.strip()
    ]
    return " ".join(keywords).strip() or topic.title


def dedupe(records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Drop repeated papers, keeping the first occurrence.

    A record is a duplicate when its bare DOI *or* its normalised title was
    already seen.  Matching on the title as well as the DOI matters when sources
    register the same work under different identifiers (e.g. the journal DOI from
    OpenAlex and the ``10.48550/arxiv.*`` DOI from arXiv).  Returns
    ``(kept, n_duplicates)``.
    """
    seen_doi: set[str] = set()
    seen_title: set[str] = set()
    kept: List[Dict[str, Any]] = []
    duplicates = 0
    for rec in records:
        doi = common.norm_doi(rec.get("doi"))
        title_key = common.normalize_text(rec.get("title", ""))
        if (doi and doi in seen_doi) or (title_key and title_key in seen_title):
            duplicates += 1
            continue
        if doi:
            seen_doi.add(doi)
        if title_key:
            seen_title.add(title_key)
        kept.append(rec)
    return kept, duplicates


# --------------------------------------------------------------------------- #
# Semantic Scholar
# --------------------------------------------------------------------------- #
def _s2_record(paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalise one Semantic Scholar hit; ``None`` when unusable."""
    external = paper.get("externalIds") or {}
    doi = common.norm_doi(external.get("DOI") if isinstance(external, dict) else None)
    abstract = (paper.get("abstract") or "").strip()
    if not doi or not abstract:
        return None
    return {
        "doi": doi,
        "title": (paper.get("title") or "").strip(),
        "abstract": abstract,
        "source": "semantic_scholar",
        "openalex_id": "",
        "semantic_scholar_id": paper.get("paperId") or "",
        "publication_date": paper.get("publicationDate") or "",
        "citation_count": int(paper.get("citationCount") or 0),
    }


def retrieve_semantic_scholar(
    topic: common.Topic, limit: int, client: common.HttpClient
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Paginate S2 ``/paper/search`` (offset based) until ``limit`` usable hits."""
    stats: Dict[str, Any] = {
        "attempted": bool(config.USE_SEMANTIC_SCHOLAR),
        "pages": 0,
        "raw": 0,
        "usable": 0,
        "error": None,
    }
    if not config.USE_SEMANTIC_SCHOLAR:
        stats["error"] = "disabled (QUAL_SG_USE_S2 != 1)"
        LOGGER.info("Semantic Scholar disabled -> OpenAlex only")
        return [], stats

    if config.S2_API_KEY:
        # The session carries the key; it is never logged or written to disk.
        client.session.headers["x-api-key"] = config.S2_API_KEY

    url = f"{config.S2_BASE}/paper/search"
    collected: List[Dict[str, Any]] = []
    offset = 0
    while len(collected) < limit:
        params = {
            "query": topic.query,
            "fields": S2_FIELDS,
            "limit": S2_PAGE_LIMIT,
            "offset": offset,
            "year": f"{EARLIEST_YEAR}-{year_end(topic)}",
        }
        data = client.get_json(url, params=params)
        if not isinstance(data, dict):
            stats["error"] = "request failed (429 / transport error / empty body)"
            LOGGER.warning("Semantic Scholar unavailable -> falling back to OpenAlex")
            break
        papers = data.get("data") or []
        stats["pages"] += 1
        stats["raw"] += len(papers)
        if not papers:
            break
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            rec = _s2_record(paper)
            if rec is not None:
                collected.append(rec)
                if len(collected) >= limit:
                    break
        stats["usable"] = len(collected)
        LOGGER.info(
            "S2 page %d: %d hits (%d usable, total %d so far)",
            stats["pages"], len(papers), stats["usable"], len(collected),
        )
        if len(papers) < S2_PAGE_LIMIT:
            break
        offset += S2_PAGE_LIMIT
    stats["usable"] = len(collected)
    if not collected:
        LOGGER.warning("Semantic Scholar returned 0 usable records -> OpenAlex only")
    return collected, stats


# --------------------------------------------------------------------------- #
# OpenAlex
# --------------------------------------------------------------------------- #
def _openalex_record(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalise one OpenAlex work; ``None`` when it has no DOI or abstract."""
    doi = common.norm_doi(item.get("doi"))
    abstract = common.invert_abstract(item.get("abstract_inverted_index"))
    if not doi or not abstract:
        return None
    return {
        "doi": doi,
        "title": (item.get("title") or "").strip(),
        "abstract": abstract,
        "source": "openalex",
        "openalex_id": common.openalex_short_id(item.get("id")),
        "semantic_scholar_id": "",
        "publication_date": item.get("publication_date") or "",
        "citation_count": int(item.get("cited_by_count") or 0),
    }


def retrieve_openalex(
    topic: common.Topic,
    limit: int,
    already: Sequence[Dict[str, Any]],
    client: common.HttpClient,
    query: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Page OpenAlex ``/works`` until the *total* pool reaches ``limit``."""
    stats: Dict[str, Any] = {
        "attempted": True,
        "pages": 0,
        "raw": 0,
        "usable": 0,
        "error": None,
    }
    search = query or openalex_search_query(topic)
    date_filter = (
        f"from_publication_date:{EARLIEST_YEAR}-01-01,"
        f"to_publication_date:{year_end(topic)}-12-31,has_doi:true"
    )
    url = f"{config.OPENALEX_BASE}/works"
    collected: List[Dict[str, Any]] = []
    page = 1
    while len(already) + len(collected) < limit and page <= OPENALEX_MAX_PAGE:
        # No ``sort``: OpenAlex's default is relevance to ``search``, which is the
        # closest free proxy for the paper's "300 papers *most relevant* to the
        # survey topic" (§Appendix C).  Sorting by citations instead surfaces
        # off-topic mega-surveys; academic impact is handled later in step 3.
        params = {
            "search": search,
            "filter": date_filter,
            "per_page": OPENALEX_PER_PAGE,
            "page": page,
            "mailto": config.OPENALEX_MAILTO,
        }
        data = client.get_json(url, params=params)
        if not isinstance(data, dict):
            stats["error"] = "request failed (429 / transport error / empty body)"
            LOGGER.warning("OpenAlex page %d failed", page)
            break
        results = data.get("results") or []
        stats["pages"] += 1
        stats["raw"] += len(results)
        if not results:
            break
        for item in results:
            if not isinstance(item, dict):
                continue
            rec = _openalex_record(item)
            if rec is not None:
                collected.append(rec)
                if len(already) + len(collected) >= limit:
                    break
        stats["usable"] = len(collected)
        LOGGER.info(
            "OpenAlex page %d: %d hits (%d usable, pool %d/%d)",
            page, len(results), stats["usable"], len(already) + stats["usable"], limit,
        )
        total = (data.get("meta") or {}).get("count")
        if len(already) + len(collected) >= limit:
            break
        if isinstance(total, int) and page * OPENALEX_PER_PAGE >= total:
            break
        if len(results) < OPENALEX_PER_PAGE:
            break
        page += 1
    stats["usable"] = len(collected)
    return collected, stats


# --------------------------------------------------------------------------- #
# Eq. (1) semantic similarity filter
# --------------------------------------------------------------------------- #
def semantic_filter(
    topic: common.Topic,
    records: List[Dict[str, Any]],
    keep: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Keep the ``keep`` candidates with the highest cos(topic, abstract).

    Paper Eq. (1) selects "the top-``n`` with the highest embedding similarity
    scores".  OpenAlex relevance is only a crude stand-in, so when we pull more
    raw hits than the target pool we re-rank them with the paper's embedding
    model (``config.EMBED_MODEL_NAME``).  If the model cannot be loaded the
    function degrades to "keep the first ``keep``" and says so in the stats.
    """
    stats: Dict[str, Any] = {"applied": False, "fetched": len(records), "kept": len(records), "error": None}
    if len(records) <= keep or config.RETRIEVAL_FETCH <= config.CANDIDATE_POOL:
        stats["kept"] = len(records)
        return records, stats
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        model = SentenceTransformer(config.EMBED_MODEL_NAME)
        docs = [f"{r.get('title','')}. {common.truncate(r.get('abstract',''), 2000)}" for r in records]
        query_vec = model.encode([topic.query], normalize_embeddings=True)
        doc_vecs = model.encode(docs, normalize_embeddings=True, batch_size=64)
        sims = doc_vecs @ query_vec[0]
        order = np.argsort(-sims)
        kept = [records[i] for i in order[:keep]]
        for r, i in zip(kept, order[:keep]):
            r["retrieval_similarity"] = round(float(sims[i]), 6)
        stats.update({"applied": True, "kept": len(kept), "model": config.EMBED_MODEL_NAME})
        LOGGER.info(
            "Semantic filter (%s): kept top %d of %d by topic cosine (max=%.3f, min kept=%.3f)",
            config.EMBED_MODEL_NAME, len(kept), len(records), float(sims[order[0]]), float(sims[order[keep - 1]]),
        )
        return kept, stats
    except Exception as exc:  # noqa: BLE001 - never block retrieval on embeddings
        stats["error"] = f"{type(exc).__name__}: {exc}"
        LOGGER.warning("Semantic filter unavailable (%s); keeping first %d candidates", exc, keep)
        return records[:keep], stats


# --------------------------------------------------------------------------- #
# arXiv (free, key-less; strongest coverage for CS/ML topics)
# --------------------------------------------------------------------------- #
ARXIV_BASE = "https://export.arxiv.org/api/query"
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
ARXIV_PER_QUERY = 100


def _collapse(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def retrieve_arxiv(
    topic: common.Topic,
    per_query: int,
    client: common.HttpClient,
    queries: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch topic-relevant preprints from the arXiv Atom API."""
    stats: Dict[str, Any] = {"attempted": True, "queries": [], "raw": 0, "usable": 0, "error": None}
    collected: List[Dict[str, Any]] = []
    for query in (queries or [openalex_search_query(topic)]):
        params = {
            "search_query": query,
            "start": 0,
            "max_results": per_query,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        text = client.get_text(ARXIV_BASE, params=params)
        if not text:
            stats["queries"].append({"query": query, "returned": 0, "error": "request failed"})
            continue
        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError as exc:
            stats["queries"].append({"query": query, "returned": 0, "error": str(exc)})
            continue
        returned = 0
        for entry in root.findall("a:entry", ARXIV_NS):
            raw_id = (entry.findtext("a:id", default="", namespaces=ARXIV_NS) or "").strip()
            m = re.search(r"abs/(\d{4}\.\d{4,5})", raw_id)
            if not m:
                continue
            arxiv_id = m.group(1)
            title = _collapse(entry.findtext("a:title", default="", namespaces=ARXIV_NS))
            abstract = _collapse(entry.findtext("a:summary", default="", namespaces=ARXIV_NS))
            published = (entry.findtext("a:published", default="", namespaces=ARXIV_NS) or "")[:10]
            if topic.year and published[:4].isdigit() and int(published[:4]) > topic.year:
                continue
            if not title or not abstract:
                continue
            doi = _collapse(entry.findtext("arxiv:doi", default="", namespaces=ARXIV_NS))
            # arXiv-registered DOI lets step 2 match the paper in OpenAlex.
            doi = doi or f"10.48550/arxiv.{arxiv_id}"
            collected.append(
                {
                    "doi": common.norm_doi(doi),
                    "title": title,
                    "abstract": abstract,
                    "source": "arxiv",
                    "openalex_id": "",
                    "semantic_scholar_id": "",
                    "publication_date": published,
                    "citation_count": None,
                    "arxiv_id": arxiv_id,
                }
            )
            returned += 1
        stats["queries"].append({"query": query, "returned": returned})
        LOGGER.info("arXiv '%s': %d usable", query, returned)
        time.sleep(3)  # arXiv API politeness
    stats["raw"] = len(collected)
    stats["usable"] = len(collected)
    return collected, stats


def fallback_queries(topic: common.Topic) -> List[str]:
    """Topic-derived query variants used when no LLM is available.

    Deliberately generic (derived only from the topic's own keywords/title) so the
    pipeline never depends on hand-written, topic-specific queries.
    """
    kw = topic.keywords.replace(";", " ").strip()
    return [
        openalex_search_query(topic),
        topic.title,
        f"{kw} benchmark dataset",
        f"{kw} survey review",
        f"{kw} method architecture",
    ]


def expand_queries(topic: common.Topic, anchor: Optional[Dict[str, Any]] = None) -> List[str]:
    """Build the search-query bank: topic keywords + LLM-generated variants.

    A literal topic title under-retrieves: a human writing about a topic also
    searches for its sub-tasks, method families and benchmark names.  We ask the
    configured LLM for ``config.QUERY_EXPANSION_N`` short queries and fall back to
    the static facet queries when no backend is available.
    """
    base = openalex_search_query(topic)
    queries = [base]
    scope = topic.title
    if anchor and anchor.get("abstract"):
        scope = f"{anchor.get('title','')}\n{anchor['abstract'][:1200]}"
    if config.QUERY_EXPANSION_N > 0:
        try:
            client = common.make_llm_client("judge")
            prompt = (
                "You are a research librarian. Given the topic description below, produce "
                f"{config.QUERY_EXPANSION_N} SHORT literature-search queries (3-8 words each) that together cover the "
                "topic's sub-tasks, method families, model names and benchmark datasets. "
                "Return one query per line, no numbering, no commentary.\n\n"
                f"TOPIC:\n{scope}"
            )
            text = client.chat([{"role": "user", "content": prompt}], tag="query-expansion").text
            generated = [" ".join(ln.split()) for ln in text.splitlines()]
            generated = [q.lstrip("-*0123456789. ").strip() for q in generated if 2 <= len(q.split()) <= 14]
            queries += generated[: config.QUERY_EXPANSION_N]
            LOGGER.info("LLM query expansion produced %d queries: %s", len(generated[: config.QUERY_EXPANSION_N]), generated[: config.QUERY_EXPANSION_N])
        except common.PendingLLM:
            raise  # file backend: wait for the answer instead of silently using the fallback queries
        except Exception as exc:  # noqa: BLE001 - retrieval must never depend on the LLM
            LOGGER.warning("query expansion unavailable (%s); using static facet queries", exc)
    # De-duplicate, keep order.
    seen, out = set(), []
    for q in queries + fallback_queries(topic):
        key = common.normalize_text(q)
        if key and key not in seen:
            seen.add(key)
            out.append(q)
    return out


def retrieve_europepmc(
    queries: Sequence[str],
    per_query: int,
    client: common.HttpClient,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Third independent database (Europe PMC).  Especially strong for medical /
    clinical work and, unlike OpenAlex, rarely throttles; returns abstracts."""
    stats: Dict[str, Any] = {"attempted": True, "queries": [], "raw": 0, "usable": 0, "error": None}
    records: List[Dict[str, Any]] = []
    for query in queries:
        clean = re.sub(r"\b(all|abs|ti|au|cat):", " ", query)
        clean = re.sub(r"\b(AND|OR|ANDNOT)\b", " ", clean).replace('"', " ")
        clean = " ".join(clean.split())
        got = europepmc.search(clean or query, per_query, client)
        records += got
        stats["queries"].append({"query": clean, "returned": len(got)})
        LOGGER.info("EuropePMC '%s': %d records", clean[:60], len(got))
    stats["raw"] = stats["usable"] = len(records)
    return records, stats


def retrieve_crossref(
    queries: Sequence[str],
    per_query: int,
    client: common.HttpClient,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Second independent database (Crossref); keeps the pipeline alive when
    OpenAlex throttles.  Abstracts are usually absent and are filled in step 2."""
    stats: Dict[str, Any] = {"attempted": True, "queries": [], "raw": 0, "usable": 0, "error": None}
    records: List[Dict[str, Any]] = []
    for query in queries:
        # Crossref takes plain bibliographic text, not arXiv/OpenAlex field syntax.
        clean = re.sub(r"\b(all|abs|ti|au|cat):", " ", query)
        clean = re.sub(r"\b(AND|OR|ANDNOT)\b", " ", clean).replace('"', " ")
        clean = " ".join(clean.split())
        got = crossref.search(clean or query, per_query, client)
        records += got
        stats["queries"].append({"query": clean, "returned": len(got)})
        LOGGER.info("Crossref '%s': %d records", clean[:60], len(got))
    stats["raw"] = stats["usable"] = len(records)
    return records, stats


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    config.ensure_dirs()
    started = time.time()

    topic = common.load_topic()
    limit = config.CANDIDATE_POOL
    fetch_limit = max(limit, config.RETRIEVAL_FETCH)
    LOGGER.info(
        "Topic '%s' | year<=%s | candidate pool=%d | raw fetch=%d",
        topic.title, year_end(topic), limit, fetch_limit,
    )

    s2_client = common.HttpClient(logger=LOGGER)
    oa_client = common.HttpClient(logger=LOGGER)
    cr_client = common.HttpClient(logger=LOGGER)     # own client: an OpenAlex 429
    ax_client = common.HttpClient(logger=LOGGER)     # must not silence Crossref/arXiv
    ep_client = common.HttpClient(logger=LOGGER)

    # --- optional anchor paper (any DOI / PDF the user supplies) ------------ #
    anchor = None
    if config.ANCHOR_DOI or config.ANCHOR_PDF:
        anchor = anchor_mod.load_anchor(oa_client)
        if anchor:
            common.write_json(ANCHOR_JSON, {k: v for k, v in anchor.items() if k != "bibliography_raw"})

    # --- query bank: topic keywords + LLM-generated facet queries ----------- #
    queries = expand_queries(topic, anchor)
    per_query = max(100, fetch_limit // max(1, len(queries)))
    LOGGER.info("Query bank: %d queries x <=%d hits each", len(queries), per_query)

    s2_records, s2_stats = retrieve_semantic_scholar(topic, fetch_limit, s2_client)

    oa_records: List[Dict[str, Any]] = []
    per_query_stats: List[Dict[str, Any]] = []
    oa_queries = queries[: config.OPENALEX_QUERY_LIMIT]
    for q in oa_queries:
        recs, st = retrieve_openalex(topic, per_query, [], oa_client, query=q)
        per_query_stats.append({"query": q, "pages": st["pages"], "raw": st["raw"], "usable": st["usable"]})
        oa_records += recs
        LOGGER.info("OpenAlex '%s': %d usable (running total %d)", q[:60], st["usable"], len(oa_records))
    oa_stats = {
        "attempted": True,
        "queries": per_query_stats,
        "raw": sum(s["raw"] for s in per_query_stats),
        "usable": sum(s["usable"] for s in per_query_stats),
        "error": None,
    }
    LOGGER.info("OpenAlex blocked/dry: %s", oa_client.blocked_until > time.time())

    cr_records, cr_stats = retrieve_crossref(queries, config.CROSSREF_PER_QUERY, cr_client)
    ep_records, ep_stats = retrieve_europepmc(queries, config.EUROPEPMC_PER_QUERY, ep_client)

    if config.USE_ARXIV:
        arxiv_records, arxiv_stats = retrieve_arxiv(topic, ARXIV_PER_QUERY, ax_client, queries=queries[: config.ARXIV_QUERY_LIMIT])
    else:
        arxiv_records, arxiv_stats = [], {"attempted": False}

    anchor_records: List[Dict[str, Any]] = []
    if anchor and config.ANCHOR_BIBLIOGRAPHY:
        anchor_records = anchor_mod.anchor_reference_records(anchor, oa_client)[: config.ANCHOR_MAX_REFS]

    core, duplicates = dedupe(s2_records + oa_records + arxiv_records + cr_records + ep_records)
    # Anchor references define the scope, so they are exempt from the Eq.(1) cut.
    candidates, filter_stats = semantic_filter(topic, core, max(0, limit - len(anchor_records)))
    existing = {common.norm_doi(c.get("doi")) for c in candidates}
    existing |= {common.normalize_text(c.get("title", "")) for c in candidates}
    anchor_added = 0
    for rec in anchor_records:
        k_doi, k_title = common.norm_doi(rec.get("doi")), common.normalize_text(rec.get("title", ""))
        if (k_doi and k_doi in existing) or (k_title and k_title in existing):
            continue
        rec["retrieval_similarity"] = None
        candidates.append(rec)
        existing.add(k_doi or k_title)
        anchor_added += 1
    raw_count = len(core) + len(anchor_records)

    log = {
        "topic": {
            "title": topic.title,
            "year": topic.year,
            "keywords": topic.keywords,
            "max_refs": topic.max_refs,
        },
        "query": {
            "semantic_scholar": topic.query,
            "openalex": openalex_search_query(topic),
            "bank": queries,
            "per_query_hits": per_query,
        },
        "anchor": (
            {"spec": anchor.get("anchor_spec"), "title": anchor.get("title"),
             "bibliography_entries": len(anchor.get("bibliography_raw") or []),
             "resolved": len(anchor_records), "added_to_pool": anchor_added}
            if anchor else None
        ),
        "year_mode": config.YEAR_MODE,
        "sources": {
            "semantic_scholar": s2_stats,
            "openalex": oa_stats,
            "arxiv": arxiv_stats,
            "crossref": cr_stats,
            "europepmc": ep_stats,
        },
        "kept": len(candidates),
        "raw_deduped": raw_count,
        "semantic_filter": filter_stats,
        "duplicates": duplicates,
        "elapsed_seconds": round(time.time() - started, 2),
    }

    common.write_json(config.STEP1_CANDIDATES, candidates)
    common.write_json(STEP1_LOG, log)

    LOGGER.info(
        "Step 1 done: kept %d candidates (S2=%d, OpenAlex=%d, duplicates=%d) in %.1fs -> %s",
        len(candidates), s2_stats["usable"], oa_stats["usable"], duplicates,
        log["elapsed_seconds"], config.STEP1_CANDIDATES,
    )


if __name__ == "__main__":
    main()
