"""Europe PMC source.

Europe PMC is free, key-less and — unlike OpenAlex during a rate-limit block —
generous.  It is the natural substitute for medical/clinical literature because it
indexes PubMed/MEDLINE plus preprints and agricultural sources, and it exposes
abstracts, citation counts and reference lists in one API.

Used for (a) retrieval and (b) filling missing abstracts / citation counts.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from . import common, config

EPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_CLIENT: Optional[common.HttpClient] = None


def _client(client: Optional[common.HttpClient] = None) -> common.HttpClient:
    global _CLIENT
    if client is not None:
        return client
    if _CLIENT is None:
        _CLIENT = common.HttpClient(logger=common.get_logger("europepmc"))
    return _CLIENT


def _clean(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def _result_to_record(res: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    title = _clean(res.get("title"))
    abstract = _clean(res.get("abstractText"))
    doi = common.norm_doi((res.get("doi") or "").rstrip("."))
    if not title:
        return None
    year = res.get("pubYear")
    try:
        year = int(year) if year else None
    except (TypeError, ValueError):
        year = None
    return {
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "source": "europepmc",
        "openalex_id": "",
        "semantic_scholar_id": "",
        "pmid": str(res.get("pmid") or res.get("id") or ""),
        "publication_date": f"{year}-01-01" if year else "",
        "citation_count": res.get("citedByCount"),
        "origin": "retrieved",
        "co_cited_count": 0,
        "year": year,
        "journal_name": _clean(res.get("journalTitle")),
        "authorString": _clean(res.get("authorString")),
    }


def search(query: str, rows: int, client: Optional[common.HttpClient] = None) -> List[Dict[str, Any]]:
    """Free-text search; returns normalised candidate records (abstracts included)."""
    client = _client(client)
    data = client.get_json(
        EPMC_BASE,
        {
            "query": query,
            "format": "json",
            "resultType": "core",
            "pageSize": min(rows, 100),
            "sort": "",
        },
    )
    if not isinstance(data, dict):
        return []
    out = []
    for res in ((data.get("resultList") or {}).get("result") or []):
        rec = _result_to_record(res)
        if rec and rec["abstract"]:
            out.append(rec)
    return out


def fetch_by_doi(doi: str, client: Optional[common.HttpClient] = None) -> Optional[Dict[str, Any]]:
    """Look up a single record by DOI (used to fill missing abstracts)."""
    doi = common.norm_doi(doi)
    if not doi:
        return None
    client = _client(client)
    data = client.get_json(
        EPMC_BASE,
        {"query": f'DOI:"{doi}"', "format": "json", "resultType": "core", "pageSize": 1},
    )
    if not isinstance(data, dict):
        return None
    for res in ((data.get("resultList") or {}).get("result") or []):
        rec = _result_to_record(res)
        if rec:
            return rec
    return None


def fetch_by_title(title: str, client: Optional[common.HttpClient] = None) -> Optional[Dict[str, Any]]:
    """Look up a single record by title (used to fill missing abstracts)."""
    title = _clean(title)
    if len(title.split()) < 3:
        return None
    client = _client(client)
    data = client.get_json(
        EPMC_BASE,
        {"query": f'TITLE:"{title[:180]}"', "format": "json", "resultType": "core", "pageSize": 3},
    )
    if not isinstance(data, dict):
        return None
    from . import bib_eval

    best, best_score = None, 0.0
    for res in ((data.get("resultList") or {}).get("result") or []):
        rec = _result_to_record(res)
        if not rec or not rec.get("abstract"):
            continue
        ok, score = bib_eval.accept_match(rec["title"], title)
        if ok and score > best_score:
            best, best_score = rec, score
    return best
