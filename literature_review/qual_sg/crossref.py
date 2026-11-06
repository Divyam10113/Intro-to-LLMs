"""Crossref source.

Crossref is free, key-less and rate-friendly, and it exposes both a
bibliographic search and reference metadata.  It is used for two jobs:

* **Retrieval** - a second, independent database alongside OpenAlex/arXiv, so the
  pipeline keeps working when one provider throttles (OpenAlex returns 429 with a
  ~27 000 s ``Retry-After`` once its daily budget is spent).
* **Anchor-bibliography resolution** - turning a paper's raw reference strings
  into real records (DOI + title + year + venue) without one API call per entry
  against OpenAlex.

Crossref often has no abstract; step 2 enriches those records by DOI from
OpenAlex, so abstracts and citation counts are still filled in.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from . import bib_eval, common, config

CROSSREF_BASE = "https://api.crossref.org"

_CLIENT: Optional[common.HttpClient] = None


def _client(client: Optional[common.HttpClient] = None) -> common.HttpClient:
    """Per-source client: a 429 on one provider must not disable the others."""
    global _CLIENT
    if client is not None:
        return client
    if _CLIENT is None:
        _CLIENT = common.HttpClient(logger=common.get_logger("crossref"))
    return _CLIENT


def _clean(text: Optional[str]) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)          # strip JATS markup
    return re.sub(r"\s+", " ", text).strip()


def _item_to_record(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    doi = common.norm_doi(item.get("DOI"))
    title = _clean((item.get("title") or [""])[0])
    if not doi or not title:
        return None
    year = None
    for key in ("published-print", "published-online", "issued", "created"):
        parts = ((item.get(key) or {}).get("date-parts") or [[None]])[0]
        if parts and parts[0]:
            year = int(parts[0])
            break
    authors = item.get("author") or []
    return {
        "doi": doi,
        "title": title,
        "abstract": _clean(item.get("abstract")),
        "source": "crossref",
        "openalex_id": "",
        "semantic_scholar_id": "",
        "publication_date": f"{year}-01-01" if year else "",
        "citation_count": item.get("is-referenced-by-count"),
        "origin": "retrieved",
        "co_cited_count": 0,
        "year": year,
        "journal_name": _clean((item.get("container-title") or [""])[0]),
        "first_author_family": _clean(((authors[0] if authors else {}).get("family"))),
    }


def search(
    query: str,
    rows: int,
    client: Optional[common.HttpClient] = None,
) -> List[Dict[str, Any]]:
    client = _client(client)
    """Bibliographic search; returns normalised candidate records."""
    data = client.get_json(
        f"{CROSSREF_BASE}/works",
        {
            "query.bibliographic": query,
            "rows": min(rows, 1000),
            "select": "DOI,title,abstract,issued,created,published-print,published-online,"
                      "container-title,author,is-referenced-by-count,type",
            "mailto": config.OPENALEX_MAILTO,
        },
    )
    if not isinstance(data, dict):
        return []
    out = []
    for item in (data.get("message") or {}).get("items") or []:
        rec = _item_to_record(item)
        if rec:
            out.append(rec)
    return out


def resolve_reference(ref_text: str, client: Optional[common.HttpClient] = None) -> Optional[Dict[str, Any]]:
    """Resolve one raw bibliography entry to a Crossref record."""
    client = _client(client)
    guess = bib_eval.guess_title_from_reference(ref_text) or ref_text
    data = client.get_json(
        f"{CROSSREF_BASE}/works",
        {"query.bibliographic": guess or ref_text, "rows": 3, "mailto": config.OPENALEX_MAILTO},
    )
    if not isinstance(data, dict):
        return None
    best, best_score = None, 0.0
    for item in (data.get("message") or {}).get("items") or []:
        rec = _item_to_record(item)
        if not rec:
            continue
        ok, score = bib_eval.accept_match(rec["title"], ref_text)
        if ok and score > best_score:
            best, best_score = rec, score
    return best
