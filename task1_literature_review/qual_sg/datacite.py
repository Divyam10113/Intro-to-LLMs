"""DataCite source.

DataCite registers the DOIs of preprints and datasets, including every
``10.48550/arXiv.*`` DOI, so it resolves exactly the entries Crossref misses
(arXiv-only papers such as CLIP or LDM, and dataset records).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from . import bib_eval, common, config

DATACITE_BASE = "https://api.datacite.org/dois"

_CLIENT: Optional[common.HttpClient] = None


def _client(client: Optional[common.HttpClient] = None) -> common.HttpClient:
    global _CLIENT
    if client is not None:
        return client
    if _CLIENT is None:
        _CLIENT = common.HttpClient(logger=common.get_logger("datacite"))
    return _CLIENT


def _clean(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def _item_to_record(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    attrs = item.get("attributes") or {}
    doi = common.norm_doi(attrs.get("doi"))
    titles = attrs.get("titles") or []
    title = _clean((titles[0] or {}).get("title")) if titles else ""
    if not doi or not title:
        return None
    year = attrs.get("publicationYear")
    descs = attrs.get("descriptions") or []
    abstract = ""
    for d in descs:
        if (d.get("descriptionType") or "").lower() in ("abstract", ""):
            abstract = _clean(d.get("description"))
            if abstract:
                break
    return {
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "source": "datacite",
        "openalex_id": "",
        "semantic_scholar_id": "",
        "publication_date": f"{year}-01-01" if year else "",
        "citation_count": None,
        "origin": "retrieved",
        "co_cited_count": 0,
        "year": year,
        "journal_name": _clean((attrs.get("publisher") or "") if isinstance(attrs.get("publisher"), str) else (attrs.get("publisher") or {}).get("name", "")),
    }


def resolve_reference(ref_text: str, client: Optional[common.HttpClient] = None) -> Optional[Dict[str, Any]]:
    """Resolve one bibliography entry to a DataCite record."""
    client = _client(client)
    guess = bib_eval.guess_title_from_reference(ref_text) or ref_text
    data = client.get_json(
        DATACITE_BASE,
        {"query": guess[:200], "page[size]": 5, "mailto": config.OPENALEX_MAILTO},
    )
    if not isinstance(data, dict):
        return None
    best, best_score = None, 0.0
    for item in ((data.get("data") or []) if isinstance(data.get("data"), list) else []):
        rec = _item_to_record(item)
        if not rec:
            continue
        ok, score = bib_eval.accept_match(rec["title"], ref_text)
        if ok and score > best_score:
            best, best_score = rec, score
    return best
