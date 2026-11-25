"""Anchor-paper support.

An *anchor* is any publication the user points the pipeline at (a DOI/OpenAlex id
or a PDF).  It plays two generic roles:

1. **Topic sharpening** - the anchor's title + abstract become the text the
   topical-relevance judge scores against, and widen the retrieval queries.  A
   one-line topic title is a weak specification of what a survey should cover;
   the anchor's abstract is a precise, human-written statement of scope.
2. **Citation-graph seeding** - the anchor's bibliography is resolved to real
   records (with abstracts) and merged into the candidate pool, i.e. classic
   *backward snowballing*.  This is what a human does when writing a review of a
   paper's area.

Nothing here is specific to any particular paper: the anchor is discovered at
run time from ``QUAL_SG_ANCHOR_DOI`` / ``QUAL_SG_ANCHOR_PDF``.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import bib_eval, common, config, crossref, datacite
from xml.etree import ElementTree

LOGGER = common.get_logger("anchor")

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

_OA_CLIENT: Optional[common.HttpClient] = None


def _openalex_client(client: Optional[common.HttpClient] = None) -> common.HttpClient:
    """Dedicated OpenAlex client: a 429 must not disable the other providers."""
    global _OA_CLIENT
    if _OA_CLIENT is None:
        _OA_CLIENT = common.HttpClient(logger=common.get_logger("anchor-openalex"))
    return _OA_CLIENT


def _resolve_arxiv_title(ref_text: str, client: Optional[common.HttpClient] = None) -> Optional[Dict[str, Any]]:
    """Resolve a bibliography entry through the arXiv title search API."""
    if client is None:
        client = common.HttpClient(logger=common.get_logger("arxiv-resolve"))
    guess = bib_eval.guess_title_from_reference(ref_text)
    if len(guess.split()) < 3:
        return None
    text = client.get_text(
        ARXIV_API,
        {"search_query": f'ti:"{guess}"', "start": 0, "max_results": 3, "sortBy": "relevance"},
    )
    if not text:
        return None
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return None
    best, best_score = None, 0.0
    for entry in root.findall("a:entry", ARXIV_NS):
        raw_id = (entry.findtext("a:id", default="", namespaces=ARXIV_NS) or "").strip()
        m = re.search(r"abs/(\d{4}\.\d{4,5})", raw_id)
        if not m:
            continue
        title = " ".join((entry.findtext("a:title", default="", namespaces=ARXIV_NS) or "").split())
        ok, score = bib_eval.accept_match(title, ref_text)
        if ok and score > best_score:
            best, best_score = {
                "doi": f"10.48550/arxiv.{m.group(1)}",
                "title": title,
                "abstract": " ".join((entry.findtext("a:summary", default="", namespaces=ARXIV_NS) or "").split()),
                "source": "arxiv",
                "openalex_id": "",
                "semantic_scholar_id": "",
                "publication_date": (entry.findtext("a:published", default="", namespaces=ARXIV_NS) or "")[:10],
                "citation_count": None,
                "origin": "anchor_ref",
                "co_cited_count": 0,
                "year": int((entry.findtext("a:published", default="0000", namespaces=ARXIV_NS) or "0000")[:4] or 0) or None,
                "journal_name": "arXiv",
            }, score
    return best


def guess_title_from_reference(ref_text: str, max_words: int = 22) -> str:
    """Heuristically pull a title out of a raw bibliography entry."""
    return bib_eval.guess_title_from_reference(ref_text, max_words)


def resolve_reference(ref_text: str, client: common.HttpClient) -> Optional[Dict[str, Any]]:
    """Find a real record for one bibliography entry.

    Crossref first (free, rate-friendly, no per-entry quota problems), OpenAlex as
    the fallback for entries Crossref does not resolve.  Both paths return the
    pipeline's candidate schema; abstracts may be empty and are filled in step 2.
    """
    rec = crossref.resolve_reference(ref_text, None)  # own client
    if rec:
        rec["origin"] = "anchor_ref"
        return rec

    for resolver in (datacite.resolve_reference, _resolve_arxiv_title):
        rec = resolver(ref_text, None)
        if rec:
            rec["origin"] = "anchor_ref"
            return rec

    title_guess = bib_eval.guess_title_from_reference(ref_text)
    if len(title_guess.split()) < 3:
        return None
    data = _openalex_client().get_json(
        f"{config.OPENALEX_BASE}/works",
        {"search": title_guess, "per-page": 3, "mailto": config.OPENALEX_MAILTO},
    )
    if not isinstance(data, dict):
        return None
    best, best_score = None, 0.0
    for item in data.get("results") or []:
        ok, score = bib_eval.accept_match(item.get("title", ""), ref_text)
        if ok and score > best_score:
            best, best_score = item, score
    if best is None:
        return None
    return _work_to_record(best, origin="anchor_ref")


def resolve_reference_in_openalex(ref_text: str, client: common.HttpClient) -> Optional[Dict[str, Any]]:
    """Backwards-compatible alias for :func:`resolve_reference`."""
    return resolve_reference(ref_text, client)


def _work_to_record(item: Dict[str, Any], origin: str) -> Dict[str, Any]:
    """Normalise an OpenAlex work into the pipeline's candidate schema."""
    abstract = common.invert_abstract(item.get("abstract_inverted_index")) or ""
    first_id, last_id = None, None
    for auth in item.get("authorships") or []:
        pos = auth.get("author_position")
        if pos == "first":
            first_id = common.openalex_short_id((auth.get("author") or {}).get("id"))
        elif pos == "last":
            last_id = common.openalex_short_id((auth.get("author") or {}).get("id"))
    source = ((item.get("primary_location") or {}).get("source") or {})
    return {
        "doi": common.norm_doi(item.get("doi")),
        "title": item.get("title", ""),
        "abstract": abstract,
        "source": "openalex",
        "openalex_id": common.openalex_short_id(item.get("id")),
        "semantic_scholar_id": "",
        "publication_date": item.get("publication_date", ""),
        "citation_count": item.get("cited_by_count"),
        "origin": origin,
        "co_cited_count": 0,
        "topics": [t.get("display_name", "") for t in (item.get("topics") or []) if isinstance(t, dict)],
        "keywords": [k.get("display_name", "") for k in (item.get("keywords") or []) if isinstance(k, dict)],
        "referenced_works": [common.openalex_short_id(r) for r in (item.get("referenced_works") or [])],
        "referenced_works_count": len(item.get("referenced_works") or []),
        "cited_by_count": item.get("cited_by_count"),
        "first_author_id": first_id,
        "last_author_id": last_id,
        "journal_id": common.openalex_short_id(source.get("id")),
        "journal_name": source.get("display_name", ""),
    }


def load_anchor(client: Optional[common.HttpClient] = None) -> Optional[Dict[str, Any]]:
    """Resolve the configured anchor (DOI/OpenAlex id or PDF) into a descriptor."""
    client = client or common.HttpClient(logger=LOGGER)
    anchor: Dict[str, Any] = {}
    spec = config.ANCHOR_DOI
    if spec:
        key = common.openalex_short_id(spec)
        data = None
        if spec.startswith("10."):
            data = client.get_json(f"{config.OPENALEX_BASE}/works/https://doi.org/{common.norm_doi(spec)}")
        if not isinstance(data, dict) or "id" not in data:
            data = client.get_json(f"{config.OPENALEX_BASE}/works/{key}")
        if isinstance(data, dict) and data.get("id"):
            anchor = _work_to_record(data, origin="anchor")
            anchor["anchor_spec"] = spec
            LOGGER.info("Anchor DOI resolved: %s", anchor["title"][:90])
        else:
            LOGGER.warning("Anchor DOI %r could not be resolved", spec)

    pdf = config.ANCHOR_PDF
    if pdf:
        path = Path(pdf)
        if not path.is_absolute():
            path = config.BASE_DIR / path
        if path.exists():
            refs = bib_eval.parse_pdf_references(path)
            anchor.setdefault("anchor_spec", str(path))
            anchor["bibliography_raw"] = refs
            LOGGER.info("Anchor PDF %s: %d bibliography entries", path.name, len(refs))
            if anchor.get("abstract") is None and anchor.get("title") is None:
                text = _pdf_head_text(path)
                anchor["title"] = anchor.get("title") or _guess_pdf_title(text)
                anchor["abstract"] = anchor.get("abstract") or _guess_pdf_abstract(text)
        else:
            LOGGER.warning("Anchor PDF not found: %s", path)
    return anchor or None


def _pdf_head_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((reader.pages[i].extract_text() or "") for i in range(min(2, len(reader.pages))))


def _guess_pdf_title(text: str) -> str:
    for line in text.splitlines():
        line = " ".join(line.split())
        if 20 <= len(line) <= 160 and not line.lower().startswith(("arxiv", "http", "abstract")):
            return line
    return ""


def _guess_pdf_abstract(text: str) -> str:
    m = re.search(r"Abstract\s*(.{200,2500}?)(?:\n\s*\n|1\.?\s+Introduction|Keywords)", text, re.S | re.I)
    return " ".join(m.group(1).split()) if m else ""


def anchor_reference_records(anchor: Dict[str, Any], client: common.HttpClient) -> List[Dict[str, Any]]:
    """Resolve the anchor's bibliography entries to OpenAlex records."""
    refs = anchor.get("bibliography_raw") or []
    if not refs:
        return []
    records: List[Dict[str, Any]] = []
    seen: set = set()
    for i, ref in enumerate(refs, start=1):
        rec = resolve_reference_in_openalex(ref["text"], client)
        if not rec:
            LOGGER.info("anchor ref [%s] unresolved: %s", ref.get("num"), ref["text"][:70])
            continue
        if not rec.get("abstract"):
            LOGGER.info("anchor ref [%s] resolved but has no abstract: %s", ref.get("num"), rec["title"][:70])
        key = rec.get("doi") or common.normalize_text(rec.get("title"))
        if key in seen:
            continue
        seen.add(key)
        rec["anchor_ref_num"] = ref.get("num")
        rec["anchor_ref_text"] = ref["text"]
        records.append(rec)
        if i % 10 == 0:
            LOGGER.info("anchor bibliography: %d/%d resolved", i, len(refs))
        time.sleep(0.2)
    LOGGER.info("anchor bibliography: %d/%d entries resolved to OpenAlex works",
                len(records), len(refs))
    return records
