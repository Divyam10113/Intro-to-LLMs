"""Metadata completion and identifier check for every record in the candidate pool.

Runs after ``fill_abstracts`` (and ``cites_anchor``), before step 3.  For each
record it:

1. **resolves the OpenAlex work** - by OpenAlex id, then DOI (batched), then an
   exact-title search accepted only on a near-identical title;
2. **fills** ``keywords`` and ``topics`` (OpenAlex display names), ``year``
   (``publication_year``, else the publication date), ``venue`` (the primary
   location's source; ``arXiv`` for arXiv-only preprints), the current
   ``cited_by_count`` and, when still missing, the abstract;
3. falls back to OpenAlex's ``/text`` tagger (same keyword and topic vocabulary,
   computed from title + abstract) for records OpenAlex does not index or has
   not tagged, recording ``keywords_source`` / ``topics_source``;
4. **checks the DOI**: it must resolve (OpenAlex, Crossref or DataCite) to a work
   whose title matches the record's title.  A DOI that does not resolve, or that
   resolves to a different paper (the failure seen in the prompt-based list,
   e.g. ``arXiv:2407.12345`` given for MedSAD-CLIP but belonging to an unrelated
   paper), is replaced by the DOI found through an exact-title search; if none is
   found the record is flagged ``doi_verified: false`` and the final export leaves
   it out.

Every action is counted in ``intermediates/enrich_metadata_stats.json``.

    python3 -m qual_sg.enrich_metadata
"""
from __future__ import annotations

import difflib
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from . import common, config

LOGGER = common.get_logger("enrich_metadata")
STATS = config.INTERMEDIATES_DIR / "enrich_metadata_stats.json"
SELECT = ("id,doi,title,publication_year,publication_date,primary_location,keywords,topics,"
          "cited_by_count,abstract_inverted_index,type,is_retracted")
TITLE_OK = 0.90  # normalised-title similarity accepted as "the same paper"


def _norm_title(text: Any) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text or ""))           # strip JATS/HTML tags
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)            # "Learn- ing" -> "Learning"
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def title_match(a: Any, b: Any) -> float:
    x, y = _norm_title(a), _norm_title(b)
    if not x or not y:
        return 0.0
    if x == y:
        return 1.0
    # tolerate subtitles / acronym prefixes ("APRIL-GAN: ...") on one side
    if len(min(x, y, key=len)) >= 25 and (x in y or y in x):
        return 0.95
    return difflib.SequenceMatcher(None, x, y).ratio()


def _names(item: Dict[str, Any], key: str) -> List[str]:
    return [e.get("display_name", "") for e in (item.get(key) or []) if isinstance(e, dict) and e.get("display_name")]


def _venue(item: Dict[str, Any]) -> Optional[str]:
    loc = item.get("primary_location") or {}
    name = ((loc.get("source") or {}).get("display_name") or "").strip()
    if name:
        return "arXiv" if name.lower().startswith("arxiv") else name
    if "arxiv" in str(item.get("doi") or "").lower() or "arxiv.org" in str(loc.get("landing_page_url") or ""):
        return "arXiv"
    return None


def _year(*values: Any) -> Optional[int]:
    for v in values:
        m = re.match(r"\s*(\d{4})", str(v or ""))
        if m and 1900 < int(m.group(1)) < 2100:
            return int(m.group(1))
    return None


def _oa_batch(client: common.HttpClient, filt: str, values: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(0, len(values), 50):
        data = client.get_json(f"{config.OPENALEX_BASE}/works",
                               {"filter": f"{filt}:" + "|".join(values[i:i + 50]), "per-page": 50,
                                "select": SELECT, "mailto": config.OPENALEX_MAILTO})
        out.extend(r for r in (data or {}).get("results") or [] if isinstance(r, dict))
    return out


def _oa_title_search(client: common.HttpClient, title: str) -> Optional[Dict[str, Any]]:
    q = re.sub(r"[^\w\s-]", " ", _norm_title(title))[:250]
    if len(q.split()) < 3:
        return None
    data = client.get_json(f"{config.OPENALEX_BASE}/works",
                           {"filter": f"title.search:{q}", "per-page": 10, "select": SELECT,
                            "mailto": config.OPENALEX_MAILTO})
    best, best_s = None, 0.0
    for item in (data or {}).get("results") or []:
        s = title_match(title, item.get("title"))
        if s > best_s:
            best, best_s = item, s
    return best if best_s >= TITLE_OK else None


def _resolve_doi(client: common.HttpClient, doi: str) -> Tuple[Optional[str], Optional[str]]:
    """(registry, registered title) for a DOI, or (None, None) when it does not resolve."""
    q = requests.utils.quote(doi, safe="/")
    if not doi.startswith("10.48550/"):
        data = client.get_json(f"https://api.crossref.org/works/{q}", {"mailto": config.OPENALEX_MAILTO})
        msg = (data or {}).get("message") or {}
        if msg.get("title"):
            return "crossref", msg["title"][0]
    data = client.get_json(f"https://api.datacite.org/dois/{q}")
    titles = (((data or {}).get("data") or {}).get("attributes") or {}).get("titles") or []
    if titles and titles[0].get("title"):
        return "datacite", titles[0]["title"]
    return None, None


def _crossref_title_search(client: common.HttpClient, title: str) -> Optional[str]:
    data = client.get_json("https://api.crossref.org/works",
                           {"query.bibliographic": title, "rows": 5, "mailto": config.OPENALEX_MAILTO})
    for it in ((data or {}).get("message") or {}).get("items") or []:
        if it.get("title") and title_match(title, it["title"][0]) >= TITLE_OK and it.get("DOI"):
            return common.norm_doi(it["DOI"])
    return None


def _crossref_venue(client: common.HttpClient, doi: str) -> Optional[str]:
    data = client.get_json(f"https://api.crossref.org/works/{requests.utils.quote(doi, safe='/')}",
                           {"mailto": config.OPENALEX_MAILTO})
    msg = (data or {}).get("message") or {}
    names = msg.get("container-title") or msg.get("event", {}).get("name") or []
    names = [names] if isinstance(names, str) else names
    if not names and msg.get("type") in ("dissertation", "report") and msg.get("institution"):
        names = [i.get("name") for i in msg["institution"] if i.get("name")]  # theses: the awarding institution
    return names[0].strip() if names else None


def _s2_citations(client: common.HttpClient, doi: str) -> Optional[int]:
    pid = ("ARXIV:" + doi.split("arxiv.", 1)[1]) if doi.startswith("10.48550/arxiv.") else "DOI:" + doi
    data = client.get_json(f"{config.S2_BASE}/paper/{pid}", {"fields": "citationCount"})
    cc = (data or {}).get("citationCount")
    return int(cc) if isinstance(cc, int) else None


def _text_tagger(client: common.HttpClient, rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data = client.get_json(f"{config.OPENALEX_BASE}/text",
                           {"title": str(rec.get("title") or "")[:500],
                            "abstract": str(rec.get("abstract") or "")[:3000],
                            "mailto": config.OPENALEX_MAILTO})
    return data if isinstance(data, dict) else None


def _apply(rec: Dict[str, Any], item: Dict[str, Any], how: str, st: Dict[str, int]) -> None:
    rec["openalex_id"] = common.openalex_short_id(item.get("id")) or rec.get("openalex_id", "")
    rec["openalex_match"] = how
    kw, tp = _names(item, "keywords"), _names(item, "topics")
    if kw:
        rec["keywords"], rec["keywords_source"] = kw, "openalex"
    if tp:
        rec["topics"], rec["topics_source"] = tp, "openalex"
    y = _year(item.get("publication_year"), item.get("publication_date"))
    if y:
        rec["year"] = y
    v = _venue(item)
    if v:
        rec["venue"] = v
    if item.get("cited_by_count") is not None:
        rec["cited_by_count"] = int(item["cited_by_count"])
    if not (rec.get("abstract") or "").strip():
        ab = common.invert_abstract(item.get("abstract_inverted_index"))
        if ab:
            rec["abstract"], rec["abstract_source"] = ab, "openalex"
            rec.pop("abstract_missing", None)
            st["abstract_filled_openalex"] += 1
    if item.get("is_retracted"):
        rec["is_retracted"] = True


def main() -> None:
    config.ensure_dirs()
    client = common.HttpClient(logger=LOGGER)
    pool: List[Dict[str, Any]] = common.read_json(config.CANONICAL_REF_SET)
    st: Dict[str, Any] = {k: 0 for k in (
        "openalex_by_id", "openalex_by_doi", "openalex_by_title", "openalex_unresolved",
        "abstract_filled_openalex", "keywords_from_text_tagger", "topics_from_text_tagger",
        "doi_verified_openalex", "doi_verified_crossref", "doi_verified_datacite",
        "doi_corrected", "doi_added", "doi_failed", "retracted", "venue_from_crossref", "venue_from_datacite", "citations_from_s2")}
    st["corrections"] = []
    st["failed"] = []

    # -- 1. resolve OpenAlex works ------------------------------------------ #
    ids = sorted({r["openalex_id"] for r in pool if re.fullmatch(r"W\d+", str(r.get("openalex_id") or ""))})
    by_id = {common.openalex_short_id(i.get("id")): i for i in _oa_batch(client, "ids.openalex", ids)}
    dois = sorted({common.norm_doi(r.get("doi")) for r in pool if common.norm_doi(r.get("doi"))})
    by_doi = {common.norm_doi(i.get("doi")): i for i in _oa_batch(client, "doi", dois)}
    LOGGER.info("OpenAlex: %d/%d ids, %d/%d DOIs resolved in batch", len(by_id), len(ids), len(by_doi), len(dois))

    for n, rec in enumerate(pool, start=1):
        doi = common.norm_doi(rec.get("doi"))
        item, how = None, ""
        cand = by_id.get(rec.get("openalex_id") or "")
        if cand is not None and title_match(rec.get("title"), cand.get("title")) >= TITLE_OK:
            item, how = cand, "id"
        if item is None and doi in by_doi and title_match(rec.get("title"), by_doi[doi].get("title")) >= TITLE_OK:
            item, how = by_doi[doi], "doi"
        if item is None and doi and doi not in by_doi:
            one = client.get_json(f"{config.OPENALEX_BASE}/works/https://doi.org/{requests.utils.quote(doi, safe='/')}",
                                  {"select": SELECT, "mailto": config.OPENALEX_MAILTO})
            if isinstance(one, dict) and one.get("id") and title_match(rec.get("title"), one.get("title")) >= TITLE_OK:
                item, how = one, "doi"  # e.g. an arXiv DOI merged into the published version
        if item is None:
            item = _oa_title_search(client, rec.get("title") or "")
            how = "title" if item else ""
        if item is not None:
            st[f"openalex_by_{how}"] += 1
            _apply(rec, item, how, st)
        else:
            st["openalex_unresolved"] += 1
            rec["openalex_match"] = "none"

        # -- 2. DOI check: must resolve to this paper ------------------------ #
        verified_by = None
        if doi and item is not None and how in ("id", "doi") and (
                common.norm_doi(item.get("doi")) == doi or how == "doi"):
            verified_by = "openalex"
        if doi and verified_by is None:
            reg, reg_title = _resolve_doi(client, doi)
            if reg and title_match(rec.get("title"), reg_title) >= TITLE_OK:
                verified_by = reg
            elif reg:
                LOGGER.warning("DOI %s resolves to a different paper (%r) - not %r", doi, reg_title, rec.get("title"))
        if verified_by is None:
            new_doi = common.norm_doi(item.get("doi")) if item is not None else ""
            if new_doi and new_doi != doi:
                reg, reg_title = _resolve_doi(client, new_doi)
                if not reg or title_match(rec.get("title"), reg_title) < TITLE_OK:
                    new_doi = ""
            if not new_doi:
                new_doi = _crossref_title_search(client, rec.get("title") or "") or ""
            if new_doi:
                st["doi_corrected" if doi else "doi_added"] += 1
                st["corrections"].append({"title": rec.get("title"), "old_doi": doi or None, "new_doi": new_doi})
                if doi:
                    rec["doi_corrected_from"] = doi
                rec["doi"] = doi = new_doi
                verified_by = "title_search"
        if verified_by:
            rec["doi_verified"] = True
            rec["doi_verified_by"] = verified_by
            if verified_by in ("openalex", "crossref", "datacite"):
                st[f"doi_verified_{verified_by}"] += 1
        else:
            rec["doi_verified"] = False
            st["doi_failed"] += 1
            st["failed"].append({"title": rec.get("title"), "doi": doi or None})

        # -- 3. year / venue fallbacks ---------------------------------------- #
        rec["year"] = rec.get("year") or _year(rec.get("publication_date"), rec.get("pub_year"))
        if not rec.get("venue"):
            rec["venue"] = rec.get("journal_name") or ("arXiv" if doi.startswith("10.48550/arxiv") else None)
        if not rec.get("venue") and doi and not doi.startswith("10.48550/"):
            rec["venue"] = _crossref_venue(client, doi)
            st["venue_from_crossref"] += bool(rec["venue"])
        if not rec.get("venue") and doi:  # e.g. theses and reports registered with DataCite
            data = client.get_json(f"https://api.datacite.org/dois/{requests.utils.quote(doi, safe='/')}")
            pub = (((data or {}).get("data") or {}).get("attributes") or {}).get("publisher")
            pub = pub.get("name") if isinstance(pub, dict) else pub
            if pub:
                rec["venue"] = str(pub).strip()
                st["venue_from_datacite"] += 1
        if rec.get("cited_by_count") in (None, "") and doi:
            rec["cited_by_count"] = _s2_citations(client, doi)
            st["citations_from_s2"] += rec["cited_by_count"] is not None

        # -- 4. keywords / topics for records OpenAlex has not tagged ---------- #
        if (not rec.get("keywords") or not rec.get("topics")) and (rec.get("abstract") or "").strip():
            tagged = _text_tagger(client, rec)
            if tagged:
                if not rec.get("keywords") and _names(tagged, "keywords"):
                    rec["keywords"], rec["keywords_source"] = _names(tagged, "keywords"), "openalex_text_tagger"
                    st["keywords_from_text_tagger"] += 1
                if not rec.get("topics") and _names(tagged, "topics"):
                    rec["topics"], rec["topics_source"] = _names(tagged, "topics"), "openalex_text_tagger"
                    st["topics_from_text_tagger"] += 1
        if rec.get("is_retracted"):
            st["retracted"] += 1
        if n % 50 == 0:
            LOGGER.info("enrich: %d/%d", n, len(pool))

    st["pool_total"] = len(pool)
    st["with_keywords"] = sum(1 for r in pool if r.get("keywords"))
    st["with_topics"] = sum(1 for r in pool if r.get("topics"))
    st["with_abstract"] = sum(1 for r in pool if (r.get("abstract") or "").strip())
    st["with_year"] = sum(1 for r in pool if r.get("year"))
    st["with_venue"] = sum(1 for r in pool if r.get("venue"))
    st["with_citation_count"] = sum(1 for r in pool if r.get("cited_by_count") not in (None, ""))
    st["doi_verified_total"] = sum(1 for r in pool if r.get("doi_verified"))
    st["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    common.write_json(config.CANONICAL_REF_SET, pool)
    common.write_json(STATS, st)
    summary = {k: v for k, v in st.items() if not isinstance(v, list)}
    LOGGER.info("enrich_metadata: %s", summary)
    print(summary)


if __name__ == "__main__":
    main()
