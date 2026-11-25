"""Full bibliography entries built from publisher metadata (approved change 3).

The selected references only carry a title and a DOI.  This module looks every
DOI up - Crossref for published work, the arXiv API for ``10.48550/arxiv.*``
preprints, OpenAlex as a fallback - and stores authors, venue, year, volume and
pages.  Entries are never written by the LLM, so an identifier always points to
the paper it is listed for.

    python3 -m qual_sg.bibliography          # -> <run>/bibliography.json + references.bib
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

import pandas as pd
import requests

from . import common, config

LOGGER = common.get_logger("bibliography")
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}


def _crossref(client: common.HttpClient, doi: str) -> Optional[Dict[str, Any]]:
    data = client.get_json("https://api.crossref.org/works/" + requests.utils.quote(doi, safe="/"),
                           params={"mailto": config.OPENALEX_MAILTO})
    msg = (data or {}).get("message")
    if not msg:
        return None
    year = None
    for field in ("published-print", "published-online", "issued", "created"):
        parts = ((msg.get(field) or {}).get("date-parts") or [[None]])[0]
        if parts and parts[0]:
            year = int(parts[0])
            break
    authors = [" ".join(p for p in (a.get("given"), a.get("family")) if p) or a.get("name", "")
               for a in msg.get("author") or []]
    return {
        "authors": [a for a in authors if a],
        "title": " ".join(((msg.get("title") or [""])[0]).split()),
        "venue": " ".join(((msg.get("container-title") or [""])[0]).split()),
        "year": year,
        "volume": msg.get("volume") or "",
        "issue": msg.get("issue") or "",
        "pages": msg.get("page") or "",
        "publisher": msg.get("publisher") or "",
        "type": msg.get("type") or "",
        "metadata_source": "crossref",
    }


def _arxiv(client: common.HttpClient, arxiv_id: str) -> Optional[Dict[str, Any]]:
    text = client.get_text("https://export.arxiv.org/api/query", params={"id_list": arxiv_id})
    if not text:
        return None
    entry = ElementTree.fromstring(text).find("a:entry", ARXIV_NS)
    if entry is None:
        return None
    published = entry.findtext("a:published", default="", namespaces=ARXIV_NS)
    return {
        "authors": [" ".join((a.findtext("a:name", default="", namespaces=ARXIV_NS)).split())
                    for a in entry.findall("a:author", ARXIV_NS)],
        "title": " ".join(entry.findtext("a:title", default="", namespaces=ARXIV_NS).split()),
        "venue": f"arXiv preprint arXiv:{arxiv_id}",
        "year": int(published[:4]) if published[:4].isdigit() else None,
        "volume": "", "issue": "", "pages": "", "publisher": "arXiv", "type": "preprint",
        "metadata_source": "arxiv",
    }


def _openalex(client: common.HttpClient, doi: str) -> Optional[Dict[str, Any]]:
    item = client.get_json(f"{config.OPENALEX_BASE}/works/https://doi.org/{doi}")
    if not item or not item.get("id"):
        return None
    biblio = item.get("biblio") or {}
    pages = "-".join(p for p in (biblio.get("first_page"), biblio.get("last_page")) if p)
    return {
        "authors": [((a.get("author") or {}).get("display_name") or "") for a in item.get("authorships") or []],
        "title": " ".join((item.get("title") or "").split()),
        "venue": (((item.get("primary_location") or {}).get("source")) or {}).get("display_name") or "",
        "year": item.get("publication_year"),
        "volume": biblio.get("volume") or "", "issue": biblio.get("issue") or "", "pages": pages,
        "publisher": "", "type": item.get("type") or "", "metadata_source": "openalex",
    }


def lookup(client: common.HttpClient, doi: str, title: str = "") -> Dict[str, Any]:
    doi = common.norm_doi(doi)
    meta: Optional[Dict[str, Any]] = None
    m = re.match(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", doi or "")
    if m:
        meta = _arxiv(client, m.group(1))
    elif doi:
        meta = _crossref(client, doi)
    if not meta and doi:
        meta = _openalex(client, doi)
    meta = meta or {"authors": [], "title": title, "venue": "", "year": None, "volume": "", "issue": "",
                    "pages": "", "publisher": "", "type": "", "metadata_source": "none"}
    meta["doi"] = doi
    meta["url"] = f"https://doi.org/{doi}" if doi else ""
    if not meta.get("title"):
        meta["title"] = title
    # Publishers mark withdrawn papers this way; flag them so a reviewer can check.
    meta["withdrawal_notice"] = bool(re.match(r"(notice of (removal|retraction)|retracted)", meta["title"], re.I))
    return meta


def _author_list(authors: List[str], limit: int = 3) -> str:
    authors = [a for a in authors if a]
    if not authors:
        return ""
    if len(authors) > limit:
        return f"{authors[0]} et al."
    return ", ".join(authors[:-1]) + (" and " if len(authors) > 1 else "") + authors[-1]


def format_text(meta: Dict[str, Any]) -> str:
    """One reference as plain text: Authors (Year). Title. Venue, vol(issue), pages. DOI."""
    parts = []
    who = _author_list(meta.get("authors") or [])
    year = meta.get("year")
    if who:
        parts.append(f"{who} ({year})." if year else f"{who}.")
    title = str(meta.get("title") or "").rstrip(".")
    parts.append(f"{title}.")
    venue = meta.get("venue") or ""
    if venue:
        tail = venue
        if meta.get("volume"):
            tail += f", {meta['volume']}" + (f"({meta['issue']})" if meta.get("issue") else "")
        if meta.get("pages"):
            tail += f", {meta['pages']}"
        parts.append(tail + ".")
    if meta.get("doi"):
        parts.append(f"https://doi.org/{meta['doi']}")
    return " ".join(parts)


def tex_escape(text: str) -> str:
    out = []
    for ch in str(text):
        if ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append("\\textasciitilde{}")
        elif ch == "^":
            out.append("\\textasciicircum{}")
        elif ch == "\\":
            out.append("\\textbackslash{}")
        else:
            out.append(ch)
    return "".join(out)


def format_tex(meta: Dict[str, Any]) -> str:
    """One reference as a LaTeX bibliography body (title in italics, DOI as URL)."""
    who = tex_escape(_author_list(meta.get("authors") or []))
    year = meta.get("year")
    title = tex_escape(str(meta.get("title") or "").rstrip("."))
    venue = tex_escape(meta.get("venue") or "")
    pieces = []
    if who:
        pieces.append(f"{who}{f' ({year})' if year else ''}.")
    pieces.append(f"\\emph{{{title}}}.")
    if venue:
        extra = ""
        if meta.get("volume"):
            extra += f", {tex_escape(meta['volume'])}" + (f"({tex_escape(meta['issue'])})" if meta.get("issue") else "")
        if meta.get("pages"):
            extra += f", {tex_escape(meta['pages'])}"
        pieces.append(f"{venue}{extra}.")
    if meta.get("doi"):
        pieces.append(f"\\url{{https://doi.org/{meta['doi']}}}")
    return " ".join(pieces)


def bibtex_entry(key: str, meta: Dict[str, Any]) -> str:
    kind = "article" if meta.get("type") in ("journal-article", "article") else (
        "inproceedings" if "proceedings" in (meta.get("type") or "") else "misc")
    fields = {
        "title": meta.get("title"), "author": " and ".join(meta.get("authors") or []),
        "year": meta.get("year"), "doi": meta.get("doi"),
        ("journal" if kind == "article" else "booktitle" if kind == "inproceedings" else "howpublished"): meta.get("venue"),
        "volume": meta.get("volume"), "number": meta.get("issue"), "pages": meta.get("pages"),
    }
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    return f"@{kind}{{{key},\n{body}\n}}"


def build(refs: List[Dict[str, Any]], client: Optional[common.HttpClient] = None) -> Dict[str, Dict[str, Any]]:
    """Metadata for ``refs`` (each ``{refNo, title, doi}``), keyed by ``refNo`` as a string."""
    client = client or common.HttpClient(logger=LOGGER)
    out: Dict[str, Dict[str, Any]] = {}
    for ref in refs:
        meta = lookup(client, ref.get("doi", ""), ref.get("title", ""))
        meta["refNo"] = int(ref["refNo"])
        meta["title_in_ranking"] = ref.get("title", "")
        out[str(ref["refNo"])] = meta
    return out


def selected_refs() -> List[Dict[str, Any]]:
    df = pd.read_csv(config.CANONICAL_RANKED_CSV)
    df = df[df["tag"] == "selected"]
    refs = []
    for row in df.to_dict("records"):
        title = str(row.get("title", "")).strip()
        abstract = str(row.get("abstract", "")).strip()
        if not title or not abstract or abstract.lower() == "nan":
            continue
        refs.append({"refNo": len(refs) + 1, "title": title, "doi": common.norm_doi(str(row.get("doi") or ""))})
    return refs


def load_or_build(refs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Cached bibliography for the run (rebuilt when the reference set changes)."""
    path = config.CANONICAL_BIBLIOGRAPHY
    if path.exists():
        bib = common.read_json(path)
        if sorted(bib) == sorted(str(r["refNo"]) for r in refs) and all(
            bib[str(r["refNo"])].get("doi") == common.norm_doi(r.get("doi")) for r in refs
        ):
            return bib
    bib = build(refs)
    common.write_json(path, bib)
    (path.parent / "references.bib").write_text(
        "\n\n".join(bibtex_entry(f"ref{k}", v) for k, v in sorted(bib.items(), key=lambda kv: int(kv[0]))) + "\n",
        encoding="utf-8",
    )
    stats = {
        "entries": len(bib),
        "by_source": dict(Counter(m["metadata_source"] for m in bib.values())),
        "with_authors": sum(1 for m in bib.values() if m.get("authors")),
        "with_venue": sum(1 for m in bib.values() if m.get("venue")),
        "withdrawal_notices": [m["title"] for m in bib.values() if m.get("withdrawal_notice")],
    }
    common.write_json(config.INTERMEDIATES_DIR / "bibliography_stats.json", stats)
    LOGGER.info("bibliography: %s", stats)
    return bib


def main() -> None:
    config.ensure_dirs()
    load_or_build(selected_refs())


if __name__ == "__main__":
    main()
