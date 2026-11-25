#!/usr/bin/env python3
"""Assemble the gap-analysis input bundle for both runs (pipeline v4 and prompted).

The bundle answers the five required inputs listed in
``DELIVERABLES/README.md``:

1. the ranked/retrieved papers with metadata (title, year, abstract, arXiv link)
   plus the selected subset;
2. post-MVFA evidence -- the nine checklist papers (SD-MAD, MadCLIP, UltraAD,
   MediCLIP, AF-CLIP, AVA, AdaCLIP, Marzullo et al., Anomaly-OV) resolved against
   this run, plus every paper the run itself retrieved that postdates MVFA;
3. the generated review and a review-number -> paper key, so ``ref [n]`` in the
   review text can be traced back to a source;
4. overlap against the anchor paper's related work / bibliography, from the PDF
   of arXiv:2403.12570 itself;
5. the topic string.

Everything is derived from artifacts already in this repository and from the
anchor PDF; no network access, no LLM.  Re-running is idempotent.

    python3 build_gap_inputs.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent          # Paper implementation/
REPO = ROOT.parent
DELIV = ROOT / "DELIVERABLES"
ANCHOR_PDF = REPO / "2403.12570v1.pdf"

import sys

sys.path.insert(0, str(ROOT))
from qual_sg import bib_eval  # noqa: E402  (pipeline's own bibliography matcher)

POST_MVFA_YEAR = 2024  # MVFA (arXiv:2403.12570) is dated 2024-03-19

# The nine papers named by the gap-analysis checklist, with the pattern used to
# look them up in this repository's data.
CHECKLIST_NINE = [
    ("SD-MAD", r"SD-?\s?MAD"),
    ("MadCLIP", r"MadCLIP"),
    ("UltraAD", r"UltraAD"),
    ("MediCLIP", r"MediCLIP"),
    ("AF-CLIP", r"AF-?\s?CLIP"),
    ("AVA", r"\bAVA\b"),
    ("AdaCLIP", r"AdaCLIP"),
    ("Marzullo et al.", r"Marzullo"),
    ("Anomaly-OV", r"Anomaly-?\s?OV"),
]

ARXIV_RE = re.compile(r"arxiv[:\s/.]*(\d{4}\.\d{4,5})", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def arxiv_url(*candidates: Any) -> str:
    """Return ``https://arxiv.org/abs/<id>`` if any field carries an arXiv id."""
    for c in candidates:
        if not c:
            continue
        m = ARXIV_RE.search(str(c))
        if m:
            return f"https://arxiv.org/abs/{m.group(1)}"
        m = re.search(r"(\d{4}\.\d{4,5})", str(c))
        if m and "arxiv" in str(c).lower():
            return f"https://arxiv.org/abs/{m.group(1)}"
    return ""


def site_url(doi: str, arxiv: str) -> str:
    """Best public link for a record: arXiv page if we have one, else the DOI."""
    if arxiv:
        return arxiv
    doi = (doi or "").strip()
    if not doi:
        return ""
    if doi.lower().startswith(("http://", "https://")):
        return doi
    if doi.lower().startswith("arxiv:"):
        return arxiv_url(doi)
    return f"https://doi.org/{doi}"


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def year_of(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip()[:4])
    except (TypeError, ValueError):
        return None


def norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def truncate(s: str, n: int = 2000) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def review_mentions(md: str, n: int) -> int:
    return len(re.findall(rf"ref\s*\[{n}\]", md))


TITLE_STOPWORDS = {"the", "of", "for", "with", "using", "on", "via", "from", "to", "and",
                   "a", "an", "in", "at", "by"}
VENUE_PREFIXES = ("in ", "arxiv", "proceedings of", "ieee", "acm ", "journal of",
                  "advances in", "conference on", "workshop on")


def looks_like_author_list(s: str) -> bool:
    """True for ``A. B. Author, C. Author, and D. Author`` style fragments."""
    if not re.search(r",\s.*,|, and | and [A-Z]", s or ""):
        return False
    proper = len(re.findall(r"\b[A-Z][a-z]+\b", s))
    stop = sum(1 for w in s.split() if w.lower().strip(",") in TITLE_STOPWORDS)
    return proper >= 3 and stop <= 1


def anchor_title(raw: str) -> str:
    """Best-effort title for a raw bibliography entry from the anchor PDF.

    ``qual_sg.bib_eval.guess_title_from_reference`` returns the author list for
    roughly a fifth of the anchor's 68 entries; this drops a leading author
    segment first and then scores the remaining segments by function-word
    density.  It is a heuristic and is labelled as such in the bundle.
    """
    segments = [s.strip().rstrip(".") for s in re.split(r"\.\s+", raw) if s.strip()]
    if segments and looks_like_author_list(segments[0]):
        segments = segments[1:]
    candidates = []
    for seg in segments:
        if seg.lower().startswith(VENUE_PREFIXES):
            continue
        words = seg.split()
        if not 3 <= len(words) <= 30:
            continue
        if seg.count(",") / max(1, len(words)) > 0.35:      # comma-dense: author list
            continue
        score = 2 * sum(1 for w in words if w.lower().strip(",") in TITLE_STOPWORDS)
        score += 2 if any(c.islower() for c in seg) else 0
        score -= seg.count(",")
        candidates.append((score, len(words), seg))
    if not candidates:
        return bib_eval.guess_title_from_reference(raw)
    return " ".join(max(candidates, key=lambda c: (c[0], c[1]))[2].split()[:24])


def split_review_body(md: str) -> str:
    """Everything before the reference list, so bibliography lines are not quoted."""
    m = re.search(r"^#+\s*(?:\d+\.\s*)?(?:Complete\s+)?References\b", md, re.M | re.I)
    return md[: m.start()] if m else md


def sentences_with(md: str, n: int) -> List[str]:
    out = []
    for para in md.split("\n"):
        if f"ref [{n}]" not in para:
            continue
        for sent in SENT_SPLIT.split(para.strip()):
            if f"ref [{n}]" in sent:
                out.append(" ".join(sent.split()))
    return out


def parse_md_reference_list(md: str) -> Dict[int, Dict[str, str]]:
    """Parse the ``**ref [n]:**`` entries of a generated review's reference list."""
    out: Dict[int, Dict[str, str]] = {}
    for line in md.split("\n"):
        # NB: the prompted review writes its 34th entry as "34:" instead of "34.",
        # so accept either separator rather than silently dropping a citation.
        m = re.match(r"\s*(\d+)\s*[.:]\s+\*\*ref \[(\d+)\]:\*\*\s+(.*\S)\s*$", line)
        if not m:
            continue
        n = int(m.group(2))
        raw = m.group(3)
        year = re.search(r"\((\d{4})\)", raw)
        title = re.sub(r"^.*?\(\d{4}\)\.\s*", "", raw)
        title = re.split(r"\.\s+\*|\s+DOI:|\s+arXiv:", title)[0].strip()
        links = re.findall(r"\]\((https?://[^)]+)\)", raw)
        dois = re.findall(r"\[(10\.[^\]]+)\]", raw)
        out[n] = {
            "raw": raw,
            "title": title,
            "year": year.group(1) if year else "",
            "link": links[0] if links else "",
            "doi": dois[0] if dois else "",
        }
    return out


# --------------------------------------------------------------------------- #
# approach configurations
# --------------------------------------------------------------------------- #
class Run:
    """Everything the builder needs to know about one run of the project."""

    def __init__(self, key: str, ranked_csv: Path, pool_json: Optional[Path],
                 refs_json: Optional[Path], review_md: Path, topic: str,
                 keywords: str, anchor_note: str):
        self.key = key
        self.ranked_csv = ranked_csv
        self.pool_json = pool_json
        self.refs_json = refs_json
        self.review_md = review_md
        self.topic = topic
        self.keywords = keywords
        self.anchor_note = anchor_note
        self.out = DELIV / key / "gap_inputs"


PIPELINE = Run(
    key="pipeline",
    ranked_csv=ROOT / "paper_after_reranking_final.csv",
    pool_json=ROOT / "final_ref_set.json",
    refs_json=ROOT / "pipeline_ref.json",
    review_md=DELIV / "pipeline" / "literature_review.md",
    topic="Generalizable Medical Image Anomaly Detection via Vision-Language Model Adaptation",
    keywords=("Vision-Language Models;CLIP;Medical Anomaly Detection;Zero-Shot;Few-Shot;"
              "Multi-Level Feature Adaptation;Anomaly Segmentation"),
    anchor_note=("Chaoqin Huang, Aofan Jiang, Jinghao Feng, Ya Zhang, Xinchao Wang, Yanfeng Wang. "
                 "'Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images', "
                 "CVPR 2024 Highlight, arXiv:2403.12570v1"),
)

PROMPTED = Run(
    key="prompted",
    ranked_csv=ROOT / "prompted_reranking.csv",
    pool_json=None,
    refs_json=ROOT / "Prompted_ref.json",
    review_md=DELIV / "prompted" / "survey.md",
    topic="Adapting Vision-Language Models for Generalizable Medical Image Anomaly Detection",
    keywords=("Vision-Language Models; Medical Images; Anomaly Detection; Zero-Shot; Few-Shot; CLIP"),
    anchor_note=("Chaoqin Huang, Aofan Jiang, Jinghao Feng, Ya Zhang, Xinchao Wang, Yanfeng Wang. "
                 "'Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images', "
                 "CVPR 2024 Highlight, arXiv:2403.12570v1"),
)

CHECKLIST_TOPIC = "Generalizable VLMs for zero/few-shot medical anomaly detection"


# --------------------------------------------------------------------------- #
# data loaders
# --------------------------------------------------------------------------- #
def load_references(run: Run) -> List[Dict[str, Any]]:
    """Normalised citation records for a run, keyed by the review's own number."""
    refs: List[Dict[str, Any]] = []
    if run.key == "pipeline":
        data = json.loads(run.refs_json.read_text(encoding="utf-8"))["references"]
        for r in data:
            refs.append({
                "number": int(r["refNoInReview"]),
                "refNo": int(r["refNo"]),
                "title": r.get("title", ""),
                "year": str(r.get("year") or ""),
                "doi": r.get("doi") or "",
                "venue": r.get("venue") or "",
                "citations": r.get("citations") or "",
                "relevance": r.get("relevance_llm_1to5", ""),
                "avg_rank": r.get("avg_rank", ""),
                "composite_score": r.get("composite_score", ""),
                "cited_by_anchor": r.get("cited_by_anchor"),
                "origin": r.get("origin", ""),
                "abstract": r.get("abstract") or "",
                "abstract_source": "pipeline_ref.json (full abstract, OpenAlex/S2)",
            })
        return refs

    # prompted: the review's own `ref [n]` numbering is authoritative
    md = run.review_md.read_text(encoding="utf-8")
    md_refs = parse_md_reference_list(md)
    payload = json.loads(run.refs_json.read_text(encoding="utf-8"))
    by_orig = {str(r.get("original_ref_id", "")).strip(): r for r in payload["references"]}
    abs_by_title = {norm_title(r.get("title", "")): r for r in payload["raw_23_field_records"]}
    abs_by_doi = {str(r.get("doi", "")).lower(): r for r in payload["raw_23_field_records"]}
    snippets = {norm_title(r["title"]): r for r in load_csv(run.ranked_csv)}
    for n in sorted(md_refs):
        md_ref = md_refs[n]
        src = by_orig.get(f"ref [{n}]", {})
        rich = abs_by_title.get(norm_title(md_ref["title"])) or {}
        if not rich:
            rich = abs_by_doi.get(str(src.get("doi", "")).lower(), {})
        abstract = str(rich.get("abstract") or "").strip()
        source = ("Prompted_ref.json raw_23_field_records['abstract'] "
                  "(agent-written summary, not a publisher abstract)")
        if not abstract:
            snip = snippets.get(norm_title(md_ref["title"]), {})
            abstract = str(snip.get("abstract") or "").strip()
            source = "prompted_reranking.csv (one-line snippet)"
        year = str(src.get("year") or md_ref["year"] or "")
        refs.append({
            "number": n,
            "refNo": src.get("refNo", ""),
            "title": src.get("title") or md_ref["title"],
            "year": year,
            "doi": src.get("doi") or md_ref["doi"],
            "venue": src.get("venue", ""),
            "citations": "",
            "relevance": src.get("relevance_llm_1to5", ""),
            "avg_rank": "",
            "composite_score": src.get("composite_score", ""),
            "cited_by_anchor": "",
            "origin": "prompt-agent",
            "abstract": abstract,
            "abstract_source": source if abstract else "none in repository",
            "_raw_ref": md_ref["raw"],
            "_link": md_ref["link"],
        })
    return refs


def load_pool(run: Run) -> List[Dict[str, Any]]:
    """Every paper the run ranked, with metadata, in rank order."""
    rows = load_csv(run.ranked_csv)
    pool: List[Dict[str, Any]] = []
    for i, r in enumerate(rows, 1):
        doi = r.get("doi", "")
        ax = arxiv_url(doi, r.get("title", ""))
        y = year_of(r.get("pub_year", ""))
        pool.append({
            "rank": i,
            "title": r.get("title", ""),
            "year": str(y) if y else "",
            "doi": doi,
            "arxiv_url": ax,
            "link": site_url(doi, ax),
            "abstract": r.get("abstract", ""),
            "abstract_source": ("paper_after_reranking_final.csv (abstract as retrieved)"
                                if run.key == "pipeline"
                                else "prompted_reranking.csv (one-line snippet)"),
            "relevance": r.get("relevance_llm_1to5", ""),
            "avg_rank": r.get("avg_rank", ""),
            "composite_score": r.get("composite_score", ""),
            "cited_by_anchor": r.get("cited_by_anchor", ""),
            "origin": r.get("origin", ""),
            "selected": r.get("tag", "") == "selected",
        })
    return pool


def enrich_pool_abstracts(run: Run, pool: List[Dict[str, Any]]) -> None:
    """The prompted ranked CSV stores one-line snippets; replace with real abstracts."""
    if run.key != "prompted":
        return
    rich = json.loads(run.refs_json.read_text(encoding="utf-8"))["raw_23_field_records"]
    by_title = {norm_title(r.get("title", "")): r for r in rich}
    by_doi = {str(r.get("doi", "")).lower(): r for r in rich}
    for row in pool:
        hit = by_title.get(norm_title(row["title"])) or by_doi.get(row["doi"].lower())
        if hit and str(hit.get("abstract") or "").strip():
            row["abstract"] = str(hit["abstract"]).strip()
            row["abstract_source"] = ("Prompted_ref.json raw_23_field_records['abstract'] "
                                      "(agent-written summary, not a publisher abstract)")
        else:
            row["abstract_source"] = "prompted_reranking.csv (one-line snippet)"


# --------------------------------------------------------------------------- #
# item builders
# --------------------------------------------------------------------------- #
def build_item1(run: Run, pool: List[Dict[str, Any]], refs: List[Dict[str, Any]]) -> None:
    fields = ["rank", "title", "year", "doi", "arxiv_url", "public_link", "abstract",
              "abstract_source", "relevance_llm_1to5", "avg_rank", "composite_score",
              "cited_by_anchor", "origin", "selected"]
    write_csv(run.out / "item1_ranked_papers.csv", pool, fields)

    by_title = {norm_title(r["title"]): r for r in refs}
    sel_fields = ["review_number", "title", "year", "doi", "arxiv_url", "public_link", "venue",
                  "citations", "relevance_llm_1to5", "abstract", "abstract_source"]
    sel_rows = []
    for row in pool:
        if not row["selected"]:
            continue
        ref = by_title.get(norm_title(row["title"]))
        sel_rows.append({
            "review_number": ref["number"] if ref else "",
            "title": row["title"],
            "year": row["year"],
            "doi": row["doi"],
            "arxiv_url": row["arxiv_url"],
            "public_link": row["link"],
            "venue": ref["venue"] if ref else "",
            "citations": ref["citations"] if ref else "",
            "relevance_llm_1to5": row["relevance"],
            "abstract": row["abstract"],
            "abstract_source": row.get("abstract_source", "paper_after_reranking_final.csv"),
        })
    write_csv(run.out / "item1_selected_papers.csv", sel_rows, sel_fields)


def build_item2(run: Run, pool: List[Dict[str, Any]], refs: List[Dict[str, Any]], md: str) -> None:
    body = split_review_body(md)
    by_title = {norm_title(r["title"]): r for r in refs}
    selected_titles = {norm_title(r["title"]) for r in refs}
    pool_by_title = {norm_title(p["title"]): p for p in pool}

    def resolve(pattern: str) -> List[Dict[str, Any]]:
        rx = re.compile(pattern, re.I)
        return [p for p in pool if rx.search(p["title"] or "")] + \
               [r for r in refs if rx.search(r["title"] or "") and
                norm_title(r["title"]) not in {norm_title(p["title"]) for p in pool}]

    manifest: Dict[str, Dict[str, Any]] = {}
    evidence: Dict[str, Dict[str, Any]] = {}

    def add(group: str, name: str, title: str, year: Any, doi: str, link: str,
            selected: bool, abstract: str, source: str) -> None:
        ref = by_title.get(norm_title(title))
        num = ref["number"] if ref else ""
        # one row per paper, even when it is both a checklist paper and one of this
        # run's own post-MVFA hits
        mkey = norm_title(title) or f"absent::{name}"
        row = manifest.get(mkey)
        if row is None:
            row = {
                "source_group": group,
                "checklist_name": name,
                "title": title,
                "year": year,
                "doi": doi,
                "arxiv_url": arxiv_url(doi, link),
                "public_link": site_url(doi, arxiv_url(doi, link)),
                "in_this_run_selected": "yes" if selected else "no",
                "review_number": num,
                "abstract_chars": len(abstract or ""),
                "abstract_source": source,
                "full_text_in_repo": "no",
            }
            manifest[mkey] = row
        else:
            groups = sorted(set(row["source_group"].split(",")) | {group})
            row["source_group"] = ",".join(groups)
            if name and name not in row["checklist_name"].split("; "):
                row["checklist_name"] = "; ".join(filter(None, [row["checklist_name"], name]))
            row["in_this_run_selected"] = "yes" if selected or row["in_this_run_selected"] == "yes" else "no"
        key = norm_title(title)
        if not (selected or group == "checklist_nine"):
            return  # kept in the manifest, but not part of the evidence pack
        if key not in evidence:
            evidence[key] = {
                "title": title, "year": year, "doi": doi,
                "link": site_url(doi, arxiv_url(doi, link)),
                "arxiv_url": arxiv_url(doi, link),
                "groups": [], "names": [], "selected": selected,
                "review_numbers": [], "abstract": abstract, "abstract_source": source,
            }
        ev = evidence[key]
        if group not in ev["groups"]:
            ev["groups"].append(group)
        if name and name not in ev["names"]:
            ev["names"].append(name)
        if num and num not in ev["review_numbers"]:
            ev["review_numbers"].append(num)

    # (a) the nine checklist papers, resolved against this run
    for name, pattern in CHECKLIST_NINE:
        hits = resolve(pattern)
        if not hits:
            manifest[f"absent::{name}"] = {
                "source_group": "checklist_nine", "checklist_name": name, "title": "",
                "year": "", "doi": "", "arxiv_url": "", "public_link": "",
                "in_this_run_selected": "absent",
                "review_number": "",
                "abstract_chars": 0,
                "abstract_source": "not present anywhere in this repository",
                "full_text_in_repo": "no",
            }
            continue
        for h in hits:
            title = h["title"]
            year = h.get("year") or h.get("pub_year", "")
            abstract = h.get("abstract", "")
            source = h.get("abstract_source") or (
                "paper_after_reranking_final.csv (full abstract, OpenAlex/S2)"
                if run.key == "pipeline" else "prompted_reranking.csv")
            add("checklist_nine", name, title, year, h.get("doi", ""),
                h.get("link") or h.get("arxiv_url", ""),
                norm_title(title) in selected_titles, abstract, source)

    # (b) this run's own post-MVFA papers (year >= 2024)
    for p in pool:
        y = year_of(p["year"])
        if y is None or y < POST_MVFA_YEAR:
            continue
        add("own_post_mvfa", "", p["title"], p["year"], p["doi"], p["link"],
            p["selected"], p["abstract"], p.get("abstract_source", ""))
    for r in refs:
        if norm_title(r["title"]) in pool_by_title:
            continue
        y = year_of(r["year"])
        if y is None or y < POST_MVFA_YEAR:
            continue
        add("own_post_mvfa", "", r["title"], r["year"], r["doi"], r.get("_link", ""),
            True, r["abstract"], r["abstract_source"])

    write_csv(run.out / "item2_post_mvfa_manifest.csv", list(manifest.values()),
              ["source_group", "checklist_name", "title", "year", "doi", "arxiv_url",
               "public_link", "in_this_run_selected", "review_number", "abstract_chars",
               "abstract_source", "full_text_in_repo"])

    # fetch manifest: what full text would still have to be obtained
    fetch = []
    for row in manifest.values():
        if row["in_this_run_selected"] == "absent":
            continue  # no record in this repository, so no link to fetch either
        if "own_post_mvfa" in row["source_group"] and row["in_this_run_selected"] != "yes":
            continue
        fetch.append({
            "title": row["title"],
            "year": row["year"],
            "checklist_name": row["checklist_name"],
            "arxiv_url": row["arxiv_url"],
            "doi": row["doi"],
            "public_link": row["public_link"],
            "why": "post-MVFA paper -- related-work/method sections are the gap evidence; "
                   "full text not stored in this repository",
        })
    write_csv(run.out / "item2_fulltext_fetch_manifest.csv", fetch,
              ["title", "year", "checklist_name", "arxiv_url", "doi", "public_link", "why"])

    # human-readable evidence pack
    lines = [
        f"# Item 2 - post-MVFA evidence ({run.key} run)",
        "",
        f"Anchor: {run.anchor_note}",
        "",
        "Scope, per the checklist: (a) the nine named post-MVFA papers, and (b) every",
        f"paper this run itself retrieved that postdates MVFA (year >= {POST_MVFA_YEAR}).",
        "",
        "**Source policy: repository artifacts only.** No full texts were downloaded.",
        "Each entry therefore carries the metadata, abstract and links this repository",
        "holds, plus the sentences of this run's own review that discuss the paper.",
        "`item2_fulltext_fetch_manifest.csv` lists the exact URLs still required for",
        "related-work/method text.",
        "",
        "This evidence pack covers the nine checklist papers and every post-MVFA paper in",
        "this run's *selected* set; the remaining ranked rows stay in",
        "`item2_post_mvfa_manifest.csv`.",
        "",
        "---",
        "",
    ]
    for ev in sorted(evidence.values(), key=lambda e: (-(year_of(e["year"]) or 0), e["title"])):
        groups = ", ".join(ev["groups"])
        names = f" (checklist: {', '.join(ev['names'])})" if ev["names"] else ""
        lines += [
            f"## {ev['title']}{names}",
            "",
            f"- group: {groups} | in this run's selected set: {'yes' if ev['selected'] else 'no'}",
            f"- year: {ev['year']} | doi: {ev['doi'] or 'n/a'} | link: {ev['link'] or 'n/a'}",
            f"- review numbers: {', '.join(f'ref [{n}]' for n in sorted(ev['review_numbers'])) or 'not cited by this review'}",
            f"- abstract ({ev['abstract_source']}):",
            "",
            f"> {truncate(ev['abstract'], 1200) or '*(none stored in this repository)*'}",
            "",
        ]
        sents: List[str] = []
        for n in sorted(ev["review_numbers"]):
            for s in sentences_with(body, n):
                if s not in sents:
                    sents.append(s)
        if sents:
            lines += ["- how this run's review uses it:", ""]
            lines += [f"  - {truncate(s, 400)}" for s in sents[:6]]
            lines += [""]
        elif ev["review_numbers"]:
            lines += ["- how this run's review uses it: listed in the bibliography but "
                      "never cited in the review text.", ""]
    absent = [m for m in manifest.values() if m["in_this_run_selected"] == "absent"]
    if absent:
        lines += ["---", "", "## Checklist papers absent from this repository", ""]
        for m in absent:
            lines.append(f"- **{m['checklist_name']}** - no record in this run's data "
                         f"(neither the ranked list nor the selected set).")
        lines.append("")
    (run.out / "item2_post_mvfa_evidence.md").write_text("\n".join(lines), encoding="utf-8")


def build_item3(run: Run, refs: List[Dict[str, Any]], md: str) -> None:
    (run.out / "item3_lit_review.md").write_text(md, encoding="utf-8")
    body = split_review_body(md)
    rows = []
    for r in refs:
        rows.append({
            "review_number": r["number"],
            "tex_label": f"ref{r['number']}",
            "title": r["title"],
            "year": r["year"],
            "doi": r["doi"],
            "arxiv_url": arxiv_url(r["doi"], r.get("_link", "")),
            "public_link": r.get("_link") or site_url(r["doi"], arxiv_url(r["doi"])),
            "in_text_mentions": review_mentions(body, r["number"]),
            "abstract_chars": len(r["abstract"] or ""),
            "abstract_source": r["abstract_source"],
        })
    write_csv(run.out / "item3_citation_key.csv", rows,
              ["review_number", "tex_label", "title", "year", "doi", "arxiv_url",
               "public_link", "in_text_mentions", "abstract_chars", "abstract_source"])


def build_item4(run: Run, pool: List[Dict[str, Any]], refs: List[Dict[str, Any]]) -> None:
    entries = bib_eval.parse_pdf_references(ANCHOR_PDF)
    ref_texts = [e["text"] for e in entries]
    titles = [anchor_title(t) for t in ref_texts]
    low_conf = [looks_like_author_list(t) for t in titles]

    (run.out / "item4_anchor_related_titles.txt").write_text(
        "\n".join(t for t in titles) + "\n", encoding="utf-8")
    write_csv(run.out / "item4_anchor_references.csv",
              [{"index": e["num"], "guessed_title": t,
                "title_is_low_confidence": "yes" if lc else "no",
                "raw_reference": e["text"]}
               for e, t, lc in zip(entries, titles, low_conf)],
              ["index", "guessed_title", "title_is_low_confidence", "raw_reference"])

    # related-work section straight out of the anchor PDF
    from pypdf import PdfReader
    reader = PdfReader(str(ANCHOR_PDF))
    pages = [p.extract_text() or "" for p in reader.pages]
    full = "\n".join(pages)
    m = re.search(r"2\.\s*Related Works?(.*?)(?=\n\s*3\.\s*Problem Formulation)", full, re.S)
    related = " ".join(m.group(1).split()) if m else ""

    selected_titles = [r["title"] for r in refs] or [p["title"] for p in pool if p["selected"]]
    res_sel = bib_eval.evaluate(selected_titles, ref_texts)
    res_pool = bib_eval.evaluate([p["title"] for p in pool], ref_texts)

    stored = None
    stored_path = ROOT / "qual_sg" / "intermediates" / "bib_overlap_v4.json"
    if run.key == "pipeline" and stored_path.exists():
        stored = json.loads(stored_path.read_text(encoding="utf-8"))

    # papers the anchor never discusses, ranked by this run's own ordering
    never = []
    for p in pool:
        idx, score = bib_eval.fuzzy_match(p["title"], ref_texts)
        if idx is None:
            never.append({
                "rank": p["rank"],
                "title": p["title"],
                "year": p["year"],
                "doi": p["doi"],
                "arxiv_url": p["arxiv_url"],
                "selected": "yes" if p["selected"] else "no",
                "best_match_score": score,
                "post_mvfa": "yes" if (year_of(p["year"]) or 0) >= POST_MVFA_YEAR else "no",
            })
    write_csv(run.out / "item4_never_discussed.csv", never,
              ["rank", "title", "year", "doi", "arxiv_url", "selected", "best_match_score",
               "post_mvfa"])

    lines = [
        f"# Item 4 - overlap against the anchor paper's related work ({run.key} run)",
        "",
        f"Anchor PDF: `2403.12570v1.pdf` - {len(entries)} bibliography entries parsed",
        f"by the pipeline's own matcher (`qual_sg/bib_eval.py`, threshold "
        f"{bib_eval.MATCH_THR}, longest-common-substring on normalised titles).",
        "",
        "## Overlap of this run's selected set with the anchor bibliography",
        "",
        "| metric | value |",
        "|---|---|",
        f"| anchor bibliography entries | {res_sel['n_gold_refs']} |",
        f"| this run's selected papers | {res_sel['n_candidates']} |",
        f"| matched | {res_sel['matched']} |",
        f"| precision | {res_sel['precision']} |",
        f"| recall | {res_sel['recall']} |",
        f"| F1 | {res_sel['f1']} |",
        "",
        f"Whole ranked pool ({res_pool['n_candidates']} papers): matched "
        f"{res_pool['matched']}, precision {res_pool['precision']}, recall "
        f"{res_pool['recall']}, F1 {res_pool['f1']}.",
        "",
    ]
    if stored:
        s = stored["selected"]
        lines += [
            "## The pipeline's own stored numbers (cross-check)",
            "",
            f"`qual_sg/intermediates/bib_overlap_v4.json`: matched {s['matched']}, "
            f"precision {s['precision']}, recall {s['recall']}, F1 {s['f1']} "
            f"over {s['n_gold_refs']} anchor refs and {s['n_candidates']} selected papers.",
            "",
        ]
    lines += [
        "## Anchor bibliography titles",
        "",
        f"All {len(titles)} titles are written to `item4_anchor_related_titles.txt` "
        "(one per line) and, with their raw reference strings, to "
        "`item4_anchor_references.csv`.",
        "",
        "Titles are extracted heuristically from the raw bibliography text "
        f"(`build_gap_inputs.anchor_title`); {sum(low_conf)} of {len(titles)} still look like an "
        "author list and are flagged `title_is_low_confidence=yes`, so the raw reference "
        "string should be used for those.",
        "",
        "## Papers this run retrieved that the anchor never discusses",
        "",
        f"{len(never)} of {len(pool)} ranked papers have no match in the anchor "
        "bibliography; they are a gap in themselves and are listed, in this run's own "
        "rank order, in `item4_never_discussed.csv`.",
        "",
        f"- selected by this run but never discussed by the anchor: "
        f"{sum(1 for n in never if n['selected'] == 'yes')}",
        f"- post-MVFA ({POST_MVFA_YEAR}+) and never discussed: "
        f"{sum(1 for n in never if n['post_mvfa'] == 'yes')}",
        "",
    ]
    if related:
        lines += ["## Anchor paper's own related-work section (verbatim from the PDF)", "",
                  "> " + truncate(related, 6000), ""]
    (run.out / "item4_anchor_overlap.md").write_text("\n".join(lines), encoding="utf-8")


def build_item5(run: Run) -> None:
    (run.out / "item5_topic.txt").write_text(
        "\n".join([
            f"run: {run.key}",
            f"topic: {run.topic}",
            f"keywords: {run.keywords}",
            f"anchor: {run.anchor_note}",
            f"checklist scope string: {CHECKLIST_TOPIC}",
            "",
            "The checklist's scope string and this run's topic string describe the same "
            "scope; the topic above is the string this run was actually driven by "
            "(pipeline_ref.json / Prompted_ref.json -> topic).",
            "",
        ]), encoding="utf-8")


def build_index(run: Run, pool: List[Dict[str, Any]], refs: List[Dict[str, Any]],
                md: str) -> None:
    missing = sorted({int(n) for n in re.findall(r"ref\s*\[(\d+)\]", split_review_body(md))} -
                     {r["number"] for r in refs})
    lines = [
        f"# Gap-analysis inputs - {run.key} run",
        "",
        f"Topic: **{run.topic}**",
        "",
        "| checklist item | file | content |",
        "|---|---|---|",
        f"| 1. retrieved papers + metadata | `item1_ranked_papers.csv` | {len(pool)} ranked papers: title, year, doi, arXiv link, abstract, ranking scores, selected flag |",
        f"| 1. selected subset | `item1_selected_papers.csv` | {len(refs)} papers picked out of the ranked list, with abstract + link |",
        f"| 2. post-MVFA evidence | `item2_post_mvfa_evidence.md` | the nine checklist papers + this run's own {POST_MVFA_YEAR}+ papers, with abstract and the review sentences that use them |",
        "| 2. coverage manifest | `item2_post_mvfa_manifest.csv` | one row per paper, incl. which have no record in this repository |",
        "| 2. full-text gaps | `item2_fulltext_fetch_manifest.csv` | exact URLs still needed for related-work/method text (nothing was downloaded) |",
        "| 3. generated review | `item3_lit_review.md` | the review exactly as generated |",
        "| 3. citation key | `item3_citation_key.csv` | review number `[n]` -> paper, so every claim is traceable |",
        "| 4. anchor overlap | `item4_anchor_overlap.md` | overlap of this run with the anchor's bibliography + the never-discussed papers |",
        "| 4. anchor related work | `item4_anchor_related_titles.txt`, `item4_anchor_references.csv` | the anchor paper's bibliography, parsed from `2403.12570v1.pdf` |",
        "| 4. never discussed | `item4_never_discussed.csv` | papers the anchor omits, in this run's rank order |",
        "| 5. topic string | `item5_topic.txt` | the topic string this run was driven by |",
        "",
        "All five items are derived from artifacts already in this repository "
        "(`paper_after_reranking_final.csv`, `pipeline_ref.json`, `final_ref_set.json`, "
        "`Prompted_ref.json`, `prompted_reranking.csv`, the generated reviews) and from "
        "the anchor PDF itself.  Nothing is fetched from the network and no text is "
        "generated: where the repository holds no abstract the cell says so.",
        "",
        f"Abstracts: the median selected-paper abstract in this bundle is "
        f"{sorted(len(r['abstract'] or '') for r in refs)[len(refs) // 2] if refs else 0} "
        f"characters. Check the `abstract_source` column per row before treating a value as a "
        f"publisher abstract - the prompted run stored one-line agent summaries, while the "
        f"pipeline stored publisher abstracts.",
        "",
        "Traceability check: every citation number that appears in "
        "`item3_lit_review.md` resolves "
        "through `item3_citation_key.csv`"
        + (f" (unresolved: {missing})" if missing else " (unresolved: none)")
        + ".",
        "",
        "Regenerate with `python3 build_gap_inputs.py` from `Paper implementation/`.",
        "",
    ]
    (run.out / "00_index.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
def build(run: Run) -> None:
    run.out.mkdir(parents=True, exist_ok=True)
    md = run.review_md.read_text(encoding="utf-8")
    refs = load_references(run)
    pool = load_pool(run)
    enrich_pool_abstracts(run, pool)
    build_item1(run, pool, refs)
    build_item2(run, pool, refs, md)
    build_item3(run, refs, md)
    build_item4(run, pool, refs)
    build_item5(run)
    build_index(run, pool, refs, md)
    print(f"[{run.key}] {run.out.relative_to(REPO)}: {len(pool)} ranked, "
          f"{len(refs)} selected/keys, {sum(1 for p in pool if p['selected'])} tagged selected")


def main() -> None:
    build(PIPELINE)
    build(PROMPTED)


if __name__ == "__main__":
    main()
