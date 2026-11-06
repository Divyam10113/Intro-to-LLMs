"""Metrics for a documented run, side by side with run v4.

Writes ``<run>/metrics.json`` and ``<run>/metrics.md``:

* retrieval and pool statistics (sources, anchor seeding, expansion, abstracts);
* a funnel of MVFA's own references through every stage, separating papers the
  pipeline found from papers seeded from MVFA's bibliography;
* whether key foundations and baselines reached the pool and the selection;
* the selected set's recency, preprint share, origin and venues;
* the generated survey and 2-page review: length, citation density, claim-check
  verdicts and revisions;
* bibliography accuracy and LLM calls per model.

    QUAL_SG_RUN_DIR=runs/<run> python3 -m qual_sg.run_metrics
"""
from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import bib_eval, common, config

ANCHOR_PDF = config.BASE_DIR.parent / "2403.12570v1.pdf"
V4_DIR = config.BASE_DIR / "DELIVERABLES" / "pipeline"
CITE_RE = re.compile(r"\bref\s*\[\s*(\d+)\s*\]")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

#: Foundations and baselines MVFA compares against (red flag #5), matched by title.
KEY_WORKS = {
    "CLIP": "learning transferable visual models from natural language supervision",
    "WinCLIP": "winclip",
    "APRIL-GAN": "zero few shot anomaly classification and segmentation method for cvpr 2023 vand",
    "AnomalyCLIP": "anomalyclip",
    "PatchCore": "towards total recall in industrial anomaly detection",
    "RD4AD": "anomaly detection via reverse distillation from one class embedding",
    "CFlow-AD": "cflow ad",
    "DRA": "catching both gray and black swans",
    "BGAD": "explicit boundary guided semi push pull contrastive learning",
    "BraTS": "brats",
    "MVTec AD": "mvtec ad",
    "BMAD": "bmad benchmarks for medical anomaly detection",
}


def _norm(t: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(t or "").lower())).strip()


def _year(r: Dict[str, Any]) -> Optional[int]:
    for k in ("year", "publication_year", "pub_year"):
        if r.get(k) not in (None, "") and str(r.get(k)) != "nan":
            try:
                return int(float(r[k]))
            except ValueError:
                pass
    d = str(r.get("publication_date") or "")
    return int(d[:4]) if d[:4].isdigit() else None


def gold_refs() -> List[str]:
    return [r["text"] for r in bib_eval.parse_pdf_references(ANCHOR_PDF)]


def overlap(titles: List[str], gold: List[str]) -> Dict[str, Any]:
    res = bib_eval.evaluate(titles, gold)
    return {k: res[k] for k in ("matched", "precision", "recall", "f1", "n_candidates", "n_gold_refs")}


def matched_gold(titles: List[str], gold: List[str]) -> Dict[str, int]:
    """title -> gold index, for titles that match MVFA's bibliography."""
    out = {}
    for t in titles:
        i, _ = bib_eval.fuzzy_match(t, gold)
        if i is not None:
            out[t] = i
    return out


def key_works(titles_pool: List[str], titles_sel: List[str]) -> Dict[str, Dict[str, bool]]:
    pool, sel = [_norm(t) for t in titles_pool], [_norm(t) for t in titles_sel]
    return {name: {"in_pool": any(p in t for t in pool), "selected": any(p in t for t in sel)}
            for name, p in KEY_WORKS.items()}


def selected_profile(sel: pd.DataFrame) -> Dict[str, Any]:
    years = [y for y in (_year(r) for r in sel.to_dict("records")) if y]
    dois = [str(d).lower() for d in sel.get("doi", pd.Series(dtype=str)).fillna("")]
    venue_col = next((c for c in ("journal_name", "venue") if c in sel), None)
    venues = Counter(str(v) for v in (sel[venue_col].fillna("") if venue_col else []) if str(v) and str(v) != "nan")
    return {
        "n": len(sel),
        "year_counts": dict(sorted(Counter(years).items())),
        "share_2024_plus": round(sum(y >= 2024 for y in years) / max(1, len(years)), 3),
        "share_pre_2020": round(sum(y < 2020 for y in years) / max(1, len(years)), 3),
        "median_year": sorted(years)[len(years) // 2] if years else None,
        "arxiv_only": sum(d.startswith("10.48550") for d in dois),
        "origin": dict(Counter(sel["origin"].fillna("?"))) if "origin" in sel else {},
        "relevance": dict(Counter(int(x) for x in sel["relevance_llm_1to5"].dropna())) if "relevance_llm_1to5" in sel else {},
        "top_venues": venues.most_common(12),
    }


def score_report(ranked: pd.DataFrame, gold: List[str]) -> Dict[str, Any]:
    """Relevance-score distributions, MVFA references and key works by score."""
    rel = pd.to_numeric(ranked["relevance_llm_1to5"], errors="coerce")
    judged = ranked[rel.notna()]
    gated = ranked[rel >= config.RELEVANCE_GATE]
    sel = ranked[ranked["tag"] == "selected"]

    def dist(df: pd.DataFrame) -> Dict[str, int]:
        c = Counter(int(x) for x in pd.to_numeric(df["relevance_llm_1to5"], errors="coerce").dropna())
        return {str(s): c.get(s, 0) for s in range(1, 6)}

    gold_rows = ranked[ranked["title"].isin(matched_gold(list(judged["title"]), gold))]
    keys = {}
    for name, p in KEY_WORKS.items():
        hit = ranked[ranked["title"].map(_norm).str.contains(p, regex=False)]
        if hit.empty:
            keys[name] = {"in_pool": False}
            continue
        r = hit.sort_values("relevance_llm_1to5", ascending=False).iloc[0]
        s = r["relevance_llm_1to5"]
        keys[name] = {"in_pool": True, "title": r["title"], "relevance": None if pd.isna(s) else int(s),
                      "has_abstract": isinstance(r.get("abstract"), str) and r["abstract"].strip() != "",
                      "passes_gate": bool(pd.notna(s) and s >= config.RELEVANCE_GATE),
                      "selected": r["tag"] == "selected"}
    comp = pd.to_numeric(sel["composite_score"], errors="coerce").dropna()
    return {
        "scale": "1-5 topical relevance from the title and abstract (rubric in step3_rerank.py)",
        "gate": config.RELEVANCE_GATE,
        "selection": {"method": config.SELECTION, "mmr_lambda": config.MMR_LAMBDA,
                      "anchor_priority": config.ANCHOR_REF_PRIORITY, "anchor_prior_score": config.ANCHOR_PRIOR_SCORE,
                      "require_abstract": config.REQUIRE_ABSTRACT},
        "pool_records": len(ranked),
        "not_judged": int(rel.isna().sum()),
        "distribution_judged": dist(judged),
        "passing_gate": len(gated),
        "distribution_selected": dist(sel),
        "passing_gate_not_selected": len(gated) - len(sel),
        "mvfa_refs_judged_distribution": dist(gold_rows),
        "key_works": keys,
        "selected_composite": {"min": round(float(comp.min()), 3), "median": round(float(comp.median()), 3),
                               "max": round(float(comp.max()), 3)} if len(comp) else {},
        "rationale_recorded": False,
    }


def text_stats(blocks: List[str]) -> Dict[str, Any]:
    text = "\n".join(blocks)
    words = len(text.split())
    marks = CITE_RE.findall(text)
    sents = [s for b in blocks for s in SENT_RE.split(b) if s.strip()]
    per_sent = [len(CITE_RE.findall(s)) for s in sents]
    return {
        "words": words,
        "citation_markers": len(marks),
        "distinct_refs_cited": len(set(marks)),
        "citations_per_100_words": round(100 * len(marks) / max(1, words), 2),
        "words_per_citation": round(words / max(1, len(marks)), 1),
        "sentences": len(sents),
        "sentences_with_more_than_3_citations": sum(1 for n in per_sent if n > 3),
        "sentences_without_citations": sum(1 for n in per_sent if n == 0),
    }


def review_blocks_md(path: Path) -> List[str]:
    if not path.exists():
        return []
    body = path.read_text(encoding="utf-8").split("## References")[0]
    return [b for b in re.split(r"\n## [^\n]+\n", body)[1:] if b.strip()]


def claim_totals(checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    c = Counter()
    for chk in checks:
        for k, v in (chk.get("counts") or {}).items():
            c[k] += v
    return {"sentences_checked": sum(c.values()), **dict(c),
            "texts_revised": sum(1 for chk in checks if chk.get("revised")),
            "revisions_rejected": sum(1 for chk in checks if chk.get("revision_rejected")),
            "texts_checked": len(checks)}


def bibliography_check(bib: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    mism = []
    for k, m in bib.items():
        sim = difflib.SequenceMatcher(None, _norm(m.get("title_in_ranking")), _norm(m.get("title"))).ratio()
        if sim < 0.85:
            mism.append({"refNo": k, "ranking_title": m.get("title_in_ranking"), "metadata_title": m.get("title"),
                         "similarity": round(sim, 2)})
    return {"entries": len(bib),
            "with_authors": sum(1 for m in bib.values() if m.get("authors")),
            "with_venue": sum(1 for m in bib.values() if m.get("venue")),
            "with_year": sum(1 for m in bib.values() if m.get("year")),
            "metadata_source": dict(Counter(m.get("metadata_source") for m in bib.values())),
            "title_mismatches": mism,
            "withdrawal_notices": [m.get("title") for m in bib.values() if m.get("withdrawal_notice")]}


def llm_usage() -> Dict[str, Any]:
    rows = common.read_jsonl(config.CACHE_DIR / "llm_audit.jsonl")
    by = Counter((r.get("stage", "?"), str(r.get("tag", "")).split("-")[0], r.get("model")) for r in rows)
    return {"answered_calls": len(rows),
            "by_stage_tag_model": [{"stage": s, "tag": t, "model": m, "calls": n} for (s, t, m), n in sorted(by.items(), key=str)]}


def main() -> None:
    config.ensure_dirs()
    gold = gold_refs()
    m: Dict[str, Any] = {"gold": {"source": str(ANCHOR_PDF.name), "entries": len(gold)}}

    # -- retrieval and pool ------------------------------------------------- #
    step1 = common.read_json(config.STEP1_CANDIDATES)
    pool = common.read_json(config.CANONICAL_REF_SET)
    s1log = common.read_json(config.STEP1_CANDIDATES.with_name("step1_retrieval_log.json"))
    s2 = common.read_json(config.STEP2_COCITATION)
    m["retrieval"] = {
        "step1_candidates": len(step1),
        "by_source": dict(Counter(r.get("source", "?") for r in step1)),
        "anchor_bibliography_added": sum(1 for r in step1 if r.get("origin") == "anchor_ref"),
        "query_bank": s1log.get("query", {}).get("bank"),
        "sources": {k: {kk: v.get(kk) for kk in ("raw", "usable", "error")} for k, v in (s1log.get("sources") or {}).items()},
        "semantic_filter": s1log.get("semantic_filter"),
    }
    m["expansion"] = {k: s2.get(k) for k in ("enriched_matched", "enriched_unmatched", "reference_lists",
                                             "co_cited_kept", "co_cited_without_abstract", "snowball")}
    m["pool"] = {"records": len(pool), "origin": dict(Counter(r.get("origin", "?") for r in pool)),
                 "without_abstract": sum(1 for r in pool if not str(r.get("abstract") or "").strip())}

    # -- MVFA reference funnel --------------------------------------------- #
    ranked = pd.read_csv(config.CANONICAL_RANKED_CSV)
    sel = ranked[ranked["tag"] == "selected"]
    searched = [r["title"] for r in step1 if r.get("origin") != "anchor_ref"]
    seeded_titles = {_norm(r["title"]) for r in step1 if r.get("origin") == "anchor_ref"}
    judged = ranked[ranked["relevance_llm_1to5"].notna()] if "relevance_llm_1to5" in ranked else ranked.iloc[0:0]
    sel_match = matched_gold(list(sel["title"]), gold)
    funnel = {
        "step1_all": overlap([r["title"] for r in step1], gold)["matched"],
        "step1_found_by_search": overlap(searched, gold)["matched"],
        "pool_after_expansion": overlap([r["title"] for r in pool], gold)["matched"],
        "pool_with_abstract": overlap([r["title"] for r in pool if str(r.get("abstract") or "").strip()], gold)["matched"],
        "judged": overlap(list(judged["title"]), gold)["matched"],
        "judged_relevance_ge_3": overlap(list(judged[judged["relevance_llm_1to5"] >= 3]["title"]), gold)["matched"],
        "selected": len(set(sel_match.values())),
        "selected_seeded_from_anchor": len({i for t, i in sel_match.items() if _norm(t) in seeded_titles}),
        "selected_found_by_pipeline": len({i for t, i in sel_match.items() if _norm(t) not in seeded_titles}),
    }
    m["mvfa_reference_funnel"] = funnel
    m["overlap_selected"] = overlap(list(sel["title"]), gold)
    m["key_works"] = key_works([r["title"] for r in pool], list(sel["title"]))
    m["selected_profile"] = selected_profile(sel)
    m["scores"] = score_report(ranked, gold)

    # -- generated texts ---------------------------------------------------- #
    content = common.read_json(config.CANONICAL_CONTENT) if config.CANONICAL_CONTENT.exists() else {"subsections": []}
    subs = content.get("subsections", [])
    m["survey"] = {
        "subsections": len(subs),
        **text_stats([s.get("content", "") for s in subs]),
        "words_per_subsection": [s.get("word_count") for s in subs],
        "length_flags": dict(Counter(s.get("length_flag") or "ok" for s in subs)),
        "citations_outside_given_refs": sum(len(s.get("cited_outside_given") or []) for s in subs),
        "claim_check": claim_totals([s.get("claim_check") or {} for s in subs if s.get("claim_check")]),
        "draft_vs_final_words": [sum(len(str(s.get("draft_content", "")).split()) for s in subs),
                                 sum(len(str(s.get("content", "")).split()) for s in subs)],
    }
    rev_path = config.OUT_DIR / "literature_review_2page.json"
    if rev_path.exists():
        rev = common.read_json(rev_path)
        m["review_2page"] = {"mode": rev.get("mode"), "sections": len(rev.get("sections", [])),
                             **text_stats([s.get("content", "") for s in rev.get("sections", [])]),
                             "claim_check": claim_totals([rev["claim_check"]]) if rev.get("claim_check") else None,
                             "models": rev.get("models")}
    m["v4_review_2page"] = text_stats(review_blocks_md(V4_DIR / "literature_review.md"))

    # -- run v4 side by side ----------------------------------------------- #
    v4 = pd.read_csv(V4_DIR / "paper_after_reranking_final.csv")
    v4_sel = v4[v4["tag"] == "selected"]
    m["v4"] = {"overlap_selected": overlap(list(v4_sel["title"]), gold),
               "key_works": key_works(list(v4["title"]), list(v4_sel["title"])),
               "selected_profile": selected_profile(v4_sel)}

    if config.CANONICAL_BIBLIOGRAPHY.exists():
        m["bibliography"] = bibliography_check(common.read_json(config.CANONICAL_BIBLIOGRAPHY))
    m["llm_usage"] = llm_usage()

    common.write_json(config.OUT_DIR / "metrics.json", m)
    (config.OUT_DIR / "metrics.md").write_text(render_md(m), encoding="utf-8")
    print(json.dumps({k: m[k] for k in ("mvfa_reference_funnel", "overlap_selected")}, indent=1))


def render_md(m: Dict[str, Any]) -> str:
    def row(*cells: Any) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"

    sp, v4p = m["selected_profile"], m["v4"]["selected_profile"]
    lines = ["# Run metrics", "", "## Selected references: this run vs v4", "",
             row("Measure", "This run", "v4"), row("---", "---", "---"),
             row("Overlap with MVFA's references (of %d)" % m["gold"]["entries"],
                 m["overlap_selected"]["matched"], m["v4"]["overlap_selected"]["matched"]),
             row("Overlap F1", m["overlap_selected"]["f1"], m["v4"]["overlap_selected"]["f1"]),
             row("Median year", sp["median_year"], v4p["median_year"]),
             row("Share from 2024 on", sp["share_2024_plus"], v4p["share_2024_plus"]),
             row("Share before 2020", sp["share_pre_2020"], v4p["share_pre_2020"]),
             row("arXiv-only", sp["arxiv_only"], v4p["arxiv_only"]),
             row("Origin", sp["origin"], v4p["origin"]), "",
             "## MVFA reference funnel (this run)", "", row("Stage", "MVFA references present"), row("---", "---")]
    lines += [row(k.replace("_", " "), v) for k, v in m["mvfa_reference_funnel"].items()]
    lines += ["", "## Key foundations and baselines", "", row("Work", "Pool (this run)", "Selected (this run)", "Selected (v4)"),
              row("---", "---", "---", "---")]
    for name, v in m["key_works"].items():
        lines.append(row(name, "yes" if v["in_pool"] else "no", "yes" if v["selected"] else "no",
                         "yes" if m["v4"]["key_works"][name]["selected"] else "no"))
    sc = m["scores"]
    lines += ["", "## Relevance scores", "",
              f"Scale: {sc['scale']}. Gate: score >= {sc['gate']}. Selection: {sc['selection']}. "
              f"{sc['not_judged']} of {sc['pool_records']} records not judged (no abstract). "
              "Only the score is recorded, no rationale.", "",
              row("Set", "1", "2", "3", "4", "5"), row("---", "---", "---", "---", "---", "---")]
    for label, key in (("All judged", "distribution_judged"), ("MVFA references judged", "mvfa_refs_judged_distribution"),
                       ("Selected", "distribution_selected")):
        lines.append(row(label, *sc[key].values()))
    lines += ["", f"Passing the gate: {sc['passing_gate']}; passing but not selected: {sc['passing_gate_not_selected']}. "
              f"Composite score of the selected: {sc['selected_composite']}", "",
              row("Key work", "Relevance", "Abstract", "Passes gate", "Selected"), row("---", "---", "---", "---", "---")]
    for name, v in sc["key_works"].items():
        if not v["in_pool"]:
            lines.append(row(name, "not in pool", "", "", ""))
        else:
            lines.append(row(name, v["relevance"] if v["relevance"] is not None else "not judged",
                             "yes" if v["has_abstract"] else "no", "yes" if v["passes_gate"] else "no",
                             "yes" if v["selected"] else "no"))
    lines += ["", "## Generated text", "", row("Measure", "Survey (this run)", "2-page review (this run)", "2-page review (v4)"),
              row("---", "---", "---", "---")]
    rv = m.get("review_2page") or {}
    for key in ("words", "citation_markers", "distinct_refs_cited", "citations_per_100_words",
                "sentences_with_more_than_3_citations"):
        lines.append(row(key.replace("_", " "), m["survey"].get(key), rv.get(key), m["v4_review_2page"].get(key)))
    cc = m["survey"]["claim_check"]
    lines += ["", f"Claim check (survey drafts): {cc}", "", f"Claim check (2-page review): {rv.get('claim_check')}", ""]
    if "bibliography" in m:
        b = m["bibliography"]
        lines += ["## Bibliography", "", f"{b['entries']} entries; authors {b['with_authors']}, venue {b['with_venue']}, "
                  f"year {b['with_year']}; sources {b['metadata_source']}; title mismatches {len(b['title_mismatches'])}; "
                  f"withdrawal notices {b['withdrawal_notices']}", ""]
    lines += ["## LLM calls", ""] + [row(r["stage"], r["tag"], r["model"], r["calls"]) for r in m["llm_usage"]["by_stage_tag_model"]]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
