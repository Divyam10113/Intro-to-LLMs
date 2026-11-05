"""Step 2 - run the limitation-extraction pipeline on the base paper.

  python run_pipeline.py                  # full system
  python run_pipeline.py --no_citation    # ablation: without the Citation agent / RAG
  python run_pipeline.py --no_verify      # ablation: without the Verifier

Outputs go to limitations/outputs/<run name>/.
"""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import agents
from common import (DATA_DIR, GEN_PROVIDER, OUT_DIR, TASK1_DIR, base_paper_text, chat_json, load_base_paper, read_jsonl,
                    write_json)
from retrieval import Retriever, rerank

JUDGE_THRESHOLD = 8.0   # arXiv:2601.11578: regenerate an agent whose mean rubric score is below 8/10
MAX_CITATION_PASSAGES = 30


def load_evidence():
    chunks = read_jsonl(DATA_DIR / "chunks.jsonl")
    mentions = read_jsonl(DATA_DIR / "mvfa_mentions.jsonl")
    papers = {p["pid"]: p for p in json.load(open(DATA_DIR / "papers.json"))}
    # MVFA-mention passages become citable passages too (ids M000...)
    for i, m in enumerate(mentions):
        p = papers[m["pid"]]
        chunks.append({"cid": f"M{i:03d}", "pid": m["pid"], "title": p["title"], "year": p["year"],
                       "groups": p["groups"], "section": m["section"] + " (discusses MVFA)", "part": 0,
                       "text": m["text"], "mentions_base": True})
    return chunks


def task1_review():
    p = TASK1_DIR / "item3_lit_review.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def citation_passages(retriever, chunks, abstract, method_text):
    """Query generation -> hybrid retrieval (top 20 / query) -> LLM rerank (keep >= 8)."""
    qs = chat_json(
        "You generate search queries for finding literature that reveals limitations of a paper. "
        "Return a JSON list of 6 short queries, each targeting a different potential weakness or missing "
        "capability (protocol, supervision, granularity, modalities, baselines, interpretability, ...).",
        f"ABSTRACT:\n{abstract}\n\nMETHOD (excerpt):\n{method_text[:6000]}", temperature=0, tag="queries")
    print(f"[citation] queries: {qs}")
    kept = {}
    for q in qs:
        cands = [c for c in retriever.search(q, k=20) if not c.get("mentions_base")]
        for c in rerank(abstract, q, cands, threshold=8, tag="rerank"):
            if c["cid"] not in kept or c["rerank"] > kept[c["cid"]]["rerank"]:
                kept[c["cid"]] = c
    ranked = sorted(kept.values(), key=lambda c: -c["rerank"])
    # passages where later papers explicitly discuss MVFA are always included (direct critique evidence)
    # round-robin across papers (max 3 each) so one paper with many mentions cannot fill every slot
    per_paper = {}
    for c in chunks:
        if c.get("mentions_base"):
            per_paper.setdefault(c["pid"], []).append(c)
    direct = [lst[i] for i in range(3) for lst in per_paper.values() if i < len(lst)]
    budget = max(0, MAX_CITATION_PASSAGES - min(len(direct), MAX_CITATION_PASSAGES // 2))
    passages = direct[:MAX_CITATION_PASSAGES // 2] + ranked[:budget]
    print(f"[citation] {len(direct)} MVFA-mention passages, {len(ranked)} reranked >= 8 -> using {len(passages)}")
    return passages, qs


def run_worker(name, paper, passages):
    fn = {"extractor": agents.extractor, "analyzer": agents.analyzer, "reviewer": agents.reviewer}.get(name)
    items = agents.citation(paper, passages) if name == "citation" else fn(paper)
    j1 = agents.judge(paper, name, items)
    record = {"agent": name, "judge_round1": j1, "regenerated": False}
    if j1["mean"] < JUDGE_THRESHOLD:  # one feedback round only - more rounds hurt recall (2601.11578, Tab. II/IV)
        items = (agents.citation(paper, passages, feedback=j1["feedback"]) if name == "citation"
                 else fn(paper, feedback=j1["feedback"]))
        record["judge_round2"] = agents.judge(paper, name + "-regen", items)
        record["regenerated"] = True
    record["items"] = items
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_citation", action="store_true")
    ap.add_argument("--no_verify", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--workers", type=int, default=4, help="parallel LLM calls (1 = one agent at a time)")
    ap.add_argument("--no_supp", action="store_true", help="drop the supplementary section from the paper text")
    args = ap.parse_args()
    if args.no_supp:
        os.environ["NO_SUPP"] = "1"
    name = args.name or ("full" + ("_nocite" if args.no_citation else "") + ("_noverify" if args.no_verify else "")
                         + ("_nim" if GEN_PROVIDER == "nim" else ""))
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    paper = base_paper_text()
    sections = dict(load_base_paper())
    abstract = sections.get("Abstract", paper[:2000])
    chunks = load_evidence()
    chunk_by_id = {c["cid"]: c for c in chunks}

    # ---- 1. worker agents ------------------------------------------------------
    workers = ["extractor", "analyzer", "reviewer"] + ([] if args.no_citation else ["citation"])
    passages, queries = [], []
    if not args.no_citation:
        passages, queries = citation_passages(Retriever([c for c in chunks]), chunks, abstract,
                                              sections.get("Method", ""))
        write_json(out / "citation_passages.json", {"queries": queries, "passages": passages})
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        records = list(ex.map(lambda w: run_worker(w, paper, passages), workers))
    write_json(out / "workers.json", records)
    items = [it for r in records for it in r["items"]]
    print(f"[workers] {len(items)} raw limitations: " +
          ", ".join(f"{r['agent']}={len(r['items'])} (judge {r['judge_round1']['mean']:.1f}"
                    f"{' -> regen' if r['regenerated'] else ''})" for r in records))

    # ---- 2. verifier ---------------------------------------------------------------
    agents.check_quotes(items, paper, chunk_by_id)
    if args.no_verify:
        kept = items
    else:
        grounded = [it for it in items if it["has_verified_evidence"]]
        agents.verify_support(paper, grounded, chunk_by_id)
        kept = [it for it in grounded if it["verdict"] in ("supported", "partial")]
        print(f"[verifier] {len(items)} -> {len(grounded)} with a verbatim quote -> {len(kept)} supported/partial")
    write_json(out / "verified.json", {"all": items, "kept_ids": [it["id"] for it in kept]})

    # ---- 3. master -------------------------------------------------------------------
    final = agents.master(paper, kept)
    print(f"[master] {len(final)} consolidated limitations")

    # ---- 4. gap status against later literature ---------------------------------------
    later = [c for c in chunks if (c.get("year") or 0) >= 2024 and not c.get("mentions_base")]
    review = task1_review()
    if later:
        r_later = Retriever(later)

        def status(lim):
            cands = r_later.search(lim["limitation"] + " " + lim.get("explanation", ""), k=8)
            return agents.gap_status(lim, cands, review[:12000])

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for lim, st in zip(final, ex.map(status, final)):
                lim["status"] = st.get("status")
                lim["addressed_by"] = [{"cid": c, "title": chunk_by_id.get(c, {}).get("title")}
                                       for c in st.get("addressed_by", [])]
                lim["remaining_gap"] = st.get("remaining_gap")
    write_json(out / "final_limitations.json", final)

    # ---- 5. research problem -------------------------------------------------------------
    m = re.search(r"(## \d*\.?\s*Cross-Cutting.*?)(?=\n## |\Z)", review, re.S | re.I)
    review_gaps = m.group(1) if m else review[-4000:]
    slim = [{k: lim.get(k) for k in ("lid", "limitation", "explanation", "aspect", "provenance", "importance",
                                      "status", "remaining_gap")} for lim in final]
    md = agents.formulate_problem(abstract, slim, review_gaps)
    (out / "research_gaps.md").write_text(render(md, final, records, name), encoding="utf-8")
    print(f"\n[done] {out / 'research_gaps.md'}")


def render(problem_md, final, records, name):
    lines = [f"# Research gaps and problem statement - MVFA (arXiv:2403.12570)\n",
             f"_Run: `{name}`. Generated by the multi-agent pipeline in `limitations/`._\n", problem_md.strip(),
             "\n\n---\n\n## Appendix A - limitations with evidence\n"]
    for lim in final:
        lines.append(f"### {lim['lid']}. {lim['limitation']}\n")
        lines.append(f"- **Aspect:** {lim.get('aspect')} | **Provenance:** {', '.join(lim.get('provenance', []))} "
                     f"| **Importance:** {lim.get('importance')}/5 | **Status in later work:** {lim.get('status')}")
        lines.append(f"- {lim.get('explanation', '')}")
        if lim.get("remaining_gap"):
            lines.append(f"- **Still open:** {lim['remaining_gap']}")
        if lim.get("addressed_by"):
            lines.append("- **Addressed (partly) by:** " + "; ".join(
                f"{a['title']} [{a['cid']}]" for a in lim["addressed_by"] if a.get("title")))
        for e in lim.get("evidence", [])[:4]:
            lines.append(f"  > [{e['source']}] \"{e['quote']}\"")
        lines.append("")
    lines.append("## Appendix B - agent quality scores (Judge, 1-10)\n")
    lines.append("| agent | round 1 mean | regenerated | round 2 mean |\n|---|---|---|---|")
    for r in records:
        r2 = r.get("judge_round2", {}).get("mean")
        lines.append(f"| {r['agent']} | {r['judge_round1']['mean']:.2f} | {r['regenerated']} | "
                     f"{'' if r2 is None else f'{r2:.2f}'} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
