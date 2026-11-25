"""
Stage 4 — Final multi-criteria ranking and selection of the top-5 ideas per research problem.

Pool (option (b), chosen by the user): every candidate HypoEvolve produced for a problem (6 seeds + 18 offspring),
not only the final population.
  1. Shortlist: the 12 candidates with the highest HypoEvolve fitness (each candidate's most recent Bradley-Terry
     fitness; scales are comparable across generations because each refit keeps the multiplier and anchors the offset
     on surviving candidates).
  2. Novelty evidence: closest prior work from a literature search per shortlisted idea (data/novelty_search.json;
     novelty_search.py re-runs the queries on Semantic Scholar).
  3. Review: one ResearchAgent-style review per idea scoring Novelty, Feasibility, Clarity, Impact and Validity (1-5).
  4. Families: one call per problem groups all candidates into method families.
  5. Final score = 0.7 * (mean review score - 1) / 4 + 0.3 * (fitness - min) / (max - min) over the shortlist.
  6. Selection: greedy by final score with at most 2 ideas per family (diversity rule), until 5 ideas are chosen.
  Per-criterion ranks are reported as well, so ideas can be compared on each criterion separately.
"""
import json
from pathlib import Path
from statistics import mean

from corpus import base_paper, load_surveygen
from llm_client import LLM, PendingCalls
from prompts import ranking as RK
from prompts.hyper import CODE_MAP
from prompts.researchagent import RESOURCES
from stage3_evolve import render
from utils import parse_json

ROOT = Path(__file__).resolve().parent
SHORTLIST, TOP_K, MAX_PER_FAMILY = 12, 5, 2
W_REVIEW, W_FIT = 0.7, 0.3


def latest_fitness(c):
    return c["fitness"][max(c["fitness"], key=int)]


def run(pid, problem, judge, base, novelty):
    evo = json.loads((ROOT / "outputs" / f"stage3_evolution_{pid}.json").read_text())
    cands = evo["candidates"]
    ranked = sorted(cands, key=lambda c: -latest_fitness(cands[c]))
    short = ranked[:SHORTLIST]

    fam_req = dict(system=RK.FAMILY_SYSTEM, user=RK.FAMILY_USER.format(
        problem=problem, n=len(cands),
        ideas="\n\n".join(f"[{c}] {cands[c]['title']}\n{cands[c]['summary']}" for c in sorted(cands))),
        tag=f"stage4/{pid}/families", temperature=0.2)
    rev_reqs = []
    for c in short:
        nv = novelty[c]
        prior = "\n".join(f"- {p['title']} ({p['url']}): {p['relation']}" for p in nv["closest"]) or "- none found"
        rev_reqs.append(dict(system=RK.REVIEW_SYSTEM, user=RK.REVIEW_USER.format(
            problem=problem, title=base["title"], abstract=base["abstract"], code_map=CODE_MAP, resources=RESOURCES,
            query=nv["query"], prior=prior, criteria=RK.criteria_block(), idea=render(cands[c])),
            tag=f"stage4/{pid}/review/{c}", temperature=0.2))
    resps = judge.batch([fam_req] + rev_reqs)
    families = parse_json(resps[0])["families"]
    fam_of = {c: f for f, ids in families.items() for c in ids}
    reviews = {c: parse_json(r) for c, r in zip(short, resps[1:])}

    fits = {c: latest_fitness(cands[c]) for c in short}
    lo, hi = min(fits.values()), max(fits.values())
    rows = []
    for c in short:
        sc = {k: int(reviews[c][k]["score"]) for k in RK.CRITERIA}
        m = mean(sc.values())
        final = W_REVIEW * (m - 1) / 4 + W_FIT * (fits[c] - lo) / (hi - lo)
        rows.append(dict(id=c, title=cands[c]["title"], family=fam_of.get(c, "unassigned"), scores=sc,
                         review_mean=round(m, 2), fitness=fits[c], final=round(final, 4),
                         survived=c in evo["final_population"], reviews=reviews[c], novelty=novelty[c]))
    rows.sort(key=lambda r: -r["final"])
    for crit in RK.CRITERIA:   # per-criterion ranks (1 = best; ties share the best rank)
        vals = sorted({r["scores"][crit] for r in rows}, reverse=True)
        for r in rows:
            r.setdefault("criterion_rank", {})[crit] = 1 + sum(1 for x in rows if x["scores"][crit] > r["scores"][crit])
    chosen, per_fam, skipped = [], {}, []
    for r in rows:
        if len(chosen) == TOP_K:
            break
        if per_fam.get(r["family"], 0) >= MAX_PER_FAMILY:
            skipped.append(dict(id=r["id"], reason=f"family '{r['family']}' already has {MAX_PER_FAMILY} selected ideas"))
            continue
        per_fam[r["family"]] = per_fam.get(r["family"], 0) + 1
        chosen.append(r["id"])
    for i, r in enumerate(rows, 1):
        r["overall_rank"] = i
    return dict(problem=pid, problem_text=problem, pool_size=len(cands), shortlist=short, families=families,
                ranking=rows, top5=chosen, skipped_for_diversity=skipped,
                weights=dict(review=W_REVIEW, fitness=W_FIT, max_per_family=MAX_PER_FAMILY))


def main():
    judge = LLM("judge")
    base = base_paper(load_surveygen())
    novelty = json.loads((ROOT / "data" / "novelty_search.json").read_text())
    s1 = json.loads((ROOT / "outputs" / "stage1_problems.json").read_text())
    probs = {c["id"]: c["versions"][c["best_version"]]["problem"] for c in s1["candidates"] if c["id"] in s1["proposed"]}
    pending = []
    for pid in sorted(probs):
        try:
            out = run(pid, probs[pid], judge, base, novelty)
        except PendingCalls as e:
            pending += e.keys
            continue
        p = ROOT / "outputs" / f"stage4_ranking_{pid}.json"
        p.write_text(json.dumps(out, indent=1))
        print(f"wrote {p}")
        for r in out["ranking"]:
            mark = "TOP5" if r["id"] in out["top5"] else "    "
            print(f"  {mark} {r['overall_rank']:2d} {r['id']:9s} final={r['final']:.3f} review={r['review_mean']} "
                  f"fit={r['fitness']} {r['scores']} [{r['family']}]")
    if pending:
        raise PendingCalls(pending)


if __name__ == "__main__":
    main()
