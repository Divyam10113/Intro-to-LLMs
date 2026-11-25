"""
Stage 1 — Research problem discovery (ResearchAgent, Baek et al. 2025).

Input gaps: outputs/gaps_verified.json (Stage 0 groups with the code-verified corrections of Stage 0b).
For each selected limitation group (run 3: A and B, see GROUPS; the leakage-free protocol group is attached to
every problem as the shared evaluation protocol):
  1. Problem identification  p = LLM(T_p(target paper, related papers, entities, limitations))
     — two candidates per group, the second asked to differ from the first.
  2. ReviewingAgents score every candidate separately on Clarity, Relevance, Originality,
     Feasibility and Significance (one call per criterion, 1-5 Likert with the human-induced rubric).
  3. Iterative refinement from the reviews; ResearchAgent reports saturation after ~3 iterations,
     so we run the initial round + up to 2 refinements (3 reviewed versions per candidate), keeping the
     best-scoring version of each candidate.
  4. The best candidate of each group (the top-2 by mean score from different groups) is proposed as a
     research problem.

Literature
  * target paper: MVFA (base paper)
  * related papers: the other papers of the Task 1 literature review (run 3: the 58 other selected papers of
    Task 1's final_ref_set.json; run 2: 59 QUAL-SG v4 references; run 1: 16 SurveyGen papers), plus the 10
    papers citing MVFA (Task 1 + Semantic Scholar) whose abstracts are most similar to MVFA's and that are not
    already in the review
  * entities: top-k from the entity-centric knowledge store over the whole corpus (Eq. 2)
"""
import json
import re
from pathlib import Path
from statistics import mean

from corpus import EntityStore, base_paper, load_surveygen, mvfa_citers, review_set
from llm_client import LLM
from prompts import researchagent as RA
from utils import parse_fields

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "stage1_problems.json"
# Run 3 (option a, decided with the user): only the two directions that Task 4 builds on. Group A covers Task 2's
# L3, L4, L5, L18 (mask and label dependence); group B covers L8, L10, L30, L38 plus the code-only C1 and C3 (scoring).
# Runs 1-2 also generated problems for "Compact normality modelling", which scored lowest both times.
GROUPS = ["Label-efficient adaptation", "Score calibration and robust aggregation"]
GROUP_LETTER = {"Label-efficient adaptation": "A", "Score calibration and robust aggregation": "B",
                "Compact normality modelling": "D"}
N_CANDIDATES = 2
N_REFINE = 2           # initial + 2 refinements = 3 reviewed versions
N_ENTITIES = 20
N_CITING = 10
CRITERIA = ["Clarity", "Relevance", "Originality", "Feasibility", "Significance"]


def citing_papers(n, base_abstract, papers):
    """ResearchAgent Sec. 3.2: the core paper's citing papers 'are further narrowed down based on their
    similarities of abstracts with the core paper'. Citing papers come from Task 1 (origin 'cites_mvfa') and the
    Semantic Scholar supplement (corpus.mvfa_citers). TF-IDF cosine similarity is used (swap in a sentence encoder
    if one is available)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    cit = [x for x in mvfa_citers(papers) if x.get("abstract")]
    if not cit:
        return []
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform([base_abstract] + [x["abstract"] for x in cit])
    sims = cosine_similarity(X[0], X[1:])[0]
    order = sims.argsort()[::-1][:n]
    return [dict(title=cit[i]["title"], abstract=cit[i]["abstract"], year=cit[i].get("year"),
                 source="citing:" + "+".join(cit[i]["sources"]), similarity=round(float(sims[i]), 3)) for i in order]


def fmt_related(papers):
    return "\n".join(f"[{i}] {p['title']} ({p.get('year') or 'n.d.'})\n    Abstract: {p['abstract'] or '(not available; title only)'}"
                     for i, p in enumerate(papers, 1))


def rating(text):
    m = re.search(r"Rating\s*\(1-5\)\s*:\s*\**\s*([1-5])", text)
    return int(m.group(1)) if m else None


def pick_two(ranked):
    """Top-2 problems by best mean score, from different limitation groups."""
    first = ranked[0]
    second = next(c for c in ranked[1:] if c["group"] != first["group"])
    return [first["id"], second["id"]]


def main():
    papers = load_surveygen()
    base = base_paper(papers)
    from corpus import norm_title
    rev = [p for p in review_set(papers) if p is not base]
    seen = {norm_title(p["title"]) for p in rev}
    related = rev + [c for c in citing_papers(N_CITING + len(rev), base["abstract"], papers)
                     if norm_title(c["title"]) not in seen][:N_CITING]
    store = EntityStore(papers)
    entities, _ = store.retrieve([base] + related, k=N_ENTITIES)
    s0 = json.loads((ROOT / "outputs" / "gaps_verified.json").read_text())  # Stage 0 + 0b (code-verified)
    gaps = {g["id"]: g for g in s0["gaps"]}
    groups = {g["name"]: g for g in s0["groups"]}

    gen, judge = LLM("generator"), LLM("judge")
    ctx = dict(title=base["title"], abstract=base["abstract"], related=fmt_related(related),
               resources=RA.RESOURCES, protocol=RA.PROTOCOL_C)

    def lims(group):
        # evidence tags such as [E9, E10] refer to stage0b_code_verification.json and are removed from prompts
        def head(t):  # Task 2 titles are full sentences; C-items have short titles
            return t + " " if t.rstrip().endswith((".", "?", "!")) else t + ": "
        return "\n".join(f"- {gid}. {head(gaps[gid]['title'])}{re.sub(r' ?\[E[0-9E, -]+\]', '', gaps[gid]['text'])}"
                          for gid in group["members"])

    # ---------------------------------------------------------------- 1. problem identification
    cands = {}
    for s in range(N_CANDIDATES):
        reqs, keys = [], []
        for gname in GROUPS:
            g = groups[gname]
            prev = cands.get((gname, s - 1))
            div = RA.DIVERSITY.format(previous=prev["versions"][0]["problem"]) if prev else ""
            user = RA.PROBLEM_USER.format(n_ref=len(related), n_ent=len(entities), group_name=gname,
                                          root_cause=g["root_cause"], limitations=lims(g),
                                          entities=entities, diversity=div, **ctx)
            reqs.append(dict(system=RA.PROBLEM_SYSTEM, user=user, tag=f"stage1/problem/{GROUP_LETTER[gname]}{s+1}/v0",
                             temperature=0.8))
            keys.append(gname)
        for gname, resp in zip(keys, gen.batch(reqs)):
            f = parse_fields(resp, ["Problem", "Rationale"])
            cands[(gname, s)] = dict(id=f"{GROUP_LETTER[gname]}{s+1}", group=gname, versions=[
                dict(problem=f["Problem"], rationale=f["Rationale"])])

    # ---------------------------------------------------------------- 2+3. review / refine loop
    for it in range(N_REFINE + 1):
        reqs, idx = [], []
        for c in cands.values():
            v = c["versions"][it]
            for m in CRITERIA:
                user = RA.REVIEW_PROBLEM_USER.format(metric=m, short=RA.SHORT["problem"][m],
                                                     criteria=RA.criteria_text("problem", m),
                                                     problem=v["problem"], rationale=v["rationale"], **ctx)
                reqs.append(dict(system=RA.REVIEW_PROBLEM_SYSTEM, user=user,
                                 tag=f"stage1/review/{c['id']}/v{it}/{m}", temperature=0.2))
                idx.append((c, m))
        for (c, m), resp in zip(idx, judge.batch(reqs)):
            f = parse_fields(resp, ["Review", "Feedback", "Rating (1-5)"])
            v = c["versions"][it]
            v.setdefault("reviews", {})[m] = dict(review=f.get("Review", ""), feedback=f.get("Feedback", ""),
                                                  rating=rating(resp))
        for c in cands.values():
            v = c["versions"][it]
            v["mean"] = round(mean(r["rating"] for r in v["reviews"].values()), 3)
        if it == N_REFINE:
            break
        reqs, order = [], []
        for c in cands.values():
            v = c["versions"][it]
            rv = "\n".join(f"- {m}: rating {r['rating']}/5. Review: {r['review']} Feedback: {r['feedback']}"
                           for m, r in v["reviews"].items())
            user = RA.REFINE_USER.format(problem=v["problem"], rationale=v["rationale"], reviews=rv,
                                         limitations=lims(groups[c["group"]]), **ctx)
            reqs.append(dict(system=RA.PROBLEM_SYSTEM, user=user, tag=f"stage1/problem/{c['id']}/v{it+1}",
                             temperature=0.7))
            order.append(c)
        for c, resp in zip(order, gen.batch(reqs)):
            f = parse_fields(resp, ["Problem", "Rationale"])
            c["versions"].append(dict(problem=f["Problem"], rationale=f["Rationale"]))

    # ---------------------------------------------------------------- 4. select
    for c in cands.values():
        best = max(range(len(c["versions"])), key=lambda i: (c["versions"][i]["mean"], i))
        c["best_version"] = best
        c["best_mean"] = c["versions"][best]["mean"]
    ranked = sorted(cands.values(), key=lambda c: (-c["best_mean"],
                                                   -c["versions"][c["best_version"]]["reviews"]["Feasibility"]["rating"]))
    out = dict(entities=entities, related=[dict(title=p["title"], year=p.get("year"), source=p["source"], similarity=p.get("similarity")) for p in related],
               candidates=ranked, proposed=pick_two(ranked))
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
    for c in ranked:
        print(f"{c['id']}  best v{c['best_version']}  mean {c['best_mean']}  trajectory "
              f"{[v['mean'] for v in c['versions']]}")


if __name__ == "__main__":
    main()
