"""
Stage 3 — Idea evolution with a generational genetic algorithm (HypoEvolve, Liu et al., 2026; Algorithm 1).

Per research problem (goal g):
  P0      = the 6 HypER seed ideas of Stage 2 (HypoEvolve's generation agent proposes mu literature-grounded
            hypotheses; here that role is filled by HypER, which adds provenance).
  f0      = SCORE(P0): every unordered pair compared by the pairwise judge; Bradley-Terry fit; the initial fit fixes the
            scale so that the population mean is 50 and the spread (max - min) is 60.
  for t = 1..G:
      while |O_t| < lambda:
          p1 = TOURNAMENT(P_{t-1}, size 2)
          with prob p_c: p2 = TOURNAMENT until p2 != p1; h = CROSSOVER(p1, p2)  (combination | inspiration, uniform)
          if no crossover or with prob p_m: h = MUTATION(h)                    (substitution | out-of-box, uniform)
          empty / unparsable operator output -> unchanged parent copy (fallback, recorded)
      Q_t = P_{t-1} + O_t ; f_t = SCORE(Q_t) ; P_t = Top_mu(Q_t)
      Later fits keep the multiplier a and shift the offset b_t so that survivors keep their mean score.
  return h* = argmax f_G, plus the whole final population and lineage records.

Settings from the paper: mu = 6, lambda = 6, G = 3, p_c = 0.6, p_m = 0.15.

Implementation notes / deviations (documented in the report):
  * Pairwise order (position bias): the paper shows each pair once. We do the same (BOTH_ORDERS = False); which idea
    is shown as "A" is decided by a hash of the two ids, so it is random but reproducible. Judging each pair in both
    orders and counting disagreement as a tie (BOTH_ORDERS = True) is the planned debiased variant (future work).
  * A pair that was already judged in an earlier generation produces an identical prompt and is answered from the
    cache (equivalent to a deterministic judge), so only new pairs cost calls.
  * Each offspring's operator call carries its own sample index, so two offspring produced by the same operator on
    the same parent are independent samples (as with temperature sampling in the paper), not one cached answer.
  * Bradley-Terry is fitted with the MM algorithm of Hunter (2004) with a weak symmetric prior (0.1 pseudo-tie per
    compared pair) so that undefeated candidates get finite strengths.
"""
import hashlib
import json
import math
import random
from pathlib import Path

from llm_client import LLM, PendingCalls
from prompts import hypoevolve as HE
from prompts.hyper import CODE_MAP
from prompts.researchagent import PROTOCOL_C
from utils import parse_json

ROOT = Path(__file__).resolve().parent
MU, LAMBDA, G, PC, PM = 6, 6, 3, 0.6, 0.15
BOTH_ORDERS = False
PRIOR = 0.1
SEED = 2026


# ------------------------------------------------------------------------------------------------ helpers
def render(idea):
    parts = [f"TITLE: {idea['title']}", f"SUMMARY: {idea['summary']}", f"HYPOTHESIS: {idea['hypothesis']}"]
    if idea.get("research_idea"):
        parts.append(f"RESEARCH IDEA: {idea['research_idea']}")
    parts += [f"RATIONALE: {idea.get('rationale', '')}", f"METHOD: {idea['method']}", f"EXPERIMENT: {idea['experiment']}"]
    return "\n".join(parts)


def flip(a, b):
    """Deterministic pseudo-random presentation order of a pair."""
    h = hashlib.sha256("|".join(sorted([a, b])).encode()).digest()[0]
    return h % 2 == 1


def bradley_terry(ids, results, iters=500):
    """results: list of (i, j, s) with s = 1 (i wins), 0 (j wins), 0.5 (tie)."""
    idx = {c: k for k, c in enumerate(ids)}
    n = len(ids)
    W = [0.0] * n
    N = [[0.0] * n for _ in range(n)]
    for i, j, s in results:
        a, b = idx[i], idx[j]
        W[a] += s + PRIOR / 2
        W[b] += (1 - s) + PRIOR / 2
        N[a][b] += 1 + PRIOR
        N[b][a] += 1 + PRIOR
    p = [1.0] * n
    for _ in range(iters):
        new = []
        for a in range(n):
            den = sum(N[a][b] / (p[a] + p[b]) for b in range(n) if N[a][b] > 0)
            new.append(W[a] / den if den > 0 else p[a])
        g = math.exp(sum(math.log(x) for x in new) / n)
        p = [x / g for x in new]
    return {c: p[idx[c]] for c in ids}


class Run:
    def __init__(self, pid, problem_text, seeds):
        self.pid = pid
        self.goal = f"{problem_text}\n\n{CODE_MAP}\n\nRequired evaluation protocol: {PROTOCOL_C}"
        self.problem_text = problem_text
        self.rng = random.Random(f"{SEED}-{pid}")
        self.cands = {}
        for s in seeds:
            cid = f"{pid}-g0-{s['chain']}"
            self.cands[cid] = dict(id=cid, gen=0, parents=[], operators=["hyper-seed"], chain=s["chain"],
                                   title=s["title"], summary=s["summary"], research_idea=s.get("research_idea"),
                                   hypothesis=s["hypothesis"], rationale=s.get("rationale"), method=s["method"],
                                   experiment=s["experiment"], provenance=s.get("provenance", []), fitness={})
        self.comparisons = {}   # (a, b) sorted -> dict
        self.history = []
        self.judge, self.gen = LLM("judge"), LLM("generator")

    # -------------------------------------------------------------- scoring
    def score(self, pool, t):
        reqs, meta = [], []
        pool = sorted(pool)
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                a, b = pool[i], pool[j]
                orders = [(a, b), (b, a)] if BOTH_ORDERS else [((b, a) if flip(a, b) else (a, b))]
                for x, y in orders:
                    reqs.append(dict(system=HE.COMPARE_SYSTEM, user=HE.COMPARE_USER.format(
                        problem=self.problem_text, a=render(self.cands[x]), b=render(self.cands[y])),
                        tag=f"stage3/{self.pid}/compare/{x}__vs__{y}", temperature=0.2))
                    meta.append((a, b, x, y))
        results = {}
        for (a, b, x, y), resp in zip(meta, self.judge.batch(reqs)):
            w = resp.split("WINNER=")[-1].strip()[:3].upper()
            s_xy = 1.0 if w.startswith("A") else 0.0 if w.startswith("B") else 0.5
            s_ab = s_xy if x == a else 1 - s_xy
            reasoning = resp.split("REASONING:")[-1].split("CONFIDENCE:")[0].strip()
            conf = resp.split("CONFIDENCE:")[-1].split("WINNER")[0].strip() if "CONFIDENCE:" in resp else ""
            results.setdefault((a, b), []).append(s_ab)
            self.comparisons[(a, b)] = dict(a=a, b=b, shown_first=x, winner=a if s_ab == 1 else b if s_ab == 0 else "TIE",
                                            reasoning=reasoning, confidence=conf, generation=t)
        res = [(a, b, sum(v) / len(v)) for (a, b), v in results.items()]
        strengths = bradley_terry(pool, res)
        logs = {c: math.log(strengths[c]) for c in pool}
        if t == 0:
            lo, hi = min(logs.values()), max(logs.values())
            self.a = 60.0 / (hi - lo) if hi > lo else 1.0
            b = 50.0 - self.a * sum(logs.values()) / len(logs)
        else:
            surv = [c for c in pool if t - 1 in self.cands[c]["fitness"]]
            prev = sum(self.cands[c]["fitness"][t - 1] for c in surv) / len(surv)
            b = prev - self.a * sum(logs[c] for c in surv) / len(surv)
        for c in pool:
            self.cands[c]["fitness"][t] = round(self.a * logs[c] + b, 2)
        return {c: self.cands[c]["fitness"][t] for c in pool}

    def reviews_for(self, cid, k=3):
        out = [f"vs '{self.cands[r['b'] if r['a'] == cid else r['a']]['title']}': "
               f"{'won' if r['winner'] == cid else 'tie' if r['winner'] == 'TIE' else 'lost'} — {r['reasoning']}"
               for r in self.comparisons.values() if cid in (r["a"], r["b"])]
        return "\n".join(out[-k:]) if out else "None yet."

    # -------------------------------------------------------------- evolution
    def tournament(self, pop, fit):
        x, y = self.rng.sample(pop, 2)
        return x if fit[x] >= fit[y] else y

    def evolve(self, pop, fit, t):
        plans = []
        while len(plans) < LAMBDA:
            p1 = self.tournament(pop, fit)
            plan = dict(p1=p1, cross=None, p2=None, mut=None)
            if self.rng.random() < PC:
                p2 = self.tournament(pop, fit)
                while p2 == p1:
                    p2 = self.tournament(pop, fit)
                plan.update(cross=self.rng.choice(["combination", "inspiration"]), p2=p2)
            if plan["cross"] is None or self.rng.random() < PM:
                plan["mut"] = self.rng.choice(["substitution", "out_of_box"])
            plans.append(plan)
        # crossover calls (batch), then mutation calls (batch)
        tmpl = {"combination": HE.COMBINATION, "inspiration": HE.INSPIRATION,
                "substitution": HE.SUBSTITUTION, "out_of_box": HE.OUT_OF_BOX}
        reqs, idx = [], []
        for k, pl in enumerate(plans):
            if pl["cross"]:
                hyp = "\n\n".join(f"Parent {n + 1}:\n{render(self.cands[p])}" for n, p in enumerate([pl["p1"], pl["p2"]]))
                rv = "\n".join(f"Parent {n + 1}: {self.reviews_for(p)}" for n, p in enumerate([pl["p1"], pl["p2"]]))
                reqs.append(dict(system=HE.EVOLVE_SYSTEM, user=tmpl[pl["cross"]].format(
                    goal=self.goal, hypotheses=hyp, reviews=rv, preferences=HE.PREFERENCES, output_format=HE.OUTPUT_FORMAT),
                    tag=f"stage3/{self.pid}/g{t}/o{k + 1}/{pl['cross']}", temperature=0.8, sample=100 * t + k + 1))
                idx.append(k)
        inter = {}
        for k, resp in zip(idx, self.gen.batch(reqs)):
            inter[k] = self._parse(resp)
        reqs, idx = [], []
        for k, pl in enumerate(plans):
            if pl["mut"]:
                base = inter.get(k) or self.cands[pl["p1"]]
                rv = self.reviews_for(pl["p1"])
                reqs.append(dict(system=HE.EVOLVE_SYSTEM, user=tmpl[pl["mut"]].format(
                    goal=self.goal, hypotheses=render(base), reviews=rv, preferences=HE.PREFERENCES,
                    output_format=HE.OUTPUT_FORMAT), tag=f"stage3/{self.pid}/g{t}/o{k + 1}/{pl['mut']}", temperature=0.8,
                    sample=100 * t + k + 1))  # distinct sample per offspring: identical prompts must not share an answer
                idx.append(k)
        final = dict(inter)
        for k, resp in zip(idx, self.gen.batch(reqs)):
            final[k] = self._parse(resp) or inter.get(k)
        offspring = []
        for k, pl in enumerate(plans):
            cid = f"{self.pid}-g{t}-o{k + 1}"
            ops = [o for o in (pl["cross"], pl["mut"]) if o]
            idea = final.get(k)
            if not idea:   # empty operator return -> unchanged parent copy
                idea = {f: self.cands[pl["p1"]][f] for f in ("title", "summary", "hypothesis", "rationale", "method", "experiment", "provenance")}
                ops = ops + ["fallback-copy"]
            self.cands[cid] = dict(id=cid, gen=t, parents=[p for p in (pl["p1"], pl["p2"]) if p], operators=ops,
                                   research_idea=None, fitness={}, **idea)
            offspring.append(cid)
        return offspring

    @staticmethod
    def _parse(resp):
        try:
            js = parse_json(resp)
            idea = {k.lower(): v for k, v in js.items()}
            need = ("title", "summary", "hypothesis", "rationale", "method", "experiment")
            if all(idea.get(f) for f in need):
                idea["provenance"] = idea.get("provenance") or []
                return {f: idea[f] for f in need + ("provenance",)}
        except Exception:
            pass
        return None

    def run(self):
        pop = sorted(self.cands)
        fit = self.score(pop, 0)
        self.history.append(dict(generation=0, population=pop, fitness=fit))
        for t in range(1, G + 1):
            off = self.evolve(pop, fit, t)
            pool = pop + off
            f = self.score(pool, t)
            pop = sorted(pool, key=lambda c: -f[c])[:MU]
            fit = {c: f[c] for c in pop}
            self.history.append(dict(generation=t, offspring=off, pool_fitness=f, population=pop, fitness=fit))
        best = max(pop, key=lambda c: fit[c])
        return dict(problem=self.pid, settings=dict(mu=MU, lam=LAMBDA, G=G, pc=PC, pm=PM, both_orders=BOTH_ORDERS,
                                                    prior=PRIOR, seed=SEED, scale_a=self.a),
                    best=best, final_population=pop, history=self.history, candidates=self.cands,
                    comparisons=list(self.comparisons.values()))


def main():
    seeds = json.loads((ROOT / "outputs" / "stage2_seeds.json").read_text())
    s1 = json.loads((ROOT / "outputs" / "stage1_problems.json").read_text())
    probs = {c["id"]: c["versions"][c["best_version"]]["problem"] for c in s1["candidates"] if c["id"] in s1["proposed"]}
    pending = []
    for pid in sorted(probs):
        try:
            out = Run(pid, probs[pid], [s for s in seeds["seeds"] if s["problem"] == pid]).run()
        except PendingCalls as e:
            pending += e.keys
            continue
        p = ROOT / "outputs" / f"stage3_evolution_{pid}.json"
        p.write_text(json.dumps(out, indent=1))
        print(f"wrote {p}; best = {out['best']}")
        for h in out["history"]:
            print(f"  gen {h['generation']}: " + ", ".join(f"{c.split('-', 1)[1]}={h['fitness'][c]}" for c in h["population"]))
    if pending:
        raise PendingCalls(pending)


if __name__ == "__main__":
    main()
