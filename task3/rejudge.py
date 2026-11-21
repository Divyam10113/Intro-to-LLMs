"""
rejudge.py — independent re-judging of this run with a different model (planned step of the hybrid plan).

Why a separate script: re-running stages 1-4 with only the JUDGE_* variables changed does not replay cleanly,
because later generator prompts depend on earlier judgments (Stage 1 refinements include the reviews; Stage 3
parents are chosen from the fitness). Changing the judge therefore produces new generator prompts that are not
in the cache. A full re-run needs an API model for both roles (set LLM_* instead).

This script instead re-asks the new judge EXACTLY the judge prompts of this run (read from logs/cache) and
measures agreement with the original judgments:
  Stage 1  problem reviews (1-5 per criterion)            -> Spearman, exact and within-1 agreement
  Stage 2  HypER relevance scores (0/1/2)                 -> agreement and quadratic-weighted kappa
  Stage 3  pairwise comparisons                           -> winner agreement; Bradley-Terry refit on all
                                                             comparisons per problem, Spearman vs original
  Stage 3' (optional, "stage3swap") the same comparisons with A and B swapped -> self-consistency, position bias,
                                                             Bradley-Terry on both orders averaged
  Stage 4  idea reviews (5 criteria)                      -> Spearman per criterion and for the mean;
                                                             top-5 recomputed with the new reviews (same
                                                             formula and diversity rule), overlap with ours

Usage (open model on a Colab GPU, no rate limits; see colab/T3_independent_judge.ipynb):
    export JUDGE_BACKEND=openai JUDGE_BASE_URL=http://localhost:8000/v1 JUDGE_API_KEY=none
    export JUDGE_MODEL=Qwen/Qwen3-8B JUDGE_CONCURRENCY=16
    python3 rejudge.py                 # all stages (437 calls, about 2.5M prompt tokens)
Usage (Groq):
    export JUDGE_BACKEND=openai JUDGE_BASE_URL=https://api.groq.com/openai/v1
    export JUDGE_API_KEY=gsk_...  JUDGE_MODEL=openai/gpt-oss-120b
    python3 rejudge.py stage3 stage4   # Stage 1 review prompts (~27k tokens each) exceed free-tier per-minute limits
Answers are cached under the new model's name, so an interrupted run (e.g. a daily token limit) resumes where it
stopped when the command is repeated.
Output: outputs/rejudge_<model>.json
"""
import glob
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from llm_client import LLM, FILE_BACKEND_MODEL
from stage3_evolve import bradley_terry
from stage4_rank import W_REVIEW, W_FIT, TOP_K, MAX_PER_FAMILY
from prompts import ranking as RK
from utils import parse_json

ROOT = Path(__file__).resolve().parent
PREFIX = {"stage1": "stage1/review/", "stage2": "stage2/relevance/", "stage3": "/compare/", "stage4": "/review/"}


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 3) if den else None


def qwk(x, y, k):
    n = len(x)
    O = [[0] * k for _ in range(k)]
    for a, b in zip(x, y):
        O[a][b] += 1
    hx = [sum(O[i]) for i in range(k)]
    hy = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = sum(((i - j) ** 2) * O[i][j] for i in range(k) for j in range(k))
    den = sum(((i - j) ** 2) * hx[i] * hy[j] / n for i in range(k) for j in range(k))
    return round(1 - num / den, 3) if den else None


def rating(text):
    # "Rating (1-5): 4", "Rating (1–5): **4**", "**Rating:** 4"; failing that, "4/5"
    # also "**Rating (1-5):**\n**4**" and "### Rating (1-5)\n\n**5**" (markdown around the number, line breaks, no colon)
    m = (re.findall(r"Rating(?:\s*\(\s*1\s*[-–]\s*5\s*\))?[\s:*#_]*([1-5])\b", text)
         or re.findall(r"\b([1-5])\s*/\s*5\b", text))
    return int(m[-1]) if m else None


def winner(text):
    """1.0 if A wins, 0.0 if B wins, 0.5 for a tie, None if the answer has no verdict."""
    m = re.findall(r"WINNER\**\s*[=:]\s*\**\s*(TIE|A|B)\b", text, re.I)
    if not m:
        return None
    w = m[-1].upper()
    return 1.0 if w == "A" else 0.0 if w == "B" else 0.5


def tail(rs, new, keep):
    """Up to five examples of answers that could not be read, so the reason can be checked."""
    return [dict(tag=r["tag"], answer_end=n[-300:]) for r, n in zip(rs, new) if not keep(n)][:5]


def relevance(text):
    try:
        j = parse_json(text)
    except Exception:
        return {}
    out = {}
    def key(k):   # '3.Title', '3. Title' and '3' all mean paper 3 of the chunk
        m = re.match(r"\s*(\d+)", str(k))
        return int(m.group(1)) if m else k
    pl = j.get("paper_list", j) if isinstance(j, dict) else {}
    for k, v in pl.items():
        if isinstance(v, dict) and "relevance" in v:
            try:
                out[key(k)] = int(v["relevance"])
            except (TypeError, ValueError):
                pass
    return out


def main(stages):
    judge = LLM("judge")
    if judge.cfg.model == FILE_BACKEND_MODEL or judge.cfg.backend == "file":
        raise SystemExit("Set JUDGE_BACKEND=openai and JUDGE_MODEL (and URL/key) to a different model first.")
    recs = [json.loads(Path(f).read_text()) for f in glob.glob(str(ROOT / "logs" / "cache" / "*.json"))]
    # Only this run's judgments: calls logged in logs/calls.jsonl, plus the second-hop relevance chunks reused
    # unchanged from an earlier run (chains extended in this run). The cache also holds the archived runs' records.
    this_run = {json.loads(l)["key"] for l in (ROOT / "logs" / "calls.jsonl").read_text().splitlines() if l.strip()}
    chains = json.loads((ROOT / "outputs" / "stage2_chains.json").read_text())
    hop2 = tuple(f"stage2/relevance/hop2/{k[:8]}/" for k in chains.get("hop2_scored", {}))
    orig = [r for r in recs if r["role"] == "judge" and r["model"] == FILE_BACKEND_MODEL
            and (r["key"] in this_run or (hop2 and r["tag"].startswith(hop2)))]
    report = dict(new_judge=judge.cfg.model, original_judge=FILE_BACKEND_MODEL)
    new_fitness, swap_fitness = {}, {}

    def redo(rs):
        reqs = [dict(system=r["system"], user=r["user"], tag="rejudge/" + r["tag"], temperature=r.get("temperature", 0.2),
                     sample=r.get("sample", 0)) for r in rs]
        return judge.batch(reqs)

    if "stage1" in stages:
        rs = sorted([r for r in orig if r["tag"].startswith(PREFIX["stage1"])], key=lambda r: r["tag"])
        new = redo(rs)
        pairs, per = [], defaultdict(list)
        for r, n in zip(rs, new):
            a, b = rating(r["response"]), rating(n)
            if a and b:
                pairs.append((a, b))
                per[r["tag"].split("/")[-1]].append((a, b))
        report["stage1"] = dict(n=len(pairs), asked=len(rs), unparsed_examples=tail(rs, new, rating), spearman=spearman(*zip(*pairs)),
                                exact=round(mean(a == b for a, b in pairs), 3),
                                within1=round(mean(abs(a - b) <= 1 for a, b in pairs), 3),
                                mean_original=round(mean(a for a, _ in pairs), 2), mean_new=round(mean(b for _, b in pairs), 2),
                                per_criterion={k: spearman(*zip(*v)) for k, v in per.items()})

    if "stage2" in stages:
        rs = [r for r in orig if r["tag"].startswith(PREFIX["stage2"])]
        new = redo(rs)
        xs, ys = [], []
        for r, n in zip(rs, new):
            a, b = relevance(r["response"]), relevance(n)
            for k in a:
                if k in b:
                    xs.append(a[k]); ys.append(b[k])
        report["stage2"] = dict(n=len(xs), exact=round(mean(a == b for a, b in zip(xs, ys)), 3) if xs else None,
                                qwk=qwk(xs, ys, 3) if xs else None)

    if "stage3" in stages:
        report["stage3"] = {}
        for pid in ("A1", "B1"):
            rs = [r for r in orig if r["tag"].startswith(f"stage3/{pid}/compare/")]
            new = redo(rs)
            evo = json.loads((ROOT / "outputs" / f"stage3_evolution_{pid}.json").read_text())
            both, res_o, res_n = [], [], []
            for r, n in zip(rs, new):
                x, y = r["tag"].split("/compare/")[1].split("__vs__")
                so, sn = winner(r["response"]), winner(n)
                if so is not None:
                    res_o.append((x, y, so))
                if sn is not None:
                    res_n.append((x, y, sn))
                if so is not None and sn is not None:
                    both.append((so, sn))
            decisive = [(a, b) for a, b in both if a != 0.5 and b != 0.5]
            ids = sorted(evo["candidates"])
            bo, bn = bradley_terry(ids, res_o), bradley_terry(ids, res_n)
            new_fitness[pid] = bn
            share = lambda v, w: round(sum(1 for x in v if x == w) / len(v), 3) if v else None
            report["stage3"][pid] = dict(
                n=len(rs), parsed=len(both), unparsed_examples=tail(rs, new, lambda t: winner(t) is not None),
                winner_agreement=round(mean(a == b for a, b in both), 3),
                winner_agreement_decisive=round(mean(a == b for a, b in decisive), 3) if decisive else None,
                n_decisive=len(decisive),
                # position bias check: how often each judge picks the idea shown first (A), and how often it says TIE
                share_A_original=share([a for a, _ in both], 1.0), share_A_new=share([b for _, b in both], 1.0),
                ties_original=share([a for a, _ in both], 0.5), ties_new=share([b for _, b in both], 0.5),
                bt_spearman=spearman([bo[c] for c in ids], [bn[c] for c in ids]),
                new_best=max(bn, key=bn.get), original_best=max(bo, key=bo.get))

    if "stage3swap" in stages:
        # Same comparisons with Idea A and Idea B swapped (new judge only; the original verdicts come from one
        # ranking per generation, so they cannot depend on the order). Measures the new judge's consistency with
        # itself and its position bias, and averages the two orders before refitting Bradley-Terry.
        report["stage3_swap"] = {}
        for pid in ("A1", "B1"):
            rs = [r for r in orig if r["tag"].startswith(f"stage3/{pid}/compare/")]
            fwd = redo(rs)   # already cached by the stage3 re-judging
            swapped = []
            for r in rs:
                head, rest = r["user"].split("\nIdea A:\n", 1)
                a, rest = rest.split("\n\nIdea B:\n", 1)
                b, tailtxt = rest.split("\n\nRespond in exactly", 1)
                swapped.append(dict(system=r["system"], user=head + "\nIdea A:\n" + b + "\n\nIdea B:\n" + a
                                    + "\n\nRespond in exactly" + tailtxt, tag="rejudge/swap/" + r["tag"],
                                    temperature=r.get("temperature", 0.2), sample=r.get("sample", 0)))
            bwd = judge.batch(swapped)
            evo = json.loads((ROOT / "outputs" / f"stage3_evolution_{pid}.json").read_text())
            ids = sorted(evo["candidates"])
            res_o, res_c, rows3 = [], [], []
            for r, f, g in zip(rs, fwd, bwd):
                x, y = r["tag"].split("/compare/")[1].split("__vs__")
                so, s1, s2 = winner(r["response"]), winner(f), winner(g)
                if so is not None:
                    res_o.append((x, y, so))
                if s1 is None or s2 is None:
                    continue
                s2x = 1 - s2                     # the swapped answer, from x's point of view
                comb = (s1 + s2x) / 2            # 1 or 0 if both orders agree, 0.5 if they contradict
                res_c.append((x, y, comb))
                rows3.append((so, s1, s2, s2x, comb))
            consistent = [t for t in rows3 if t[1] == t[3]]
            firsts = [t[1] for t in rows3] + [t[2] for t in rows3]   # 1.0 = the idea shown first won
            bo, bc = bradley_terry(ids, res_o), bradley_terry(ids, res_c)
            swap_fitness[pid] = bc
            report["stage3_swap"][pid] = dict(
                n=len(rs), parsed=len(rows3), unparsed_examples=tail(rs, bwd, lambda t: winner(t) is not None),
                self_consistency=round(len(consistent) / len(rows3), 3) if rows3 else None,
                first_shown_wins=round(sum(1 for v in firsts if v == 1.0) / len(firsts), 3) if firsts else None,
                ties_new=round(sum(1 for v in firsts if v == 0.5) / len(firsts), 3) if firsts else None,
                agreement_when_consistent=round(mean(t[0] == t[4] for t in consistent if t[0] is not None), 3) if consistent else None,
                n_consistent=len(consistent),
                bt_spearman_both_orders=spearman([bo[c] for c in ids], [bc[c] for c in ids]),
                new_best_both_orders=max(bc, key=bc.get), original_best=max(bo, key=bo.get))

    if "stage4" in stages:
        report["stage4"] = {}
        for pid in ("A1", "B1"):
            rk = json.loads((ROOT / "outputs" / f"stage4_ranking_{pid}.json").read_text())
            rs = sorted([r for r in orig if r["tag"].startswith(f"stage4/{pid}/review/")], key=lambda r: r["tag"])
            new = redo(rs)
            rows = {x["id"]: x for x in rk["ranking"]}
            newsc = {}
            for r, n in zip(rs, new):
                cid = r["tag"].split("/")[-1]
                try:
                    j = parse_json(n)
                    newsc[cid] = {k: int(j[k]["score"]) for k in RK.CRITERIA}
                except Exception:
                    pass
            ids = [c for c in rows if c in newsc]
            per = {k: spearman([rows[c]["scores"][k] for c in ids], [newsc[c][k] for c in ids]) for k in RK.CRITERIA}
            fits = {c: rows[c]["fitness"] for c in ids}
            lo, hi = min(fits.values()), max(fits.values())
            final = {c: W_REVIEW * (mean(newsc[c].values()) - 1) / 4 + W_FIT * (fits[c] - lo) / (hi - lo) for c in ids}
            def top(final):
                chosen, per_fam = [], defaultdict(int)
                for c in sorted(ids, key=lambda c: -final[c]):
                    if len(chosen) == TOP_K:
                        break
                    if per_fam[rows[c]["family"]] < MAX_PER_FAMILY:
                        per_fam[rows[c]["family"]] += 1
                        chosen.append(c)
                return chosen
            chosen = top(final)
            # fully independent version: the new judge's reviews AND the new judge's tournament fitness
            # (Stage 3 must be re-judged in the same command); the shortlist itself is kept
            indep = None
            if pid in new_fitness:
                nf = {c: new_fitness[pid][c] for c in ids}
                lo2, hi2 = min(nf.values()), max(nf.values())
                final2 = {c: W_REVIEW * (mean(newsc[c].values()) - 1) / 4
                          + W_FIT * ((nf[c] - lo2) / (hi2 - lo2) if hi2 > lo2 else 0) for c in ids}
                t2 = top(final2)
                indep = dict(top5=t2, overlap=len(set(t2) & set(rk["top5"])))
            indep_both = None
            if pid in swap_fitness:   # same, with the tournament judged in both orders
                nf = {c: swap_fitness[pid][c] for c in ids}
                lo3, hi3 = min(nf.values()), max(nf.values())
                final3 = {c: W_REVIEW * (mean(newsc[c].values()) - 1) / 4
                          + W_FIT * ((nf[c] - lo3) / (hi3 - lo3) if hi3 > lo3 else 0) for c in ids}
                t3 = top(final3)
                indep_both = dict(top5=t3, overlap=len(set(t3) & set(rk["top5"])))
            report["stage4"][pid] = dict(n=len(ids), spearman_per_criterion=per,
                                         spearman_mean=spearman([rows[c]["review_mean"] for c in ids],
                                                                [mean(newsc[c].values()) for c in ids]),
                                         top5_original=rk["top5"], top5_new=chosen,
                                         top5_overlap=len(set(chosen) & set(rk["top5"])),
                                         top5_fully_independent=indep,
                                         top5_fully_independent_both_orders=indep_both)
    out = ROOT / "outputs" / f"rejudge_{re.sub(r'[^A-Za-z0-9.-]+', '_', judge.cfg.model)}.json"
    if out.exists():   # keep results of stages run earlier (e.g. stage3 today, stage4 tomorrow)
        old = json.loads(out.read_text())
        report = {**old, **report}
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    # stage3swap (Stage 3 comparisons again with A and B swapped, 336 extra calls) runs only when named:
    #     python3 rejudge.py stage1 stage2 stage3 stage3swap stage4
    main(sys.argv[1:] or ["stage1", "stage2", "stage3", "stage4"])
