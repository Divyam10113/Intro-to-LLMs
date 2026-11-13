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
  Stage 4  idea reviews (5 criteria)                      -> Spearman per criterion and for the mean;
                                                             top-5 recomputed with the new reviews (same
                                                             formula and diversity rule), overlap with ours

Usage (Groq example):
    export JUDGE_BACKEND=openai JUDGE_BASE_URL=https://api.groq.com/openai/v1
    export JUDGE_API_KEY=gsk_...  JUDGE_MODEL=llama-3.3-70b-versatile
    python3 rejudge.py                 # all stages
    python3 rejudge.py stage1 stage4   # a subset
Answers are cached under the new model's name, so an interrupted run resumes where it stopped.
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
    m = re.findall(r"Rating\s*\(1-5\)\s*:\s*\**\s*([1-5])", text)
    return int(m[-1]) if m else None


def winner(text):
    w = text.split("WINNER=")[-1].strip()[:3].upper()
    return 1.0 if w.startswith("A") else 0.0 if w.startswith("B") else 0.5


def relevance(text):
    try:
        j = parse_json(text)
    except Exception:
        return {}
    out = {}
    for k, v in j.items():
        if isinstance(v, dict) and "relevance" in v:
            out[k] = int(v["relevance"])
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, dict) and "relevance" in vv:
                    out[kk] = int(vv["relevance"])
    return out


def main(stages):
    judge = LLM("judge")
    if judge.cfg.model == FILE_BACKEND_MODEL or judge.cfg.backend == "file":
        raise SystemExit("Set JUDGE_BACKEND=openai and JUDGE_MODEL (and URL/key) to a different model first.")
    recs = [json.loads(Path(f).read_text()) for f in glob.glob(str(ROOT / "logs" / "cache" / "*.json"))]
    # Only this run's judgments: calls logged in logs/calls.jsonl, plus the prompts reused unchanged from the
    # previous run (Stage 0 grouping, HypER relevance). The cache also holds the archived run's records.
    this_run = {json.loads(l)["key"] for l in (ROOT / "logs" / "calls.jsonl").read_text().splitlines() if l.strip()}
    orig = [r for r in recs if r["role"] == "judge" and r["model"] == FILE_BACKEND_MODEL
            and (r["key"] in this_run or r["tag"].startswith(("stage0/", "stage2/relevance")))]
    report = dict(new_judge=judge.cfg.model, original_judge=FILE_BACKEND_MODEL)

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
        report["stage1"] = dict(n=len(pairs), spearman=spearman(*zip(*pairs)),
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
            agree, res_o, res_n = [], [], []
            for r, n in zip(rs, new):
                x, y = r["tag"].split("/compare/")[1].split("__vs__")
                so, sn = winner(r["response"]), winner(n)
                agree.append(so == sn)
                res_o.append((x, y, so)); res_n.append((x, y, sn))
            ids = sorted(evo["candidates"])
            bo, bn = bradley_terry(ids, res_o), bradley_terry(ids, res_n)
            report["stage3"][pid] = dict(n=len(rs), winner_agreement=round(mean(agree), 3),
                                         bt_spearman=spearman([bo[c] for c in ids], [bn[c] for c in ids]),
                                         new_best=max(bn, key=bn.get), original_best=max(bo, key=bo.get))

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
            chosen, per_fam = [], defaultdict(int)
            for c in sorted(ids, key=lambda c: -final[c]):
                if len(chosen) == TOP_K:
                    break
                if per_fam[rows[c]["family"]] < MAX_PER_FAMILY:
                    per_fam[rows[c]["family"]] += 1
                    chosen.append(c)
            report["stage4"][pid] = dict(n=len(ids), spearman_per_criterion=per,
                                         spearman_mean=spearman([rows[c]["review_mean"] for c in ids],
                                                                [mean(newsc[c].values()) for c in ids]),
                                         top5_original=rk["top5"], top5_new=chosen,
                                         top5_overlap=len(set(chosen) & set(rk["top5"])))
    out = ROOT / "outputs" / f"rejudge_{re.sub(r'[^A-Za-z0-9.-]+', '_', judge.cfg.model)}.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:] or ["stage1", "stage2", "stage3", "stage4"])
