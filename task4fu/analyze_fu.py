"""Follow-up analysis: applies RULES_FU.md exactly (frozen before any follow-up run).

    python analyze_fu.py [results.csv] [out.md]

Reads only results/results.csv (written by train2.py after evaluate_target.py) and, for the
side-by-side table, round 3's results.csv. Writes results/analysis_fu.md and prints the outcome.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

FU = Path(__file__).resolve().parent
SEEDS = (8, 9, 10, 11, 12)
R3_SEEDS = (3, 4, 5)
ARMS = ("mvfa", "mil_m10")
NAMES = {"mvfa": "MVFA (re-run)", "mil_m10": "MIL+10%"}
PAPER_PIX, R0_AUPRO = 90.27, 65.35          # MVFA paper Brain pixel AUROC; round 1's R0 Brain AUPRO
K = ("img_auroc", "pix_auroc", "aupro")


def load(path):
    """d[arm][seed] = {metric: points, best_epoch, diag_p}; Brain fold, Brain test set, non-smoke rows."""
    d = {a: {} for a in ARMS}
    if not Path(path).exists():
        return d
    for r in csv.DictReader(open(path)):
        if "SMOKE" in r.get("notes", "") or r["arm"] not in ARMS or r["fold"] != "Brain" or r["eval_dataset"] != "Brain":
            continue
        if r["run_id"] != f"{r['arm']}_Brain_s{r['seed']}":
            continue
        v = {k: float(r[k]) * 100 for k in K if r[k] not in ("", None)}
        v.update(best_epoch=int(r["best_epoch"]), diag_p=float(r["diag_inmask_p"]))
        d[r["arm"]][int(r["seed"])] = v
    return d


def test(d, seeds=SEEDS):
    pairs = [s for s in seeds if s in d["mvfa"] and s in d["mil_m10"]
             and "aupro" in d["mvfa"][s] and "aupro" in d["mil_m10"][s]]
    dau = {s: d["mil_m10"][s]["aupro"] - d["mvfa"][s]["aupro"] for s in pairs}
    dpix = {s: d["mil_m10"][s]["pix_auroc"] - d["mvfa"][s]["pix_auroc"] for s in pairs}
    dimg = {s: d["mil_m10"][s]["img_auroc"] - d["mvfa"][s]["img_auroc"] for s in pairs}
    pos = sum(v > 0 for v in dau.values())                       # ties count against MIL+10 %
    n = len(pairs)
    p = binomtest(pos, n, 0.5, alternative="greater").pvalue if n else float("nan")
    mpix = float(np.mean(list(dpix.values()))) if n else float("nan")
    return dict(pairs=pairs, n=n, pos=pos, p=p, dau=dau, dpix=dpix, dimg=dimg, mpix=mpix)


def outcome(t, need=len(SEEDS)):
    if t["n"] < need:
        return f"incomplete ({t['n']}/{need} seed pairs; AUPRO higher on {t['pos']}/{t['n']}); cannot be CONFIRMED"
    if t["pos"] == need and t["mpix"] > 0:
        return "CONFIRMED"
    if t["pos"] == need - 1:
        return "holds on most seeds (not confirmed)"
    return "did not replicate"


def fmt(v):
    return "–" if not v else (f"{np.mean(v):.2f}" if len(v) == 1 else f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}")


def main(path=None, out=None):
    path = Path(path or FU / "results" / "results.csv")
    out = Path(out or FU / "results" / "analysis_fu.md")
    d = load(path)
    t = test(d)
    res = outcome(t)
    L = ["# Follow-up analysis: MIL+10 % vs MVFA on Brain (RULES_FU.md, applied by `analyze_fu.py`)", "",
         f"Source: `{path.name}`. Points = AUROC/AUPRO × 100. Seeds {list(SEEDS)}; round 3's seeds are never pooled.", "",
         "## Per seed (Brain test set)", "",
         "| seed | MVFA img | MVFA pix | MVFA AUPRO | MIL+10% img | MIL+10% pix | MIL+10% AUPRO | Δ AUPRO | Δ pix | Δ img |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        a, b = d["mvfa"].get(s, {}), d["mil_m10"].get(s, {})
        g = lambda x, k: f"{x[k]:.2f}" if k in x else "–"
        dd = lambda m: f"{m[s]:+.2f}" if s in m else "–"
        L.append(f"| {s} | {g(a, 'img_auroc')} | {g(a, 'pix_auroc')} | {g(a, 'aupro')} | {g(b, 'img_auroc')} | "
                 f"{g(b, 'pix_auroc')} | {g(b, 'aupro')} | {dd(t['dau'])} | {dd(t['dpix'])} | {dd(t['dimg'])} |")
    L += ["", "## FU-H: MIL+10 % beats MVFA on Brain (the one pre-registered test)", "",
          f"- AUPRO higher on **{t['pos']}/{t['n']}** seed pairs; one-sided sign test p = {t['p']:.4f}",
          f"- Mean pixel AUROC Δ (MIL+10 % − MVFA): {t['mpix']:+.2f}",
          f"- **Outcome: {res}**", "",
          "## Descriptive (not tested)", ""]
    for a in ARMS:
        L.append(f"- {NAMES[a]} Brain, seeds 8–12: img {fmt([v['img_auroc'] for v in d[a].values()])}, "
                 f"pix {fmt([v['pix_auroc'] for v in d[a].values()])}, AUPRO {fmt([v['aupro'] for v in d[a].values()])}; "
                 f"best epochs {[d[a][s]['best_epoch'] for s in SEEDS if s in d[a]]}; "
                 f"in-mask {fmt([v['diag_p'] for v in d[a].values()]) if d[a] else '–'}")
    if t["n"]:
        L.append(f"- Mean image AUROC Δ (MIL+10 % − MVFA): {np.mean(list(t['dimg'].values())):+.2f}")
    m = d["mil_m10"]
    if m:
        L.append(f"- MIL+10 % pixel AUROC above the MVFA paper's {PAPER_PIX} on "
                 f"{sum(v['pix_auroc'] > PAPER_PIX for v in m.values())}/{len(m)} seeds "
                 f"(mean {np.mean([v['pix_auroc'] for v in m.values()]):.2f}). Different protocol: context only.")
        L.append(f"- MIL+10 % AUPRO above R0's {R0_AUPRO} (released checkpoint, round 1) on "
                 f"{sum(v['aupro'] > R0_AUPRO for v in m.values())}/{len(m)} seeds. Context only.")
    r3 = FU.parent / "task4r3" / "results" / "results.csv"
    if r3.exists():
        d3 = load(r3)
        t3 = test(d3, R3_SEEDS)
        L += ["", "## Round 3 seeds 3–5 (the post-hoc observation; side by side, never pooled)", "",
              "| arm | seeds | img | pix | AUPRO |", "|---|---|---|---|---|"]
        for a in ARMS:
            vs = [d3[a][s] for s in R3_SEEDS if s in d3[a]]
            L.append(f"| {NAMES[a]} | 3–5 | {fmt([v['img_auroc'] for v in vs])} | {fmt([v['pix_auroc'] for v in vs])} | "
                     f"{fmt([v['aupro'] for v in vs])} |")
        L.append(f"\nRound 3: AUPRO higher on {t3['pos']}/{t3['n']} seeds; mean pix Δ {t3['mpix']:+.2f}.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"FU-H: {res}  (AUPRO higher {t['pos']}/{t['n']}, p = {t['p']:.4f}, mean pix Δ {t['mpix']:+.2f}) -> {out}")
    return res


if __name__ == "__main__":
    main(*sys.argv[1:3])
