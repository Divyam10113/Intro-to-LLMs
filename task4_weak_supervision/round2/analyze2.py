"""Apply the plan's §6 rules (fixed before any run; operational details G1-15)
mechanically to results/results.csv; writes results/analysis.md.

    python analyze2.py
"""
import csv
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

from common2 import R1, RESULTS2

T = ["Brain", "Liver", "Retina_RESC"]
PRIMARY = ["mvfa", "mil", "pu", "g", "pu_m100", "g_m100"]
NAMES = {"mvfa": "MVFA (re-run)", "mil": "MIL", "pu": "PU (π=1%)", "g": "G (gated MIL)",
         "pu_m100": "PU+100%", "g_m100": "G+100%", "pu_pi5": "PU-π5"}
K = ("img_auroc", "pix_auroc", "aupro")


def load():
    rows = list(csv.DictReader(open(RESULTS2 / "results.csv")))
    d = defaultdict(dict)       # d[arm][(dataset, seed)] = {metric: value in points}
    for r in rows:
        arm = r["arm"] + ("_pi5" if r["run_id"].endswith("_pi5") else "")
        if r["run_id"].split("_s")[-1].split("_")[0] != r["seed"]:
            continue
        v = {k: float(r[k]) * 100 for k in K if r[k] not in ("", None)}
        v.update(best_epoch=int(r["best_epoch"]), diag_p=float(r["diag_inmask_p"]), diag_pg=float(r["diag_inmask_pg"]))
        d[arm][(r["eval_dataset"], int(r["seed"]))] = v
    return d


def r0():
    out = {}
    for r in csv.DictReader(open(R1 / "results" / "results.csv")):
        if r["config"] == "R0":
            out[r["target"]] = {k: float(r[k]) * 100 for k in K}
    return out


def seeds(d, arm, t, ss=(0, 1, 2)):
    return [s for s in ss if (t, s) in d.get(arm, {})]


def mean(d, arm, t, k, ss=(0, 1, 2)):
    v = [d[arm][(t, s)][k] for s in seeds(d, arm, t, ss) if k in d[arm][(t, s)]]
    return float(np.mean(v)) if v else float("nan")


def fmt(d, arm, t, k, ss=(0, 1, 2)):
    v = [d[arm][(t, s)][k] for s in seeds(d, arm, t, ss) if k in d[arm][(t, s)]]
    if not v:
        return "–"
    return f"{np.mean(v):.2f}" if len(v) == 1 else f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}"


def beats(d, arm, ss=(0, 1, 2)):
    pairs = [(t, s) for t in T for s in ss if (t, s) in d.get(arm, {}) and (t, s) in d.get("mvfa", {})]
    dau = [d[arm][p]["aupro"] - d["mvfa"][p]["aupro"] for p in pairs]
    pos = sum(x > 0 for x in dau)
    pix_hi = sum(mean(d, arm, t, "pix_auroc", ss) > mean(d, "mvfa", t, "pix_auroc", ss) for t in T)
    img_ok = all(mean(d, arm, t, "img_auroc", ss) >= mean(d, "mvfa", t, "img_auroc", ss) - 0.5 for t in T)
    p = binomtest(pos, len(pairs), 0.5, alternative="greater").pvalue if pairs else float("nan")
    return dict(n=len(pairs), pos=pos, pix_hi=pix_hi, img_ok=img_ok, p=p, dau=dau)


def main():
    d, R0 = load(), r0()
    L = ["# Round 2 analysis (§6 rules applied mechanically by `analyze2.py`)\n",
         "All numbers are in points (×100). Mean ± std over seeds 0–2 unless marked; Chest/HIS/OCT17 = seed 0 "
         "image AUROC only (OCT17 from the RESC+OCT17 fold).\n"]
    # ---- main table
    L += ["## Main table (masked targets)\n", "| target | arm | n | img AUROC | pix AUROC | AUPRO | best epochs |",
          "|---|---|---|---|---|---|---|"]
    for t in T:
        L.append(f"| {t} | R0 (released, round 1; different protocol) | 1 | {R0[t]['img_auroc']:.2f} | "
                 f"{R0[t]['pix_auroc']:.2f} | {R0[t]['aupro']:.2f} | – |")
        for arm in PRIMARY + ["pu_pi5"]:
            ss = (0, 1, 2) if arm != "pu_pi5" else (0,)
            n = len(seeds(d, arm, t, ss))
            if n:
                be = ",".join(str(d[arm][(t, s)]["best_epoch"]) for s in seeds(d, arm, t, ss))
                L.append(f"| {t} | {NAMES[arm]} | {n} | " + " | ".join(fmt(d, arm, t, k, ss) for k in K) + f" | {be} |")
    L += ["", "## Image AUROC on the mask-less held-out datasets (seed 0)\n", "| dataset | " +
          " | ".join(NAMES[a] for a in ["mvfa", "mil", "pu", "g"]) + " |", "|---|---|---|---|---|"]
    for t in ["Retina_OCT2017", "Chest", "Histopathology"]:
        L.append(f"| {t} | " + " | ".join(fmt(d, a, t, "img_auroc", (0,)) for a in ["mvfa", "mil", "pu", "g"]) + " |")

    # ---- idea #1
    L += ["", "## Rule: idea #1 (gated MIL) against Task 3's hypothesis\n"]
    gd = {t: mean(d, "g", t, "pix_auroc") - mean(d, "mil", t, "pix_auroc") for t in T}
    gi = {t: mean(d, "g", t, "img_auroc", (0,)) - mean(d, "mil", t, "img_auroc", (0,)) for t in ["Chest", "Histopathology"]}
    sup = all(v >= 2 for v in gd.values()) and all(v >= -0.5 for v in gi.values())
    fals = np.nanmean(list(gd.values())) <= 0
    v1 = "SUPPORTED" if sup else ("FALSIFIED (no pixel-AUROC gain over MIL)" if fals else
                                  "NOT SUPPORTED (gain below +2 on at least one target)")
    L += [f"- G − MIL pixel AUROC (mean over seeds): " + ", ".join(f"{t} {v:+.2f}" for t, v in gd.items()),
          f"- G − MIL image AUROC, seed 0: " + ", ".join(f"{t} {v:+.2f}" for t, v in gi.items()),
          f"- **Verdict: {v1}**"]
    # ---- idea #3
    L += ["", "## Rule: refined idea #3 (PU, fixed π = 1 %)\n"]
    pd_ = {t: mean(d, "pu", t, "pix_auroc") - mean(d, "mil", t, "pix_auroc") for t in T}
    n3 = sum(v >= 1.5 for v in pd_.values())
    L += [f"- PU − MIL pixel AUROC: " + ", ".join(f"{t} {v:+.2f}" for t, v in pd_.items()),
          f"- **Verdict: {'SUPPORTED' if n3 >= 2 else 'NOT SUPPORTED'}** ({n3}/3 targets ≥ +1.5; rule needs ≥ 2)",
          "- The estimated-π part of Task 3's hypothesis: reported as failed in round 1 (B11), not retested.",
          f"- Sensitivity π = 5 % (seed 0), PU-π5 − PU pixel AUROC: " +
          ", ".join(f"{t} {mean(d, 'pu_pi5', t, 'pix_auroc', (0,)) - mean(d, 'pu', t, 'pix_auroc', (0,)):+.2f}" for t in T)]
    # ---- beating MVFA
    L += ["", "## Rule: does an arm beat MVFA (re-run under the same rule)?\n",
          "| arm | AUPRO Δ>0 pairs | sign-test p | pix higher on targets | img within 0.5 on all | seeds 0–2 result | seeds 3–4 |",
          "|---|---|---|---|---|---|---|"]
    qualify = []
    for arm in ["g", "pu", "g_m100", "pu_m100"]:
        b = beats(d, arm)
        if not b["n"]:
            continue
        if b["pos"] >= 8 and b["pix_hi"] >= 2 and b["img_ok"]:
            res = "qualifies → needs seeds 3–4"
            qualify.append(arm)
        elif b["pos"] >= 7:
            res = "suggestive"
        else:
            res = "does not beat MVFA"
        c = beats(d, arm, (3, 4))
        conf = "–"
        if c["n"]:
            ok = c["pos"] >= 5 and c["pix_hi"] >= 2
            conf = f"{c['pos']}/{c['n']}, pix {c['pix_hi']}/3 → " + ("**BEATS MVFA**" if ok and res.startswith("qual") else "not confirmed (suggestive)")
        L.append(f"| {NAMES[arm]} | {b['pos']}/{b['n']} | {b['p']:.3f} | {b['pix_hi']}/3 | {'yes' if b['img_ok'] else 'no'} | {res} | {conf} |")
    L += ["", "Paired AUPRO Δ vs MVFA (Brain s0,s1,s2 | Liver | RESC):"]
    for arm in ["g", "pu", "g_m100", "pu_m100"]:
        b = beats(d, arm)
        if b["n"]:
            L.append(f"- {NAMES[arm]}: " + " ".join(f"{x:+.2f}" for x in b["dau"]))
    # ---- retention
    L += ["", "## Retention of MVFA's localisation (mask-free arms)\n",
          "(pix − 50) / (MVFA re-run − 50), mean over seeds; [vs R0 in brackets, round 1's definition]\n",
          "| arm | " + " | ".join(T) + " | mean |", "|---|---|---|---|---|"]
    for arm in ["mil", "pu", "g"]:
        rv, r0v = [], []
        for t in T:
            rv.append((mean(d, arm, t, "pix_auroc") - 50) / (mean(d, "mvfa", t, "pix_auroc") - 50))
            r0v.append((mean(d, arm, t, "pix_auroc") - 50) / (R0[t]["pix_auroc"] - 50))
        L.append(f"| {NAMES[arm]} | " + " | ".join(f"{a:.3f} [{b:.3f}]" for a, b in zip(rv, r0v)) +
                 f" | {np.nanmean(rv):.3f} [{np.nanmean(r0v):.3f}] |")
    # ---- diagnostic
    L += ["", "## Task 3's diagnostic for #1: share of top-k patches inside the true mask (masked sources' valid/ abnormal images)\n",
          "| arm | by p | by p·g |", "|---|---|---|"]
    for arm in PRIMARY:
        vals = [v for v in d.get(arm, {}).values()]
        if vals:
            L.append(f"| {NAMES[arm]} | {np.nanmean([v['diag_p'] for v in vals]):.3f} | {np.nanmean([v['diag_pg'] for v in vals]):.3f} |")
    L += ["", f"Arms qualifying for confirmation seeds: {qualify or 'none'}"]
    (RESULTS2 / "analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return qualify


if __name__ == "__main__":
    main()
