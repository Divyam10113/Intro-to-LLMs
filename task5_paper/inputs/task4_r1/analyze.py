"""Step 5: apply the pre-registered decision rules to results/results.csv.

    python analyze.py            # prints and writes results/analysis.md

Works on partial results (e.g. Liver only, for the early-stop check).
All differences are in AUROC/AUPRO points (x100).
"""
import csv
import json
import math
from collections import defaultdict

import numpy as np

from common import RESULTS_DIR, TARGETS

PTS = 100.0
NEG = 0.3  # +-0.3 points = "no effect" band


def load(path=RESULTS_DIR / "results.csv"):
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    for r in rows:
        for k in ("img_auroc", "pix_auroc", "aupro"):
            r[k] = float(r[k]) if r.get(k) not in (None, "") else math.nan
    return rows


def sign_p(k, n):
    """One-sided sign test P(X >= k), X ~ Bin(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def main():
    rows = load()
    untagged = lambda r, cfg: r["config"] == cfg and r["run_id"] == f"{cfg}_{r['target']}_s{r['seed']}"
    r0 = {r["target"]: r for r in rows if r["config"] == "R0"}
    by = defaultdict(dict)  # by[cfg][(target, seed)] = row (last one wins)
    for r in rows:
        if r["config"] in ("mh", "mh_ctrl", "m", "b0") and untagged(r, r["config"]):
            by[r["config"]][(r["target"], int(r["seed"]))] = r
    out = ["# Task 4 analysis (Step 5 rules)\n"]

    # ---- table (a): M-h vs M-h-ctrl vs R0
    paper = {r["target"]: r for r in csv.DictReader(open(RESULTS_DIR / "paper_numbers.csv"))}
    pis = json.load(open(RESULTS_DIR / "pi.json")) if (RESULTS_DIR / "pi.json").exists() else {}
    est = (json.load(open(RESULTS_DIR / "pi_estimated_seg.json"))
           if (RESULTS_DIR / "pi_estimated_seg.json").exists() else {})
    out.append("## (a) M-h vs M-h-ctrl\n")
    out.append("| target | config | n | img AUROC | pix AUROC | AUPRO |")
    out.append("|---|---|---|---|---|---|")
    fmt = lambda v: f"{np.mean(v) * PTS:.2f} ± {np.std(v, ddof=1) * PTS:.2f}" if len(v) > 1 else (
        f"{v[0] * PTS:.2f}" if v else "–")
    for t in TARGETS:
        if t in paper:
            out.append(f"| {t} | R0 (paper) | – | {paper[t]['img_auroc']} | {paper[t]['pix_auroc']} | – |")
        if t in r0:
            out.append(f"| {t} | R0 (reproduced) | 1 | {fmt([r0[t]['img_auroc']])} | {fmt([r0[t]['pix_auroc']])} "
                       f"| {fmt([r0[t]['aupro']])} |")
        for cfg in ("mh", "mh_ctrl"):
            rs = [r for (tt, _), r in sorted(by[cfg].items()) if tt == t]
            if rs:
                out.append(f"| {t} | {cfg} | {len(rs)} | " + " | ".join(fmt([r[k] for r in rs])
                           for k in ("img_auroc", "pix_auroc", "aupro")) + " |")
        if t in pis:
            out.append(f"| {t} | π used (D1) | | " + ", ".join(f"{d}: {v['pi']:.3f}" for d, v in pis[t].items())
                       + " | | |")
        if t in est:
            out.append(f"| {t} | π estimator, raw (report only) | | "
                       + ", ".join(f"{d}: {v['pi_raw']:.3f}" for d, v in est[t].items()) + " | | |")

    deltas = {"pix_auroc": [], "aupro": []}
    per_t = defaultdict(lambda: defaultdict(list))
    out.append("\n**Paired Δ = M-h − M-h-ctrl (points)**\n")
    out.append("| target | seed | Δ pix AUROC | Δ AUPRO |")
    out.append("|---|---|---|---|")
    for key in sorted(set(by["mh"]) & set(by["mh_ctrl"])):
        a, b = by["mh"][key], by["mh_ctrl"][key]
        d = {k: (a[k] - b[k]) * PTS for k in deltas}
        for k in deltas:
            deltas[k].append(d[k])
            per_t[key[0]][k].append(d[k])
        out.append(f"| {key[0]} | {key[1]} | {d['pix_auroc']:+.2f} | {d['aupro']:+.2f} |")

    n = len(deltas["aupro"])
    out.append("")
    if n:
        k_pos = sum(x > 0 for x in deltas["aupro"])
        t_pos = sum(np.mean(per_t[t]["aupro"]) > 0 for t in per_t)
        mean_all = float(np.mean(deltas["aupro"]))
        out.append(f"AUPRO Δ > 0 in **{k_pos}/{n}** pairs (one-sided sign test p = {sign_p(k_pos, n):.3f}); "
                   f"mean Δ > 0 on {t_pos}/{len(per_t)} targets; overall mean Δ AUPRO = {mean_all:+.2f}, "
                   f"pix AUROC = {np.mean(deltas['pix_auroc']):+.2f}.\n")
        if n == 9:
            if abs(mean_all) <= NEG or k_pos <= 6:
                verdict = "**No effect beyond fine-tuning** (negative result)."
            elif k_pos >= 8 and t_pos >= 2:
                verdict = "**The PU term on mask-less data consistently improves localisation beyond fine-tuning.**"
            elif k_pos == 7:
                verdict = "**Suggestive improvement** (7/9, p ≈ 0.09)."
            else:
                verdict = "8/9 pairs positive but mean Δ > 0 on < 2 targets: report as suggestive, not consistent."
            out.append("Rule: " + verdict + "\n")
        else:
            out.append(f"({n} of 9 pairs available; main rule not applied yet.)\n")
        # early stop, on Liver's 3 seeds
        L = per_t.get("Liver")
        if L and len(L["aupro"]) == 3:
            both_small = all(all(abs(x) <= NEG for x in L[k]) for k in ("aupro", "pix_auroc"))
            mixed = all(min(L[k]) < 0 < max(L[k]) for k in ("aupro", "pix_auroc"))
            es = both_small and mixed
            out.append(f"**Early-stop check (Liver, 3 seeds):** Δ AUPRO {['%+.2f' % x for x in L['aupro']]}, "
                       f"Δ pix AUROC {['%+.2f' % x for x in L['pix_auroc']]} → "
                       + ("**EARLY STOP**: run `./run_all.sh rest_early_stop` (seed 0 only on Brain and RESC)."
                          if es else "continue: run `./run_all.sh rest` (3 seeds on Brain and RESC).") + "\n")
        # the lambda exception
        if ("Liver", 0) in by["mh"] and ("Liver", 0) in by["mh_ctrl"]:
            if by["mh"][("Liver", 0)]["aupro"] <= by["mh_ctrl"][("Liver", 0)]["aupro"]:
                out.append("M-h ≤ M-h-ctrl on Liver seed 0 (AUPRO): the λ = 0.5 exception applies "
                           "(`./run_all.sh lam05`), Liver seed 0 only; report both.\n")
    # M-h vs R0
    beats = []
    for t in TARGETS:
        rs = [r for (tt, _), r in by["mh"].items() if tt == t]
        if rs and t in r0:
            m = {k: np.mean([r[k] for r in rs]) for k in ("img_auroc", "pix_auroc", "aupro")}
            ok = (m["pix_auroc"] > r0[t]["pix_auroc"] and m["aupro"] > r0[t]["aupro"]
                  and (r0[t]["img_auroc"] - m["img_auroc"]) * PTS <= 0.5)
            beats.append(ok)
            out.append(f"M-h vs R0 on {t}: pix {(m['pix_auroc'] - r0[t]['pix_auroc']) * PTS:+.2f}, "
                       f"AUPRO {(m['aupro'] - r0[t]['aupro']) * PTS:+.2f}, "
                       f"img {(m['img_auroc'] - r0[t]['img_auroc']) * PTS:+.2f} → {'exceeds' if ok else 'does not'}")
    if len(beats) == 3:
        out.append("\nRule: " + ("**Matches or exceeds MVFA's released model** (caveat: R0 was selected on test)."
                                 if sum(beats) >= 2 else "Does not exceed R0 on ≥ 2 of 3 targets.") + "\n")

    # ---- table (b): mask-free
    out.append("\n## (b) Mask-free: M and B0\n")
    out.append("| target | M pix (s0) | B0 pix | M AUPRO (s0) | B0 AUPRO | retention (M−.5)/(R0−.5) | AUPRO M/R0 | M img |")
    out.append("|---|---|---|---|---|---|---|---|")
    ret, m_beats, m_tie = [], 0, 0
    for t in TARGETS:
        m, b = by["m"].get((t, 0)), by["b0"].get((t, 0))
        if not (m and t in r0):
            continue
        rt = (m["pix_auroc"] - 0.5) / (r0[t]["pix_auroc"] - 0.5)
        ret.append(rt)
        bp = f"{b['pix_auroc'] * PTS:.2f}" if b else "–"
        ba = f"{b['aupro'] * PTS:.2f}" if b else "–"
        if b:
            dp, da = (m["pix_auroc"] - b["pix_auroc"]) * PTS, (m["aupro"] - b["aupro"]) * PTS
            m_beats += dp > 0 and da > 0
            m_tie += abs(dp) <= NEG and abs(da) <= NEG
        out.append(f"| {t} | {m['pix_auroc'] * PTS:.2f} | {bp} | {m['aupro'] * PTS:.2f} | {ba} | {rt:.3f} | "
                   f"{m['aupro'] / r0[t]['aupro']:.3f} | {m['img_auroc'] * PTS:.2f} |")
    ms = [by["m"][("Liver", s)] for s in (0, 1, 2) if ("Liver", s) in by["m"]]
    if len(ms) > 1:
        out.append(f"\nLiver M over {len(ms)} seeds: pix {fmt([r['pix_auroc'] for r in ms])}, "
                   f"AUPRO {fmt([r['aupro'] for r in ms])}. Other targets single-seed.")
    if len(ret) == 3:
        mr = float(np.mean(ret))
        if mr >= 0.8 and m_beats >= 2:
            v = "**Without any masks, PU learning keeps most of MVFA's localisation and beats plain MIL.**"
        elif mr >= 0.8 and m_tie >= 2:
            v = "**Image labels alone go a long way, but the PU loss adds little over MIL.**"
        elif mr >= 0.8:
            v = "Retention ≥ 0.8, but M vs B0 is mixed: report per target."
        else:
            v = "**Masks still matter.** Report retention per target and the AUPRO ratio."
        out.append(f"\nMean retention = {mr:.3f}; M > B0 (pix and AUPRO) on {m_beats}/3. Rule: {v}")
    under = [r["run_id"] for c in ("m", "b0") for r in by[c].values() if "under-trained" in r.get("notes", "")]
    if under:
        out.append(f"\nPossibly under-trained (loss still falling): {', '.join(under)}")

    # ---- secondary / exception runs (tagged): reported separately, never in the rules
    tagged = [r for r in rows if r["config"] in ("mh", "mh_ctrl", "m", "b0")
              and r["run_id"] != f"{r['config']}_{r['target']}_s{r['seed']}"]
    if tagged:
        out.append("\n## Secondary runs (not used by any Step 5 rule; DECISIONS D1-D3, plan exceptions)\n")
        out.append("| run | pix AUROC | AUPRO | img AUROC | vs primary (pix / AUPRO) | notes |")
        out.append("|---|---|---|---|---|---|")
        for r in tagged:
            prim = by[r["config"]].get((r["target"], int(r["seed"])))
            vs = (f"{(r['pix_auroc'] - prim['pix_auroc']) * PTS:+.2f} / {(r['aupro'] - prim['aupro']) * PTS:+.2f}"
                  if prim else "–")
            out.append(f"| {r['run_id']} | {r['pix_auroc'] * PTS:.2f} | {r['aupro'] * PTS:.2f} | "
                       f"{r['img_auroc'] * PTS:.2f} | {vs} | {r.get('notes', '')} |")

    text = "\n".join(out)
    (RESULTS_DIR / "analysis.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
