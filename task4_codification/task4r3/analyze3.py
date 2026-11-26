"""Apply round 3's rules (plan §5, written out in RULES.md; both frozen before any
run, hashes in report/ISSUES_LOG.md) mechanically to results/results.csv.
Writes results/analysis.md and results/per_seed.md. A copy of round 2's
analyze2.py with §5's rules in place of round 2's §6.

    python analyze3.py                 # results/results.csv
    python analyze3.py some.csv outdir # any results file (used by the tests)
"""
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from common2 import R1, RESULTS2

T = ["Brain", "Liver", "Retina_RESC"]                  # masked targets (RESC from the RESC+OCT17 fold)
S = (3, 4, 5)                                           # round 3 seeds for the 9 pairs
S_BRAIN = (3, 4, 5, 6, 7)                               # H2 (mvfa, mil only)
ARMS = ["mvfa", "mil", "pu_m100", "mil_m100", "mil_m10", "mil_m25"]
NAMES = {"mvfa": "MVFA (re-run)", "mil": "MIL", "pu_m100": "PU+100%", "mil_m100": "MIL+100%",
         "mil_m10": "MIL+10%", "mil_m25": "MIL+25%"}
K = ("img_auroc", "pix_auroc", "aupro")
R2_RESULTS = R1.parent / "task4r2" / "results" / "results.csv"


# ----------------------------------------------------------------------------- data
def load(path):
    """d[arm][(dataset, seed)] = {metric: points, best_epoch, epochs, diag_p}. Only full
    (non-smoke) rows; a row whose run_id seed disagrees with its seed column is skipped."""
    d = defaultdict(dict)
    if not Path(path).exists():
        return d
    for r in csv.DictReader(open(path)):
        if "SMOKE" in r.get("notes", "") or r["run_id"].split("_s")[-1].split("_")[0] != r["seed"]:
            continue
        v = {k: float(r[k]) * 100 for k in K if r[k] not in ("", None)}
        v.update(best_epoch=int(r["best_epoch"]), epochs=int(r["epochs"]), diag_p=float(r["diag_inmask_p"]))
        d[r["arm"]][(r["eval_dataset"], int(r["seed"]))] = v
    return d


def vals(d, arm, t, k, ss=S):
    return [d[arm][(t, s)][k] for s in ss if (t, s) in d.get(arm, {}) and k in d[arm][(t, s)]]


def mean(d, arm, t, k, ss=S):
    v = vals(d, arm, t, k, ss)
    return float(np.mean(v)) if v else float("nan")


def fmt(d, arm, t, k, ss=S):
    v = vals(d, arm, t, k, ss)
    if not v:
        return "–"
    return f"{np.mean(v):.2f}" if len(v) == 1 else f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}"


def sign_p(pos, n):
    return binomtest(pos, n, 0.5, alternative="greater").pvalue if n else float("nan")


def holm(ps):
    """Holm step-down adjusted p-values; ps: {name: raw p} (nan = not tested)."""
    items = sorted(((p, k) for k, p in ps.items() if not math.isnan(p)))
    m, out, run = len(items), {k: float("nan") for k in ps}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


# ----------------------------------------------------------------------------- rules
def beats(d, arm, ss=S):
    """H3's rule ingredients (round 2's rule): paired AUPRO, pixel AUROC per target, image AUROC per target."""
    pairs = [(t, s) for t in T for s in ss if (t, s) in d.get(arm, {}) and (t, s) in d.get("mvfa", {})
             and "aupro" in d[arm][(t, s)] and "aupro" in d["mvfa"][(t, s)]]
    dau = [d[arm][p]["aupro"] - d["mvfa"][p]["aupro"] for p in pairs]
    pos = sum(x > 0 for x in dau)
    dpix = {t: mean(d, arm, t, "pix_auroc", ss) - mean(d, "mvfa", t, "pix_auroc", ss) for t in T}
    dimg = {t: mean(d, arm, t, "img_auroc", ss) - mean(d, "mvfa", t, "img_auroc", ss) for t in T}
    pix_hi = sum(v > 0 for v in dpix.values())
    img_ok = all(v >= -0.5 for v in dimg.values())
    return dict(n=len(pairs), pos=pos, p=sign_p(pos, len(pairs)), pix_hi=pix_hi, img_ok=img_ok, dau=dau,
                pairs=pairs, dpix=dpix, dimg=dimg)


def beats_outcome(b, p_adj):
    """Confirmed: AUPRO higher in >= 8/9 pairs (scaled: ceil(8n/9)) with Holm-adjusted p <= 0.05,
    pixel AUROC higher on >= 2/3 targets, image AUROC not > 0.5 lower on any target.
    Else >= 7/9 (ceil(7n/9)) -> suggestive again; else did not replicate."""
    n = b["n"]
    if not n:
        return "not run"
    if b["pos"] >= math.ceil(8 * n / 9) and p_adj <= 0.05 and b["pix_hi"] >= 2 and b["img_ok"]:
        return "CONFIRMED"
    if b["pos"] >= math.ceil(7 * n / 9):
        return "suggestive again"
    return "did not replicate"


def h1(d):
    ret = {t: (mean(d, "mil", t, "pix_auroc") - 50) / (mean(d, "mvfa", t, "pix_auroc") - 50) for t in T}
    m = float(np.mean(list(ret.values())))
    if all(v >= 0.90 for v in ret.values()):
        out = "NEARLY MATCHES (retention >= 0.90 on all 3 targets)"
    elif m >= 0.75:
        out = "KEEPS MOST (mean retention >= 0.75)"
    else:
        out = "FALLS SHORT"
    low = [t for t, v in ret.items() if v < 0.60]
    return ret, m, out, low


def h2(d):
    ss = [s for s in S_BRAIN if ("Brain", s) in d.get("mil", {}) and ("Brain", s) in d.get("mvfa", {})]
    dau = {s: d["mil"][("Brain", s)]["aupro"] - d["mvfa"][("Brain", s)]["aupro"] for s in ss}
    pos, n = sum(v > 0 for v in dau.values()), len(ss)
    dpix = mean(d, "mil", "Brain", "pix_auroc", ss) - mean(d, "mvfa", "Brain", "pix_auroc", ss)
    return dict(n=n, pos=pos, p=sign_p(pos, n), dau=dau, dpix=dpix, seeds=ss)


def h2_outcome(r, p_adj):
    n, pos = r["n"], r["pos"]
    if not n:
        return "not run"
    if n == 5 and pos == 5 and r["dpix"] > 0 and p_adj <= 0.05:
        return "CONFIRMED"
    if pos / n >= 0.8:
        extra = f" ({pos}/{n} seeds, p = {r['p']:.3f} uncorrected)" if pos == n else ""
        return "holds on most seeds (not confirmed)" + extra
    return "did not replicate"


def matches(b):
    return all(v >= -1 for v in b["dpix"].values())


# ----------------------------------------------------------------------------- report
def main(path=None, outdir=None):
    path = Path(path) if path else RESULTS2 / "results.csv"
    outdir = Path(outdir) if outdir else RESULTS2
    d, d2 = load(path), load(R2_RESULTS)
    L = ["# Round 3 analysis (plan §5 rules, applied mechanically by `analyze3.py`)\n",
         f"Source: `{path.name}`. Points = AUROC/AUPRO × 100. Mean ± std over seeds 3–5 unless marked. "
         "A pair is (target, seed). Sign tests are one-sided; Holm within each tier.\n"]
    L.append(f"Runs present: {len({(a, t, s) for a in d for (t, s) in d[a] if t in T})} (arm, masked target, seed) "
             f"cells; arms: {', '.join(NAMES.get(a, a) for a in ARMS if a in d) or 'none'}.\n")

    # ---- main table
    L += ["## Main table (masked targets, seeds 3–5)\n",
          "| target | arm | n | img AUROC | pix AUROC | AUPRO | best epochs |", "|---|---|---|---|---|---|---|"]
    for t in T:
        for arm in ARMS:
            ss = [s for s in S if (t, s) in d.get(arm, {})]
            if ss:
                be = ",".join(str(d[arm][(t, s)]["best_epoch"]) for s in ss)
                L.append(f"| {t} | {NAMES[arm]} | {len(ss)} | " + " | ".join(fmt(d, arm, t, k) for k in K) + f" | {be} |")
    L += ["", "## OCT17 image AUROC (RESC+OCT17 fold, seeds 3–5)\n", "| arm | img AUROC |", "|---|---|"]
    for arm in ARMS:
        if vals(d, arm, "Retina_OCT2017", "img_auroc"):
            L.append(f"| {NAMES[arm]} | {fmt(d, arm, 'Retina_OCT2017', 'img_auroc')} |")

    # ---- Tier 1
    L += ["", "## Tier 1: confirm rounds 1–2\n"]
    ret, mret, o1, low = h1(d)
    L += ["### H1 (P1): MIL without masks keeps most of MVFA's localisation (estimate, no test)\n",
          "- Retention (pix − 50) / (MVFA − 50): " + ", ".join(f"{t} {v:.3f}" for t, v in ret.items()) +
          f"; mean {mret:.3f}",
          f"- **H1: {o1}**" + (f"; below 0.60: {', '.join(low)}" if low else "")]
    r2_, b3 = h2(d), beats(d, "pu_m100")
    adj1 = holm({"H2": r2_["p"], "H3": b3["p"]})
    o2, o3 = h2_outcome(r2_, adj1["H2"]), beats_outcome(b3, adj1["H3"])
    L += ["", "### H2 (P2): MIL beats MVFA on Brain\n",
          f"- Brain seeds {list(r2_['seeds'])}: MIL − MVFA AUPRO per seed: " +
          ", ".join(f"s{s} {v:+.2f}" for s, v in r2_["dau"].items()),
          f"- AUPRO higher on {r2_['pos']}/{r2_['n']} seeds; sign test p = {r2_['p']:.3f} raw, "
          f"{adj1['H2']:.3f} Holm (with H3)",
          f"- Mean Brain pixel AUROC Δ (MIL − MVFA, same seeds): {r2_['dpix']:+.2f}",
          f"- **H2: {o2}**"]
    L += ["", "### H3 (P3): PU+100% beats MVFA\n"] + _beats_lines(b3, adj1["H3"], o3)

    # ---- Tier 2
    b4, b5 = beats(d, "mil_m100"), beats(d, "mil_m10")
    adj2 = holm({"H4": b4["p"], "H5": b5["p"]})
    o4 = beats_outcome(b4, adj2["H4"])
    o4 = "SUPPORTED (beats MVFA)" if o4 == "CONFIRMED" else ("not run" if o4 == "not run" else
                                                           ("MATCHES" if matches(b4) else "TRAILS"))
    o5b = beats_outcome(b5, adj2["H5"])
    o5 = "not run" if o5b == "not run" else ("BEATS MVFA" if o5b == "CONFIRMED" and matches(b5) else
                                             ("SUPPORTED (matches MVFA)" if matches(b5) else "TRAILS"))
    L += ["", "## Tier 2: extend\n", "### H4: MIL+100% beats MVFA\n"] + _beats_lines(b4, adj2["H4"], o4)
    L += ["", "### H5: MIL with 10 % of the masks matches MVFA\n",
          "- Rule: mean pixel Δ (MIL+10% − MVFA) ≥ −1 on every target; \"beats\" if it also meets H3's rule."
          ] + _beats_lines(b5, adj2["H5"], o5)
    L += ["", "### p-values\n", "| hypothesis | tier | raw p | Holm-adjusted p | outcome |", "|---|---|---|---|---|",
          f"| H2 | 1 | {r2_['p']:.4f} | {adj1['H2']:.4f} | {o2} |",
          f"| H3 | 1 | {b3['p']:.4f} | {adj1['H3']:.4f} | {o3} |",
          f"| H4 | 2 | {b4['p']:.4f} | {adj2['H4']:.4f} | {o4} |",
          f"| H5 (beats part) | 2 | {b5['p']:.4f} | {adj2['H5']:.4f} | {o5} |",
          f"| H1 | 1 | – | – | {o1} |"]

    # ---- descriptive
    L += ["", "## Descriptive (no test)\n", "### Label-budget curve: retention of MVFA's pixel AUROC\n",
          "| masks | arm | " + " | ".join(T) + " | mean |", "|---|---|---|---|---|---|"]
    for lab, arm in (("0 %", "mil"), ("10 %", "mil_m10"), ("25 %", "mil_m25"), ("100 %", "mil_m100"),
                     ("100 % (MVFA)", "mvfa")):
        rv = [(mean(d, arm, t, "pix_auroc") - 50) / (mean(d, "mvfa", t, "pix_auroc") - 50) for t in T]
        if not all(math.isnan(x) for x in rv):
            L.append(f"| {lab} | {NAMES[arm]} | " + " | ".join(f"{x:.3f}" for x in rv) + f" | {np.nanmean(rv):.3f} |")
    L += ["", "AUPRO along the same curve (mean over seeds):", "",
          "| masks | " + " | ".join(T) + " |", "|---|---|---|---|"]
    for lab, arm in (("0 %", "mil"), ("10 %", "mil_m10"), ("25 %", "mil_m25"), ("100 % +MIL", "mil_m100"),
                     ("MVFA", "mvfa")):
        if any(vals(d, arm, t, "aupro") for t in T):
            L.append(f"| {lab} | " + " | ".join(fmt(d, arm, t, "aupro") for t in T) + " |")

    L += ["", "### Round 3 (20 epochs, seeds 3–5) vs round 2 (5 epochs, seeds 0–2); never pooled\n",
          "| arm | target | R2 pix | R3 pix | R2 AUPRO | R3 AUPRO | R2 img | R3 img |", "|---|---|---|---|---|---|---|---|"]
    for arm in ("mvfa", "mil", "pu_m100"):
        for t in T:
            if vals(d, arm, t, "pix_auroc") or vals(d2, arm, t, "pix_auroc", (0, 1, 2)):
                L.append(f"| {NAMES[arm]} | {t} | {fmt(d2, arm, t, 'pix_auroc', (0, 1, 2))} | {fmt(d, arm, t, 'pix_auroc')} | "
                         f"{fmt(d2, arm, t, 'aupro', (0, 1, 2))} | {fmt(d, arm, t, 'aupro')} | "
                         f"{fmt(d2, arm, t, 'img_auroc', (0, 1, 2))} | {fmt(d, arm, t, 'img_auroc')} |")

    L += ["", "### Best-epoch histogram (one count per run; epochs are 0-based)\n"]
    runs = {}
    for arm in d:
        for (t, s), v in d[arm].items():
            runs[(arm, "RESC_OCT" if t.startswith("Retina") else t, s)] = (v["best_epoch"], v["epochs"])
    if runs:
        last = sum(be == ep - 1 for be, ep in runs.values())
        last2 = sum(be >= ep - 2 for be, ep in runs.values())
        hist = Counter(be for be, _ in runs.values())
        L += ["| best epoch | " + " | ".join(str(k) for k in sorted(hist)) + " |",
              "|---|" + "---|" * len(hist), "| runs | " + " | ".join(str(hist[k]) for k in sorted(hist)) + " |", "",
              f"- {last}/{len(runs)} runs picked their last epoch; {last2}/{len(runs)} one of the last two.",
              "- **" + ("Most runs still pick their last epoch: the runs are still under-trained at this schedule."
                        if last > len(runs) / 2 else "Most runs do not pick their last epoch.") + "**"]
        per_arm = defaultdict(list)
        for (arm, _, _), (be, ep) in runs.items():
            per_arm[arm].append(be == ep - 1)
        L.append("- Last-epoch share per arm: " + ", ".join(f"{NAMES.get(a, a)} {sum(v)}/{len(v)}"
                                                          for a, v in per_arm.items()))

    L += ["", "### In-mask diagnostic (round 2's): share of top-k patches inside the true mask\n",
          "(masked sources' valid/ abnormal images, selected checkpoint; by p)\n", "| arm | by p |", "|---|---|"]
    for arm in ARMS:
        v = [x["diag_p"] for x in d.get(arm, {}).values()]
        if v:
            L.append(f"| {NAMES[arm]} | {np.nanmean(v):.3f} |")

    flags = [(arm, t, s, v["img_auroc"]) for arm in d for (t, s), v in d[arm].items()
             if "img_auroc" in v and v["img_auroc"] < 50]
    L += ["", "### Image AUROC below 50 (flagged)\n"]
    L += [f"- {NAMES.get(a, a)}, {t}, seed {s}: {x:.2f}" for a, t, s, x in sorted(flags)] or ["- none"]

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.md").write_text("\n".join(L) + "\n")
    per_seed(d, outdir)
    print("\n".join(L))
    return dict(H1=o1, H2=o2, H3=o3, H4=o4, H5=o5, holm1=adj1, holm2=adj2)


def _beats_lines(b, p_adj, outcome):
    if not b["n"]:
        return ["- not run"]
    return [f"- AUPRO higher than MVFA in {b['pos']}/{b['n']} pairs; sign test p = {b['p']:.3f} raw, "
            f"{p_adj:.3f} Holm-adjusted",
            "- Paired AUPRO Δ: " + " ".join(f"{t[:5]}-s{s} {x:+.2f}" for (t, s), x in zip(b["pairs"], b["dau"])),
            "- Mean pixel AUROC Δ: " + ", ".join(f"{t} {v:+.2f}" for t, v in b["dpix"].items()) +
            f" (higher on {b['pix_hi']}/3)",
            "- Mean image AUROC Δ: " + ", ".join(f"{t} {v:+.2f}" for t, v in b["dimg"].items()) +
            f" (all ≥ −0.5: {'yes' if b['img_ok'] else 'no'})",
            f"- **Outcome: {outcome}**"]


def per_seed(d, outdir):
    L = ["# Round 3 per-seed results (every recorded run)\n", "Points ×100.\n",
         "| arm | eval dataset | seed | img AUROC | pix AUROC | AUPRO | best epoch | in-mask p |",
         "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS + sorted(set(d) - set(ARMS)):
        for (t, s) in sorted(d.get(arm, {}), key=lambda x: (x[0], x[1])):
            v = d[arm][(t, s)]
            L.append(f"| {NAMES.get(arm, arm)} | {t} | {s} | " +
                     " | ".join(f"{v[k]:.2f}" if k in v else "–" for k in K) +
                     f" | {v['best_epoch']} | {v['diag_p']:.3f} |")
    (outdir / "per_seed.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    out = main(*sys.argv[1:3])
    print(json.dumps({k: v for k, v in out.items() if k.startswith("H")}, indent=1))
