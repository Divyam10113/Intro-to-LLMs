"""Round 3 hand-off for the paper (plan §4 Session B step 4): build report/paper_inputs/task4_r3/.

    python make_paper_inputs3.py

Reads only results/ (and round 2's results.csv for the R2-vs-R3 table) and the figures that
make_figures.py wrote; no GPU, no test set. Every number is computed with analyze3.py's own
functions (imported, not changed), so the tables cannot drift from results/analysis.md; the
hypothesis outcomes are copied from analysis.md's p-value table, the frozen rules' output.
Layout follows round 1's task4/report/paper_inputs/ (PAPER_PLAN.md was not supplied; ISSUES_LOG G3).
"""
import csv
import re
import shutil
from pathlib import Path

import numpy as np

import analyze3 as A

R3 = Path(__file__).resolve().parent
RES = R3 / "results"
REP = R3 / "report"
OUT = REP / "paper_inputs" / "task4_r3"
R2_CSV = R3.parent / "task4r2" / "results" / "results.csv"
TN = {"Brain": "Brain", "Liver": "Liver", "Retina_RESC": "RESC"}


def write(name, header, rows, title):
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    md = [f"# {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    (OUT / f"{name}.md").write_text("\n".join(md) + "\n")


def sd(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = A.load(RES / "results.csv")
    for f in ("results.csv", "analysis.md", "per_seed.md"):
        shutil.copy(RES / f, OUT / f)

    # main table: mean ± std over seeds 3-5
    rows = []
    for t in A.T:
        for a in A.ARMS:
            rows.append([TN[t], A.NAMES[a], len(A.vals(d, a, t, "pix_auroc"))] +
                        [A.fmt(d, a, t, k) for k in A.K] + [",".join(str(d[a][(t, s)]["best_epoch"]) for s in A.S)])
    write("table_main", ["target", "arm", "n seeds", "img AUROC", "pix AUROC", "AUPRO", "best epochs (0-based)"], rows,
          "Round 3 main table (masked targets, seeds 3–5, points)")

    write("table_oct17", ["arm", "OCT17 img AUROC (seeds 3–5)"],
          [[A.NAMES[a], A.fmt(d, a, "Retina_OCT2017", "img_auroc")] for a in A.ARMS],
          "OCT17 image AUROC (RESC+OCT17 fold)")

    # hypotheses: statistics recomputed, outcome labels copied from analysis.md (frozen rules' output)
    txt = (RES / "analysis.md").read_text()
    outc = dict(re.findall(r"^\| (H\d)[^|]*\| \d \| [^|]+\| [^|]+\| ([^|]+) \|$", txt, re.M))
    outc["H1"] = re.search(r"^\| H1 \| 1 \| – \| – \| ([^|]+) \|$", txt, re.M).group(1)
    ret, rm, _, _ = A.h1(d)
    h2 = A.h2(d)
    b = {a: A.beats(d, a) for a in ("pu_m100", "mil_m100", "mil_m10")}
    p1 = A.holm({"H2": h2["p"], "H3": b["pu_m100"]["p"]})
    p2 = A.holm({"H4": b["mil_m100"]["p"], "H5": b["mil_m10"]["p"]})
    pix = lambda bb: " / ".join(f"{bb['dpix'][t]:+.2f}" for t in A.T)
    rows = [
        ["H1", "MIL (no masks) keeps MVFA's localisation", "retention " + " / ".join(f"{ret[t]:.3f}" for t in A.T) +
         f" (mean {rm:.3f})", "–", "–", outc["H1"].strip()],
        ["H2", "MIL beats MVFA on Brain", f"AUPRO higher {h2['pos']}/{h2['n']} seeds; pix Δ {h2['dpix']:+.2f}",
         f"{h2['p']:.4f}", f"{p1['H2']:.4f}", outc["H2"].strip()],
        ["H3", "PU+100% beats MVFA", f"AUPRO higher {b['pu_m100']['pos']}/{b['pu_m100']['n']} pairs; pix Δ {pix(b['pu_m100'])}",
         f"{b['pu_m100']['p']:.4f}", f"{p1['H3']:.4f}", outc["H3"].strip()],
        ["H4", "MIL+100% beats MVFA", f"AUPRO higher {b['mil_m100']['pos']}/{b['mil_m100']['n']} pairs; pix Δ {pix(b['mil_m100'])}",
         f"{b['mil_m100']['p']:.4f}", f"{p2['H4']:.4f}", outc["H4"].strip()],
        ["H5", "MIL+10% matches MVFA", f"AUPRO higher {b['mil_m10']['pos']}/{b['mil_m10']['n']} pairs; pix Δ {pix(b['mil_m10'])}",
         f"{b['mil_m10']['p']:.4f}", f"{p2['H5']:.4f}", outc["H5"].strip()],
    ]
    write("table_hypotheses", ["#", "hypothesis", "statistic (pix Δ: Brain / Liver / RESC)", "raw p (one-sided sign test)",
                               "Holm p (within tier)", "outcome (frozen rules)"], rows,
          "Round 3 hypotheses under the frozen rules (RULES.md, analyze3.py)")

    rows = []
    for a, bb in b.items():
        for (t, s), x in zip(bb["pairs"], bb["dau"]):
            rows.append([A.NAMES[a], TN[t], s, f"{x:+.2f}"])
    for s, x in h2["dau"].items():
        rows.append(["MIL (H2)", "Brain", s, f"{x:+.2f}"])
    write("table_paired_deltas", ["arm", "target", "seed", "AUPRO Δ vs MVFA (same seed)"], rows,
          "Paired AUPRO differences against the re-run MVFA")

    # label-budget curve
    bud = [("0 %", "mil"), ("10 %", "mil_m10"), ("25 %", "mil_m25"), ("100 % + MIL", "mil_m100"), ("100 % (MVFA)", "mvfa")]
    rows = []
    for lab, a in bud:
        r = [(A.mean(d, a, t, "pix_auroc") - 50) / (A.mean(d, "mvfa", t, "pix_auroc") - 50) for t in A.T]
        rows.append([lab, A.NAMES[a]] + [f"{x:.3f}" for x in r] + [f"{np.mean(r):.3f}"] +
                    [A.fmt(d, a, t, "aupro") for t in A.T])
    write("table_budget", ["masks", "arm", "retention Brain", "retention Liver", "retention RESC", "retention mean",
                           "AUPRO Brain", "AUPRO Liver", "AUPRO RESC"], rows,
          "Label-budget curve (descriptive, no test): pixel-AUROC retention vs MVFA and AUPRO")

    # round 2 (5 epochs, seeds 0-2) vs round 3 (20 epochs, seeds 3-5); never pooled
    d2 = A.load(R2_CSV)
    rows = []
    for a in ("mvfa", "mil", "pu_m100"):
        for t in A.T:
            rows.append([A.NAMES[a], TN[t]] + [x for k in ("pix_auroc", "aupro", "img_auroc")
                                               for x in (A.fmt(d2, a, t, k, (0, 1, 2)), A.fmt(d, a, t, k))])
    write("table_r2_vs_r3", ["arm", "target", "R2 pix", "R3 pix", "R2 AUPRO", "R3 AUPRO", "R2 img", "R3 img"], rows,
          "Round 2 (5 epochs, seeds 0–2) vs round 3 (20 epochs, seeds 3–5), never pooled")

    # in-mask diagnostic. Measured per run on the fold's masked SOURCES' valid/ abnormal images, so it is
    # listed per fold (seeds 3-5, each run once). analysis.md's pooled number averages every row of the arm:
    # each RESC+OCT17 run counts twice (its RESC and OCT17 rows carry the same value) and Brain seeds 6-7
    # are included (ISSUES_LOG G3). Both are given; the per-fold values are the ones to cite.
    fold_of = {"Brain": "Brain fold (sources Liver, RESC)", "Liver": "Liver fold (sources Brain, RESC)",
               "Retina_RESC": "RESC+OCT17 fold (sources Brain, Liver)"}
    rows = []
    for a in A.ARMS:
        per = [np.mean([d[a][(t, s)]["diag_p"] for s in A.S]) for t in A.T]
        pooled = np.nanmean([x["diag_p"] for x in d[a].values()])
        rows.append([A.NAMES[a]] + [f"{x:.3f}" for x in per] + [f"{np.mean(per):.3f}", f"{pooled:.3f}"])
    write("table_inmask", ["arm"] + [fold_of[t] for t in A.T] + ["mean of folds", "analysis.md pooled (all rows)"],
          rows, "In-mask diagnostic: share of top-k patches inside the true mask (masked sources' valid/ "
          "abnormal images, selected checkpoint, seeds 3–5)")

    # headline numbers with their source
    num = [["H1 retention mean", f"{rm:.3f}", "analysis.md H1"]]
    num += [[f"H1 retention {TN[t]}", f"{ret[t]:.3f}", "analysis.md H1"] for t in A.T]
    for a in A.ARMS:
        for t in A.T:
            for k in ("pix_auroc", "aupro", "img_auroc"):
                v = A.vals(d, a, t, k)
                num.append([f"{A.NAMES[a]} {TN[t]} {k} mean", f"{np.mean(v):.2f}", "analysis.md main table"])
                num.append([f"{A.NAMES[a]} {TN[t]} {k} std", f"{sd(v):.2f}", "analysis.md main table"])
    write("paper_numbers", ["quantity", "value", "source"], num, "Round 3 numbers for the paper")

    for f in ("liver_heatmaps_r3.png", "liver_heatmaps_r3.pdf", "liver_heatmaps_r3_images.txt",
              "budget_curve_r3.png", "budget_curve_r3.pdf"):
        shutil.copy(REP / f, OUT / f)
    print("wrote", OUT, sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
