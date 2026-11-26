"""Reconcile paper/inputs/results.json with the Task 4 reports of rounds 2, 3 and the follow-up.

Round 1 is checked by verify_results.py (unchanged). This script compares every cell of:
  * round 2: the main table in task4_r2/results/analysis.md (mean ± std per arm and target);
  * round 3: the frozen paper tables in task4_r3/report/paper_inputs/task4_r3/ (table_main, table_budget,
    table_inmask, table_oct17, table_hypotheses, table_paired_deltas);
  * follow-up: the per-seed table and FU-H line of task4_fu/results/analysis_fu.md.
Tolerance 0.011 for points (2-decimal rounding), 0.0011 for ratios and p-values.  Run from paper/.
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "inputs"
R = json.loads((IN / "results.json").read_text(encoding="utf-8"))["rounds"]
errors, n = [], 0
TGT = {"Brain": "Brain", "Liver": "Liver", "RESC": "Retina_RESC", "Retina_RESC": "Retina_RESC"}
ARM = {"MVFA (re-run)": "mvfa", "MIL": "mil", "PU+100%": "pu_m100", "MIL+100%": "mil_m100", "MIL+10%": "mil_m10",
       "MIL+25%": "mil_m25", "PU (π=1%)": "pu", "G (gated MIL)": "g", "G+100%": "g_m100"}


def eq(label, actual, expected, tol=0.011):
    global n
    n += 1
    if actual is None or expected is None or abs(float(actual) - float(expected)) > tol:
        errors.append(f"{label}: results.json={actual} report={expected}")


def ms(text):
    """'87.84 ± 0.87' -> (87.84, 0.87); '81.18' -> (81.18, None)."""
    m = re.match(r"\s*([+-]?\d+(?:\.\d+)?)\s*(?:±\s*(\d+(?:\.\d+)?))?", text.replace("−", "-"))
    return (float(m.group(1)), float(m.group(2)) if m.group(2) else None) if m else (None, None)


def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- round 2: analysis.md main table
r2 = R["r2"]
for line in (IN / "task4_r2/results/analysis.md").read_text(encoding="utf-8").splitlines():
    c = [x.strip() for x in line.strip("|").split("|")]
    if len(c) == 7 and c[1] in ARM and c[0] in TGT:
        box = r2["configs"][ARM[c[1]]][TGT[c[0]]]
        for col, m in zip(c[3:6], ("img_auroc", "pix_auroc", "aupro")):
            v, s = ms(col)
            eq(f"r2/{c[1]}/{c[0]}/{m}", box[m]["mean"], v)
            if s is not None:
                eq(f"r2/{c[1]}/{c[0]}/{m}/std", box[m]["std"], s)
for arm, vals in {"mil": (1.088, 0.987, 0.716, 0.930), "pu": (1.071, 0.901, 0.351, 0.774),
                  "g": (0.951, 0.962, 0.828, 0.914)}.items():       # analysis.md "Retention" table
    for t, v in zip(("Brain", "Liver", "Retina_RESC", "_mean"), vals):
        eq(f"r2/retention/{arm}/{t}", r2["retention"][arm][t], v, 0.0011)
be = r2["best_epoch"]["excluding"]
eq("r2/best_epoch/n", be["n_runs"], 57, 0); eq("r2/best_epoch/last", be["last"], 37, 0); eq("r2/best_epoch/last2", be["last_two"], 54, 0)

# ---------------------------------------------------------------- round 3: frozen paper tables
r3, P3 = R["r3"], IN / "task4_r3/report/paper_inputs/task4_r3"
for r in rows(P3 / "table_main.csv"):
    box = r3["configs"][ARM[r["arm"]]][TGT[r["target"]]]
    for col, m in (("img AUROC", "img_auroc"), ("pix AUROC", "pix_auroc"), ("AUPRO", "aupro")):
        v, s = ms(r[col])
        eq(f"r3/{r['arm']}/{r['target']}/{m}", box[m]["mean"], v)
        eq(f"r3/{r['arm']}/{r['target']}/{m}/std", box[m]["std"], s)
    eq(f"r3/{r['arm']}/{r['target']}/best_epochs", 0 if box["best_epoch"] == [int(x) for x in r["best epochs (0-based)"].split(",")] else 1, 0, 0)
for r in rows(P3 / "table_budget.csv"):
    arm = ARM[r["arm"]]
    for t, col in (("Brain", "retention Brain"), ("Liver", "retention Liver"), ("Retina_RESC", "retention RESC"),
                   ("_mean", "retention mean")):
        if arm != "mvfa":
            eq(f"r3/budget/{arm}/{t}", r3["retention"][arm][t], float(r[col]), 0.0011)
    for t, col in (("Brain", "AUPRO Brain"), ("Liver", "AUPRO Liver"), ("Retina_RESC", "AUPRO RESC")):
        eq(f"r3/budget/{arm}/{t}/aupro", r3["configs"][arm][t]["aupro"]["mean"], ms(r[col])[0])
FOLD = {"Brain fold (sources Liver, RESC)": "Brain", "Liver fold (sources Brain, RESC)": "Liver",
        "RESC+OCT17 fold (sources Brain, Liver)": "RESC_OCT"}
for r in rows(P3 / "table_inmask.csv"):
    arm = ARM[r["arm"]]
    for col, fold in FOLD.items():
        eq(f"r3/inmask/{arm}/{fold}", r3["inmask"][arm]["per_fold"][fold], float(r[col]), 0.0011)
    eq(f"r3/inmask/{arm}/mean", r3["inmask"][arm]["mean_of_folds"], float(r["mean of folds"]), 0.0011)
for r in rows(P3 / "table_oct17.csv"):
    v, s = ms(r["OCT17 img AUROC (seeds 3–5)"])
    box = r3["configs"][ARM[r["arm"]]]["Retina_OCT2017"]["img_auroc"]
    eq(f"r3/oct17/{r['arm']}", box["mean"], v); eq(f"r3/oct17/{r['arm']}/std", box["std"], s)
HYP = {"H2": ("mil-mvfa", "Brain"), "H3": ("pu_m100-mvfa", None), "H4": ("mil_m100-mvfa", None), "H5": ("mil_m10-mvfa", None)}
for r in rows(P3 / "table_hypotheses.csv"):
    h = r["#"]
    if h == "H1":
        nums = [float(x) for x in re.findall(r"\d+\.\d+", r["statistic (pix Δ: Brain / Liver / RESC)"])]
        for t, v in zip(("Brain", "Liver", "Retina_RESC", "_mean"), nums):
            eq(f"r3/H1/{t}", r3["retention"]["mil"][t], v, 0.0011)
        continue
    eq(f"r3/{h}/raw_p", r3["tests"][h]["raw_p"], float(r["raw p (one-sided sign test)"]), 0.0011)
    eq(f"r3/{h}/holm_p", r3["tests"][h]["holm_p"], float(r["Holm p (within tier)"]), 0.0011)
    pair, tgt = HYP[h]
    stat = r["statistic (pix Δ: Brain / Liver / RESC)"].replace("−", "-")
    k, nn = map(int, re.search(r"(\d+)/(\d+)", stat).groups())
    pr = r3["paired"][pair]
    got = (pr["per_target"][tgt]["aupro_positive"], pr["per_target"][tgt]["n"]) if tgt else \
          (pr["main_seeds"]["positive"]["aupro"], pr["main_seeds"]["n_pairs"])
    eq(f"r3/{h}/positive", got[0], k, 0); eq(f"r3/{h}/n", got[1], nn, 0)
    deltas = [float(x) for x in re.findall(r"[+-]\d+\.\d+", stat.split("pix Δ")[1])]
    if tgt:
        eq(f"r3/{h}/pix", pr["per_target"][tgt]["mean_delta_pix_auroc"], deltas[0])
    else:
        for t, v in zip(("Brain", "Liver", "Retina_RESC"), deltas):
            eq(f"r3/{h}/pix/{t}", pr["main_seeds"]["mean_delta_by_target"][t]["pix_auroc"], v)
AP = {"PU+100%": "pu_m100-mvfa", "MIL+100%": "mil_m100-mvfa", "MIL+10%": "mil_m10-mvfa", "MIL (H2)": "mil-mvfa"}
for r in rows(P3 / "table_paired_deltas.csv"):
    pp = {(d["target"], d["seed"]): d for d in r3["paired"][AP[r["arm"]]]["per_pair"]}
    eq(f"r3/delta/{r['arm']}/{r['target']}/s{r['seed']}", pp[(TGT[r["target"]], int(r["seed"]))]["aupro"],
       float(r["AUPRO Δ vs MVFA (same seed)"].replace("−", "-")))
eq("r3/best_epoch/last", r3["best_epoch"]["last"], 11, 0); eq("r3/best_epoch/last_two", r3["best_epoch"]["last_two"], 18, 0)
eq("r3/best_epoch/n", r3["best_epoch"]["n_runs"], 58, 0)

# ---------------------------------------------------------------- follow-up: analysis_fu.md
fu = R["fu"]
pp = {d["seed"]: d for d in fu["paired"]["mil_m10-mvfa"]["per_pair"]}
for line in (IN / "task4_fu/results/analysis_fu.md").read_text(encoding="utf-8").splitlines():
    c = [x.strip() for x in line.strip("|").split("|")]
    if len(c) == 10 and c[0].isdigit():
        s = int(c[0])
        for i, m in enumerate(("img_auroc", "pix_auroc", "aupro")):
            eq(f"fu/s{s}/mvfa/{m}", fu["configs"]["mvfa"]["Brain"][m]["values"][fu["configs"]["mvfa"]["Brain"]["seeds"].index(s)], float(c[1 + i]))
            eq(f"fu/s{s}/mil_m10/{m}", fu["configs"]["mil_m10"]["Brain"][m]["values"][fu["configs"]["mil_m10"]["Brain"]["seeds"].index(s)], float(c[4 + i]))
        eq(f"fu/s{s}/dAUPRO", pp[s]["aupro"], float(c[7])); eq(f"fu/s{s}/dpix", pp[s]["pix_auroc"], float(c[8]))
        eq(f"fu/s{s}/dimg", pp[s]["img_auroc"], float(c[9].replace("−", "-")))
t = fu["paired"]["mil_m10-mvfa"]["per_target"]["Brain"]
eq("fu/positive", t["aupro_positive"], 5, 0); eq("fu/p", t["sign_test_p_aupro"], 0.0312, 0.0011)
eq("fu/mean_dpix", t["mean_delta_pix_auroc"], 5.35); eq("fu/mean_daupro", t["mean_delta_aupro"], 14.70)
eq("fu/mean_dimg", t["mean_delta_img_auroc"], 1.08)
for arm, (img, pix, au) in {"mvfa": ((63.29, 9.80), (87.00, 0.72), (57.38, 2.01)),
                            "mil_m10": ((64.37, 5.83), (92.36, 0.98), (72.08, 3.44))}.items():
    b = fu["configs"][arm]["Brain"]
    for m, (v, s) in zip(("img_auroc", "pix_auroc", "aupro"), (img, pix, au)):
        eq(f"fu/{arm}/{m}", b[m]["mean"], v); eq(f"fu/{arm}/{m}/std", b[m]["std"], s)

if errors:
    print("\n".join(errors))
    sys.exit(1)
print(f"PASS: {n} values checked against the round 2, round 3 and follow-up reports.")
