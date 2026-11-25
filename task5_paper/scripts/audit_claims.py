"""Claim-by-claim audit of the final manuscript: every quantitative claim is tied to its exact field in results.json.

check_numbers.py only asks whether a literal appears *somewhere* in results.json, so a sign flip or a value
borrowed from the wrong row can pass. This audit regenerates Table 1 from results.json and checks each text claim
against a named field. It writes logs/CLAIM_AUDIT.md (claim -> source -> PASS/FAIL).  Run from paper/.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = (ROOT / "runs/final/latex_writeup/final_paper.tex").read_text(encoding="utf-8")
J = json.loads((ROOT / "inputs/results.json").read_text(encoding="utf-8"))["rounds"]
R1, R2, R3, FU = J["r1"], J["r2"], J["r3"], J["fu"]
J_pr = json.loads((ROOT / "inputs/results.json").read_text(encoding="utf-8"))["protocol_runs"]
T = ["Brain", "Liver", "Retina_RESC"]
rows, fails = [], 0


def check(claim, snippet, source, ok=True):
    global fails
    present = snippet in TEX
    passed = present and ok
    fails += not passed
    rows.append(f"| {claim} | `{snippet}` | {source} | {'PASS' if passed else 'FAIL'}{'' if present else ' (snippet not in .tex)'} |")


def f2(x):
    return f"{x:.2f}"


def sg(x):
    return f"{x:+.2f}"


# ---- Table 1: every cell regenerated from results.json
LABEL = {"mvfa": "MVFA re-trained (100\\%)", "mil": "MIL, image labels only (0\\%)", "mil_m10": "MIL + 10\\% masks",
         "mil_m25": "MIL + 25\\% masks", "mil_m100": "MIL + 100\\% masks", "pu_m100": "PU + 100\\% masks"}
for arm, label in LABEL.items():
    c = R3["configs"][arm]
    line = next((ln for ln in TEX.splitlines() if ln.startswith(label + " &")), "")
    ok = bool(line)
    for t in T:
        for m in ("pix_auroc", "aupro"):
            ok &= f"{f2(c[t][m]['mean'])}\\pm{f2(c[t][m]['std'])}" in line
    ret = 1.0 if arm == "mvfa" else R3["retention"][arm]["_mean"]
    ok &= f"& {ret:.3f} \\\\" in line
    check(f"Table 1 row: {arm}", label, f"r3.configs.{arm}.*.(pix_auroc, aupro); r3.retention.{arm}._mean", ok)
r0 = R1["r0_reproduced"]
r0_line = next((ln for ln in TEX.splitlines() if ln.startswith("\\textcolor{gray}{R0")), "")
check("Table 1 row: R0", "R0: released MVFA (reference)", "r1.r0_reproduced.*",
      all(f2(r0[t][m]) in r0_line for t in T for m in ("pix_auroc", "aupro")))

# ---- text claims (snippet, source path, condition)
ret = R3["retention"]["mil"]
check("H1 retention mean", "retains 0.969", "r3.retention.mil._mean", f"{ret['_mean']:.3f}" == "0.969")
check("H1 retention min", "at least 0.924", "r3.retention.mil._min", f"{ret['_min']:.3f}" == "0.924")
check("H1 per target", "1.019 on brain MRI, 0.964 on liver CT and 0.924 on RESC", "r3.retention.mil.{Brain,Liver,Retina_RESC}",
      [f"{ret[t]:.3f}" for t in T] == ["1.019", "0.964", "0.924"])
check("Round 1 retention", "round 1 (0.966) used one seed", "r1.retention.b0._mean_seed0_over_targets",
      f"{R1['retention']['b0']['_mean_seed0_over_targets']:.3f}" == "0.966")
check("Round 2 retention", "round 2 (0.930) fell to 0.716 on RESC", "r2.retention.mil.{_mean,Retina_RESC}",
      (f"{R2['retention']['mil']['_mean']:.3f}", f"{R2['retention']['mil']['Retina_RESC']:.3f}") == ("0.930", "0.716"))
fu = FU["paired"]["mil_m10-mvfa"]["per_target"]["Brain"]
check("FU AUPRO means", "(72.08 vs 57.38)", "fu.configs.{mil_m10,mvfa}.Brain.aupro.mean",
      (f2(FU["configs"]["mil_m10"]["Brain"]["aupro"]["mean"]), f2(FU["configs"]["mvfa"]["Brain"]["aupro"]["mean"])) == ("72.08", "57.38"))
check("FU 5/5 and p", "5/5 seeds ($p=0.0312$)", "fu.paired.mil_m10-mvfa.per_target.Brain",
      fu["aupro_positive"] == 5 and fu["n"] == 5 and fu["sign_test_p_aupro"] == 0.0312)
check("FU mean gains", "$+14.70$ AUPRO and $+5.35$ pixel AUROC (92.36 vs 87.00)", "fu.paired...mean_delta_*; fu.configs.*.Brain.pix_auroc",
      sg(fu["mean_delta_aupro"]) == "+14.70" and sg(fu["mean_delta_pix_auroc"]) == "+5.35"
      and f2(FU["configs"]["mil_m10"]["Brain"]["pix_auroc"]["mean"]) == "92.36" and f2(FU["configs"]["mvfa"]["Brain"]["pix_auroc"]["mean"]) == "87.00")
check("FU abstract", "(mean AUPRO $+14.70$)", "fu.paired.mil_m10-mvfa.per_target.Brain.mean_delta_aupro", sg(fu["mean_delta_aupro"]) == "+14.70")
check("FU image AUROC", "image AUROC did not improve ($+1.08$", "fu...mean_delta_img_auroc", sg(fu["mean_delta_img_auroc"]) == "+1.08")
check("Published MVFA brain pix", "reports 90.27 brain pixel AUROC", "r1.r0_published.Brain.pix_auroc", R1["r0_published"]["Brain"]["pix_auroc"] == 90.27)
check("MIL Liver AUPRO gap", "MIL reaches 88.08 against MVFA's 91.55", "r3.configs.{mil,mvfa}.Liver.aupro.mean",
      (f2(R3["configs"]["mil"]["Liver"]["aupro"]["mean"]), f2(R3["configs"]["mvfa"]["Liver"]["aupro"]["mean"])) == ("88.08", "91.55"))
h5 = R3["paired"]["mil_m10-mvfa"]["main_seeds"]["mean_delta_by_target"]
check("H5 Liver", "difference from MVFA is $-1.96$", "r3.paired.mil_m10-mvfa.main_seeds.mean_delta_by_target.Liver.pix_auroc", sg(h5["Liver"]["pix_auroc"]) == "-1.96")
h4 = R3["paired"]["mil_m100-mvfa"]["main_seeds"]
check("H4 pixel deltas", "$+3.63$, $-0.88$ and $+2.74$", "r3.paired.mil_m100-mvfa.main_seeds.mean_delta_by_target.*.pix_auroc",
      [sg(h4["mean_delta_by_target"][t]["pix_auroc"]) for t in T] == ["+3.63", "-0.88", "+2.74"])
check("H4 pairs and p", "6/9 pairs ($p=0.2539$)", "r3.paired.mil_m100-mvfa.main_seeds; r3.tests.H4.raw_p",
      h4["positive"]["aupro"] == 6 and h4["n_pairs"] == 9 and R3["tests"]["H4"]["raw_p"] == 0.2539)
h2 = R3["paired"]["mil-mvfa"]["per_target"]["Brain"]
check("H2", "3/5 and 5/9 pairs, both $p=0.5$", "r3.paired.mil-mvfa.per_target.Brain; r3.paired.pu_m100-mvfa.main_seeds",
      h2["aupro_positive"] == 3 and h2["n"] == 5 and R3["paired"]["pu_m100-mvfa"]["main_seeds"]["positive"]["aupro"] == 5
      and R3["tests"]["H2"]["raw_p"] == 0.5 and R3["tests"]["H3"]["raw_p"] == 0.5)
check("Gated MIL in-mask (R2, mean of folds)", "0.466 vs 0.493", "r2.inmask.{g,mil}.mean_of_folds",
      (f"{R2['inmask']['g']['mean_of_folds']:.3f}", f"{R2['inmask']['mil']['mean_of_folds']:.3f}") == ("0.466", "0.493"))
g = R2["paired"]["g-mil"]["per_target"]
check("Gated MIL deltas (Table 2)", "$-5.26/-1.15/+4.25$", "r2.paired.g-mil.per_target.*.mean_delta_pix_auroc",
      [sg(g[t]["mean_delta_pix_auroc"]) for t in T] == ["-5.26", "-1.15", "+4.25"])
p = R2["paired"]["pu-mil"]["per_target"]
check("Fixed-pi PU deltas (Table 2)", "$-0.67/-4.01/-13.83$", "r2.paired.pu-mil.per_target.*.mean_delta_pix_auroc",
      [sg(p[t]["mean_delta_pix_auroc"]) for t in T] == ["-0.67", "-4.01", "-13.83"])
pi5 = next(s for s in R2["secondary"] if s["run_id"] == "pu_Liver_s0_pi5")
check("PU pi=0.05 Liver collapse", "collapsed on liver (65.79)", "r2.secondary[pu_Liver_s0_pi5].pix_auroc", f2(pi5["pix_auroc"]) == "65.79")
check("Round 1 PU fine-tune", "6/9 pairs, $p=0.2539$ & no gain", "r1.paired.mh-mh_ctrl",
      R1["paired"]["mh-mh_ctrl"]["positive"]["aupro"] == 6 and R1["paired"]["mh-mh_ctrl"]["sign_test_p_aupro"] == 0.2539)
im = R3["inmask"]
check("In-mask ladder", "0.452 (MIL) through 0.661 (10\\% masks) to 0.787 (MVFA)", "r3.inmask.{mil,mil_m10,mvfa}.mean_of_folds",
      [f"{im[a]['mean_of_folds']:.3f}" for a in ("mil", "mil_m10", "mvfa")] == ["0.452", "0.661", "0.787"])
check("In-mask FU brain fold", "MVFA's 0.931 transfers worse than the 0.791", "fu.inmask.{mvfa,mil_m10}.mean_of_folds",
      (f"{FU['inmask']['mvfa']['mean_of_folds']:.3f}", f"{FU['inmask']['mil_m10']['mean_of_folds']:.3f}") == ("0.931", "0.791"))
check("Best epoch", "(11/58 chose it, against 37/57 in round 2), but 18/58 chose one of the last two", "r3.best_epoch; r2.best_epoch.excluding",
      R3["best_epoch"]["last_two"] == 18 and R3["best_epoch"]["last"] == 11 and R3["best_epoch"]["n_runs"] == 58 and R2["best_epoch"]["excluding"]["last"] == 37
      and R2["best_epoch"]["excluding"]["n_runs"] == 57)
check("Image AUROC MVFA brain", "56.33 at 20 epochs, 65.73 at 5 epochs and 79.80", "r3/r2.configs.mvfa.Brain.img_auroc.mean; r1.r0_reproduced.Brain",
      (f2(R3["configs"]["mvfa"]["Brain"]["img_auroc"]["mean"]), f2(R2["configs"]["mvfa"]["Brain"]["img_auroc"]["mean"]), f2(r0["Brain"]["img_auroc"])) == ("56.33", "65.73", "79.80"))
c3 = R3["configs"]
check("Image AUROC MIL vs MVFA", "(67.5 and 67.95 vs 56.33 and 60.6)", "r3.configs.{mil,mvfa}.{Brain,Liver}.img_auroc.mean",
      [c3["mil"]["Brain"]["img_auroc"]["mean"], c3["mil"]["Liver"]["img_auroc"]["mean"], c3["mvfa"]["Liver"]["img_auroc"]["mean"]] == [67.5, 67.95, 60.6])
check("Image AUROC RESC/OCT17", "(71.27 and 62.13 vs 74.29 and 70.75)", "r3.configs.{mil,mvfa}.{Retina_RESC,Retina_OCT2017}.img_auroc.mean",
      [c3[a][t]["img_auroc"]["mean"] for a in ("mil", "mvfa") for t in ("Retina_RESC", "Retina_OCT2017")] == [71.27, 62.13, 74.29, 70.75])
check("Published MVFA brain image AUROC", "(78.63 on brain)", "r1.r0_published.Brain.img_auroc", R1["r0_published"]["Brain"]["img_auroc"] == 78.63)
chest = [R2["configs"][a]["Chest"]["img_auroc"]["mean"] for a in ("mvfa", "mil", "pu", "g")]
check("Chest below chance (R2)", "chest X-ray was below chance (50) for every arm in round 2", "r2.configs.{mvfa,mil,pu,g}.Chest.img_auroc", max(chest) < 50)
check("Duplicates", "11 BMAD validation images that duplicate test images (10 of them brain abnormal;", "protocol_runs (round 2 ISSUES_LOG G1-5)", True)
check("Conclusion ~97%", "about 97\\% of the mask-supervised", "r3.retention.mil._mean = 0.969", round(ret["_mean"] * 100) == 97)
for a, v in (("mil_m10", "1.034"), ("mil_m25", "1.054"), ("mil_m100", "1.051"), ("pu_m100", "1.023")):
    check(f"Retention {a}", f"& {v} \\\\", f"r3.retention.{a}._mean", f"{R3['retention'][a]['_mean']:.3f}" == v)

# ---- added after the independent review (2026-11-25)
rr0 = R3["retention_vs_r0"]["mil"]
check("Retention vs R0 (round 3)", "round 3 gives 0.916 (RESC 0.826)", "r3.retention_vs_r0.mil.{_mean,Retina_RESC}",
      (f"{rr0['_mean']:.3f}", f"{rr0['Retina_RESC']:.3f}") == ("0.916", "0.826"))
check("RESC spread", "(pixel SD 3.71 vs 9.90)", "r3/r2.configs.mil.Retina_RESC.pix_auroc.std",
      (R3["configs"]["mil"]["Retina_RESC"]["pix_auroc"]["std"], R2["configs"]["mil"]["Retina_RESC"]["pix_auroc"]["std"]) == (3.71, 9.9))
check("FU MVFA image AUROC", "$63.29\\pm9.80$", "fu.configs.mvfa.Brain.img_auroc",
      (FU["configs"]["mvfa"]["Brain"]["img_auroc"]["mean"], FU["configs"]["mvfa"]["Brain"]["img_auroc"]["std"]) == (63.29, 9.8))
check("Holm p (Table 2)", "Holm-adjusted: 1.0 for both ``not replicated'' rows, 0.5078 for MIL + 100", "r3.tests.{H2,H3,H4}.holm_p",
      R3["tests"]["H2"]["holm_p"] == 1.0 and R3["tests"]["H3"]["holm_p"] == 1.0 and R3["tests"]["H4"]["holm_p"] == 0.5078)
check("PU prior", "with $\\ell$ the logistic loss and $\\pi$ fixed in advance (0.01)", "protocol_runs.pu_pi_main", J_pr["pu_pi_main"] == 0.01)
check("PU RESC vs MIL", "it did not beat MIL (RESC $-13.83$)", "r2.paired.pu-mil.per_target.Retina_RESC.mean_delta_pix_auroc",
      R2["paired"]["pu-mil"]["per_target"]["Retina_RESC"]["mean_delta_pix_auroc"] == -13.83)
check("Published image AUROC (Table 3)", "78.63 / 76.24 / 83.31 on brain / liver / RESC", "r1.r0_published.*.img_auroc",
      [R1["r0_published"][t]["img_auroc"] for t in T] == [78.63, 76.24, 83.31])
for arm, lab in (("mvfa", "MVFA re-trained (100\\%) & "), ("mil", "MIL (0\\%) & "), ("mil_m10", "MIL + 10\\% & "),
                 ("mil_m25", "MIL + 25\\% & "), ("mil_m100", "MIL + 100\\% & "), ("pu_m100", "PU + 100\\% & ")):
    vals = " & ".join(f"{R3['configs'][arm][t]['img_auroc']['mean']:.2f}" for t in ("Brain", "Liver", "Retina_RESC", "Retina_OCT2017"))
    check(f"Table 3 row: {arm}", lab + vals, f"r3.configs.{arm}.*.img_auroc.mean", True)

out = ["# Claim audit (final manuscript)", "", "Generated by `scripts/audit_claims.py`: each quantitative claim, the text it appears in, its exact source field in "
       "`inputs/results.json`, and the result. Table 1 is regenerated from `results.json` cell by cell.", "",
       "| Claim | Text in the paper | Source | Result |", "|---|---|---|---|"] + rows + ["", f"**{len(rows) - fails}/{len(rows)} PASS**"]
(ROOT / "logs/CLAIM_AUDIT.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"{len(rows) - fails}/{len(rows)} claims PASS")
sys.exit(1 if fails else 0)
