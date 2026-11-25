"""PaperOrchestra raw materials for the FINAL paper (Task 4 rounds 1-3 + follow-up).

Extends prepare_raw_materials.py (round 1 only). Every number is read from paper/inputs/results.json; nothing
is typed by hand. Writes paper/inputs/raw_materials/ in the layout the patched PaperOrchestra fork reads
(idea.md, experiment_log.md, related_work_outline.md, citation_pool.json, refs.bib, figures/ + info.json).
The round-1 materials stay in paper/inputs/raw_materials_round1/.

    python -E -P scripts/prepare_raw_materials_final.py        # from paper/
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "inputs"
RAW = DATA / "raw_materials"
FIG = RAW / "figures"
FIG.mkdir(parents=True, exist_ok=True)
J = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
R1, R2, R3, FU = (J["rounds"][k] for k in ("r1", "r2", "r3", "fu"))
PR = J["protocol_runs"]
T = ["Brain", "Liver", "Retina_RESC"]
TN = {"Brain": "Brain MRI", "Liver": "Liver CT", "Retina_RESC": "RESC (retinal OCT)"}
NAME = {"mvfa": "MVFA (re-run, full masks)", "mil": "MIL (no masks)", "mil_m10": "MIL+10 % masks",
        "mil_m25": "MIL+25 % masks", "mil_m100": "MIL+100 % masks", "pu_m100": "PU+100 % masks"}


def ms(s):
    return f"{s['mean']:.2f}" + (f" ± {s['std']:.2f}" if s.get("std") is not None else "")


# ------------------------------------------------------------------ figure 1: label-budget curve (AUPRO)
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.3, 1.9))
xs = [0, 1, 2, 3]
colors = {"Brain": "#1f77b4", "Liver": "#d62728", "Retina_RESC": "#2ca02c"}
for t in T:
    arms = ["mil", "mil_m10", "mil_m25", "mil_m100"]
    m = [R3["configs"][a][t]["aupro"]["mean"] for a in arms]
    s = [R3["configs"][a][t]["aupro"]["std"] for a in arms]
    ax.errorbar(xs, m, yerr=s, color=colors[t], marker="o", ms=3, lw=1, capsize=2, label=TN[t].split(" (")[0])
    ax.axhline(R3["configs"]["mvfa"][t]["aupro"]["mean"], color=colors[t], ls="--", lw=0.8)
ax.set_xticks(xs, ["0 %\n(MIL only)", "10 %", "25 %", "100 %"])
ax.set_xlabel("abnormal masked-source images that keep their mask", fontsize=6.5)
ax.set_ylabel("AUPRO (target)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6, frameon=False)
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "fig1_budget.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(FIG / "fig1_budget.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
(FIG / "info.json").write_text(json.dumps([{
    "name": "fig1_budget.pdf",
    "caption": "Label-budget curve (round 3, seeds 3-5, mean ± SD): target AUPRO when masks are kept for 0/10/25/100 % "
               "of the masked sources' abnormal images and MIL is used on all other images. Dashed: MVFA re-run "
               "with full masks. Drawn from results.json."}], indent=1), encoding="utf-8")

# ------------------------------------------------------------------ figure 2: round 3 Liver heat maps (fixed images)
HM = DATA / "task4_r3/report/paper_inputs/task4_r3"
shutil.copy2(HM / "liver_heatmaps_r3.pdf", FIG / "fig2_liver_heatmaps_r3.pdf")
shutil.copy2(HM / "liver_heatmaps_r3_images.txt", FIG / "fig2_liver_heatmaps_r3_images.txt")
info = json.loads((FIG / "info.json").read_text(encoding="utf-8"))
info.append({"name": "fig2_liver_heatmaps_r3.pdf",
             "caption": "Liver CT heat maps, round 3, seed 3, for four test images fixed by round 1's rule before any "
                        "map was viewed (cyan: ground-truth outline). Columns: image, R0, MVFA, MIL, PU+100 %, "
                        "MIL+100 %, MIL+10 %, MIL+25 %. Illustrative only."})
(FIG / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")

# ------------------------------------------------------------------ experiment log
L = ["# Experiment log: Task 4, final (all values from `results.json`; rounds are never pooled)", "",
     "## Framing (PAPER_PLAN §5.2 applied to round 3 + follow-up)", "",
     "Round 3's verdict: one result confirmed at the top grade (H1), one 'matches' (H4), two retractions (H2, H3), "
     "one 'trails' (H5). The pre-registered follow-up then CONFIRMED MIL+10 % > MVFA on Brain. Under §5.2 this is "
     "a **partial win → mixed paper**: the confirmed results are the headline; everything else goes in one compact "
     "'what did not hold' table. No claim of beating MVFA overall, and no claim against the published MVFA numbers "
     "(different protocol: context only).", "",
     "## Protocol (rounds 2, 3 and the follow-up)", "",
     f"Leave-one-modality-out on BMAD's six datasets, {PR['folds']} folds, RESC and OCT17 held out together; "
     f"every arm trained from scratch on at most {PR['train_images_per_class_per_source']} normal + "
     f"{PR['train_images_per_class_per_source']} abnormal images per source (OCT17: {PR['oct17_normal_train_images']} normal); "
     f"checkpoint chosen on source validation image AUROC ({PR['validation_images_per_class_cap']} images per class cap, "
     f"{PR['valid_test_duplicates_removed']} valid/test duplicates removed); frozen CLIP ViT-L/14 at {PR['image_size']} px "
     f"({PR['patches_per_image']} patches), MIL top-k = {PR['topk_patches']} patches; Adam lr {PR['adam_lr']}, batch {PR['batch_size']}. "
     f"Round 2: {R2['epochs'][0]} epochs, seeds {R2['main_seeds']}. Round 3: {R3['epochs'][0]} epochs, seeds {R3['main_seeds']} "
     f"(Brain 3-7 for H2). Follow-up: {FU['epochs'][0]} epochs, seeds {FU['main_seeds']}, Brain only. Rules frozen (hashed) before each round.", "",
     "## Table 1 source: round 3 (pixel AUROC / AUPRO, mean ± SD over seeds 3-5)", "",
     "| arm | " + " | ".join(f"{TN[t]} pix | {TN[t]} AUPRO" for t in T) + " | retention (mean) |",
     "|---|" + "---|" * 7]
for a in ["mvfa", "mil", "mil_m10", "mil_m25", "mil_m100", "pu_m100"]:
    c = R3["configs"][a]
    ret = R3["retention"].get(a, {}).get("_mean", 1.0)
    L.append(f"| {NAME[a]} | " + " | ".join(f"{ms(c[t]['pix_auroc'])} | {ms(c[t]['aupro'])}" for t in T) + f" | {ret:.3f} |")
r0 = R1["r0_reproduced"]
L += ["", "R0 (released checkpoint, measured in round 1; selected on the target test set, different protocol): " +
      ", ".join(f"{TN[t]} {r0[t]['pix_auroc']:.2f} / {r0[t]['aupro']:.2f}" for t in T) + ".", "",
      "## Hypotheses (frozen rules) and outcomes", ""]
p = R3["paired"]
L += [f"- H1 MIL keeps MVFA's localisation: retention {R3['retention']['mil']['Brain']:.3f} / "
      f"{R3['retention']['mil']['Liver']:.3f} / {R3['retention']['mil']['Retina_RESC']:.3f} (mean "
      f"{R3['retention']['mil']['_mean']:.3f}) → NEARLY MATCHES (≥ 0.90 on all targets). Earlier rounds: round 1 B0 "
      f"{R1['retention']['b0']['_mean_seed0_over_targets']:.3f} of R0 (1 seed); round 2 {R2['retention']['mil']['_mean']:.3f}.",
      f"- H2 MIL beats MVFA on Brain: AUPRO higher on {p['mil-mvfa']['per_target']['Brain']['aupro_positive']}/"
      f"{p['mil-mvfa']['per_target']['Brain']['n']} seeds, p = {R3['tests']['H2']['raw_p']} → did not replicate.",
      f"- H3 PU+100 % beats MVFA: {p['pu_m100-mvfa']['main_seeds']['positive']['aupro']}/{p['pu_m100-mvfa']['main_seeds']['n_pairs']} "
      f"pairs, p = {R3['tests']['H3']['raw_p']} → did not replicate.",
      f"- H4 MIL+100 % beats MVFA: {p['mil_m100-mvfa']['main_seeds']['positive']['aupro']}/9 pairs, p = {R3['tests']['H4']['raw_p']}; "
      "pix Δ " + " / ".join(f"{p['mil_m100-mvfa']['main_seeds']['mean_delta_by_target'][t]['pix_auroc']:+.2f}" for t in T) +
      " → MATCHES (within 1 point everywhere), does not beat.",
      "- H5 MIL+10 % matches MVFA: pix Δ " + " / ".join(
          f"{p['mil_m10-mvfa']['main_seeds']['mean_delta_by_target'][t]['pix_auroc']:+.2f}" for t in T) + " → TRAILS (Liver).",
      f"- FU-H MIL+10 % beats MVFA on Brain (seeds 8-12): AUPRO higher on "
      f"{FU['paired']['mil_m10-mvfa']['per_target']['Brain']['aupro_positive']}/5, p = "
      f"{FU['paired']['mil_m10-mvfa']['per_target']['Brain']['sign_test_p_aupro']}; mean Δ AUPRO "
      f"{FU['paired']['mil_m10-mvfa']['per_target']['Brain']['mean_delta_aupro']:+.2f}, Δ pix "
      f"{FU['paired']['mil_m10-mvfa']['per_target']['Brain']['mean_delta_pix_auroc']:+.2f}, Δ img "
      f"{FU['paired']['mil_m10-mvfa']['per_target']['Brain']['mean_delta_img_auroc']:+.2f} → CONFIRMED (Brain only).",
      "- Round 2 idea #1 (normal-bank-gated MIL) vs MIL, pix Δ " + " / ".join(
          f"{R2['paired']['g-mil']['per_target'][t]['mean_delta_pix_auroc']:+.2f}" for t in T) + " → falsified.",
      "- Round 2 idea #3 (nnPU, fixed π = 1 %) vs MIL, pix Δ " + " / ".join(
          f"{R2['paired']['pu-mil']['per_target'][t]['mean_delta_pix_auroc']:+.2f}" for t in T) + " → not supported.",
      f"- Round 1 PU fine-tune vs control: AUPRO up in {R1['paired']['mh-mh_ctrl']['positive']['aupro']}/9 pairs, "
      f"p = {R1['paired']['mh-mh_ctrl']['sign_test_p_aupro']} → no consistent gain; the π estimator failed.", "",
      "## Diagnostics and caveats", "",
      "- In-mask share of the top-k patches on source valid/ images (round 3, mean of folds): " + ", ".join(
          f"{NAME[a]} {R3['inmask'][a]['mean_of_folds']:.3f}" for a in ["mvfa", "mil", "mil_m10", "mil_m25", "mil_m100"]) +
      f". Follow-up Brain fold: MVFA {FU['inmask']['mvfa']['mean_of_folds']:.3f} vs MIL+10 % {FU['inmask']['mil_m10']['mean_of_folds']:.3f}.",
      f"- Target image AUROC is weak under source-only selection: MVFA Brain {R3['configs']['mvfa']['Brain']['img_auroc']['mean']:.2f} "
      f"at 20 epochs vs {R2['configs']['mvfa']['Brain']['img_auroc']['mean']:.2f} at 5 epochs vs R0 {r0['Brain']['img_auroc']:.2f}.",
      f"- Schedule: {R3['best_epoch']['last']}/{R3['best_epoch']['n_runs']} round-3 runs picked their last epoch "
      f"(round 2: {R2['best_epoch']['excluding']['last']}/{R2['best_epoch']['excluding']['n_runs']}).",
      "- Limitations: subsample and 20 epochs (MVFA: full data, 50 epochs); 3 seeds (5 on Brain); one base model; "
      "Chest/HIS only in round 2 (Chest below chance for every arm); RESC held out with OCT17 (not comparable with MVFA's paper).", ""]
(RAW / "experiment_log.md").write_text("\n".join(L) + "\n", encoding="utf-8")

# ------------------------------------------------------------------ idea.md
idea = """# Paper idea and constraints (final: Task 4 rounds 1-3 + follow-up)

## Framing
Mixed paper (PAPER_PLAN §5.2, partial win). Headline 1: image labels alone (plain top-k MIL) nearly match mask
supervision for MVFA's localisation on unseen modalities (H1, confirmed in three rounds). Headline 2 (Brain only):
masks for 10 % of the abnormal source images plus MIL beat full-mask MVFA (pre-registered follow-up, 5/5 seeds).
Secondary: MIL added to full masks matches MVFA (H4). Everything else (H2, H3, H5, round 2's ideas #1 and #3, round 1's
PU) goes into one compact "what did not hold" table. Never claim to beat MVFA overall or the published MVFA numbers.

## Method
MVFA (frozen CLIP ViT-L/14 + residual adapters at 4 levels) trains segmentation adapters with Dice + Focal on the
masked sources only. MIL replaces that with, per level, BCE(mean of the top-k = 3 patch probabilities, image label)
on every source image. A mask budget b keeps masks for a seeded share b of the masked sources' abnormal images.
MIL+100 % keeps all masks and adds MIL on the mask-less sources. Tested refinements (from the LLM pipeline):
normal-bank-gated top-k selection, and nnPU with an estimated or fixed class prior.

## Structure and 4+1 budget
1 Introduction; 2 Related Work (2.1 vision-language medical anomaly detection; 2.2 weak supervision / MIL;
2.3 PU learning); 3 Problem Formulation; 4 Methodology (4.1 data and protocol; 4.2 study design); 5 Experiments
(Table 1 round 3, Figure 1 label budget, Table 2 hypothesis record, diagnostics, limitations); 6 Conclusion.
Content ends on page 4; page 5 = references only. Numbers only from results.json; citations only from refs.bib.
"""
(RAW / "idea.md").write_text(idea, encoding="utf-8")

# ------------------------------------------------------------------ related work outline + citation registry
pool = json.loads((DATA / "citation_pool.json").read_text(encoding="utf-8"))
sections = {"2.1": "Medical anomaly detection and vision-language adaptation",
            "2.2": "Weakly supervised and mask-free anomaly localisation",
            "2.3": "Positive-unlabelled learning and mixture-proportion estimation"}
out = ["# Related Work outline (curated Task 1 pool + approved extra refs only)", ""]
for sid, title in sections.items():
    out += [f"## {sid} {title}", ""]
    out += [f"- `\\cite{{{q['citation_key']}}}` {q['title']}: {q.get('inclusion_reason', '')}" for q in pool if q.get("section") == sid]
    out.append("")
(RAW / "related_work_outline.md").write_text("\n".join(out) + "\n", encoding="utf-8")
shutil.copy2(DATA / "citation_pool.json", RAW / "citation_pool.json")
shutil.copy2(DATA / "refs.bib", RAW / "refs.bib")
print("wrote", RAW)
