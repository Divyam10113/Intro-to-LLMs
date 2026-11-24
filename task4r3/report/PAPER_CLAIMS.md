# Paper claims from Task 4 (rounds 1–3)

Built from round 3's frozen-rule outcomes (plan §6). Each claim names its source row in
[`results/analysis.md`](../results/analysis.md) (AN), [`per_seed.md`](../results/per_seed.md) (PS) or the
tables in [`paper_inputs/task4_r3/`](paper_inputs/task4_r3/) (PI). Points = AUROC/AUPRO × 100; mean over seeds 3–5.

## Sorting by outcome (plan §6)

| Hypothesis | §5 outcome | Use in the paper |
|---|---|---|
| **H1:** MIL without masks keeps MVFA's localisation | **Confirmed: NEARLY MATCHES** | **Headline:** title, abstract, contribution 1 |
| **H4:** MIL+100 % beats MVFA | **MATCHES** (not "beats") | Results section and contribution 2, worded as *matches*, never *beats* (note 1) |
| **H2:** MIL beats MVFA on Brain | Did not replicate (3/5) | "What did not hold" table |
| **H3:** PU+100 % beats MVFA | Did not replicate (5/9) | "What did not hold" table |
| **H5:** MIL+10 % matches MVFA | Trails (Liver pix Δ −1.96) | "What did not hold" table, with the budget curve as description |
| **FU-H (follow-up, 2026-11-24):** MIL+10 % beats MVFA on Brain | **CONFIRMED** (AUPRO 5/5 pre-registered seeds 8–12, p = 0.031; mean pix Δ +5.35) | **Second headline, Brain only:** contribution 4, Results. See the follow-up section at the end |
| Round 2, idea #1 (normal-bank-gated MIL) | Falsified | "What did not hold" table |
| Rounds 1–2, idea #3 (nnPU, estimated π, then fixed π) | Not supported | "What did not hold" table |

**Note 1.** Plan §6 lists three outcome groups. H4's "matches" is in none of them: it is not "supported", and it is not "did not replicate / trails". It is placed in the Results section, worded exactly as the rule's outcome ("matches MVFA: pixel AUROC within 1 point on every target"). The teammate can move it at gate G2.

**Nothing qualifies as "suggestive / holds on most seeds"** under round 3's rules.

## Title options

1. *Image Labels Are Nearly Enough: Weakly Supervised Adapters for Cross-Modality Medical Anomaly Localisation*
2. *How Far Do Image Labels Go? A Pre-Registered, Three-Round Study of MIL for Zero-Shot Medical Anomaly Segmentation*
3. *Replicating LLM-Proposed Weak Supervision for Medical Anomaly Localisation: What Held and What Did Not*

## One-sentence contribution

Under a leave-one-modality-out protocol with source-only model selection and pre-registered rules, plain top-k MIL trained only on image labels retains 97 % of mask-supervised MVFA's pixel AUROC on unseen modalities (≥ 92 % on each). The result holds across three rounds, two schedules and fresh seeds, while two LLM-proposed refinements and two earlier apparent wins do not survive replication.

## Three contributions

1. **Image labels nearly match masks for cross-modality localisation (H1, confirmed).**
   - MIL retains 1.019 / 0.964 / 0.924 of MVFA's pixel AUROC on Brain / Liver / RESC (mean 0.969). *AN: "H1"; PI: `table_hypotheses`, `table_budget` row 0 %.*
   - It is consistent over rounds: 0.967 (round 1), 0.930 (round 2), 0.969 (round 3). *Round 1 `TASK4_REPORT.md`; round 2 `VERDICT.md`.*
2. **Adding MIL on the mask-less sources to mask training matches MVFA (H4, "matches"), and a small mask budget sits between the two (descriptive).**
   - MIL+100 % pix Δ +3.63 / −0.88 / +2.74. *AN: "H4"; PI: `table_hypotheses`.*
   - Mean retention 0.969 → 1.034 → 1.054 → 1.051 at 0 / 10 / 25 / 100 % masks. *AN: "Label-budget curve"; PI: `table_budget`, `budget_curve_r3.png`.*
   - The in-mask share of the top-k patches rises with the mask share: 0.45 → 0.66 → 0.71 → 0.78. *PI: `table_inmask`.*
3. **A pre-registered replication record (negative results reported in full).**
   - Of four positive signals from rounds 1–2, one (P1) holds. Two (P2, P3) do not replicate at 4× the schedule on fresh seeds, and one (P4) carries over in part.
   - Both of the LLM pipeline's top-ranked ideas (#1, #3) were falsified or unsupported in round 2.
   - *AN: "H2", "H3", "H5", "p-values"; round 2 `VERDICT.md`.*

## Main table (round 3; PI: `table_main`)

| Target | Metric | MVFA | MIL (no masks) | MIL+10 % | MIL+25 % | MIL+100 % | PU+100 % |
|---|---|---|---|---|---|---|---|
| Brain | pix AUROC | 87.84 ± 0.87 | 88.56 ± 2.77 | 92.63 ± 0.76 | 92.42 ± 0.80 | 91.48 ± 0.98 | 89.82 ± 0.61 |
| Brain | AUPRO | 59.31 ± 2.15 | 63.77 ± 3.77 | 73.29 ± 2.99 | 73.18 ± 2.72 | 69.91 ± 3.48 | 65.75 ± 1.72 |
| Liver | pix AUROC | 97.21 ± 0.01 | 95.50 ± 0.12 | 95.25 ± 1.61 | 96.64 ± 0.22 | 96.33 ± 0.29 | 96.57 ± 0.20 |
| Liver | AUPRO | 91.55 ± 0.09 | 88.08 ± 0.68 | 88.55 ± 2.32 | 90.39 ± 0.36 | 90.84 ± 0.01 | 91.04 ± 0.28 |
| RESC | pix AUROC | 86.14 ± 0.69 | 83.40 ± 3.71 | 86.80 ± 1.05 | 88.08 ± 0.89 | 88.88 ± 0.41 | 87.26 ± 0.98 |
| RESC | AUPRO | 57.66 ± 1.39 | 54.31 ± 6.52 | 58.03 ± 2.16 | 59.88 ± 2.04 | 61.97 ± 1.04 | 57.78 ± 2.89 |
| — | Retention (mean) | 1.000 | **0.969** | 1.034 | 1.054 | 1.051 | — |

Image AUROC (Brain / Liver / RESC / OCT17) belongs in a second table or a column group: MVFA 56.3 / 60.6 / 74.3 / 70.8; MIL 67.5 / 68.0 / 71.3 / 62.1 (PI: `table_main`, `table_oct17`). The paper must say that target image AUROC is weak for every arm under source-only selection (limitation 2).

## Against the base paper (MVFA, CVPR 2024): context only, not a claim

These are different protocols, so this section is context, not a claim.
- **The paper's numbers** (round 1's `paper_inputs/table_a_main`) were trained on the full data for 50 epochs, and their checkpoint was selected on the test set.
- **R0** is round 1's measurement of the released checkpoint.
- **Round 3** trains from scratch, 20 epochs, on at most 1,000 images per source, with source-only selection. RESC is held out together with OCT17, so RESC has 2 masked sources instead of 3.

| Target | Metric | MVFA paper | R0 (released, measured) | Round 3 MVFA re-run | Best round 3 arm |
|---|---|---|---|---|---|
| Brain | pix AUROC | 90.27 | 89.68 | 87.84 | **92.63** (MIL+10 %); 92.42 (MIL+25 %); 91.48 (MIL+100 %) |
| Brain | AUPRO | – | 65.35 | 59.31 | **73.29** (MIL+10 %); 69.91 (MIL+100 %) |
| Liver | pix AUROC | 97.85 | 97.92 | 97.21 | 96.64 (MIL+25 %) |
| RESC | pix AUROC | 92.05 | 90.44 | 86.14 | 88.88 (MIL+100 %) |
| Brain / Liver / RESC | img AUROC | 78.63 / 76.24 / 83.31 | 79.80 / 81.19 / 88.99 | 56.33 / 60.60 / 74.29 | 67.52 / 68.05 / 72.71 |

**Reading:**
- **Brain localisation is above the published model for every arm that adds MIL to some masks,** even though those arms had less data, a shorter schedule and no test-set selection. The best is +2.4 pix AUROC over the paper and +7.9 AUPRO over R0. This is post hoc and Brain-only (report §4), so it is a lead for a pre-registered follow-up, not a claim. **Update 2026-11-24: for MIL+10 % vs MVFA, the pre-registered follow-up CONFIRMED it (see the follow-up section at the end). The comparison with the paper itself stays context only.**
- **On Liver and RESC, nothing reaches the paper.**
- **Image-level detection is far below the paper on every target** (limitation 2).

## "What did not hold" (one line each)

| Claim tested | Where | Rule | Result |
|---|---|---|---|
| MIL beats MVFA on Brain (P2) | R3, H2 | AUPRO higher on 5/5 seeds | 3/5, p = 0.50; mean pix Δ −0.81 |
| PU+100 % beats MVFA (P3) | R3, H3 | ≥ 8/9 pairs, Holm p ≤ 0.05 | 5/9, p = 0.50 (round 2: 7/9) |
| MIL with 10 % masks matches MVFA | R3, H5 | pix Δ ≥ −1 on every target | Liver −1.96 (Brain +4.79, RESC +0.66) |
| Normal-bank-gated MIL improves on MIL (idea #1) | R2 | G − MIL ≥ +2 pix on all targets | −5.26 / −1.15 / +4.25; the gate selects lesion patches no more often (0.40 vs 0.43) |
| Fixed-π nnPU improves on MIL (idea #3, refined) | R2 | PU − MIL ≥ +1.5 pix on ≥ 2 targets | 0/3 (−0.67 / −4.01 / −13.83); collapses at π = 5 % |
| nnPU with estimated π (idea #3) | R1 | Consistent gain | No consistent gain; the π estimator failed (round 1, B11) |

## Figures

1. **Liver heat maps:** `paper_inputs/task4_r3/liver_heatmaps_r3.png/.pdf`, with columns R0 (reference), MVFA, MIL, PU+100 %, MIL+100 %, MIL+10 %, MIL+25 %. The images are fixed by round 1's rule (`liver_heatmaps_r3_images.txt`). The caption must say: 4 images, seed 3, illustrative.
2. **Label-budget curve:** `paper_inputs/task4_r3/budget_curve_r3.png/.pdf`, with AUPRO and pixel retention against the share of masks, per target, and MVFA dashed.

## Limitations (for the paper)

1. A 20-epoch schedule on at most 1,000 images per source (MVFA uses 50 epochs on the full data).
2. Source-only checkpoint selection, as the protocol requires, does not track target image AUROC: MVFA Brain image AUROC is 56.3 at 20 epochs vs 65.7 at 5, against 79.8 for the released (test-selected) checkpoint.
3. Three seeds (five on Brain for H2); one-sided sign tests can confirm only at 8/9 or better.
4. RESC and OCT17 are held out together, so they are not comparable with MVFA's paper. Chest and HIS come from round 2 (one seed; Chest below chance for every arm).
5. A single MVFA codebase with fixed k = 3, π = 1 % and batch 1; no hyper-parameter sweep.
6. The Brain gains of the budget arms are post hoc (report §4) and are not claimed.

## Framing

`PAPER_PLAN.md` §5.2 was not available to this session (ISSUES_LOG G3-2), so the applicable framing is left to the teammate. Based on the outcomes above, the framing that fits is: **a positive headline (H1) inside an honest replication study**. Lead with the image-label result. Give H4 and the budget curve as the practical recipe ("add image-label MIL to whatever masks you have"). Report the retractions in the "what did not hold" table, so the headline cannot be read as selectively reported.

---

## Follow-up (pre-registered, 2026-11-24): MIL+10 % vs MVFA on Brain, CONFIRMED

*Added after Task 4 closed. Source: `../../task4fu/report/FU_REPORT.md`, `../../task4fu/results/analysis_fu.md`. Rule frozen before the run (`task4fu/RULES_FU.md`).*

- **Outcome: CONFIRMED.** MIL+10 % AUPRO is higher than MVFA's on **5/5** fresh seeds (8–12), one-sided sign test p = 0.031. Mean Δ AUPRO **+14.70** (range +12.47 to +19.07). Mean Δ pixel AUROC **+5.35** (higher on every seed).
- **Means (seeds 8–12):** MIL+10 %: pix **92.36**, AUPRO **72.08**, img 64.37. MVFA re-run: 87.00 / 57.38 / 63.29. Round 3's seeds 3–5 gave 92.63 / 73.29 for MIL+10 %, so the new seeds reproduce it.
- **Sentence the rule allows:** *"With masks for 10 % of the abnormal source images and image labels elsewhere, MVFA's adapters localise better than full-mask MVFA on unseen brain MRI (AUPRO higher on 5/5 pre-registered seeds, p = 0.031)."*
- **Suggested contribution 4:** *A small mask budget plus image labels beats full masks on the target where masks transfer worst (Brain; pre-registered, 5/5 seeds).*
- **Context, not a claim:** MIL+10 % Brain pixel AUROC is above the MVFA paper's 90.27 on 5/5 seeds (mean +2.09), and its AUPRO is above R0's 65.35 on 5/5 seeds (mean +6.73). The protocols differ (subsample, 20 epochs, source-only selection vs the paper's full data, 50 epochs, test-set selection). Wording: "*exceeds the published Brain localisation numbers despite a cheaper protocol*", never "*beats MVFA*".
- **Limits, to state next to it:**
  - Brain only. On Liver, MIL+10 % trails MVFA (H5, pix Δ −1.96), and RESC was not re-tested.
  - Image AUROC is not improved (mean Δ +1.08, lower on 2/5 seeds).
  - The mechanism is untested. MVFA's top-k falls inside source masks more often (in-mask 0.93 vs 0.79), yet it localises worse on Brain, which is consistent with full masks over-fitting source lesion appearance. That is a hypothesis.
- **Framing change:** the paper can now lead with two confirmed results: **H1** (image labels nearly match masks, all targets) and **FU-H** (10 % masks + MIL beats full masks on Brain). The "what did not hold" table is unchanged.
