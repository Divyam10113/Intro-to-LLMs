# Follow-up rule (frozen before any follow-up run)

**Why this run exists.** In round 3, MIL with masks for 10 % of the masked sources' abnormal images (`mil_m10`) was ahead of the re-run MVFA on Brain AUPRO on every seed (3–5), by 10.9–16.3 points. Its Brain pixel AUROC (92.63) was also above the MVFA paper's (90.27). Both were **post-hoc** observations (round 3 report §4, VERDICT weakness 3, ISSUES_LOG G3-5), so neither could be claimed. This follow-up tests the Brain effect once, under a rule fixed in advance, on seeds that have never been used.

**Runs.** 10 runs: `mvfa` and `mil_m10` on the Brain fold, seeds **8, 9, 10, 11, 12**, 20 epochs. They use round 3's code, data subsample, source-validation checkpointing and evaluation, all unchanged (`task4fu/README.md`). Seeds 3–7 are never pooled with them.

## The one pre-registered test (FU-H)

**FU-H: MIL+10 % beats MVFA on Brain.** For each seed s, Δ_AUPRO(s) = AUPRO(mil_m10, s) − AUPRO(mvfa, s) on the Brain test set; Δ_pix(s) is defined the same way for pixel AUROC.

| Outcome | Condition |
|---|---|
| **CONFIRMED** | Δ_AUPRO > 0 on **5 of 5** seeds (one-sided sign test p = 0.031 ≤ 0.05; one test, so no correction) **and** mean Δ_pix > 0 |
| **holds on most seeds (not confirmed)** | Δ_AUPRO > 0 on 4 of 5 seeds |
| **did not replicate** | Δ_AUPRO > 0 on 3 or fewer of 5 seeds, **or** 5/5 with mean Δ_pix ≤ 0 |
| **incomplete** | Fewer than 5 seed pairs have both runs. The result is reported as "k/n pairs", and CONFIRMED cannot be reached |

- Ties (Δ = 0) count against MIL+10 %, as in rounds 2–3.
- Points = AUROC/AUPRO × 100, read from `results/results.csv`, rows not marked SMOKE.
- Only `evaluate_target.py` reads the Brain test set, after selection on source validation.

## Descriptive (reported, not tested)

- Image AUROC: per-seed and mean Δ (MIL+10 % − MVFA). Round 3 found MVFA's Brain image AUROC near chance at 20 epochs.
- MIL+10 % Brain pixel AUROC against the **MVFA paper's 90.27**: on how many seeds it is above, and the mean. This compares different protocols (the paper trains on full data for 50 epochs and selects on the test set), so it is context, not a claim.
- MIL+10 % Brain AUPRO against **R0's 65.35** (round 1's measurement of the released checkpoint): same caveat.
- Round 3's seeds 3–5 for the same two arms, side by side and never pooled.
- Best epochs and the in-mask diagnostic.

## What the paper may say

- **CONFIRMED:** "With masks for 10 % of the abnormal source images and image labels elsewhere, MVFA's adapters localise better than full-mask MVFA on unseen brain MRI (AUPRO higher on 5/5 pre-registered seeds, p = 0.031)." The comparison with the published model stays descriptive.
- **Any other outcome:** the round 3 Brain observation goes in the "what did not hold" table (or "holds on most seeds" for 4/5), and the paper's headline stays H1.

Frozen hashes: `report/FROZEN.sha256` (this file and `analyze_fu.py`). `go.sh` refuses to launch if they change.
