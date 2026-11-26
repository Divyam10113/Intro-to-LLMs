# Experiment log: Task 4, final (all values from `results.json`; rounds are never pooled)

## Framing (PAPER_PLAN §5.2 applied to round 3 + follow-up)

Round 3's verdict: one result confirmed at the top grade (H1), one 'matches' (H4), two retractions (H2, H3), one 'trails' (H5). The pre-registered follow-up then CONFIRMED MIL+10 % > MVFA on Brain. Under §5.2 this is a **partial win → mixed paper**: the confirmed results are the headline; everything else goes in one compact 'what did not hold' table. No claim of beating MVFA overall, and no claim against the published MVFA numbers (different protocol: context only).

## Protocol (rounds 2, 3 and the follow-up)

Leave-one-modality-out on BMAD's six datasets, 5 folds, RESC and OCT17 held out together; every arm trained from scratch on at most 500 normal + 500 abnormal images per source (OCT17: 242 normal); checkpoint chosen on source validation image AUROC (250 images per class cap, 11 valid/test duplicates removed); frozen CLIP ViT-L/14 at 240 px (289 patches), MIL top-k = 3 patches; Adam lr 0.0001, batch 1. Round 2: 5 epochs, seeds [0, 1, 2]. Round 3: 20 epochs, seeds [3, 4, 5] (Brain 3-7 for H2). Follow-up: 20 epochs, seeds [8, 9, 10, 11, 12], Brain only. Rules frozen (hashed) before each round.

## Table 1 source: round 3 (pixel AUROC / AUPRO, mean ± SD over seeds 3-5)

| arm | Brain MRI pix | Brain MRI AUPRO | Liver CT pix | Liver CT AUPRO | RESC (retinal OCT) pix | RESC (retinal OCT) AUPRO | retention (mean) |
|---|---|---|---|---|---|---|---|
| MVFA (re-run, full masks) | 87.84 ± 0.87 | 59.31 ± 2.15 | 97.21 ± 0.01 | 91.55 ± 0.09 | 86.14 ± 0.69 | 57.66 ± 1.39 | 1.000 |
| MIL (no masks) | 88.56 ± 2.77 | 63.77 ± 3.77 | 95.50 ± 0.12 | 88.08 ± 0.68 | 83.40 ± 3.71 | 54.31 ± 6.52 | 0.969 |
| MIL+10 % masks | 92.63 ± 0.76 | 73.29 ± 2.99 | 95.25 ± 1.61 | 88.55 ± 2.32 | 86.80 ± 1.05 | 58.03 ± 2.16 | 1.034 |
| MIL+25 % masks | 92.42 ± 0.80 | 73.18 ± 2.72 | 96.64 ± 0.22 | 90.39 ± 0.36 | 88.08 ± 0.89 | 59.88 ± 2.04 | 1.054 |
| MIL+100 % masks | 91.48 ± 0.98 | 69.91 ± 3.48 | 96.33 ± 0.29 | 90.84 ± 0.01 | 88.88 ± 0.41 | 61.97 ± 1.04 | 1.051 |
| PU+100 % masks | 89.82 ± 0.61 | 65.75 ± 1.72 | 96.57 ± 0.20 | 91.04 ± 0.28 | 87.26 ± 0.98 | 57.78 ± 2.89 | 1.023 |

R0 (released checkpoint, measured in round 1; selected on the target test set, different protocol): Brain MRI 89.68 / 65.35, Liver CT 97.92 / 92.34, RESC (retinal OCT) 90.44 / 68.75.

## Hypotheses (frozen rules) and outcomes

- H1 MIL keeps MVFA's localisation: retention 1.019 / 0.964 / 0.924 (mean 0.969) → NEARLY MATCHES (≥ 0.90 on all targets). Earlier rounds: round 1 B0 0.966 of R0 (1 seed); round 2 0.930.
- H2 MIL beats MVFA on Brain: AUPRO higher on 3/5 seeds, p = 0.5 → did not replicate.
- H3 PU+100 % beats MVFA: 5/9 pairs, p = 0.5 → did not replicate.
- H4 MIL+100 % beats MVFA: 6/9 pairs, p = 0.2539; pix Δ +3.63 / -0.88 / +2.74 → MATCHES (within 1 point everywhere), does not beat.
- H5 MIL+10 % matches MVFA: pix Δ +4.79 / -1.96 / +0.66 → TRAILS (Liver).
- FU-H MIL+10 % beats MVFA on Brain (seeds 8-12): AUPRO higher on 5/5, p = 0.0312; mean Δ AUPRO +14.70, Δ pix +5.35, Δ img +1.08 → CONFIRMED (Brain only).
- Round 2 idea #1 (normal-bank-gated MIL) vs MIL, pix Δ -5.26 / -1.15 / +4.25 → falsified.
- Round 2 idea #3 (nnPU, fixed π = 1 %) vs MIL, pix Δ -0.67 / -4.01 / -13.83 → not supported.
- Round 1 PU fine-tune vs control: AUPRO up in 6/9 pairs, p = 0.2539 → no consistent gain; the π estimator failed.

## Diagnostics and caveats

- In-mask share of the top-k patches on source valid/ images (round 3, mean of folds): MVFA (re-run, full masks) 0.787, MIL (no masks) 0.452, MIL+10 % masks 0.661, MIL+25 % masks 0.711, MIL+100 % masks 0.779. Follow-up Brain fold: MVFA 0.931 vs MIL+10 % 0.791.
- Target image AUROC is weak under source-only selection: MVFA Brain 56.33 at 20 epochs vs 65.73 at 5 epochs vs R0 79.80.
- Schedule: 11/58 round-3 runs picked their last epoch (round 2: 37/57).
- Limitations: subsample and 20 epochs (MVFA: full data, 50 epochs); 3 seeds (5 on Brain); one base model; Chest/HIS only in round 2 (Chest below chance for every arm); RESC held out with OCT17 (not comparable with MVFA's paper).

