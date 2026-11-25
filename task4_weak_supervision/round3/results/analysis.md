# Round 3 analysis (plan §5 rules, applied mechanically by `analyze3.py`)

Source: `results.csv`. Points = AUROC/AUPRO × 100. Mean ± std over seeds 3–5 unless marked. A pair is (target, seed). Sign tests are one-sided; Holm within each tier.

Runs present: 58 (arm, masked target, seed) cells; arms: MVFA (re-run), MIL, PU+100%, MIL+100%, MIL+10%, MIL+25%.

## Main table (masked targets, seeds 3–5)

| target | arm | n | img AUROC | pix AUROC | AUPRO | best epochs |
|---|---|---|---|---|---|---|
| Brain | MVFA (re-run) | 3 | 56.33 ± 6.73 | 87.84 ± 0.87 | 59.31 ± 2.15 | 14,14,16 |
| Brain | MIL | 3 | 67.50 ± 3.70 | 88.56 ± 2.77 | 63.77 ± 3.77 | 12,19,18 |
| Brain | PU+100% | 3 | 67.52 ± 8.31 | 89.82 ± 0.61 | 65.75 ± 1.72 | 16,10,12 |
| Brain | MIL+100% | 3 | 60.30 ± 0.65 | 91.48 ± 0.98 | 69.91 ± 3.48 | 19,13,8 |
| Brain | MIL+10% | 3 | 61.21 ± 6.40 | 92.63 ± 0.76 | 73.29 ± 2.99 | 18,19,13 |
| Brain | MIL+25% | 3 | 58.62 ± 7.55 | 92.42 ± 0.80 | 73.18 ± 2.72 | 16,13,16 |
| Liver | MVFA (re-run) | 3 | 60.60 ± 2.53 | 97.21 ± 0.01 | 91.55 ± 0.09 | 19,17,18 |
| Liver | MIL | 3 | 67.95 ± 4.11 | 95.50 ± 0.12 | 88.08 ± 0.68 | 14,17,10 |
| Liver | PU+100% | 3 | 63.86 ± 4.01 | 96.57 ± 0.20 | 91.04 ± 0.28 | 13,15,16 |
| Liver | MIL+100% | 3 | 63.42 ± 3.21 | 96.33 ± 0.29 | 90.84 ± 0.01 | 13,17,19 |
| Liver | MIL+10% | 3 | 65.64 ± 6.50 | 95.25 ± 1.61 | 88.55 ± 2.32 | 19,14,10 |
| Liver | MIL+25% | 3 | 68.05 ± 6.12 | 96.64 ± 0.22 | 90.39 ± 0.36 | 8,19,13 |
| Retina_RESC | MVFA (re-run) | 3 | 74.29 ± 2.95 | 86.14 ± 0.69 | 57.66 ± 1.39 | 19,18,14 |
| Retina_RESC | MIL | 3 | 71.27 ± 0.83 | 83.40 ± 3.71 | 54.31 ± 6.52 | 18,18,15 |
| Retina_RESC | PU+100% | 3 | 71.37 ± 1.45 | 87.26 ± 0.98 | 57.78 ± 2.89 | 19,14,19 |
| Retina_RESC | MIL+100% | 3 | 72.25 ± 2.66 | 88.88 ± 0.41 | 61.97 ± 1.04 | 14,18,17 |
| Retina_RESC | MIL+10% | 3 | 71.58 ± 2.03 | 86.80 ± 1.05 | 58.03 ± 2.16 | 16,11,15 |
| Retina_RESC | MIL+25% | 3 | 72.71 ± 1.50 | 88.08 ± 0.89 | 59.88 ± 2.04 | 14,15,19 |

## OCT17 image AUROC (RESC+OCT17 fold, seeds 3–5)

| arm | img AUROC |
|---|---|
| MVFA (re-run) | 70.75 ± 11.09 |
| MIL | 62.13 ± 3.99 |
| PU+100% | 65.09 ± 2.37 |
| MIL+100% | 61.84 ± 3.62 |
| MIL+10% | 59.27 ± 3.74 |
| MIL+25% | 64.63 ± 2.90 |

## Tier 1: confirm rounds 1–2

### H1 (P1): MIL without masks keeps most of MVFA's localisation (estimate, no test)

- Retention (pix − 50) / (MVFA − 50): Brain 1.019, Liver 0.964, Retina_RESC 0.924; mean 0.969
- **H1: NEARLY MATCHES (retention >= 0.90 on all 3 targets)**

### H2 (P2): MIL beats MVFA on Brain

- Brain seeds [3, 4, 5, 6, 7]: MIL − MVFA AUPRO per seed: s3 -0.40, s4 +2.85, s5 +10.93, s6 -6.95, s7 +6.77
- AUPRO higher on 3/5 seeds; sign test p = 0.500 raw, 1.000 Holm (with H3)
- Mean Brain pixel AUROC Δ (MIL − MVFA, same seeds): -0.81
- **H2: did not replicate**

### H3 (P3): PU+100% beats MVFA

- AUPRO higher than MVFA in 5/9 pairs; sign test p = 0.500 raw, 1.000 Holm-adjusted
- Paired AUPRO Δ: Brain-s3 +4.22 Brain-s4 +4.21 Brain-s5 +10.91 Liver-s3 -0.68 Liver-s4 -0.63 Liver-s5 -0.23 Retin-s3 +2.43 Retin-s4 -4.83 Retin-s5 +2.74
- Mean pixel AUROC Δ: Brain +1.97, Liver -0.64, Retina_RESC +1.12 (higher on 2/3)
- Mean image AUROC Δ: Brain +11.19, Liver +3.26, Retina_RESC -2.92 (all ≥ −0.5: no)
- **Outcome: did not replicate**

## Tier 2: extend

### H4: MIL+100% beats MVFA

- AUPRO higher than MVFA in 6/9 pairs; sign test p = 0.254 raw, 0.508 Holm-adjusted
- Paired AUPRO Δ: Brain-s3 +7.92 Brain-s4 +6.85 Brain-s5 +17.04 Liver-s3 -0.61 Liver-s4 -0.79 Liver-s5 -0.74 Retin-s3 +5.72 Retin-s4 +1.51 Retin-s5 +5.70
- Mean pixel AUROC Δ: Brain +3.63, Liver -0.88, Retina_RESC +2.74 (higher on 2/3)
- Mean image AUROC Δ: Brain +3.97, Liver +2.83, Retina_RESC -2.04 (all ≥ −0.5: no)
- **Outcome: MATCHES**

### H5: MIL with 10 % of the masks matches MVFA

- Rule: mean pixel Δ (MIL+10% − MVFA) ≥ −1 on every target; "beats" if it also meets H3's rule.
- AUPRO higher than MVFA in 4/9 pairs; sign test p = 0.746 raw, 0.746 Holm-adjusted
- Paired AUPRO Δ: Brain-s3 +10.89 Brain-s4 +16.31 Brain-s5 +14.74 Liver-s3 -1.75 Liver-s4 -1.57 Liver-s5 -5.70 Retin-s3 +3.67 Retin-s4 -2.35 Retin-s5 -0.21
- Mean pixel AUROC Δ: Brain +4.79, Liver -1.96, Retina_RESC +0.66 (higher on 2/3)
- Mean image AUROC Δ: Brain +4.88, Liver +5.05, Retina_RESC -2.71 (all ≥ −0.5: no)
- **Outcome: TRAILS**

### p-values

| hypothesis | tier | raw p | Holm-adjusted p | outcome |
|---|---|---|---|---|
| H2 | 1 | 0.5000 | 1.0000 | did not replicate |
| H3 | 1 | 0.5000 | 1.0000 | did not replicate |
| H4 | 2 | 0.2539 | 0.5078 | MATCHES |
| H5 (beats part) | 2 | 0.7461 | 0.7461 | TRAILS |
| H1 | 1 | – | – | NEARLY MATCHES (retention >= 0.90 on all 3 targets) |

## Descriptive (no test)

### Label-budget curve: retention of MVFA's pixel AUROC

| masks | arm | Brain | Liver | Retina_RESC | mean |
|---|---|---|---|---|---|
| 0 % | MIL | 1.019 | 0.964 | 0.924 | 0.969 |
| 10 % | MIL+10% | 1.126 | 0.959 | 1.018 | 1.034 |
| 25 % | MIL+25% | 1.121 | 0.988 | 1.054 | 1.054 |
| 100 % | MIL+100% | 1.096 | 0.981 | 1.076 | 1.051 |
| 100 % (MVFA) | MVFA (re-run) | 1.000 | 1.000 | 1.000 | 1.000 |

AUPRO along the same curve (mean over seeds):

| masks | Brain | Liver | Retina_RESC |
|---|---|---|---|
| 0 % | 63.77 ± 3.77 | 88.08 ± 0.68 | 54.31 ± 6.52 |
| 10 % | 73.29 ± 2.99 | 88.55 ± 2.32 | 58.03 ± 2.16 |
| 25 % | 73.18 ± 2.72 | 90.39 ± 0.36 | 59.88 ± 2.04 |
| 100 % +MIL | 69.91 ± 3.48 | 90.84 ± 0.01 | 61.97 ± 1.04 |
| MVFA | 59.31 ± 2.15 | 91.55 ± 0.09 | 57.66 ± 1.39 |

### Round 3 (20 epochs, seeds 3–5) vs round 2 (5 epochs, seeds 0–2); never pooled

| arm | target | R2 pix | R3 pix | R2 AUPRO | R3 AUPRO | R2 img | R3 img |
|---|---|---|---|---|---|---|---|
| MVFA (re-run) | Brain | 88.32 ± 0.27 | 87.84 ± 0.87 | 61.55 ± 0.74 | 59.31 ± 2.15 | 65.73 ± 1.97 | 56.33 ± 6.73 |
| MVFA (re-run) | Liver | 96.80 ± 0.61 | 97.21 ± 0.01 | 90.40 ± 1.13 | 91.55 ± 0.09 | 71.16 ± 8.86 | 60.60 ± 2.53 |
| MVFA (re-run) | Retina_RESC | 87.92 ± 0.34 | 86.14 ± 0.69 | 58.93 ± 2.05 | 57.66 ± 1.39 | 79.98 ± 6.60 | 74.29 ± 2.95 |
| MIL | Brain | 91.71 ± 1.49 | 88.56 ± 2.77 | 70.63 ± 5.05 | 63.77 ± 3.77 | 66.93 ± 2.80 | 67.50 ± 3.70 |
| MIL | Liver | 96.19 ± 0.34 | 95.50 ± 0.12 | 89.26 ± 1.37 | 88.08 ± 0.68 | 69.90 ± 3.10 | 67.95 ± 4.11 |
| MIL | Retina_RESC | 77.14 ± 9.90 | 83.40 ± 3.71 | 48.80 ± 10.68 | 54.31 ± 6.52 | 76.47 ± 2.42 | 71.27 ± 0.83 |
| PU+100% | Brain | 88.77 ± 0.93 | 89.82 ± 0.61 | 63.32 ± 2.70 | 65.75 ± 1.72 | 70.51 ± 2.68 | 67.52 ± 8.31 |
| PU+100% | Liver | 96.91 ± 0.58 | 96.57 ± 0.20 | 91.19 ± 1.19 | 91.04 ± 0.28 | 73.83 ± 2.89 | 63.86 ± 4.01 |
| PU+100% | Retina_RESC | 88.60 ± 1.54 | 87.26 ± 0.98 | 60.38 ± 3.40 | 57.78 ± 2.89 | 77.49 ± 1.66 | 71.37 ± 1.45 |

### Best-epoch histogram (one count per run; epochs are 0-based)

| best epoch | 8 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| runs | 2 | 3 | 1 | 2 | 7 | 9 | 5 | 6 | 5 | 7 | 11 |

- 11/58 runs picked their last epoch; 18/58 one of the last two.
- **Most runs do not pick their last epoch.**
- Last-epoch share per arm: MVFA (re-run) 2/11, MIL 1/11, PU+100% 2/9, MIL+100% 2/9, MIL+10% 2/9, MIL+25% 2/9

### In-mask diagnostic (round 2's): share of top-k patches inside the true mask

(masked sources' valid/ abnormal images, selected checkpoint; by p)

| arm | by p |
|---|---|
| MVFA (re-run) | 0.766 |
| MIL | 0.417 |
| PU+100% | 0.741 |
| MIL+100% | 0.732 |
| MIL+10% | 0.597 |
| MIL+25% | 0.654 |

### Image AUROC below 50 (flagged)

- none
