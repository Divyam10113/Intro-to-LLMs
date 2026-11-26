# Follow-up analysis: MIL+10 % vs MVFA on Brain (RULES_FU.md, applied by `analyze_fu.py`)

Source: `results.csv`. Points = AUROC/AUPRO × 100. Seeds [8, 9, 10, 11, 12]; round 3's seeds are never pooled.

## Per seed (Brain test set)

| seed | MVFA img | MVFA pix | MVFA AUPRO | MIL+10% img | MIL+10% pix | MIL+10% AUPRO | Δ AUPRO | Δ pix | Δ img |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 53.71 | 85.76 | 53.97 | 60.59 | 91.33 | 68.04 | +14.07 | +5.58 | +6.88 |
| 9 | 56.97 | 86.97 | 57.25 | 68.68 | 92.01 | 70.53 | +13.28 | +5.04 | +11.71 |
| 10 | 70.02 | 87.45 | 58.74 | 70.25 | 92.56 | 73.37 | +14.63 | +5.11 | +0.23 |
| 11 | 76.96 | 87.39 | 58.14 | 66.06 | 93.93 | 77.20 | +19.07 | +6.54 | -10.90 |
| 12 | 58.80 | 87.44 | 58.80 | 56.26 | 91.94 | 71.27 | +12.47 | +4.50 | -2.54 |

## FU-H: MIL+10 % beats MVFA on Brain (the one pre-registered test)

- AUPRO higher on **5/5** seed pairs; one-sided sign test p = 0.0312
- Mean pixel AUROC Δ (MIL+10 % − MVFA): +5.35
- **Outcome: CONFIRMED**

## Descriptive (not tested)

- MVFA (re-run) Brain, seeds 8–12: img 63.29 ± 9.80, pix 87.00 ± 0.72, AUPRO 57.38 ± 2.01; best epochs [15, 17, 15, 13, 17]; in-mask 0.93 ± 0.00
- MIL+10% Brain, seeds 8–12: img 64.37 ± 5.83, pix 92.36 ± 0.98, AUPRO 72.08 ± 3.44; best epochs [12, 14, 13, 17, 14]; in-mask 0.79 ± 0.02
- Mean image AUROC Δ (MIL+10 % − MVFA): +1.08
- MIL+10 % pixel AUROC above the MVFA paper's 90.27 on 5/5 seeds (mean 92.36). Different protocol: context only.
- MIL+10 % AUPRO above R0's 65.35 (released checkpoint, round 1) on 5/5 seeds. Context only.

## Round 3 seeds 3–5 (the post-hoc observation; side by side, never pooled)

| arm | seeds | img | pix | AUPRO |
|---|---|---|---|---|
| MVFA (re-run) | 3–5 | 56.33 ± 6.73 | 87.84 ± 0.87 | 59.31 ± 2.15 |
| MIL+10% | 3–5 | 61.21 ± 6.40 | 92.63 ± 0.76 | 73.29 ± 2.99 |

Round 3: AUPRO higher on 3/3 seeds; mean pix Δ +4.79.
