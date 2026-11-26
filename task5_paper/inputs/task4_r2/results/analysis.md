# Round 2 analysis (§6 rules applied mechanically by `analyze2.py`)

All numbers are in points (×100). Mean ± std over seeds 0–2 unless marked; Chest/HIS/OCT17 = seed 0 image AUROC only (OCT17 from the RESC+OCT17 fold).

## Main table (masked targets)

| target | arm | n | img AUROC | pix AUROC | AUPRO | best epochs |
|---|---|---|---|---|---|---|
| Brain | R0 (released, round 1; different protocol) | 1 | 79.80 | 89.68 | 65.35 | – |
| Brain | MVFA (re-run) | 3 | 65.73 ± 1.97 | 88.32 ± 0.27 | 61.55 ± 0.74 | 4,3,4 |
| Brain | MIL | 3 | 66.93 ± 2.80 | 91.71 ± 1.49 | 70.63 ± 5.05 | 4,4,3 |
| Brain | PU (π=1%) | 3 | 66.73 ± 2.03 | 91.04 ± 2.23 | 69.42 ± 5.54 | 4,3,4 |
| Brain | G (gated MIL) | 3 | 64.80 ± 1.39 | 86.45 ± 2.12 | 55.36 ± 6.12 | 4,4,3 |
| Brain | PU+100% | 3 | 70.51 ± 2.68 | 88.77 ± 0.93 | 63.32 ± 2.70 | 3,4,2 |
| Brain | G+100% | 3 | 65.76 ± 2.07 | 89.82 ± 0.75 | 65.96 ± 2.84 | 4,3,4 |
| Brain | PU-π5 | 1 | 67.29 | 88.24 | 61.33 | 4 |
| Liver | R0 (released, round 1; different protocol) | 1 | 81.19 | 97.92 | 92.34 | – |
| Liver | MVFA (re-run) | 3 | 71.16 ± 8.86 | 96.80 ± 0.61 | 90.40 ± 1.13 | 2,4,4 |
| Liver | MIL | 3 | 69.90 ± 3.10 | 96.19 ± 0.34 | 89.26 ± 1.37 | 3,4,4 |
| Liver | PU (π=1%) | 3 | 68.44 ± 4.87 | 92.18 ± 3.48 | 81.57 ± 5.15 | 3,4,4 |
| Liver | G (gated MIL) | 3 | 72.16 ± 2.54 | 95.04 ± 1.31 | 85.75 ± 3.25 | 4,4,4 |
| Liver | PU+100% | 3 | 73.83 ± 2.89 | 96.91 ± 0.58 | 91.19 ± 1.19 | 3,4,3 |
| Liver | G+100% | 3 | 69.94 ± 4.97 | 96.51 ± 0.41 | 90.30 ± 0.88 | 3,3,4 |
| Liver | PU-π5 | 1 | 72.38 | 65.79 | 46.56 | 4 |
| Retina_RESC | R0 (released, round 1; different protocol) | 1 | 88.99 | 90.44 | 68.75 | – |
| Retina_RESC | MVFA (re-run) | 3 | 79.98 ± 6.60 | 87.92 ± 0.34 | 58.93 ± 2.05 | 2,3,4 |
| Retina_RESC | MIL | 3 | 76.47 ± 2.42 | 77.14 ± 9.90 | 48.80 ± 10.68 | 4,4,4 |
| Retina_RESC | PU (π=1%) | 3 | 77.18 ± 1.01 | 63.31 ± 11.35 | 33.06 ± 8.65 | 4,3,3 |
| Retina_RESC | G (gated MIL) | 3 | 79.14 ± 1.12 | 81.39 ± 5.73 | 52.44 ± 5.60 | 4,4,4 |
| Retina_RESC | PU+100% | 3 | 77.49 ± 1.66 | 88.60 ± 1.54 | 60.38 ± 3.40 | 4,4,3 |
| Retina_RESC | G+100% | 3 | 72.28 ± 6.59 | 87.79 ± 0.47 | 58.59 ± 1.01 | 4,3,4 |
| Retina_RESC | PU-π5 | 1 | 77.81 | 49.11 | 18.59 | 4 |

## Image AUROC on the mask-less held-out datasets (seed 0)

| dataset | MVFA (re-run) | MIL | PU (π=1%) | G (gated MIL) |
|---|---|---|---|---|
| Retina_OCT2017 | 81.18 | 70.90 | 63.30 | 62.93 |
| Chest | 44.57 | 45.67 | 49.02 | 43.78 |
| Histopathology | 61.70 | 72.13 | 60.69 | 70.80 |

## Rule: idea #1 (gated MIL) against Task 3's hypothesis

- G − MIL pixel AUROC (mean over seeds): Brain -5.26, Liver -1.15, Retina_RESC +4.25
- G − MIL image AUROC, seed 0: Chest -1.89, Histopathology -1.32
- **Verdict: FALSIFIED (no pixel-AUROC gain over MIL)**

## Rule: refined idea #3 (PU, fixed π = 1 %)

- PU − MIL pixel AUROC: Brain -0.67, Liver -4.01, Retina_RESC -13.83
- **Verdict: NOT SUPPORTED** (0/3 targets ≥ +1.5; rule needs ≥ 2)
- The estimated-π part of Task 3's hypothesis: reported as failed in round 1 (B11), not retested.
- Sensitivity π = 5 % (seed 0), PU-π5 − PU pixel AUROC: Brain -0.39, Liver -30.35, Retina_RESC -24.88

## Rule: does an arm beat MVFA (re-run under the same rule)?

| arm | AUPRO Δ>0 pairs | sign-test p | pix higher on targets | img within 0.5 on all | seeds 0–2 result | seeds 3–4 |
|---|---|---|---|---|---|---|
| G (gated MIL) | 0/9 | 1.000 | 0/3 | no | does not beat MVFA | – |
| PU (π=1%) | 3/9 | 0.910 | 1/3 | no | does not beat MVFA | – |
| G+100% | 5/9 | 0.500 | 1/3 | no | does not beat MVFA | – |
| PU+100% | 7/9 | 0.090 | 3/3 | no | suggestive | – |

Paired AUPRO Δ vs MVFA (Brain s0,s1,s2 | Liver | RESC):
- G (gated MIL): -1.36 -11.99 -5.21 -8.71 -0.87 -4.36 -10.35 -6.26 -2.87
- PU (π=1%): +0.89 +10.59 +12.14 -4.58 -6.65 -15.27 -15.47 -36.35 -25.80
- G+100%: +2.40 +8.42 +2.43 -0.74 +2.16 -1.72 +1.25 -2.04 -0.23
- PU+100%: -1.75 +5.13 +1.94 +0.26 +0.96 +1.15 +2.61 -2.52 +4.25

## Retention of MVFA's localisation (mask-free arms)

(pix − 50) / (MVFA re-run − 50), mean over seeds; [vs R0 in brackets, round 1's definition]

| arm | Brain | Liver | Retina_RESC | mean |
|---|---|---|---|---|
| MIL | 1.088 [1.051] | 0.987 [0.964] | 0.716 [0.671] | 0.930 [0.895] |
| PU (π=1%) | 1.071 [1.034] | 0.901 [0.880] | 0.351 [0.329] | 0.774 [0.748] |
| G (gated MIL) | 0.951 [0.919] | 0.962 [0.940] | 0.828 [0.776] | 0.914 [0.878] |

## Task 3's diagnostic for #1: share of top-k patches inside the true mask (masked sources' valid/ abnormal images)

| arm | by p | by p·g |
|---|---|---|
| MVFA (re-run) | 0.729 | 0.659 |
| MIL | 0.426 | 0.439 |
| PU (π=1%) | 0.267 | 0.306 |
| G (gated MIL) | 0.400 | 0.402 |
| PU+100% | 0.715 | 0.644 |
| G+100% | 0.696 | 0.624 |

Arms qualifying for confirmation seeds: none
