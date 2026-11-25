# Task 4 analysis (Step 5 rules)

## (a) M-h vs M-h-ctrl

| target | config | n | img AUROC | pix AUROC | AUPRO |
|---|---|---|---|---|---|
| Liver | R0 (paper) | – | 76.24 | 97.85 | – |
| Liver | R0 (reproduced) | 1 | 81.19 | 97.92 | 92.34 |
| Liver | mh | 3 | 80.29 ± 0.63 | 96.97 ± 0.18 | 91.39 ± 0.27 |
| Liver | mh_ctrl | 3 | 81.63 ± 0.37 | 96.99 ± 0.31 | 91.07 ± 0.41 |
| Liver | π used (D1) | | Retina_OCT2017: 0.065, Chest: 0.065, Histopathology: 0.065 | | |
| Liver | π estimator, raw (report only) | | Retina_OCT2017: 1.000, Chest: 0.100, Histopathology: 0.986 | | |
| Brain | R0 (paper) | – | 78.63 | 90.27 | – |
| Brain | R0 (reproduced) | 1 | 79.80 | 89.68 | 65.35 |
| Brain | mh | 3 | 80.25 ± 0.72 | 91.71 ± 0.71 | 71.57 ± 2.07 |
| Brain | mh_ctrl | 3 | 80.31 ± 0.96 | 90.05 ± 0.74 | 66.38 ± 2.12 |
| Brain | π used (D1) | | Retina_OCT2017: 0.064, Chest: 0.064, Histopathology: 0.064 | | |
| Brain | π estimator, raw (report only) | | Retina_OCT2017: 0.949, Chest: -0.058, Histopathology: 0.756 | | |
| Retina_RESC | R0 (paper) | – | 83.31 | 92.05 | – |
| Retina_RESC | R0 (reproduced) | 1 | 88.99 | 90.44 | 68.75 |
| Retina_RESC | mh | 3 | 89.20 ± 0.16 | 88.76 ± 2.69 | 67.45 ± 6.15 |
| Retina_RESC | mh_ctrl | 3 | 88.92 ± 0.08 | 89.76 ± 0.46 | 66.94 ± 1.14 |
| Retina_RESC | π used (D1) | | Retina_OCT2017: 0.009, Chest: 0.009, Histopathology: 0.009 | | |
| Retina_RESC | π estimator, raw (report only) | | Retina_OCT2017: 1.000, Chest: -0.017, Histopathology: 0.941 | | |

**Paired Δ = M-h − M-h-ctrl (points)**

| target | seed | Δ pix AUROC | Δ AUPRO |
|---|---|---|---|
| Brain | 0 | +1.73 | +5.42 |
| Brain | 1 | +1.66 | +5.35 |
| Brain | 2 | +1.58 | +4.82 |
| Liver | 0 | +0.20 | +0.66 |
| Liver | 1 | -0.15 | -0.13 |
| Liver | 2 | -0.12 | +0.44 |
| Retina_RESC | 0 | -1.82 | -1.92 |
| Retina_RESC | 1 | +1.54 | +6.27 |
| Retina_RESC | 2 | -2.72 | -2.83 |

AUPRO Δ > 0 in **6/9** pairs (one-sided sign test p = 0.254); mean Δ > 0 on 3/3 targets; overall mean Δ AUPRO = +2.01, pix AUROC = +0.21.

Rule: **No effect beyond fine-tuning** (negative result).

**Early-stop check (Liver, 3 seeds):** Δ AUPRO ['+0.66', '-0.13', '+0.44'], Δ pix AUROC ['+0.20', '-0.15', '-0.12'] → continue: run `./run_all.sh rest` (3 seeds on Brain and RESC).

M-h vs R0 on Liver: pix -0.96, AUPRO -0.94, img -0.90 → does not
M-h vs R0 on Brain: pix +2.02, AUPRO +6.22, img +0.45 → exceeds
M-h vs R0 on Retina_RESC: pix -1.68, AUPRO -1.30, img +0.20 → does not

Rule: Does not exceed R0 on ≥ 2 of 3 targets.


## (b) Mask-free: M and B0

| target | M pix (s0) | B0 pix | M AUPRO (s0) | B0 AUPRO | retention (M−.5)/(R0−.5) | AUPRO M/R0 | M img |
|---|---|---|---|---|---|---|---|
| Liver | 49.90 | 94.42 | 17.33 | 85.47 | -0.002 | 0.188 | 74.09 |
| Brain | 89.87 | 89.46 | 64.50 | 64.53 | 1.005 | 0.987 | 68.87 |
| Retina_RESC | 80.13 | 89.57 | 58.52 | 67.29 | 0.745 | 0.851 | 84.53 |

Liver M over 3 seeds: pix 59.45 ± 15.51, AUPRO 26.28 ± 15.50. Other targets single-seed.

Mean retention = 0.583; M > B0 (pix and AUPRO) on 0/3. Rule: **Masks still matter.** Report retention per target and the AUPRO ratio.

Possibly under-trained (loss still falling): b0_Liver_s0, b0_Brain_s0

## Secondary runs (not used by any Step 5 rule; DECISIONS D1-D3, plan exceptions)

| run | pix AUROC | AUPRO | img AUROC | vs primary (pix / AUPRO) | notes |
|---|---|---|---|---|---|
| m_Retina_RESC_s0_r0det | 82.11 | 60.19 | 87.25 | +1.98 / +1.67 | det_init=r0 (R0 det adapters: not strictly mask-free); GPU shared by up to 6 jobs: gpu_minutes is wall time on a shared card |
| mh_Liver_s0_pi0.30 | 95.99 | 89.59 | 80.22 | -0.89 / -1.67 | loss still falling (last 10% < 0.95x previous 10%): possibly under-trained; pi_override=0.3 (sensitivity); GPU shared by up to 6 jobs: gpu_minutes is wall time on a shared card |
| m_Liver_s0_r0det | 38.01 | 13.39 | 74.58 | -11.89 / -3.94 | loss still falling (last 10% < 0.95x previous 10%): possibly under-trained; det_init=r0 (R0 det adapters: not strictly mask-free); GPU shared by up to 6 jobs: gpu_minutes is wall time on a shared card |
| m_Liver_s0_pi0.05 | 97.14 | 87.28 | 69.28 | +47.23 / +69.95 | det_init=fresh; pi_override=0.05 (sensitivity); GPU shared by up to 6 jobs: gpu_minutes is wall time on a shared card |
| m_Brain_s0_r0det | 87.02 | 57.23 | 81.86 | -2.85 / -7.26 | loss still falling (last 10% < 0.95x previous 10%): possibly under-trained; det_init=r0 (R0 det adapters: not strictly mask-free); GPU shared by up to 6 jobs: gpu_minutes is wall time on a shared card |
| m_Liver_s0_2S | 59.88 | 36.19 | 63.41 | +9.98 / +18.86 | det_init=fresh; GPU shared by up to 6 jobs: gpu_minutes is wall time on a shared card |
