# Experimental log — Round 1 (all result values from `results.json`)

## Framing and protocol

Pre-registered rule §5.1 selects an **analysis paper (F-C)**. Neither the hybrid PU fine-tune nor the fully mask-free PU model meets the registered rule. B0 is an informative, unregistered observation. Do not describe these results as an improvement over MVFA.

The study uses the BMAD leave-one-modality-out benchmark with 6 datasets. Targets are Brain, Liver, Retina_RESC; the source pool includes 3 datasets with pixel masks and 3 without them. Each source is subsampled to at most 500 normal and 500 abnormal training images, with a held-out 10% subset for prior estimation. The backbone is frozen CLIP ViT-L/14 with 4 adapter levels. Main runs use 2000 updates and seeds 0, 1, 2. Pixel metrics use AUPRO integrated through FPR 0.30.
Model selection in the released R0 checkpoints used the target test set; these fine-tunes use their last training step and do not evaluate the target during training.

## Table 1 source — M-h versus the matched control

Values are mean ± sample standard deviation across seeds; R0 is shown separately as published and reproduced.

| Target | Model | Image AUROC | Pixel AUROC | AUPRO |
|---|---|---:|---:|---:|
| Liver | R0 (published) | 76.24 | 97.85 | — |
| Liver | R0 (reproduced) | 81.19 | 97.92 | 92.34 |
| Liver | M-h | 80.29 ± 0.63 | 96.97 ± 0.18 | 91.39 ± 0.27 |
| Liver | M-h-ctrl | 81.63 ± 0.37 | 96.99 ± 0.31 | 91.07 ± 0.41 |
| Brain | R0 (published) | 78.63 | 90.27 | — |
| Brain | R0 (reproduced) | 79.80 | 89.68 | 65.35 |
| Brain | M-h | 80.25 ± 0.72 | 91.71 ± 0.71 | 71.57 ± 2.07 |
| Brain | M-h-ctrl | 80.31 ± 0.96 | 90.05 ± 0.74 | 66.38 ± 2.12 |
| Retina_RESC | R0 (published) | 83.31 | 92.05 | — |
| Retina_RESC | R0 (reproduced) | 88.99 | 90.44 | 68.75 |
| Retina_RESC | M-h | 89.20 ± 0.16 | 88.76 ± 2.69 | 67.45 ± 6.15 |
| Retina_RESC | M-h-ctrl | 88.92 ± 0.08 | 89.76 ± 0.46 | 66.94 ± 1.14 |

Paired M-h minus M-h-ctrl AUPRO was positive in 6 of 9 matched runs (one-sided sign-test p = 0.2539). Mean paired AUPRO differences were +5.20 on Brain, +0.33 on Liver, and +0.51 on RESC. The pre-registered criterion is not met. The Brain result is target-specific: AUPRO improved in all 3 paired Brain seeds; RESC is mixed and Liver is near zero.

## Table 2 source — fully mask-free models

M and B0 reinitialize both adapter sets; M uses fixed π = 0.30 and B0 uses top-k MIL. Values outside Liver are single-seed observations.

| Target | Model | Image AUROC | Pixel AUROC | AUPRO | Pixel retention vs R0 | AUPRO/R0 |
|---|---|---:|---:|---:|---:|---:|
| Liver | M | 73.79 ± 2.52 | 59.45 ± 15.51 | 26.28 ± 15.50 | -0.002 | 0.188 |
| Liver | B0 (single seed) | 75.17 (single seed) | 94.42 (single seed) | 85.47 (single seed) | 0.927 | 0.926 |
| Brain | M (single seed) | 68.87 (single seed) | 89.87 (single seed) | 64.50 (single seed) | 1.005 | 0.987 |
| Brain | B0 (single seed) | 63.51 (single seed) | 89.46 (single seed) | 64.53 (single seed) | 0.994 | 0.987 |
| Retina_RESC | M (single seed) | 84.53 (single seed) | 80.13 (single seed) | 58.52 (single seed) | 0.745 | 0.851 |
| Retina_RESC | B0 (single seed) | 76.98 (single seed) | 89.57 (single seed) | 67.29 (single seed) | 0.978 | 0.979 |

M's mean seed-0 pixel-retention across targets is 0.583; B0's is 0.966. M exceeds B0 in pixel AUROC and AUPRO on 0 of 3 targets. This evidence does not meet the label-efficiency framing rule. B0's 0.93–0.99 retention range is an unregistered single-seed observation.

## Why π estimation failed

The corrected lower-tail estimator was evaluated against known patch fractions on masked source data. True fractions in `results.json` range from 0.008 to 0.119. At the deepest segmentation level, the corresponding estimates range from -0.286 to 1.000 (reported raw, before clipping), far from the true rates in several cases. This violates the irreducibility assumption needed for mixture-proportion estimation on these MVFA patch-score distributions. M-h therefore used the mean true patch fraction from masked sources (D1), while mask-free M used the pre-registered fixed π = 0.30 (D2); these are different information regimes.

## Secondary sensitivity runs

Sensitivity runs are kept separate and are not included in means or pre-registered decisions. Their π overrides were selected after inspecting mask statistics. For example, M on Liver with π = 0.05 produced pixel AUROC {next(x['pix_auroc'] for x in round1['secondary'] if x['run_id']=='m_Liver_s0_pi0.05'):.2f} and AUPRO {next(x['aupro'] for x in round1['secondary'] if x['run_id']=='m_Liver_s0_pi0.05'):.2f}; this is post-hoc evidence only.

## Limitations and disclosure

The study is zero-shot and uses a subsampled training pool. M-h and its control have three seeds; M and B0 have a single seed outside Liver, and the Liver pixel metric for M has high seed variability. R0 inherits target-test checkpoint selection bias. RESC is evaluated while OCT2017 is in the source pool, so they share a retinal OCT modality. Pixel AUROC on Liver is dominated by the large normal background. The π overrides in secondary runs are post-hoc. Round 2 is absent and must remain a separate experiment if added later.
The work used an AI-augmented pipeline across literature review, limitation extraction, ideation, code implementation, and manuscript drafting; human verification is recorded in the supplied Task 4 transcript and the paper verification log.
