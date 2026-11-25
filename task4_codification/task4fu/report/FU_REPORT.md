# Task 4 follow-up report: MIL+10 % vs MVFA on Brain (seeds 8–12)

**Outcome under the frozen rule (`RULES_FU.md`, applied by `analyze_fu.py`): CONFIRMED.**
AUPRO higher on **5/5** pre-registered seeds (one-sided sign test p = 0.031), mean pixel AUROC Δ **+5.35**.

- Run 2026-11-24, 20:37:29–21:38:02 (about 61 min), 10 runs, 5 per GPU, 0 failures, 0 retries.
- Source: [`results/results.csv`](../results/results.csv) (10 rows, one per run) and [`results/analysis_fu.md`](../results/analysis_fu.md) (written by the frozen `analyze_fu.py`). Every number below is copied from those two files. Points = AUROC/AUPRO × 100.
- Frozen-hash check: `RULES_FU.md` and `analyze_fu.py` matched `report/FROZEN.sha256` before launch (`go.sh`) and again after the analysis.

## 1. What was tested and why

Round 3 found, **post hoc**, that MIL with masks for 10 % of the masked sources' abnormal images (`mil_m10`) beat the re-run MVFA on Brain AUPRO on all 3 seeds (3–5). That could not be claimed. This follow-up tests it once, under a rule fixed before any run (FU-2), on seeds 8–12, which no earlier round used. The code, data subsample, 20-epoch schedule, source-validation checkpointing and evaluation are round 3's, byte for byte (FU-1, checked by a test).

## 2. Every run (Brain fold, evaluated on the Brain test set)

| seed | arm | img AUROC | pix AUROC | AUPRO | best epoch (0-based) | val img AUROC | in-mask p | job min |
|---|---|---|---|---|---|---|---|---|
| 8 | MVFA | 53.71 | 85.76 | 53.97 | 15 | 97.17 | 0.93 | 57.3 |
| 8 | MIL+10 % | 60.59 | 91.33 | 68.04 | 12 | 96.63 | 0.79 | 60.4 |
| 9 | MVFA | 56.97 | 86.97 | 57.25 | 17 | 96.96 | 0.93 | 57.5 |
| 9 | MIL+10 % | 68.68 | 92.01 | 70.53 | 14 | 96.59 | 0.80 | 60.4 |
| 10 | MVFA | 70.02 | 87.45 | 58.74 | 15 | 97.47 | 0.93 | 57.4 |
| 10 | MIL+10 % | 70.25 | 92.56 | 73.37 | 13 | 97.21 | 0.77 | 59.8 |
| 11 | MVFA | 76.96 | 87.39 | 58.14 | 13 | 97.39 | 0.93 | 56.7 |
| 11 | MIL+10 % | 66.06 | 93.93 | 77.20 | 17 | 97.24 | 0.79 | 59.8 |
| 12 | MVFA | 58.80 | 87.44 | 58.80 | 17 | 97.23 | 0.93 | 56.6 |
| 12 | MIL+10 % | 56.26 | 91.94 | 71.27 | 14 | 97.23 | 0.81 | 59.8 |

All 20 epochs ran in every run (94,840 updates each). The selected checkpoint was never the last epoch (indices 12–17 of 0–19). Job-minutes: 585.6 in total (MVFA 285.5, MIL+10 % 300.1).

## 3. The pre-registered test (FU-H)

| seed | Δ AUPRO | Δ pix AUROC | Δ img AUROC |
|---|---|---|---|
| 8 | +14.07 | +5.58 | +6.88 |
| 9 | +13.28 | +5.04 | +11.71 |
| 10 | +14.63 | +5.11 | +0.23 |
| 11 | +19.07 | +6.54 | −10.90 |
| 12 | +12.47 | +4.50 | −2.54 |
| **mean** | **+14.70** | **+5.35** | **+1.08** |

- Δ AUPRO > 0 on 5/5 seeds → sign test p = 0.0312 ≤ 0.05. Mean Δ pix > 0. **Rule outcome: CONFIRMED.**
- The effect is not marginal. The smallest AUPRO gain is +12.47, and the seed-to-seed spread of each arm is 2–3.5 points. Pixel AUROC is higher on every seed too (+4.50 to +6.54), although the rule only needed the mean.

## 4. Descriptive (not tested)

**Means over seeds 8–12, with round 3's seeds 3–5 alongside (never pooled):**

| arm | seeds | img AUROC | pix AUROC | AUPRO |
|---|---|---|---|---|
| MVFA (re-run) | 8–12 | 63.29 ± 9.80 | 87.00 ± 0.72 | 57.38 ± 2.01 |
| MIL+10 % | 8–12 | 64.37 ± 5.83 | **92.36 ± 0.98** | **72.08 ± 3.44** |
| MVFA (re-run) | 3–5 (round 3) | 56.33 ± 6.73 | 87.84 ± 0.87 | 59.31 ± 2.15 |
| MIL+10 % | 3–5 (round 3) | 61.21 ± 6.40 | 92.63 ± 0.76 | 73.29 ± 2.99 |

The new seeds reproduce round 3's numbers closely. MIL+10 % is within 0.3 pix and 1.2 AUPRO of its round 3 mean, and MVFA within 0.9 pix and 1.9 AUPRO.

- **Image AUROC is not better.** Mean Δ +1.08, with 3 seeds up and 2 down (−10.90 on seed 11). Both arms stay far below the MVFA paper's 78.63 on Brain. The confirmed gain is in localisation only.
- **Against the published model (different protocol: context only).** MIL+10 % Brain pixel AUROC is above the MVFA paper's 90.27 on 5/5 seeds (mean 92.36, +2.09). AUPRO is above R0's 65.35 (round 1's measurement of the released checkpoint) on 5/5 seeds (mean 72.08, +6.73). The paper trains on full data for 50 epochs and selects on the test set; our runs use the subsample, 20 epochs and source-validation selection.
- **In-mask diagnostic.** This is the share of top-3 patches inside the true mask, on the source `valid/` abnormal images. MVFA scores 0.93 on every seed; MIL+10 % scores 0.77–0.81. So full masks concentrate the top-k on source lesions more, yet transfer worse to unseen Brain MRI. One reading is that full-mask training fits the source modalities' lesion appearance more tightly. That is a hypothesis; nothing here tests it.

## 5. What this does and does not support

**Supports, per `RULES_FU.md`, "What the paper may say":**
> With masks for 10 % of the abnormal source images and image labels elsewhere, MVFA's adapters localise better than full-mask MVFA on unseen brain MRI (AUPRO higher on 5/5 pre-registered seeds, p = 0.031).

**Does not support:**
- **Any claim beyond Brain.** In round 3, MIL+10 % *trailed* MVFA on Liver (H5, pix Δ −1.96), and RESC was not tested here. The claim is target-specific.
- **"Beats the published MVFA."** The comparison with 90.27 / 65.35 crosses protocols and stays descriptive.
- **Better image-level detection.** See §4.
- **Why it happens.** The in-mask contrast is suggestive only.

## 6. Deviations and incidents (also in `ISSUES_LOG.md`)

- **FU-5 (GPU wait skipped).** `go.sh` was in its free-GPU wait (3 checks, 60 s apart) when the teammate asked to skip it. It was stopped and relaunched with `FORCE=1` at 20:36:57. Tests ran again and passed (59). Both GPUs had been checked empty just before (0 % load, at most 161 MiB used). Nothing else changed.
- **FU-6 (≈550 MB contexts on GPU 0).** Each of the 5 GPU‑1 runs also held an idle CUDA context of about 550 MB on GPU 0. The cause is `pin_memory=True` with no `torch.cuda.set_device`, so the pin-memory thread initialises `cuda:0`. It has no effect on results, and round 3 had the same behaviour (same code). It was left unchanged because `train2.py` must stay identical to round 3.
- The GPU smoke test passed (both arms, 50 steps, `run.json` written, no row added to `results.csv`).
