# Round 3 GPU checks (written by gpu_checks.py)

## GPU smoke test (2026-11-23 10:57:03)

Each round 3 arm ran 50 steps on the Brain fold (seed 3, 1 epoch), with validation, the diagnostic and a 16-image target evaluation. Smoke runs are never written to results.csv.

| arm | exit | steps | loss first → last | finite | val AUROC | eval rows | s/step |
|---|---|---|---|---|---|---|---|
| mvfa | 0 | 50 | 6.354 → 1.354 | True | 0.775 | 1 | 0.105 |
| mil | 0 | 50 | 3.264 → 1.740 | True | 0.556 | 1 | 0.104 |
| pu_m100 | 0 | 50 | 8.341 → 2.935 | True | 0.678 | 1 | 0.106 |
| mil_m100 | 0 | 50 | 6.354 → 1.453 | True | 0.741 | 1 | 0.104 |
| mil_m10 | 0 | 50 | 6.354 → 1.532 | True | 0.775 | 1 | 0.104 |
| mil_m25 | 0 | 50 | 6.354 → 2.311 | True | 0.775 | 1 | 0.118 |

**Smoke test: PASS**

## Timing check (2026-11-23 10:57:40)

First 201 steps of each arm, 2 copies each, 12 concurrent jobs on 2 GPUs (round 2's G2-5 method). Steady s/step = the average over steps 50–200.

| arm | s/step (2 runs) | mean |
|---|---|---|
| mvfa | 0.0703, 0.0693 | 0.0698 |
| mil | 0.0751, 0.0660 | 0.0706 |
| pu_m100 | 0.0777, 0.0804 | 0.0791 |
| mil_m100 | 0.0761, 0.0686 | 0.0724 |
| mil_m10 | 0.0754, 0.0784 | 0.0769 |
| mil_m25 | 0.0673, 0.0788 | 0.0731 |

GPU memory per job (sampled during timing): max 4966 MiB, median 4966 MiB.

**Projection rule (fixed before any run, R3-4):** round-2-equivalent run-epochs = run-epochs × slowdown, where slowdown = mean over MVFA, MIL and PU+100 % of (measured s/step ÷ round 2's training s/step on the same Brain fold). The plan's 58 runs are 1160 run-epochs at round 2's speed; the cap is 1400 (4× round 2's 345, ≈ 6½ h).

Slowdown: mvfa 0.0698 / 0.0381 = 1.83, mil 0.0706 / 0.0434 = 1.63, pu_m100 0.0791 / 0.0458 = 1.73; mean **1.73**.
Per-epoch validation and fixed per-run minutes (for the wall-clock estimate) are round 2's medians per fold: Brain val 0.17 / fixed 1.42, Liver val 0.15 / fixed 0.57, RESC_OCT val 0.15 / fixed 0.61

- Cuts none: 58 runs = 1160 run-epochs × 1.73 = **2005** round-2-equivalent run-epochs (5.81× round 2); 6360 job-minutes (5.90× round 2's 1077); wall-clock on 12 slots ≈ 9.5 h
- Cuts ['mil_m25']: 49 runs = 980 run-epochs × 1.73 = **1694** round-2-equivalent run-epochs (4.91× round 2); 5339 job-minutes (4.96× round 2's 1077); wall-clock on 12 slots ≈ 8.3 h
- Cuts ['mil_m25', 'brain_extra']: 45 runs = 900 run-epochs × 1.73 = **1556** round-2-equivalent run-epochs (4.51× round 2); 4876 job-minutes (4.53× round 2's 1077); wall-clock on 12 slots ≈ 7.3 h
- Still above the cap after both cuts: the remaining runs are never cut (plan §4 step 4); running them anyway and logging the overrun.

**Decision: cut mil_m25 + brain_extra.** 45 runs, ≈ 7.3 h wall-clock if both GPUs stay free.


## Re-measurement on the real runs (2026-11-23 11:05), which replaces the decision above (R3-11)

The timing check's 1.73× slowdown compared **unlike things**:
- **today's number** is s/step over steps 50–200, measured while all 12 jobs were starting together (CLIP load, data reads, 14 % CPU I/O wait), about 15 s of steps;
- **round 2's number** comes from whole 4,742-step epochs.

Round 2's own timing check, over the same kind of startup window (ISSUES_LOG G2-5), also read 0.04–0.07 s/step. Today's 0.07–0.08 is therefore in line with round 2's startup.

The first full epochs of the 45-run queue (started 10:57:41) gave a fair comparison. "Epoch 1" below is the second epoch; epoch 0 includes startup in both rounds.

| run (12 concurrent jobs) | round 2, epoch 1 | round 3, epoch 1 | ratio |
|---|---|---|---|
| MVFA, Brain (r2 seeds 0–2; r3 seeds 3–5) | 178, 170, 178 s | 195, 195, 194 s | **1.11** |
| MVFA, RESC_OCT (r2 seed 0; r3 seeds 3–5) | 152 s | 175, 165, 165 s | 1.10 |
| MVFA, Liver (r3 seeds 3–5) | — | 195, 184, 194 s | — |
| MIL, Brain epoch 0 (r2 seeds 0–2; r3 seeds 3–5) | 202, 197, 220 s | 231, 230, 232 s | 1.11 |

With slowdown **1.11**, the plan's 58 runs are 1,160 × 1.11 = **1,288** round-2-equivalent run-epochs, under the 1,400 cap. Projected wall-clock is about 6.2 h on 12 slots.

**Decision: no cut.** All 58 runs. The teammate approved the restart (verbatim): *"yeah restart it with all 58 if it's not slower"*. The 45-run queue was stopped at 11:06 after one or two epochs. No result row had been written. Its 12 partial runs were moved to `/mnt/HDD1/divyam/task4/r3/aborted_2026-11-23_45run_queue/` and its logs to `logs/aborted_2026-11-23_45run_queue/`. The 58-run queue started at 11:06:36 from scratch. `logs/projection.json` was recomputed at 1.11× for `status.py`.
