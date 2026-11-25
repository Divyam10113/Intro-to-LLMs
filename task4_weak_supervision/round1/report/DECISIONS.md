# Task 4: deviations from the plan, fixed before any GPU run

Written 2026-11-21, **before any GPU run and before any target-set result was
seen**. The only numbers looked at so far are: the π estimator's outputs on
source datasets (CPU smoke test, `validate_pi.py`), the true anomalous-patch
fractions of the masked source datasets (from their masks), and smoke-test
metrics on 16 Liver test images (meaningless at that size, not used for any
choice). The author delegated these choices to the LLM ("do as you feel
like, just ensure to document everything"); each row says why.

Everything not listed here follows `TASK4_PLAN_RECOMMENDED.md` unchanged
(targets, S = 2,000, λ = 1, the λ = 0.5 exception, seeds, subsampling, the
Step 5 rules, the early stop).

## D1. π for M-h: from the masked sources' masks, not the estimator

- **Plan:** estimate π_d for each mask-less source with the R0 segmentation branch (Step 2.1).
- **Problem:** the formula as written cannot work (issues log A4). Once fixed, it is correct on synthetic data, but on MVFA's real patch scores it returns about 1 (→ the 0.30 clip) at the deep levels. That happens even on masked sources, where the truth is known (Brain 0.010 vs ≈ 1.0; RESC 0.119 vs ≈ 1.0; issues log B11).
- **Decision:** for target T, every mask-less source gets the same
  **π_T = mean over T's masked sources of the true anomalous-patch fraction of their abnormal training images** (14×14-px cell anomalous if > 50% covered by the mask; subsampled train + π lists).
  - Liver target: (Brain 0.010 + RESC 0.119) / 2 ≈ 0.065.
  - Brain target: (Liver 0.008 + RESC 0.119) / 2 ≈ 0.064.
  - RESC target: (Liver 0.008 + Brain 0.010) / 2 ≈ 0.009.

  Exact values: `results/pi.json`, produced by `set_pi.py`.
- **Why:** it uses only masks that M-h already trains on (same images, same run). It never touches the target. And it is a real measured anomaly rate rather than a clip value.
- **Kept for the report:** the estimator's outputs (segmentation and detection branch, all sources) go to `results/pi_estimated_seg.json` and `pi_estimated_det.json`.
- **Sensitivity (secondary, reported separately):** M-h on Liver seed 0 with π = 0.30, i.e. what the plan's procedure effectively gives (`--pi_override 0.30 --tag pi0.30`).

## D2. π for M: fixed at 0.30, the value the plan's procedure returns

- **Plan:** π for M from R0's detection branch (Step 2.7).
- **Problem:**
  - that estimate is ≈ 1 → 0.30 for every source (B11);
  - R0's detection adapters were trained with mask gradients (B2);
  - and M may not use masks, so D1's approach is not allowed.
- **Decision:** **π = 0.30 for every source**, written to `results/pi_maskfree.json`.
- **Why:** this is exactly what the pre-registered procedure produces, so it is the least-choice option. It also uses no mask information, whereas a smaller constant would have been chosen after seeing the masked datasets' fractions.
- **Sensitivity (secondary):** M on Liver seed 0 with π = 0.05 (`--pi_override 0.05 --tag pi0.05`). The value was **chosen after seeing mask statistics**, so this run is not mask-free in spirit; report it only as a sensitivity check.

## D3. M and B0: detection adapters re-trained from scratch with image labels

- **Plan:** M and B0 keep R0's detection adapters, re-initialise only the segmentation adapters, and call that mask-free (citing Task 3 E6).
- **Problem:** in MVFA's training the detection adapters do receive mask gradients (issues log B2), so M as planned is not mask-free.
- **Decision (primary M / B0):** re-initialise **both** adapter sets (MVFA's initialisation, fixed seed 0, independent of the run seed) and train them jointly for S steps on pairs from all sources:
  - segmentation adapters: PU loss (M) or top-k MIL (B0);
  - detection adapters: **MVFA's detection loss, unchanged**, with image labels.

  CLIP stays frozen and no mask is ever loaded (enforced by the `MaskLeakError` assert).
- **Why:** it makes the plan's headline claim ("no mask is ever used") true at little cost (a few extra GPU minutes per run).
- **Secondary (as planned, reported separately):** M with R0's detection adapters, seed 0, all three targets (`--det_init r0 --tag r0det`). Report it as "segmentation adapters trained without masks on top of MVFA's (mask-influenced) detection adapters".
- **Consequence:** the retention rule (Step 5) compares a fully mask-free model with R0. M's image AUROC now comes from mask-free detection adapters trained for only S steps, so it may be lower than R0's. Report it.

## D4. MVFA's detection loss is replicated with its quirk

MVFA applies `BCEWithLogitsLoss` to the mean patch *probability* (a value in [0, 1]), i.e. it treats a probability as a logit (`train_zero.py:125-131`). D3 uses this loss exactly as MVFA does, since "MVFA's detection loss" is what the plan specifies and R0 was trained with it (issues log B12).

## D5. M-h-ctrl skips the PU iterations instead of stepping on λ = 0

- **Plan:** M-h-ctrl is M-h with λ = 0.
- **Decision:** M-h-ctrl runs the identical mask-loss updates in the identical order, and the PU iterations are skipped rather than run as zero-gradient steps. A zero-gradient step is not a no-op with Adam, because momentum keeps moving the weights (issues log B4).
- **Consequence:** M-h makes S updates (S/2 mask + S/2 PU) and M-h-ctrl makes S/2.

## D6. The π estimator itself (kept, corrected)

The estimator is corrected (lower tails; F_norm ∈ [0.10, 0.50]; 1-SE threshold choice, issues log A4). It is still run for every target in phase 0, so the report can show the B11 finding. It no longer sets π for any run (D1, D2).

## Run list after these decisions (≈ 3.5–4.5 GPU-hours)

| Phase | Runs |
|---|---|
| phase0 | `set_pi.py` (CPU); dry run; R0 × 3; estimator seg × 3, det × 3 and `validate_pi.py` × 3 (report only) |
| liver / rest | M-h, M-h-ctrl × seeds 0–2 × targets (or the early-stop subset), as planned |
| maskfree | M (3 targets s0 + Liver s1, s2), B0 (3 targets s0), fully mask-free (D3) |
| secondary | M-h Liver s0 π = 0.30; M Liver s0 π = 0.05; M with R0 detection adapters × 3 targets s0 |
| optional | λ = 0.5 exception; M at 2S on Liver s0 |
