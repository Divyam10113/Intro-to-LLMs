# Round 3 rules (frozen before any round 3 run)

The plan's §5 (`Plans/TASK4_ROUND3_PLAN.md`) written out exactly as `analyze3.py` applies it.
Both files are hashed into `report/FROZEN.sha256` and `report/ISSUES_LOG.md` (R3-5) before any GPU run.
`tests/test_r3.py::test_rules_files_match_frozen_hashes` fails if either file changes afterwards.
Any later change must be logged as a new ISSUES_LOG row with its reason.

## Definitions

- **Points:** AUROC or AUPRO × 100.
- **Targets:** Brain, Liver and RESC (from the RESC + OCT17 fold). OCT17 image AUROC is descriptive only.
- **Pair:** (target, seed) over seeds 3, 4 and 5, giving 9 pairs. A pair counts only when both arms have a row for it.
- **Paired Δ:** arm − MVFA (re-run in round 3) on the same pair. A pair counts as "higher" only if Δ > 0, so ties count against the arm.
- **Means:** means over seeds 3–5 per target, unless a rule says otherwise.
- **Retention:** (pix − 50) / (MVFA pix − 50) per target, using means over seeds 3–5 of this round's MVFA.
- **Sign test:** one-sided binomial with p = ½: P(X ≥ pos | n). Over 9 pairs, 8/9 gives p = 0.020 and 9/9 gives p = 0.002. Over 5 seeds, 5/5 gives p = 0.031.
- **Holm correction:** standard Holm step-down within each tier. The Tier 1 family is {H2, H3}; the Tier 2 family is {H4, H5's "beats" part}. Holm runs on the sign-test p-values only. Raw and adjusted p are both reported.
  - With two hypotheses, the smaller p must be ≤ 0.025 and the larger ≤ 0.05.
  - H2's best possible p is 0.031, so H2 can pass only at the second step, i.e. when H3's p is ≤ 0.025.
- **Missing runs:** if a run failed twice, it is reported, and the thresholds scale with the pairs present. "8/9" becomes ⌈8n/9⌉ and "7/9" becomes ⌈7n/9⌉.
- **Rows used:** rows marked SMOKE, and rows whose run_id seed disagrees with the seed column, are ignored.

## Tier 1: confirm

| | Hypothesis | Outcome rule |
|---|---|---|
| **H1** (P1) | MIL without masks keeps most of MVFA's localisation | Retention ≥ 0.90 on all 3 targets → **NEARLY MATCHES**. Otherwise, mean retention ≥ 0.75 → **KEEPS MOST**. Otherwise **FALLS SHORT**. Any target with retention < 0.60 is named. This is an estimate with no test and no correction. |
| **H2** (P2) | MIL beats MVFA on Brain | Uses the Brain seeds 3–7 that have both MIL and MVFA. **CONFIRMED** if all of: n = 5; AUPRO higher on 5/5 seeds; mean Brain pixel AUROC Δ > 0 (same seeds); Holm-adjusted p ≤ 0.05. Otherwise, if pos/n ≥ 0.8 → **holds on most seeds (not confirmed)**; when pos = n, the text adds "(n/n seeds, p = … uncorrected)". Otherwise **did not replicate**. |
| **H3** (P3) | PU+100 % beats MVFA | **CONFIRMED** if all of: AUPRO higher in ≥ 8/9 pairs; Holm-adjusted p ≤ 0.05; mean pixel AUROC higher on ≥ 2/3 targets; mean image AUROC not more than 0.5 below MVFA's on any target. Otherwise, ≥ 7/9 pairs → **suggestive again**. Otherwise **did not replicate**. |

## Tier 2: extend

| | Hypothesis | Outcome rule |
|---|---|---|
| **H4** | MIL+100 % beats MVFA | H3's rule, with Holm over {H4, H5}. If it passes → **SUPPORTED (beats MVFA)**. Otherwise, mean pixel Δ ≥ −1 on every target → **MATCHES**. Otherwise **TRAILS**. |
| **H5** | MIL with 10 % of the masks matches MVFA | Mean pixel Δ (MIL+10 % − MVFA) ≥ −1 on every target → **SUPPORTED (matches MVFA)**. If it also meets H3's rule (Holm over {H4, H5}) → **BEATS MVFA**. Otherwise **TRAILS**. |

## Descriptive (no test)

- **Label-budget curve:** retention and AUPRO at 0 / 10 / 25 / 100 % masks (MIL, MIL+10 %, MIL+25 %, MIL+100 %), plus MVFA, per target.
- **Round 3 vs round 2:** MVFA, MIL and PU+100 % at 20 epochs on seeds 3–5 against 5 epochs on seeds 0–2. Shown side by side, never pooled.
- **Best-epoch histogram:** one count per run. If more than half of the runs pick their last epoch, the analysis states that the runs are still under-trained.
- **In-mask diagnostic:** round 2's, for every arm.
- **Low image AUROC:** every image AUROC below 50 is flagged.

## Interpretations fixed here (where the plan's wording leaves a choice)

1. **H1 seeds.** H1 uses seeds 3–5 on all 3 targets. The extra Brain seeds 6–7 are used only for H2, so every target has the same seed set.
2. **Holm runs on p-values only.** H3's non-p conditions (pixel and image) do not affect H2's Holm step. H2 passes the second step whenever H3's raw p ≤ 0.025, i.e. 8/9 or 9/9 pairs, even if H3 then fails a non-p condition.
3. **Fewer than 5 Brain seeds.** If the extra Brain seeds are cut (plan §4 step 4, cut b), H2 cannot be confirmed. It is reported as "holds on most seeds" if pos/n ≥ 0.8, otherwise "did not replicate".
4. **H5's "beats" outcome needs both conditions:** H3's rule, plus "matches" (mean pixel Δ ≥ −1 on every target). They can disagree: H3's rule needs the pixel AUROC to be higher on only 2/3 targets, so the third target can still be more than 1 point lower.
