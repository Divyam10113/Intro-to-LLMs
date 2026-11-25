# Round 2 verdict

Written by Claude Code (Opus 5.5) after all runs, as plan §7 requires. Every number is from
[`results/results.csv`](../results/results.csv) via [`analysis.md`](../results/analysis.md); points = AUROC/AUPRO × 100.

## A. Are these good results?

**Overall: No improvement, but informative.** Neither Task 3 idea beats plain MIL or MVFA under the pre-registered rules. The data show clearly why: the bank gate does not select more lesion patches, and PU is fragile with respect to π. The one robust finding is that plain image-label MIL nearly matches mask supervision.

| Arm | Verdict | Evidence |
|---|---|---|
| **G** (idea #1, gated MIL) | **Negative** | G − MIL pixel AUROC: Brain −5.26, Liver −1.15, RESC +4.25, which falsifies Task 3's hypothesis (needs ≥ +2 on all three). AUPRO is below MVFA in **9/9** paired runs. Its mechanism fails Task 3's own diagnostic: 0.40 of selected patches are inside lesions vs 0.43 for plain MIL. |
| **PU** (refined #3, π = 1 %) | **Negative** | PU − MIL pix: −0.67, −4.01, −13.83 (0/3 ≥ +1.5). It collapses on RESC (pix 63.3 ± 11.4). At π = 5 % it collapses on Liver (65.8) and RESC (49.1). |
| **G+100 %** | **No improvement, but informative** | Matches MVFA: AUPRO Δ > 0 in 5/9 pairs; pix 89.82 / 96.51 / 87.79 vs MVFA 88.32 / 96.80 / 87.92. Adding G on the mask-less sources neither helps nor hurts localisation. It lowers OCT17 image AUROC (45.2 / 45.9 / 64.7 vs 81.2 / 77.4 / 66.9). |
| **PU+100 %** | **Negative / inconclusive** (the rules call it "suggestive") | AUPRO Δ > 0 in **7/9** pairs (sign test p = 0.09). Pixel AUROC is higher on 3/3 targets, but only by +0.45 / +0.11 / +0.68. RESC image AUROC is 2.5 lower. The rule needs 8/9 plus confirmation seeds; it got neither. Noise cannot be ruled out. |
| **MIL** (baseline, not a Task 3 idea) | Strongest arm without masks (observation, not a pre-registered claim) | Retention of MVFA's pixel AUROC is 0.930. On Brain it beats MVFA on 3/3 seeds (AUPRO 70.6 ± 5.1 vs 61.6 ± 0.7; pix 91.7 vs 88.3). It is unstable on RESC (pix 76.9 / 67.4 / 87.1). |

**Compared with round 1:**
- **MIL without masks:** round 1 found that mask-free top-k MIL (B0) kept 0.967 of R0's localisation. Round 2 repeats this from scratch, against a re-run MVFA, over 3 seeds: 0.930 (0.895 against R0).
- **The Brain effect:** round 1 found a Brain-only AUPRO gain for PU (+5.2). Round 2 shows that gain is not specific to PU. Every image-label loss beats MVFA on Brain (MIL 70.6, PU 69.4, MVFA 61.6).
- **π:** round 1's mask-free PU collapsed at π = 0.30. Round 2 shows it also collapses at 0.05 and is unstable on RESC even at 0.01.

**What the mask-free arms keep:** retention (pix − 50) / (MVFA − 50) is MIL 0.930, G 0.914 and PU 0.774 (means over the 3 targets). **What each mask budget saves** (G, seed 0, report §3.5): with masks on 10 % of the masked sources' abnormal images, Liver AUPRO goes from 82.1 to 87.1, and to 89.0 at 25 %, against 90.1 at 100 % and MVFA's 90.8. That is 62 % and 87 % of the gap closed. On Brain, G+10 % (pix 90.24, AUPRO 66.4) already exceeds MVFA (88.54 / 62.3). Neither budget beats plain MIL with no masks (Liver AUPRO 88.1), so the budget mainly buys back what the gate lost. One seed only.

**The three strongest numbers**
1. MIL, no masks, Brain: AUPRO **70.6 ± 5.1 vs MVFA 61.6 ± 0.7**, higher on all 3 seeds. That is image-label training beating mask supervision on an unseen modality.
2. MIL retains **0.930** of MVFA's pixel AUROC without a single mask (Liver 96.19 vs 96.80).
3. PU+100 % is ≥ MVFA in AUPRO on **7/9** pairs and in pixel AUROC on 3/3 targets, the only consistent sign among the four tested arms (but see weakness 3).

**The three weakest or most worrying numbers**
1. **37/57 runs selected their last epoch** (54/57 one of the last two). The re-run MVFA trails the released checkpoint by 9–14 image AUROC (Brain 65.7 vs 79.8). Every arm is under-trained at 5 epochs. *Not shown:* whether the ranking holds at MVFA's schedule. The experiment that would show it: the same 4 arms at 20–50 epochs, at least Liver and Brain, 3 seeds.
2. **Chest image AUROC is below chance for every arm** (43.8–49.0, MVFA 44.6). G is below chance on OCT17 in 2 of 3 seeds. Idea #1's "image AUROC on Chest and HIS" condition is therefore weakly measured. *Not shown:* why Chest fails. The experiment: a Chest-fold run with full-size Chest training data, or per-epoch target image AUROC logged for analysis only.
3. PU+100 %'s margins are sub-point, and four arms were tested. Its p = 0.09 has no multiplicity correction, and it was not confirmed on seeds 3–4. *Not shown:* that it beats MVFA. The experiment: seeds 3–7 for PU+100 % and MVFA.

## B. Can this be published at a workshop?

**Answer: Submit as an honest negative / mixed-result paper.**

It cannot be a positive result: nothing beats MVFA under the rules. It is not "not ready" either: the evidence, the baseline and the process meet the checklist below. The paper should say plainly that the schedule is short (limitation 1). The paper's claim is a negative one: Task 3's two top ideas do not improve on plain MIL under Task 3's protocol, and plain MIL nearly matches mask supervision. A short schedule weakens that claim but does not invalidate it, because all arms share the schedule and the comparisons are paired.

| # | Checklist item | Status | Evidence |
|---|---|---|---|
| 1 | Claims match the evidence | **Yes** | Every claim in the summary and in this verdict cites a row of `analysis.md` or `per_seed.md`. The contribution sentence below is negative and backed by §3.3 of the report. |
| 2 | A fair baseline | **Yes, with a caveat** | MVFA was re-run under the same rule (from scratch, same data, same epochs, same selection), with 3 seeds and paired Δ (§3.3). The caveat: it is a 5-epoch MVFA, below the released checkpoint (limitation 1). |
| 3 | Pre-registered rules applied as written | **Yes** | §6 was implemented in `analyze2.py` before any result and applied unchanged. Both ideas were falsified or unsupported by it, and PU+100 % was held to "suggestive". Operational details were fixed at G1 before any run (G1-15). |
| 4 | No selection on target test sets | **Yes** | Only `evaluate_target.py` reads test sets, and a test enforces this (`test_only_evaluate_target_reads_test_sets`). Each `run.json` records `eval_reads`. Checkpoints were chosen on source `valid/`, with test duplicates removed (G1-5). |
| 5 | Limitations stated | **Yes** | Report §5 covers: the RESC/OCT17 protocol difference, the 5-epoch schedule, the subsample, π fixed rather than estimated, a single MVFA codebase, and the published or R0 numbers being selected on the test set. It also adds Chest below chance and the saturated validation. |
| 6 | Reproducibility | **Partly** | Code, configs (`run_all.sh`), seeds, split lists, 41 tests and `results.csv` are in the repository. The trained adapters and maps are on the server HDD only (about 50 MB per run). Publishing them, or a release archive, would make this "yes". |
| 7 | The AI's role is disclosed | **Yes** | Claude Code wrote the code, ran it and wrote the report. The teammate's words are logged verbatim at G1, G2 and later (ISSUES_LOG), and the transcript is in `report/transcript/`. |
| 8 | A clear one-sentence contribution | **Yes** | See below. |

**Contribution (one sentence):** Under a leave-one-modality-out protocol with source-only model selection, neither of an LLM pipeline's two top-ranked ideas for training MVFA's segmentation adapters from image labels (normal-bank-gated MIL and fixed-prior nnPU) improves on plain top-k MIL, which by itself retains 93 % of mask-supervised MVFA's pixel AUROC and beats it on brain MRI.

This verdict judges the evidence and the paper's readiness only. No venue or deadline is assumed, because no call for papers was supplied.

## What to do next (ordered)

1. **Write it up as an honest negative / mixed-result paper.** Lead with the two falsified ideas and the MIL retention result, and put the short-schedule limitation in the abstract.
2. **If one more GPU session is available, rerun MVFA, MIL and PU+100 % at a longer schedule** (20+ epochs, Liver and Brain, 3 seeds). This is the single experiment most likely to change the conclusions (weakness 1).
3. **Run seeds 3–7 for PU+100 % and MVFA** to settle the "suggestive" 7/9 (weakness 3).
4. **Next Task 3 idea: #4, mosaic MIL (`A1-g2-o2`)**, then #2 (gated mosaic), as the plan prescribes when both ideas fail. The evidence for this choice:
   - plain MIL's top-k lands inside lesions only 43 % of the time, against 73 % for MVFA (diagnostic);
   - MIL is unstable on RESC (pix 67–87 across seeds).

   Both point at localisation precision, which mosaic spatial-contrast supervision targets directly. Bank gating (#1) did not help (40 %).
5. Archive the HDD run folders, with adapters and maps, alongside the repository for reproducibility (checklist item 6).
