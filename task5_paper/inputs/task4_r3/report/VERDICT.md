# Round 3 verdict (final round of Task 4)

Written by Claude Code (Opus 5.5) after all 58 runs, as plan §4 Session B step 3 requires. Every number is from
[`results/results.csv`](../results/results.csv) via [`analysis.md`](../results/analysis.md) and [`per_seed.md`](../results/per_seed.md).
Points = AUROC/AUPRO × 100. The rules were frozen before any run (`RULES.md`, `analyze3.py`, `report/FROZEN.sha256`).

## A. Are these good results?

**Overall: one robust positive, one useful "matches", and two retractions.**
- **The robust positive:** image-label MIL without masks nearly matches mask-supervised MVFA, at the rule's top grade.
- **The "matches":** combining MIL with MVFA's masks matches MVFA.
- **The retractions:** at 4× the training on fresh seeds, round 2's "MIL beats MVFA on Brain" and "PU+100 % beats MVFA" do not hold.

No arm *beats* MVFA under the rules.

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| **H1** (P1) | MIL without masks keeps MVFA's localisation | **Confirmed: NEARLY MATCHES** | Retention (pix − 50)/(MVFA − 50): Brain 1.019, Liver 0.964, RESC 0.924; mean **0.969**, ≥ 0.90 on every target. Round 1: 0.967 (1 seed). Round 2: 0.930. Now 3 seeds at 20 epochs. |
| **H2** (P2) | MIL beats MVFA on Brain | **Did not replicate** | AUPRO higher on 3/5 Brain seeds (−0.40, +2.85, +10.93, −6.95, +6.77), p = 0.50 (Holm 1.00). Mean pix Δ −0.81. Round 2's 70.6 vs 61.6 became 63.8 vs 59.3, and MIL's Brain AUPRO spans 52.6–67.8 over 5 seeds. |
| **H3** (P3) | PU+100 % beats MVFA | **Did not replicate** | AUPRO higher in 5/9 pairs (round 2: 7/9), p = 0.50 (Holm 1.00). Pix Δ +1.97 / −0.64 / +1.12. Image AUROC −2.92 on RESC. |
| **H4** | MIL+100 % beats MVFA | **MATCHES (does not beat)** | AUPRO higher in 6/9 pairs, p = 0.25 (Holm 0.51). Pix Δ **+3.63 / −0.88 / +2.74**: within −1 on every target. Ahead on all 6 Brain and RESC pairs and behind on all 3 Liver pairs, by < 1 AUPRO (post hoc). |
| **H5** | MIL+10 % matches MVFA | **TRAILS** | Pix Δ +4.79 / **−1.96** / +0.66. Liver misses the −1 limit. Per-seed Liver pix Δ is −0.87 / −1.19 / −3.80; seed 5 contributes most, but without it the mean would still be −1.03. AUPRO higher in 4/9 pairs. On Brain all 3 seeds are ahead by 10.9–16.3 AUPRO (post hoc). |

**Compared with rounds 1–2:**
- **P1 is now the one result that has held three times:**
  - round 1: 0.967 of R0, 1 seed, 5 epochs;
  - round 2: 0.930, 3 seeds, 5 epochs;
  - round 3: 0.969, 3 fresh seeds, 20 epochs.
  MIL's RESC instability in round 2 (pix 67–87) is gone (79–86).
- **P2 and P3 were not pre-registered** (P2) or were only "suggestive" (P3). Neither survives fresh seeds and a longer schedule.
- **P4 (a small mask budget recovers most of the gap) carries over only in part.** With plain MIL, 10 % of the masks lifts mean retention from 0.969 to 1.034 and passes MVFA on Brain and RESC. On Liver it trails (H5).

**The three strongest numbers**
1. **MIL, no masks: retention 0.969, and ≥ 0.92 on every target** (Liver pix 95.50 vs 97.21; RESC 83.40 vs 86.14; Brain 88.56 vs 87.84). It is pre-registered, confirmed, and consistent over three rounds.
2. **MIL+100 %: pixel AUROC within 0.9 of MVFA on Liver and ahead by 3.6 on Brain and 2.7 on RESC.** AUPRO is higher on 6/6 Brain and RESC pairs. Adding the image-label loss on the mask-less sources costs nothing measurable and may help off-Liver.
3. **Brain with 100 masks (MIL+10 %): AUPRO 73.3 ± 3.0 vs MVFA 59.3 ± 2.2,** ahead on every seed (pix 92.6 vs 87.8). This is post hoc and is not a test (see weakness 3).

**The three weakest or most worrying numbers**
1. **Target image AUROC fell with 4× training:**
   - MVFA Brain 65.7 → **56.3** (seeds 51.8–64.1, near chance), Liver 71.2 → 60.6, RESC 80.0 → 74.3;
   - source validation kept rising to 0.965, so source-validation selection does not track target image-level detection;
   - the released checkpoint (R0, selected on the test set) is at 79.8 on Brain.

   *Not shown:* whether this is overfitting to the sources or a property of the protocol. *The experiment:* log per-epoch target image AUROC for analysis only (never for selection) on one fold and 2 seeds.
2. **Two of round 2's three tested positives were noise or schedule artefacts.** H2 (5 seeds) and H3 (9 pairs) are both at p = 0.50. Any single-target, 3-seed effect from rounds 1–2 should be treated as unconfirmed. *Not shown:* that anything other than P1 is robust to seed choice.
3. **The Brain gains of the budget arms are large but post hoc,** and Brain is the target where per-seed spread is largest (MIL AUPRO 52.6–67.8). *Not shown:* that a 10 % budget beats MVFA on Brain under a pre-registered test. *The experiment:* MIL+10 % vs MVFA on Brain, seeds 8–12, with the 5/5 rule.

## B. Can this be published at a workshop?

**Answer: Yes, as a pre-registered, three-round study whose headline is the confirmed result (H1), with the retractions reported in full.**

The headline is no longer a negative-only claim. H1 was pre-registered, frozen and confirmed at the rule's top grade, after a schedule and seed change that removed two other positives. That makes it more credible, not less. H4 gives a second, weaker positive ("matches"). The retractions (H2, H3) and the trailing H5 go in one compact "what did not hold" table, as plan §6 requires.

| # | Checklist item | Status | Evidence |
|---|---|---|---|
| 1 | Claims match the evidence | **Yes** | Each claim in `PAPER_CLAIMS.md` names its `analysis.md` row and its §5 outcome. Post-hoc observations are labelled as such and are not used as claims. |
| 2 | A fair baseline | **Yes, with a caveat** | MVFA is re-run under the same rule: from scratch, same data, same 20 epochs, same selection, paired by seed. The caveat: its target image AUROC is far below the released checkpoint's (weakness 1). |
| 3 | Pre-registered rules applied as written | **Yes** | `RULES.md` and `analyze3.py` were hashed before any GPU run (R3-9); `go.sh` checked the hashes at launch, and they still match. Two positives were retracted by the rules. One operational slip (the timing-check cut, R3-11) was caught and reversed before any result existed, under the same frozen cut rule, and is logged. |
| 4 | No selection on target test sets | **Yes** | Only `evaluate_target.py` reads test sets. Each `run.json` lists exactly its own target(s). Checkpoints were chosen on source `valid/`, with test duplicates removed (round 2's G1-5). |
| 5 | Limitations stated | **Yes** | Report §5: schedule and subsample, source-validation selection, 3 seeds, the confound in the round comparison, Chest/HIS not rerun, a single codebase, heat maps illustrative only. |
| 6 | Reproducibility | **Partly** | Code, configs, seeds, 80 tests, frozen rules, `results.csv` and the scripts for the paper tables and figures are in the repository. Adapters and maps (3.5 GB) are on the server HDD only. Archiving them would make this "yes". |
| 7 | The AI's role is disclosed | **Yes** | Claude Code wrote the code changes, rules, launcher, analysis and reports. The teammate's words are logged verbatim (ISSUES_LOG R3-3, R3-3b, R3-11, G2). The transcript is in `report/transcript/`. |
| 8 | A clear one-sentence contribution | **Yes** | See below. |

**Contribution (one sentence):** Under a leave-one-modality-out protocol with source-only model selection and pre-registered rules, plain top-k MIL trained only on image labels retains 97 % of mask-supervised MVFA's pixel AUROC on unseen modalities (≥ 92 % on each). This holds across three rounds, two schedules and fresh seeds, while two LLM-proposed refinements and two earlier apparent wins do not survive replication.

No venue or deadline is assumed, because no call for papers was supplied.

## What to do next (ordered)

1. **Write the paper around H1, with H4 as the secondary result** and the "what did not hold" table (H2, H3, H5, plus round 2's falsified ideas #1 and #3). Put the source-selection limitation (weakness 1) in the abstract. `PAPER_CLAIMS.md` has the structure, and `paper_inputs/task4_r3/` has the tables and figures.
2. **If one more GPU session is available:**
   - pre-register MIL+10 % vs MVFA on Brain (seeds 8–12, 5/5 rule) to test the largest post-hoc effect (weakness 3);
   - log per-epoch target image AUROC, for analysis only (weakness 1).
3. **Archive the HDD run folders** (`/mnt/HDD1/divyam/task4/r3/runs/`, 3.5 GB) with the repository (checklist item 6).
4. **Future work: idea #4, mosaic MIL.** MIL's top-3 patches land inside lesions 45 % of the time, against 79 % for MVFA, and only 16 % on the RESC+OCT17 fold. Localisation precision, not image-level ranking, is what masks add, and that is the gap #4 targets.
