# Task 4, round 2: refine round 1's idea and codify the next Task 3 idea, to beat MVFA

**What Task 4 is** (course brief):
- take the top-5 ideas from Task 3 and codify the best one with an LLM;
- find what blocks it (gaps in the spec, inconsistencies, compute) and give the LLM feedback;
- refine the idea;
- get results to compare with the baseline, and say where the idea stands.

This plan continues that loop. Round 1 codified one Task 3 idea. Round 2 does two things with Round 1's feedback:
1. **Refines round 1's idea** (`A1-g1-o6`, witness-rate / PU), fixing what round 1 found wrong with it.
2. **Codifies the next Task 3 idea** (`A1-g1-o2`, normal-bank-gated MIL), Task 3's rank-1 idea.

Both are tested under Task 3's own protocol, against MVFA re-run the same way. At the end Claude Code writes a **verdict** (§7): do the results hold up, and are they publishable at a workshop?

**Who runs it:** a teammate, with **Claude Code** as the LLM, on the same server as round 1 (2 × RTX PRO 6000 Blackwell). **Budget: about 2.5–3 wall-clock hours.**

Nothing here is a new idea: every method below is a Task 3 idea, used with the hyper-parameters and protocol Task 3 wrote for it.

**Naming:** the problem is **A1** throughout (train MVFA's segmentation adapters from image-level labels). "Idea #1", "#3" are ranks inside A1's top 5: **#1 = `A1-g1-o2`** (normal-bank-gated MIL, new this round), **#3 = `A1-g1-o6`** (witness-rate / PU, codified in round 1 and refined this round). Task 3's other problem, **B1** (per-image scoring for few-shot MVFA), is not used in Task 4.

```
Task 3 (A1 problem + top-5 ideas)
   │  choose idea #3 (Task 3's recommendation)
   ▼
Round 1: codify #3 → refine to PU → issues + feedback → results            (done, 2026-11-22)
   │  feedback (§3): the π estimator failed; the PU gain was real on Brain; plain MIL worked
   ▼
Round 2:  refine #3  ──►  PU with a fixed π, no estimator
          codify  #1  ──►  gated MIL
          both under Task 3's protocol, vs MVFA re-run the same way
   ▼
Verdict (§7): does it beat MVFA? is it publishable?
```

---

## 1. Starting point: Task 3's output

**Source:** branch `task3_kevalina`, `task3/outputs/stage4_ranking_A1.json` and `stage3_evolution_A1.json`, summarised in `Task3 top 5 ideas for A1.pdf`.

**Problem A1** (Task 3, stage 1):
- MVFA's segmentation adapters, the only part that draws anomaly maps, learn only from the three source datasets with pixel masks (Brain MRI, Liver CT, RESC).
- OCT17, Chest X-ray and histopathology train only the detection adapters.
- Question: can the segmentation adapters be trained from **image-level labels**, and how much of the mask-supervised localisation do they keep on the unseen modality?

**Task 3's protocol for every A1 idea:**
- leave-one-modality-out zero-shot training on BMAD, with RESC and OCT17 held out together;
- all source masks discarded; detection adapters and their loss unchanged;
- checkpoints selected on **source** validation by image AUROC;
- mask-supervised MVFA and plain top-k MIL re-run under the same rule;
- 3 seeds;
- pixel AUROC and AUPRO on Brain, Liver and RESC; image AUROC on all folds.

**Top 5, and what each one claims:**

| Rank | ID | Idea | Task 3 hypothesis (what would falsify it) |
|---|---|---|---|
| 1 | `A1-g1-o2` | **Normal-bank-gated MIL** | +2 pixel AUROC over plain top-k MIL on all three masked targets; image AUROC on Chest and HIS not down by > 0.5. Falsified if no pixel-AUROC gain over plain MIL |
| 2 | `A1-g3-o3` | Gated mosaic MIL | +1.5 over mosaic MIL and +3 over top-k MIL |
| 3 | `A1-g1-o6` | **Witness-rate-adaptive MIL (PU family)** | +1.5 pixel AUROC over the best fixed k ∈ {0.5 %, 1 %, 5 %}. Estimated π ranks the masked sources like their true lesion areas. Falsified if adaptive k does not beat the best fixed k |
| 4 | `A1-g2-o2` | Mosaic MIL | +2 pixel AUROC over consensus MIL |
| 5 | `A1-g3-o2` | Per-image adaptive k | +1.5 over per-dataset witness-rate k |

---

## 2. Round 1 in brief (done)

Full record: [`TASK4_REPORT.md`](../task4_share/task4/report/TASK4_REPORT.md), [`ISSUES_LOG.md`](../task4_share/task4/report/ISSUES_LOG.md) (29 issues with the feedback given), [`DECISIONS.md`](../task4_share/task4/report/DECISIONS.md) (D1–D6).

- **Idea chosen:** #3, as Task 3's summary recommended (cleanest, cheapest, one change against plain MIL).
- **Refinement:** witness rate → full nnPU loss (run 1's `A1-g2-o6`). It was run two ways: added to MVFA's mask training on the mask-less sources (M-h), and mask-free (M). Plain top-k MIL (B0) ran as the mask-free baseline.
- **Results:**
  - **M-h vs the same fine-tune without the PU term:** no consistent gain. AUPRO rose in 6/9 seed pairs: +5.2 on Brain (3/3 seeds), nothing on Liver, mixed on RESC.
  - **Mask-free PU (M):** collapsed on Liver (pixel AUROC 49.9 at π = 0.30). The same run at π = 0.05 reached 97.1, but that value was picked after seeing mask statistics, so it is not a valid result.
  - **Plain top-k MIL with no masks (B0):** kept **93–99 %** of MVFA's localisation.
  - **Task 3's diagnostic** (estimated π against true lesion fraction) failed: the estimator returns about 1.0 where the truth is 0.008–0.12 (B11).

---

## 3. From round 1's feedback to round 2

| Round 1 finding | Decision for round 2 |
|---|---|
| The π estimator does not work on MVFA's scores (B10, B11; D1, D2, D6). Task 3's hypothesis for #3 depends on it | **Refine #3, do not drop it:** no estimator. Use a **fixed π** from Task 3's own fixed-k grid (the idea's baseline): π = **1 %** as the main arm (the value the idea itself starts with, and plain MIL's k) and π = **5 %** as a sensitivity arm. This tests the PU loss, not the estimator |
| PU gave +5 AUPRO on Brain on all 3 seeds, but nothing elsewhere | Keep PU as a full arm with 3 seeds on every target, so the Brain effect gets a proper test |
| Fine-tuning from MVFA's released checkpoints (selected on the test set) made even the control lose (A5) | Train **from scratch** under Task 3's protocol, with source-validation checkpoints, for every arm |
| Plain MIL worked without masks (B0) | MIL is the baseline; **#1 gated MIL builds directly on it** and needs no π. It is Task 3's rank-1 idea, so it is codified this round |
| #4, #2 (mosaics) and #5 (per-image k) | #5's baseline needs the failed estimator; #4 and #2 stay for a later round if both arms are falsified |

**How round 2 can beat MVFA** also comes from Task 3:
- the Task 3 summary's recommendation 3: "keep the masks where they exist and add this loss on the three mask-less datasets";
- A1's analysis (iii): masks for 10 % and 25 % of abnormal source images.

So round 2 also runs both losses with all masks kept (100 %).

---

## 4. Steps

### Step 1: Spec check before coding (gate G1)
Gaps and ambiguities in the two specs, found by comparing them with round 1's code. Log each one in `task4r2/report/ISSUES_LOG.md` in round 1's format (*issue → how found → feedback → fix*). The proposed resolutions:

1. **"Level 4" features for the bank (#1).** Task 3's reviewer flagged that one level gates all four. Feedback: use frozen CLIP layer-24 features for the bank and gate every level with them, as the spec says.
2. **Bank size {5k, 20k} (#1).** Use 20k, the spec's main value. No sweep (compute).
3. **π for #3 (#3).** Use the fixed π above, set before any run; no estimator (B11).
4. **PU loss details (#3).** Reuse round 1's non-negative PU risk on the patch logit z = s₁ − s₀, with a normal/abnormal pair from one dataset per step (A1, A2).
5. **"Source validation."** The spec doesn't name the split. MVFA trains on the sources' `test/` folders (B1). Feedback: use the sources' BMAD `valid/` folders, after checking that no file appears in both.
6. **"Mask-free" detection adapters.** In MVFA they also receive mask gradients (B2, D3). Feedback: train both adapter sets from scratch, with the detection loss unchanged, including its quirk (B12, D4).
7. **RESC and OCT17 held out together.** This differs from round 1 and from MVFA's paper protocol. Feedback: follow Task 3, and mark the published numbers as a different protocol.
8. **"Released learning rate and epochs" (50 epochs).** That does not fit 3 hours. Feedback: **5 epochs** on round 1's subsample (≤ 1,000 images per source), checkpoint after each epoch chosen on source validation. Log it as the brief's compute issue.
9. **Hard negatives (weight 2 on normal top-k) (#1).** Keep, as in the spec.
10. **Seeds** set initialisation and data order (round 1 varied only data order).

### Step 2: Implement and test on CPU (gate G2)
- Reuse round 1's code (`task4/`): data, model wrapper, MIL and PU losses, AUPRO, tests, results CSV.
- New code:
  - `build_normal_bank.py`: per source dataset, frozen layer-24 patch features of ≤ 500 normal images, greedy coreset to 20,000 vectors;
  - `gated_mil()`: gate g = 1 − max cosine similarity to the bank (detached); select top-k by p × g; k = 1 %;
  - the mask-budget option (a fixed, seeded share of abnormal masked-source images keeps its masks; Dice + Focal on those only);
  - checkpoint selection on source validation by image AUROC;
  - `evaluate_target.py`, the only code that reads a target test set;
  - `analyze.py`, which applies §6 mechanically.
- CPU checks:
  - the gate is 0 for a patch identical to a bank entry;
  - selection indices carry no gradient;
  - PU: round 1's checks (hand calculation, π = 0 equals BCE);
  - no mask reaches the loss in mask-free runs (round 1's `MaskLeakError`);
  - no target file in any training, bank or validation list;
  - different seeds give different initial weights.

### Step 3: Run

| ID | What | Folds × seeds | Runs |
|---|---|---|---|
| **MVFA** | Mask-supervised MVFA, re-run under the same rule | Brain, Liver, RESC + OCT17 × 0–2; Chest, HIS × 0 | 11 |
| **MIL** | Plain top-k MIL, all masks discarded (round 1's B0, from scratch) | same | 11 |
| **PU** | Refined idea #3: PU with fixed π = 1 %, all masks discarded | same | 11 |
| **G** | Idea #1: gated MIL, all masks discarded | same | 11 |
| **PU-π5** | PU with π = 5 % (sensitivity, Task 3's fixed-k grid) | Brain, Liver, RESC + OCT17 × 0 | 3 |
| **PU+100 %**, **G+100 %** | Each loss on the mask-less sources + all masks (Task 3 recommendation 3): the main attempt at beating MVFA | Brain, Liver, RESC + OCT17 × 0–2 | 18 |
| Confirmation | Seeds 3–4 for MVFA and the one variant that met the "beats MVFA" rule. **Only if one did** | Brain, Liver, RESC + OCT17 | 12 |
| Budget points | G+10 %, G+25 % as Task 3 specifies. **Only if time is left** | Brain, Liver × 0 | 4 |
| R0 | MVFA's released checkpoints (round 1 numbers, reference only) | — | 0 |

**65 runs** by default (77 with confirmation, 81 with the budget points). OCT17's image AUROC comes from the RESC + OCT17 fold's model. Each run appends one row to `task4r2/results/results.csv` in round 1's format.

### Step 4: Read the results, then write the report and the verdict (gate G3)
- Apply §6's rules, which are fixed before any run.
- Report every number, including the losses: per-seed tables, mean ± std, paired Δ, and Liver heat maps (round 1's four images).
- The report follows round 1's structure:
  1. idea and refinement;
  2. issues log with feedback;
  3. results;
  4. rules applied;
  5. limitations.
- Then write `task4r2/report/VERDICT.md` as §7 requires.
- If both ideas are falsified, name #4 as the next idea, with the evidence.

---

## 5. The agentic loop (Claude Code)

One loop, with three stops where the teammate gives feedback:

```
spec check ──G1──► build ⇄ test (fix until the CPU checks pass, ≤ 10 tries) ──G2──► run all (background)
     ──► check each result (crash → retry once; NaN or AUROC < 0.5 → flag) ──► apply §6
     ──► if an arm "beats MVFA": run the confirmation seeds ──► write report + VERDICT ──G3──► done
```

- **G1:** the teammate approves or edits the Step 1 resolutions.
- **G2:** the teammate reviews the code and the test output before any GPU time.
- **G3:** the teammate reads the report and the verdict.
- Claude Code logs the teammate's words verbatim as the feedback, as the brief asks.

**Three rules the agent must not break:**
1. Target test sets are read only by `evaluate_target.py`, after training ends. Nothing is chosen on a target.
2. `results.csv` is append-only, and every run is reported.
3. No method that is not in Task 3's outputs.

---

## 6. Decision rules (pre-registered)

**Each idea against Task 3's own hypothesis**
- **#1 gated MIL supported:** G − MIL ≥ +2 pixel AUROC on all three masked targets (mean over seeds), and image AUROC on Chest and HIS not down by more than 0.5. Falsified if there is no pixel-AUROC gain over MIL.
- **#3 as refined (PU, fixed π):** PU − MIL ≥ +1.5 pixel AUROC on at least 2 of 3 masked targets (Task 3's threshold, against its fixed-k baseline). The estimated-π part of Task 3's hypothesis is reported as failed (B11), not retested.

**Beating MVFA** (each of G, PU, G+100 %, PU+100 % against MVFA re-run under the same rule; Δ paired by target and seed, 9 pairs)

| Result | Claim |
|---|---|
| AUPRO Δ > 0 in ≥ 8/9 pairs, pixel AUROC higher on ≥ 2/3 targets, image AUROC not lower by more than 0.5, **and it holds again on seeds 3–4** | "Beats MVFA" (sign test p ≈ 0.02); name the arm |
| 7/9, or not confirmed on seeds 3–4 | Suggestive |
| ≤ 6/9 | Does not beat MVFA |

Four arms are tested, so a claim needs the confirmation seeds; without them the best arm is "suggestive" at most.

**Published MVFA numbers and R0** are shown as reference only. MVFA selected them on the test set (A5), and RESC follows a different protocol (Step 1 item 7).

---

## 7. Verdict and workshop readiness (Claude Code must write this)

After the report, Claude Code writes `task4r2/report/VERDICT.md`, **in plain language, with one clear conclusion for each question**. It must be honest: no hype, and it must say plainly what is not shown.

**A. Are these good results?** One of four verdicts per arm and overall, each backed by the numbers:

| Verdict | Means |
|---|---|
| **Clear win** | Beats MVFA under §6 and confirmed on fresh seeds |
| **Partial win** | Beats MVFA on some targets or metrics only; say which |
| **No improvement, but informative** | Matches or trails MVFA, with a clear explanation from the data |
| **Negative / inconclusive** | No effect, or the evidence cannot separate it from noise |

It must also say:
- how it compares with round 1's result;
- what the mask-free arms keep of MVFA's localisation (retention, as in round 1) and what each mask budget saves;
- the three strongest numbers, and the three weakest or most worrying ones.

**B. Can this be published at a workshop?** The brief (Task 5) says that with good results and analysis the paper can go to a workshop on AI-generated papers; it is optional. Claude Code answers with exactly one of:
- **Submit as a positive result,**
- **Submit as an honest negative / mixed-result paper,**
- **Not ready**, with the specific missing items.

Then it justifies the answer against this checklist, one line each with evidence (yes / partly / no):
1. Claims match the evidence: every sentence in the abstract and contributions is backed by a table row.
2. A fair baseline: MVFA re-run under the same rule, 3+ seeds, paired comparisons.
3. Pre-registered rules applied as written, including the ones that went against us.
4. No selection on the target test sets (the guardrails of §5).
5. Limitations are stated: the RESC / OCT17 protocol difference, the 5-epoch schedule, the subsample, π fixed rather than estimated, a single MVFA codebase, and the published MVFA numbers having been selected on the test set.
6. Reproducibility: code, configs, seeds, split lists and the `results.csv` are in the repository.
7. The AI's role is disclosed: Claude Code wrote the code and the report, and the teammate's feedback is logged (the issues log and transcript).
8. The contribution is clear in one sentence, and Claude Code writes that sentence.

Rules for this section:
- It judges the **evidence and the paper's readiness**. It does not predict whether a specific workshop will accept it, and it does not name a venue or a deadline unless the teammate supplies the call for papers. If one is supplied, it checks the paper against that call.
- Any "good" claim must cite the number behind it; any "not shown" claim must say what experiment would show it.
- It ends with a short, ordered list: what to do next, given the verdict (for example, "write up as a negative result", or "run #4 next").

---

## 8. Compute (≈ 2.5–3 hours)

- One update costs 0.018 s. With 6 jobs per GPU, the server does about 860k updates per hour (B14).
- With **E = 5 epochs** on the subsample (N ≈ 4,500 updates per epoch; × 1.5 for paired MIL and PU steps), about 2 minutes of wall-clock per run once 12 jobs share the 2 GPUs:

| Block | Runs | ≈ Wall-clock |
|---|---|---|
| Setup, normal banks, CPU checks, 1-epoch timing | — | 15 min |
| MVFA + MIL + PU + G | 44 | 1 h 30 min |
| PU-π5 + the two +100 % arms | 21 | 40 min |
| Evaluation, report, verdict | — | 20 min |
| **Default total** | 65 | **≈ 2 h 45 min** |
| Confirmation (only if an arm qualifies) | 12 | + 25 min |
| Budget points (only if time is left) | 4 | + 10 min |

These are estimates. After the 1-epoch timing, the agent re-checks the total. If it would pass 3 hours, the cuts are, in order:
1. no budget points;
2. 4 epochs instead of 5;
3. Chest and HIS folds only for MVFA, MIL and G;
4. PU-π5 on one fold.

Log each cut as a compute issue.

---

## 9. Decisions carried over from round 1 (Task 4)

The decisions taken at the end of round 1, and what each means for this round. Full text: [`DECISIONS.md`](../task4_share/task4/report/DECISIONS.md), [`ISSUES_LOG.md`](../task4_share/task4/report/ISSUES_LOG.md), [`TASK4_REPORT.md`](../task4_share/task4/report/TASK4_REPORT.md) §4–5.

| Round 1 | Decision | In round 2 |
|---|---|---|
| A5 | Never select a checkpoint on the target test set | Kept: source-validation selection (Task 3's rule) |
| B1 | MVFA's training data are the sources' `test/` folders | Kept. Validation uses `valid/` (Step 1 item 5) |
| B2, **D3** | "Mask-free" means both adapter sets are trained without masks | Kept for MIL, PU and G |
| B12, **D4** | MVFA's detection loss is replicated with its quirk | Kept (Task 3: detection loss unchanged) |
| B4, **D5** | A control skips steps instead of zero-weighting them (Adam) | Kept for every comparison |
| B10, B11, **D1, D2, D6** | The π estimator fails on MVFA's scores; π was set from masks (D1) or fixed at 0.30 (D2) | **Refined:** a fixed π from Task 3's own k grid (1 %, with 5 % as sensitivity), no estimator, no mask information |
| A2 | Paired batch via a rewritten forward | Reused |
| B13, B14 | Concurrency rule; 6 jobs per GPU | Reused |
| B15 | R0 reproduces MVFA's released checkpoint, not the paper | R0 kept as a reference row |
| Report §4 | PU is not a reliable gain overall (Brain only); plain top-k MIL keeps most of MVFA's localisation without masks | Why PU is refined, not dropped, and why #1 (MIL-based) is codified next to it |

---

## Appendix: kickoff prompt for Claude Code

Start Claude Code in the project folder on the GPU server, where round 1's `task4/` code and data paths live, and paste:

```
Read Plans/TASK4_ROUND2_PLAN.md completely, then task4_share/task4/report/TASK4_REPORT.md,
DECISIONS.md and ISSUES_LOG.md, and Task 3's A1 outputs on branch task3_kevalina
(task3/outputs/stage3_evolution_A1.json, stage4_ranking_A1.json).
Execute round 2 exactly as the plan says, following the loop in section 5. Work in task4r2/.
Stop at gates G1, G2 and G3 and wait for me; log my feedback verbatim in
task4r2/report/ISSUES_LOG.md. Read target test sets only through evaluate_target.py.
Do not add any method that is not in Task 3's outputs. Keep the whole round within 3 hours of GPU time.
When the results are in, write task4r2/report/VERDICT.md exactly as section 7 says: a clear verdict on
how good the results are and whether they can be submitted to a workshop, with evidence for every line.
Be honest: say plainly what is not shown.
```
