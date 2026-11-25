# Task 4, round 3 (final): image labels instead of masks for MVFA, and mosaic MIL

**This is the last round of Task 4.** It ends with a step that picks the positive results and frames the paper around them (Step 6).

---

## 0. The idea, in brief

**Problem.** MVFA detects anomalies in a medical modality it never saw in training. The part that draws its anomaly maps, the segmentation adapters, learns only from pixel masks, so the three source datasets without masks (OCT17, Chest X-ray, histopathology) never teach it where lesions are.

**Question** (Task 3, problem A1). Can the segmentation adapters learn from **image labels alone** ("this image is abnormal"), and how much of MVFA's localisation survives on the unseen modality?

**Idea 1: plain MIL** (Task 3's main A1 experiment). Treat each image as a bag of patches. An abnormal image must have a few anomalous patches; a normal image must have none. At each of MVFA's four adapter levels l, with patch anomaly probabilities p_{l,j} and image label y ∈ {0, 1}:

  L_MIL = Σ_l w_l · BCE( mean_{j ∈ TopK(p_l)} p_{l,j} , y ),  k = 1 % of patches (3 of 289)

Every mask is discarded. MVFA's detection adapters and their loss are unchanged.

**Idea 2: mosaic MIL** (Task 3's idea #4, `A1-g2-o2`). Plain MIL only learns that an image contains a lesion somewhere. A mosaic of four images, each resized to 120 px and placed in one quadrant, has a free label for **every quadrant**: the label of the image placed there. So the model must learn *where* the abnormality is, with no annotation.
- **Normal quadrant:** every patch is pushed towards 0.
- **Abnormal quadrant:** consensus MIL inside the quadrant.
- **Consensus MIL** (`A1-g1-o1`, Task 3's baseline for #4): choose the top-k patches on the geometric mean of the four levels' maps, c_j = (Π_l p_{l,j})^{1/4}, then apply BCE(mean p, 1) at every level.

Half of the training steps are mosaics, the other half single images with consensus MIL.

**What would count as success.** Fixed before any run (§6):
- **Idea 1:** MIL keeps ≥ 90 % of MVFA's localisation on every target ("nearly matches"), and beats MVFA on Brain again, on fresh seeds.
- **Idea 2:** mosaic MIL beats consensus MIL by ≥ 2 points of pixel AUROC (Task 3's hypothesis).

**What would falsify them:**
- **Idea 1:** retention below 0.60 on any target.
- **Idea 2:** no pixel-AUROC gain over consensus MIL.

**Why it matters.** Pixel masks are the most expensive annotation in medical imaging. If image labels give most of the localisation, MVFA-style detectors can train on far more data.

---

## 1. Where this comes from: Task 3 and rounds 1–2

```
Task 3 (A1 problem + top-5 ideas)
  ├─ Round 1: #3 → PU refinement ─────────► no consistent gain; π estimator fails; plain MIL keeps 93–99 %
  ├─ Round 2: #1 gated MIL, #3 fixed π ───► both falsified; plain MIL keeps 93 %, beats MVFA on Brain;
  │                                          every run under-trained (5 epochs)
  ▼
Round 3 (final): plain MIL at MVFA's full 50 epochs, 5 fresh seeds, Task 3's secondary analyses;
                 idea #4 mosaic MIL; confirmations of every positive result carried forward
  ▼
Step 6: pick the positive results ──► paper framing (PAPER_FRAMING.md) ──► Task 5
```

**Task 3 sources:** branch `task3_kevalina`, `task3/outputs/stage4_ranking_A1.json` and `stage3_evolution_A1.json`. Copies are in [`task4_share/Plans/`](../task4_share/Plans/).

**A1's secondary analyses** (Task 3 `problem_text`):
- **(i)** a control where the segmentation branch's text anchors are replaced by a learned linear patch classifier trained with the same MIL loss;
- **(ii)** a per-level breakdown;
- **(iii)** label budgets with masks for 10 % and 25 % of the abnormal source images.

**Protocol** (shared by every A1 idea):
- leave-one-modality-out on BMAD, with RESC and OCT17 held out together;
- checkpoints chosen on source validation by image AUROC;
- MVFA and plain MIL re-run under the same rule;
- MVFA's released learning rate and epochs.

**Top 5 and status:**

| Rank | ID | Idea | Status |
|---|---|---|---|
| 1 | `A1-g1-o2` | Normal-bank-gated MIL | Falsified (round 2) |
| 2 | `A1-g3-o3` | Gated mosaic MIL | Not run: it uses #1's gate, which selected no more lesion patches than plain MIL (0.40 vs 0.43) |
| 3 | `A1-g1-o6` | Witness-rate-adaptive MIL (PU) | Not supported (rounds 1–2): the estimator fails, and fixed-π PU collapses |
| 4 | `A1-g2-o2` | **Mosaic MIL** | **Codified this round** |
| 5 | `A1-g3-o2` | Per-image adaptive k from normal-bank exceedances | Not run: it uses the same bank |

**Records:**
- **Round 1:** [`TASK4_REPORT.md`](../task4_share/task4/report/TASK4_REPORT.md), [`DECISIONS.md`](../task4_share/task4/report/DECISIONS.md), [`ISSUES_LOG.md`](../task4_share/task4/report/ISSUES_LOG.md).
- **Round 2:** [`TASK4R2_REPORT.md`](../task4_share/task4r2/report/TASK4R2_REPORT.md), [`VERDICT.md`](../task4_share/task4r2/report/VERDICT.md), [`ISSUES_LOG.md`](../task4_share/task4r2/report/ISSUES_LOG.md).

---

## 2. Positive results carried forward, and how round 3 confirms them

Each positive result from rounds 1–2 becomes a **pre-registered test** in round 3, on **fresh seeds (10–14)** at MVFA's full 50-epoch schedule. A result that holds again can be a headline claim. One that does not is reported as "did not replicate". Rounds are never pooled.

| # | Positive result | Where | Status then | Round-3 test |
|---|---|---|---|---|
| P1 | Plain MIL without masks keeps most of MVFA's localisation: 0.967 of R0 (R1, 1 seed), 0.930 of re-run MVFA (R2, 3 seeds) | R1 B0; R2 §3.4 | Measured twice, both times under-trained | **Q1**, the main result: retention at 50 epochs, 5 seeds |
| P2 | MIL beats MVFA on Brain on 3/3 seeds (AUPRO 70.6 ± 5.1 vs 61.6 ± 0.7), and also beats the released checkpoint (65.4) | R2 §3.1 | Not pre-registered | **C1:** confirmatory test on 5 fresh seeds |
| P3 | Round 1's Brain gain for PU (+5.2 AUPRO) is shared by every image-label loss | R1 §3; R2 §3.4 | Explained by P2 | Covered by C1; no separate test |
| P4 | Small mask budgets recover most of the gap: with G, 10 % / 25 % masks closed 62 % / 87 % of the Liver AUPRO gap; on Brain G+10 % beat MVFA | R2 §3.5 | 1 seed, G arm | **Q6:** budget curve for **MIL**, 3 seeds, 3 targets |
| P5 | PU+100 % ≥ MVFA in AUPRO on 7/9 pairs and in pixel AUROC on 3/3 targets | R2 §3.3 | "Suggestive", unconfirmed | **C2:** confirmatory test on 5 fresh seeds |
| P6 | Mask supervision, not the weak loss, decides how much of the selection lands in lesions (0.70–0.73 vs 0.43 for MIL) | R2 §3.4 | Diagnostic | Diagnostic again for MIL, C-MIL and MO. **Q2** tests whether mosaics raise it |
| P7 | A data finding: 11 BMAD `valid/` images are byte-identical to `test/` images (10 Brain abnormal, 1 OCT17 normal) | R2 G1-5 | Checked by MD5 | Re-checked in Step 1; reported as a benchmark note |
| P8 | Process: no target selection, append-only results, 41 tests, rules frozen before the runs | R1–R2 | Held in both rounds | Kept and extended (§9) |

---

## 3. Arms

| ID | What | Source |
|---|---|---|
| **MVFA** | Mask-supervised MVFA, re-run under the shared rule | A1 protocol |
| **MIL** | Idea 1: per-level top-k MIL, all masks discarded (round 2's `mil`) | A1 main experiment |
| **MIL-lin** | MIL with a learned linear patch classifier per level in place of the text anchors; same loss; its output is the map | A1 secondary (i) |
| **MIL+10 %, MIL+25 %** | Masks kept for a fixed, seeded 10 % / 25 % of the masked sources' abnormal images (Dice + Focal, as MVFA); MIL loss on everything else | A1 secondary (iii); P4 |
| **C-MIL** | Consensus MIL | `A1-g1-o1`, #4's baseline |
| **MO** | Idea 2: mosaic MIL, mosaic probability 0.5, on top of C-MIL | #4 `A1-g2-o2` |
| **MO-nc** | MO with all-normal or all-abnormal mosaics only (no spatial contrast) | #4's ablation |
| **MO-p25** | MO with mosaic probability 0.25 | #4's grid |
| **PU+100 %** | Round 2's `pu_m100`, code unchanged: masked sources as MVFA; nnPU with π = 1 % on the mask-less sources | P5 confirmation |

**Per-level breakdown**, secondary (ii): no extra training. `evaluate_target.py` scores each level's map separately for MVFA, MIL and MO.

---

## 4. Steps

### Step 1: Spec check, build, test (Claude Code session A)

Log every gap in `task4r3/report/ISSUES_LOG.md` in rounds 1–2's format (*issue → how found → feedback → fix*). Default resolutions, adopted unless the teammate objects at G1:
1. **Quadrant ↔ patch grid.** 240 px gives 17 × 17 patches of 14 px, but a 120 px quadrant is 8.57 patches wide. A patch belongs to the quadrant containing its centre; the middle row and column, which cross quadrant edges, get no mosaic loss.
2. **Mosaic sampling.** Four images from one source dataset, drawn with a seeded RNG. MO and MO-p25 use 1–3 abnormal quadrants (uniform). MO-nc is all-normal or all-abnormal with equal chance. A mosaic step replaces a single-image step, so every arm has the same number of steps.
3. **Detection loss on a mosaic.** Task 3 does not say. Default: MVFA's detection loss with label 1 if any quadrant is abnormal, otherwise 0.
4. **k inside a quadrant.** 1 % of about 72 patches, with the minimum of 3, gives k = 3.
5. **MIL-lin head.** One `nn.Linear(d, 1)` per level on the adapted segmentation tokens (logit → sigmoid), in the segmentation optimiser, initialised with the run seed.
6. **Budget set.** Per seed, a seeded 10 % / 25 % of each masked source's abnormal training images, with 25 % ⊃ 10 %.
7. **Per-level maps.** Extra `results.csv` rows with `level = 1..4`; `level = all` is the usual combined map.
8. **Selection.** Source-validation image AUROC after every epoch; best epoch kept, ties going to the later one. Last-epoch target numbers are logged as a secondary column and never used for any choice.
9. **Seeds.** Main runs use seeds **10–14**, which are fresh relative to rounds 1–2; the pilot uses seed 100.
10. **P7 re-check.** MD5 of every `valid/` file against every `test/` file.

**Build:**
- Copy `task4r2/` to `task4r3/`. Reuse the data, model, MIL, mask-budget, PU (`pu_m100`) and evaluation code unchanged.
- New code:
  - `mosaic.py`, `consensus_mil()` and the `mil_lin` head;
  - `train3.py`, with arms `mvfa mil mil_lin mil_m10 mil_m25 cmil mo mo_nc mo_p25 pu_m100`;
  - per-level evaluation;
  - `analyze3.py` (§6);
  - `select_positive.py` (Step 6);
  - `status.py`, a one-screen progress summary;
  - `check_budget.py` (Step 3).
- Every arm trains for `--epochs 50`.
- `run_all.sh` is **resumable**: it skips runs whose `run.json` is marked done.

**Tests**, extending round 2's 41 checks:
- **Mosaics:** quadrant labels match their source images; edge patches get no loss; MO-nc never mixes labels.
- **Masks:** no mask is loaded in MIL, MIL-lin, C-MIL or any MO arm; budget arms load masks only for their budget set.
- **Gradients:** consensus selection carries no gradient; the MIL-lin head replaces the anchors in the segmentation branch only.
- **Leakage:** no held-out file is in any training, mosaic or validation list, and only `evaluate_target.py` reads test sets.

**GPU smoke test:** every arm for 50 steps on one fold, plus one evaluation.

### Step 2: Chest check (session A)
Round 2 found Chest image AUROC below chance for every arm. Before any GPU time, check:
- the Chest test labels and class balance;
- the prompt used for Chest;
- the sign convention in `evaluate_target.py`;
- that Chest scores well *as a source* on source validation.

If it's a bug, fix it and add a test. If it isn't, log "Chest is hard for MVFA-style models under this protocol".

**Gate G1:** the teammate reads the Step 1 resolutions, the test and smoke output, and the Chest finding, then replies "go" or edits. The reply is logged verbatim.

### Step 3: Pilot: reproduction check and timing (unattended, about 2 h)
**The schedule is fixed at 50 epochs for every arm.** This is MVFA's released schedule and Task 3's protocol ("released learning rate and epochs"). Round 2 cut it to 5 for compute (G1-8), and every arm was under-trained (G3-1). A schedule picked from source validation is not used, because that score saturates (0.94–0.97) and can look flat while target localisation is still improving (round 2, limitation 2).

The pilot checks two things only; it chooses nothing.
- **Reproduction check.** MVFA on the Brain and Liver folds, 50 epochs, seed 100 (2 runs in parallel). These folds use the same sources as MVFA's own protocol, so the result can be compared with R0.
  - **Pass:** within 2 pixel-AUROC points of R0 (89.68 and 97.92).
  - **Fail:** Claude Code checks fp16, the learning rate and the data lists, and logs them. It continues either way, and a failure is limitation 1 in the report.
- **Timing.** One epoch of each arm type on the Brain fold, seed 100, run alongside: MIL, MIL-lin, C-MIL, MO and PU+100 %. Mosaic and paired-PU steps can be slower than single-image steps.

`check_budget.py` turns the measured times into a projection for the 138 runs at 50 epochs, applies §8's cuts if needed, and writes `report/BUDGET.md`. It also logs the SHA-256 of `analyze3.py`, `select_positive.py` and `RULES.md`, which freezes all rules before the main runs. The pilot's target numbers are logged and never used to choose anything.

### Step 4: Main runs (unattended)

| Order | Block | Arms | Folds × seeds | Runs |
|---|---|---|---|---|
| 1 | Core | MVFA, MIL | Brain, Liver, RESC+OCT × 10–14; Chest, HIS × 10–12 | 30 + 12 |
| 2 | Idea 2 | C-MIL, MO | Brain, Liver, RESC+OCT × 10–14; Chest, HIS × 10–12 | 30 + 12 |
| 3 | P5 confirmation | PU+100 % | Brain, Liver, RESC+OCT × 10–14 | 15 |
| 4 | Secondary (iii) | MIL+10 %, MIL+25 % | 3 masked folds × 10–12 | 18 |
| 5 | Secondary (i) | MIL-lin | 3 masked folds × 10–12 | 9 |
| 6 | #4 ablation | MO-nc | 3 masked folds × 10–12 | 9 |
| 7 | #4 grid | MO-p25 | 3 masked folds × 10 | 3 |
| | **Total** | | | **138** |

Blocks run in this order. Each run appends its rows to `task4r3/results/results.csv`. The Liver seed-10 runs save maps for the heat-map figure.

### Step 5: Analysis, report, verdict (Claude Code session C)
- `analyze3.py` applies §6 unchanged and writes `results/analysis.md` and `results/per_seed.md`.
- Claude Code then writes `report/TASK4R3_REPORT.md` with every result, positive and negative, in round 2's structure: idea, issues and feedback, results, rules applied, limitations, compute.
- It also writes `report/VERDICT.md` (§7).

### Step 6: Pick the positive results and frame the paper (Claude Code session C)
This is the final step of Task 4. `select_positive.py` sorts every result into tiers using the rules frozen in Step 3. Claude Code then writes `report/PAPER_FRAMING.md`, the brief the Task 5 paper is built on.

**Tiers:**

| Tier | What qualifies | How the paper may use it |
|---|---|---|
| **A: confirmed** | Passes its pre-registered rule in round 3 (Q1 "nearly matches" or "keeps most"; Q2, Q3 or Q4 supported after Holm; C1 or C2 confirmed) | Headline: title, abstract, contributions |
| **B: replicated** | Positive in round 2 *and* round 3 in the same direction, without passing a confirmatory rule (e.g. C2 positive in 11/15 pairs) | Results section, worded as "consistent across two rounds" |
| **C: exploratory** | Positive in round 3 only, not pre-registered as a test (e.g. a per-level or budget pattern) | Discussion only, labelled exploratory |
| **D: not positive** | Failed its rule, did not replicate, or is negative | Not a headline. Kept in **one compact table and one paragraph**: "What did not work" |

**The framing rules** (`PAPER_FRAMING.md` must follow them):
1. **The paper is built around Tier A.** The title, the abstract's main sentence and every contribution come from Tier A results. With no Tier A result, the headline is the strongest Tier B one, worded as consistent across two rounds rather than confirmed.
2. **The abstract leads with the positives.** One clause mentions the refinements that did not help, e.g. "…and two LLM-proposed refinements did not improve on it". That clause is the paper's "LLM ideas tested honestly" story, which an AI-generated-papers workshop values.
3. **Negatives are compressed, not removed.** Tier D goes into one table (each idea, its rule, the outcome) and one paragraph. That keeps the Task 3 → Task 4 trace the course requires, and it protects the positive claims from a "selective reporting" objection.
4. **No silent re-slicing.** A positive result found by restricting to a target, seed or metric after the fact is Tier C at most, never a headline.
5. **Every claim cites its row** in `analysis.md` and its tier.

**`PAPER_FRAMING.md` contains:**
- the title (2–3 options);
- a one-sentence contribution;
- a 3-item contributions list, each item tagged with its tier and its `analysis.md` row;
- the abstract's key sentences;
- the main table, built from the Tier A and B results;
- the "What did not work" table;
- the figure list: the Liver heat maps, plus the budget curve if Q6 is clean;
- the limitations;
- which `PAPER_PLAN.md` §5.2 framing applies: method, mixed or analysis paper.

**Hand-off to Task 5.** Claude Code writes `report/paper_inputs/` in the layout `PAPER_PLAN.md` §11 expects, with round 3 as `task4_r3`. Rounds 1–2 appear in the paper only through the carried-forward rows of §2 and the "What did not work" table, each labelled by round.

**Gate G2:** the teammate reads the report, the verdict and `PAPER_FRAMING.md`. The reply is logged verbatim, and Task 4 is closed.

---

## 5. Working with Claude Code on a Pro plan

A Pro plan has usage limits that reset every few hours, so Claude Code does the short, thinking-heavy parts and scripts and GPUs do the long work.

| Session | When | What |
|---|---|---|
| **A** | Day 1 | Steps 1–2, gate G1, launch the pilot. Most of the tokens go here |
| **B** | After the pilot | `check_budget.py`, launch the main runs. Short |
| **C** | After the main runs | Steps 5–6, gate G2 |

- **Hand-off file.** Each session ends by updating `task4r3/STATE.md`: what is done, what is running, the next command, and the open issues. Each session starts by reading it, so a session cut off by the usage limit resumes there.
- **No babysitting.** Long jobs run in `tmux` (`tmux new -s r3 './run_all.sh main'`), and Claude Code ends the session after launching them. The teammate can check progress with `python status.py` without using Claude.
- **Cheap reading.** Claude Code reads `status.py` output, `analysis.md` and failed-run logs only, never full training logs. Use `/compact` between steps.
- **Model.** Use the default model for building and running, and a stronger one, if the plan offers it, for the spec check (A) and for Steps 5–6 (C).
- **Failures** are handled by the script: one retry, then the run is marked `FAILED` and reported. Nothing is dropped silently.

---

## 6. Decision rules (pre-registered in `RULES.md` and `analyze3.py`, frozen in Step 3)

Points = AUROC/AUPRO × 100. A **pair** is (masked target, seed), so 5 seeds give 15 pairs. Retention = (pix − 50) / (MVFA − 50), as in rounds 1–2.

**Tests.** One-sided sign tests, with **Holm** correction inside each family at α = 0.05; raw and adjusted p-values are both reported. Over 15 pairs, 12/15 gives p = 0.018, 13/15 p = 0.004 and 14/15 p = 0.0005. Over 5 pairs, 5/5 gives p = 0.031.

**Family A: round-3 hypotheses** (MO vs C-MIL, C-MIL vs MIL, MIL vs MVFA, MO vs MVFA). With 4 tests, the first Holm step needs p ≤ 0.0125, which means at least 13 of 15 pairs.

**Q1. Idea 1: how much does MIL keep without masks?** This is an estimate, not a test. Report retention per target (mean ± std, and per seed), and apply the first label that holds:
1. **"Nearly matches":** ≥ 0.90 on every masked target.
2. **"Keeps most":** mean ≥ 0.75. This is Task 3's A1 threshold (`A1-g0-C1`: 75 %, falsified below 60 %).
3. **"Falls short":** otherwise.

Any target below 0.60 is named.

**Q2. Idea 2: mosaic MIL vs consensus MIL** (Task 3's hypothesis):
- **Supported:** the mean pixel-AUROC Δ over the 3 targets is ≥ +2, and its Holm-adjusted p ≤ 0.05.
- **Falsified:** that mean is ≤ 0.
- **Otherwise:** partial; say which targets.
- **Spatial contrast:** it matters if MO − MO-nc > 0 on ≥ 2 of 3 targets.

**Q3. Consensus vs per-level MIL** (`A1-g1-o1`'s falsifier): C-MIL − MIL AUPRO, Holm-adjusted p ≤ 0.05.

**Q4. Does a mask-free arm (MIL, MO) beat MVFA?**
- **"Beats":** AUPRO Δ has a Holm-adjusted p ≤ 0.05, pixel AUROC is higher on ≥ 2 of 3 targets, and image AUROC is not lower by more than 0.5 on any masked target.
- **"Matches":** mean pixel Δ ≥ −1 on every target.
- **Otherwise:** "trails".

**Family B: confirmations of carried-forward results** (C1, C2; Holm over 2 tests: first step p ≤ 0.025, second p ≤ 0.05):
- **C1 (P2): MIL beats MVFA on Brain.** Confirmed if MIL − MVFA Brain AUPRO > 0 on 5/5 seeds (p = 0.031), with mean pixel Δ > 0. Because 5/5 is p = 0.031, it is confirmed only at Holm's second step, i.e. if C2's p ≤ 0.025. Otherwise it is reported as "5/5 seeds, p = 0.031 uncorrected" (Tier B).
- **C2 (P5): PU+100 % beats MVFA.** Round 2's rule on 15 fresh pairs: AUPRO Δ has an adjusted p ≤ 0.05, pixel AUROC is higher on ≥ 2 of 3 targets, and image AUROC is not lower by more than 0.5 on any masked target.

**Descriptive** (Tier C at most, never a test):
- **Q5. Text anchors** (secondary (i)): they matter if MIL − MIL-lin ≥ +2 pixel AUROC on ≥ 2 of 3 targets; they don't if |Δ| < 1 on all three; otherwise mixed.
- **Q6. Label budget** (secondary (iii)): the retention curve at 0 / 10 / 25 / 100 % masks per target, and the smallest budget with mean retention ≥ 0.95.
- **Q7. Per level** (secondary (ii)): retention per level for MIL and MO.
- **Q8. The schedule's effect:** the round-3 arm ranking at 50 epochs against round 2's at 5 epochs. Never pooled.

**Diagnostics** (report only):
- the in-mask fraction of selected patches (MVFA's was 0.73);
- the share of normal-image patches above 0.5 (collapse monitor);
- best-epoch histograms;
- any image AUROC below 50, flagged.

---

## 7. Verdict (`task4r3/report/VERDICT.md`)

Same format and honesty rules as round 2's §7:
- **(A) Are these good results?**
  - one verdict per arm and overall: clear win / partial win / no improvement but informative / negative;
  - the three strongest and the three weakest numbers;
  - the outcome of each carried-forward test: P1, C1, P4 and C2 confirmed or not replicated.
- **(B) Workshop readiness:** one answer and round 2's 8-item checklist, plus two items:
  9. **Novelty check:** a short, targeted search for prior work that trains a CLIP-based anomaly detector's segmentation from image labels, or uses mosaics for free quadrant labels. Name the closest works and say what is new; mark anything unverified.
  10. **Agreement with `PAPER_FRAMING.md`:** the headline claims are all Tier A or B.
- **(C) Future work.** This is the final round, so this is a list of what the paper's future-work section should say, not a plan for round 4.

No venue or deadline is assumed unless a call for papers is supplied.

---

## 8. Compute

These are estimates from round 2's measurement: 1,024 job-minutes for 69 five-epoch runs at 12 concurrent jobs on 2 GPUs, ≈ 3 job-minutes per epoch per run. `check_budget.py` recomputes them from the pilot's real timing; mosaic and paired-PU steps may cost more.

| Block | ≈ Wall-clock on 2 GPUs (1 GPU: × 2) |
|---|---|
| Session A (build, tests, Chest check, smoke) | 2–4 h of Claude time; GPU nearly idle |
| Pilot (2 MVFA runs × 50 epochs, plus 1-epoch timings) | ≈ 2 h |
| Main, 138 runs × **50 epochs** | ≈ 29 h |
| Session C (analysis, report, verdict, framing) | 1–2 h |

**Cap: 30 h of wall-clock for the main block on 2 GPUs.** If the projected time exceeds it, `check_budget.py` cuts in this order and logs each cut as a compute issue:
1. drop MO-p25;
2. Chest and HIS at seed 10 only;
3. MIL-lin, the budget arms and MO-nc at seeds 10–11;

The epochs are **never** lowered. If the projection is still over 30 h after cuts 1–3, the runs simply take longer, because they are unattended; the overrun is logged.

**Never cut:**
- the Core block (MVFA, MIL × 5 seeds);
- idea 2's main comparison (C-MIL, MO × 5 seeds);
- the C2 confirmation (PU+100 % × 5 seeds).

**Storage:** about 50 MB per run → ≈ 7 GB on the HDD (`/mnt/HDD1/divyam/task4/r3/`).

---

## 9. Agent rules

1. **Test sets:** target test sets are read only by `evaluate_target.py`, after training, and nothing is chosen on a target. The pilot's reproduction check is logged and used for nothing.
2. **Results:** `results.csv` is append-only, and every run appears in the report and in `analysis.md`. Step 6 selects what the *paper* headlines; it never deletes results.
3. **Methods:** no method that is not in Task 3's outputs or carried forward from rounds 1–2.
4. **Rules:** `analyze3.py`, `select_positive.py` and `RULES.md` are hashed before the main runs. Any later change is logged as a deviation and reported.
5. **Feedback:** the teammate's words at G1 and G2 are logged verbatim, as the brief asks.

---

## Appendix A: Kickoff prompts

**Session A** (Claude Code in the project folder on the GPU server):
```
Read Plans/TASK4_ROUND3_PLAN.md fully (start with §0 and §2), then task4r2/README.md,
task4r2/report/VERDICT.md and task4r2/report/ISSUES_LOG.md, and Task 3's A1 files in Plans/
(stage4_ranking_A1.json problem_text; stage3_evolution_A1.json A1-g1-o1 and A1-g2-o2).
Do Steps 1-2 in task4r3/: log the spec gaps with the plan's default resolutions, build, test,
run the Chest check and the GPU smoke test. Stop at gate G1 with a short summary. After my "go",
launch the pilot in tmux, update task4r3/STATE.md and end. Read target test sets only through
evaluate_target.py. Add no method that is not in Task 3's outputs or the plan's §2.
```

**Session B** (after the pilot):
```
Read task4r3/STATE.md and Plans/TASK4_ROUND3_PLAN.md §4 Steps 3-4 and §8. Run check_budget.py,
check BUDGET.md, the reproduction check and the rule hashes, launch ./run_all.sh main in tmux, update STATE.md and end.
```

**Session C** (when `python status.py` shows the main runs are done):
```
Read task4r3/STATE.md and Plans/TASK4_ROUND3_PLAN.md §4 Steps 5-6, §6 and §7. Run analyze3.py,
write TASK4R3_REPORT.md (every result) and VERDICT.md. Then run select_positive.py and write
PAPER_FRAMING.md and report/paper_inputs/ exactly by Step 6's tiers and framing rules. Headline
only Tier A/B results; keep the Tier D results in one compact table. Stop at gate G2.
```

**Resume** (whenever a session was cut off by the usage limit):
```
Read task4r3/STATE.md and continue from the next step it lists. Do not redo finished steps.
```

## Appendix B: Decisions carried over

| ID | Decision | In round 3 |
|---|---|---|
| R1 A5 | Never select on the target test set | Kept; source-validation selection |
| R1 B1, R2 G1-5 | Train on the sources' `test/`, validate on `valid/` minus the 11 duplicates | Kept; duplicates re-checked (P7) |
| R1 D3, D4 | Mask-free means both adapter sets; MVFA's detection loss with its quirk | Kept |
| R2 G1-7 | Five folds, RESC + OCT17 held out together | Kept |
| R2 G1-8 | 5 epochs on a subsample (compute) | **Changed:** fixed at MVFA's 50 epochs; subsample kept so the schedule is the only change |
| R2 G1-13, G1-14 | k = 3; batch 1; the same image order across arms | Kept, also inside mosaic quadrants |
| R2 G1-19 | In-mask diagnostic | Kept, extended to C-MIL and MO |
| R2 G2-2 | The CPU smoke test was a mistake once the gate was delegated | GPU smoke test |
| R2 G3-1 | Under-training | Addressed by the fixed 50-epoch schedule; the best-epoch histogram checks it |
| R2 G3-2 | Chest below chance | Check step before GPU time |
| R2 G3-5 | Next idea is #4 | Codified here |
| User, 2026-11-22 | MIL-based paper with #4 as the one idea; carry the positive results forward; final round; frame the paper around the positives | §0, §2, Step 6 |
