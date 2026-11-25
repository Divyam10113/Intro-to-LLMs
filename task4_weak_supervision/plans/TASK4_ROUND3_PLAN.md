# Task 4, round 3 (final): firm up the positive results, then extend them

**In one sentence.** Rerun the arms that gave positive results in rounds 1–2, with 4× the epochs and fresh seeds, to confirm them; then run two new arms that combine those results: plain MIL with all masks, and plain MIL with only 10–25 % of the masks.

**Budget:** at most **3.5×** round 2's compute (round 2: 69 runs × 5 epochs = 345 run-epochs, ≈ 1¾ h on 2 GPUs). Round 3: 58 runs × 20 epochs = 1,160 run-epochs, ≈ 5½ h on the same 2 × RTX PRO 6000 Blackwell server.

**Safety:**
- round 2's code is reused as it is;
- the only code change is adding two arm names to round 2's existing mask-budget mechanism;
- every rule is fixed before the runs.

The fuller round-3 draft (mosaic MIL, 138 runs) is archived in [`archive/TASK4_ROUND3_PLAN_v1_full.md`](archive/TASK4_ROUND3_PLAN_v1_full.md) and is not used.

---

## 1. The idea

**Problem** (Task 3, A1). MVFA's segmentation adapters, the part that draws anomaly maps, learn only from pixel masks. Can they learn from **image labels** ("this image is abnormal"), and how much localisation survives on an unseen modality?

**Method: plain top-k MIL** (Task 3's main A1 experiment). At each of MVFA's 4 adapter levels l, the image's score is the mean of its k most anomalous patch probabilities, trained with BCE against the image label y:

  L_MIL = Σ_l w_l · BCE( mean_{j ∈ TopK(p_l)} p_{l,j} , y ),  k = 3 of 289 patches (1 %)

**What rounds 1–2 found** (the evidence this round builds on):

| # | Positive result | Evidence | Status |
|---|---|---|---|
| P1 | MIL with **no masks** keeps most of MVFA's localisation | R1: 0.967 of R0 (1 seed); R2: 0.930 of re-run MVFA (3 seeds) | Measured twice, both under-trained |
| P2 | MIL **beats MVFA on Brain** | R2: AUPRO 70.6 ± 5.1 vs 61.6 ± 0.7, higher on 3/3 seeds; also above the released checkpoint (65.4) | Not pre-registered |
| P3 | Adding a weak loss on the mask-less sources to MVFA's mask training gives ≥ MVFA | R2: PU+100 % AUPRO higher in 7/9 pairs, pixel AUROC higher on 3/3 targets; G+100 % matched MVFA | "Suggestive", unconfirmed |
| P4 | A **small mask budget** recovers most of the gap | R2 (gated MIL, 1 seed): 10 % / 25 % masks closed 62 % / 87 % of the Liver AUPRO gap; on Brain, 10 % masks beat MVFA | 1 seed, on the weaker MIL variant |

**The main weakness behind all four:** every round-2 run was under-trained. 37 of 57 runs picked their last epoch (G3-1), and the re-run MVFA trailed the released checkpoint by 9–14 points of image AUROC.

**Round 3 in two tiers:**
- **Tier 1, confirm (priority):** retest P1, P2 and P3 with the same arms and code, at **20 epochs** (4× round 2) on **fresh seeds**.
- **Tier 2, extend (new, built on Tier 1):** combine the positives.
  - **MIL+100 %:** P3's recipe with the best weak loss. Plain MIL beat both PU and G without masks (R2), so it is the natural loss to add on the mask-less sources. This is Task 3's summary recommendation 3.
  - **MIL+10 %, MIL+25 %:** P4's budgets with plain MIL instead of gated MIL. This is Task 3's A1 analysis (iii).

  Hypothesis: with masks for only 10 % of the abnormal images, MIL matches MVFA.

**Not run:** idea #4, mosaic MIL. It needs new training code and has no evidence yet, so it is named as future work (round 2 verdict, next step 4).

---

## 2. From Task 3 through rounds 1–2

```
Task 3 A1 (plain MIL is its main experiment; top-5 refinements #1–#5)
  ├─ Round 1: #3 (PU) ─────────► no consistent gain; plain MIL keeps 0.967 (P1)
  ├─ Round 2: #1 (gated MIL), #3 fixed π ─► both falsified; P1–P4 found; all runs under-trained
  ▼
Round 3 (final): feedback = "train longer, confirm on fresh seeds" (R2 verdict, next steps 2–3)
  Tier 1: MVFA, MIL, PU+100 % at 20 epochs, fresh seeds   → confirm P1, P2, P3
  Tier 2: MIL+100 %, MIL+10 %, MIL+25 %                    → extend P3, P4
  ▼
Verdict + paper claims → Task 5
```

**Records:**
- **Round 1:** [`TASK4_REPORT.md`](../task4_share/task4/report/TASK4_REPORT.md), [`DECISIONS.md`](../task4_share/task4/report/DECISIONS.md).
- **Round 2:** [`TASK4R2_REPORT.md`](../task4_share/task4r2/report/TASK4R2_REPORT.md), [`VERDICT.md`](../task4_share/task4r2/report/VERDICT.md), [`ISSUES_LOG.md`](../task4_share/task4r2/report/ISSUES_LOG.md).

**Everything else is unchanged from round 2:**
- the protocol: from scratch; leave-one-modality-out with RESC + OCT17 held out together; source-validation checkpoints;
- the data subsample;
- k = 3, batch 1, π = 1 % for PU;
- the guardrails (only `evaluate_target.py` reads test sets; append-only results);
- the 41 tests.

---

## 3. Runs

All runs use the three masked folds: **Brain**, **Liver** and **RESC + OCT17** (that fold's model is also scored on OCT17). Chest and HIS are not rerun (round 2's numbers stand). Every run is 20 epochs with source-validation selection after each epoch. Seeds **3–5** are fresh; round 2 used 0–2.

| Order | Tier | Arm | What | Folds × seeds | Runs |
|---|---|---|---|---|---|
| 1 | 1 | **MVFA** | Mask-supervised MVFA, re-run (round 2's `mvfa`) | 3 × {3, 4, 5} | 9 |
| 2 | 1 | **MIL** | Plain MIL, no masks (round 2's `mil`) | 3 × {3, 4, 5} | 9 |
| 3 | 1 | **PU+100 %** | MVFA's masks + nnPU (π = 1 %) on the mask-less sources (round 2's `pu_m100`) | 3 × {3, 4, 5} | 9 |
| 4 | 1 | MVFA, MIL | Two extra Brain seeds, so P2 is tested on 5 seeds | Brain × {6, 7} | 4 |
| 5 | 2 | **MIL+100 %** | MVFA's masks on the masked sources + MIL on the mask-less sources (**new arm** `mil_m100`) | 3 × {3, 4, 5} | 9 |
| 6 | 2 | **MIL+10 %** | Masks for a seeded 10 % of the masked sources' abnormal images; MIL on everything else (**new arm** `mil_m10`) | 3 × {3, 4, 5} | 9 |
| 7 | 2 | **MIL+25 %** | Same, with 25 % (**new arm** `mil_m25`) | 3 × {3, 4, 5} | 9 |
| | | | | **Total** | **58** |

- **Order:** blocks run in this order, longest folds first within a block, so the Tier 1 results are in first.
- **Maps:** the Liver seed-3 runs save maps for the heat-map figure (round 1's four images).
- **The code change** (`train2.py`, about 3 lines; round 2's mask-budget mechanism already does the rest):
  ```python
  ARMS += ["mil_m100", "mil_m10", "mil_m25"]
  MASK_BUDGET.update({"mil_m100": 1.0, "mil_m10": 0.10, "mil_m25": 0.25})
  WEAK.update({"mil_m100": "mil", "mil_m10": "mil", "mil_m25": "mil"})
  ```
- **New tests**, added to the 41:
  - `mil_m100` loads masks only on the masked sources;
  - `mil_m10` and `mil_m25` load masks only for their seeded budget set, with 25 % ⊃ 10 % for the same seed;
  - MIL, MVFA and the new arms see the same image order for the same seed.

---

## 4. Steps

### Session A (Claude Code): set up, test, launch
1. **Set up.** Copy `task4r2/` to `task4r3/`. Point the outputs at round 3: `TASK4R2_RESULTS=task4r3/results` and `TASK4_RUNS=/mnt/HDD1/divyam/task4/r3/runs`. Clear the copied `results/` and `logs/`. Reuse round 2's normal banks (`BANKS`), which the in-mask diagnostic reads. If they are not found, rebuild them with `./run_all.sh banks` (≈ 1.5 min).
2. **Code.** Add the three arm names (§3), the new tests, and a `run_all.sh r3` mode that queues the 58 runs in §3's order with `--epochs 20`. Write `analyze3.py`: a copy of `analyze2.py` with §5's rules. Then hash `analyze3.py` and `RULES.md` into `report/ISSUES_LOG.md`, which freezes the rules before any run.
3. **Test.** `pytest` (41 + new), then a **GPU** smoke test: each arm for 50 steps on the Brain fold (round 2's lesson G2-2).
4. **Timing check.** Measure the first 200 steps of each arm at full concurrency (as round 2 did, G2-5) and project the total. If the projection exceeds **4× round 2** (≈ 1,400 run-epochs, about 6½ h wall-clock), cut in this order: (a) MIL+25 %; (b) the two extra Brain seeds. Log any cut as a compute issue. Tier 1's first three blocks and MIL+100 % / MIL+10 % are never cut.
5. **Gate G1.** Show the teammate the code diff, the test and smoke output and the projection. Log their reply verbatim.
6. **Launch.** `tmux new -s r3 './run_all.sh r3'`. Write `task4r3/STATE.md` (what's done, what's running, the next command) and end the session. The runs are unattended; `python status.py` shows progress without Claude.

### Session B (Claude Code): analyse, report, verdict, paper claims
1. **Analyse.** Run `analyze3.py`; it writes `results/analysis.md` and `results/per_seed.md`.
2. **Report.** Write `report/TASK4R3_REPORT.md` in round 2's structure (idea, issues and feedback, results, rules applied, limitations, compute), with every run reported.
3. **Verdict.** Write `report/VERDICT.md` in round 2's format: one verdict per hypothesis, the strongest and weakest numbers, and the workshop checklist.
4. **Paper claims.** Write `report/PAPER_CLAIMS.md` (§6) and `report/paper_inputs/` in the layout `PAPER_PLAN.md` §11 expects (round 3 as `task4_r3`).
5. **Gate G2.** The teammate reads the report, the verdict and `PAPER_CLAIMS.md`. Log the reply; Task 4 is closed.

**On a Claude Pro plan:**
- Two short sessions are enough. Session A uses most of the tokens.
- If a usage limit cuts a session, start a new one with: *"Read task4r3/STATE.md and continue from the next step. Do not redo finished steps."*
- Claude reads `status.py` output and `analysis.md`, never full training logs.

---

## 5. Hypotheses and rules (fixed before the runs, in `RULES.md` and `analyze3.py`)

Points = AUROC/AUPRO × 100. A **pair** is (target, seed): 3 targets × 3 seeds = 9 pairs (Brain alone has 5 seeds for H2). Retention = (pix − 50) / (MVFA − 50), against this round's MVFA. Tests are one-sided sign tests: over 9 pairs, 8/9 gives p = 0.020 and 9/9 p = 0.002; over 5 seeds, 5/5 gives p = 0.031. Holm correction is applied within each tier, and raw and adjusted p-values are both reported.

**Tier 1: confirm the previous evidence**

| | Hypothesis | Confirmed if | Otherwise |
|---|---|---|---|
| **H1** (P1) | MIL without masks keeps most of MVFA's localisation | **"Nearly matches":** retention ≥ 0.90 on all 3 targets. **"Keeps most":** mean retention ≥ 0.75 (Task 3's A1 threshold) | "Falls short"; any target below 0.60 is named |
| **H2** (P2) | MIL beats MVFA on Brain | AUPRO higher on 5/5 Brain seeds and mean Brain pixel AUROC higher (Holm with H3) | 4/5: "holds on most seeds" (not confirmed); ≤ 3/5: "did not replicate" |
| **H3** (P3) | PU+100 % beats MVFA | Round 2's rule: AUPRO higher in ≥ 8/9 pairs (Holm-adjusted p ≤ 0.05), pixel AUROC higher on ≥ 2/3 targets, and image AUROC not more than 0.5 lower on any target | 7/9: "suggestive again"; ≤ 6/9: "did not replicate" |

H1 is an estimate, not a test, so it needs no correction. Holm over H2 and H3 needs p ≤ 0.025 at its first step. H2's best possible p is 0.031 (5/5), so H2 can be confirmed only at the second step, i.e. when H3 also passes. Otherwise a 5/5 Brain result is reported as "5/5 seeds, p = 0.031 uncorrected", in the "holds on most seeds" row of §6.

**Tier 2: extend**

| | Hypothesis | Supported if | Otherwise |
|---|---|---|---|
| **H4** | MIL+100 % beats MVFA | Same rule as H3 (Holm with H5) | "Matches" if the mean pixel Δ ≥ −1 on every target; else "trails" |
| **H5** | MIL with 10 % of the masks matches MVFA | Mean pixel Δ (MIL+10 % − MVFA) ≥ −1 on every target. **"Beats"** if it also meets H3's rule (Holm with H4) | "Trails" |

**Descriptive** (no test):
- the label-budget curve: retention at 0 / 10 / 25 / 100 % masks (MIL, MIL+10 %, MIL+25 %, MIL+100 %), per target;
- round 3 vs round 2 for each arm: the effect of 20 vs 5 epochs (never pooled);
- best-epoch histogram: if most runs still pick their last epoch, the report says the runs are still under-trained;
- round 2's in-mask diagnostic for every arm;
- any image AUROC below 50, flagged.

---

## 6. Paper claims (`PAPER_CLAIMS.md`)

The paper is built around the results that hold. `PAPER_CLAIMS.md` sorts each hypothesis by its §5 outcome:

| Outcome | Use in the paper |
|---|---|
| **Confirmed / supported** (H1 "nearly matches" or "keeps most"; H2–H5 passing their rule) | Headline: title, abstract, contributions |
| **Suggestive / holds on most seeds** | Results section, worded as such |
| **Did not replicate / trails** | One compact "what did not hold" table with one line each, plus the negative rounds-1–2 ideas (#1, #3). This keeps the Task 3 → Task 4 trace and protects the headline from a selective-reporting objection |

**It also contains:**
- 2–3 title options;
- a one-sentence contribution;
- 3 contributions, each citing its `analysis.md` row;
- the main table;
- the figures (Liver heat maps, plus the budget curve);
- the limitations;
- the `PAPER_PLAN.md` §5.2 framing that applies.

---

## 7. Compute

Round 2 measured ≈ 3.2 min of job time per epoch per run for these arms, at 12 concurrent jobs on 2 GPUs (`results.csv`, `total_minutes`).

| Item | Run-epochs | ≈ Wall-clock (2 GPUs; 1 GPU: × 2) | × round 2 |
|---|---|---|---|
| Round 2 (reference) | 345 | 1¾ h | 1× |
| Round 3, Tier 1 (31 runs) | 620 | 3 h | 1.8× |
| Round 3, Tier 2 (27 runs) | 540 | 2½ h | 1.6× |
| **Round 3 total (58 runs)** | **1,160** | **≈ 5½ h** | **3.4×** |
| Cap (cuts in Session A, step 4) | 1,400 | ≈ 6½ h | 4× |

Add about 1 h for setup, tests and the smoke test, and about 1 h of Claude time for Session B. Storage is about 50 MB per run, ≈ 3 GB in total.

---

## Appendix: kickoff prompts

**Session A** (Claude Code in the project folder on the GPU server):
```
Read Plans/TASK4_ROUND3_PLAN.md fully, then task4r2/README.md and task4r2/report/VERDICT.md.
Do Session A (plan §4): copy task4r2 to task4r3, add the three arm names and tests exactly as §3
shows, write analyze3.py with §5's rules and freeze them, run the tests and a GPU smoke test,
do the timing check, then stop at gate G1 with a short summary. After my "go", launch
./run_all.sh r3 in tmux, write task4r3/STATE.md and end. Change nothing else in round 2's code.
Read target test sets only through evaluate_target.py.
```

**Session B** (when `python status.py` shows all runs are done):
```
Read task4r3/STATE.md and Plans/TASK4_ROUND3_PLAN.md §4-§6. Run analyze3.py, then write
TASK4R3_REPORT.md (every run), VERDICT.md, PAPER_CLAIMS.md and report/paper_inputs/ as the
plan says. Apply §5's rules exactly as frozen; say plainly what did not hold. Stop at gate G2.
```
