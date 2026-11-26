# Task 4 execution plan: mask-free localisation for MVFA

**Who this is for:** an LLM coding agent (VS Code Copilot agent mode, Codex CLI, Cline, Cursor or similar) working in the team repository, with a human supervisor.

**How to use it:** this file lives in the repository as `plans/TASK4_PLAN.md`, next to `plans/PAPER_PLAN.md` (the final paper, which reports this task's results) and `plans/README.md` (order of execution). Point the agent's instruction file at it (`AGENTS.md`, `.github/copilot-instructions.md` or `.clinerules`; see Appendix A). Then give the start prompt in Appendix B. All outputs go to `task4/`. The agent executes the plan phase by phase, logs everything, and **stops at every human gate (marked 🛑)** until the supervisor answers.

---

## 0. Ground rules for the agent (read first, follow always)

1. **Follow the phases in order.** Do not start a phase until the previous phase's exit criteria are met and, where marked 🛑, the human has approved.
2. **One change at a time.** Every experiment differs from its parent by exactly one documented change. Never bundle changes.
3. **No test leakage.**
   - Within a training run, checkpoints and early stopping are chosen **only** on the validation split of the **source** datasets, never on the held-out target.
   - Configuration tuning (Phase 6) may look **only** at the development fold (`F_brain`). The clean evaluation folds are run once per frozen configuration and their numbers are never used to decide anything.
4. **Masks in mask-free configs.** In B0, B1, B2 and M, no pixel mask of any source dataset may reach the training loss, checkpoint selection or tuning. Masks may be read **only** for evaluation on the held-out fold and for the analysis-only comparison in §5.6.
5. **Log every iteration** in the files listed in §2.2, at the moment it happens, not afterwards.
6. **Validate before trusting.** Every new component needs a passing unit test (P4.3) before it is used in a GPU run. Every GPU result passes the sanity checks in §5.5 before it is recorded as valid.
7. **Stay within budget.** Track GPU hours in `task4/logs/BUDGET.md`. Never start a run that would exceed the remaining budget (§7.3).
8. **Never commit** datasets, model weights, checkpoints, API keys or tokens. Never `git push` without the human's approval.
9. **If unsure, stop and ask.** A clear question to the human is better than a guess. Record the question and the answer in `DECISIONS.md`.
10. **Report honestly.** A negative result is a valid result. Never present the development fold as a clean result, and never drop runs that went badly.

---

## 1. Background: what is being built and why

### 1.1 The base paper
MVFA (Huang et al., CVPR 2024): *Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images*.
- Code: `https://github.com/MediaBrain-SJTU/MVFA-AD` (Task 3 verified commit `a30bf7bb2c51772a36e7be58d3c072be480620fe`).
- MVFA adds two sets of adapters to a frozen CLIP model:
  - **detection adapters**: one anomaly score per image;
  - **segmentation adapters**: a patch-level anomaly map.
- Benchmark: BMAD, six datasets (Brain MRI, Liver CT, retinal OCT RESC, retinal OCT OCT17, Chest X-ray, histopathology).
- Zero-shot setting: train on five datasets, test on the unseen sixth.

### 1.2 The research problem (Task 3, run 3, problem A1)
> In MVFA's zero-shot training, the segmentation adapters, the only part that produces anomaly maps, are trained with Dice and Focal losses **only on the three source datasets that have pixel masks** (Brain MRI, Liver CT, RESC). OCT17, Chest X-ray and histopathology train only the detection adapters. **Can the segmentation adapters be trained from image-level labels alone, and how much mask-supervised localisation does that retain on the unseen modality?**

Code facts found by Task 3 (re-verify in Phase 1):

| Fact | Location in MVFA code |
|---|---|
| Dice + Focal seg loss applied only when `seg_idx > 0` (Brain, Liver, RESC) | `train_zero.py` line ~134 |
| Mask-less datasets train only the detection loss (`loss = det_loss`) | `train_zero.py` line ~160 |
| The zero-shot script tests on the **target test loader during training** and keeps the best test checkpoint (model selection on test) | `train_zero.py` line ~104 |
| A `valid/` split exists in BMAD and is used for few-shot supports | `dataset/medical_few.py` line ~133 |

### 1.3 The idea being codified
Task 4 codifies **one idea**: run 3's **witness-rate-adaptive MIL** (`A1-g1-o6`, ranked #3 in run 3's A1 top 5). It is then refined, through the feedback loop, into a **non-negative positive-unlabelled (nnPU) loss** that uses the idea's own lesion-fraction estimate (from run 1's top A1 idea, `A1-g2-o6`).

| Component | Source | Role |
|---|---|---|
| Per-dataset lesion fraction π, estimated without masks by mixture-proportion estimation | Run 3 `A1-g1-o6` (the codified idea) | Answers "how many patches are lesion?" |
| nnPU patch loss weighted by π | Run 1 `A1-g2-o6` (the refinement) | Replaces the top-k heuristic with a loss that uses every patch |
| Normal-bank gating | Run 3 `A1-g1-o2` | **Competitor only** (config B2), never combined with the PU loss |
| Masks kept where they exist + PU loss on mask-less datasets | Our extension | Attempt to beat MVFA (config M-h) |

**Do not combine gating with the PU loss.** The PU risk is unbiased only if the unlabelled set is *all* patches of abnormal images, mixed with proportion π. Removing patches first changes that mixture.

### 1.4 The experiment ladder

| ID | Config | Exactly one change from | Purpose |
|---|---|---|---|
| **R0** | MVFA with masks, original losses, **source-validation model selection** | the released code (only the selection rule changes) | State-of-the-art reference under a fair protocol |
| **B0** | Plain top-k MIL, k = 1% of patches, at all four levels; all masks removed | R0 (seg loss replaced) | The problem's own baseline |
| **B1** | k per dataset from the estimated π | B0 | Does choosing *how many* help? |
| **B2** | B1 + normal-bank gating of which patches are selected | B1 | Strongest MIL competitor |
| **M** | nnPU loss with π (no top-k, no gating) | B1 (loss form) | **Main method** |
| **M-h** | Hybrid: MVFA Dice+Focal on masked sources + nnPU (with π) on mask-less sources | R0 (adds PU term) and M (adds masks) | Attempt to beat MVFA |

---

## 2. Repository layout and logs

### 2.1 Layout (create in Phase 0)

```
task4/
  README.md                # how to run; filled in during Phase 8
  mvfa/                    # MVFA-AD at the pinned commit, our changes as clearly marked diffs
  src/
    losses.py              # topk_mil, gated_mil, nnpu_loss
    prior.py               # best-bin mixture-proportion estimator (π)
    bank.py                # normal-patch bank + coreset (B2 only)
    sampler.py             # paired normal/abnormal sampling per source dataset
    metrics.py             # image AUROC, pixel AUROC, AUPRO
    guards.py              # leakage and sanity guards (§5.5)
    train_weak.py          # training entry point: --config configs/<ID>.yaml --fold <F> --seed <s>
  configs/                 # one YAML per config ID and per tuning candidate
  tests/                   # unit tests (P4.3); pytest
  runners/
    modal_app.py           # GPU runner for Modal (agent can launch it)
    colab_runner.ipynb     # GPU runner for Colab (human runs it)
  results/
    results.csv            # one row per run (§2.3)
    runs/<run_id>/         # metrics.json, config.yaml, log.txt, small sample heat maps
  logs/
    ISSUES_LOG.md
    EXPERIMENT_LOG.md
    IMPROVEMENT_LOG.md
    DECISIONS.md
    BUDGET.md
  report/
    TASK4_REPORT.md
    paper_inputs/          # hand-off to plans/PAPER_PLAN.md (Phase 8, step 7)
```

`.gitignore` must exclude data, weights, checkpoints, `.env` and any `*.key`.

### 2.2 Logs (keep them current)

**`logs/ISSUES_LOG.md`**: the core Task 4 deliverable. One entry per issue:
```
### I-<nn>: <short title>
- Phase / step:
- Type: spec gap | inconsistency | ambiguity | compute limit | bug | result anomaly
- How it was found: (reading code, unit test, run check, human review)
- Evidence: (file:line, error message, metric)
- Feedback given to the model: (the exact instruction, or the human's answer)
- Fix: (what changed; commit hash)
- Outcome: (test passes / metric before -> after)
```

**`logs/EXPERIMENT_LOG.md`**: one line per GPU run: run_id, config, fold, seed, start/end time, GPU hours, status, link to `results/runs/<run_id>/`.

**`logs/IMPROVEMENT_LOG.md`**: one entry per improvement round (P6.3).

**`logs/DECISIONS.md`**: every human gate and every question asked, with the answer.

**`logs/BUDGET.md`**: total budget, hours used, hours remaining, updated after every run.

### 2.3 `results/results.csv` columns
`run_id, config_id, parent_id, change, fold, seed, git_commit, gpu, gpu_hours, best_epoch, sel_metric_source_val, img_auroc, pix_auroc, aupro, pi_estimates_json, collapsed, valid, notes`

Never delete rows. Mark invalid runs `valid=false`, with the reason.

---

## 3. Fixed protocol (shared by every config)

### 3.1 Folds
Leave-one-modality-out on BMAD. RESC and OCT17 are both retinal OCT, so they are **held out together**.

| Fold | Held out | Pixel metrics? | Use |
|---|---|---|---|
| `F_brain` | Brain MRI | yes | **Development fold**: pilot and tuning (Phases 5–6). Reported, but marked "dev". |
| `F_liver` | Liver CT | yes | Clean evaluation |
| `F_retina` | RESC + OCT17 | yes (RESC only) | Clean evaluation |
| `F_chest` | Chest X-ray | no (image AUROC only) | Clean evaluation, Tier 3 |
| `F_his` | Histopathology | no (image AUROC only) | Clean evaluation, Tier 3 |

### 3.2 Model selection inside a run
- Hold out the `valid/` split of every **source** dataset.
- Select the checkpoint by mean **image AUROC on the source validation splits**. This is the same rule for every config, so nothing uses masks.
- For R0 and M-h, also record source-validation pixel AUROC on the masked sources (logged, not used for selection, so all configs share one rule).
- Never evaluate on the held-out fold before training ends. Evaluate the selected checkpoint exactly once.

### 3.3 Metrics
- **Image AUROC** on every fold.
- **Pixel AUROC** and **AUPRO** (area under the per-region overlap curve, integrated up to FPR 0.3) on folds with masks.
- **Retention** = (M − 0.5) / (R0 − 0.5) for pixel AUROC, i.e. the share of R0's margin over chance that a mask-free config keeps. The problem's target is ≥ 80%.
- Report mean ± standard deviation over seeds.

### 3.4 Seeds
Seeds `0, 1, 2`. Fix Python, NumPy and PyTorch seeds, and log the cuDNN determinism setting.

### 3.5 Everything else stays as in MVFA
Backbone, image size, adapter levels and level weights, optimizer, learning rate, epochs and the detection loss all stay at MVFA's defaults, unless a logged issue requires a change.

---

## 4. Phases

### Phase 0: Setup and decisions (no GPU)

**Steps**
1. Create the layout in §2.1 and the empty log files.
2. Ask the human (🛑, record in `DECISIONS.md`):
   - GPU runner: **Modal** (the agent launches runs itself) or **Colab** (the agent prepares, the human runs and pastes back results);
   - total GPU budget in hours;
   - whether anyone else works on problem B1 (score calibration), and whether R0 runs are shared with them.
3. Add MVFA-AD at the pinned commit to `task4/mvfa/` and record the commit hash.
4. Write `.gitignore`.

**Exit criteria:** layout exists; runner and budget recorded; MVFA pinned.

### Phase 1: Code audit and specification freeze (no GPU)

**Goal:** check the method against MVFA's real code, and write the exact specification with every gap fixed. Every gap goes into `ISSUES_LOG.md`. This is Task 4's "identify issues before codification" step.

**Steps**
1. **Audit MVFA's code.** Find and record, with file:line:
   - (a) where patch-level anomaly probabilities are produced at each of the four levels (the tensor that the Dice/Focal loss consumes), and its shape;
   - (b) the batch size and how batches are drawn per dataset;
   - (c) the dataset names and the `seg_idx` / `CLASS_INDEX` mapping;
   - (d) the seg and det loss weights per level;
   - (e) the model-selection code (expected: on the target test set; Task 3 finding C2);
   - (f) how the `valid/` split is loaded;
   - (g) the backbone and image size;
   - (h) whether the patch grid is the same at every level.
2. **Re-verify Task 3's four code facts** (§1.2). Log any difference.
3. **Fix the known gaps.** Each becomes an issue entry with its resolution:
   - **I-01, the PU loss was half-written in its source idea.** Use this exact definition. Classes: 1 = lesion, 0 = normal. N = patches of **normal** images (labelled normal). U = patches of **abnormal** images (unlabelled, lesion fraction π). With per-patch logit z and the sigmoid BCE loss ℓ(z, y):
     ```
     R_N_neg = mean_{N} ℓ(z, 0)          # normal patches scored as normal
     R_N_pos = mean_{N} ℓ(z, 1)          # normal patches scored as lesion
     R_U_pos = mean_{U} ℓ(z, 1)          # unlabelled patches scored as lesion
     R_pos   = R_U_pos - (1 - π) * R_N_pos   # estimate of π * E_lesion[ℓ(z,1)]
     risk    = (1 - π) * R_N_neg + max(0, R_pos)
     ```
     Kiryo et al.'s nnPU gradient rule is an optional knob: if `R_pos < -β`, minimise `-γ * R_pos` instead of `risk`, with defaults β = 0 and γ = 1.
     - π is per source dataset.
     - The risk is computed per level and combined with MVFA's level weights.
     - Normal images contribute to N only; abnormal images contribute to U only.
   - **I-02, the batch composition.** The PU loss needs N and U from the **same source dataset** in each step. If MVFA's batch size is 1, implement `sampler.py` to yield pairs (1 normal + 1 abnormal from the same dataset) or accumulate over a pair. Keep the effective images per optimizer step equal to MVFA's, or log the change.
   - **I-03, where π is estimated.** On a held-out part of each source dataset's **training** images: 10% of normal and abnormal images, stratified and fixed by seed. Never on the held-out fold. Patches from this part are not used in the loss.
   - **I-04, which level π comes from.** Level-4 patch probabilities, used at all four levels, as the idea specifies. Per-level π is an improvement-round knob (P6.1).
   - **I-05, the π estimator is numerically unstable for small tails.** Best-bin estimator:
     ```
     π = min over thresholds t of  (1 - F_abn(t)) / (1 - F_norm(t)),
         over thresholds where 1 - F_norm(t) >= m_min (default 0.01)
     ```
     F are the empirical CDFs of level-4 patch probabilities of abnormal and normal images. Clip π to [0.005, 0.30], with a 2-epoch warm-up at π = 0.01 and an update every 2 epochs. Use a fixed threshold grid (e.g. 200 quantiles of the pooled scores).
   - **I-06, model selection on test** (MVFA). Replace it with the source-validation rule in §3.2, in every config including R0. Expect R0 to score below the paper; record both numbers.
   - **I-07, AUPRO is not in MVFA's code** (verify). Implement it in `metrics.py` with the standard per-region overlap definition, integrated to FPR 0.3, and test it (P4.3).
   - Any further gap found in step 1: add I-08, I-09, ….
4. **Write `task4/SPEC.md`**: the frozen specification of R0, B0, B1, B2, M and M-h, with the exact tensors, losses, sampling, π schedule and selection rule.

🛑 **Gate 1: specification review.** Show the human `SPEC.md` and the issue list. Proceed only after approval; record it in `DECISIONS.md`.

**Exit criteria:** `SPEC.md` approved; issues I-01…I-07 (and any new ones) logged with resolutions.

### Phase 2: Environment and data (light GPU)

**Steps**
1. Get BMAD and the CLIP weights exactly as MVFA's README describes. Record the download sources, file counts per dataset and split, and checksums where available.
2. Build the GPU runner chosen in Phase 0:
   - **Modal:** `runners/modal_app.py` takes `--config --fold --seed`, mounts the data volume, runs `src/train_weak.py`, writes `results/runs/<run_id>/metrics.json` and prints a one-line JSON summary. The agent launches it with `modal run` and reads the result.
   - **Colab:** `runners/colab_runner.ipynb` does the same. The agent writes the exact cell to run and asks the human to paste back `metrics.json` and the last 50 log lines.
3. **Smoke test:** run MVFA's released test script on its released checkpoint for one dataset, if available. Record the numbers.

**Exit criteria:** data present and counted; the runner works end-to-end on a 1-epoch smoke run; `BUDGET.md` started.

### Phase 3: Reproduce the reference R0 (GPU)

**Steps**
1. Implement R0: MVFA's original losses, with the selection rule replaced (I-06). That's one change.
2. Run R0 on `F_brain`, seed 0. Record the time per run; this sets the budget plan (§7.3).
3. Compare with the MVFA paper's zero-shot numbers for Brain MRI:
   - **within 5 points of pixel AUROC:** accept;
   - **otherwise:** open an issue, check data, preprocessing and the selection rule, and try at most 3 fixes. If it's still off, stop at Gate 2 and explain.

🛑 **Gate 2: reference accepted.** Show the R0 numbers, the paper's numbers and the time per run. The human approves R0 as the reference.

**Exit criteria:** R0 on `F_brain` seed 0 is valid and approved; cost per run is known.

### Phase 4: Codify with the agentic loop (CPU first)

Build the configs **in ladder order: B0 → B1 → B2 → M → M-h**. Each one follows the build loop.

#### P4.1 Build loop (per config)
1. **Plan:** write in `EXPERIMENT_LOG.md` what changes from the parent config, citing `SPEC.md`.
2. **Implement** the one change in `src/` and a `configs/<ID>.yaml`.
3. **Unit-test on CPU** (P4.3). Loop: run the tests → read failures → fix → rerun, at most 5 rounds. Each failed round that revealed a real problem gets an `ISSUES_LOG` entry.
4. **Self-review:** before any GPU run, answer in the log:
   - Does the code match `SPEC.md` line by line?
   - Can any mask reach a mask-free config's loss or selection?
   - Can the held-out fold reach training or selection?
   - Is exactly one thing different from the parent?
5. **GPU check run:** 1 fold (`F_brain`), 1 seed, **2 epochs only**. Check §5.5.
6. If the checks fail, diagnose and fix (at most 3 rounds per failure type), logging each round. If it's still failing, stop and ask the human.

#### P4.2 Per-config notes
- **B0:** at each level, positive term = BCE(mean of top-k patch probabilities of an abnormal image, 1); negative term = BCE(mean of top-k patch probabilities of a normal image, 0); k = 1% of patches, at least 3. Detection branch unchanged.
- **B1:** B0 with k_d = max(3, π_d × number of patches) per source dataset; π from `prior.py` with the schedule in I-05.
- **B2:** B1 with the selection score p_j × g_j, where g_j = 1 − max cosine(f_j, bank_d) (detached).
  - `bank_d` is a greedy coreset (default 20,000 vectors) of frozen CLIP level-4 patch features from up to 500 normal **training** images of source dataset d.
  - The bank is built once per fold, from source datasets only.
- **M:** replace the B1 MIL term with `nnpu_loss` (I-01); π from B1's estimator.
- **M-h:** masked sources keep MVFA's Dice + Focal; mask-less sources (OCT17, Chest, histopathology, whichever are in the fold's source set) use `nnpu_loss` with their π. Total seg loss = MVFA seg loss + λ × PU loss, with λ = 1 by default.

#### P4.3 Unit tests (must pass before any GPU run)
| Test | Checks |
|---|---|
| `test_nnpu_toy` | On a hand-made batch with known logits, the risk equals a hand computation (tolerance 1e-6); `max(0,·)` activates when `R_pos < 0` |
| `test_nnpu_reduces_to_pn` | With π = 0 and U empty, the risk equals the plain BCE on N |
| `test_prior_synthetic` | Samples: normal ~ N(0,1); abnormal = mixture of (1−π*)·N(0,1) + π*·N(3,1) for π* ∈ {0.02, 0.1, 0.25}; the estimate is within ±0.03 of π* with 50k samples |
| `test_prior_clip` | Degenerate inputs (identical distributions, empty tail) return a clipped π, never NaN |
| `test_topk_mil` | k = max(3, ⌈frac·P⌉); the gradient flows only to selected patches |
| `test_gating` | Patches identical to bank entries get g ≈ 0; random far vectors get g > 0.5 |
| `test_sampler_pairs` | Every batch has normal and abnormal images from the same source dataset; the held-out fold never appears |
| `test_no_mask_leak` | In B0, B1, B2 and M, the training step raises if a mask tensor is passed to the loss (guard in `guards.py`) |
| `test_no_target_in_train` | The train and selection loaders contain no file from the held-out fold (path check) |
| `test_metrics` | Image and pixel AUROC equal `sklearn.metrics.roc_auc_score` on random data; AUPRO equals 1.0 for a perfect map and ≈ 0 for an inverted one |

**Exit criteria (Phase 4):** all six configs implemented; all tests pass; each config passed its 2-epoch check run; issues logged.

### Phase 5: Pilot and go/no-go (GPU)

**Steps**
1. On `F_brain`, run **R0, B0, M** for seeds 0 and 1, and **B1 and B2** for seed 0. Use full epochs.
2. Compute the **seed-noise estimate** σ = the larger of R0's and B0's standard deviations of pixel AUROC across seeds 0 and 1. Set the **improvement threshold δ = max(0.5 points, σ)**.
3. **π check (analysis only, §5.6):** compare the estimated π per source dataset with the true lesion fraction from masks.
4. **Go/no-go:**
   - **GO with M as the main method** if M is valid (§5.5), not collapsed, and M ≥ B0 − δ on pixel AUROC.
   - **FALLBACK** to **B2** as the main method if M collapses or diverges in both seeds after the P4.1 fix rounds, or M < B0 − δ. Log the fallback as an issue and a decision. M is still reported.
   - **STOP and ask** if both M and B2 collapse.

🛑 **Gate 3: pilot review.** Show the pilot table, δ, the π comparison, the go/no-go decision and the budget projection (§7.3). The human approves the main method and the plan for Phase 6–7.

### Phase 6: Bounded improvement loop (GPU, development fold only)

The goal is to improve the **main method** (M, or B2 after a fallback) and **M-h**, using `F_brain` only, until the stop rule says further improvement is not credible.

#### P6.1 What may be changed (the knob list; one knob per round)
| Knob | Default | Candidates | Applies to |
|---|---|---|---|
| π warm-up epochs | 2 | 1, 4 | M, M-h, B1, B2 |
| π update interval | 2 epochs | 1, 4 | same |
| π clip upper bound | 0.30 | 0.15, 0.50 | same |
| π per level (vs. level-4 for all) | off | on | same |
| nnPU gradient rule (Kiryo β/γ) | off | on (β = 0, γ = 1) | M, M-h |
| PU loss weight λ_seg relative to MVFA's seg weight | 1.0 | 0.5, 2.0 | M, M-h |
| Hybrid weight λ | 1.0 | 0.5, 2.0 | M-h |
| Bank size (B2 fallback only) | 20,000 | 5,000 | B2 |
| k minimum / top-k fraction floor (B2 fallback only) | 3 / 0.5% | 5 / 1% | B2 |

Do **not** change: the backbone, epochs, optimizer, learning rate, data, folds, metrics or selection rule. A change outside this list needs human approval (log it in `DECISIONS.md`).

#### P6.2 One improvement round
1. **Hypothesis:** pick the knob most likely to help, based on evidence (collapse signs, π versus the truth, training curves), and state it in one sentence.
2. **Run:** the candidate on `F_brain`, seeds 0 and 1.
3. **Validate** with §5.5. An invalid run doesn't count as an attempt; fix or discard it, and log it.
4. **Accept** only if the mean pixel AUROC over the two seeds improves on the current best by **≥ δ** **and** both seeds individually improve (sign consistency). Otherwise **reject**, and the current best stays.
5. **Log** the round (P6.3).

#### P6.3 `IMPROVEMENT_LOG.md` entry
```
### Round <n> — <config> — knob: <name> = <value>
- Hypothesis:
- Evidence that motivated it:
- Result (F_brain, seeds 0,1): pix AUROC <a ± s> vs best <b ± s>; AUPRO; img AUROC
- Validity checks: pass/fail (details)
- Decision: ACCEPT (new best) | REJECT
- Self-critique: (anything suspicious? could the gain be noise or leakage?)
- Consecutive non-improving rounds: <k>
- GPU hours used so far / remaining:
```

#### P6.4 Stop rule: when improvement ends
Stop the loop for a config as soon as **any** of these holds, and write the reason in `IMPROVEMENT_LOG.md` as **"STOP: <reason>"**:
1. **Plateau:** 3 consecutive rounds rejected.
2. **Exhausted:** every knob in P6.1 that applies has been tried once from the current best.
3. **Round cap:** 8 rounds for that config.
4. **Budget:** the Phase 6 share of the budget (§7.3) is used up.
5. **Ceiling (M-h only):** M-h exceeds R0 by ≥ δ on `F_brain` and two more rounds don't improve it.

When every config has stopped, write in `IMPROVEMENT_LOG.md`:
> **"Improvement stopped: further gains are not distinguishable from seed noise (δ = …) within the allowed knobs and budget."**

Then freeze the configs (copy them to `configs/final/`, commit, record the hash).

🛑 **Gate 4: freeze review.** Show the improvement log summary and the frozen configs. The human approves before the clean evaluation.

### Phase 7: Clean evaluation (GPU)

Run the frozen configs **once** per fold and seed. Do not tune anything after seeing these numbers. If a clean run fails for technical reasons (a crash, not a bad score), it may be rerun with the same config. Log why.

**Run in this order; stop when the budget runs out:**
- **Tier 1 (must):** R0, B0, main method, M-h on `F_liver` and `F_retina`, seeds 0–2.
- **Tier 2:** B1 and B2 (or M, if B2 became main) on `F_liver` and `F_retina`, seeds 0–2; plus seed 2 for every config on `F_brain`.
- **Tier 3:**
  - π sensitivity: the main method with π × 0.5 and π × 2, on `F_liver`, seed 0;
  - image AUROC on `F_chest` and `F_his` for R0, B0, the main method and M-h, seed 0.

**Exit criteria:** Tier 1 complete and valid.

### Phase 8: Analysis and report (no GPU)

**Steps**
1. **Tables** (mean ± std), with clean folds and the dev fold kept separate:
   - pixel AUROC, AUPRO, image AUROC per config and fold;
   - retention versus R0 (target ≥ 80%);
   - M-h versus R0: does it beat the base paper's reference under the fair protocol?
2. **Ablation reading:** B0 → B1 (effect of π), B1 → B2 (effect of gating), B1 → M (effect of the PU loss), R0 → M-h (effect of adding mask-less data).
3. **π analysis:** the estimate against the truth per dataset; the π sensitivity results.
4. **Qualitative:** 4 heat maps per config on `F_liver` (2 good, 2 bad, chosen by a fixed rule, not cherry-picked).
5. **Honesty checks:**
   - state where the main method lost;
   - state that `F_brain` was used for tuning;
   - state the R0-versus-paper gap and its cause (I-06).
6. **Write `report/TASK4_REPORT.md`:**
   1. Chosen idea and why (from Task 3: run 3's A1 top 5, idea #3; refinement from run 1's #1).
   2. Issues found and how they were fixed (a summary of `ISSUES_LOG.md`, including the LLM feedback loop).
   3. Method as finally implemented (`SPEC.md` plus accepted improvements).
   4. Setup: data, folds, protocol, compute.
   5. Results against state of the art (R0 and the paper) and against the baselines.
   6. Improvement loop and stop reason.
   7. Limitations.
   8. Reproduction commands.

7. **Export the hand-off for the paper** (`plans/PAPER_PLAN.md` reads only these files) to `task4/report/paper_inputs/`:
   - `main_table.csv`: per config × fold, mean ± std over seeds of pixel AUROC, AUPRO, image AUROC and retention, with the dev fold flagged;
   - `ablation_ladder.csv`: B0 → B1 → B2 → M and R0 → M-h differences, with δ;
   - `pi_vs_truth.csv` and `pi_vs_truth.png`;
   - `heatmaps.png`: the fixed-rule qualitative examples from step 4;
   - `ablation_bars.png`;
   - `summary.md`: the main method finally used (M or B2), δ, the improvement stop reason, total GPU hours, R0 against the paper's numbers, and a short list of the most important `ISSUES_LOG` entries.

   Every number in these files must come from `results/results.csv` (valid runs only) and match the §5.7 re-check.

🛑 **Gate 5: final review.** The human reviews the report and the paper hand-off, and decides on any commit or push. Then the paper's Stage B (`plans/PAPER_PLAN.md` §9) can start.

---

## 5. Validation (applies throughout)

### 5.1 Static checks before every GPU run
- Unit tests pass.
- The `git diff` against the parent config touches only the declared change.
- The config file exists and its hash is recorded.

### 5.2 Runtime guards (`guards.py`, always on)
- **Mask guard:** in mask-free configs, any mask tensor reaching the loss raises an error.
- **Fold guard:** any file of the held-out fold in a training or selection loader raises an error.
- **NaN guard:** NaN or inf in a loss stops the run, saving the last finite state.

### 5.3 Training-curve checks (from `log.txt`)
- The seg loss decreases over the first epochs (not flat, not exploding).
- π stays inside its clip range, and isn't pinned at a bound for more than 50% of updates.

### 5.4 Output checks
- **Collapse:** the standard deviation of pixel scores on the held-out fold is < 1e-4, **or** more than 95% of pixels share one score → `collapsed = true`, run invalid.
- **Inverted maps:** pixel AUROC < 0.45 → flag it, and check the sign convention before accepting.
- **Detection sanity:** image AUROC must not drop by more than 5 points from R0 on the same fold and seed, because the detection branch is unchanged. A larger drop means a bug.

### 5.5 A run is **valid** only if
5.2 raised nothing, 5.3 holds, 5.4 shows no collapse, metrics were computed on the selected checkpoint (source validation), and the row in `results.csv` is complete.

### 5.6 π against the truth (analysis only)
For each masked source dataset, compute the true lesion fraction from masks at the patch grid: the share of patches with ≥ 50% mask overlap in abnormal images. Compare it with the estimated π. **Never feed this back into training or tuning.**

### 5.7 Re-check of headline numbers
Before writing the report, recompute the headline numbers (main method and M-h, clean folds) from saved predictions with a separate script, and confirm they match `results.csv` to 4 decimals.

---

## 6. Decision rules at a glance

| Situation | Action |
|---|---|
| A unit test fails 5 rounds in a row | Stop and ask the human |
| A run fails a validity check 3 times with different fixes | Stop and ask |
| R0 is more than 5 points below the paper after 3 fixes | Stop at Gate 2 |
| M collapses or falls below B0 − δ in the pilot | Fallback: B2 becomes the main method |
| Both M and B2 fail | Stop and ask |
| A candidate improves by ≥ δ and both seeds improve | Accept |
| 3 rejected rounds in a row, all knobs tried, 8 rounds, or the budget is spent | Stop improving that config |
| A clean-fold number looks surprising | Report it; do **not** tune on it |
| Budget would be exceeded | Stop and ask before starting the run |

---

## 7. Compute and budget

### 7.1 What costs GPU time
Every config trains MVFA's adapters once per fold and seed. The cost per run (T hours) is measured in Phase 3.

### 7.2 Run count
| Phase | Runs |
|---|---|
| Phase 3 | 1 |
| Phase 4 check runs | 5 × short (2 epochs) |
| Phase 5 pilot | 8 (R0, B0, M × 2 seeds; B1, B2 × 1 seed) |
| Phase 6 | 2 per round, at most 8 rounds × 2 configs = up to 32 |
| Phase 7 Tier 1 | 4 configs × 2 folds × 3 seeds = 24 |
| Phase 7 Tier 2 | 2 configs × 2 folds × 3 seeds + 6 configs × 1 seed = 18 |
| Phase 7 Tier 3 | 2 + 8 = 10 |

### 7.3 Budget split
After Phase 3, with T known and the total budget H:
- reserve **Tier 1 of Phase 7 first** (24 × T);
- then the Phase 5 pilot (8 × T);
- give Phase 6 at most **30%** of what remains, and the rest to Tiers 2–3.

If Tier 1 plus the pilot doesn't fit in H, stop at Gate 2 and propose cuts to the human: fewer seeds, one clean fold, or fewer epochs for every config alike.

---

## 8. Expected outcomes (for honest reporting)

- **Mask-free configs (B0–M)** are not expected to beat mask-trained MVFA. The claim to test is retention ≥ 80% and **M > B0** (and M versus B2).
- **M-h** is the realistic chance to beat MVFA, because it adds localisation signal from the datasets MVFA ignores.
- **If B2 beats M,** report it. The finding is then that the choice of patches matters more than the loss form.

---

## Appendix A: Instruction file for the agent

Save as `AGENTS.md` (Codex), `.github/copilot-instructions.md` (Copilot) or `.clinerules` (Cline):

```
You are executing plans/TASK4_PLAN.md. Read it in full before doing anything. Its outputs go to task4/.
- Follow its phases in order and obey section 0 (ground rules) at all times.
- Stop at every gate marked 🛑 and wait for the human.
- Log every issue, run, improvement round and decision in task4/logs/ as the plan specifies, when it happens.
- Never commit data, weights, checkpoints or keys. Never push without approval.
- When in doubt, ask.
```

## Appendix B: Start prompt

> Read `plans/README.md`, `plans/TASK4_PLAN.md` and the instruction file. Start Phase 0. Create the layout and logs, then ask me the Phase 0 questions. Do not go beyond Phase 0 until I answer.

## Appendix C: References
- MVFA: Huang et al., *Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images*, CVPR 2024; code `MediaBrain-SJTU/MVFA-AD`.
- nnPU: Kiryo, Niu, du Plessis and Sugiyama, *Positive-Unlabeled Learning with Non-Negative Risk Estimator*, NeurIPS 2017.
- Mixture-proportion estimation (best-bin): Scott, *A Rate of Convergence for Mixture Proportion Estimation*, AISTATS 2015.
- Task 3 outputs (branch `task3_kevalina`): run 3 `outputs/stage4_ranking_A1.json`, `stage3_evolution_A1.json`; run 1 `runs/run1_surveygen341/outputs/` (idea `A1-g2-o6`).
- Task 4 reference papers suggested by the course: DeepScientist, Arbor, AutoSOTA, EvoScientist (the agentic loop in Phases 4 and 6 follows their implement → run → critique → revise pattern).
