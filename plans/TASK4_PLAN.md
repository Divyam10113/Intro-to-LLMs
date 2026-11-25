# Task 4 plan (simple, low-compute): beat MVFA's localisation with mask-less data

**The idea being codified:** run 3's **witness-rate-adaptive MIL** (`A1-g1-o6`, problem A1, top-5 #3), refined into a **positive-unlabelled (PU) loss** (run 1's `A1-g2-o6`). It's used to let MVFA learn localisation from the **three datasets it currently ignores for localisation** (OCT17, Chest X-ray, histopathology).

**Why it's cheap:**
- we **don't retrain MVFA**: we start from its **released checkpoints**;
- CLIP and the detection adapters stay **frozen**;
- only the **segmentation adapters** are fine-tuned, for a **fixed, short number of steps**;
- no tuning loop, and one seed for most runs.

**Target:** a few GPU hours in total on a free or cheap Colab GPU (T4 or L4).

**Who runs it:** you, with an LLM (ChatGPT, Copilot or Claude) writing and fixing the code. Task 4 requires "codify … by using LLMs" and a record of the issues found and the feedback given, so keep the short log in Step 6.

---

## 1. What we test (5 runs per target dataset)

Targets: **Brain MRI, Liver CT, RESC**, the three BMAD datasets with pixel masks, where localisation can be measured. Zero-shot setting: train on the other datasets, test on the target.

| ID | What | Training cost | Purpose |
|---|---|---|---|
| **R0** | MVFA's **released zero-shot checkpoint** for the target, evaluated as is | none (inference only) | The paper's result, reproduced |
| **M-h** | R0 + **fine-tune the segmentation adapters for S steps** with MVFA's **mask loss on the masked sources** **+ λ × PU loss on the mask-less sources** | short | **Main method:** can mask-less data improve localisation? |
| **M-h-ctrl** | Same as M-h, **without the PU term** (mask loss only), same S steps | short | **Control:** separates "fine-tuned a bit longer" from "learned from the mask-less data" |
| **M** *(optional)* | Detection adapters from R0; **segmentation adapters re-initialised** and trained with the **PU loss only**, on all sources, **no masks**; same S steps | short | Label efficiency: how much localisation without any mask? |
| **B0** *(optional)* | Like M, but with plain **top-k MIL** (k = 1%) instead of the PU loss | short | The problem's baseline for M |

**Must-run:** R0, M-h and M-h-ctrl on the 3 targets: 3 inference runs + 6 short fine-tunes. **Optional, if time is left:** M and B0.

**Why M is honest without masks:** in MVFA's code, the detection adapters are trained with image-level labels only (Task 3 evidence E6). Only the segmentation adapters ever see masks (E7), so re-initialising them removes all mask information.

---

## 2. The method (keep it exactly this simple)

**For each mask-less source dataset d (OCT17, Chest, histopathology; whichever are sources for this target):**

1. **Estimate the lesion fraction π_d once, before fine-tuning:**
   - run the R0 model on a held-out 10% of d's **training** images;
   - take the segmentation patch probabilities at the deepest level;
   - split them into normal-image patches (CDF F_norm) and abnormal-image patches (F_abn);
   - compute **π_d = min over thresholds t of (1 − F_abn(t)) / (1 − F_norm(t))**, using only thresholds with **1 − F_norm(t) ≥ 0.01**;
   - clip π_d to **[0.005, 0.30]**.

   Because R0 is already trained, its scores are meaningful from the start: **no warm-up and no re-estimation schedule** is needed. For M and B0 (re-initialised), estimate π with the R0 model as well.
2. **The PU loss** (per level, combined with MVFA's level weights). Use z = the patch logit, ℓ = binary cross-entropy, N = patches of a normal image, U = patches of an abnormal image from the same dataset:
   ```
   risk = (1 − π_d) · mean_N ℓ(z, 0)  +  max(0, mean_U ℓ(z, 1) − (1 − π_d) · mean_N ℓ(z, 1))
   ```
3. **Batches:** MVFA uses batch size 1. For the PU term, take **one normal and one abnormal image from the same mask-less dataset** and accumulate gradients over the pair.
4. **M-h loss** = MVFA's original Dice + Focal loss (masked sources, unchanged) + **λ × PU loss** (mask-less sources), alternating batches between the two. Default **λ = 1**.
5. **Fixed budget:** S steps (default **S = 2,000**, adjust after the timing test). The learning rate is MVFA's zero-shot default. **The final model is the last step.** There's no checkpoint selection on any test set: MVFA's own code selects on the test set (finding C2), and we don't.

**Fixed hyperparameters, no tuning loop.** One allowed exception: if M-h ≤ M-h-ctrl on Liver CT, try **λ = 0.5** once, on Liver CT only, and report both.

---

## 3. Steps

### Step 1: Setup (no GPU, about half a day)
1. Clone MVFA-AD at commit `a30bf7bb2c51772a36e7be58d3c072be480620fe`.
2. **Check that the released checkpoints exist** for the zero-shot setting and each target (see the repo's README).
   - If they don't: train R0 once per target with MVFA's script, a reduced number of epochs, and source-validation selection. That's the only expensive fallback; log it.
3. Download BMAD and the CLIP weights as MVFA's README describes. Never commit them.
4. **Subsample to save compute:** for fine-tuning, use at most **500 normal + 500 abnormal training images per source dataset** (fixed list, seed 0, saved to `task4/splits/`). Hold out 10% of each for π estimation.
5. Copy the paper's **zero-shot image AUROC and pixel AUROC** for Brain, Liver and RESC from MVFA's results table into `task4/results/paper_numbers.csv`. Copy them exactly; never type numbers from memory.

### Step 2: Reproduce R0 (inference only)
1. Evaluate the released checkpoint on each target's test set: image AUROC and pixel AUROC, using MVFA's evaluation code.
2. Compare with `paper_numbers.csv`. **Expect a close match.** If one is off by more than 2 points, check the preprocessing before going on; log it.
3. **Time it:** measure seconds per training step on your GPU with a 50-step dry run of M-h. Set S so that each fine-tune takes **≤ 30 minutes**, or lower it to fit your GPU.

### Step 3: Implement (LLM-assisted, on CPU first)
1. Ask the LLM to implement, inside MVFA's training script:
   - the π estimator;
   - the PU loss;
   - the paired sampler for mask-less datasets;
   - the M-h / M-h-ctrl / M / B0 switches;
   - freezing everything except the segmentation adapters (except in M and B0, which re-initialise them).
2. **Three quick checks before any GPU run:**
   - **PU loss:** on a hand-made toy batch, it equals a hand calculation; with π = 0 and no U patches, it reduces to plain BCE on N.
   - **π estimator:** on synthetic data (normal ~ N(0,1); abnormal = 90% N(0,1) + 10% N(3,1)), it returns about 0.10 (±0.03).
   - **No leakage:** assert that no target-dataset file is in any training list, and (M, B0) that no mask tensor reaches the loss.
3. Record every problem the LLM's first version had, and the feedback you gave, in the issues log (Step 6).

### Step 4: Run (the must-run set)
For each target (Brain, Liver, RESC):
1. estimate π for the mask-less sources (save it to `task4/results/pi.json`);
2. run **M-h** and **M-h-ctrl**, seed 0, S steps;
3. evaluate both on the target's test set (image AUROC, pixel AUROC);
4. append every run to `task4/results/results.csv` with columns `run_id, config, target, seed, steps, lambda, pi_json, img_auroc, pix_auroc, gpu_minutes, notes`.

**If time is left, in this order:**
- seeds 1 and 2 for M-h and M-h-ctrl on Liver CT (to see the noise);
- M and B0 on Liver CT;
- the λ = 0.5 exception.

### Step 5: Read the results (simple rules, written in advance)
| Result | What you can claim |
|---|---|
| **M-h > R0** on pixel AUROC for **≥ 2 of 3 targets**, image AUROC not down by more than 0.5 points, **and M-h > M-h-ctrl** | **The mask-less data helps:** the method improves on MVFA's localisation, and the improvement comes from the PU term, not from extra fine-tuning |
| M-h > R0, but M-h ≈ M-h-ctrl (within 0.3 points) | The gain comes from fine-tuning, not from the PU term. Report it honestly |
| M-h ≤ R0 | No improvement. Report the π values and the control as an analysis |
| (Optional) **M** keeps ≥ 80% of R0's margin: (M − 0.5)/(R0 − 0.5) ≥ 0.8 on pixel AUROC, and M > B0 | **Localisation without any masks** is largely possible, and the PU loss beats plain MIL |

> [!note] Comparing fairly
> R0's checkpoints were chosen by MVFA on the **test set** (finding C2), and **M-h and M-h-ctrl start from those same checkpoints**, so they inherit that bias. Only **M-h against M-h-ctrl** is a fully clean comparison (same start, same budget, only the PU term differs). Report M-h against R0 with this caveat, not as a clean "beats MVFA" claim.
>
> **RESC:** MVFA's protocol trains with OCT17 (another retinal OCT dataset) in the source set when RESC is the target. We keep MVFA's protocol for comparability, and note this as a limitation.

### Step 6: Write up (no GPU)
Produce `task4/report/TASK4_REPORT.md` with:
1. **Idea and refinement:** witness-rate MIL → PU loss → hybrid. Why this idea; cite Task 3's IDs.
2. **Issues log**, a table with one row per issue: *issue → how it was found → feedback given to the LLM → fix*. Start it with the known specification gaps:
   - the PU formula was half-written in the source idea;
   - batch size 1 versus PU's need for paired normal and abnormal images;
   - where to estimate π;
   - the π estimator is unstable for small tails;
   - MVFA selects models on the test set;
   - compute: a full retrain was too expensive, hence the warm-start design.
3. **Results table:** R0 (paper), R0 (reproduced), M-h, M-h-ctrl, and M and B0 if run. Image and pixel AUROC per target, plus π per mask-less dataset.
4. **Which rule from Step 5 applies,** and the honest conclusion.
5. **Limitations:**
   - one seed for most runs;
   - fine-tune subsamples;
   - the zero-shot setting only;
   - R0 selected on test;
   - the RESC/OCT17 overlap.

**Hand-off to the paper** (`plans/PAPER_PLAN.md`): copy `results.csv`, `paper_numbers.csv`, `pi.json`, the results table and two heat-map examples per config (on Liver CT) to `task4/report/paper_inputs/`.

---

## 4. Compute budget (rough; measure in Step 2)

| Item | Runs | Cost each |
|---|---|---|
| R0 inference | 3 | minutes |
| π estimation | 3 | minutes (inference on 10% subsets) |
| M-h, M-h-ctrl | 6 | ≤ 30 min each (fixed S) |
| **Must-run total** | | **about 3–4 GPU hours** |
| Optional: extra seeds, M, B0, λ = 0.5 (Liver only) | up to 7 | ≤ 30 min each |

If even this is too much: halve S, or run Liver CT first and add Brain and RESC only if Liver shows a gain.

---

## 5. What to hand in for Task 4
- The code (MVFA fork with the changes) and the notebook used to run it.
- `task4/report/TASK4_REPORT.md` with the issues log and the results table.
- `task4/results/results.csv`, `paper_numbers.csv`, `pi.json`.
- A record of the LLM conversations used for coding (links or exported text).
