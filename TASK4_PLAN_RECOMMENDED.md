# Task 4 plan, RECOMMENDED tier (≈ 14 GPU hours, ≈ 17–18 with buffer)

**Goal:** two results, for a reputable workshop on AI-generated papers:
1. a controlled, multi-seed answer to *does a positive-unlabelled (PU) loss on MVFA's mask-less datasets improve localisation beyond plain fine-tuning?*;
2. **how much localisation MVFA keeps with no masks at all** when its segmentation adapters are trained with the PU loss instead, and whether that beats plain MIL.

**The idea being codified:**
- run 3's **witness-rate-adaptive MIL** (`A1-g1-o6`, problem A1, top-5 #3);
- refined into a **PU loss** (run 1's `A1-g2-o6`);
- applied to the three datasets MVFA never uses for localisation (OCT17, Chest X-ray, histopathology).

**Why it's cheap:**
- start from **MVFA's released zero-shot checkpoints**;
- freeze CLIP and the detection adapters;
- fine-tune **only the segmentation adapters** for a fixed number of steps;
- no tuning loop.

**Who runs it:** you, with an LLM (ChatGPT, Copilot or Claude) writing and fixing the code. Keep the issues log (Step 6); Task 4 requires it.

> Compared with the base plan (`TASK4_PLAN.md`), this tier adds **3 seeds** for the two main configs on every target, **AUPRO** for every run and **pre-registered decision rules** (the credible tier, `TASK4_PLAN_CREDIBLE.md`), **plus** the mask-free runs **M** and **B0** on every target.

---

## 1. Runs

Targets (zero-shot): **Liver CT, Brain MRI, RESC**, the BMAD datasets with pixel masks. **Run Liver CT first** (see the early-stop rule in Step 5).

| ID | What | Seeds | Runs |
|---|---|---|---|
| **R0** | MVFA's released zero-shot checkpoint for the target, evaluated as is | — | 3 (inference) |
| **M-h** | R0 + fine-tune the **segmentation adapters** for S steps: MVFA's mask loss on the masked sources **+ λ × PU loss on the mask-less sources** | 0, 1, 2 | 9 fine-tunes |
| **M-h-ctrl** | The same fine-tune **without the PU term** (mask loss only), same S steps and same seeds | 0, 1, 2 | 9 fine-tunes |
| **M** (mask-free) | Detection adapters from R0; **segmentation adapters re-initialised** and trained with the **PU loss only**, on **all** source datasets; **no mask is ever used** | 0 on all 3 targets; 1 and 2 on Liver CT | 5 trainings |
| **B0** (mask-free baseline) | Like M, but with plain **top-k MIL** (k = 1% of patches, at least 3) instead of the PU loss | 0 | 3 trainings |

**Total:** 3 inference runs + π estimations + **26 short trainings** (18 for M-h / M-h-ctrl, 8 for M / B0).

**Why M and B0 are mask-free:** in MVFA's code, the detection adapters are trained with image-level labels only (Task 3 evidence E6); only the segmentation adapters ever see masks (E7). Re-initialising the segmentation adapters and never loading a mask removes all mask information.

**What the seed controls:** the random order of the fine-tuning data. The subsample lists and π are fixed per target, the same for every seed.

---

## 2. The method (exactly this, nothing more)

For each **mask-less source dataset d** (OCT17, Chest, histopathology, whichever are sources for the target):

1. **Estimate π_d once, before fine-tuning, with the R0 model:**
   - run R0 on a held-out 10% of d's **training** images;
   - take the segmentation patch probabilities at the deepest level;
   - with F_norm the CDF over normal-image patches and F_abn over abnormal-image patches:
     **π_d = min over t of (1 − F_abn(t)) / (1 − F_norm(t))**, using only thresholds with **1 − F_norm(t) ≥ 0.01**;
   - clip to **[0.005, 0.30]** and save to `task4/results/pi.json`.
2. **PU loss** per level, combined with MVFA's level weights. z = patch logit, ℓ = binary cross-entropy, N = patches of a normal image, U = patches of an abnormal image from the same dataset:
   ```
   risk = (1 − π_d) · mean_N ℓ(z, 0)  +  max(0, mean_U ℓ(z, 1) − (1 − π_d) · mean_N ℓ(z, 1))
   ```
3. **Batches:** MVFA uses batch size 1. For the PU term, take one normal and one abnormal image from the same mask-less dataset and accumulate gradients over the pair.
4. **M-h loss:** MVFA's original Dice + Focal on the masked sources, unchanged, + **λ × PU** on the mask-less sources, alternating between the two. **λ = 1.**
5. **Fixed budget:** S steps (default **2,000**; set in Step 2 so that one fine-tune is ≤ 30 minutes). MVFA's zero-shot learning rate. **The final model is the last step**; no checkpoint is chosen on any test set.
6. **M-h-ctrl:** identical to M-h with λ = 0.
7. **M (mask-free):**
   - keep R0's frozen CLIP and **detection adapters**; **re-initialise the segmentation adapters** (MVFA's initialisation);
   - train them with the **PU loss only**, on **all** source datasets (the masked ones are used **only with their image labels**);
   - same S steps and learning rate; last step = final model;
   - **π for M comes from the detection branch.** Estimate π_d for **every** source dataset with step 1's estimator, but on R0's **detection-branch** patch maps (they never saw masks), not on the segmentation branch. Save to `pi_maskfree.json`.
8. **B0 (mask-free baseline):** as M, but the loss per level is top-k MIL. For abnormal images, BCE(mean of the top-k patch probabilities, 1); for normal images, BCE(mean of the top-k, 0); k = max(3, 1% of patches). No π needed.

> [!note] M may be under-trained
> M and B0 train re-initialised adapters for the same S steps as a fine-tune. If M's training loss is still clearly falling at the end, report it as **under-trained**. If time is left, run **2S on Liver seed 0 only**.

**Fixed hyperparameters.** One allowed exception: if M-h ≤ M-h-ctrl on Liver CT in seed 0, run **λ = 0.5** once, on Liver seed 0 only, and report both. Don't switch λ for the other runs.

---

## 3. Steps

### Step 1: Setup (no GPU)
1. Clone MVFA-AD at commit `a30bf7bb2c51772a36e7be58d3c072be480620fe`.
2. **Check that the released zero-shot checkpoints exist** for each target.
   - If they don't, train R0 once per target with MVFA's script (shortened, source-validation selection). That costs several extra hours, so log it.
3. Download BMAD and the CLIP weights as MVFA's README describes. Never commit them.
4. **Subsample** each source dataset: at most 500 normal + 500 abnormal training images, fixed list (seed 0), saved to `task4/splits/`. Hold out 10% for π.
5. Copy the paper's zero-shot **image AUROC and pixel AUROC** for Liver, Brain and RESC from MVFA's results table (arXiv 2403.12570) into `task4/results/paper_numbers.csv`. Copy exactly.

### Step 2: Reproduce R0 and time it
1. Evaluate R0 on each target: image AUROC, pixel AUROC and **AUPRO**.
2. Compare with `paper_numbers.csv`. If either AUROC is off by more than 2 points, check the preprocessing first, and log it.
3. A 50-step dry run of M-h: set S so that one fine-tune takes ≤ 30 minutes on your GPU.

### Step 3: Implement (LLM-assisted, CPU first)
1. The LLM implements:
   - the π estimator, the PU loss and the paired sampler;
   - the λ switch;
   - the M / B0 switches: re-initialise the segmentation adapters; PU-only or top-k-MIL loss on all sources; π from the detection branch;
   - freezing everything except the segmentation adapters;
   - **AUPRO**: per-region overlap, integrated to FPR 0.3 and normalised, computed per image in chunks to save memory.
2. **Checks before any GPU run:**
   - **PU loss:** equals a hand calculation on a toy batch; with π = 0 and no U patches, it equals plain BCE on N.
   - **π estimator:** on synthetic data (normal ~ N(0,1); abnormal = 90% N(0,1) + 10% N(3,1)), it returns 0.10 ± 0.03.
   - **AUPRO:** a perfect map gives 1.0; an inverted map gives about 0. Image and pixel AUROC match `sklearn` on random data.
   - **No leakage:** no target-dataset file appears in any training or π list.
   - **No masks in M / B0:** the training step raises an error if a mask tensor reaches the loss in M or B0 (a simple assert).
3. Log every problem in the LLM's first versions, and the feedback you gave, in the issues log.

### Step 4: Run (Liver CT first)
1. **Liver CT:** π; then M-h and M-h-ctrl for seeds 0, 1 and 2; evaluate each (image AUROC, pixel AUROC, AUPRO).
2. **Early-stop check** (Step 5). Then **Brain MRI** and **RESC** the same way.
3. **Mask-free runs:** π from the detection branch (once per target), then **M** (seed 0) and **B0** (seed 0) on all three targets, and **M** seeds 1 and 2 on Liver CT.
4. Append every run to `task4/results/results.csv`: `run_id, config, target, seed, steps, lambda, pi_json, img_auroc, pix_auroc, aupro, gpu_minutes, notes`.

### Step 5: Read the results (rules fixed in advance)
Let **Δ = M-h − M-h-ctrl**, paired by target and seed: 9 pairs for pixel AUROC, and 9 for AUPRO.

| Result | Claim |
|---|---|
| **AUPRO Δ > 0 in ≥ 8 of 9 pairs** and mean Δ > 0 on ≥ 2 of 3 targets | **"The PU term on mask-less data consistently improves localisation beyond fine-tuning."** (8/9 positive has a one-sided sign-test p ≈ 0.02.) |
| 7 of 9 | "Suggestive improvement" (p ≈ 0.09); say so |
| ≤ 6 of 9, or mean Δ within ±0.3 points | **No effect beyond fine-tuning.** Report it as a negative result, with π and the per-seed spread |
| **M-h > R0** (mean over seeds) on ≥ 2 of 3 targets, image AUROC not down by more than 0.5 points | "Matches or exceeds MVFA's released model", **with the caveat below** |

**Mask-free results (M, B0):**

| Result | Claim |
|---|---|
| Mean pixel-AUROC **retention** (M − 0.5)/(R0 − 0.5) ≥ 0.8 over the 3 targets, and M > B0 (pixel AUROC and AUPRO) on ≥ 2 of 3 targets | **"Without any masks, PU learning keeps most of MVFA's localisation and beats plain MIL."** |
| Retention ≥ 0.8, but M ≈ B0 (within 0.3 points) | "Image labels alone go a long way, but the PU loss adds little over MIL" |
| Retention < 0.8 | "Masks still matter." Report the retention per target, and AUPRO's ratio M / R0 |

Use Liver's 3 seeds of M to show how much M varies; the M-vs-B0 comparison on the other targets is single-seed, so say so.

**Early stop (M-h vs M-h-ctrl only):** if, after 3 seeds on Liver, the AUPRO and pixel-AUROC Δ are each within ±0.3 points with mixed signs, run **seed 0 only** on Brain and RESC to confirm, then write the negative result. That saves about 4 hours. The mask-free runs are done either way.

> [!warning] What the comparison with MVFA does and doesn't show
> MVFA chose its released checkpoints **on the test set** (Task 3 finding C2), and **M-h starts from those same checkpoints**, so it inherits that bias.
> - **M-h against M-h-ctrl is a fully clean comparison:** same starting point, same budget, only the PU term differs.
> - **M-h against R0 is not a clean "beats MVFA" claim;** report it with this caveat.
> - The RESC target also follows MVFA's protocol, in which OCT17 (another retinal OCT dataset) is a source. State that as a limitation.

### Step 6: Report and hand-off (no GPU)
`task4/report/TASK4_REPORT.md`:
1. **Idea and refinement:** witness-rate MIL → PU loss → hybrid fine-tune, with Task 3 IDs and why this idea.
2. **Issues log:** one row per issue: *issue → how found → feedback given to the LLM → fix*. Start it with:
   - the half-written PU formula;
   - batch size 1 versus PU's need for paired images;
   - where to estimate π;
   - the π estimator's instability on small tails;
   - MVFA's test-set model selection;
   - the full retrain was too expensive, hence the warm start.
3. **Results tables:**
   - (a) per target, R0 (paper), R0 (reproduced), M-h and M-h-ctrl (mean ± std over 3 seeds): image AUROC, pixel AUROC, AUPRO, π per mask-less dataset, Δ per pair;
   - (b) mask-free: M and B0 per target (seed 0; Liver M with 3 seeds), retention against R0, AUPRO ratio, and π from the detection branch.
4. **Which Step 5 rules apply** (both tables), and the honest conclusion.
5. **Limitations:**
   - zero-shot only;
   - subsampled fine-tuning;
   - 3 seeds;
   - starting checkpoints selected on test;
   - M and B0 single-seed outside Liver, and possibly under-trained at S steps;
   - the RESC/OCT17 overlap.

**Hand-off to the paper:** `results.csv`, `paper_numbers.csv`, `pi.json`, `pi_maskfree.json`, both results tables and Liver heat maps for R0, M-h, M-h-ctrl, M and B0, copied to `task4/report/paper_inputs/`.

---

## 4. Budget
| Item | Runs | Time |
|---|---|---|
| R0 evaluation with AUPRO, π estimation | 3 + 3 | ≈ 1 h |
| M-h + M-h-ctrl, 3 seeds × 3 targets | 18 × ≤ 30 min | ≈ 9 h |
| M (3 targets seed 0 + Liver seeds 1–2) and B0 (3 targets) | 8 × ≤ 30 min | ≈ 4 h |
| **Total** | | **≈ 14 GPU hours** (≈ 17–18 with a 25% buffer) |
| Optional: M at 2S on Liver seed 0, the λ = 0.5 exception | 1–2 | ≈ 1 h |

These assume ≤ 30 minutes per fine-tune (set in Step 2). On a free T4, expect roughly double, or halve S.
