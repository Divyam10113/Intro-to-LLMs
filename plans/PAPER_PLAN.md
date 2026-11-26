# Final task: research paper plan, v2 (PaperOrchestra + our Tasks 1–4, with Task 4 round 1 in hand)

**Goal:** a research paper in **4+1 format** (4 pages of content plus 1 page of references only) that consolidates the **latest** outputs of Tasks 1–4. It is drafted with Google Research's **PaperOrchestra** and verified by us section by section, as the instructor asks.

**Executed by:** a coding agent (Claude Code, Copilot agent mode, Codex CLI, Cursor or similar). **The agent runs end to end without stopping to ask.** The human gives a few inputs at the start (all with defaults, §2) and does **one review at the end** (Gate 1, §10). Anything the agent could not settle goes into `paper/OPEN_ITEMS.md` for that review, not into a question.

**Version:** v2, 2026-11-22. v1 is in `archive/PAPER_PLAN_v1.md`.

**What changed since v1:**
1. **Task 4 round 1 is finished.** Its outputs are in `task4_r1/` (see `MANIFEST.md`). The paper no longer waits for results: §5's framing rule has been applied and gives an **analysis paper** (F-C), not a method paper.
2. **The method section follows what was actually run**, not what v1 expected. The π estimator failed (§0.2); M-h took π from the masked sources (D1); M and B0 are mask-free in both adapter sets (D3).
3. **Round 2 is coming** (`TASK4_ROUND2_PLAN.md`). It will replace or extend the Experiments section, so the inputs are organised by round (§4.3) and §11 says how to swap.
4. **The citation pool needs no new search** for the recent zero-shot medical anomaly-detection papers: IQE-CLIP and MADPOT are already in Task 1's pool. **No fixed number of references.**
5. **Fewer human steps:** v1's five gates are now one final review; the agent applies the rules, logs its choices and does not ask questions (§1 rule 9, §10).
6. **A script is provided:** `tools/build_results.py` builds `results.json` from any round's folder, tested against round 1's report.

**Sources:** only the team's GitHub repository `Divyam10113/Intro-to-LLMs` and Task 4's outputs. Do **not** use any older local folder or Google Drive copy.

---

## 0. What the paper is about

### 0.1 Problem (Task 3, run 3, A1)
MVFA's segmentation adapters, the only part that produces anomaly maps, learn only from expert pixel masks on 3 of BMAD's 6 datasets; the other 3 never teach localisation. *Can localisation be learned from image-level labels alone, and how much of mask-supervised localisation does that keep on an unseen modality?*

### 0.2 What Task 4 round 1 did and found
Run from `TASK4_PLAN_RECOMMENDED.md`, with deviations fixed before any GPU run in `task4_r1/report/DECISIONS.md`. Zero-shot, leave-one-modality-out, targets Brain MRI, Liver CT, RESC; warm start from MVFA's released checkpoints (R0).

| Part | What it was |
|---|---|
| **M-h** | R0 + a short fine-tune of the segmentation adapters: MVFA's mask loss on the masked sources + λ × a non-negative PU loss on the three mask-less sources (witness-rate idea `A1-g1-o6`, refined to PU as `A1-g2-o6`). π per target = the masked sources' true anomalous-patch fraction (D1) |
| **M-h-ctrl** | The same fine-tune without the PU term (the clean control) |
| **M** | Mask-free: both adapter sets re-initialised, PU loss only, π = 0.30 fixed (D2, D3) |
| **B0** | Mask-free baseline: same, with plain top-k MIL |

**Findings** (all numbers must be rebuilt from `results.json`; the values below are for orientation only):
1. **PU on mask-less data gives no consistent gain over plain fine-tuning.** AUPRO rises in 6 of 9 paired runs (sign test p = 0.25). The gain is large and seed-consistent on Brain (+5.2 AUPRO, +1.7 pixel AUROC, 3/3 seeds), absent on Liver, unstable on RESC.
2. **M-h does not beat the released model R0** (only on Brain). R0 was selected on the test set (Task 3 finding C2), so this comparison favours R0.
3. **The mask-free PU model fails:** retention 0.58 of R0's pixel-AUROC margin; chance level on Liver (49.9) because π = 0.30 is far too high (the true Liver lesion fraction is about 0.8 %).
4. **π cannot be estimated from MVFA's patch scores,** even against ground truth: the estimator returns about 1.0 where the true fractions are 0.008–0.119 (B10, B11). Every patch of an abnormal image outscores the normal patches at deep levels, so the mixture assumption fails.
5. **Plain top-k MIL with no masks (B0) keeps 93–99 % of R0's localisation** on all three targets. Not pre-registered, single seed per target.
6. **A code audit of MVFA:** MVFA's detection adapters also receive mask gradients (B2); its zero-shot training data are the sources' `test/` folders (B1); its detection loss feeds a probability to `BCEWithLogitsLoss` (B12); its released checkpoints reproduce the README numbers, not the paper's (B15); checkpoints are selected on the target test set (A5, Task 3 C2).

### 0.3 Framing: an honest analysis paper
A method paper is not supported, so the paper is built on findings 1, 3, 4, 5 and 6 as an **analysis of how far image-level labels go on MVFA**, with a clean negative result on PU learning and a strong simple baseline. No result may be described as an improvement unless a rule in §5 says so.

**Interim read on workshop readiness (round 1 only):** an honest mixed-result paper is submittable in principle; a positive-result paper is not. The final decision is made at the final review (Gate 1) using round 2's `VERDICT.md` if it exists (§11).

### 0.4 Interpreting the instructor's brief

| Instructor's words | What it means for us | How the plan does it |
|---|---|---|
| "consolidating the outputs of all the prior assignments" | Each section traces to a named output file | Source map (§4) |
| Abstract: "LLM in the loop by feeding the rest of the paper" | Written **last**, from the finished paper | Generated in Phase 4, then checked |
| Introduction (i): "research gap/limitation extraction … followed by problem formulation" | Motivation from **Task 2's limitations**, verified against the code by Task 3, plus **Task 3's problem A1** | §4.2 row 1 |
| Introduction (ii): "main novel contributions of the **implemented** solution" | Contributions only for what Task 4 built **and** measured | §0.3 and §4.2; each bullet tied to a result |
| Related work: "literature review … manual filtering … hierarchical catalogue" | Task 1's paper set, filtered by a human, organised as subsections | Citation pool (§6.2) |
| Problem formulation: "fed to LLM … mathematical definition … manually verified" | Formal definition of mask-free localisation and of PU learning at the patch level | LLM draft (§4.3) + human check (§7) |
| Methodology: "dataset, approach … avoid low-level experimentation details" | Subsections *Dataset* and *Approach*, high level only | §4.2 |
| Experiments: "output of codification … verbalised … elaboration of the experiments" | Task 4's results and logs, explained | Numbers only from `results.json` |
| Conclusion: "LLM … what conclusions can be drawn … manually verified" | LLM proposes; only supported conclusions stay | §7 check 5 |
| "workshops on AI generated papers (optional)" | Keep a full AI-use record | §8 |
| "4+1" | 4 content pages + 1 references page | Page budget (§3) and check (§7) |

---

## 1. Ground rules for the agent

1. Follow the phases in order. **Do not stop to ask.** Pick the conservative default, log it (rule 9), and continue. The only stop is the final review.
2. **Use only the latest sources in §4.1.** Ignore older local copies, earlier iterations and Drive folders.
3. **Every number comes from `paper/inputs/results.json`**, built by `tools/build_results.py` from the Task 4 round folders. Nothing is typed by hand or invented.
4. **Every citation comes from the citation pool** (§6.2): Task 1's checked papers plus `extra_refs.bib`. **There is no target number of references.** The reference list is as long as the text needs, and the only limit is that it fits on page 5.
5. **Claim only what Task 4 measured:** targets Brain MRI, Liver CT and RESC; M-h and M-h-ctrl with 3 seeds; M on Liver with 3 seeds; **everything else single-seed** and labelled so. State that MVFA's released checkpoints (R0) were selected on the test set (Task 3 finding C2), while ours were not.
6. **Never pool rounds.** Round 1 (warm start, 2,000 steps) and round 2 (from scratch, Task 3's protocol) are different experiments. Report them separately.
7. Never commit API keys, data or weights; never push without the human's approval.
8. Log every LLM run, human edit, check and decision in `paper/logs/` (§8).
9. **If unsure, do not ask.** Choose the more conservative option (the weaker claim, the smaller change, the default in §2), record it in `paper/logs/DECISIONS.md`, add a line to `paper/OPEN_ITEMS.md`, and continue. Stop only when blocked: a missing API key, or a check that cannot pass after the 3 refinement rounds.

---

## 2. Human inputs (given once, all optional)
Nothing here blocks the run. If an input is missing, the agent uses the default and notes it in `OPEN_ITEMS.md`; the human can fill it in at any time, including at the final review.

| Input | Default if not given |
|---|---|
| **Authors and affiliation** | "Anonymous authors" placeholder |
| **Gemini API key** (environment variable only) | PaperOrchestra cannot run without it. The agent checks once, prepares everything else (pool, drafts, scripts, raw materials), and lists the missing key as the single blocker |
| **Template** | CVPR 2025 (PaperOrchestra's `templates/cvpr2025`) |
| **Workshop target and its call for papers** | None. The disclosure is written anyway; the final review checks the call if one is given |
| **Round 2 timing** | Write now from round 1; update later (§11) |

Starting the run authorises cloning PaperOrchestra (Apache-2.0) at a pinned commit. Committing and pushing are the only actions the agent does not take on its own.

---

## 3. Format and page budget (4+1)

- **Template:** PaperOrchestra's `templates/cvpr2025` (MVFA is a CVPR paper) unless the human names another.
- **The instructor's 4+1 format:** title, authors, abstract, all sections, **every figure and table** and the captions within **4 pages**. **Page 5 holds references only.** No appendix unless the instructor allows one. Over budget means cut text, not squeeze the layout.

| Section | Pages | Content |
|---|---|---|
| Title + Abstract | 0.30 | about 150 words |
| 1 Introduction | 0.55 | (i) background → problem, 0.35; (ii) 3–4 contribution bullets, 0.20 |
| 2 Related work | 0.50 | 3 subsections (§4.2), about 400 words |
| 3 Problem formulation | 0.40 | setup, PU view of patches, objective, evaluation |
| 4 Methodology | 0.60 | 4.1 Dataset; 4.2 Approach (hybrid fine-tune, mask-free variants, π), 0.35 of text; Figure 1, 0.25 |
| 5 Experiments, results and discussion | 1.45 | setup paragraph, 0.20; **Table 1** (M-h vs control), **Table 2** (mask-free), **Figure 2** (Liver heat maps), the π finding in text, discussion and limitations, 1.25 |
| 6 Conclusion | 0.20 | 3–4 sentences |
| **Total content** | **4.00** | |
| References | page 5 only | no target count; as many as the text cites, as long as they fit one page |

At most 2 tables and 1 figure in §5. The π-estimator evidence is one or two sentences with numbers, not a table.

---

## 4. Sources and section map

### 4.1 Latest sources

| Task | Branch @ commit, or folder | Files used |
|---|---|---|
| **Task 1**, literature review | `Vismay` @ `8a312f7` → `task1/` | `final_ref_set.json` (594 papers: checked DOI, abstract, year, venue, relevance 1–5, origin including `cites_mvfa`); `RUN_REPORT.md` |
| **Task 2**, limitations | `task2` @ `efbdb3e` → `task2/` | `final_limitations.json` (the run Task 3 used); `TASK2_DOCUMENTATION.md` |
| **Task 3**, problems and ideas | `task3_kevalina` @ `9cc58d7` → `task3/` | `run3/explorer_data.json` (problem A1); `gaps_verified.json`; `stage0b_code_verification.json` (E1–E20, C1–C4); `stage4_ranking_A1.json`, `stage3_evolution_A1.json` (idea `A1-g1-o6`, competitor `A1-g1-o2`); run 1 `stage3_evolution_A1.json` (`A1-g2-o6`); `rejudge_gemini-3.8-flash.json`; `reports/Task3_Submission.pdf` |
| **Task 4, round 1** | `task4_r1/` (finished 2026-11-22) | `report/TASK4_REPORT.md` (results + issues log), `report/DECISIONS.md`, `report/ISSUES_LOG.md`, `results/results.csv`, `paper_numbers.csv`, `pi*.json`, `report/paper_inputs/` (tables, Liver heat maps), `report/transcript/TRANSCRIPT.md` (AI-use record) |
| **Task 4, round 2** | `task4_r2/` (not yet) | Same layout, plus `report/VERDICT.md` |

### 4.2 Section by section

| Section | Source | Content rules |
|---|---|---|
| **1(i) Background and problem** | Task 2 limitations **L3** (the "zero-shot" setting still uses masks and labels), **L18** (the cost of masks is never measured), **L5** (the treatment of mask-less datasets is unspecified; the code shows they are dropped from the segmentation loss); evidence **E7, E8** (`train_zero.py`) and **E16**; the A1 problem statement | Why MVFA's localisation depends on expensive masks, and why half its training data never teaches localisation |
| **1(ii) Contributions** | §0.2 findings and the framing in §0.3 | Candidate bullets, each kept only if a result backs it: (a) a controlled test of a patch-level PU loss on MVFA's mask-less datasets: no consistent gain, a consistent gain on Brain (finding 1); (b) the finding that π cannot be estimated from MVFA's patch scores, tested against ground truth (finding 4); (c) a mask-free baseline: plain top-k MIL keeps 93–99 % of the released model's localisation, single seed (finding 5); (d) an audit of MVFA's zero-shot training with file/line evidence (finding 6). **No bullet claims an improvement over MVFA** |
| **2 Related work** | Task 1's pool, filtered (§6.2), plus `extra_refs.bib` | **2.1** Medical anomaly detection and vision-language adaptation: MVFA, WinCLIP, APRIL-GAN, AnomalyCLIP, AdaCLIP, VCP-CLIP, and the recent medical zero-shot works already in the pool (IQE-CLIP, MADPOT). **2.2** Weakly supervised and mask-free anomaly localisation (MIL, image-level supervision, pseudo-masks). **2.3** Positive-unlabelled learning and mixture-proportion estimation. State that other papers' numbers use different protocols and are not compared in a table |
| **3 Problem formulation** | A1 statement + §4.3 skeleton + `task4_r1/report/TASK4_REPORT.md` §1 | LLM drafts; a human verifies. Includes the irreducibility assumption of mixture-proportion estimation and the fact that it fails on MVFA's patch scores |
| **4 Methodology** | `TASK4_PLAN_RECOMMENDED.md` §2, `DECISIONS.md` D1–D5 | 4.1 Dataset: BMAD's six datasets, leave-one-modality-out, targets Brain MRI, Liver CT and RESC, which datasets have masks. 4.2 Approach: the hybrid fine-tune (M-h) and its control; the mask-free variants (M, B0) with both adapter sets retrained; π (what was used and why it was not estimated); no checkpoint selection on the target. **No** hyperparameter tables, file names or server details |
| **5 Experiments** | `results.json`, `TASK4_REPORT.md` | Setup and protocol. **Table 1:** per target, R0 (published and reproduced), M-h and M-h-ctrl mean ± std over 3 seeds (image AUROC, pixel AUROC, AUPRO), with the paired result (AUPRO up in k of 9 pairs, sign-test p). **Table 2:** M and B0 per target, retention against R0, AUPRO ratio. **Figure 2:** the Liver heat maps for R0, M-h, M-h-ctrl, M and B0. Text: the π finding with numbers; the sensitivity runs flagged as chosen after seeing mask statistics; limitations (zero-shot only, subsampled, 3 seeds, R0 selected on the test set, M and B0 single-seed outside Liver, the RESC / OCT17 overlap, pixel metrics dominated by background on Liver) |
| **6 Conclusion** | the finished sections | LLM proposes; checked (§7) |
| **Abstract** | the finished paper | Generated last |

### 4.3 Problem formulation skeleton (the LLM writes it; a human verifies)
- **Source datasets** 𝒟₁…𝒟_S, each with images x and image labels y ∈ {0, 1}. Pixel masks exist only for a subset 𝒮_m. Target modality t is unseen.
- **Patches:** a frozen CLIP encoder with segmentation adapters gives, for each image x and each of 4 levels l, patch scores z_{l,j}(x).
- **Latent patch labels** v_j ∈ {0, 1}: in a normal image every v_j = 0; in an abnormal image an unknown fraction π_d of patches have v_j = 1.
- **PU view, per dataset d:** N = patches of normal images ~ p₀; U = patches of abnormal images ~ π_d p₁ + (1 − π_d) p₀. π_d is identifiable only under irreducibility (the anomalous patch distribution is not contained in the normal one). On MVFA's deep levels this fails empirically (finding 4).
- **PU risk:** (1 − π)·mean_N ℓ(z, 0) + max(0, mean_U ℓ(z, 1) − (1 − π)·mean_N ℓ(z, 1)), with z the two-class logit s₁ − s₀ (so sigmoid(z) is MVFA's patch probability). Must match `pu.py` in `task4_r1/`.
- **Hybrid objective (M-h):** MVFA's Dice + Focal on 𝒮_m + λ × PU on the remaining datasets. **Mask-free (M):** PU only, on all sources. **B0:** top-k MIL, k = max(3, 1 % of patches).
- **Evaluation:** pixel AUROC and AUPRO (FPR ≤ 0.3) and image AUROC on t.
- **Retention** = (AUROC_pix − 0.5) / (AUROC_pix(R0) − 0.5).

---

## 5. Choosing the framing from Task 4's results

### 5.1 Round 1 (applied now)
The pre-registered rules of `TASK4_PLAN_RECOMMENDED.md` Step 5 were applied by `analyze.py`. Apply the **first** rule that holds:

| # | Condition | Framing | Status |
|---|---|---|---|
| F-A | M-h > R0 on pixel AUROC for ≥ 2 of 3 targets, image AUROC not down by more than 0.5, **and** M-h > M-h-ctrl | Method paper | **No.** M-h exceeds R0 only on Brain; AUPRO Δ positive in 6/9 pairs |
| F-B | M keeps ≥ 80 % of R0's pixel-AUROC margin and M > B0 | Label-efficiency paper | **No.** Retention 0.58; M > B0 on 0 of 3 targets |
| F-C | Otherwise | **Analysis paper:** an honest negative or mixed result | **Applies** |

Title direction: "…how far image-level labels go for localisation in MVFA…". Not "…improves…".

### 5.2 Round 2 (when it arrives; rules fixed now)
`task4_r2/report/VERDICT.md` gives a verdict per arm and overall. Map it as:

| Round 2 verdict | Framing |
|---|---|
| **Clear win** (beats MVFA under its rule, confirmed on fresh seeds) | **Method paper** on round 2's arm. Round 1 becomes a compressed motivating study (π failure, Brain signal, MIL baseline) |
| **Partial win** | **Mixed paper:** round 2's winning targets and metrics as the main result; round 1 as the controlled test of PU |
| **No improvement but informative**, or **negative** | **Analysis paper** as in round 1, strengthened by round 2's from-scratch comparison |

The agent applies the rule and records the framing in `DECISIONS.md`; the human sees it at the final review. The contributions, title, discussion and conclusion follow it.

---

## 6. Inputs to prepare

### 6.1 `paper/inputs/results.json`: the only source of numbers
Built by `tools/build_results.py`, which reads each round's `results/results.csv`, `paper_numbers.csv` and `pi*.json`:
```
python tools/build_results.py --round r1=task4_r1 --pair r1=mh:mh_ctrl --retention r1=m:b0 --out paper/inputs/results.json
```
- per config and target: image AUROC, pixel AUROC and AUPRO (mean, sample std, n, per-seed values);
- R0 as published and as reproduced;
- paired differences (M-h minus M-h-ctrl) per pair and per target, the positive counts and the one-sided sign-test p;
- retention and the AUPRO ratio against R0 for M and B0;
- π files;
- secondary runs (sensitivity and variant runs), kept apart and never averaged.

Verified on 2026-11-22: every mean ± std in `TASK4_REPORT.md` §3(a) and (b) is reproduced exactly. Note: the report's per-target mean of the paired differences averages the rounded per-pair values, so it can differ from `results.json` by 0.01 (for example Liver pixel AUROC −0.02 against −0.03). `check_numbers.py` must allow ±0.01.

**Automatic check:** the agent compares every cell of `results.json` with the tables in `TASK4_REPORT.md` (±0.01) and records the result in `VERIFICATION_LOG.md`. A mismatch is a blocker; a pass needs no human step.

### 6.2 Citation pool: the only allowed references
1. Start from Task 1's `final_ref_set.json`. Keep **`relevance_score ≥ 3`**.
2. The agent maps papers to §4.2's three subsections, using `topics`, `keywords` and `origin`. **No fixed number:** a paper enters the pool if the text has a specific sentence it supports, and each subsection takes as many as it needs. MVFA, BMAD, its main baselines (WinCLIP, APRIL-GAN, AnomalyCLIP, AdaCLIP, VCP-CLIP) and the strongest papers that cite MVFA go first. **IQE-CLIP and MADPOT are already in the pool** (relevance 5).
3. **Manual filtering (the instructor's requirement) is done by rule, then reviewed once.** The agent drops papers that are off-topic by their abstract, and logs each drop with a one-line reason in `citation_candidates.csv`. The human skims that list at the final review. **If the references overflow page 5,** drop the lowest-relevance papers first, then the ones supporting only a single minor sentence, until it fits.
4. **`extra_refs.bib`** (checked identifiers) already covers: Elkan & Noto 2008, nnPU (Kiryo 2017), MPE (Ramaswamy 2016), attention MIL (Ilse 2018). MVTec AD (AUPRO), CLIP, MVFA and BMAD are in Task 1's pool. A paper found during planning, *Beyond Pooling: Matching for Robust Generalization under Data Heterogeneity* (arXiv 2602.07154, AISTATS 2026), is **not** in the pool; the agent may add it to `extra_refs.bib` if a sentence needs it, after checking that its identifier resolves to that title.
5. Generate `paper/inputs/refs.bib` with stable keys (for example `huang2024mvfa`).

### 6.3 PaperOrchestra raw materials (`paper/inputs/raw_materials/`)
Use the format found in Phase 1. Content:
- **`idea.md`:** §0 (problem, method as run, findings, framing), §4.3, the forced section structure and page budget (§3), "cite only keys in `refs.bib`; use only numbers in `results.json`; claim no improvement over MVFA".
- **`experiment_log.md`:** generated from `results.json` and Task 4's report: setup, protocol, results, the π analysis, the sensitivity runs (flagged), limitations, and the issues found during codification (`ISSUES_LOG.md`).
- **`figures/`:** Figure 1, the method diagram; Figure 2, `task4_r1/report/paper_inputs/liver_heatmaps.pdf` (the same four images for every model, fixed before any map was viewed).
- **`related_work_outline.md`:** the three subsections with their approved papers.

---

## 7. Verification (the instructor's "manually verified", made systematic)

Run after every generation round. Log in `paper/logs/VERIFICATION_LOG.md`, one table per section: claim → source → pass/fail → fix.

1. **Numbers (`check_numbers.py`, to be written):** every number in the `.tex` matches `results.json` (±0.01 for rounding) or is structural.
2. **Citations (`check_citations.py`):** every `\cite` key is in `refs.bib`, and every `refs.bib` entry is in the pool. The agent checks each cited paper's abstract against its sentence; doubtful pairs go to `OPEN_ITEMS.md`.
3. **Maths (agent, then the independent reviewer of check 7):** every symbol is defined; the PU risk matches `pu.py`; the hybrid objective matches M-h; the protocol matches `TASK4_PLAN_RECOMMENDED.md` §1–§3 and `DECISIONS.md`.
4. **Claims (agent; uncertain items go to `OPEN_ITEMS.md`):** every sentence in the Introduction, Contributions, Discussion and Conclusion is tagged with its source. Remove untagged claims. Specifically:
   - **no claim of improvement over MVFA** unless a rule in §5 supports it;
   - no claim beyond the targets and seeds actually run; single-seed results are labelled;
   - the B0 finding is described as an observation, not a pre-registered result;
   - sensitivity runs chosen after seeing mask statistics are labelled as such;
   - it is stated that R0 was selected on the test set while our runs were not;
   - M-h's π comes from the masked sources' masks (D1), which is mask information M-h already trains on;
   - no "first ever" novelty claim unless the related-work check supports it.
5. **Conclusion and abstract:** generated last by feeding the finished paper to the LLM (the instructor's procedure), then checks 1, 2 and 4 are re-run.
6. **Format (`check_pages.py`):** the PDF compiles, the content ends on page 4, the references fit on page 5 and nothing else is there, no undefined references, figures legible.
7. **Independent review:** a different model from the writer answers a reviewer prompt asking for unsupported claims, overstatements and unclear maths. Every point is fixed or answered in the log.

**Refinement stop rule:** at most **3 rounds**; stop early when checks 1, 2 and 6 pass **and** a review round raises no new major issue; if round 3 still fails, fix it by hand and log it.

---

## 8. AI-use record

- **`paper/logs/GENERATION_LOG.md`:** each PaperOrchestra run (date, models, settings, runtime, cost if visible) and each human edit (section, change, reason).
- **`paper/runs/round<k>/`:** the raw outputs of each round.
- **Task 4's record:** `task4_r1/report/ISSUES_LOG.md` (29 issues, with the feedback given to the LLM) and `report/transcript/TRANSCRIPT.md` (the full conversation with Claude Code that wrote and ran the code).
- **A disclosure paragraph:** Tasks 1–4 were produced with an AI-augmented pipeline (QUAL-SG literature review; LLM limitation extraction; ResearchAgent + HypER + HypoEvolve ideation; LLM-assisted codification with logged feedback); the paper was drafted with PaperOrchestra and verified by the authors.

---

## 9. PaperOrchestra: setup and fork

**What it is:** Google Research's open-source multi-agent paper writer (Apache-2.0; arXiv 2604.05018; `github.com/google-research/paper-orchestra`). Agents: Outline, Literature Review, Section Writing, Content Refinement (simulated peer review), Plotting. **Input:** a raw-materials folder and a LaTeX template folder. **Output:** a LaTeX manuscript.

**Needs:** Python 3.11; a `.env` with `GEMINI_API_KEY` (or Vertex AI, or `OPENAI_API_KEY`); `SEMANTIC_SCHOLAR_API_KEY` for its own search (not needed if F1 works). Cost and runtime are not documented, so measure them in the pilot.

**Run:**
```
python paper_writing_cli.py --raw_materials_dir paper/inputs/raw_materials --latex_template_dir templates/cvpr2025 --output_dir paper/runs/round<k> --use_plotting true
```

**Not documented, so the agent finds out by reading the code in Phase 1:** the raw-material file format; whether it accepts a given reference list; whether the outline can be forced; whether a page budget can be set.

| ID | Change in the fork (`paper/paper-orchestra/`, commit pinned) | Why | Fallback |
|---|---|---|---|
| **F1** | The Literature Review agent uses only `refs.bib` and the pool, with no Semantic Scholar search | Related work must come from Task 1; prevents invented citations | Let it search, then rewrite §2 and the `.bib` from the pool |
| **F2** | Force the outline to the §3 sections and the §4.2 subsections | The instructor's structure has a separate Problem Formulation | Edit the outline between the Outline and Writing steps |
| **F3** | Put the §3 page budget in the writing and refinement prompts | 4+1 | Trim by hand in Phase 6 |
| **F4** | Refinement prompt: never change numbers or citation keys; never add a claim of improvement; flag doubtful claims | Protects the results and the honest framing | Checks 1, 2 and 4 catch it |

Log each finding and change in `paper/logs/ISSUES_LOG.md`.

---

## 10. Phases

The agent runs Phases 0–6 in one go. `TASK5_STAGE_A_PLAN.md` gives the step-by-step detail for Phases 0–2. There is **one human gate, at the end**.

| Phase | Steps |
|---|---|
| **0. Setup** | Create `paper/` (inputs, scripts, logs, runs); copy `Task5_inputs_v2/` into `paper/inputs/`. Read §2's inputs from the environment or use the defaults. Clone PaperOrchestra at a pinned commit; keys as environment variables only |
| **1. Fork, citation pool, early drafts** | Read PaperOrchestra's code; F1–F4 or fallbacks. Build the pool and `refs.bib` (§6.2). Draft by LLM and check (§7): §2, §3, §4, §1(i) |
| **2. Pilot and scripts** | One PaperOrchestra run with a **dummy** `experiment_log.md` (numbers clearly fake, marked DUMMY) to test the pipeline, format, page fit and cost; delete the dummy outputs afterwards, keeping only the notes. Write `check_numbers.py`, `check_citations.py`, `check_pages.py` and test them on the dummy file |
| **3. Results in** | Run `build_results.py` (round 1) → `results.json`; run the automatic check against `TASK4_REPORT.md` (§6.1); copy Task 4's figures; apply the framing rule (F-C, §5.1) |
| **4. Full generation** | Final raw materials (§6.3); full PaperOrchestra run with plotting; refinement rounds under the §7 stop rule |
| **5. Verification** | Checks 1–7 of §7, in the instructor's order: Related work → Problem formulation → Methodology → Experiments → Introduction → Conclusion → Abstract. Write `VERIFICATION_LOG.md` |
| **6. Polish and hand-over** | Page fit (check 6), independent review (check 7), AI-use disclosure, final PDF, `OPEN_ITEMS.md` |

### Gate 1: the only human review (end of Phase 6)
The human reads the final PDF with `VERIFICATION_LOG.md` (claim → source → pass/fail) and `OPEN_ITEMS.md` (everything the agent could not settle, each with its default and what to change). This review is the instructor's "manually verified". The human also fills in any §2 input left at its default, and decides on commit, push and any workshop submission. The agent takes none of those three actions.

---

## 11. Updating when round 2 arrives (the swap)

Round 2 runs `TASK4_ROUND2_PLAN.md` and delivers a folder with the same layout as `task4_r1/`: `results/results.csv`, `paper_numbers.csv`, `report/TASK4_REPORT.md`, `report/VERDICT.md`, `report/paper_inputs/`. Its configs are named differently (MVFA, MIL, PU, G, …) and its protocol is different (from scratch, RESC held out together with OCT17), so **never merge it into round 1's numbers**.

1. **Drop the folder in as `paper/inputs/task4_r2/`.** Check that its `results.csv` has the same columns and that main runs are named `<config>_<target>_s<seed>`; if not, adapt `is_main()` in `build_results.py` and log it.
2. **Rebuild:**
   ```
   python tools/build_results.py --round r1=task4_r1 --round r2=task4_r2 \
       --pair r1=mh:mh_ctrl --pair r2=<arm>:<baseline> --retention r1=m:b0 --out paper/inputs/results.json
   ```
   Use `--pair` for each comparison round 2's report makes (for example gated MIL against MIL, each arm against MVFA).
3. **Automatic check again** for the round-2 numbers (§6.1). Read `VERDICT.md` and apply §5.2.
4. **Re-frame** by the §5.2 rule. Update the contributions, title, discussion and conclusion.
5. **Re-budget §5 of the paper** (1.25 pages): round 2's main table, then a compressed round 1 (one small table, or text with the headline numbers), one figure. Cut round 1 first if space is short.
6. **Add** to the related-work and methodology sections what round 2 added (the normal bank, the fixed-π PU), and to the limitations the round-2 caveats (5 epochs on a subsample; RESC and OCT17 held out together, so MVFA's published RESC number is a different protocol).
7. **Regenerate** (Phase 4) and re-run all checks (Phase 5). Use `VERDICT.md`'s workshop checklist as the starting point for the final review (Gate 1).

---

## 12. Final update: rounds 2, 3 and the follow-up (done 2026-11-25)

Task 4 ended with round 3 (`TASK4_ROUND3_PLAN.md`) and a pre-registered Brain follow-up. §11 was applied to all three at once:
- **Numbers.** `results.json` now holds rounds `r1`, `r2`, `r3` and `fu`, never pooled. It is built by the extended `tools/build_results.py` and reconciled with every report table (554 values).
- **Framing (§5.2).** A partial win gives a **mixed paper**. Two headlines are confirmed under frozen rules:
  - MIL with image labels only keeps 0.969 of MVFA's pixel-AUROC margin;
  - 10 % masks plus MIL beat full-mask MVFA on brain MRI, on 5/5 pre-registered seeds.

  The rest goes in one "what did not hold" table.
- **Generation.** PaperOrchestra was not re-run: its round-1 draft needed heavy manual fixing, and the Gemini quota had run out. An LLM agent revised the checked draft instead. The PaperOrchestra inputs are regenerated, so a later re-run with a new key is possible.
- **Checks.** Every check passes, including a new claim-by-claim audit (42/42).
- **Where things are:** the details are in `paper/README.md`, `paper/logs/` and `paper/OPEN_ITEMS.md`; the final PDF is `paper/runs/final/latex_writeup/final_paper.pdf`.

---

## Appendix A: Instruction file for the agent (paper part)

```
You are executing plans/PAPER_PLAN.md (v2). Read it fully first, then MANIFEST.md.
- Use only the sources in its §4.1 (latest GitHub outputs and Task 4's outputs).
- Follow its phases in order and run to the end without asking questions. Defaults in §2; unsettled items go to paper/OPEN_ITEMS.md. The only human review is Gate 1, at the end.
- Numbers only from paper/inputs/results.json; citations only from paper/inputs/refs.bib.
- Never claim an improvement over MVFA unless a rule in §5 says so. Never pool Task 4 rounds.
- Log every LLM run, edit, check and decision in paper/logs/.
- Never commit keys, data or weights; never commit or push. When unsure, take the conservative option and log it.
```

## Appendix B: Start prompt
> Read `MANIFEST.md`, `plans/README.md` and `plans/PAPER_PLAN.md`, then run Phases 0–6 to the end. Don't ask me questions: use the defaults in §2 and log anything unsettled in `paper/OPEN_ITEMS.md`. Stop only if you are blocked (for example, no Gemini key), and tell me exactly what is missing. I'll review at the end (Gate 1).

## Appendix C: Key sources
- MVFA: Huang et al., CVPR 2024, `github.com/MediaBrain-SJTU/MVFA-AD`.
- PaperOrchestra: Song, Song, Pfister and Yoon, arXiv 2604.05018, `github.com/google-research/paper-orchestra`.
- nnPU: Kiryo et al., NeurIPS 2017. Mixture-proportion estimation: Ramaswamy, Scott and Tewari, ICML 2016.
- Team repository: `Divyam10113/Intro-to-LLMs` (branches in §4.1).
