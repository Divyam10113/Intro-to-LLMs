# Final task: research paper plan (PaperOrchestra + our Tasks 1–4)

**Goal:** a research paper in **4+1 format** (confirmed: 4 pages of content, plus 1 page for references only) that consolidates the **latest** outputs of Tasks 1–4. It is drafted with Google Research's **PaperOrchestra** and verified by us section by section, as the instructor asks.

**Executed by:** a coding agent (Copilot agent mode, Codex CLI, Cline, Cursor or similar) with a human supervisor. The agent stops at every gate marked 🛑.

**Companion plan:** `TASK4_PLAN.md` in the same folder. Task 4 (codifying problem A1) produces the results this paper reports. Read `README.md` in this folder for the order of execution.

**Sources:** only the team's GitHub repository `Divyam10113/Intro-to-LLMs` and Task 4's outputs. Do **not** use any older local folder or Google Drive copy.

---

## 0. What the paper is about

**Problem (Task 3, run 3, A1):** MVFA's segmentation adapters, the only part that produces anomaly maps, learn only from expert pixel masks on 3 of BMAD's 6 datasets; the other 3 never teach localisation. *Can localisation be learned from image-level labels alone, and how much of mask-supervised localisation does that keep on an unseen modality?*

**Method (Task 4, per `TASK4_PLAN.md`, simple low-compute version):**
- witness-rate-adaptive MIL: each mask-less dataset's lesion fraction π is estimated without masks, once, from MVFA's released model;
- refined into a **non-negative positive-unlabelled (nnPU) loss** weighted by π;
- **M-h (main):** MVFA's released zero-shot checkpoint, with only the segmentation adapters fine-tuned briefly, using MVFA's mask loss on the masked sources plus λ × the PU loss on the three mask-less sources;
- **M-h-ctrl:** the same fine-tune without the PU term, to show whether the gain comes from the mask-less data;
- **optional M / B0:** segmentation adapters re-initialised and trained mask-free with the PU loss, or with top-k MIL.

**The framing depends on Task 4's outcome.** It is chosen in Phase 4 by the rule in §4. Until then, nothing about results may be written as fact.

### 0.1 Interpreting the instructor's brief

| Instructor's words | What it means for us | How the plan does it |
|---|---|---|
| "consolidating the outputs of all the prior assignments" | Each section traces to a named output file | Source map (§3) |
| Abstract: "LLM in the loop by feeding the rest of the paper" | Written **last**, from the finished paper | Generated in Phase 5, then checked |
| Introduction (i): "research gap/limitation extraction … followed by problem formulation" | Motivation from **Task 2's limitations**, as verified against the code by Task 3, plus **Task 3's problem A1** | §3.2 row 1 |
| Introduction (ii): "main novel contributions of the **implemented** solution" | Contributions only for what Task 4 actually built **and** measured | Written after Task 4; each bullet tied to a result (§4) |
| Related work: "literature review … manual filtering of unrelated papers … hierarchical catalogue (section, subsection)" | Task 1's paper set, filtered by a human, organised as subsections | Citation pool (§5.2), PaperOrchestra change F1 |
| Problem formulation: "fed to LLM … mathematical definition … manually verified" | Formal definition of mask-free localisation, and of PU learning at the patch level | LLM draft (§3.3) + human check (§6) |
| Methodology: "dataset, approach … avoid low-level experimentation and implementation details" | Subsections *Dataset* and *Approach*, high level only | Section rules (§3.2) |
| Experiments: "output of codification … verbalised … elaboration of the experiments performed" | Task 4's results and logs, explained | Numbers only from `results.json` (§5.1) |
| Conclusion: "LLM … what conclusions can be drawn … manually verified" | LLM proposes; only supported conclusions stay | §6 check 5 |
| "workshops on AI generated papers (optional)" | Keep a full AI-use record | §7 |
| "4+1" | 4 content pages + 1 references page | Page budget (§2) and check (§6) |

---

## 1. Ground rules for the agent

1. Follow the phases in order; stop at every 🛑 gate.
2. **Use only the latest sources in §3.1.** Ignore older local copies, earlier iterations and Drive folders.
3. **Every number comes from `paper/inputs/results.json`**, which is built by script from Task 4's outputs (§5.1). Nothing is typed by hand or invented.
4. **Every citation comes from the citation pool** (§5.2): Task 1's checked papers plus a small, human-approved `extra_refs.bib`.
5. **Claim only what Task 4 measured:** the targets Brain MRI, Liver CT and RESC, mostly one seed. Label single-seed results as such, and say that MVFA's released checkpoints (R0) were selected on the test set (Task 3 finding C2), while ours were not.
6. Never commit API keys, data or weights; never push without the human's approval.
7. Log every LLM run, human edit, check and decision in `paper/logs/` (§7).
8. If unsure, stop and ask; record it in `paper/logs/DECISIONS.md`.

---

## 2. Format and page budget (4+1)

- **Template:** PaperOrchestra's `templates/cvpr2025` (MVFA is a CVPR paper) **unless the human names another at Gate 0**.
- **The instructor's 4+1 format:** title, authors, abstract, all sections, **every figure and table** and the captions within **4 pages**. **Page 5 holds references only.** No appendix unless the instructor allows one. Over budget means cut text, not squeeze the layout.

| Section | Pages | Content |
|---|---|---|
| Title + Abstract | 0.30 | about 150 words |
| 1 Introduction | 0.55 | (i) background → problem, 0.35; (ii) 3–4 contribution bullets, 0.20 |
| 2 Related work | 0.50 | 3 subsections (§3.2), about 400 words |
| 3 Problem formulation | 0.40 | setup, PU view of patches, objective, evaluation |
| 4 Methodology | 0.60 | 4.1 Dataset; 4.2 Approach (π estimation → nnPU loss → hybrid), 0.35 of text; Figure 1, 0.25 |
| 5 Experiments, results and discussion | 1.45 | setup paragraph, 0.20; results with at most 2 tables + 1 figure, discussion and limitations, 1.25 |
| 6 Conclusion | 0.20 | 3–4 sentences |
| **Total content** | **4.00** | |
| References | page 5 only | about 20–30 references |

This is the same budget as `TASK5_STAGE_A_PLAN.md`: 2.25 pages are drafted in Stage A, and 1.75 pages are reserved for Stage B.

---

## 3. Sources and section map

### 3.1 Latest sources (verify the commit hashes at Gate 0; use newer commits if they exist)

| Task | Branch @ commit | Files used |
|---|---|---|
| **Task 1**, literature review | `Vismay` @ `8a312f7` | `literature_review/runs/v6_2026-11-16/final_ref_set.json` (594 papers with checked DOI, abstract, year, venue, citation count, OpenAlex keywords and topics, relevance 1–5, origin including `cites_mvfa`); `RUN_REPORT.md` |
| **Task 2**, limitations | `task2` @ `efbdb3e` | `limitations/outputs/full_nim/final_limitations.json` (the run Task 3 used); `DELIVERABLES/task2/TASK2_DOCUMENTATION.md` |
| **Task 3**, problems and ideas | `task3_kevalina` @ `9cc58d7` | `task3/outputs/explorer_data.json` (problem A1: statement, rationale, scores); `gaps_verified.json` (A1 group "Label-efficient adaptation": L3, L4, L5, L18); `stage0b_code_verification.json` (E1, E6, E7, E8, E15, E16 and their verdicts); `stage4_ranking_A1.json`, `stage3_evolution_A1.json` (idea `A1-g1-o6`, competitor `A1-g1-o2`); run 1 `task3/runs/run1_surveygen341/outputs/stage3_evolution_A1.json` (refinement `A1-g2-o6`); `rejudge_gemini-3.8-flash.json`; `task3/reports/Task3_Submission.pdf` |
| **Task 4**, codification | produced by `TASK4_PLAN.md` | `task4/report/TASK4_REPORT.md` (issues log + results table); `task4/results/results.csv`, `paper_numbers.csv`, `pi.json`; `task4/report/paper_inputs/` (hand-off, `TASK4_PLAN.md` Step 6) |

### 3.2 Section by section

| Section | Source | Content rules |
|---|---|---|
| **1(i) Background and problem** | Task 2 limitations **L3** (the "zero-shot" setting still uses masks and labels; code-confirmed), **L18** (the cost of masks is never measured; confirmed), **L5** (the treatment of mask-less datasets is unspecified; the code shows they're dropped from the seg loss); evidence **E7, E8** (`train_zero.py` ~134, ~160) and **E16** (few-shot supports with masks); the A1 problem statement | Why MVFA's localisation depends on expensive masks, and why half its training data never teaches localisation |
| **1(ii) Contributions** | Task 3's ideas (`A1-g1-o6` → nnPU from `A1-g2-o6`; competitor `A1-g1-o2`) and **Task 4's results** | Chosen by the framing rule (§4). Candidates: (a) patch-level PU formulation of mask-free training for CLIP adapters, with mask-free π estimation; (b) a hybrid that uses MVFA's unused mask-less datasets; (c) a control (M-h against M-h-ctrl) showing whether the gain comes from the PU term on mask-less data rather than from extra fine-tuning; (d) a low-compute, warm-start recipe with no test-set checkpoint selection (unlike MVFA's code, finding C2). Keep only those backed by results |
| **2 Related work** | Task 1's pool, filtered (§5.2), plus `extra_refs.bib` | **2.1** Medical anomaly detection and vision-language adaptation (MVFA, its relatives, papers that cite MVFA). **2.2** Weakly supervised and mask-free anomaly localisation (MIL, image-level supervision, pseudo-masks). **2.3** Positive-unlabelled learning and mixture-proportion estimation |
| **3 Problem formulation** | A1 statement + `TASK4_PLAN.md` §2 (the method as run) | LLM drafts from the skeleton in §3.3; a human verifies |
| **4 Methodology** | `TASK4_PLAN.md` §2 and `TASK4_REPORT.md` | 4.1 Dataset: BMAD's six datasets, zero-shot leave-one-modality-out with targets Brain MRI, Liver CT and RESC, and which datasets have masks. 4.2 Approach: π estimation → nnPU loss → hybrid fine-tune of the segmentation adapters from MVFA's released model; no test-set checkpoint selection. **No** hyperparameter tables, file names or Colab details |
| **5 Experiments** | `results.json`, `TASK4_REPORT.md` (results table and issues log) | Setup and protocol; main table per target (R0 as published, R0 reproduced, M-h, M-h-ctrl, and M / B0 if run); π per mask-less dataset; one qualitative figure; limitations (one seed, subsamples, zero-shot only, R0 selected on test, RESC/OCT17 overlap) |
| **6 Conclusion** | the finished sections | LLM proposes; checked (§6) |
| **Abstract** | the finished paper | Generated last |

### 3.3 Problem formulation skeleton (the LLM writes it; a human verifies)
- **Source datasets** 𝒟₁…𝒟_S, each with images x and image labels y ∈ {0, 1}. Pixel masks exist only for a subset 𝒮_m ⊂ {1…S}. Target modality t is unseen.
- **Patches:** a frozen CLIP encoder with segmentation adapters g_θ gives, for each image x and each of 4 levels l, patch scores z_{l,j}(x).
- **Latent patch labels** v_j ∈ {0, 1}:
  - in a normal image (y = 0), every v_j = 0;
  - in an abnormal image (y = 1), an unknown fraction π_d of patches have v_j = 1.
- **PU view, per dataset d:**
  - **N** = patches of normal images, ~ p₀;
  - **U** = patches of abnormal images, ~ π_d p₁ + (1 − π_d) p₀;
  - π_d is identifiable under the usual irreducibility assumption of mixture-proportion estimation, and is estimated from p₀ and the U mixture.
- **Objective (mask-free, optional M):** minimise the non-negative PU risk of `TASK4_PLAN.md` §2 at every level, using only image labels.
- **Hybrid objective:** MVFA's Dice + Focal loss on 𝒮_m, plus λ × the PU risk on the remaining datasets.
- **Evaluation:** localisation (pixel AUROC, AUPRO) and detection (image AUROC) on t, under leave-one-modality-out.
- **Target:** retention = (AUROC_pix − 0.5)/(AUROC_pix(R0) − 0.5) ≥ 0.8 for mask-free; > R0 for the hybrid.

---

## 4. Choosing the framing from Task 4's results (Phase 4)

Use the rules fixed in advance in `TASK4_PLAN.md` Step 5, on the must-run results (targets Brain MRI, Liver CT, RESC). Apply the **first** rule that holds:

| # | Condition | Framing | Title direction |
|---|---|---|---|
| F-A | M-h > R0 on pixel AUROC for ≥ 2 of 3 targets, image AUROC not down by more than 0.5 points, **and** M-h > M-h-ctrl | **Method paper:** "Mask-less data improves MVFA's localisation" | "…learning localisation from mask-less data with positive-unlabelled learning …" |
| F-B | (if M and B0 were run) M keeps ≥ 80% of R0's pixel-AUROC margin and M > B0 | **Label-efficiency paper:** "How far image labels go" | "…mask-free localisation via positive-unlabelled learning …" |
| F-C | Otherwise | **Analysis paper:** an honest negative or mixed result (π values, fine-tuning versus PU effect) | "…when does image-level supervision help …" |

The human confirms the framing at 🛑 Gate 3. The contributions, title, discussion and conclusion follow it.

---

## 5. Inputs to prepare

### 5.1 `paper/inputs/results.json`: the only source of numbers
Built by `paper/scripts/build_results.py` from `task4/results/results.csv`, `paper_numbers.csv` and `pi.json`:
- per config and target: image and pixel AUROC (mean ± std if more than one seed, with the number of seeds);
- R0 as published (`paper_numbers.csv`) and R0 as reproduced;
- retention against R0 for M and B0, if run;
- π per mask-less dataset;
- the protocol facts: targets, source datasets, subsample sizes, steps S, λ, seeds, GPU minutes.

Cross-check it against the results table in `TASK4_REPORT.md`.

🛑 **Gate 2:** the human compares `results.json` with `TASK4_REPORT.md`.

### 5.2 Citation pool: the only allowed references
1. Start from Task 1's `final_ref_set.json`. Keep **`relevance_score ≥ 3`**.
2. The agent proposes about 40 papers mapped to §3.2's three subsections, using `topics`, `keywords` and `origin`. MVFA, its main baselines and the strongest papers that cite MVFA go first.
3. **The human removes unrelated papers** (the instructor's "manual filtering"; 🛑 Gate 1).
4. **`extra_refs.bib`** covers what Task 1 never searched for. Every entry needs a checked identifier (DOI, or the arXiv DOI `10.48550/arxiv.*`), and the human approves each one. At least:
   - Kiryo et al. 2017, nnPU (arXiv 1703.00593);
   - Elkan & Noto 2008, learning from positive and unlabelled data;
   - one mixture-proportion estimation reference (for example Scott 2015, or Ramaswamy et al. 2016);
   - one multiple-instance learning reference (for example Ilse et al. 2018, attention-based MIL);
   - the AUPRO / per-region-overlap metric source (MVTec AD, Bergmann et al.).
5. Generate `paper/inputs/refs.bib` with stable keys (for example `huang2024mvfa`).

### 5.3 PaperOrchestra raw materials (`paper/inputs/raw_materials/`)
Use the format found in Phase 1. Content:
- **`idea.md`:**
  - §0 (problem and method);
  - §3.3 (formulation skeleton);
  - the framing chosen in §4, and its contributions;
  - the forced section structure and page budget (§2);
  - "cite only keys in `refs.bib`; use only numbers in `results.json`".
- **`experiment_log.md`:**
  - generated from `results.json` and Task 4's report and logs: setup, protocol, the ladder, results, π analysis;
  - the improvement loop summary and stop reason;
  - the limitations;
  - the issues found during codification (the issues log in `TASK4_REPORT.md`), for the discussion and the AI-use note.
- **`figures/`:**
  - Figure 1, the method diagram (by the Plotting agent or by hand);
  - Task 4's exported figures (π against the truth; qualitative heat maps; ablation bars).
- **`related_work_outline.md`:** the three subsections with their approved papers.

---

## 6. Verification (the instructor's "manually verified", made systematic)

Run after every generation round. Log in `paper/logs/VERIFICATION_LOG.md`, one table per section: claim → source → pass/fail → fix.

1. **Numbers (script `check_numbers.py`):** every number in the `.tex` matches `results.json` (rounding allowed) or is structural.
2. **Citations (script `check_citations.py`):** every `\cite` key is in `refs.bib`, and every `refs.bib` entry is in the pool. The human spot-checks that each cited paper supports its sentence (from its abstract).
3. **Maths (human):**
   - every symbol is defined;
   - the PU risk matches `TASK4_PLAN.md` §2 step 2 exactly;
   - the π estimator matches §2 step 1;
   - the hybrid objective matches M-h;
   - the protocol matches `TASK4_PLAN.md` §1–§3.
4. **Claims (human, LLM-assisted):** every sentence in the Introduction, Contributions, Discussion and Conclusion is tagged with its source. Remove untagged claims. Specifically:
   - no claim beyond the targets and seeds actually run;
   - dev-fold numbers are labelled as such;
   - it is stated that MVFA's released checkpoints were selected on the test set (finding C2), while ours used none;
   - no "first ever" novelty claim unless the related-work check supports it.
5. **Conclusion and abstract:** generated last by feeding the finished paper to the LLM (the instructor's procedure), then checks 1, 2 and 4 are re-run.
6. **Format (script `check_pages.py`):** the PDF compiles, the content ends on page 4, the references fit on page 5, there are no undefined references, and the figures are legible.
7. **Independent review:** a different model from the writer answers a reviewer prompt asking for unsupported claims, overstatements and unclear maths. Every point is fixed or answered in the log.

**Refinement stop rule:**
- at most **3 rounds**;
- stop early when checks 1, 2 and 6 pass **and** a review round raises no new major issue;
- if round 3 still fails, fix it by hand and log it.

---

## 7. AI-use record

- **`paper/logs/GENERATION_LOG.md`:** each PaperOrchestra run (date, models, settings, runtime, cost if visible) and each human edit (section, change, reason).
- **`paper/runs/round<k>/`:** the raw outputs of each round.
- **A disclosure paragraph:**
  - Tasks 1–4 were produced with an AI-augmented pipeline: QUAL-SG literature review; LLM limitation extraction; ResearchAgent + HypER + HypoEvolve ideation; LLM-assisted codification following `TASK4_PLAN.md`;
  - the paper was drafted with PaperOrchestra and verified by the authors.

---

## 8. PaperOrchestra: setup and fork

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
| **F2** | Force the outline to the §2 sections and the §3.2 subsections | The instructor's structure has a separate Problem Formulation | Edit the outline between the Outline and Writing steps |
| **F3** | Put the §2 page budget in the writing and refinement prompts | 4+1 | Trim by hand in Phase 6 |
| **F4** | Refinement prompt: never change numbers or citation keys; flag doubtful claims | Protects the results | Checks 1–2 catch it |

Log each finding and change in `paper/logs/ISSUES_LOG.md`.

---

## 9. Phases

The paper runs in **two stages**, so most of it is ready the moment Task 4 finishes.

### Stage A: can start now (Task 4's method is fixed in `TASK4_PLAN.md` §2)

| Phase | Steps | Gate |
|---|---|---|
| **0. Setup and inputs from the human** | Create `paper/` (inputs, scripts, logs, runs). Ask for and record the 🛑 **Gate 0 inputs** (§9.1). Clone PaperOrchestra at a pinned commit; keys as environment variables only | 🛑 Gate 0 |
| **1. Fork, citation pool and early drafts** | Read PaperOrchestra's code; make F1–F4 or choose fallbacks. Build the citation pool and `refs.bib` (§5.2). Draft by LLM, verify by hand: §2 Related work outline, §3 Problem formulation (from §3.3 + `TASK4_PLAN.md` §2), §4.1 Dataset, §4.2 Approach, §1(i) Background | 🛑 Gate 1: filtered reference list and early drafts |
| **2. Pilot run** | One PaperOrchestra run with a **dummy** `experiment_log.md` (clearly fake numbers, marked DUMMY) to test the pipeline, format, page fit and cost. **Never keep any dummy number** | — |

### Stage B: after Task 4 finishes (`TASK4_PLAN.md` Step 6, report and hand-off)

| Phase | Steps | Gate |
|---|---|---|
| **3. Results in** | Run `build_results.py` → `results.json`; copy Task 4's exported figures; choose the framing (§4) | 🛑 Gate 2 (numbers) and 🛑 Gate 3 (framing) |
| **4. Full generation** | Build the final raw materials (§5.3); full PaperOrchestra run with plotting; refinement rounds under the §6 stop rule | — |
| **5. Section verification** | Checks 3 and 4, in the instructor's order: Related work → Problem formulation → Methodology → Experiments → Introduction → Conclusion → Abstract | 🛑 Gate 4: the human signs off each section |
| **6. Final polish** | Page fit (check 6), independent review (check 7), AI-use disclosure, final PDF | 🛑 Gate 5: final approval; the human decides on commit, push and any workshop submission |

### 9.1 Gate 0 inputs (defaults in brackets; the human confirms or changes each)
1. **Template:** [CVPR 2025 from PaperOrchestra].
2. **Location in the repo:** [`plans/` for both plan files, `task4/` for Task 4 and `paper/` for the paper, on the `Vismay` branch]. Separate `task4` / `paper` branches are an alternative.
3. **Scope:** [A1 only. Problem B1 from Task 3 is mentioned only as a direction identified but not implemented, unless a teammate supplies codified B1 results].
4. **Authors and affiliation:** [no default; must be given by the human].
5. **API access:** [a Gemini API key with some budget; a Semantic Scholar key only if F1 fails].
6. **Task 2 version:** [`full_nim`, as used by Task 3].
7. **Timing:** [Stage A in parallel with Task 4, Stage B after it].
8. **Workshop target:** [none; disclosure written anyway].

---

## Appendix A: Instruction file for the agent (paper part)

```
You are executing plans/PAPER_PLAN.md. Read it fully first.
- Use only the sources in its §3.1 (latest GitHub outputs and Task 4's outputs).
- Follow its phases in order; stop at every gate marked 🛑 and wait for the human.
- Numbers only from paper/inputs/results.json; citations only from paper/inputs/refs.bib.
- Log every LLM run, edit, check and decision in paper/logs/.
- Never commit keys, data or weights; never push without approval. When unsure, ask.
```

## Appendix B: Start prompt
> Read `plans/README.md` and `plans/PAPER_PLAN.md`. Do Phase 0: create the `paper/` layout and ask me the Gate 0 questions (§9.1). Don't go further until I answer.

## Appendix C: Key sources
- MVFA: Huang et al., CVPR 2024, `github.com/MediaBrain-SJTU/MVFA-AD`.
- PaperOrchestra: Song, Song, Pfister and Yoon, arXiv 2604.05018, `github.com/google-research/paper-orchestra`.
- nnPU: Kiryo et al., NeurIPS 2017. Mixture-proportion estimation: Scott, AISTATS 2015.
- Team repository: `Divyam10113/Intro-to-LLMs` (branches in §3.1).
