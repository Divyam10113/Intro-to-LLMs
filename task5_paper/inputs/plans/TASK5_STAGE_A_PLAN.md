# Task 5 (final paper): what to do while Task 4 runs

> **Status, 2026-11-22: partly superseded by `PAPER_PLAN.md` v2.** Task 4 **round 1 has finished** (recommended tier; outputs in `task4_r1/`, listed in `MANIFEST.md`), so this plan's "while Task 4 runs" framing no longer applies, but its **steps A0–A7 are still the detailed instructions for Phases 0–2** of the v2 plan. What changed:
> - **A0 (decisions):** the tier is decided (recommended); the framing is decided (F-C, analysis paper); only authors, API key, template and workshop target remain (v2 §2).
> - **A1 (hand-off agreement):** done. The real files are in `task4_r1/`; use them as they are.
> - **A2 (citation pool):** unchanged, except IQE-CLIP and MADPOT are already in Task 1's pool.
> - **A3 (drafts):** write §3 and §4 from the method **as run** (v2 §0.2, `DECISIONS.md`), not from the plan's expectation: π was not estimated (D1, D2) and M / B0 retrain both adapter sets (D3). The 1(ii) contributions skeleton can now be filled from v2 §4.2.
> - **A6 (scripts):** `tools/build_results.py` already exists and is tested against round 1; write the other three.
> - **Round 2** will update the Experiments section; see v2 §11.
> - **No fixed number of references, and no stops.** Wherever the text below says "about 40", "keep about 20–30", "🛑" or "with your OK", read: the agent decides by rule, logs it, and carries on; the references need only fit on page 5; the human reviews once at the end (v2 Gate 1).


**Situation:** a teammate is running Task 4 (`TASK4_PLAN_RECOMMENDED.md`, or another tier). The paper (4+1 format: 4 content pages + 1 references page) can't report results yet, but **most of it doesn't depend on them**.

This plan covers everything that can be finished **now**, with **no GPU and no Task 4 code**, so that when Task 4's results arrive only the results-dependent parts remain. It's Stage A of `PAPER_PLAN.md`, written out step by step.

**Sources:** only the latest GitHub outputs (`Divyam10113/Intro-to-LLMs`: `Vismay` @ `8a312f7` = Task 1 v6; `task2` @ `efbdb3e` = Task 2; `task3_kevalina` @ `9cc58d7` = Task 3 run 3) and the plans in this folder.

---

## The format: 4+1 pages (the instructor's requirement)

**4 pages of content + 1 page for references only.** Every step below is held to it.

**What counts toward the 4 pages:** the title and author block, the abstract, all 7 sections (Introduction, Related work, Problem formulation, Methodology, Experiments/results/discussion, Conclusion), **every figure and table**, and the captions. **The 5th page holds references only;** nothing else may spill onto it. **No appendix** unless the instructor explicitly allows one.

**Template:** CVPR 2025 two-column (PaperOrchestra's `templates/cvpr2025`). One full page of plain two-column text is roughly **800 words**; figures, tables and equations take space away from that.

**Page budget** (fixed now; the drafts must respect it):

| Part | Pages | ≈ Words of text | Written in | Notes |
|---|---|---|---|---|
| Title, authors, abstract | 0.30 | ≈ 150 (abstract) | Stage B (last) | Reserve the space now |
| 1(i) Background and problem | 0.35 | ≈ 270 | **Stage A** | |
| 1(ii) Contributions | 0.20 | ≈ 120 | Skeleton now, final in Stage B | 3–4 bullets |
| 2 Related work | 0.50 | ≈ 400 | **Stage A** | 3 subsections |
| 3 Problem formulation | 0.40 | ≈ 220 + equations | **Stage A** | |
| 4 Methodology (text) | 0.35 | ≈ 280 | **Stage A** | 4.1 Dataset, 4.2 Approach |
| Figure 1 (method diagram) | 0.25 | — | **Stage A** | Single-column width |
| 5 Setup paragraph | 0.20 | ≈ 160 | **Stage A** | |
| 5 Results, tables, figures, discussion, limitations | 1.25 | ≈ 450 + 2 tables + 1 figure | Stage B | The largest block: kept free for Task 4 |
| 6 Conclusion | 0.20 | ≈ 120 | Stage B | |
| **Total** | **4.00** | | | **References: page 5 only** (as many as the text cites; no target count) |

**Rules:**
- Stage A drafts may use **at most 2.25 pages** in total (1(i) 0.35 + contributions 0.20 + 2: 0.50 + 3: 0.40 + 4: 0.35 + Figure 1: 0.25 + setup 0.20), so that Stage B keeps its 1.75 pages (title and abstract 0.30, results 1.25, conclusion 0.20).
- A draft that runs over budget is **cut, not squeezed:** no smaller fonts, no shrunken margins, and no `\vspace` tricks that break the template.
- The page count is checked by **compiling**, not by counting words (step A6, `check_pages.py`).

---

## What can and can't be done now

| Paper part | Now? | Why |
|---|---|---|
| Title (working) | ✅ draft | The topic is fixed: problem A1; the final title depends on the framing |
| 1(i) Background and problem | ✅ full | From Task 2 and Task 3 only |
| 1(ii) Contributions | ⏳ skeleton | Depends on which Task 4 rule applies |
| 2 Related work | ✅ full | From Task 1 + extra references |
| 3 Problem formulation | ✅ full | From A1 + Task 4's method (§2 of the Task 4 plan) |
| 4 Methodology | ✅ full | Dataset + approach; the method is fixed in the Task 4 plan |
| 5 Experiments | ⏳ setup paragraph only | The results come from Task 4 |
| 6 Conclusion, Abstract | ❌ | Written last, from the finished paper |
| Figure 1 (method diagram) | ✅ | Doesn't need results |
| Figures 2–3, tables | ⏳ templates only | Filled from Task 4's hand-off |
| PaperOrchestra setup and fork | ✅ | Pilot with **dummy** numbers |
| Check scripts | ✅ | Tested on a dummy results file |

---

## Step A0: inputs and defaults (no questions)
Record the values used in `paper/logs/DECISIONS.md`; anything not supplied takes its default (v2 §2):
1. **Authors and affiliation** (no default; needed for the title block).
2. **Template:** CVPR 2025, PaperOrchestra's `templates/cvpr2025`, unless the instructor names another.
3. **Where the paper lives:** default `paper/` on the `Vismay` branch, **not committed until you approve**.
4. **Scope:** A1 only; B1 mentioned as identified but not implemented.
5. **Task 4 tier** your teammate is running (base, credible or recommended). It decides whether the paper has a mask-free (M / B0) part.
6. **API access for PaperOrchestra:** a Gemini API key (as an environment variable only) and, optionally, a Semantic Scholar key.
7. **Workshop target**, if any (affects the disclosure section and maybe the template).
8. **The 4+1 rules** (confirm once with the instructor if unsure): do the title, abstract, figures and tables count toward the 4 pages (assumed **yes**)? Is an appendix allowed (assumed **no**)? Is the 5th page for references only (assumed **yes**)?

## Step A1: agree the hand-off with the Task 4 teammate (about 30 minutes)
So that Stage B is mechanical, confirm with them:
- **File names and location:** `task4/results/results.csv` with exactly the columns of their tier plan (`run_id, config, target, seed, steps, lambda, pi_json, img_auroc, pix_auroc, aupro, gpu_minutes, notes`), plus `paper_numbers.csv`, `pi.json` (and `pi_maskfree.json` for the recommended tier), `task4/report/TASK4_REPORT.md` and `task4/report/paper_inputs/`.
- **Config names** exactly as in the plans: `R0`, `M-h`, `M-h-ctrl`, `M`, `B0`.
- **Heat maps:** Liver CT, the same 2 test images for every config, chosen by a fixed rule (for example the first two abnormal images by file name), saved as PNG.
- **The issues log** as a table (issue, how found, feedback to the LLM, fix). The paper's discussion and AI-use note use it.

## Step A2: citation pool (agent, 2–3 hours)
1. **Shortlist from Task 1** (`final_ref_set.json`, 594 papers; 301 have relevance ≥ 3):
   - keep `relevance_score ≥ 3`;
   - the agent assigns every paper the text needs (**no fixed count**) to one subsection, using `topics`, `keywords` and `origin`:
     - **2.1 Medical anomaly detection and vision-language adaptation:** MVFA, its benchmark (BMAD) and baselines (CLIP-based AD, memory-bank methods), and the strongest papers citing MVFA (`cites_mvfa`, relevance 4–5);
     - **2.2 Weakly supervised and mask-free localisation:** MIL, image-level supervision, pseudo-masks;
     - **2.3 Positive-unlabelled learning and mixture-proportion estimation:** mostly from the extra list (step 3), since Task 1 didn't search this area.
   - Output: `paper/inputs/citation_candidates.csv` (doi, title, year, relevance, origin, subsection, one-line reason).
2. **Filter it by rule** (the instructor's "manual filtering"): drop off-topic papers with a logged reason; if the references overflow page 5, drop the lowest-relevance first. The human skims the list at the final review.
3. **Extra references** (`extra_refs.bib`). Each one's DOI or arXiv id is **checked to resolve to the right title** before use:

   | Topic | Candidate | ID to check |
   |---|---|---|
   | nnPU | Kiryo et al., NeurIPS 2017 | arXiv 1703.00593 |
   | PU learning | Elkan & Noto, KDD 2008 | DOI 10.1145/1401890.1401920 |
   | Mixture-proportion estimation | Ramaswamy, Scott & Tewari, ICML 2016 | arXiv 1603.02501 |
   | Multiple-instance learning | Ilse, Tomczak & Welling, ICML 2018 | arXiv 1802.04712 |
   | AUPRO / PRO metric | Bergmann et al. (MVTec AD), IJCV 2021 | DOI 10.1007/s11263-020-01400-4 |
   | CLIP (if not already in the pool) | Radford et al., ICML 2021 | arXiv 2103.00020 |
4. **Generate `paper/inputs/refs.bib`** from the approved pool, with stable keys (`huang2024mvfa`, …).

## Step A3: draft the sections that don't need results (LLM in the loop, then you verify; 1–2 days)
For each section: give the LLM **only** its source material (below) and the page budget, then check it with the steps in A6. Save LaTeX fragments in `paper/drafts/`.

| Draft | Source material for the LLM | Length (4-page budget) | Your checks |
|---|---|---|---|
| `sec1_background.tex`, 1(i) | Task 2 limitations L3, L5, L18; code evidence E7, E8, E16 (`stage0b_code_verification.json`); A1's statement (`explorer_data.json` → `problems.A1`) | **0.35 page** (≈ 270 words) | Each code fact matches its file:line evidence |
| `sec1_contributions_SKELETON.tex`, 1(ii) | `PAPER_PLAN.md` §3.2 (the candidate contributions) | **0.20 page**: 3–4 bullets **with [RESULT] placeholders** | Nothing claims a result |
| `sec2_related_work.tex` | The **filtered** pool, grouped by subsection, plus the papers' abstracts | **0.50 page** (≈ 400 words), 3 subsections | Every `\cite` key is in `refs.bib`; each citation supports its sentence |
| `sec3_problem_formulation.tex` | `PAPER_PLAN.md` §3.3 skeleton + `TASK4_PLAN*.md` §2 | **0.40 page** (≈ 220 words + equations) | Every symbol defined; the PU risk and the π estimator match the Task 4 plan exactly |
| `sec4_methodology.tex` | Task 4 plan §1–§2 (high level only) | **0.35 page** of text (≈ 280 words): 4.1 Dataset, 4.2 Approach (+ Figure 1, 0.25 page) | No hyperparameter tables, file names or Colab details; M / B0 included only if the tier includes them |
| `sec5_setup.tex` (first paragraph of §5) | Task 4 plan §1 + Step 1 (targets, sources, subsampling, seeds, metrics) | **0.20 page** (≈ 160 words) | Matches the tier being run |
| `fig1_method.tex` or `.pdf` | §2 of the Task 4 plan | **0.25 page**, single-column width | Shows: frozen CLIP → detection adapters (frozen) / segmentation adapters (trained) → mask loss on masked sources + λ·PU loss on mask-less sources, with π estimated once |

**Writing rules for every draft:**
- **stay within the draft's page budget** (the 4+1 table above). After each draft, compile all the Stage A drafts together in the CVPR template and record the pages used in `paper/logs/PAGE_BUDGET.md`. **If the total goes over 2.25 pages, cut text** rather than squeezing the layout;
- no numbers except the setup facts (dataset names, number of targets, seeds);
- no claims about outcomes;
- results-dependent sentences become `[RESULT: …]` placeholders.

## Step A4: PaperOrchestra setup (agent; 3–5 hours)
1. Clone `github.com/google-research/paper-orchestra` (Apache-2.0) at a pinned commit into `paper/paper-orchestra/`. Python 3.11 environment.
2. **Answer from its code what the README doesn't say,** and log the answers in `paper/logs/ISSUES_LOG.md`:
   - (a) the exact file names and format of the raw-materials folder;
   - (b) where the Literature Review agent searches, and whether it can take our `refs.bib`;
   - (c) whether the outline can be forced;
   - (d) whether a page budget can be set.
3. **Make the four changes** from `PAPER_PLAN.md` §8, or choose their fallbacks:
   - **F1:** cite only from our pool;
   - **F2:** force our 7 sections;
   - **F3:** the 4+1 budget: the per-section page and word targets from the table above go into the outline and writing prompts, and the refinement agent may not lengthen a section past its budget;
   - **F4:** never change numbers or citation keys.
4. **Prepare the raw materials** in the format found in step 2:
   - `idea.md`: problem A1, the method (Task 4 §2), the formulation, the required sections and page budget, the citation and number rules, and our drafts from A3 as guidance;
   - `experiment_log_TEMPLATE.md`: every table and statement with `{{placeholders}}` mapped to `results.json` keys;
   - `figures/fig1_method.*`;
   - `related_work_outline.md`.

## Step A5: pilot run with DUMMY numbers (needs the Gemini key; about half a day)
1. **Make a dummy results file:** `paper/inputs/DUMMY_results.json`, with clearly fake values (for example every AUROC 0.111), file name and contents marked **DUMMY**.
2. **Run PaperOrchestra once** (no plotting). Measure the runtime and the API cost.
3. **Check the pipeline, not the science:**
   - the 7-section structure;
   - citations only from `refs.bib`;
   - **the page fit: content ends on page 4, the references fit on page 5, and nothing else is on page 5;**
   - the template;
   - whether our drafts were respected.
4. **Fix the inputs or the fork,** and log each change.
5. **Delete every dummy output** (`paper/runs/pilot*/`) after the review. Keep only the notes in `GENERATION_LOG.md`.

## Step A6: verification scripts (agent, 2–3 hours)
Write them now and test them on the dummy file:

| Script | Does |
|---|---|
| `build_results.py` | `task4/results/*.csv` + `pi*.json` → `paper/inputs/results.json`: per config and target, mean ± std over seeds, number of seeds, retention, π; R0 as published and as reproduced |
| `check_numbers.py` | Every number in the `.tex` must match `results.json` (rounding allowed) or be structural; otherwise it fails |
| `check_citations.py` | Every `\cite` key is in `refs.bib`, and every `refs.bib` entry is in the approved pool |
| `check_pages.py` | **4+1 check:** the PDF compiles; all content (title through conclusion, every figure and table) ends **on page 4**; page 5 contains **only** the references and they fit on it; no appendix; no undefined references or citations. It also reports pages used per section, against the budget table |

Test each on: a correct case (passes), a wrong number (fails), an unknown citation (fails), **a draft whose content spills onto page 5 (fails)** and **a reference list longer than one page (fails)**.

## Step A7: the "ready for results" checklist
Stage A is done when all of these are true:
- [ ] A0 inputs and defaults recorded (authors, template, keys)
- [ ] Hand-off format agreed with the Task 4 teammate (A1)
- [ ] Citation pool filtered by rule; `refs.bib` built; extra references checked (A2)
- [ ] Drafts of 1(i), 2, 3, 4 and the setup paragraph written **and checked**; the contributions skeleton has placeholders only (A3)
- [ ] Figure 1 ready (0.25 page)
- [ ] **4+1 budget respected:** the Stage A drafts compile to **≤ 2.25 pages**, recorded in `PAGE_BUDGET.md`; `check_pages.py` passes on the pilot
- [ ] PaperOrchestra forked (F1–F4 or fallbacks); raw materials and the experiment-log template ready (A4)
- [ ] Pilot run reviewed; dummy outputs deleted (A5)
- [ ] The 4 scripts pass their tests (A6)

---

## When Task 4's results arrive (Stage B, `PAPER_PLAN.md` §9)
1. Run `build_results.py` → `results.json`, then the automatic comparison with `TASK4_REPORT.md` (v2 §6.1).
2. **Choose the framing** with the pre-registered rules (`PAPER_PLAN.md` §4, which mirrors the Task 4 plan's Step 5).
3. **Fill the experiment log** from the template, and the contributions from the skeleton. Add Figures 2–3 from `paper_inputs/`. **The results, tables, figures and discussion must fit in their 1.25 pages:** at most 2 tables and 1 figure.
4. **Full PaperOrchestra run,** then at most 3 refinement rounds.
5. **Experiments section,** then the Introduction contributions, then the **Conclusion**, then the **Abstract**, with the LLM fed the finished sections, in the instructor's order.
6. **The 7 checks** (`PAPER_PLAN.md` §6), including the **4+1 page check**, section sign-off, and the final PDF.

**Expected Stage B effort, once Stage A is complete:** about 2–3 days, most of it verification.

## Effort summary for Stage A
| Step | Who | Effort |
|---|---|---|
| A0, A1 | You (+ teammate) | ≈ 1 h |
| A2 | Agent + you (filtering) | 2–3 h |
| A3 | LLM + you (checks) | 1–2 days |
| A4 | Agent | 3–5 h |
| A5 | Agent + you | ≈ ½ day |
| A6 | Agent | 2–3 h |
| **Total** | | **≈ 3–4 working days, no GPU** |
